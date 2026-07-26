#!/usr/bin/env python3
"""Run a warm-resident dwagon-local GLM-5.2 PP=1/TP=2 benchmark.

One SGLang parent owns both dwagon RTX 3090 devices in the current PCI/NUMA
order. KTransformers owns all 112 physical CPU cores, both NUMA nodes, and two
CPUInfer pools. The server is launched once, receives one semantic coherency
warm-up, and then serves selected deterministic 7,744-input/128-output
concurrency 1, 2, and/or 3 cases without a restart or cache flush. The default
remains the matched c1/c2 run. A c1-only run admits a correspondingly smaller
KV pool for VRAM-sensitive MTP and stream-prefill experiments. Every selected
profile retains a 256-token scheduler admission reserve and a measured VRAM
safety floor.

The process lifecycle is reused from the proven GLM-4.7 local TP2 harness. This
GLM-5.2 harness adds fail-closed GPU inventory, free-VRAM, CUDA-process
ownership, and exact server-capacity gates before benchmark timing is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from types import FrameType
from typing import Final, Literal, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import httpx  # noqa: E402

from exo.shared.types.common import Host  # noqa: E402
from scripts import run_sglang_kt_glm47_pp2_local_diagnostic as pp2  # noqa: E402
from scripts import run_sglang_kt_glm47_tp2_local_diagnostic as tp2  # noqa: E402
from scripts import run_sglang_kt_glm52_pp3_benchmark as glm52  # noqa: E402

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ExpertPlacementStrategy = Literal["uniform", "frequency"]
type MlaKvBW8Backend = Literal["marlin"]
type CapacityCommandRunner = Callable[
    [tuple[str, ...], float],
    subprocess.CompletedProcess[str],
]
type PrefaultCommandRunner = Callable[
    [tuple[str, ...], float],
    subprocess.CompletedProcess[str],
]
type ProcessIdentityReader = Callable[[int], ProcessIdentity]

TARGET_PROFILE: Final = "glm52_bf16_amxint4_pp1_tp2_dwagon_benchmark_v3"
ORDERED_GPU_UUIDS: Final = (
    "GPU-a442b72e-6727-6322-ba5d-5a9512b79886",
    "GPU-63a7760a-6164-0758-9228-03dbf35d721c",
)
EXPECTED_GPU_PCI_SUFFIXES: Final = (":16:00.0", ":d8:00.0")
PHYSICAL_CPUS: Final = tuple(range(112))
NUMA_NODES: Final = (0, 1)
CPU_INFER_THREADS: Final = 112
THREADPOOL_COUNT: Final = 2
MODEL_LAYER_COUNT: Final = 78
ROUTED_LAYER_COUNT: Final = 75
ROUTED_EXPERT_COUNT: Final = 256
TOTAL_ROUTED_EXPERT_POSITIONS: Final = ROUTED_LAYER_COUNT * ROUTED_EXPERT_COUNT
LOCAL_MLA_HEAD_COUNT: Final = 32
HYBRID_CHECKPOINT_MANIFEST_NAME: Final = "hybrid-checkpoint-manifest.json"
HYBRID_CHECKPOINT_MANIFEST_KIND: Final = "glm52_amxint4_ampere_w8a16_hybrid_checkpoint"
HYBRID_CHECKPOINT_MANIFEST_MAXIMUM_BYTES: Final = 8 * 1024 * 1024
_COMPACT_MLA_BACKEND_MARKER: Final = re.compile(
    r"\bTP(?P<rank>[01])\] Loaded compact MLA kv_b W8 "
    r"backend=(?P<backend>triton|marlin) "
    r"local_heads=(?P<local_heads>[0-9]+)"
)
EXPERT_BYTES_PER_TP_RANK: Final = 36 * 1024 * 1024
ADMITTED_CHUNKED_PREFILL_SIZES: Final = (2_048, 4_096, 8_192)

DEFAULT_RUNTIME_PYTHON: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/"
    "14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455/"
    "venv/bin/python"
)
DEFAULT_RUNTIME_INSTALL_RECEIPT: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/"
    "14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455/"
    "install-receipt.json"
)
DEFAULT_RUNTIME_INSTALL_RECEIPT_SHA256: Final = (
    "77ddc2f4c4f84b628a81d0d05868e0f973441f023da385483384abde2b0aa924"
)
DEFAULT_SOURCE_DIRECTORY: Final = "/var/lib/exo/sources/ktransformers-glm47-f9ca696"
DEFAULT_MODEL_PATH: Final = "/mnt/sanic/glm52"
DEFAULT_KTRANSFORMERS_WEIGHT_PATH: Final = "/mnt/sanic/glm52-AMXINT4"
DEFAULT_DWAGON_IP: Final = "192.168.40.24"
DEFAULT_DWAGON_SOCKET_INTERFACE: Final = "ens13f0np0"
DEFAULT_DISTRIBUTED_PORT: Final = 62700
DEFAULT_SERVICE_PORT: Final = 62710
DEFAULT_CONTEXT_LENGTH: Final = 9_216
DEFAULT_BENCHMARK_INPUT_TOKENS: Final = 7_744
DEFAULT_BENCHMARK_OUTPUT_TOKENS: Final = 128
DEFAULT_SERVER_RANDOM_SEED: Final = 20_260_725
ADMITTED_BENCHMARK_CONCURRENCIES: Final = (1, 2, 3)
DEFAULT_BENCHMARK_CONCURRENCIES: Final = (1, 2)
DEFAULT_SCHEDULER_TOKEN_HEADROOM: Final = 256
DEFAULT_MAXIMUM_TOTAL_TOKENS: Final = 16_000
DEFAULT_KV_CACHE_DTYPE: Final = "bfloat16"
DEFAULT_STATIC_MEMORY_FRACTION: Final = 0.933
DEFAULT_MINIMUM_PRELAUNCH_FREE_VRAM_MIB: Final = 20_480
DEFAULT_MINIMUM_POSTREADINESS_FREE_VRAM_MIB: Final = 512
MODEL_HEADER_BF16_GIB: Final = 34.6476
MTP_LAYER_78_BF16_GIB: Final = 18.5388
MODEL_HEADER_PER_TP_RANK_MIB: Final = math.ceil(MODEL_HEADER_BF16_GIB * 1_024 / 2)
MODEL_WORKSPACE_RESERVE_MIB: Final = 2_048
_RESULT_FILENAME: Final = "glm52-tp2-local-benchmark-result.json"
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024
_NVIDIA_SMI_TIMEOUT_SECONDS: Final = 10.0
_GPU_INVENTORY_COMMAND: Final = (
    "nvidia-smi",
    (
        "--query-gpu=uuid,index,pci.bus_id,compute_cap,memory.total,memory.free,"
        "memory.used"
    ),
    "--format=csv,noheader,nounits",
)
_GPU_COMPUTE_PROCESS_COMMAND: Final = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
)
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_SHARED_HOST_WEIGHT_MANIFEST_KIND: Final = "kt_shared_host_weights_manifest"
_SHARED_HOST_WEIGHT_CONTENT_KIND: Final = "kt_shared_host_weights_content"
_SHARED_HOST_WEIGHT_MANIFEST_MAXIMUM_BYTES: Final = 64 * 1024 * 1024
_SHARED_HOST_WEIGHT_PREFAULT_PLAN_KIND: Final = (
    "glm52_shared_host_weight_numa_prefault_plan"
)
_SHARED_HOST_WEIGHT_PREFAULT_RESULT_KIND: Final = (
    "glm52_shared_host_weight_numa_prefault_result"
)
_SHARED_HOST_WEIGHT_PREFAULT_PLAN_FILENAME: Final = (
    "shared-host-weight-prefault-plan.json"
)
_SAFETENSORS_HEADER_MAXIMUM_BYTES: Final = 64 * 1024 * 1024
_PREFAULT_READ_CHUNK_BYTES: Final = 8 * 1024 * 1024
_NUMACTL_EXECUTABLE: Final = Path("/usr/bin/numactl")
_EXPERT_RECORDER_DIRECTORY: Final = Path("/tmp")
_CONCURRENCY_TWO_PROMPT_MARKERS: Final = (
    "Jadeite",
    "Kestrel",
)
_CONCURRENCY_THREE_PROMPT_MARKERS: Final = (
    "Larkspur",
    "Mica",
    "Nuthatch",
)
_SPECULATIVE_ACCEPT_RATE_FIELD: Final = "spec_accept_rate"
_SPECULATIVE_ACCEPT_LENGTH_FIELD: Final = "spec_accept_length"
_SPECULATIVE_ACCEPT_TOKEN_COUNT_FIELD: Final = "spec_accept_token_num"
_SPECULATIVE_DRAFT_TOKEN_COUNT_FIELD: Final = "spec_draft_token_num"
_SPECULATIVE_VERIFICATION_COUNT_FIELD: Final = "spec_verify_ct"
_SPECULATIVE_ACCEPT_HISTOGRAM_FIELD: Final = "spec_accept_histogram"


class Glm52Tp2BenchmarkError(RuntimeError):
    """Raised when the local GLM-5.2 TP2 run lacks complete evidence."""


@dataclass(frozen=True, slots=True)
class TargetVerificationTiming:
    """One target-verification timing exposed by SGLang stream metadata."""

    field_name: str
    value: float


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingMetrics:
    """Strict optional speculative metrics accumulated from one request."""

    spec_accept_rate: float | None
    spec_accept_length: float | None
    spec_accept_token_num: int | None
    spec_draft_token_num: int | None
    spec_verify_ct: int | None
    spec_accept_histogram: tuple[int, ...] | None
    target_verification_timings: tuple[TargetVerificationTiming, ...]

    def merged_with(
        self,
        newer: SpeculativeDecodingMetrics,
    ) -> SpeculativeDecodingMetrics:
        timings_by_name = {
            timing.field_name: timing
            for timing in (
                *self.target_verification_timings,
                *newer.target_verification_timings,
            )
        }
        return SpeculativeDecodingMetrics(
            spec_accept_rate=(
                newer.spec_accept_rate
                if newer.spec_accept_rate is not None
                else self.spec_accept_rate
            ),
            spec_accept_length=(
                newer.spec_accept_length
                if newer.spec_accept_length is not None
                else self.spec_accept_length
            ),
            spec_accept_token_num=(
                newer.spec_accept_token_num
                if newer.spec_accept_token_num is not None
                else self.spec_accept_token_num
            ),
            spec_draft_token_num=(
                newer.spec_draft_token_num
                if newer.spec_draft_token_num is not None
                else self.spec_draft_token_num
            ),
            spec_verify_ct=(
                newer.spec_verify_ct
                if newer.spec_verify_ct is not None
                else self.spec_verify_ct
            ),
            spec_accept_histogram=(
                newer.spec_accept_histogram
                if newer.spec_accept_histogram is not None
                else self.spec_accept_histogram
            ),
            target_verification_timings=tuple(
                timings_by_name[name] for name in sorted(timings_by_name)
            ),
        )

    def receipt(self) -> JsonObject:
        return {
            "spec_accept_rate": self.spec_accept_rate,
            "spec_accept_length": self.spec_accept_length,
            "spec_accept_token_num": self.spec_accept_token_num,
            "spec_draft_token_num": self.spec_draft_token_num,
            "spec_verify_ct": self.spec_verify_ct,
            "spec_accept_histogram": (
                None
                if self.spec_accept_histogram is None
                else list(self.spec_accept_histogram)
            ),
            "target_verification_timings": {
                timing.field_name: timing.value
                for timing in self.target_verification_timings
            },
        }


@dataclass(frozen=True, slots=True)
class SpeculativeRequestMetricsObservation:
    request_index: int
    stream_events_with_metrics: int
    metrics: SpeculativeDecodingMetrics | None

    def receipt(self) -> JsonObject:
        return {
            "request_index": self.request_index,
            "stream_events_with_metrics": self.stream_events_with_metrics,
            "metrics": None if self.metrics is None else self.metrics.receipt(),
        }


@dataclass(frozen=True, slots=True)
class SpeculativeCaseMetricsObservation:
    concurrency: int
    requests: tuple[SpeculativeRequestMetricsObservation, ...]

    def receipt(self) -> JsonObject:
        return {
            "concurrency": self.concurrency,
            "requests": [request.receipt() for request in self.requests],
        }


def _is_target_verification_timing_field(field_name: str) -> bool:
    normalized = field_name.lower()
    verification_scoped = ("verify" in normalized or "verification" in normalized) and (
        "spec" in normalized or "target" in normalized
    )
    timing_scoped = any(
        marker in normalized for marker in ("_time", "_latency", "_duration")
    )
    return verification_scoped and timing_scoped and "_timeout" not in normalized


def _optional_nonnegative_integer_metric(
    meta_info: Mapping[str, JsonValue],
    field_name: str,
) -> int | None:
    if field_name not in meta_info:
        return None
    raw_value = meta_info[field_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 0:
        raise Glm52Tp2BenchmarkError(
            f"SGLang stream meta_info field {field_name!r} must be "
            "a nonnegative integer"
        )
    return raw_value


def _optional_nonnegative_float_metric(
    meta_info: Mapping[str, JsonValue],
    field_name: str,
    *,
    maximum: float | None = None,
) -> float | None:
    if field_name not in meta_info:
        return None
    raw_value = meta_info[field_name]
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise Glm52Tp2BenchmarkError(
            f"SGLang stream meta_info field {field_name!r} must be numeric"
        )
    value = float(raw_value)
    if (
        not math.isfinite(value)
        or value < 0.0
        or (maximum is not None and value > maximum)
    ):
        bound = (
            "nonnegative and finite"
            if maximum is None
            else f"finite and between 0 and {maximum:g}"
        )
        raise Glm52Tp2BenchmarkError(
            f"SGLang stream meta_info field {field_name!r} must be {bound}"
        )
    return value


def parse_speculative_decoding_metrics(
    meta_info: Mapping[str, JsonValue],
) -> SpeculativeDecodingMetrics | None:
    """Parse only speculative fields actually present in one SSE event."""

    known_fields = {
        _SPECULATIVE_ACCEPT_RATE_FIELD,
        _SPECULATIVE_ACCEPT_LENGTH_FIELD,
        _SPECULATIVE_ACCEPT_TOKEN_COUNT_FIELD,
        _SPECULATIVE_DRAFT_TOKEN_COUNT_FIELD,
        _SPECULATIVE_VERIFICATION_COUNT_FIELD,
        _SPECULATIVE_ACCEPT_HISTOGRAM_FIELD,
    }
    timing_field_names = tuple(
        sorted(
            field_name
            for field_name in meta_info
            if _is_target_verification_timing_field(field_name)
        )
    )
    if not any(field_name in meta_info for field_name in known_fields) and not (
        timing_field_names
    ):
        return None

    raw_histogram = meta_info.get(_SPECULATIVE_ACCEPT_HISTOGRAM_FIELD)
    histogram: tuple[int, ...] | None = None
    if _SPECULATIVE_ACCEPT_HISTOGRAM_FIELD in meta_info:
        if not isinstance(raw_histogram, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_histogram
        ):
            raise Glm52Tp2BenchmarkError(
                "SGLang stream meta_info field "
                f"{_SPECULATIVE_ACCEPT_HISTOGRAM_FIELD!r} must be a list of "
                "nonnegative integers"
            )
        histogram = tuple(cast(list[int], raw_histogram))

    timings = tuple(
        TargetVerificationTiming(
            field_name=field_name,
            value=cast(
                float,
                _optional_nonnegative_float_metric(meta_info, field_name),
            ),
        )
        for field_name in timing_field_names
    )
    return SpeculativeDecodingMetrics(
        spec_accept_rate=_optional_nonnegative_float_metric(
            meta_info,
            _SPECULATIVE_ACCEPT_RATE_FIELD,
            maximum=1.0,
        ),
        spec_accept_length=_optional_nonnegative_float_metric(
            meta_info,
            _SPECULATIVE_ACCEPT_LENGTH_FIELD,
        ),
        spec_accept_token_num=_optional_nonnegative_integer_metric(
            meta_info,
            _SPECULATIVE_ACCEPT_TOKEN_COUNT_FIELD,
        ),
        spec_draft_token_num=_optional_nonnegative_integer_metric(
            meta_info,
            _SPECULATIVE_DRAFT_TOKEN_COUNT_FIELD,
        ),
        spec_verify_ct=_optional_nonnegative_integer_metric(
            meta_info,
            _SPECULATIVE_VERIFICATION_COUNT_FIELD,
        ),
        spec_accept_histogram=histogram,
        target_verification_timings=timings,
    )


class SpeculativeMetricsRecorder:
    """Thread-safe per-case accumulator for concurrent SGLang SSE streams."""

    def __init__(self, concurrency: int) -> None:
        if concurrency <= 0:
            raise ValueError("speculative metrics concurrency must be positive")
        self._concurrency = concurrency
        self._lock = Lock()
        self._metrics_by_request: dict[int, SpeculativeDecodingMetrics] = {}
        self._event_counts_by_request: dict[int, int] = {}

    def observe(
        self,
        request_index: int,
        meta_info: Mapping[str, JsonValue],
    ) -> None:
        if not 0 <= request_index < self._concurrency:
            raise Glm52Tp2BenchmarkError(
                f"speculative metrics request index {request_index} is outside "
                f"concurrency {self._concurrency}"
            )
        parsed = parse_speculative_decoding_metrics(meta_info)
        if parsed is None:
            return
        with self._lock:
            prior = self._metrics_by_request.get(request_index)
            self._metrics_by_request[request_index] = (
                parsed if prior is None else prior.merged_with(parsed)
            )
            self._event_counts_by_request[request_index] = (
                self._event_counts_by_request.get(request_index, 0) + 1
            )

    def observation(self) -> SpeculativeCaseMetricsObservation:
        with self._lock:
            return SpeculativeCaseMetricsObservation(
                concurrency=self._concurrency,
                requests=tuple(
                    SpeculativeRequestMetricsObservation(
                        request_index=request_index,
                        stream_events_with_metrics=(
                            self._event_counts_by_request.get(request_index, 0)
                        ),
                        metrics=self._metrics_by_request.get(request_index),
                    )
                    for request_index in range(self._concurrency)
                ),
            )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    run_id: str
    result_directory: Path
    runtime_python: str
    runtime_install_receipt: Path
    runtime_install_receipt_sha256: str
    local_source_directory: str
    model_path: str
    ktransformers_weight_path: str
    dwagon_ip: str
    dwagon_socket_interface: str
    distributed_port: int
    service_port: int
    benchmark_concurrencies: tuple[int, ...]
    context_length: int
    maximum_total_tokens: int
    benchmark_input_tokens: int
    benchmark_output_tokens: int
    chunked_prefill_size: int
    resident_gpu_expert_budget_total: int
    kt_gpu_experts_ratio: str | None
    expert_placement_strategy: ExpertPlacementStrategy
    init_expert_location: Path | None
    init_expert_location_sha256: str | None
    kv_cache_dtype: Literal["bfloat16", "fp8_e4m3"]
    mla_kv_b_w8_backend: MlaKvBW8Backend
    enable_two_batch_overlap: bool
    enable_amx_fine_grained_decode: bool
    enable_stream_prefill: bool
    stream_prefill_token_threshold: int
    stream_prefill_experts_per_chunk: int
    enable_mtp: bool
    enable_shared_host_weights: bool
    shared_host_weights_manifest: Path | None
    shared_host_weights_content_id: str | None
    shared_host_weights_state_directory: Path | None
    capture_representative_routing: bool
    static_memory_fraction: float
    minimum_prelaunch_free_vram_mib: int
    minimum_postreadiness_free_vram_mib: int
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Glm52Tp2ProcessSpec(tp2.Tp2LocalProcessSpec):
    """Exact inert launch description for the current dwagon TP2 topology."""

    target_profile: str = TARGET_PROFILE
    ordered_gpu_uuids: tuple[str, str] = ORDERED_GPU_UUIDS
    resident_gpu_experts: int = 0
    ktransformers_weight_path: str = DEFAULT_KTRANSFORMERS_WEIGHT_PATH
    context_length: int = DEFAULT_CONTEXT_LENGTH
    maximum_total_tokens: int = DEFAULT_MAXIMUM_TOTAL_TOKENS
    maximum_running_requests: int = 2
    kv_cache_dtype: Literal["bfloat16", "fp8_e4m3"] = DEFAULT_KV_CACHE_DTYPE
    mla_kv_b_w8_backend: MlaKvBW8Backend = "marlin"
    enable_two_batch_overlap: bool = False
    enable_amx_fine_grained_decode: bool = False
    enable_stream_prefill: bool = False
    stream_prefill_token_threshold: int = 4_096
    stream_prefill_experts_per_chunk: int = 4
    enable_mtp: bool = False
    enable_shared_host_weights: bool = False
    shared_host_weights_manifest: Path | None = None
    shared_host_weights_content_id: str | None = None
    shared_host_weights_state_directory: Path | None = None
    capture_representative_routing: bool = False
    chunked_prefill_size: int = 2_048
    resident_gpu_expert_budget_total: int = 0
    kt_gpu_experts_ratio: str | None = None
    expert_placement_strategy: ExpertPlacementStrategy = "uniform"
    init_expert_location: Path | None = None
    init_expert_location_sha256: str | None = None

    def __post_init__(self) -> None:
        exact_topology = (
            self.target_profile == TARGET_PROFILE
            and self.pipeline_rank == 0
            and self.pipeline_parallel_size == 1
            and self.tensor_parallel_size == 2
            and self.node_count == 1
            and self.node_rank == 0
            and self.ordered_gpu_uuids == ORDERED_GPU_UUIDS
            and self.cpu_cores == PHYSICAL_CPUS
            and self.memory_nodes == NUMA_NODES
            and self.cpu_infer_threads == CPU_INFER_THREADS
            and self.threadpool_count == THREADPOOL_COUNT
            and self.resident_gpu_experts == 0
        )
        if not exact_topology:
            raise ValueError(
                "GLM-5.2 TP2 process spec differs from the current dwagon topology"
            )
        for path_name, raw_path in (
            ("runtime executable", self.executable),
            ("model path", self.model_path),
            ("KTransformers weight path", self.ktransformers_weight_path),
        ):
            if not Path(raw_path).is_absolute():
                raise ValueError(f"{path_name} must be absolute")
        if not 0.8 <= self.static_memory_fraction <= 0.95:
            raise ValueError(
                "GLM-5.2 TP2 static memory fraction must be between 0.8 and 0.95"
            )
        if self.service_endpoint == self.distributed_coordinator:
            raise ValueError("service and distributed endpoints must be distinct")
        if self.context_length <= 0 or self.maximum_total_tokens <= 0:
            raise ValueError("token capacities must be positive")
        if self.kv_cache_dtype not in {"bfloat16", "fp8_e4m3"}:
            raise ValueError("unsupported GLM-5.2 KV cache dtype")
        if self.mla_kv_b_w8_backend != "marlin":
            raise ValueError(
                "the focused GLM-5.2 TP2 harness requires the Marlin compact "
                "MLA kv_b W8 backend"
            )
        if self.maximum_running_requests not in ADMITTED_BENCHMARK_CONCURRENCIES:
            raise ValueError(
                "the focused GLM-5.2 TP2 harness requires concurrency 1, 2, or 3"
            )
        if self.chunked_prefill_size not in ADMITTED_CHUNKED_PREFILL_SIZES:
            raise ValueError("GLM-5.2 chunked prefill must be 2048, 4096, or 8192")
        if (
            not 0
            <= self.resident_gpu_expert_budget_total
            <= (TOTAL_ROUTED_EXPERT_POSITIONS)
        ):
            raise ValueError("resident GPU expert total is outside GLM-5.2 bounds")
        if self.resident_gpu_expert_budget_total == 0:
            if (
                self.kt_gpu_experts_ratio is not None
                or self.expert_placement_strategy != "uniform"
                or self.init_expert_location is not None
                or self.init_expert_location_sha256 is not None
            ):
                raise ValueError(
                    "zero-resident profile must use uniform placement without "
                    "a ratio or frequency artifact"
                )
        else:
            if self.kt_gpu_experts_ratio is None:
                raise ValueError("resident profile requires an exact KT ratio")
            try:
                ratio = float(self.kt_gpu_experts_ratio)
            except ValueError as error:
                raise ValueError("KT GPU expert ratio is not numeric") from error
            if (
                not math.isfinite(ratio)
                or not 0.0 < ratio <= 1.0
                or int(ratio * TOTAL_ROUTED_EXPERT_POSITIONS)
                != self.resident_gpu_expert_budget_total
            ):
                raise ValueError(
                    "KT GPU expert ratio does not encode the pinned total budget"
                )
            frequency_contract = (
                self.init_expert_location,
                self.init_expert_location_sha256,
            )
            if self.expert_placement_strategy == "frequency":
                if any(value is None for value in frequency_contract):
                    raise ValueError("frequency placement requires a path and SHA-256")
                assert self.init_expert_location is not None
                if not self.init_expert_location.is_absolute():
                    raise ValueError("frequency placement input path must be absolute")
                if (
                    self.init_expert_location_sha256 is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        self.init_expert_location_sha256,
                    )
                    is None
                ):
                    raise ValueError("frequency placement input SHA-256 is invalid")
            elif any(value is not None for value in frequency_contract):
                raise ValueError("uniform placement cannot carry a frequency artifact")
        if self.enable_stream_prefill and (
            self.stream_prefill_token_threshold <= 0
            or self.stream_prefill_experts_per_chunk not in {1, 2, 4, 8, 16}
        ):
            raise ValueError("invalid bounded stream-prefill configuration")
        shared_contract = (
            self.shared_host_weights_manifest,
            self.shared_host_weights_content_id,
            self.shared_host_weights_state_directory,
        )
        if self.enable_shared_host_weights:
            if any(value is None for value in shared_contract):
                raise ValueError(
                    "shared host weights require manifest, content ID, and state"
                )
            if (
                self.shared_host_weights_content_id is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    self.shared_host_weights_content_id,
                )
                is None
            ):
                raise ValueError("shared host weight content ID must be SHA-256")
            for path in (
                self.shared_host_weights_manifest,
                self.shared_host_weights_state_directory,
            ):
                if path is None or not path.is_absolute():
                    raise ValueError("shared host weight paths must be absolute")
        elif any(value is not None for value in shared_contract):
            raise ValueError(
                "shared host weight contract was supplied without enabling it"
            )

    @property
    def command(self) -> tuple[str, ...]:
        return (
            self.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--kt-weight-path",
            self.ktransformers_weight_path,
            "--kt-method",
            "AMXINT4",
            "--kt-cpuinfer",
            str(self.cpu_infer_threads),
            "--kt-threadpool-count",
            str(self.threadpool_count),
            "--kt-numa-nodes",
            *(str(memory_node) for memory_node in self.memory_nodes),
            *(
                ("--kt-num-gpu-experts", "0")
                if self.resident_gpu_expert_budget_total == 0
                else (
                    "--kt-gpu-experts-ratio",
                    cast(str, self.kt_gpu_experts_ratio),
                )
            ),
            "--kt-max-deferred-experts-per-token",
            "0",
            "--kt-expert-placement-strategy",
            self.expert_placement_strategy,
            *(
                (
                    "--init-expert-location",
                    str(self.init_expert_location),
                )
                if self.expert_placement_strategy == "frequency"
                else ()
            ),
            *(
                (
                    "--kt-gpu-prefill-token-threshold",
                    str(self.stream_prefill_token_threshold),
                    "--kt-stream-prefill",
                    "--kt-stream-prefill-experts-per-chunk",
                    str(self.stream_prefill_experts_per_chunk),
                    "--kt-stream-prefill-ring-slots",
                    "2",
                    "--kt-stream-prefill-safety-margin-mb",
                    "512",
                )
                if self.enable_stream_prefill
                else ()
            ),
            "--pp-size",
            "1",
            "--tp-size",
            "2",
            "--nnodes",
            "1",
            "--node-rank",
            "0",
            "--dist-init-addr",
            str(self.distributed_coordinator),
            "--host",
            self.service_endpoint.ip,
            "--port",
            str(self.service_endpoint.port),
            "--context-length",
            str(self.context_length),
            "--max-total-tokens",
            str(self.maximum_total_tokens),
            "--mem-fraction-static",
            str(self.static_memory_fraction),
            "--max-running-requests",
            str(self.maximum_running_requests),
            "--chunked-prefill-size",
            str(self.chunked_prefill_size),
            "--attention-backend",
            "flashinfer",
            "--kv-cache-dtype",
            self.kv_cache_dtype,
            *(() if self.enable_mtp else ("--load-format", "safetensors")),
            "--random-seed",
            str(DEFAULT_SERVER_RANDOM_SEED),
            "--moe-a2a-backend",
            "none",
            *(("--enable-two-batch-overlap",) if self.enable_two_batch_overlap else ()),
            *(
                (
                    "--load-format",
                    "safetensors",
                    "--ep-size",
                    "1",
                    "--speculative-algorithm",
                    "NEXTN",
                    "--speculative-draft-model-path",
                    self.model_path,
                    "--speculative-draft-load-format",
                    "safetensors",
                    "--speculative-num-steps",
                    "1",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "2",
                    "--speculative-moe-a2a-backend",
                    "none",
                )
                if self.enable_mtp
                else ()
            ),
            "--disable-cuda-graph",
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--served-model-name",
            glm52.SERVED_MODEL_NAME,
            "--watchdog-timeout",
            "3000",
            "--trust-remote-code",
            *(
                (
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--expert-distribution-recorder-buffer-size",
                    "-1",
                )
                if self.capture_representative_routing
                else ()
            ),
            "--disable-radix-cache",
        )

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        environment: tuple[tuple[str, str], ...] = (
            ("CUDA_VISIBLE_DEVICES", ",".join(self.ordered_gpu_uuids)),
            ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
            ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
            ("SGLANG_MLA_KV_B_W8_BACKEND", self.mla_kv_b_w8_backend),
        )
        if self.enable_amx_fine_grained_decode:
            environment = (*environment, ("KT_AMX_FINE_GRAINED_DECODE", "1"))
        if self.enable_shared_host_weights:
            assert self.shared_host_weights_manifest is not None
            assert self.shared_host_weights_content_id is not None
            assert self.shared_host_weights_state_directory is not None
            environment = (
                *environment,
                ("KT_SHARED_HOST_WEIGHTS", "1"),
                (
                    "KT_SHARED_HOST_WEIGHTS_MANIFEST",
                    str(self.shared_host_weights_manifest),
                ),
                (
                    "KT_SHARED_HOST_WEIGHTS_CONTENT_ID",
                    self.shared_host_weights_content_id,
                ),
                (
                    "KT_SHARED_HOST_WEIGHTS_STATE_DIR",
                    str(self.shared_host_weights_state_directory),
                ),
            )
        return environment

    def receipt(self) -> JsonObject:
        return {
            "schema_version": 1,
            "target_profile": self.target_profile,
            "model": {
                "model_path": self.model_path,
                "ktransformers_weight_path": self.ktransformers_weight_path,
                "compute_dtype": "bfloat16",
                "activation_dtype": "bfloat16",
                "weight_storage": "checkpoint_contract",
                "kv_cache_dtype": self.kv_cache_dtype,
                "mla_kv_b_w8_backend": self.mla_kv_b_w8_backend,
                "ktransformers_method": "AMXINT4",
                "causal_layer_range": [0, MODEL_LAYER_COUNT],
                "mtp_layer_78_loaded": self.enable_mtp,
            },
            "parallelism": {
                "parent_process_count": 1,
                "pipeline_parallel_size": 1,
                "tensor_parallel_size": 2,
                "node_count": 1,
                "node_rank": 0,
            },
            "gpu_workers": [
                {
                    "tensor_parallel_rank": rank,
                    "gpu_uuid": gpu_uuid,
                    "expected_pci_bus_suffix": EXPECTED_GPU_PCI_SUFFIXES[rank],
                    "matching_numa_node": rank,
                    "numa_local_physical_cpu_ids": list(
                        range(rank * 56, (rank + 1) * 56)
                    ),
                }
                for rank, gpu_uuid in enumerate(self.ordered_gpu_uuids)
            ],
            "cpu": {
                "physical_cpu_ids": list(self.cpu_cores),
                "numa_nodes": list(self.memory_nodes),
                "cpu_infer_threads": self.cpu_infer_threads,
                "threadpool_count": self.threadpool_count,
            },
            "experts": {
                "resident_gpu_experts": 0,
                "resident_gpu_expert_budget_scope": "global_across_routed_layers",
                "resident_gpu_expert_budget_total": (
                    self.resident_gpu_expert_budget_total
                ),
                "kt_gpu_experts_ratio": self.kt_gpu_experts_ratio,
                "resident_bytes_per_tp_rank": (
                    self.resident_gpu_expert_budget_total * EXPERT_BYTES_PER_TP_RANK
                ),
                "max_deferred_experts_per_token": 0,
                "placement_strategy": self.expert_placement_strategy,
                "frequency_input": (
                    None
                    if self.init_expert_location is None
                    else {
                        "path": str(self.init_expert_location),
                        "sha256": self.init_expert_location_sha256,
                    }
                ),
                "deferred_expert_lane": "excluded_as_approximate_model",
            },
            "capacity": {
                "context_length": self.context_length,
                "maximum_total_tokens": self.maximum_total_tokens,
                "chunked_prefill_size": self.chunked_prefill_size,
                "scheduler_token_headroom": DEFAULT_SCHEDULER_TOKEN_HEADROOM,
                "maximum_running_requests": self.maximum_running_requests,
                "bf16_non_routed_non_mtp_header_gib_total": MODEL_HEADER_BF16_GIB,
                "estimated_header_mib_per_tp_rank": MODEL_HEADER_PER_TP_RANK_MIB,
                "excluded_mtp_layer_78_bf16_gib": MTP_LAYER_78_BF16_GIB,
            },
            "service_endpoint": self.service_endpoint.model_dump(mode="json"),
            "distributed_coordinator": self.distributed_coordinator.model_dump(
                mode="json"
            ),
            "static_memory_fraction": self.static_memory_fraction,
            "paper_optimizations": {
                "two_batch_attention_moe_overlap": self.enable_two_batch_overlap,
                "fine_grained_amx_decode_dependencies": (
                    self.enable_amx_fine_grained_decode
                ),
                "bounded_stream_loading_prefill": self.enable_stream_prefill,
                "kt_backed_mtp": self.enable_mtp,
                "zero_copy_shared_host_weights": (self.enable_shared_host_weights),
                "representative_route_capture": (self.capture_representative_routing),
            },
            "argv": list(self.command),
            "environment": {name: value for name, value in self.environment},
        }


@dataclass(frozen=True, slots=True)
class GpuMemoryDevice:
    uuid: str
    index: int
    pci_bus_id: str
    compute_capability: str
    total_mib: int
    free_mib: int
    used_mib: int


@dataclass(frozen=True, slots=True)
class GpuComputeProcess:
    gpu_uuid: str
    pid: int
    used_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class GpuCapacitySnapshot:
    observed_at_utc: str
    inventory_command: tuple[str, ...]
    inventory_stdout_sha256: str
    compute_process_command: tuple[str, ...]
    compute_process_stdout_sha256: str
    devices: tuple[GpuMemoryDevice, ...]
    compute_processes: tuple[GpuComputeProcess, ...]


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    parent_pid: int
    process_group_id: int
    session_id: int
    start_time_ticks: int
    environment_entries: frozenset[bytes]


@dataclass(frozen=True, slots=True)
class SharedHostWeightPrefaultFile:
    relative_path: str
    absolute_path: Path
    size_bytes: int
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class SharedHostWeightPrefaultExtent:
    offset_bytes: int
    length_bytes: int
    tensor_count: int


@dataclass(frozen=True, slots=True)
class SharedHostWeightPrefaultNodeSummary:
    numa_node: int
    file_count: int
    extent_count: int
    tensor_count: int
    expected_bytes: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256_bytes(encoded)


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _shared_weight_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest path must be a non-empty string"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
        or "\x00" in value
    ):
        raise Glm52Tp2BenchmarkError(
            f"shared-host-weight manifest path is unsafe: {value!r}"
        )
    return value


def _load_shared_host_weight_prefault_files(
    config: BenchmarkConfig,
) -> tuple[tuple[SharedHostWeightPrefaultFile, ...], str, tuple[int, ...]]:
    manifest_path = config.shared_host_weights_manifest
    expected_content_id = config.shared_host_weights_content_id
    if manifest_path is None or expected_content_id is None:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault lacks its manifest identity"
        )
    try:
        manifest_status_before = manifest_path.lstat()
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            f"shared-host-weight manifest is unavailable: {manifest_path}"
        ) from error
    if (
        manifest_path.is_symlink()
        or not stat.S_ISREG(manifest_status_before.st_mode)
        or manifest_status_before.st_size <= 0
        or manifest_status_before.st_size > _SHARED_HOST_WEIGHT_MANIFEST_MAXIMUM_BYTES
    ):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest must be a bounded regular non-symlink file"
        )
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest_value = cast(object, json.loads(manifest_raw))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest is unreadable or invalid"
        ) from error
    manifest_status_after = manifest_path.lstat()
    if _stable_file_identity(manifest_status_before) != _stable_file_identity(
        manifest_status_after
    ):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest changed while being read"
        )
    if not isinstance(manifest_value, dict):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest must be a JSON object"
        )
    manifest = cast(JsonObject, manifest_value)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != _SHARED_HOST_WEIGHT_MANIFEST_KIND
    ):
        raise Glm52Tp2BenchmarkError("unsupported shared-host-weight manifest schema")
    content_id = manifest.get("content_id")
    if content_id != expected_content_id:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest content ID differs from the "
            "configured identity"
        )
    raw_numa_nodes = manifest.get("numa_nodes")
    if not isinstance(raw_numa_nodes, list) or raw_numa_nodes != list(NUMA_NODES):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest NUMA order must be [0, 1]"
        )
    numa_nodes = tuple(cast(list[int], raw_numa_nodes))
    raw_files_value = manifest.get("files")
    if not isinstance(raw_files_value, list) or not raw_files_value:
        raise Glm52Tp2BenchmarkError("shared-host-weight manifest has no files")

    checkpoint_root = Path(config.ktransformers_weight_path).resolve(strict=True)
    if not checkpoint_root.is_dir():
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight checkpoint root is not a directory"
        )
    content_rows: list[JsonValue] = []
    files: list[SharedHostWeightPrefaultFile] = []
    for raw_file in cast(list[object], raw_files_value):
        if not isinstance(raw_file, dict):
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight manifest file entry is not an object"
            )
        file_value = cast(dict[str, object], raw_file)
        relative_path = _shared_weight_relative_path(file_value.get("path"))
        size_bytes = file_value.get("size_bytes")
        sha256 = file_value.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise Glm52Tp2BenchmarkError(
                f"shared-host-weight manifest metadata is invalid: {relative_path}"
            )
        candidate = checkpoint_root.joinpath(*PurePosixPath(relative_path).parts)
        current = checkpoint_root
        try:
            for part in PurePosixPath(relative_path).parts:
                current /= part
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise Glm52Tp2BenchmarkError(
                        "shared-host-weight checkpoint path traverses a "
                        f"symlink: {relative_path}"
                    )
            resolved = candidate.resolve(strict=True)
            _ = resolved.relative_to(checkpoint_root)
            status = resolved.lstat()
        except (OSError, ValueError) as error:
            raise Glm52Tp2BenchmarkError(
                f"shared-host-weight checkpoint file is unavailable: {relative_path}"
            ) from error
        if not stat.S_ISREG(status.st_mode) or status.st_size != size_bytes:
            raise Glm52Tp2BenchmarkError(
                f"shared-host-weight checkpoint file changed size or type: "
                f"{relative_path}"
            )
        mount_read_only = bool(os.statvfs(resolved).f_flag & os.ST_RDONLY)
        if not mount_read_only and status.st_mode & 0o222:
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight checkpoint must be immutable before "
                f"prefault: {relative_path}"
            )
        content_rows.append(
            {
                "path": relative_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
        files.append(
            SharedHostWeightPrefaultFile(
                relative_path=relative_path,
                absolute_path=resolved,
                size_bytes=size_bytes,
                sha256=sha256,
                identity=_stable_file_identity(status),
            )
        )
    relative_paths = tuple(file.relative_path for file in files)
    if relative_paths != tuple(sorted(relative_paths)) or len(
        set(relative_paths)
    ) != len(relative_paths):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight manifest paths are not unique and sorted"
        )
    calculated_content_id = _canonical_sha256(
        {
            "files": content_rows,
            "kind": _SHARED_HOST_WEIGHT_CONTENT_KIND,
            "schema_version": 1,
        }
    )
    if calculated_content_id != expected_content_id:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight content ID does not match its file table"
        )
    return tuple(files), _sha256_bytes(manifest_raw), numa_nodes


def _pread_exact(
    file_descriptor: int,
    length: int,
    offset: int,
    description: str,
) -> bytes:
    contents = bytearray()
    while len(contents) < length:
        chunk = os.pread(
            file_descriptor,
            min(1024 * 1024, length - len(contents)),
            offset + len(contents),
        )
        if not chunk:
            raise Glm52Tp2BenchmarkError(f"short read while reading {description}")
        contents.extend(chunk)
    return bytes(contents)


def _safetensors_numa_extents(
    file: SharedHostWeightPrefaultFile,
    *,
    include_mtp: bool,
) -> dict[int, tuple[SharedHostWeightPrefaultExtent, ...]]:
    result: dict[int, list[SharedHostWeightPrefaultExtent]] = {
        node: [] for node in NUMA_NODES
    }
    if file.absolute_path.suffix != ".safetensors":
        return {node: () for node in NUMA_NODES}
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(file.absolute_path, flags)
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            f"cannot open safetensors shard for prefault planning: {file.relative_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if _stable_file_identity(before) != file.identity:
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight shard changed before header parsing: "
                f"{file.relative_path}"
            )
        header_length = int.from_bytes(
            _pread_exact(
                descriptor,
                8,
                0,
                f"{file.relative_path} safetensors header length",
            ),
            byteorder="little",
        )
        if (
            header_length <= 0
            or header_length > _SAFETENSORS_HEADER_MAXIMUM_BYTES
            or 8 + header_length > before.st_size
        ):
            raise Glm52Tp2BenchmarkError(
                f"invalid safetensors header length in {file.relative_path}"
            )
        try:
            raw_header = cast(
                object,
                json.loads(
                    _pread_exact(
                        descriptor,
                        header_length,
                        8,
                        f"{file.relative_path} safetensors header",
                    )
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise Glm52Tp2BenchmarkError(
                f"invalid safetensors header JSON in {file.relative_path}"
            ) from error
        if not isinstance(raw_header, dict):
            raise Glm52Tp2BenchmarkError(
                f"safetensors header is not an object: {file.relative_path}"
            )
        data_start = 8 + header_length
        all_ranges: list[tuple[int, int]] = []
        for tensor_name, raw_metadata in cast(
            dict[object, object],
            raw_header,
        ).items():
            if tensor_name == "__metadata__":
                continue
            if not isinstance(tensor_name, str) or not isinstance(raw_metadata, dict):
                raise Glm52Tp2BenchmarkError(
                    f"invalid safetensors tensor row in {file.relative_path}"
                )
            metadata = cast(dict[object, object], raw_metadata)
            raw_offsets = metadata.get("data_offsets")
            if (
                not isinstance(raw_offsets, list)
                or len(raw_offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_offsets
                )
            ):
                raise Glm52Tp2BenchmarkError(
                    f"invalid tensor offsets in {file.relative_path}"
                )
            begin, end = cast(list[int], raw_offsets)
            if begin < 0 or end <= begin or data_start + end > before.st_size:
                raise Glm52Tp2BenchmarkError(
                    f"out-of-range tensor extent in {file.relative_path}"
                )
            all_ranges.append((begin, end))
            matching_nodes = [
                node for node in NUMA_NODES if f".numa.{node}." in tensor_name
            ]
            if ".numa." in tensor_name and len(matching_nodes) != 1:
                raise Glm52Tp2BenchmarkError(
                    f"unknown NUMA tensor name in {file.relative_path}: {tensor_name}"
                )
            if not matching_nodes:
                continue
            if not include_mtp and tensor_name.startswith("blk.78."):
                continue
            node = matching_nodes[0]
            absolute_begin = data_start + begin
            length = end - begin
            extents = result[node]
            if (
                extents
                and extents[-1].offset_bytes + extents[-1].length_bytes
                == absolute_begin
            ):
                previous = extents[-1]
                extents[-1] = SharedHostWeightPrefaultExtent(
                    offset_bytes=previous.offset_bytes,
                    length_bytes=previous.length_bytes + length,
                    tensor_count=previous.tensor_count + 1,
                )
            else:
                extents.append(
                    SharedHostWeightPrefaultExtent(
                        offset_bytes=absolute_begin,
                        length_bytes=length,
                        tensor_count=1,
                    )
                )
        expected_begin = 0
        for begin, end in sorted(all_ranges):
            if begin != expected_begin:
                raise Glm52Tp2BenchmarkError(
                    f"safetensors data ranges have a gap or overlap: "
                    f"{file.relative_path}"
                )
            expected_begin = end
        if data_start + expected_begin != before.st_size:
            raise Glm52Tp2BenchmarkError(
                f"safetensors data ranges do not cover {file.relative_path}"
            )
        after = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(after):
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight shard changed during header parsing: "
                f"{file.relative_path}"
            )
    finally:
        os.close(descriptor)
    return {node: tuple(result[node]) for node in NUMA_NODES}


def _write_prefault_plan(
    config: BenchmarkConfig,
    files: tuple[SharedHostWeightPrefaultFile, ...],
    *,
    manifest_sha256: str,
    numa_nodes: tuple[int, ...],
) -> tuple[Path, str, str, tuple[SharedHostWeightPrefaultNodeSummary, ...], int]:
    node_file_rows: dict[int, list[JsonValue]] = {node: [] for node in numa_nodes}
    summaries: list[SharedHostWeightPrefaultNodeSummary] = []
    node_totals: dict[int, list[int]] = {node: [0, 0, 0, 0] for node in numa_nodes}
    for file in files:
        extents_by_node = _safetensors_numa_extents(
            file,
            include_mtp=config.enable_mtp,
        )
        for node in numa_nodes:
            extents = extents_by_node[node]
            if not extents:
                continue
            expected_bytes = sum(extent.length_bytes for extent in extents)
            tensor_count = sum(extent.tensor_count for extent in extents)
            node_file_rows[node].append(
                {
                    "absolute_path": str(file.absolute_path),
                    "relative_path": file.relative_path,
                    "size_bytes": file.size_bytes,
                    "identity": list(file.identity),
                    "extent_count": len(extents),
                    "tensor_count": tensor_count,
                    "expected_bytes": expected_bytes,
                    "extents": [
                        {
                            "offset_bytes": extent.offset_bytes,
                            "length_bytes": extent.length_bytes,
                            "tensor_count": extent.tensor_count,
                        }
                        for extent in extents
                    ],
                }
            )
            node_totals[node][0] += 1
            node_totals[node][1] += len(extents)
            node_totals[node][2] += tensor_count
            node_totals[node][3] += expected_bytes
    for node in numa_nodes:
        file_count, extent_count, tensor_count, expected_bytes = node_totals[node]
        if not file_count or not extent_count or not expected_bytes:
            raise Glm52Tp2BenchmarkError(
                f"shared-host-weight prefault has no NUMA {node} extents"
            )
        summaries.append(
            SharedHostWeightPrefaultNodeSummary(
                numa_node=node,
                file_count=file_count,
                extent_count=extent_count,
                tensor_count=tensor_count,
                expected_bytes=expected_bytes,
            )
        )
    if len({summary.expected_bytes for summary in summaries}) != 1:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight NUMA prefault byte counts are asymmetric"
        )
    selected_bytes = sum(summary.expected_bytes for summary in summaries)
    artifact_bytes = sum(file.size_bytes for file in files)
    plan: JsonObject = {
        "schema_version": 1,
        "kind": _SHARED_HOST_WEIGHT_PREFAULT_PLAN_KIND,
        "manifest": str(config.shared_host_weights_manifest),
        "manifest_sha256": manifest_sha256,
        "content_id": config.shared_host_weights_content_id,
        "include_mtp_layer_78": config.enable_mtp,
        "artifact_file_count": len(files),
        "artifact_bytes": artifact_bytes,
        "selected_bytes": selected_bytes,
        "skipped_bytes": artifact_bytes - selected_bytes,
        "nodes": [
            {
                **cast(JsonObject, asdict(summary)),
                "files": node_file_rows[summary.numa_node],
            }
            for summary in summaries
        ],
    }
    plan["plan_content_sha256"] = _canonical_sha256(plan)
    encoded = (
        json.dumps(
            plan,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    if len(encoded) > _SHARED_HOST_WEIGHT_MANIFEST_MAXIMUM_BYTES:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault plan exceeds its safety bound"
        )
    plan_path = config.result_directory / _SHARED_HOST_WEIGHT_PREFAULT_PLAN_FILENAME
    try:
        descriptor = os.open(
            plan_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o400,
        )
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            f"cannot create immutable shared-host-weight prefault plan: {plan_path}"
        ) from error
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise Glm52Tp2BenchmarkError(
                    "short write for shared-host-weight prefault plan"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    tp2._fsync_directory(config.result_directory)
    return (
        plan_path,
        _sha256_bytes(encoded),
        cast(str, plan["plan_content_sha256"]),
        tuple(summaries),
        artifact_bytes,
    )


def _memory_cache_snapshot() -> JsonObject:
    wanted = {
        "MemFree",
        "Cached",
        "Active(file)",
        "Inactive(file)",
    }
    values: JsonObject = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise Glm52Tp2BenchmarkError(
            "cannot record the host page-cache state"
        ) from error
    for line in lines:
        name, separator, remainder = line.partition(":")
        if separator and name in wanted:
            fields = remainder.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise Glm52Tp2BenchmarkError(f"unexpected /proc/meminfo row for {name}")
            values[f"{name}_kib"] = int(fields[0])
    if set(values) != {f"{name}_kib" for name in wanted}:
        raise Glm52Tp2BenchmarkError("host page-cache snapshot is incomplete")
    return values


def _execute_prefault_worker_plan(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    numa_node: int,
) -> JsonObject:
    try:
        plan_status_before = plan_path.lstat()
        plan_raw = plan_path.read_bytes()
        plan_status_after = plan_path.lstat()
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            f"cannot read shared-host-weight prefault plan: {plan_path}"
        ) from error
    if (
        plan_path.is_symlink()
        or not stat.S_ISREG(plan_status_before.st_mode)
        or plan_status_before.st_size <= 0
        or plan_status_before.st_size > _SHARED_HOST_WEIGHT_MANIFEST_MAXIMUM_BYTES
        or _stable_file_identity(plan_status_before)
        != _stable_file_identity(plan_status_after)
        or _sha256_bytes(plan_raw) != expected_plan_sha256
    ):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault worker plan identity is invalid"
        )
    try:
        raw_plan = cast(object, json.loads(plan_raw))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault worker plan is invalid JSON"
        ) from error
    if not isinstance(raw_plan, dict):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault worker plan is not an object"
        )
    plan = cast(dict[str, object], raw_plan)
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != _SHARED_HOST_WEIGHT_PREFAULT_PLAN_KIND
    ):
        raise Glm52Tp2BenchmarkError(
            "unsupported shared-host-weight prefault worker plan"
        )
    plan_content_sha256 = plan.get("plan_content_sha256")
    content_without_hash = {
        name: value for name, value in plan.items() if name != "plan_content_sha256"
    }
    if (
        not isinstance(plan_content_sha256, str)
        or _canonical_sha256(cast(JsonObject, content_without_hash))
        != plan_content_sha256
    ):
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault plan content hash is invalid"
        )
    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list):
        raise Glm52Tp2BenchmarkError("shared-host-weight prefault plan lacks node rows")
    matching_rows = [
        row
        for row in cast(list[object], raw_nodes)
        if isinstance(row, dict)
        and cast(dict[object, object], row).get("numa_node") == numa_node
    ]
    if len(matching_rows) != 1:
        raise Glm52Tp2BenchmarkError(
            f"shared-host-weight prefault plan lacks NUMA {numa_node}"
        )
    node_row = cast(dict[str, object], matching_rows[0])
    raw_files = node_row.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise Glm52Tp2BenchmarkError(
            f"shared-host-weight prefault plan has no NUMA {numa_node} files"
        )

    buffer = bytearray(_PREFAULT_READ_CHUNK_BYTES)
    buffer_view = memoryview(buffer)
    completed_bytes = 0
    completed_extents = 0
    completed_tensors = 0
    started = time.monotonic()
    for raw_file in cast(list[object], raw_files):
        if not isinstance(raw_file, dict):
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight prefault file row is invalid"
            )
        file_row = cast(dict[str, object], raw_file)
        raw_path = file_row.get("absolute_path")
        raw_identity = file_row.get("identity")
        raw_extents = file_row.get("extents")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or not isinstance(raw_identity, list)
            or len(raw_identity) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_identity
            )
            or not isinstance(raw_extents, list)
            or not raw_extents
        ):
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight prefault file contract is invalid"
            )
        expected_identity = tuple(cast(list[int], raw_identity))
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(raw_path, flags)
        except OSError as error:
            raise Glm52Tp2BenchmarkError(
                f"cannot open prefault shard: {raw_path}"
            ) from error
        file_bytes = 0
        file_extents = 0
        file_tensors = 0
        try:
            before = os.fstat(descriptor)
            if _stable_file_identity(before) != expected_identity:
                raise Glm52Tp2BenchmarkError(
                    f"prefault shard identity changed: {raw_path}"
                )
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_RANDOM"):
                os.posix_fadvise(
                    descriptor,
                    0,
                    0,
                    os.POSIX_FADV_RANDOM,
                )
            previous_end = -1
            for raw_extent in cast(list[object], raw_extents):
                if not isinstance(raw_extent, dict):
                    raise Glm52Tp2BenchmarkError(
                        f"invalid prefault extent for {raw_path}"
                    )
                extent = cast(dict[object, object], raw_extent)
                offset_bytes = extent.get("offset_bytes")
                length_bytes = extent.get("length_bytes")
                tensor_count = extent.get("tensor_count")
                if (
                    isinstance(offset_bytes, bool)
                    or not isinstance(offset_bytes, int)
                    or isinstance(length_bytes, bool)
                    or not isinstance(length_bytes, int)
                    or isinstance(tensor_count, bool)
                    or not isinstance(tensor_count, int)
                    or offset_bytes < 0
                    or length_bytes <= 0
                    or tensor_count <= 0
                    or offset_bytes < previous_end
                    or offset_bytes + length_bytes > before.st_size
                ):
                    raise Glm52Tp2BenchmarkError(
                        f"invalid prefault extent bounds for {raw_path}"
                    )
                remaining = length_bytes
                read_offset = offset_bytes
                while remaining:
                    requested = min(remaining, len(buffer))
                    observed = os.preadv(
                        descriptor,
                        [buffer_view[:requested]],
                        read_offset,
                    )
                    if observed <= 0:
                        raise Glm52Tp2BenchmarkError(
                            f"short prefault read for {raw_path}"
                        )
                    remaining -= observed
                    read_offset += observed
                previous_end = offset_bytes + length_bytes
                file_bytes += length_bytes
                file_extents += 1
                file_tensors += tensor_count
            after = os.fstat(descriptor)
            if _stable_file_identity(before) != _stable_file_identity(after):
                raise Glm52Tp2BenchmarkError(
                    f"prefault shard changed while reading: {raw_path}"
                )
        finally:
            os.close(descriptor)
        if (
            file_bytes != file_row.get("expected_bytes")
            or file_extents != file_row.get("extent_count")
            or file_tensors != file_row.get("tensor_count")
        ):
            raise Glm52Tp2BenchmarkError(
                f"prefault file totals differ from plan: {raw_path}"
            )
        completed_bytes += file_bytes
        completed_extents += file_extents
        completed_tensors += file_tensors
    elapsed_seconds = time.monotonic() - started
    if (
        completed_bytes != node_row.get("expected_bytes")
        or completed_extents != node_row.get("extent_count")
        or completed_tensors != node_row.get("tensor_count")
        or len(raw_files) != node_row.get("file_count")
    ):
        raise Glm52Tp2BenchmarkError(
            f"NUMA {numa_node} prefault totals differ from plan"
        )
    return {
        "schema_version": 1,
        "kind": _SHARED_HOST_WEIGHT_PREFAULT_RESULT_KIND,
        "status": "completed",
        "numa_node": numa_node,
        "file_count": len(raw_files),
        "extent_count": completed_extents,
        "tensor_count": completed_tensors,
        "completed_bytes": completed_bytes,
        "elapsed_seconds": elapsed_seconds,
        "affinity_cpu_count": len(os.sched_getaffinity(0)),
        "readahead_policy": "POSIX_FADV_RANDOM",
    }


def _prefault_worker_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="NUMA-bound GLM-5.2 shared-weight prefault worker"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--plan-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--numa-node",
        type=int,
        choices=NUMA_NODES,
        required=True,
    )
    try:
        parsed = parser.parse_args(arguments)
        receipt = _execute_prefault_worker_plan(
            plan_path=cast(Path, parsed.plan),
            expected_plan_sha256=cast(str, parsed.plan_sha256),
            numa_node=cast(int, parsed.numa_node),
        )
    except (Glm52Tp2BenchmarkError, OSError, ValueError) as error:
        print(f"shared-host-weight prefault worker failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def _run_prefault_command(
    command: tuple[str, ...],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )


def prefault_shared_host_weights(
    config: BenchmarkConfig,
    command_runner: PrefaultCommandRunner = _run_prefault_command,
) -> JsonObject:
    """NUMA-prefault only the AMX tensor extents used by this exact run."""

    if not config.enable_shared_host_weights:
        return {
            "enabled": False,
            "policy": "not_applicable_without_shared_host_weights",
        }
    if not _NUMACTL_EXECUTABLE.is_file():
        raise Glm52Tp2BenchmarkError(
            "shared-host-weight prefault requires /usr/bin/numactl"
        )
    files, manifest_sha256, numa_nodes = _load_shared_host_weight_prefault_files(config)
    (
        plan_path,
        plan_sha256,
        plan_content_sha256,
        node_summaries,
        artifact_bytes,
    ) = _write_prefault_plan(
        config,
        files,
        manifest_sha256=manifest_sha256,
        numa_nodes=numa_nodes,
    )
    summary_by_node = {summary.numa_node: summary for summary in node_summaries}

    cache_before = _memory_cache_snapshot()
    started = time.monotonic()
    commands = [
        (
            str(_NUMACTL_EXECUTABLE),
            f"--cpunodebind={node}",
            f"--membind={node}",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "_prefault-worker",
            "--plan",
            str(plan_path),
            "--plan-sha256",
            plan_sha256,
            "--numa-node",
            str(node),
        )
        for node in numa_nodes
    ]
    with ThreadPoolExecutor(max_workers=len(numa_nodes)) as executor:
        futures = [
            executor.submit(
                command_runner,
                command,
                config.request_timeout_seconds,
            )
            for command in commands
        ]
        results = [future.result() for future in futures]
    elapsed_seconds = time.monotonic() - started
    workers: list[JsonValue] = []
    for index, result in enumerate(results):
        node = numa_nodes[index]
        stderr = result.stderr or ""
        if result.returncode != 0:
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight prefault reader "
                f"for NUMA {node} exited {result.returncode}; stderr SHA-256 "
                f"{_sha256_bytes(stderr.encode())}"
            )
        stdout = result.stdout or ""
        if len(stdout.encode()) > 1024 * 1024:
            raise Glm52Tp2BenchmarkError(
                f"NUMA {node} prefault worker response is too large"
            )
        try:
            raw_worker = cast(object, json.loads(stdout))
        except json.JSONDecodeError as error:
            raise Glm52Tp2BenchmarkError(
                f"NUMA {node} prefault worker response is not JSON"
            ) from error
        if not isinstance(raw_worker, dict):
            raise Glm52Tp2BenchmarkError(
                f"NUMA {node} prefault worker response is not an object"
            )
        worker = cast(JsonObject, raw_worker)
        expected = summary_by_node[node]
        exact_worker_contract: Mapping[str, JsonValue] = {
            "schema_version": 1,
            "kind": _SHARED_HOST_WEIGHT_PREFAULT_RESULT_KIND,
            "status": "completed",
            "numa_node": node,
            "file_count": expected.file_count,
            "extent_count": expected.extent_count,
            "tensor_count": expected.tensor_count,
            "completed_bytes": expected.expected_bytes,
            "readahead_policy": "POSIX_FADV_RANDOM",
        }
        for name, expected_value in exact_worker_contract.items():
            if worker.get(name) != expected_value:
                raise Glm52Tp2BenchmarkError(
                    f"NUMA {node} prefault worker field {name} differs from its plan"
                )
        workers.append(
            {
                "worker_index": index,
                "numa_node": node,
                "file_count": expected.file_count,
                "extent_count": expected.extent_count,
                "tensor_count": expected.tensor_count,
                "expected_bytes": expected.expected_bytes,
                "command_sha256": _canonical_sha256(list(commands[index])),
                "returncode": result.returncode,
                "stdout_sha256": _sha256_bytes(stdout.encode()),
                "stderr_sha256": _sha256_bytes(stderr.encode()),
                "worker_receipt": worker,
            }
        )
    for file in files:
        try:
            observed_identity = _stable_file_identity(file.absolute_path.lstat())
        except OSError as error:
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight checkpoint disappeared after prefault: "
                f"{file.relative_path}"
            ) from error
        if observed_identity != file.identity:
            raise Glm52Tp2BenchmarkError(
                "shared-host-weight checkpoint changed during prefault: "
                f"{file.relative_path}"
            )
    cache_after = _memory_cache_snapshot()
    total_bytes = sum(summary.expected_bytes for summary in node_summaries)
    return {
        "enabled": True,
        "completed": True,
        "policy": "node_bound_exact_safetensors_extent_prefault",
        "first_touch_contract": "numa_local_membind_per_tensor_extent",
        "kernel_readahead_disabled": True,
        "cache_eviction_requested": False,
        "manifest": str(config.shared_host_weights_manifest),
        "manifest_sha256": manifest_sha256,
        "content_id": config.shared_host_weights_content_id,
        "numa_nodes": list(numa_nodes),
        "include_mtp_layer_78": config.enable_mtp,
        "artifact_file_count": len(files),
        "artifact_bytes": artifact_bytes,
        "selected_file_count": max(summary.file_count for summary in node_summaries),
        "skipped_bytes": artifact_bytes - total_bytes,
        "expected_bytes": total_bytes,
        "completed_bytes": total_bytes,
        "worker_count": len(numa_nodes),
        "prefault_plan": {
            "path": str(plan_path),
            "sha256": plan_sha256,
            "plan_content_sha256": plan_content_sha256,
        },
        "elapsed_seconds": elapsed_seconds,
        "effective_gib_per_second": (
            total_bytes / 1024**3 / elapsed_seconds if elapsed_seconds > 0.0 else None
        ),
        "host_page_cache_before": cache_before,
        "host_page_cache_after": cache_after,
        "workers": workers,
    }


def build_process_spec(config: BenchmarkConfig) -> Glm52Tp2ProcessSpec:
    return Glm52Tp2ProcessSpec(
        executable=config.runtime_python,
        model_path=config.model_path,
        service_endpoint=Host(ip=config.dwagon_ip, port=config.service_port),
        distributed_coordinator=Host(
            ip=config.dwagon_ip,
            port=config.distributed_port,
        ),
        static_memory_fraction=config.static_memory_fraction,
        ktransformers_weight_path=config.ktransformers_weight_path,
        context_length=config.context_length,
        maximum_total_tokens=config.maximum_total_tokens,
        maximum_running_requests=max(config.benchmark_concurrencies),
        kv_cache_dtype=config.kv_cache_dtype,
        mla_kv_b_w8_backend=config.mla_kv_b_w8_backend,
        enable_two_batch_overlap=config.enable_two_batch_overlap,
        enable_amx_fine_grained_decode=(config.enable_amx_fine_grained_decode),
        enable_stream_prefill=config.enable_stream_prefill,
        stream_prefill_token_threshold=(config.stream_prefill_token_threshold),
        stream_prefill_experts_per_chunk=(config.stream_prefill_experts_per_chunk),
        enable_mtp=config.enable_mtp,
        enable_shared_host_weights=config.enable_shared_host_weights,
        shared_host_weights_manifest=config.shared_host_weights_manifest,
        shared_host_weights_content_id=(config.shared_host_weights_content_id),
        shared_host_weights_state_directory=(
            config.shared_host_weights_state_directory
        ),
        chunked_prefill_size=config.chunked_prefill_size,
        resident_gpu_expert_budget_total=(config.resident_gpu_expert_budget_total),
        kt_gpu_experts_ratio=config.kt_gpu_experts_ratio,
        expert_placement_strategy=config.expert_placement_strategy,
        init_expert_location=config.init_expert_location,
        init_expert_location_sha256=config.init_expert_location_sha256,
        capture_representative_routing=(config.capture_representative_routing),
    )


def _lifecycle_config(config: BenchmarkConfig) -> tp2.Tp2LocalDiagnosticConfig:
    return tp2.Tp2LocalDiagnosticConfig(
        run_id=config.run_id,
        result_directory=config.result_directory,
        dwagon_runtime_python=config.runtime_python,
        dwagon_runtime_install_receipt=config.runtime_install_receipt,
        dwagon_runtime_install_receipt_sha256=(config.runtime_install_receipt_sha256),
        dwagon_model_path=config.model_path,
        local_source_directory=config.local_source_directory,
        dwagon_ip=config.dwagon_ip,
        dwagon_socket_interface=config.dwagon_socket_interface,
        distributed_port=config.distributed_port,
        service_port=config.service_port,
        static_memory_fraction=config.static_memory_fraction,
        # The reused process lifecycle does not inspect this GLM-4.7-only field.
        resident_gpu_experts=1,
        readiness_timeout_seconds=config.readiness_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        cleanup_timeout_seconds=config.cleanup_timeout_seconds,
        warmup_count=1,
        sample_count=1,
    )


def _hybrid_checkpoint_manifest_receipt(
    config: BenchmarkConfig,
    model_root: Path,
) -> JsonObject | None:
    manifest_path = model_root / HYBRID_CHECKPOINT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        root_status = model_root.lstat()
        manifest_status = manifest_path.lstat()
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint manifest identity is unavailable"
        ) from error
    if (
        model_root.is_symlink()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or root_status.st_mode & 0o222
        or manifest_status.st_mode & 0o222
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint root and manifest must be immutable regular paths"
        )
    if not 0 < manifest_status.st_size <= HYBRID_CHECKPOINT_MANIFEST_MAXIMUM_BYTES:
        raise Glm52Tp2BenchmarkError("hybrid checkpoint manifest exceeds its bound")
    try:
        raw_manifest_value = cast(
            object,
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint manifest is unreadable or invalid"
        ) from error
    if not isinstance(raw_manifest_value, dict):
        raise Glm52Tp2BenchmarkError("hybrid checkpoint manifest is not an object")
    raw_manifest = cast(dict[str, object], raw_manifest_value)
    content_id = raw_manifest.get("content_id")
    if (
        raw_manifest.get("schema_version") != 1
        or raw_manifest.get("kind") != HYBRID_CHECKPOINT_MANIFEST_KIND
        or not isinstance(content_id, str)
        or len(content_id) != 64
        or any(character not in "0123456789abcdef" for character in content_id)
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint manifest identity does not match the W8A16 contract"
        )

    quantization_value = raw_manifest.get("quantization")
    if not isinstance(quantization_value, dict):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint quantization contract is incompatible"
        )
    quantization = cast(dict[str, object], quantization_value)
    if (
        quantization.get("activation_dtype") != "BF16"
        or quantization.get("serialized_weight_dtype")
        != "INT8_biased_by_128_packed_in_INT32"
        or quantization.get("temporary_bf16_expansion_at_load") is not False
        or quantization.get("mla_kv_b_compact_to_compact_marlin_repack_at_load")
        is not True
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint quantization contract is incompatible"
        )

    expert_checkpoint_value = raw_manifest.get("expert_checkpoint")
    if not isinstance(expert_checkpoint_value, dict):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint does not bind the configured AMXINT4 experts"
        )
    expert_checkpoint = cast(dict[str, object], expert_checkpoint_value)
    if (
        expert_checkpoint.get("method") != "AMXINT4"
        or expert_checkpoint.get("weight_path") != config.ktransformers_weight_path
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint does not bind the configured AMXINT4 experts"
        )
    expert_content_id = expert_checkpoint.get("content_id")
    if (
        not isinstance(expert_content_id, str)
        or len(expert_content_id) != 64
        or any(character not in "0123456789abcdef" for character in expert_content_id)
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint expert content identity is invalid"
        )
    if (
        config.enable_shared_host_weights
        and expert_content_id != config.shared_host_weights_content_id
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid and shared-host expert content identities differ"
        )

    raw_files_value = raw_manifest.get("files")
    if not isinstance(raw_files_value, list):
        raise Glm52Tp2BenchmarkError("hybrid checkpoint file table is absent")
    raw_files = cast(list[object], raw_files_value)
    file_receipts: dict[str, tuple[str, int]] = {}
    for raw_file_value in raw_files:
        if not isinstance(raw_file_value, dict):
            raise Glm52Tp2BenchmarkError(
                "hybrid checkpoint file table contains a non-object"
            )
        raw_file = cast(dict[str, object], raw_file_value)
        relative_path = raw_file.get("path")
        sha256 = raw_file.get("sha256")
        size_bytes = raw_file.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or PurePosixPath(relative_path).is_absolute()
            or ".." in PurePosixPath(relative_path).parts
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or relative_path in file_receipts
        ):
            raise Glm52Tp2BenchmarkError(
                "hybrid checkpoint file table contains an invalid entry"
            )
        file_path = model_root / relative_path
        try:
            file_status = file_path.lstat()
        except OSError as error:
            raise Glm52Tp2BenchmarkError(
                f"hybrid checkpoint file is unavailable: {relative_path}"
            ) from error
        if (
            file_path.is_symlink()
            or not file_path.is_file()
            or file_status.st_size != size_bytes
            or file_status.st_mode & 0o222
        ):
            raise Glm52Tp2BenchmarkError(
                f"hybrid checkpoint file identity differs: {relative_path}"
            )
        file_receipts[relative_path] = (sha256, size_bytes)

    for relative_path in ("config.json", "model.safetensors.index.json"):
        expected = file_receipts.get(relative_path)
        if expected is None or _sha256_file(model_root / relative_path) != expected[0]:
            raise Glm52Tp2BenchmarkError(
                f"hybrid checkpoint {relative_path} differs from its manifest"
            )

    output_value = raw_manifest.get("output")
    if not isinstance(output_value, dict):
        raise Glm52Tp2BenchmarkError("hybrid checkpoint output summary is absent")
    output = cast(dict[str, object], output_value)
    payload_bytes = output.get("payload_bytes")
    shard_count = output.get("shard_count")
    tensor_count = output.get("tensor_count")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (payload_bytes, shard_count, tensor_count)
    ):
        raise Glm52Tp2BenchmarkError(
            "hybrid checkpoint output summary contains invalid counts"
        )
    payload_bytes = cast(int, payload_bytes)
    shard_count = cast(int, shard_count)
    tensor_count = cast(int, tensor_count)
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": 1,
        "kind": HYBRID_CHECKPOINT_MANIFEST_KIND,
        "content_id": content_id,
        "storage": "persistent_compact_w8_with_bf16_sensitive_tensors",
        "activation_dtype": "BF16",
        "kv_cache_dtype": config.kv_cache_dtype,
        "temporary_bf16_expansion_at_load": False,
        "mla_kv_b_marlin_compact_to_compact_repack": True,
        "file_count": len(file_receipts),
        "payload_bytes": payload_bytes,
        "shard_count": shard_count,
        "tensor_count": tensor_count,
        "expert_content_id": expert_content_id,
    }


def verify_runtime_and_checkpoint_contract(config: BenchmarkConfig) -> JsonObject:
    runtime = Path(config.runtime_python)
    source_directory = Path(config.local_source_directory)
    if not runtime.is_file() or not os.access(runtime, os.X_OK):
        raise Glm52Tp2BenchmarkError(
            f"runtime executable is missing or not executable: {runtime}"
        )
    if not source_directory.is_dir():
        raise Glm52Tp2BenchmarkError(
            f"runtime source directory is missing: {source_directory}"
        )
    if not config.runtime_install_receipt.is_file():
        raise Glm52Tp2BenchmarkError(
            f"runtime install receipt is missing: {config.runtime_install_receipt}"
        )
    observed_receipt_sha256 = _sha256_file(config.runtime_install_receipt)
    if observed_receipt_sha256 != config.runtime_install_receipt_sha256:
        raise Glm52Tp2BenchmarkError(
            "runtime install receipt SHA-256 differs from the configured identity"
        )

    model_root = Path(config.model_path)
    hybrid_manifest = _hybrid_checkpoint_manifest_receipt(config, model_root)
    checkpoint_receipts: list[JsonValue] = []
    for role, raw_root in (
        (
            (
                "persistent_w8a16_gpu_model"
                if hybrid_manifest is not None
                else "bf16_model"
            ),
            config.model_path,
        ),
        ("amxint4_ktransformers", config.ktransformers_weight_path),
    ):
        root = Path(raw_root)
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        if not root.is_dir() or not config_path.is_file() or not index_path.is_file():
            raise Glm52Tp2BenchmarkError(
                f"{role} checkpoint lacks config.json or model index: {root}"
            )
        if config_path.stat().st_size > 2 * 1024 * 1024:
            raise Glm52Tp2BenchmarkError(f"{role} config exceeds its size bound")
        try:
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Glm52Tp2BenchmarkError(
                f"{role} config is unreadable or invalid"
            ) from error
        if not isinstance(raw_config, dict):
            raise Glm52Tp2BenchmarkError(f"{role} config is not a JSON object")
        if (
            raw_config.get("model_type") != "glm_moe_dsa"
            or raw_config.get("num_hidden_layers") != MODEL_LAYER_COUNT
            or raw_config.get("num_nextn_predict_layers") != 1
        ):
            raise Glm52Tp2BenchmarkError(
                f"{role} config is not the expected GLM-5.2 78+1 checkpoint"
            )
        checkpoint_receipts.append(
            {
                "role": role,
                "root": str(root),
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "model_index_path": str(index_path),
                "model_index_size_bytes": index_path.stat().st_size,
                "causal_layer_count": MODEL_LAYER_COUNT,
                "nextn_predict_layer_count": 1,
            }
        )
    frequency_input: JsonObject | None = None
    if config.init_expert_location is not None:
        try:
            frequency_status = config.init_expert_location.lstat()
        except OSError as error:
            raise Glm52Tp2BenchmarkError(
                "frequency placement input is unavailable"
            ) from error
        if (
            config.init_expert_location.is_symlink()
            or not config.init_expert_location.is_file()
            or frequency_status.st_size <= 0
            or frequency_status.st_size > 512 * 1024 * 1024
        ):
            raise Glm52Tp2BenchmarkError(
                "frequency placement input must be a bounded regular non-symlink file"
            )
        observed_frequency_sha256 = _sha256_file(config.init_expert_location)
        if observed_frequency_sha256 != config.init_expert_location_sha256:
            raise Glm52Tp2BenchmarkError(
                "frequency placement input SHA-256 differs from profile"
            )
        frequency_input = {
            "path": str(config.init_expert_location),
            "sha256": observed_frequency_sha256,
            "size_bytes": frequency_status.st_size,
        }
    return {
        "runtime": {
            "executable": str(runtime),
            "install_receipt": str(config.runtime_install_receipt),
            "install_receipt_sha256": observed_receipt_sha256,
            "source_directory": str(source_directory),
        },
        "checkpoints": checkpoint_receipts,
        "compact_mla_kv_b_w8": hybrid_manifest is not None,
        "hybrid_checkpoint_manifest": hybrid_manifest,
        "frequency_placement_input": frequency_input,
        "mtp_policy": {
            "causal_layer_range": [0, MODEL_LAYER_COUNT],
            "layer_78_is_nextn_predict": True,
            "layer_78_loaded": config.enable_mtp,
            "layer_78_experts": (
                "persistent_amxint4" if config.enable_mtp else "not_loaded"
            ),
            "speculative_decoding_enabled": config.enable_mtp,
        },
        "shared_host_weights": {
            "enabled": config.enable_shared_host_weights,
            "manifest": (
                None
                if config.shared_host_weights_manifest is None
                else str(config.shared_host_weights_manifest)
            ),
            "content_id": config.shared_host_weights_content_id,
            "state_directory": (
                None
                if config.shared_host_weights_state_directory is None
                else str(config.shared_host_weights_state_directory)
            ),
        },
    }


def _run_capacity_command(
    command: tuple[str, ...],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _parse_nonnegative_integer(raw: str, description: str) -> int:
    value = int(raw.strip())
    if value < 0:
        raise ValueError(f"{description} is negative")
    return value


def _parse_optional_nonnegative_integer(
    raw: str,
    description: str,
) -> int | None:
    if raw.strip() in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    return _parse_nonnegative_integer(raw, description)


def _parse_compute_capability(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", raw.strip())
    if match is None:
        raise ValueError(f"invalid compute capability {raw!r}")
    return int(match.group(1)), int(match.group(2))


def _completed_capacity_command(
    command: tuple[str, ...],
    command_runner: CapacityCommandRunner,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = command_runner(command, _NVIDIA_SMI_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        raise Glm52Tp2BenchmarkError(
            f"capacity command could not complete: {' '.join(command)}"
        ) from error
    if completed.returncode != 0:
        raise Glm52Tp2BenchmarkError(
            f"capacity command exited {completed.returncode}: "
            f"{completed.stderr.strip()[-1000:]}"
        )
    return completed


def collect_gpu_capacity_snapshot(
    command_runner: CapacityCommandRunner = _run_capacity_command,
) -> GpuCapacitySnapshot:
    """Collect bounded GPU memory and CUDA-process evidence."""

    inventory = _completed_capacity_command(_GPU_INVENTORY_COMMAND, command_runner)
    compute_processes = _completed_capacity_command(
        _GPU_COMPUTE_PROCESS_COMMAND,
        command_runner,
    )
    try:
        inventory_rows = tuple(
            csv.reader(inventory.stdout.splitlines(), skipinitialspace=True)
        )
        process_rows = tuple(
            csv.reader(compute_processes.stdout.splitlines(), skipinitialspace=True)
        )
    except csv.Error as error:
        raise Glm52Tp2BenchmarkError("nvidia-smi returned invalid CSV") from error

    devices: list[GpuMemoryDevice] = []
    for row_number, row in enumerate(inventory_rows, start=1):
        normalized = tuple(field.strip() for field in row)
        if not normalized or all(not field for field in normalized):
            continue
        if len(normalized) != 7:
            raise Glm52Tp2BenchmarkError(
                f"GPU inventory row {row_number} has {len(normalized)} fields"
            )
        try:
            device = GpuMemoryDevice(
                uuid=normalized[0],
                index=_parse_nonnegative_integer(normalized[1], "GPU index"),
                pci_bus_id=normalized[2].lower(),
                compute_capability=normalized[3],
                total_mib=_parse_nonnegative_integer(normalized[4], "total GPU memory"),
                free_mib=_parse_nonnegative_integer(normalized[5], "free GPU memory"),
                used_mib=_parse_nonnegative_integer(normalized[6], "used GPU memory"),
            )
            _parse_compute_capability(device.compute_capability)
        except ValueError as error:
            raise Glm52Tp2BenchmarkError(
                f"GPU inventory row {row_number} is invalid: {error}"
            ) from error
        if (
            device.total_mib <= 0
            or device.free_mib > device.total_mib
            or device.used_mib > device.total_mib
        ):
            raise Glm52Tp2BenchmarkError(
                f"GPU inventory row {row_number} has inconsistent memory"
            )
        devices.append(device)

    applications: list[GpuComputeProcess] = []
    for row_number, row in enumerate(process_rows, start=1):
        normalized = tuple(field.strip() for field in row)
        if not normalized or all(not field for field in normalized):
            continue
        if len(normalized) != 3:
            raise Glm52Tp2BenchmarkError(
                f"GPU process row {row_number} has {len(normalized)} fields"
            )
        try:
            pid = _parse_nonnegative_integer(normalized[1], "CUDA process PID")
            if pid == 0:
                raise ValueError("CUDA process PID is zero")
            applications.append(
                GpuComputeProcess(
                    gpu_uuid=normalized[0],
                    pid=pid,
                    used_memory_mib=_parse_optional_nonnegative_integer(
                        normalized[2], "CUDA process memory"
                    ),
                )
            )
        except ValueError as error:
            raise Glm52Tp2BenchmarkError(
                f"GPU process row {row_number} is invalid: {error}"
            ) from error

    device_uuids = [device.uuid for device in devices]
    if len(set(device_uuids)) != len(device_uuids):
        raise Glm52Tp2BenchmarkError("GPU inventory contains duplicate UUIDs")
    return GpuCapacitySnapshot(
        observed_at_utc=_utc_now(),
        inventory_command=_GPU_INVENTORY_COMMAND,
        inventory_stdout_sha256=_sha256_bytes(inventory.stdout.encode()),
        compute_process_command=_GPU_COMPUTE_PROCESS_COMMAND,
        compute_process_stdout_sha256=_sha256_bytes(compute_processes.stdout.encode()),
        devices=tuple(sorted(devices, key=lambda device: device.index)),
        compute_processes=tuple(
            sorted(
                applications,
                key=lambda application: (application.gpu_uuid, application.pid),
            )
        ),
    )


def _devices_by_uuid(
    snapshot: GpuCapacitySnapshot,
) -> Mapping[str, GpuMemoryDevice]:
    return {device.uuid: device for device in snapshot.devices}


def validate_prelaunch_gpu_capacity(
    snapshot: GpuCapacitySnapshot,
    config: BenchmarkConfig,
) -> JsonObject:
    """Reject stale GPU owners, old PCI ordering, or inadequate launch room."""

    by_uuid = _devices_by_uuid(snapshot)
    missing = [gpu_uuid for gpu_uuid in ORDERED_GPU_UUIDS if gpu_uuid not in by_uuid]
    if missing:
        raise Glm52Tp2BenchmarkError(
            f"prelaunch GPU inventory lacks pinned UUIDs: {missing}"
        )
    selected_processes = [
        process
        for process in snapshot.compute_processes
        if process.gpu_uuid in ORDERED_GPU_UUIDS
    ]
    if selected_processes:
        owners = [(process.gpu_uuid, process.pid) for process in selected_processes]
        raise Glm52Tp2BenchmarkError(
            f"pinned GPUs already have CUDA compute owners: {owners}"
        )

    device_evidence: list[JsonValue] = []
    for rank, (gpu_uuid, expected_pci_suffix) in enumerate(
        zip(ORDERED_GPU_UUIDS, EXPECTED_GPU_PCI_SUFFIXES, strict=True)
    ):
        device = by_uuid[gpu_uuid]
        if not device.pci_bus_id.endswith(expected_pci_suffix):
            raise Glm52Tp2BenchmarkError(
                f"TP rank {rank} GPU {gpu_uuid} is at {device.pci_bus_id}, "
                f"expected PCI suffix {expected_pci_suffix}"
            )
        if config.kv_cache_dtype == "fp8_e4m3" and _parse_compute_capability(
            device.compute_capability
        ) < (8, 9):
            raise Glm52Tp2BenchmarkError(
                f"TP rank {rank} GPU {gpu_uuid} has compute capability "
                f"{device.compute_capability}; Triton fp8e4nv KV kernels require "
                "SM89 or newer"
            )
        fraction_requirement_mib = (
            math.ceil(device.total_mib * config.static_memory_fraction)
            + config.minimum_postreadiness_free_vram_mib
        )
        model_requirement_mib = (
            MODEL_HEADER_PER_TP_RANK_MIB
            + MODEL_WORKSPACE_RESERVE_MIB
            + math.ceil(
                config.resident_gpu_expert_budget_total
                * EXPERT_BYTES_PER_TP_RANK
                / 1024**2
            )
        )
        required_free_mib = max(
            config.minimum_prelaunch_free_vram_mib,
            fraction_requirement_mib,
            model_requirement_mib,
        )
        if device.free_mib < required_free_mib:
            raise Glm52Tp2BenchmarkError(
                f"TP rank {rank} GPU {gpu_uuid} has {device.free_mib} MiB free, "
                f"requires at least {required_free_mib} MiB"
            )
        device_evidence.append(
            {
                **cast(JsonObject, asdict(device)),
                "tensor_parallel_rank": rank,
                "matching_numa_node": rank,
                "required_free_mib": required_free_mib,
                "fraction_requirement_mib": fraction_requirement_mib,
                "model_plus_workspace_requirement_mib": model_requirement_mib,
                "resident_expert_requirement_mib": (
                    config.resident_gpu_expert_budget_total
                    * EXPERT_BYTES_PER_TP_RANK
                    // 1024**2
                ),
            }
        )
    return {
        "passed": True,
        "policy": "empty_compute_owner_set_and_capacity_floor",
        "ordered_gpu_uuids": list(ORDERED_GPU_UUIDS),
        "devices": device_evidence,
        "compute_processes_on_selected_gpus": [],
    }


def _read_process_identity(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessIdentity:
    process_root = proc_root / str(pid)
    try:
        stat_fields = (process_root / "stat").read_text().rsplit(")", 1)[1].split()
        environment_entries = frozenset(
            (process_root / "environ").read_bytes().split(b"\0")
        )
        return ProcessIdentity(
            parent_pid=int(stat_fields[1]),
            process_group_id=int(stat_fields[2]),
            session_id=int(stat_fields[3]),
            start_time_ticks=int(stat_fields[19]),
            environment_entries=environment_entries,
        )
    except (OSError, UnicodeError, IndexError, ValueError) as error:
        raise Glm52Tp2BenchmarkError(
            f"cannot prove CUDA process identity for PID {pid}"
        ) from error


def _has_owned_ancestor(
    parent_pid: int,
    owned_parent_pid: int,
    process_identity_reader: ProcessIdentityReader,
) -> bool:
    """Return whether a live process ancestry chain reaches the owned parent."""

    current_pid = parent_pid
    visited: set[int] = set()
    for _ in range(128):
        if current_pid == owned_parent_pid:
            return True
        if current_pid <= 1 or current_pid in visited:
            return False
        visited.add(current_pid)
        current_pid = process_identity_reader(current_pid).parent_pid
    return False


def validate_postreadiness_gpu_capacity(
    snapshot: GpuCapacitySnapshot,
    config: BenchmarkConfig,
    running: tp2.RunningParent,
    *,
    process_identity_reader: ProcessIdentityReader = _read_process_identity,
) -> JsonObject:
    """Require sufficient remaining VRAM and benchmark-owned CUDA contexts."""

    by_uuid = _devices_by_uuid(snapshot)
    owner_entry = f"EXO_BENCHMARK_OWNER_TOKEN={running.owned.owner_token}".encode()
    process_receipts: list[JsonValue] = []
    seen_pids: set[int] = set()
    for rank, gpu_uuid in enumerate(ORDERED_GPU_UUIDS):
        device = by_uuid.get(gpu_uuid)
        if device is None:
            raise Glm52Tp2BenchmarkError(
                f"post-readiness GPU inventory lacks TP rank {rank} UUID {gpu_uuid}"
            )
        if device.free_mib < config.minimum_postreadiness_free_vram_mib:
            raise Glm52Tp2BenchmarkError(
                f"TP rank {rank} GPU {gpu_uuid} retained only "
                f"{device.free_mib} MiB free"
            )
        applications = [
            process
            for process in snapshot.compute_processes
            if process.gpu_uuid == gpu_uuid
        ]
        if not applications:
            raise Glm52Tp2BenchmarkError(
                f"TP rank {rank} GPU {gpu_uuid} has no observable CUDA owner"
            )
        for application in applications:
            if application.pid in seen_pids:
                continue
            identity = process_identity_reader(application.pid)
            same_owned_namespace = (
                identity.process_group_id == running.owned.process_group_id
                and identity.session_id == running.owned.pid
            )
            owned_descendant = _has_owned_ancestor(
                identity.parent_pid,
                running.owned.pid,
                process_identity_reader,
            )
            inherited_owner_token = owner_entry in identity.environment_entries
            started_after_owned_parent = (
                identity.start_time_ticks >= running.owned.start_time_ticks
            )
            token_and_start_proof = inherited_owner_token and started_after_owned_parent
            if not (same_owned_namespace or owned_descendant or token_and_start_proof):
                raise Glm52Tp2BenchmarkError(
                    f"CUDA PID {application.pid} is not owned by this TP2 run "
                    f"(namespace={same_owned_namespace}, "
                    f"ancestor={owned_descendant}, "
                    f"owner_token={inherited_owner_token}, "
                    f"started_after_parent={started_after_owned_parent})"
                )
            seen_pids.add(application.pid)
            process_receipts.append(
                {
                    "pid": application.pid,
                    "parent_pid": identity.parent_pid,
                    "process_group_id": identity.process_group_id,
                    "session_id": identity.session_id,
                    "ownership_proof": (
                        "process_group_and_session"
                        if same_owned_namespace
                        else (
                            "ancestor_chain"
                            if owned_descendant
                            else "owner_token_and_start_time"
                        )
                    ),
                    "owner_token_sha256": _sha256_bytes(
                        running.owned.owner_token.encode()
                    ),
                }
            )
    return {
        "passed": True,
        "policy": "minimum_remaining_vram_and_owned_cuda_processes",
        "minimum_free_vram_mib": config.minimum_postreadiness_free_vram_mib,
        "devices": [
            cast(JsonObject, asdict(by_uuid[gpu_uuid]))
            for gpu_uuid in ORDERED_GPU_UUIDS
        ],
        "owned_cuda_processes": process_receipts,
    }


def validate_server_capacity(
    server_info: JsonObject,
    config: BenchmarkConfig,
    spec: Glm52Tp2ProcessSpec,
) -> JsonObject:
    """Require the live server to expose the exact TP2 and token-capacity contract."""

    expected: JsonObject = {
        "model_path": spec.model_path,
        "kt_weight_path": spec.ktransformers_weight_path,
        "tp_size": 2,
        "pp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
        "dist_init_addr": str(spec.distributed_coordinator),
        "kt_method": "AMXINT4",
        "kt_cpuinfer": 112,
        "kt_threadpool_count": 2,
        "kt_numa_nodes": [0, 1],
        "kt_num_gpu_experts": (
            0 if spec.resident_gpu_expert_budget_total == 0 else None
        ),
        "kt_gpu_experts_ratio": (
            None
            if spec.kt_gpu_experts_ratio is None
            else float(spec.kt_gpu_experts_ratio)
        ),
        "kt_max_deferred_experts_per_token": 0,
        "kt_expert_placement_strategy": spec.expert_placement_strategy,
        "init_expert_location": (
            "trivial"
            if spec.init_expert_location is None
            else str(spec.init_expert_location)
        ),
        "mem_fraction_static": spec.static_memory_fraction,
        "attention_backend": "flashinfer",
        "kv_cache_dtype": spec.kv_cache_dtype,
        "enable_two_batch_overlap": spec.enable_two_batch_overlap,
        "moe_a2a_backend": "none",
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "disable_shared_experts_fusion": True,
        "chunked_prefill_size": spec.chunked_prefill_size,
        "context_length": config.context_length,
        "max_total_tokens": config.maximum_total_tokens,
        "max_total_num_tokens": config.maximum_total_tokens,
        "max_running_requests": spec.maximum_running_requests,
        "served_model_name": glm52.SERVED_MODEL_NAME,
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "trust_remote_code": True,
    }
    if spec.enable_stream_prefill:
        expected.update(
            {
                "kt_gpu_prefill_token_threshold": (spec.stream_prefill_token_threshold),
                "kt_stream_prefill": True,
                "kt_stream_prefill_experts_per_chunk": (
                    spec.stream_prefill_experts_per_chunk
                ),
                "kt_stream_prefill_ring_slots": 2,
                "kt_stream_prefill_safety_margin_mb": 512,
            }
        )
    if spec.enable_mtp:
        expected.update(
            {
                "ep_size": 1,
                "load_format": "safetensors",
                "speculative_algorithm": "EAGLE",
                "speculative_draft_model_path": spec.model_path,
                "speculative_draft_load_format": "safetensors",
                "speculative_num_steps": 1,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 2,
                "speculative_moe_a2a_backend": "none",
            }
        )
    if spec.capture_representative_routing:
        expected.update(
            {
                "expert_distribution_recorder_mode": "stat",
                "expert_distribution_recorder_buffer_size": -1,
            }
        )
    for field_name, expected_value in expected.items():
        observed_value = server_info.get(field_name)
        if (
            type(observed_value) is not type(expected_value)
            or observed_value != expected_value
        ):
            raise Glm52Tp2BenchmarkError(
                f"server capacity/identity field {field_name} is "
                f"{observed_value!r}, expected {expected_value!r}"
            )

    raw_internal_states = server_info.get("internal_states")
    if not isinstance(raw_internal_states, list) or not raw_internal_states:
        raise Glm52Tp2BenchmarkError(
            "server_info did not expose scheduler internal states"
        )
    expected_scheduler_state = {
        "effective_max_running_requests_per_dp": spec.maximum_running_requests,
        "max_total_tokens": config.maximum_total_tokens,
        "pp_max_micro_batch_size": spec.maximum_running_requests,
    }
    for state_index, raw_state in enumerate(raw_internal_states):
        if not isinstance(raw_state, dict):
            raise Glm52Tp2BenchmarkError(
                f"server scheduler state {state_index} is not an object"
            )
        for field_name, expected_value in expected_scheduler_state.items():
            if raw_state.get(field_name) != expected_value:
                raise Glm52Tp2BenchmarkError(
                    f"server scheduler state {state_index} field {field_name} "
                    f"is {raw_state.get(field_name)!r}, expected {expected_value!r}"
                )
    return {
        "passed": True,
        "validation": "exact_tp2_identity_and_scheduler_capacity",
        "top_level": expected,
        "scheduler_state": expected_scheduler_state,
        "scheduler_state_count": len(raw_internal_states),
        "server_info_sha256": _canonical_sha256(server_info),
    }


def _prepare_benchmark_prompts(
    tokenizer: glm52.BenchmarkTokenizer,
    config: BenchmarkConfig,
) -> Mapping[int, tuple[glm52.PreparedPrompt, ...]]:
    return {
        1: (
            glm52._prepare_long_context_prompt(
                tokenizer,
                config.benchmark_input_tokens,
            ),
        ),
        2: tuple(
            glm52._prepare_long_context_prompt(
                tokenizer,
                config.benchmark_input_tokens,
                marker,
            )
            for marker in _CONCURRENCY_TWO_PROMPT_MARKERS
        ),
        3: tuple(
            glm52._prepare_long_context_prompt(
                tokenizer,
                config.benchmark_input_tokens,
                marker,
            )
            for marker in _CONCURRENCY_THREE_PROMPT_MARKERS
        ),
    }


def _run_concurrency_case(
    client: httpx.Client,
    tokenizer: glm52.BenchmarkTokenizer,
    prepared_prompts: tuple[glm52.PreparedPrompt, ...],
    output_token_count: int,
    speculative_metrics_recorder: SpeculativeMetricsRecorder | None,
) -> glm52.BenchmarkCaseObservation:
    if speculative_metrics_recorder is None:
        return glm52.run_concurrency_case(
            client,
            tokenizer,
            prepared_prompts,
            output_token_count,
        )
    return glm52.run_concurrency_case(
        client,
        tokenizer,
        prepared_prompts,
        output_token_count,
        stream_meta_info_observer=speculative_metrics_recorder.observe,
    )


def _speculative_decoding_receipt(
    enabled: bool,
    recorders: list[SpeculativeMetricsRecorder],
) -> JsonObject:
    if not enabled:
        return {}
    return {
        "speculative_decoding_metrics": {
            "schema_version": 1,
            "source": "native_generate_sse_meta_info",
            "cases": [recorder.observation().receipt() for recorder in recorders],
        }
    }


def _write_result(config: BenchmarkConfig, payload: JsonObject) -> Path:
    destination = config.result_directory / _RESULT_FILENAME
    temporary = config.result_directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    tp2._fsync_directory(config.result_directory)
    return destination


def _log_receipt(config: BenchmarkConfig) -> JsonObject | None:
    path = config.result_directory / "rank-0.log"
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size > _LOG_MAXIMUM_BYTES:
        raise Glm52Tp2BenchmarkError("TP2 parent log exceeds the size bound")
    return {
        "rank": 0,
        "path": str(path),
        "size_bytes": size,
        "sha256": _sha256_file(path),
    }


def validate_compact_mla_backend_attestation(
    log_path: Path,
    spec: Glm52Tp2ProcessSpec,
) -> JsonObject:
    try:
        status = log_path.stat()
    except OSError as error:
        raise Glm52Tp2BenchmarkError(
            "compact MLA backend attestation log is unavailable"
        ) from error
    if not log_path.is_file() or status.st_size > _LOG_MAXIMUM_BYTES:
        raise Glm52Tp2BenchmarkError(
            "compact MLA backend attestation log is invalid or oversized"
        )
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise Glm52Tp2BenchmarkError(
            "compact MLA backend attestation log is unreadable"
        ) from error

    expected_backend: Literal["marlin"] = "marlin"
    expected_module_count = MODEL_LAYER_COUNT + int(spec.enable_mtp)
    observations: dict[int, dict[str, int]] = {0: {}, 1: {}}
    for match in _COMPACT_MLA_BACKEND_MARKER.finditer(log_text):
        tensor_parallel_rank = int(match.group("rank"))
        backend = match.group("backend")
        local_heads = int(match.group("local_heads"))
        if local_heads != LOCAL_MLA_HEAD_COUNT:
            raise Glm52Tp2BenchmarkError(
                "compact MLA backend loaded an unexpected local-head count"
            )
        rank_counts = observations[tensor_parallel_rank]
        rank_counts[backend] = rank_counts.get(backend, 0) + 1

    for tensor_parallel_rank, rank_counts in observations.items():
        unexpected_backends = {
            backend: count
            for backend, count in rank_counts.items()
            if backend != expected_backend and count > 0
        }
        observed_count = rank_counts.get(expected_backend, 0)
        if unexpected_backends or observed_count != expected_module_count:
            raise Glm52Tp2BenchmarkError(
                "compact MLA backend attestation differs on TP rank "
                f"{tensor_parallel_rank}: expected {expected_module_count} "
                f"{expected_backend} modules, observed {rank_counts}"
            )

    return {
        "passed": True,
        "source": "merged_parent_log_after_readiness",
        "requested_backend": spec.mla_kv_b_w8_backend,
        "expected_runtime_backend": expected_backend,
        "expected_module_count_per_tp_rank": expected_module_count,
        "expected_local_heads_per_module": LOCAL_MLA_HEAD_COUNT,
        "observations": [
            {
                "tensor_parallel_rank": tensor_parallel_rank,
                "backend": expected_backend,
                "module_count": observations[tensor_parallel_rank][expected_backend],
                "local_heads_per_module": LOCAL_MLA_HEAD_COUNT,
            }
            for tensor_parallel_rank in range(2)
        ],
    }


def validate_concurrent_generation_admission(
    benchmark_case: glm52.BenchmarkCaseObservation,
) -> JsonObject:
    """Require every submitted request to generate before any one completes."""

    intervals = [
        {
            "request_index": request.request_index,
            "generation_started_seconds": request.observation.ttft_seconds,
            "request_completed_seconds": request.observation.end_to_end_seconds,
        }
        for request in benchmark_case.requests
    ]
    if benchmark_case.concurrency == 1:
        return {
            "passed": True,
            "policy": "single_request_no_overlap_required",
            "intervals": intervals,
        }

    latest_generation_start = max(
        request.observation.ttft_seconds for request in benchmark_case.requests
    )
    earliest_request_completion = min(
        request.observation.end_to_end_seconds for request in benchmark_case.requests
    )
    overlap_seconds = earliest_request_completion - latest_generation_start
    if overlap_seconds <= 0.0:
        raise Glm52Tp2BenchmarkError(
            f"c{benchmark_case.concurrency} requests were client-concurrent but "
            "not server-resident together: the latest first token arrived "
            f"{-overlap_seconds:.3f}s after the earliest request completed"
        )
    return {
        "passed": True,
        "policy": "all_requests_generate_before_any_request_completes",
        "overlap_seconds": overlap_seconds,
        "intervals": intervals,
    }


def run_benchmark(config: BenchmarkConfig) -> JsonObject:
    """Launch once, warm once, benchmark selected cases, and write a receipt."""

    spec = build_process_spec(config)
    lifecycle_config = _lifecycle_config(config)
    process_spec = spec.receipt()
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    owner_token = uuid.uuid4().hex
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    runtime_and_checkpoint_contract: JsonObject | None = None
    shared_host_weight_prefault: JsonObject | None = None
    prelaunch_snapshot: GpuCapacitySnapshot | None = None
    prelaunch_capacity_gate: JsonObject | None = None
    postreadiness_snapshot: GpuCapacitySnapshot | None = None
    postreadiness_capacity_gate: JsonObject | None = None
    server_capacity_gate: JsonObject | None = None
    compact_mla_backend_attestation: JsonObject | None = None
    benchmark_prompts: Mapping[int, tuple[glm52.PreparedPrompt, ...]] | None = None
    semantic_prompt: glm52.PreparedPrompt | None = None
    semantic: glm52.SemanticObservation | None = None
    benchmark_cases: list[glm52.BenchmarkCaseObservation] = []
    speculative_metrics_recorders: list[SpeculativeMetricsRecorder] = []
    concurrency_admission: list[JsonObject] = []
    routing_capture: JsonObject | None = None
    running: tp2.RunningParent | None = None
    readiness: tuple[JsonObject, ...] = ()
    readiness_ownership: JsonObject | None = None
    distributed_coordinator_ownership: JsonObject | None = None
    server_info: JsonObject | None = None
    launch_evidence: JsonObject | None = None
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []
    cleanup_complete = False
    journal_created = False
    journal_cleared = False
    journal_clear_reason: str | None = None
    partial_start: JsonObject | None = None
    partial_start_cleanup_verified: bool | None = None
    parent_launch_failure: JsonObject | None = None
    signal_state = pp2._ManagedSignalState()
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, signal_state.handle)
        try:
            runtime_and_checkpoint_contract = verify_runtime_and_checkpoint_contract(
                config
            )
            if config.enable_shared_host_weights:
                glm52._status(
                    "prefaulting the immutable shared AMXINT4 artifact across "
                    "both NUMA nodes before launch"
                )
            shared_host_weight_prefault = prefault_shared_host_weights(config)
            prelaunch_snapshot = collect_gpu_capacity_snapshot()
            prelaunch_capacity_gate = validate_prelaunch_gpu_capacity(
                prelaunch_snapshot,
                config,
            )
            signal_state.checkpoint()

            tokenizer = glm52._load_tokenizer(config.model_path)
            semantic_prompt = glm52._prepare_semantic_prompt(tokenizer)
            benchmark_prompts = _prepare_benchmark_prompts(tokenizer, config)
            signal_state.checkpoint()

            with signal_state.defer():
                running = tp2.start_local_parent(spec, lifecycle_config, owner_token)
                launch_evidence = running.launch_evidence
                journal_created = True
                tp2._write_ownership_journal(lifecycle_config, running)
            readiness = tp2.wait_for_parent_readiness(
                spec,
                running,
                config.readiness_timeout_seconds,
            )
            readiness_ownership = tp2.verify_owned_service_listener(spec, running)
            if runtime_and_checkpoint_contract["compact_mla_kv_b_w8"] is True:
                compact_mla_backend_attestation = (
                    validate_compact_mla_backend_attestation(
                        config.result_directory / "rank-0.log",
                        spec,
                    )
                )
            postreadiness_snapshot = collect_gpu_capacity_snapshot()
            postreadiness_capacity_gate = validate_postreadiness_gpu_capacity(
                postreadiness_snapshot,
                config,
                running,
            )
            signal_state.checkpoint()

            rank_zero_url = f"http://{spec.service_endpoint}"
            with httpx.Client(
                base_url=rank_zero_url,
                timeout=config.request_timeout_seconds,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                info_response = client.get("/server_info")
                info_response.raise_for_status()
                server_info = glm52._strict_response_object(
                    info_response,
                    "server_info",
                )
                server_capacity_gate = validate_server_capacity(
                    server_info,
                    config,
                    spec,
                )
                distributed_coordinator_ownership = (
                    tp2.observe_distributed_coordinator_listener_ownership(
                        spec,
                        running,
                    )
                )

                glm52._status(
                    "running semantic coherency warm-up on the resident TP2 server"
                )
                semantic = glm52.run_semantic_warmup(
                    client,
                    tokenizer,
                    semantic_prompt,
                )
                if running.process.poll() is not None:
                    raise Glm52Tp2BenchmarkError(
                        "the TP2 parent exited during semantic warm-up"
                    )
                signal_state.checkpoint()

                glm52._status(
                    "semantic warm-up passed; starting "
                    f"{config.benchmark_input_tokens}-input/"
                    f"{config.benchmark_output_tokens}-output "
                    f"{'/'.join(f'c{value}' for value in config.benchmark_concurrencies)} "
                    "cases "
                    "without a cache flush or restart"
                )
                for concurrency in config.benchmark_concurrencies:
                    glm52._status(f"starting local TP2 concurrency {concurrency} case")
                    speculative_metrics_recorder = (
                        SpeculativeMetricsRecorder(concurrency)
                        if config.enable_mtp
                        else None
                    )
                    if speculative_metrics_recorder is not None:
                        speculative_metrics_recorders.append(
                            speculative_metrics_recorder
                        )
                    benchmark_case = _run_concurrency_case(
                        client,
                        tokenizer,
                        benchmark_prompts[concurrency],
                        config.benchmark_output_tokens,
                        speculative_metrics_recorder,
                    )
                    benchmark_cases.append(benchmark_case)
                    concurrency_admission.append(
                        validate_concurrent_generation_admission(benchmark_case)
                    )
                    if running.process.poll() is not None:
                        raise Glm52Tp2BenchmarkError(
                            f"the TP2 parent exited during concurrency {concurrency}"
                        )
                    signal_state.checkpoint()
                    glm52._status(
                        f"local TP2 c{concurrency} complete: "
                        f"wall={benchmark_case.case_wall_seconds:.3f}s, "
                        "aggregate output="
                        f"{benchmark_case.aggregate_output_tokens_per_second:.3f} "
                        "tok/s"
                    )
                if config.capture_representative_routing:
                    from scripts import glm52_routing_profiles as routing

                    glm52._status(
                        "timed cases complete; capturing representative expert "
                        "routes on the same resident server"
                    )
                    routing_capture = routing.capture_routing(
                        server_url=rank_zero_url,
                        recorder_directory=_EXPERT_RECORDER_DIRECTORY,
                        output_directory=config.result_directory / "routing",
                        runtime_python=Path(config.runtime_python),
                        timeout_seconds=config.request_timeout_seconds,
                    )
                    if running.process.poll() is not None:
                        raise Glm52Tp2BenchmarkError(
                            "the TP2 parent exited during routing capture"
                        )
                    signal_state.checkpoint()
        except BaseException as error:
            if isinstance(error, tp2.Tp2ParentLaunchError):
                launch_evidence = error.launch_evidence
                journal_created = journal_created or error.journal_created
                journal_cleared = journal_cleared or error.journal_cleared
                if error.journal_cleared:
                    journal_clear_reason = "popen_failed_before_process_creation"
                parent_launch_failure = {
                    "popen_attempted": error.popen_attempted,
                    "popen_returned": False,
                    "process_created": False,
                    "journal_error": error.journal_error,
                }
            elif isinstance(error, tp2.Tp2PartialParentStartError):
                partial_start = error.process_evidence
                partial_start_cleanup_verified = error.cleanup_verified
                launch_evidence = error.launch_evidence
                journal_created = journal_created or error.journal_created
                cleanup.append(
                    {
                        "rank": 0,
                        "partial_start": True,
                        **error.cleanup_evidence,
                    }
                )
            failure = (
                Glm52Tp2BenchmarkError(str(error))
                if isinstance(
                    error,
                    (
                        pp2.Pp2LocalDiagnosticError,
                        tp2.Tp2LocalDiagnosticError,
                        glm52.Glm52Pp3BenchmarkError,
                    ),
                )
                else error
            )
        finally:
            signal_state.begin_cleanup()
            if running is not None:
                try:
                    receipt = tp2.stop_local_parent(
                        running,
                        config.cleanup_timeout_seconds,
                    )
                    cleanup.append(
                        {
                            "rank": 0,
                            **receipt.model_dump(mode="json"),
                        }
                    )
                except BaseException as error:
                    cleanup.append(
                        {
                            "rank": 0,
                            "host_name": running.owned.host_name,
                            "ownership_verified": False,
                            "terminated": False,
                            "forced": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if failure is None:
                        failure = error
            if running is not None:
                cleanup_complete = (
                    len(cleanup) == 1
                    and cast(dict[str, object], cleanup[0]).get("ownership_verified")
                    is True
                    and cast(dict[str, object], cleanup[0]).get("terminated") is True
                )
            elif partial_start is not None:
                cleanup_complete = partial_start_cleanup_verified is True
            else:
                cleanup_complete = True
            if journal_created and not journal_cleared and cleanup_complete:
                try:
                    tp2._clear_ownership_journal(lifecycle_config)
                    journal_cleared = True
                    journal_clear_reason = (
                        "verified_registered_parent_cleanup"
                        if running is not None
                        else "verified_partial_parent_cleanup"
                        if partial_start is not None
                        else "popen_failed_before_process_creation"
                    )
                except BaseException as error:
                    if failure is None:
                        failure = error
    finally:
        for managed_signal, previous_handler in previous_handlers.items():
            signal.signal(managed_signal, previous_handler)

    journal_path = tp2._ownership_journal_path(lifecycle_config)
    journal_retained = journal_path.exists()
    parent_registered = running is not None
    parent_process_created = parent_registered or partial_start is not None
    log_receipt = _log_receipt(config)
    exact_case_set_complete = (
        tuple(case.concurrency for case in benchmark_cases)
        == config.benchmark_concurrencies
    )
    compact_mla_attestation_required = (
        runtime_and_checkpoint_contract is not None
        and runtime_and_checkpoint_contract.get("compact_mla_kv_b_w8") is True
    )
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm52_bf16_amxint4_pp1_tp2_dwagon_benchmark",
        "status": (
            "passed"
            if (
                failure is None
                and parent_registered
                and cleanup_complete
                and not journal_retained
                and exact_case_set_complete
                and prelaunch_capacity_gate is not None
                and postreadiness_capacity_gate is not None
                and server_capacity_gate is not None
                and (
                    not compact_mla_attestation_required
                    or compact_mla_backend_attestation is not None
                )
            )
            else "failed"
        ),
        "run_id": config.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "single_resident_parent": True,
        "cache_flush_between_semantic_and_benchmark": False,
        "restart_between_benchmark_cases": False,
        "topology": {
            "scope": "dwagon_local",
            "host_count": 1,
            "parent_process_count": 1,
            "gpu_worker_count": 2,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 2,
            "ordered_gpu_uuids": list(ORDERED_GPU_UUIDS),
            "gpu_to_numa": [
                {
                    "tensor_parallel_rank": rank,
                    "gpu_uuid": gpu_uuid,
                    "numa_node": rank,
                    "physical_cpu_ids": list(range(rank * 56, (rank + 1) * 56)),
                }
                for rank, gpu_uuid in enumerate(ORDERED_GPU_UUIDS)
            ],
            "nvlink_p2p_allowed": True,
        },
        "configuration": {
            "benchmark_concurrencies": list(config.benchmark_concurrencies),
            "context_length": config.context_length,
            "maximum_total_tokens": config.maximum_total_tokens,
            "benchmark_input_tokens": config.benchmark_input_tokens,
            "benchmark_output_tokens": config.benchmark_output_tokens,
            "kv_cache_dtype": config.kv_cache_dtype,
            "mla_kv_b_w8_backend": config.mla_kv_b_w8_backend,
            "enable_two_batch_overlap": config.enable_two_batch_overlap,
            "enable_amx_fine_grained_decode": (config.enable_amx_fine_grained_decode),
            "enable_stream_prefill": config.enable_stream_prefill,
            "stream_prefill_token_threshold": (config.stream_prefill_token_threshold),
            "stream_prefill_experts_per_chunk": (
                config.stream_prefill_experts_per_chunk
            ),
            "enable_mtp": config.enable_mtp,
            "enable_shared_host_weights": config.enable_shared_host_weights,
            "capture_representative_routing": (config.capture_representative_routing),
            "static_memory_fraction": config.static_memory_fraction,
            "minimum_prelaunch_free_vram_mib": (config.minimum_prelaunch_free_vram_mib),
            "minimum_postreadiness_free_vram_mib": (
                config.minimum_postreadiness_free_vram_mib
            ),
        },
        "runtime_and_checkpoint_contract": runtime_and_checkpoint_contract,
        "shared_host_weight_prefault": shared_host_weight_prefault,
        "process_spec": process_spec,
        "process_spec_sha256": _canonical_sha256(process_spec),
        "launch": (
            launch_evidence
            if launch_evidence is not None
            else {
                "process_created": False,
                "evidence_origin": "no_successful_popen",
            }
        ),
        "processes": ([] if running is None else [tp2._process_receipt(running)]),
        "partial_start": partial_start,
        "parent_launch_failure": parent_launch_failure,
        "readiness": list(readiness),
        "readiness_ownership": readiness_ownership,
        "distributed_coordinator_ownership": (distributed_coordinator_ownership),
        "capacity_and_vram": {
            "fail_closed": True,
            "model_sizing": {
                "bf16_non_routed_non_mtp_header_gib_total": (MODEL_HEADER_BF16_GIB),
                "estimated_header_mib_per_tp_rank": (MODEL_HEADER_PER_TP_RANK_MIB),
                "workspace_reserve_mib_per_gpu": MODEL_WORKSPACE_RESERVE_MIB,
                "excluded_mtp_layer_78_bf16_gib": MTP_LAYER_78_BF16_GIB,
            },
            "prelaunch_snapshot": (
                None
                if prelaunch_snapshot is None
                else cast(JsonObject, asdict(prelaunch_snapshot))
            ),
            "prelaunch_gate": prelaunch_capacity_gate,
            "postreadiness_snapshot": (
                None
                if postreadiness_snapshot is None
                else cast(JsonObject, asdict(postreadiness_snapshot))
            ),
            "postreadiness_gate": postreadiness_capacity_gate,
            "server_capacity_gate": server_capacity_gate,
        },
        "server_info": server_info,
        "compact_mla_backend_attestation": compact_mla_backend_attestation,
        "semantic_warmup": (
            None if semantic is None else cast(JsonObject, asdict(semantic))
        ),
        "benchmark_prompt": (
            None
            if benchmark_prompts is None
            else {
                "kind": "deterministic_long_context_incident_timeline_qa",
                "input_tokens_per_request": config.benchmark_input_tokens,
                "input_ids_sha256_by_concurrency": {
                    str(concurrency): [
                        prompt.input_ids_sha256
                        for prompt in benchmark_prompts[concurrency]
                    ]
                    for concurrency in config.benchmark_concurrencies
                },
                "max_new_tokens": config.benchmark_output_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
                "sampling_seed": glm52.DEFAULT_SAMPLING_SEED,
            }
        ),
        "benchmark_cases": [
            cast(JsonObject, asdict(benchmark_case))
            for benchmark_case in benchmark_cases
        ],
        **_speculative_decoding_receipt(
            config.enable_mtp,
            speculative_metrics_recorders,
        ),
        "concurrency_admission": concurrency_admission,
        "routing_capture": routing_capture,
        "planned_parent_process_count": 1,
        "started_parent_process_count": int(parent_process_created),
        "registered_parent_process_count": int(parent_registered),
        "all_planned_processes_started": parent_registered,
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "managed_signal": signal_state.signal_number,
        "ownership_journal": {
            "path": str(journal_path),
            "created": journal_created,
            "cleared_before_receipt": journal_cleared,
            "clear_reason": journal_clear_reason,
            "retained": journal_retained,
        },
        "logs": [] if log_receipt is None else [log_receipt],
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    result_path = _write_result(config, payload)
    if failure is not None:
        raise Glm52Tp2BenchmarkError(
            f"local GLM-5.2 TP2 benchmark failed; evidence is in {result_path}: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    if payload["status"] != "passed":
        raise Glm52Tp2BenchmarkError(
            f"local GLM-5.2 TP2 benchmark was incomplete; see {result_path}"
        )
    return payload


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _memory_fraction(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.8 <= value <= 0.95:
        raise argparse.ArgumentTypeError(
            "value must be finite and between 0.8 and 0.95"
        )
    return value


def _sha256_argument(raw: str) -> str:
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--runtime-python", default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument(
        "--runtime-install-receipt",
        type=Path,
        default=Path(DEFAULT_RUNTIME_INSTALL_RECEIPT),
    )
    parser.add_argument(
        "--runtime-install-receipt-sha256",
        type=_sha256_argument,
        default=DEFAULT_RUNTIME_INSTALL_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--local-source-directory",
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--ktransformers-weight-path",
        default=DEFAULT_KTRANSFORMERS_WEIGHT_PATH,
    )
    parser.add_argument("--dwagon-ip", default=DEFAULT_DWAGON_IP)
    parser.add_argument(
        "--dwagon-socket-interface",
        default=DEFAULT_DWAGON_SOCKET_INTERFACE,
    )
    parser.add_argument(
        "--distributed-port",
        type=_positive_int,
        default=DEFAULT_DISTRIBUTED_PORT,
    )
    parser.add_argument(
        "--service-port",
        type=_positive_int,
        default=DEFAULT_SERVICE_PORT,
    )
    parser.add_argument(
        "--context-length",
        type=_positive_int,
        default=DEFAULT_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--benchmark-input-tokens",
        type=_positive_int,
        default=DEFAULT_BENCHMARK_INPUT_TOKENS,
    )
    parser.add_argument(
        "--benchmark-output-tokens",
        type=_positive_int,
        default=DEFAULT_BENCHMARK_OUTPUT_TOKENS,
    )
    parser.add_argument(
        "--benchmark-concurrencies",
        type=int,
        choices=ADMITTED_BENCHMARK_CONCURRENCIES,
        nargs="+",
        default=DEFAULT_BENCHMARK_CONCURRENCIES,
        help=(
            "ordered unique cases to run from 1, 2, and 3; default is 1 2. "
            "Use 1 for the VRAM-sensitive MTP and stream-prefill lane"
        ),
    )
    parser.add_argument(
        "--max-total-tokens",
        dest="maximum_total_tokens",
        type=_positive_int,
        default=None,
        help=(
            "must equal max selected concurrency times input-plus-output "
            "tokens plus 256 admission tokens; defaults to 16000 for the "
            "canonical 7744/128 c1/c2 profile"
        ),
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        choices=ADMITTED_CHUNKED_PREFILL_SIZES,
        default=2_048,
    )
    parser.add_argument(
        "--resident-gpu-expert-budget-total",
        type=_nonnegative_int,
        default=0,
        help=(
            "exact global resident budget across all 75 routed layers; this "
            "is deliberately not KT's per-layer --kt-num-gpu-experts value"
        ),
    )
    parser.add_argument(
        "--kt-gpu-experts-ratio",
        help=(
            "decimal ratio whose runtime int(ratio * 19200) must equal the "
            "global resident budget"
        ),
    )
    parser.add_argument(
        "--kt-expert-placement-strategy",
        choices=("uniform", "frequency"),
        default="uniform",
    )
    parser.add_argument("--init-expert-location", type=Path)
    parser.add_argument(
        "--init-expert-location-sha256",
        type=_sha256_argument,
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("bfloat16", "fp8_e4m3"),
        default=DEFAULT_KV_CACHE_DTYPE,
        help=(
            "KV cache storage dtype; bfloat16 is canonical on SM86 because "
            "Triton's NVIDIA E4M3 path requires newer GPU architecture"
        ),
    )
    parser.add_argument(
        "--mla-kv-b-w8-backend",
        choices=("marlin",),
        default="marlin",
        help=(
            "Direct compact kv_b_proj backend. The focused OSDI26 path is "
            "fail-closed on Marlin."
        ),
    )
    parser.add_argument(
        "--enable-two-batch-overlap",
        action="store_true",
        help=(
            "Enable the exact KT attention/CPU-MoE two-request overlap path; "
            "single-stream requests remain unsplit"
        ),
    )
    parser.add_argument(
        "--enable-amx-fine-grained-decode",
        action="store_true",
        help=(
            "Enable per-slice/per-expert AMX dependencies through the "
            "KT_AMX_FINE_GRAINED_DECODE=1 runtime contract"
        ),
    )
    parser.add_argument(
        "--enable-stream-prefill",
        action="store_true",
        help=(
            "Use the bounded two-slot BF16 expert ring for prompts at or above "
            "the configured threshold"
        ),
    )
    parser.add_argument(
        "--stream-prefill-token-threshold",
        type=_positive_int,
        default=4_096,
    )
    parser.add_argument(
        "--stream-prefill-experts-per-chunk",
        type=_positive_int,
        default=4,
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help=(
            "Enable the exact one-step GLM-5.2 NEXTN path with layer-78 "
            "routed experts kept in persistent AMXINT4"
        ),
    )
    parser.add_argument(
        "--enable-shared-host-weights",
        action="store_true",
        help=(
            "Bind AMXINT4 kernels directly to immutable safetensors mappings "
            "instead of allocating a second anonymous host copy"
        ),
    )
    parser.add_argument("--shared-host-weights-manifest", type=Path)
    parser.add_argument(
        "--shared-host-weights-content-id",
        type=_sha256_argument,
    )
    parser.add_argument("--shared-host-weights-state-directory", type=Path)
    parser.add_argument(
        "--capture-representative-routing",
        action="store_true",
        help=(
            "After timed cases, capture and materialize expert routes for the "
            "fixed coding/agent corpus without restarting the server"
        ),
    )
    parser.add_argument(
        "--static-memory-fraction",
        type=_memory_fraction,
        default=DEFAULT_STATIC_MEMORY_FRACTION,
    )
    parser.add_argument(
        "--minimum-prelaunch-free-vram-mib",
        type=_positive_int,
        default=DEFAULT_MINIMUM_PRELAUNCH_FREE_VRAM_MIB,
    )
    parser.add_argument(
        "--minimum-postreadiness-free-vram-mib",
        type=_positive_int,
        default=DEFAULT_MINIMUM_POSTREADINESS_FREE_VRAM_MIB,
    )
    parser.add_argument(
        "--readiness-timeout-seconds",
        type=_positive_float,
        default=3_600.0,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=1_800.0,
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=_positive_float,
        default=60.0,
    )
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> BenchmarkConfig:
    run_id = cast(str, arguments.run_id)
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise Glm52Tp2BenchmarkError(
            "run_id must contain only safe identifier characters"
        )
    distributed_port = cast(int, arguments.distributed_port)
    service_port = cast(int, arguments.service_port)
    if (
        distributed_port == service_port
        or distributed_port > 65_535
        or service_port > 65_535
    ):
        raise Glm52Tp2BenchmarkError(
            "distributed and service ports must be distinct valid TCP ports"
        )
    runtime_python = cast(str, arguments.runtime_python)
    local_source_directory = cast(str, arguments.local_source_directory)
    model_path = cast(str, arguments.model_path)
    ktransformers_weight_path = cast(str, arguments.ktransformers_weight_path)
    for path_name, raw_path in (
        ("runtime executable", runtime_python),
        ("runtime source directory", local_source_directory),
        ("model path", model_path),
        ("KTransformers weight path", ktransformers_weight_path),
    ):
        if not Path(raw_path).is_absolute():
            raise Glm52Tp2BenchmarkError(f"{path_name} must be absolute")

    context_length = cast(int, arguments.context_length)
    input_tokens = cast(int, arguments.benchmark_input_tokens)
    output_tokens = cast(int, arguments.benchmark_output_tokens)
    if input_tokens + output_tokens > context_length:
        raise Glm52Tp2BenchmarkError(
            "benchmark input plus output tokens exceed context length"
        )
    benchmark_concurrencies = tuple(
        cast(list[int] | tuple[int, ...], arguments.benchmark_concurrencies)
    )
    if (
        not benchmark_concurrencies
        or benchmark_concurrencies != tuple(sorted(set(benchmark_concurrencies)))
        or any(
            concurrency not in ADMITTED_BENCHMARK_CONCURRENCIES
            for concurrency in benchmark_concurrencies
        )
    ):
        raise Glm52Tp2BenchmarkError(
            "benchmark concurrencies must be a nonempty ordered unique subset of 1,2,3"
        )
    benchmark_working_set_tokens = max(benchmark_concurrencies) * (
        input_tokens + output_tokens
    )
    exact_maximum_total_tokens = (
        benchmark_working_set_tokens + DEFAULT_SCHEDULER_TOKEN_HEADROOM
    )
    requested_maximum_total_tokens = cast(
        int | None,
        arguments.maximum_total_tokens,
    )
    maximum_total_tokens = (
        exact_maximum_total_tokens
        if requested_maximum_total_tokens is None
        else requested_maximum_total_tokens
    )
    if maximum_total_tokens != exact_maximum_total_tokens:
        raise Glm52Tp2BenchmarkError(
            "max total tokens must exactly fit the selected concurrency "
            "input-plus-output budget "
            f"plus {DEFAULT_SCHEDULER_TOKEN_HEADROOM} scheduler admission "
            f"tokens ({exact_maximum_total_tokens})"
        )
    chunked_prefill_size = cast(int, arguments.chunked_prefill_size)
    resident_gpu_expert_budget_total = cast(
        int,
        arguments.resident_gpu_expert_budget_total,
    )
    kt_gpu_experts_ratio = cast(str | None, arguments.kt_gpu_experts_ratio)
    expert_placement_strategy = cast(
        ExpertPlacementStrategy,
        arguments.kt_expert_placement_strategy,
    )
    init_expert_location = cast(Path | None, arguments.init_expert_location)
    init_expert_location_sha256 = cast(
        str | None,
        arguments.init_expert_location_sha256,
    )
    if resident_gpu_expert_budget_total > TOTAL_ROUTED_EXPERT_POSITIONS:
        raise Glm52Tp2BenchmarkError(
            "resident GPU expert budget exceeds 75 * 256 routed positions"
        )
    if resident_gpu_expert_budget_total == 0:
        if (
            kt_gpu_experts_ratio is not None
            or expert_placement_strategy != "uniform"
            or init_expert_location is not None
            or init_expert_location_sha256 is not None
        ):
            raise Glm52Tp2BenchmarkError(
                "zero-resident profile must use uniform placement without "
                "ratio or frequency input"
            )
    else:
        if kt_gpu_experts_ratio is None:
            raise Glm52Tp2BenchmarkError(
                "nonzero global resident budget requires --kt-gpu-experts-ratio"
            )
        try:
            parsed_ratio = float(kt_gpu_experts_ratio)
        except ValueError as error:
            raise Glm52Tp2BenchmarkError(
                "KT GPU expert ratio is not numeric"
            ) from error
        if (
            not math.isfinite(parsed_ratio)
            or not 0.0 < parsed_ratio <= 1.0
            or int(parsed_ratio * TOTAL_ROUTED_EXPERT_POSITIONS)
            != resident_gpu_expert_budget_total
        ):
            raise Glm52Tp2BenchmarkError(
                "KT GPU expert ratio does not encode the exact global budget"
            )
        frequency_contract = (
            init_expert_location,
            init_expert_location_sha256,
        )
        if expert_placement_strategy == "frequency":
            if any(value is None for value in frequency_contract):
                raise Glm52Tp2BenchmarkError(
                    "frequency placement requires a hashed init-expert-location"
                )
            assert init_expert_location is not None
            init_expert_location = init_expert_location.resolve()
        elif any(value is not None for value in frequency_contract):
            raise Glm52Tp2BenchmarkError(
                "uniform placement cannot carry a frequency input"
            )
    minimum_prelaunch_free_vram_mib = cast(
        int,
        arguments.minimum_prelaunch_free_vram_mib,
    )
    minimum_postreadiness_free_vram_mib = cast(
        int,
        arguments.minimum_postreadiness_free_vram_mib,
    )
    if minimum_prelaunch_free_vram_mib <= minimum_postreadiness_free_vram_mib:
        raise Glm52Tp2BenchmarkError(
            "prelaunch free-VRAM floor must exceed post-readiness floor"
        )
    enable_shared_host_weights = cast(
        bool,
        arguments.enable_shared_host_weights,
    )
    shared_host_weights_manifest = cast(
        Path | None,
        arguments.shared_host_weights_manifest,
    )
    shared_host_weights_content_id = cast(
        str | None,
        arguments.shared_host_weights_content_id,
    )
    shared_host_weights_state_directory = cast(
        Path | None,
        arguments.shared_host_weights_state_directory,
    )
    shared_contract = (
        shared_host_weights_manifest,
        shared_host_weights_content_id,
        shared_host_weights_state_directory,
    )
    if enable_shared_host_weights != all(
        value is not None for value in shared_contract
    ):
        raise Glm52Tp2BenchmarkError(
            "shared host weights must be enabled with manifest, content ID, "
            "and state directory together"
        )
    if shared_host_weights_manifest is not None:
        shared_host_weights_manifest = shared_host_weights_manifest.resolve()
    if shared_host_weights_state_directory is not None:
        shared_host_weights_state_directory = (
            shared_host_weights_state_directory.resolve()
        )
    stream_prefill_experts_per_chunk = cast(
        int,
        arguments.stream_prefill_experts_per_chunk,
    )
    if stream_prefill_experts_per_chunk not in {1, 2, 4, 8, 16}:
        raise Glm52Tp2BenchmarkError(
            "stream-prefill experts per chunk must be one of 1, 2, 4, 8, 16"
        )
    return BenchmarkConfig(
        run_id=run_id,
        result_directory=cast(Path, arguments.result_directory).resolve(),
        runtime_python=runtime_python,
        runtime_install_receipt=cast(
            Path,
            arguments.runtime_install_receipt,
        ).resolve(),
        runtime_install_receipt_sha256=cast(
            str,
            arguments.runtime_install_receipt_sha256,
        ),
        local_source_directory=local_source_directory,
        model_path=model_path,
        ktransformers_weight_path=ktransformers_weight_path,
        dwagon_ip=cast(str, arguments.dwagon_ip),
        dwagon_socket_interface=cast(str, arguments.dwagon_socket_interface),
        distributed_port=distributed_port,
        service_port=service_port,
        benchmark_concurrencies=benchmark_concurrencies,
        context_length=context_length,
        maximum_total_tokens=maximum_total_tokens,
        benchmark_input_tokens=input_tokens,
        benchmark_output_tokens=output_tokens,
        chunked_prefill_size=chunked_prefill_size,
        resident_gpu_expert_budget_total=resident_gpu_expert_budget_total,
        kt_gpu_experts_ratio=kt_gpu_experts_ratio,
        expert_placement_strategy=expert_placement_strategy,
        init_expert_location=init_expert_location,
        init_expert_location_sha256=init_expert_location_sha256,
        kv_cache_dtype=cast(
            Literal["bfloat16", "fp8_e4m3"],
            arguments.kv_cache_dtype,
        ),
        mla_kv_b_w8_backend=cast(
            MlaKvBW8Backend,
            arguments.mla_kv_b_w8_backend,
        ),
        enable_two_batch_overlap=cast(
            bool,
            arguments.enable_two_batch_overlap,
        ),
        enable_amx_fine_grained_decode=cast(
            bool,
            arguments.enable_amx_fine_grained_decode,
        ),
        enable_stream_prefill=cast(bool, arguments.enable_stream_prefill),
        stream_prefill_token_threshold=cast(
            int,
            arguments.stream_prefill_token_threshold,
        ),
        stream_prefill_experts_per_chunk=(stream_prefill_experts_per_chunk),
        enable_mtp=cast(bool, arguments.enable_mtp),
        enable_shared_host_weights=enable_shared_host_weights,
        shared_host_weights_manifest=shared_host_weights_manifest,
        shared_host_weights_content_id=shared_host_weights_content_id,
        shared_host_weights_state_directory=(shared_host_weights_state_directory),
        capture_representative_routing=cast(
            bool,
            arguments.capture_representative_routing,
        ),
        static_memory_fraction=cast(float, arguments.static_memory_fraction),
        minimum_prelaunch_free_vram_mib=minimum_prelaunch_free_vram_mib,
        minimum_postreadiness_free_vram_mib=(minimum_postreadiness_free_vram_mib),
        readiness_timeout_seconds=cast(
            float,
            arguments.readiness_timeout_seconds,
        ),
        request_timeout_seconds=cast(float, arguments.request_timeout_seconds),
        cleanup_timeout_seconds=cast(float, arguments.cleanup_timeout_seconds),
    )


def main() -> int:
    if sys.argv[1:2] == ["_prefault-worker"]:
        return _prefault_worker_main(sys.argv[2:])
    try:
        config = _config_from_arguments(_parser().parse_args())
        payload = run_benchmark(config)
    except (
        Glm52Tp2BenchmarkError,
        tp2.Tp2LocalDiagnosticError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as error:
        print(f"GLM-5.2 local TP2 benchmark failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "topology": payload["topology"],
                "capacity_and_vram": payload["capacity_and_vram"],
                "benchmark_cases": payload["benchmark_cases"],
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(config.result_directory / _RESULT_FILENAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
