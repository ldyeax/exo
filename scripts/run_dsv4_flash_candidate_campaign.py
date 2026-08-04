#!/usr/bin/env python3
"""Run a resumable, fail-closed DSV4 optimization candidate campaign.

The controller launches at most one candidate per invocation.  Cheap offline
and kernel gates are represented as prerequisites in the manifest; a candidate
gets one five-phase exact/near screening receipt before a second launch is
spent on confirmation.  Baseline and confirmation launches collect three
repetitions so comparisons use prompt-phase observations rather than a single
headline rate.

No remote execution is supported.  The launcher must be a local file, the
fixed contract pins TP2/EP2/PP1 and GPUs 0,1, and every model run must prove
524K capacity, CUDA graphs, admitted OSCAR INT2 storage, CPU offload, plan
provenance, strict OpenCode/tool coherency, and payload traffic on every local
NVLink counter.  Generic FP8/INT4 and selective-BF16 cache experiments are not
representable by this campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Self, cast, final

import torch

try:
    from scripts import benchmark_dsv4_flash_hotspot as hotspot
    from scripts import validate_dsv4_flash_coherency as coherency
except ModuleNotFoundError:
    import benchmark_dsv4_flash_hotspot as hotspot
    import validate_dsv4_flash_coherency as coherency


MANIFEST_FORMAT: Final = "dsv4_candidate_campaign_v1"
RESULT_FORMAT: Final = "dsv4_candidate_campaign_result_v1"
SUMMARY_FORMAT: Final = "dsv4_candidate_campaign_summary_v1"
SAFE_ENV_KEY = re.compile(r"^(DSV4|SGLANG|KT|NCCL)_[A-Z0-9_]+$")
SAFE_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# These are the only fixed serving-contract values a candidate may replace.
# Each key is validated again below against a deliberately small value domain;
# listing a key here must never become a generic escape hatch from the fixed
# Oscar/graph/topology contract.
CANDIDATE_OVERRIDE_KEYS: Final = frozenset(
    {
        "DSV4_DSPARK_FIXED_VERIFY_LEN",
        "DSV4_CPUINFER_THREADS",
        "DSV4_STAGE_KT_AVX_TAIL_OVERLAY",
        "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY",
        "KT_MXFP4_AVX_SCALE_FOLD_MODE",
        "KT_SINGLE_NUMA_INLINE_DISPATCH",
        "KT_TASK_QUEUE_PIN_FIRST_CORE",
    }
)
ADMITTED_CPUINFER_THREAD_COUNTS: Final = frozenset({"56"})
KT_CPU_OPTIMIZED_CANDIDATE_ID: Final = "g14-oscar-int2-cpu-inline-scale-lut-n128"
KT_CPU_OPTIMIZED_CANDIDATE_PATH: Final = (
    "/var/lib/exo/experiments/dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/"
    "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
)
KT_CPU_OPTIMIZED_CACHE_ROOT: Final = (
    "/var/lib/exo/cache/dsv4-cpu-optimized-serving-overlays"
)
KT_CPU_OPTIMIZED_CANDIDATE_SHA256: Final = (
    "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043"
)
KT_CPU_OPTIMIZED_SELECTION_RECEIPT: Final = (
    "scripts/data/dsv4_flash_mxfp4_avx_scale_fold_2026-08-04.json"
)
KT_CPU_OPTIMIZED_SELECTION_RECEIPT_SHA256: Final = (
    "c0ec50a98958a767ab1353aa51a10372de21901b259805c94531c033faef1ed6"
)
KT_CPU_OPTIMIZED_ENVIRONMENT: Final = {
    "DSV4_STAGE_KT_AVX_TAIL_OVERLAY": "0",
    "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
    "DSV4_KT_CPU_OPTIMIZED_CANDIDATE": KT_CPU_OPTIMIZED_CANDIDATE_PATH,
    "DSV4_KT_CPU_OPTIMIZED_CACHE_ROOT": KT_CPU_OPTIMIZED_CACHE_ROOT,
    "KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1",
    "KT_SINGLE_NUMA_INLINE_DISPATCH": "1",
    "KT_TASK_QUEUE_PIN_FIRST_CORE": "1",
}
OSCAR_MODEL_ID: Final = "deepseek-ai/DeepSeek-V4-Flash"
REQUIRED_LOCAL_CAMPAIGN_LAUNCHER: Final = (
    "scripts/dsv4_flash_hybrid_ep2_dwagon_opencode.sh"
)
REQUIRED_LOCAL_LAUNCH_CHAIN: Final = (
    "scripts/dsv4_flash_hybrid_ep2_dwagon.sh",
    "scripts/dsv4_flash_fwuff_parity.sh",
    "scripts/dsv4_flash_0731_tp2_dwagon.sh",
)
OSCAR_ENVIRONMENT_CONTRACT: Final = {
    "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE": "1",
    "SGLANG_DSV4_INT4_KV_STORAGE": "0",
    "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE": "0",
    "SGLANG_DSV4_SM86_C128_BF16_STORAGE": "0",
}
OSCAR_STATIC_SERVER_INFO: Final = {
    "dsv4_oscar_int2_kv_storage": True,
    "dsv4_oscar_algorithm": "oscar-int2-asym-g64-v1",
    "dsv4_oscar_model_id": OSCAR_MODEL_ID,
    "dsv4_kv_storage_mode": "oscar_int2_asymmetric+protected_swa_bfloat16",
    "dsv4_swa_kv_bytes_per_token": 1_024,
    "dsv4_c4_kv_bytes_per_token": 272,
    "dsv4_c128_kv_bytes_per_token": 272,
    "dsv4_oscar_c4_scorer": True,
    "dsv4_oscar_c4_scorer_algorithm": ("oscar-int2-c4-asym-c128-fp32-adjacent4-v1"),
    "dsv4_oscar_masked_writer_execution": "device-uniform-live-mask-row-v1",
    "dsv4_oscar_c4_masked_writer_execution": "device-uniform-live-mask-row-v1",
    "dsv4_oscar_c4_query_rotation_execution": ("once-per-query-stable-workspace-v1"),
    "dsv4_c4_indexer_bytes_per_token": 40,
    "dsv4_int4_kv_storage": False,
    "dsv4_int4_c4_indexer_storage": False,
    "dsv4_sm86_c128_bf16_storage": False,
}
OSCAR_DYNAMIC_SERVER_INFO_KEYS: Final = frozenset(
    {
        "dsv4_oscar_artifact_sha256",
        "dsv4_oscar_model_config_sha256",
        "dsv4_oscar_artifact_provenance_sha256",
        "dsv4_oscar_checkpoint_sha256",
        "dsv4_oscar_checkpoint_fingerprint_sha256",
        "dsv4_oscar_admission_sha256",
        "dsv4_oscar_admission_receipt_sha256",
        "dsv4_oscar_wo_a_absorption_state",
    }
)
OSCAR_OWNED_SERVER_INFO_KEYS: Final = frozenset(OSCAR_STATIC_SERVER_INFO) | (
    OSCAR_DYNAMIC_SERVER_INFO_KEYS
)
MINIMUM_COMPLETION_LENGTH_RATIO: Final = 0.90
MAXIMUM_COMPLETION_LENGTH_RATIO: Final = 1.10
NATURAL_STOP_MAXIMUM_SPREAD_DENOMINATOR: Final = 4
ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND: Final = 80.0
ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND: Final = 90.0
SM86_SMALL_BATCH_LOG_PATTERN: Final = re.compile(
    r"\bTP(?P<tp_rank>\d+) EP(?P<ep_rank>\d+)\].*"
    r"V4 MXFP4 SM86 small-batch GEMM enabled: "
    r"block_n=(?P<block_n>\d+) split_k=(?P<split_k>\d+) "
    r"stages=(?P<num_stages>\d+)"
)
EXPECTED_SM86_SMALL_BATCH_RANKS: Final = frozenset({(0, 0), (1, 1)})
EXPECTED_SM86_SMALL_BATCH_CONFIG: Final = (128, 2, 4)
EXPECTED_SM86_SMALL_BATCH_SERVER_CONFIG: Final = {
    "block_n": 128,
    "split_k": 2,
    "num_stages": 4,
    "num_warps": 4,
}
EXPECTED_SM86_SMALL_BATCH_CALL_PARAMETERS: Final = (
    "out_dtype",
    "lhs_dtype",
    "rhs_dtype",
    "precision_config",
    "m",
    "n",
    "k",
    "routing_data",
    "can_use_persistent_tma",
    "can_use_fused_scatter",
    "enforce_bitwise_invariance",
    "epilogue_effective_itemsize",
    "constraints",
)
KT_MXFP4_AVX_SCALE_FOLD_MODE: Final = "lut-v1"
KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK: Final = 128
KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MINIMUM: Final = 118
KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MAXIMUM: Final = 126
KT_WORKER_RECORD_KEYS: Final = frozenset(
    {
        "pid",
        "gpu_id",
        "tp_rank",
        "pp_rank",
        "dp_rank",
        "moe_ep_rank",
        "moe_dp_rank",
        "telemetry",
        "validation_error",
    }
)
KT_SINGLE_NUMA_INLINE_DISPATCH_TELEMETRY_KEYS: Final = frozenset(
    {
        "environment_enabled",
        "expected_numa_id",
        "required_worker_count",
        "singleton_instance_count",
        "registered_configuration_count",
        "dispatch_count_before_startup_probe",
        "dispatch_count_after_startup_probe",
        "startup_probe_advanced_dispatch_count",
        "task_queue_affinity",
        "single_numa_inline_dispatch",
        "worker_pool_affinity",
        "startup_probe",
        "task_queue_live_cpu_affinity",
        "worker_live_cpu_affinities",
        "all_live_worker_affinities_exact",
        "all_worker_cpus_in_expected_numa",
    }
)
KT_MXFP4_AVX_SCALE_FOLD_TELEMETRY_KEYS: Final = frozenset(
    {
        "schema_version",
        "requested_mode",
        "configuration_valid",
        "architecture_supported",
        "execution_mode",
        "n_block",
        "fold_safe_minimum",
        "fold_safe_maximum",
        "lut_identity",
        "lut_hash_algorithm",
        "lut_hash",
        "lut_bytes",
        "buffers_constructed",
        "buffers_finalized",
        "buffers_admitted",
        "buffers_rejected",
        "whole_buffer_domain_finalized",
        "whole_buffer_domain_admitted",
        "scale_bytes_audited",
        "unsafe_scale_bytes",
        "nan_scale_bytes",
        "invalid_mode_requests",
        "observed_scale_minimum",
        "observed_scale_maximum",
        "decode_dispatch_count",
        "prefill_dispatch_count",
        "real_dispatch_count",
        "scale_fold_dispatch_count",
        "lut_decode_dispatch_count",
        "lut_prefill_dispatch_count",
        "exponent_decode_dispatch_count",
        "exponent_prefill_dispatch_count",
        "fallback_dispatch_count",
        "fallback_decode_dispatch_count",
        "fallback_prefill_dispatch_count",
        "zero_invalid_or_fallback_counts",
    }
)
KT_MXFP4_AVX_FIRST_DECODE_LOG_PATTERN: Final = re.compile(
    r"\[KT\]\[MXFP4\] first AVX-512 decode dispatch "
    r"\(pid=(?P<pid>\d+) m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+) "
    r"group=(?P<group>\d+) thread=(?P<thread_index>\d+)/"
    r"(?P<thread_count>\d+) scale_fold=(?P<mode>[a-z0-9-]+) "
    r"domain_finalized=(?P<domain_finalized>[01]) "
    r"n_block=(?P<n_block>\d+)\)"
)


class CampaignError(RuntimeError):
    pass


class ProcessStatus(Protocol):
    def poll(self) -> int | None: ...


class HttpStatusResponse(Protocol):
    status: int

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...


@final
@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    kind: str = "file"


@final
@dataclass(frozen=True)
class OscarContract:
    calibration_artifact: Artifact
    checkpoint_fingerprint: Artifact
    admission_receipt: Artifact
    model_id: str


@final
@dataclass(frozen=True)
class Candidate:
    identifier: str
    priority: int
    description: str
    prerequisites: tuple[str, ...]
    environment: dict[str, str]
    expected_server_info: dict[str, object]
    expected_plan: Path
    expected_plan_sha256: str | None
    plan_materialized_at_launch: bool
    artifacts: tuple[Artifact, ...]
    policy: str
    predicted: dict[str, object]


@final
@dataclass(frozen=True)
class Gates:
    model_path: Path
    agents_path: Path
    server_url: str
    chat_url: str
    input_tokens: int
    output_tokens: int
    baseline_repetitions: int
    screen_repetitions: int
    confirmation_repetitions: int
    coherency_repetitions: int
    startup_timeout_seconds: float
    request_timeout_seconds: float
    shutdown_timeout_seconds: float
    maximum_median_ttft_seconds: float
    minimum_screen_decode_ratio: float
    minimum_acceptance_ratio: float
    maximum_target_verify_ratio: float
    minimum_confirm_ci_ratio: float
    minimum_headroom_mib: int
    bootstrap_samples: int


@final
@dataclass(frozen=True)
class CampaignManifest:
    path: Path
    launcher: Path
    launcher_sha256: str
    fixed_environment: dict[str, str]
    controlled_environment_keys: frozenset[str]
    oscar_contract: OscarContract
    source_artifacts: tuple[Artifact, ...]
    gates: Gates
    baseline: Candidate
    candidates: tuple[Candidate, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sm86_small_batch_log(log_path: Path) -> dict[str, object]:
    """Prove the specialization selected once in both fixed TP2/EP2 ranks."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    rank_hits: dict[tuple[int, int], int] = {}
    for match in SM86_SMALL_BATCH_LOG_PATTERN.finditer(text):
        rank = (int(match.group("tp_rank")), int(match.group("ep_rank")))
        config = (
            int(match.group("block_n")),
            int(match.group("split_k")),
            int(match.group("num_stages")),
        )
        if config != EXPECTED_SM86_SMALL_BATCH_CONFIG:
            raise CampaignError(
                f"SM86 small-batch log reported unexpected config {config}"
            )
        rank_hits[rank] = rank_hits.get(rank, 0) + 1
    if frozenset(rank_hits) != EXPECTED_SM86_SMALL_BATCH_RANKS:
        raise CampaignError(
            "SM86 small-batch log did not prove specialization on both TP/EP ranks"
        )
    if any(count != 1 for count in rank_hits.values()):
        raise CampaignError("SM86 small-batch specialization log was not one per rank")
    return {
        "expected_tp_ep_ranks": [list(rank) for rank in sorted(rank_hits)],
        "rank_hit_counts": [
            {"tp_rank": rank[0], "ep_rank": rank[1], "count": rank_hits[rank]}
            for rank in sorted(rank_hits)
        ],
        "selected_config": {
            "block_n": EXPECTED_SM86_SMALL_BATCH_CONFIG[0],
            "split_k": EXPECTED_SM86_SMALL_BATCH_CONFIG[1],
            "num_stages": EXPECTED_SM86_SMALL_BATCH_CONFIG[2],
        },
    }


def validate_sm86_small_batch_server_telemetry(
    info: Mapping[str, object],
) -> dict[str, object]:
    """Prove both TP workers selected only the qualified V4 specialization."""

    expected_summary: dict[str, object] = {
        "dsv4_sm86_small_batch_gemm_expected_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_reporting_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_active_worker_count": 2,
        "dsv4_sm86_small_batch_gemm_all_workers_active": True,
    }
    summary_mismatches = {
        key: {"expected": expected, "observed": info.get(key)}
        for key, expected in expected_summary.items()
        if info.get(key) != expected
    }
    if summary_mismatches:
        raise CampaignError(
            f"SM86 small-batch worker summary mismatch: {summary_mismatches}"
        )

    raw_workers = info.get("dsv4_sm86_small_batch_gemm_worker_telemetry")
    if not isinstance(raw_workers, list) or len(raw_workers) != 2:
        raise CampaignError(
            "SM86 small-batch telemetry must contain exactly two TP workers"
        )

    workers: list[dict[str, object]] = []
    ranks: set[tuple[int, int]] = set()
    gpu_ids: set[int] = set()
    pids: set[int] = set()
    for raw_worker in raw_workers:
        if not isinstance(raw_worker, dict):
            raise CampaignError("SM86 small-batch worker telemetry is malformed")
        worker = cast(dict[str, object], raw_worker)
        tp_rank = worker.get("tp_rank")
        pp_rank = worker.get("pp_rank")
        dp_rank = worker.get("dp_rank")
        gpu_id = worker.get("gpu_id")
        pid = worker.get("pid")
        if (
            not isinstance(tp_rank, int)
            or isinstance(tp_rank, bool)
            or not isinstance(pp_rank, int)
            or isinstance(pp_rank, bool)
            or dp_rank not in (None, 0)
            or not isinstance(gpu_id, int)
            or isinstance(gpu_id, bool)
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
        ):
            raise CampaignError("SM86 small-batch worker identity is malformed")
        ranks.add((tp_rank, pp_rank))
        gpu_ids.add(gpu_id)
        pids.add(pid)

        worker_contract: dict[str, object] = {
            "configured": True,
            "patch_state": "installed",
            "patch_installed": True,
            "patch_error": None,
            "selected_config": EXPECTED_SM86_SMALL_BATCH_SERVER_CONFIG,
            "expected_call_parameters": list(EXPECTED_SM86_SMALL_BATCH_CALL_PARAMETERS),
        }
        worker_mismatches = {
            key: {"expected": expected, "observed": worker.get(key)}
            for key, expected in worker_contract.items()
            if worker.get(key) != expected
        }
        if worker_mismatches:
            raise CampaignError(
                f"SM86 small-batch worker contract mismatch: {worker_mismatches}"
            )

        selection_count = worker.get("selection_count")
        raw_signatures = worker.get("observed_signatures")
        if (
            not isinstance(selection_count, int)
            or isinstance(selection_count, bool)
            or selection_count <= 0
            or not isinstance(raw_signatures, list)
            or not raw_signatures
        ):
            raise CampaignError(
                "SM86 small-batch worker did not publish positive selection proof"
            )
        signature_selection_count = 0
        observed_k: set[int] = set()
        for raw_signature in raw_signatures:
            if not isinstance(raw_signature, dict):
                raise CampaignError("SM86 small-batch signature is malformed")
            signature = cast(dict[str, object], raw_signature)
            values = tuple(
                signature.get(key)
                for key in (
                    "m",
                    "logical_rows",
                    "n",
                    "k",
                    "local_experts",
                    "selection_count",
                )
            )
            if not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in values
            ):
                raise CampaignError("SM86 small-batch signature is malformed")
            m, logical_rows, n, k, local_experts, count = cast(
                tuple[int, int, int, int, int, int], values
            )
            if (
                m not in (8, 16, 24, 32, 40, 48)
                or logical_rows != m // 8
                or n != 4096
                or k not in (2048, 4096)
                or not 6 <= local_experts <= 22
                or count <= 0
            ):
                raise CampaignError(
                    "SM86 small-batch telemetry contains an unqualified signature"
                )
            observed_k.add(k)
            signature_selection_count += count
        if signature_selection_count != selection_count:
            raise CampaignError("SM86 small-batch signature counts do not reconcile")
        if observed_k != {2048, 4096}:
            raise CampaignError(
                "SM86 small-batch telemetry did not prove both W13 and W2 kernels"
            )
        workers.append(worker)

    if ranks != {(0, 0), (1, 0)} or gpu_ids != {0, 1} or len(pids) != 2:
        raise CampaignError(
            "SM86 small-batch telemetry did not prove distinct TP0/TP1 GPU workers"
        )
    return {
        **expected_summary,
        "tp_pp_ranks": [list(rank) for rank in sorted(ranks)],
        "gpu_ids": sorted(gpu_ids),
        "worker_pids": sorted(pids),
        "workers": workers,
    }


def validate_kt_single_numa_inline_dispatch_server_telemetry(
    info: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact two worker PIDs from the admitted EP2 inline path."""

    expected_summary: dict[str, object] = {
        "kt_single_numa_inline_dispatch_configured": True,
        "kt_single_numa_inline_dispatch_expected_worker_count": 2,
        "kt_single_numa_inline_dispatch_reporting_worker_count": 2,
        "kt_single_numa_inline_dispatch_active_worker_count": 2,
        "kt_single_numa_inline_dispatch_invalid_worker_count": 0,
        "kt_single_numa_inline_dispatch_duplicate_worker_count": 0,
        "kt_single_numa_inline_dispatch_rank_coverage_valid": True,
        "kt_single_numa_inline_dispatch_ep2_topology_valid": True,
        "kt_single_numa_inline_dispatch_all_workers_active": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": info.get(key)}
        for key, expected in expected_summary.items()
        if info.get(key) != expected
    }
    if mismatches:
        raise CampaignError(
            f"single-NUMA inline-dispatch summary mismatch: {mismatches}"
        )
    raw_workers = info.get("kt_single_numa_inline_dispatch_worker_telemetry")
    if not isinstance(raw_workers, list) or len(raw_workers) != 2:
        raise CampaignError(
            "single-NUMA inline-dispatch telemetry must contain exactly two workers"
        )

    workers: list[dict[str, object]] = []
    ranks: set[tuple[int, int, int]] = set()
    pids: set[int] = set()
    for raw_worker in raw_workers:
        if not isinstance(raw_worker, dict) or set(raw_worker) != KT_WORKER_RECORD_KEYS:
            raise CampaignError(
                "single-NUMA inline-dispatch worker record has the wrong schema"
            )
        worker = cast(dict[str, object], raw_worker)
        pid = worker["pid"]
        gpu_id = worker["gpu_id"]
        tp_rank = worker["tp_rank"]
        pp_rank = worker["pp_rank"]
        moe_ep_rank = worker["moe_ep_rank"]
        if any(
            type(value) is not int
            for value in (pid, gpu_id, tp_rank, pp_rank, moe_ep_rank)
        ):
            raise CampaignError(
                "single-NUMA inline-dispatch worker identity is malformed"
            )
        assert isinstance(pid, int)
        assert isinstance(gpu_id, int)
        assert isinstance(tp_rank, int)
        assert isinstance(pp_rank, int)
        assert isinstance(moe_ep_rank, int)
        if (
            pid <= 0
            or gpu_id != tp_rank
            or tp_rank not in (0, 1)
            or pp_rank != 0
            or moe_ep_rank != tp_rank
            or worker["dp_rank"] not in (None, 0)
            or worker["moe_dp_rank"] not in (None, 0)
            or worker["validation_error"] is not None
        ):
            raise CampaignError(
                "single-NUMA inline-dispatch worker identity violates EP2"
            )
        telemetry = worker["telemetry"]
        if (
            not isinstance(telemetry, dict)
            or set(telemetry) != KT_SINGLE_NUMA_INLINE_DISPATCH_TELEMETRY_KEYS
        ):
            raise CampaignError(
                "single-NUMA inline-dispatch nested telemetry has the wrong schema"
            )
        before_dispatch = telemetry["dispatch_count_before_startup_probe"]
        after_dispatch = telemetry["dispatch_count_after_startup_probe"]
        exact_nested = {
            "environment_enabled": True,
            "required_worker_count": 56,
            "singleton_instance_count": 1,
            "registered_configuration_count": 1,
            "startup_probe_advanced_dispatch_count": True,
            "all_live_worker_affinities_exact": True,
            "all_worker_cpus_in_expected_numa": True,
        }
        if (
            any(
                telemetry.get(key) != expected for key, expected in exact_nested.items()
            )
            or type(before_dispatch) is not int
            or type(after_dispatch) is not int
            or before_dispatch < 0
            or after_dispatch <= before_dispatch
        ):
            raise CampaignError(
                "single-NUMA inline-dispatch nested telemetry is not admitted"
            )
        ranks.add((tp_rank, moe_ep_rank, gpu_id))
        pids.add(pid)
        workers.append(worker)
    if ranks != {(0, 0, 0), (1, 1, 1)} or len(pids) != 2:
        raise CampaignError(
            "single-NUMA inline-dispatch did not prove two distinct EP workers"
        )
    return {
        **expected_summary,
        "worker_pids": sorted(pids),
        "tp_ep_gpu_ranks": [list(rank) for rank in sorted(ranks)],
        "workers": workers,
    }


def _validate_mxfp4_avx_scale_fold_telemetry(
    telemetry: object,
) -> dict[str, object]:
    if (
        not isinstance(telemetry, dict)
        or set(telemetry) != KT_MXFP4_AVX_SCALE_FOLD_TELEMETRY_KEYS
    ):
        raise CampaignError("MXFP4 AVX scale-fold telemetry has the wrong schema")
    typed = cast(dict[str, object], telemetry)
    boolean_keys = (
        "configuration_valid",
        "architecture_supported",
        "whole_buffer_domain_finalized",
        "whole_buffer_domain_admitted",
        "zero_invalid_or_fallback_counts",
    )
    integer_keys = (
        "schema_version",
        "n_block",
        "fold_safe_minimum",
        "fold_safe_maximum",
        "lut_bytes",
        "buffers_constructed",
        "buffers_finalized",
        "buffers_admitted",
        "buffers_rejected",
        "scale_bytes_audited",
        "unsafe_scale_bytes",
        "nan_scale_bytes",
        "invalid_mode_requests",
        "observed_scale_minimum",
        "observed_scale_maximum",
        "decode_dispatch_count",
        "prefill_dispatch_count",
        "real_dispatch_count",
        "scale_fold_dispatch_count",
        "lut_decode_dispatch_count",
        "lut_prefill_dispatch_count",
        "exponent_decode_dispatch_count",
        "exponent_prefill_dispatch_count",
        "fallback_dispatch_count",
        "fallback_decode_dispatch_count",
        "fallback_prefill_dispatch_count",
    )
    string_keys = (
        "requested_mode",
        "execution_mode",
        "lut_identity",
        "lut_hash_algorithm",
        "lut_hash",
    )
    if (
        any(type(typed[key]) is not bool for key in boolean_keys)
        or any(type(typed[key]) is not int for key in integer_keys)
        or any(type(typed[key]) is not str for key in string_keys)
    ):
        raise CampaignError("MXFP4 AVX scale-fold telemetry types are malformed")
    exact_contract: dict[str, object] = {
        "schema_version": 1,
        "requested_mode": KT_MXFP4_AVX_SCALE_FOLD_MODE,
        "configuration_valid": True,
        "architecture_supported": True,
        "n_block": KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK,
        "fold_safe_minimum": 2,
        "fold_safe_maximum": 252,
        "lut_identity": "mxfp4-e2m1-bf16-ue8m0-lut-v1",
        "lut_hash_algorithm": "fnv1a64-le",
        "lut_hash": "06d1a83dbf20f545",
        "lut_bytes": 16_384,
        "whole_buffer_domain_finalized": True,
        "whole_buffer_domain_admitted": True,
        "buffers_rejected": 0,
        "unsafe_scale_bytes": 0,
        "nan_scale_bytes": 0,
        "invalid_mode_requests": 0,
        "fallback_dispatch_count": 0,
        "fallback_decode_dispatch_count": 0,
        "fallback_prefill_dispatch_count": 0,
        "zero_invalid_or_fallback_counts": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": typed[key]}
        for key, expected in exact_contract.items()
        if typed[key] != expected
    }
    if mismatches:
        raise CampaignError(f"MXFP4 AVX scale-fold contract mismatch: {mismatches}")
    counters = [cast(int, typed[key]) for key in integer_keys]
    if min(counters) < 0:
        raise CampaignError("MXFP4 AVX scale-fold telemetry has a negative count")
    constructed = cast(int, typed["buffers_constructed"])
    finalized = cast(int, typed["buffers_finalized"])
    admitted = cast(int, typed["buffers_admitted"])
    if constructed <= 0 or (constructed, finalized, admitted) != (
        constructed,
        constructed,
        constructed,
    ):
        raise CampaignError(
            "MXFP4 AVX scale-fold did not admit every constructed buffer"
        )
    if cast(int, typed["scale_bytes_audited"]) <= 0:
        raise CampaignError("MXFP4 AVX scale-fold audited no scale bytes")
    observed_minimum = cast(int, typed["observed_scale_minimum"])
    observed_maximum = cast(int, typed["observed_scale_maximum"])
    if not (
        KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MINIMUM
        <= observed_minimum
        <= observed_maximum
        <= KT_MXFP4_AVX_SCALE_FOLD_OBSERVED_MAXIMUM
    ):
        raise CampaignError("MXFP4 AVX scale-fold observed range is outside [118,126]")
    decode_dispatches = cast(int, typed["decode_dispatch_count"])
    prefill_dispatches = cast(int, typed["prefill_dispatch_count"])
    real_dispatches = cast(int, typed["real_dispatch_count"])
    expected_execution_mode = (
        "not-executed" if real_dispatches == 0 else KT_MXFP4_AVX_SCALE_FOLD_MODE
    )
    if (
        real_dispatches != decode_dispatches + prefill_dispatches
        or typed["lut_decode_dispatch_count"] != decode_dispatches
        or typed["lut_prefill_dispatch_count"] != prefill_dispatches
        or typed["exponent_decode_dispatch_count"] != 0
        or typed["exponent_prefill_dispatch_count"] != 0
        or typed["scale_fold_dispatch_count"] != real_dispatches
        or typed["execution_mode"] != expected_execution_mode
    ):
        raise CampaignError("MXFP4 AVX scale-fold dispatch counters do not reconcile")
    return typed


def validate_mxfp4_avx_scale_fold_server_telemetry(
    info: Mapping[str, object],
    *,
    inline_worker_pids: Sequence[int],
) -> dict[str, object]:
    """Prove N128 LUT folding on the same two workers as inline dispatch."""

    expected_summary: dict[str, object] = {
        "kt_mxfp4_avx_scale_fold_configured": True,
        "kt_mxfp4_avx_scale_fold_requested_mode": (KT_MXFP4_AVX_SCALE_FOLD_MODE),
        "kt_mxfp4_avx_scale_fold_expected_n_block": (KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK),
        "kt_mxfp4_avx_scale_fold_expected_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_reporting_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_active_worker_count": 2,
        "kt_mxfp4_avx_scale_fold_invalid_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_duplicate_worker_count": 0,
        "kt_mxfp4_avx_scale_fold_rank_coverage_valid": True,
        "kt_mxfp4_avx_scale_fold_ep2_topology_valid": True,
        "kt_mxfp4_avx_scale_fold_all_workers_active": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": info.get(key)}
        for key, expected in expected_summary.items()
        if info.get(key) != expected
    }
    if mismatches:
        raise CampaignError(f"MXFP4 AVX scale-fold summary mismatch: {mismatches}")
    expected_pids = set(inline_worker_pids)
    if (
        len(inline_worker_pids) != 2
        or len(expected_pids) != 2
        or any(type(pid) is not int or pid <= 0 for pid in inline_worker_pids)
    ):
        raise CampaignError("inline worker PID proof must contain two exact PIDs")
    raw_workers = info.get("kt_mxfp4_avx_scale_fold_worker_telemetry")
    if not isinstance(raw_workers, list) or len(raw_workers) != 2:
        raise CampaignError(
            "MXFP4 AVX scale-fold telemetry must contain exactly two workers"
        )

    workers: list[dict[str, object]] = []
    ranks: set[tuple[int, int, int]] = set()
    pids: set[int] = set()
    for raw_worker in raw_workers:
        if not isinstance(raw_worker, dict) or set(raw_worker) != KT_WORKER_RECORD_KEYS:
            raise CampaignError("MXFP4 AVX scale-fold worker has the wrong schema")
        worker = cast(dict[str, object], raw_worker)
        pid = worker["pid"]
        gpu_id = worker["gpu_id"]
        tp_rank = worker["tp_rank"]
        pp_rank = worker["pp_rank"]
        moe_ep_rank = worker["moe_ep_rank"]
        if any(
            type(value) is not int
            for value in (pid, gpu_id, tp_rank, pp_rank, moe_ep_rank)
        ):
            raise CampaignError("MXFP4 AVX scale-fold worker identity is malformed")
        assert isinstance(pid, int)
        assert isinstance(gpu_id, int)
        assert isinstance(tp_rank, int)
        assert isinstance(pp_rank, int)
        assert isinstance(moe_ep_rank, int)
        if (
            pid <= 0
            or gpu_id != tp_rank
            or tp_rank not in (0, 1)
            or pp_rank != 0
            or moe_ep_rank != tp_rank
            or worker["dp_rank"] not in (None, 0)
            or worker["moe_dp_rank"] not in (None, 0)
            or worker["validation_error"] is not None
        ):
            raise CampaignError("MXFP4 AVX scale-fold worker violates EP2")
        _validate_mxfp4_avx_scale_fold_telemetry(worker["telemetry"])
        ranks.add((tp_rank, moe_ep_rank, gpu_id))
        pids.add(pid)
        workers.append(worker)
    if ranks != {(0, 0, 0), (1, 1, 1)} or pids != expected_pids:
        raise CampaignError(
            "MXFP4 AVX scale-fold workers do not match the two inline EP PIDs"
        )
    return {
        **expected_summary,
        "worker_pids": sorted(pids),
        "tp_ep_gpu_ranks": [list(rank) for rank in sorted(ranks)],
        "workers": workers,
    }


def validate_mxfp4_avx_scale_fold_dispatch_log(
    log_path: Path,
    *,
    expected_worker_pids: Sequence[int],
) -> dict[str, object]:
    """Correlate two post-warmup real decode dispatches to admitted workers."""

    expected_pids = set(expected_worker_pids)
    if (
        len(expected_worker_pids) != 2
        or len(expected_pids) != 2
        or any(type(pid) is not int or pid <= 0 for pid in expected_worker_pids)
    ):
        raise CampaignError("dispatch-log proof requires two exact worker PIDs")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(KT_MXFP4_AVX_FIRST_DECODE_LOG_PATTERN.finditer(text))
    if len(matches) != 2:
        raise CampaignError(
            "MXFP4 AVX scale-fold log must contain exactly two first decode "
            f"dispatch records; found {len(matches)}"
        )
    records: list[dict[str, object]] = []
    observed_pids: set[int] = set()
    for match in matches:
        values = {
            key: int(match.group(key))
            for key in (
                "pid",
                "m",
                "n",
                "k",
                "group",
                "thread_index",
                "thread_count",
                "domain_finalized",
                "n_block",
            )
        }
        mode = match.group("mode")
        if (
            values["pid"] <= 0
            or values["m"] <= 0
            or values["n"] <= 0
            or values["k"] <= 0
            or values["group"] != 32
            or values["k"] % values["group"] != 0
            or values["thread_count"] <= 0
            or not 0 <= values["thread_index"] < values["thread_count"]
            or mode != KT_MXFP4_AVX_SCALE_FOLD_MODE
            or values["domain_finalized"] != 1
            or values["n_block"] != KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK
        ):
            raise CampaignError(
                "MXFP4 AVX scale-fold first-dispatch record violates the exact "
                "N128 LUT contract"
            )
        observed_pids.add(values["pid"])
        records.append({**values, "scale_fold_mode": mode})
    if observed_pids != expected_pids:
        raise CampaignError(
            "MXFP4 AVX scale-fold first-dispatch PIDs do not match inline workers"
        )
    records.sort(key=lambda record: cast(int, record["pid"]))
    return {
        "expected_worker_pids": sorted(expected_pids),
        "observed_worker_pids": sorted(observed_pids),
        "record_count": len(records),
        "records": records,
    }


SOURCE_TREE_SUFFIXES: Final = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".py",
        ".pyi",
        ".rs",
        ".toml",
    }
)


def sha256_source_tree(path: Path) -> str:
    """Hash source names and bytes while excluding generated caches/builds."""
    if path.is_symlink() or not path.is_dir():
        raise CampaignError(f"source-tree artifact is not a directory: {path}")
    selected = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and item.suffix in SOURCE_TREE_SUFFIXES
        and not {".git", "__pycache__", "build", "dist"}.intersection(
            item.relative_to(path).parts
        )
    )
    if not selected:
        raise CampaignError(f"source-tree artifact contains no source files: {path}")
    digest = hashlib.sha256()
    for item in selected:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def derive_endpoint(url: str, path: str, query: str = "") -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CampaignError(f"invalid local server URL: {url!r}")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise CampaignError("candidate campaign only permits local HTTP endpoints")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))


def _required_dict(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{label} must be a non-empty string")
    return value


def _required_int(value: object, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _required_number(value: object, *, label: str, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < minimum
    ):
        raise CampaignError(f"{label} must be numeric and >= {minimum}")
    return float(value)


def _resolve_path(raw: object, *, root: Path, label: str) -> Path:
    value = Path(_required_string(raw, label=label))
    return value if value.is_absolute() else (root / value).resolve()


def _parse_environment(value: object, *, label: str) -> dict[str, str]:
    raw = _required_dict(value, label=label)
    parsed: dict[str, str] = {}
    for key, item in raw.items():
        if not SAFE_ENV_KEY.fullmatch(key):
            raise CampaignError(f"{label} contains unsafe environment key {key!r}")
        if not isinstance(item, str) or "\x00" in item:
            raise CampaignError(f"{label}.{key} must be a NUL-free string")
        parsed[key] = item
    return parsed


def _parse_artifacts(value: object, *, root: Path, label: str) -> tuple[Artifact, ...]:
    if not isinstance(value, list):
        raise CampaignError(f"{label} must be a list")
    artifacts: list[Artifact] = []
    for index, raw_item in enumerate(cast(list[object], value)):
        item = _required_dict(raw_item, label=f"{label}[{index}]")
        digest = _required_string(item.get("sha256"), label=f"{label}[{index}].sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CampaignError(f"{label}[{index}].sha256 is not lowercase SHA-256")
        kind = item.get("kind", "file")
        if kind not in {"file", "source_tree"}:
            raise CampaignError(f"{label}[{index}].kind is unsupported")
        artifacts.append(
            Artifact(
                path=_resolve_path(
                    item.get("path"), root=root, label=f"{label}[{index}].path"
                ),
                sha256=digest,
                kind=cast(str, kind),
            )
        )
    return tuple(artifacts)


def _parse_oscar_contract(
    value: object, *, root: Path, label: str = "oscar_contract"
) -> OscarContract:
    raw = _required_dict(value, label=label)
    expected_keys = {
        "calibration_artifact",
        "checkpoint_fingerprint",
        "admission_receipt",
        "model_id",
    }
    if set(raw) != expected_keys:
        raise CampaignError(f"{label} fields must be exactly {sorted(expected_keys)}")

    def parse_file(field: str) -> Artifact:
        artifacts = _parse_artifacts([raw[field]], root=root, label=f"{label}.{field}")
        artifact = artifacts[0]
        if artifact.kind != "file":
            raise CampaignError(f"{label}.{field} must be a file artifact")
        return artifact

    model_id = _required_string(raw.get("model_id"), label=f"{label}.model_id")
    if model_id != OSCAR_MODEL_ID:
        raise CampaignError(f"{label}.model_id must be {OSCAR_MODEL_ID!r}")
    return OscarContract(
        calibration_artifact=parse_file("calibration_artifact"),
        checkpoint_fingerprint=parse_file("checkpoint_fingerprint"),
        admission_receipt=parse_file("admission_receipt"),
        model_id=model_id,
    )


def _parse_candidate(
    value: object,
    *,
    root: Path,
    label: str,
) -> Candidate:
    raw = _required_dict(value, label=label)
    identifier = _required_string(raw.get("id"), label=f"{label}.id")
    if not SAFE_CANDIDATE_ID.fullmatch(identifier):
        raise CampaignError(f"{label}.id is not a safe candidate identifier")
    raw_prerequisites = raw.get("prerequisites", [])
    if not isinstance(raw_prerequisites, list) or any(
        not isinstance(item, str) or not SAFE_CANDIDATE_ID.fullmatch(item)
        for item in raw_prerequisites
    ):
        raise CampaignError(f"{label}.prerequisites must contain candidate IDs")
    expected_sha = raw.get("expected_plan_sha256")
    if expected_sha is not None and (
        not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        raise CampaignError(f"{label}.expected_plan_sha256 is invalid")
    policy = _required_string(raw.get("policy"), label=f"{label}.policy")
    if policy not in {"speed", "memory_noninferior", "baseline"}:
        raise CampaignError(f"{label}.policy is unsupported")
    return Candidate(
        identifier=identifier,
        priority=_required_int(raw.get("priority", 0), label=f"{label}.priority"),
        description=_required_string(
            raw.get("description"), label=f"{label}.description"
        ),
        prerequisites=tuple(cast(list[str], raw_prerequisites)),
        environment=_parse_environment(
            raw.get("environment", {}), label=f"{label}.environment"
        ),
        expected_server_info=_required_dict(
            raw.get("expected_server_info", {}),
            label=f"{label}.expected_server_info",
        ),
        expected_plan=_resolve_path(
            raw.get("expected_plan"), root=root, label=f"{label}.expected_plan"
        ),
        expected_plan_sha256=cast(str | None, expected_sha),
        plan_materialized_at_launch=raw.get("plan_materialized_at_launch") is True,
        artifacts=_parse_artifacts(
            raw.get("artifacts", []), root=root, label=f"{label}.artifacts"
        ),
        policy=policy,
        predicted=_required_dict(raw.get("predicted", {}), label=f"{label}.predicted"),
    )


def load_manifest(path: Path) -> CampaignManifest:
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    raw = _required_dict(loaded, label="manifest")
    if raw.get("format") != MANIFEST_FORMAT:
        raise CampaignError(f"manifest format must be {MANIFEST_FORMAT}")
    root = path.resolve().parents[2] if path.parent.name == "data" else path.parent
    fixed_environment = _parse_environment(
        raw.get("fixed_environment"), label="fixed_environment"
    )
    oscar_contract = _parse_oscar_contract(raw.get("oscar_contract"), root=root)
    mandatory_fixed = {
        "DSV4_CONTEXT_LENGTH": "524288",
        "DSV4_MAX_TOTAL_TOKENS": "524288",
        "DSV4_TENSOR_PARALLEL_SIZE": "2",
        "DSV4_EXPERT_PARALLEL_SIZE": "2",
        "DSV4_PIPELINE_PARALLEL_SIZE": "1",
        "DSV4_CUDA_VISIBLE_DEVICES": "0,1",
        "DSV4_KV_CACHE_DTYPE": "fp8_e4m3",
        "DSV4_PREFILL_GRAPH_BACKEND": "breakable",
        "DSV4_DECODE_GRAPH_BACKEND": "full",
        "DSV4_DISABLE_SPECULATIVE": "0",
        "DSV4_TARGET_VERIFY_EAGER": "0",
        "DSV4_DSPARK_FIXED_VERIFY_LEN": "4",
        "DSV4_CPUINFER_THREADS": "56",
        "DSV4_KT_THREADPOOL_COUNT": "1",
        "DSV4_KT_NUMA_NODES": "0 1",
        # Keep target geometry rooted at g14 so the separately materialized
        # three-layer draft plan never silently expands with a target ceiling.
        "DSV4_GPU_EXPERTS_PER_LAYER": "14",
        **OSCAR_ENVIRONMENT_CONTRACT,
        "DSV4_OSCAR_CALIBRATION_PATH": str(oscar_contract.calibration_artifact.path),
        "DSV4_OSCAR_CHECKPOINT_FINGERPRINT_PATH": str(
            oscar_contract.checkpoint_fingerprint.path
        ),
        "DSV4_OSCAR_ADMISSION_RECEIPT_PATH": str(oscar_contract.admission_receipt.path),
        "DSV4_OSCAR_MODEL_ID": oscar_contract.model_id,
    }
    mismatched = {
        key: (fixed_environment.get(key), expected)
        for key, expected in mandatory_fixed.items()
        if fixed_environment.get(key) != expected
    }
    if mismatched:
        raise CampaignError(
            f"fixed environment violates serving contract: {mismatched}"
        )
    raw_controlled = raw.get("controlled_environment_keys")
    if not isinstance(raw_controlled, list) or any(
        not isinstance(key, str) or not SAFE_ENV_KEY.fullmatch(key)
        for key in raw_controlled
    ):
        raise CampaignError("controlled_environment_keys must contain safe keys")
    controlled = frozenset(cast(list[str], raw_controlled))
    if not set(fixed_environment).issubset(controlled):
        raise CampaignError("every fixed environment key must be controlled")
    if "KT_MXFP4_N_BLOCK" in controlled or "KT_MXFP4_N_BLOCK" in fixed_environment:
        raise CampaignError(
            "KT_MXFP4_N_BLOCK is compile-time provenance and must not be a "
            "campaign environment override"
        )

    raw_gates = _required_dict(raw.get("gates"), label="gates")
    gates = Gates(
        model_path=_resolve_path(
            raw_gates.get("model_path"), root=root, label="gates.model_path"
        ),
        agents_path=_resolve_path(
            raw_gates.get("agents_path"), root=root, label="gates.agents_path"
        ),
        server_url=_required_string(
            raw_gates.get("server_url"), label="gates.server_url"
        ),
        chat_url=_required_string(raw_gates.get("chat_url"), label="gates.chat_url"),
        input_tokens=_required_int(
            raw_gates.get("input_tokens"), label="gates.input_tokens", minimum=1
        ),
        output_tokens=_required_int(
            raw_gates.get("output_tokens"), label="gates.output_tokens", minimum=1
        ),
        baseline_repetitions=_required_int(
            raw_gates.get("baseline_repetitions"),
            label="gates.baseline_repetitions",
            minimum=2,
        ),
        screen_repetitions=_required_int(
            raw_gates.get("screen_repetitions"),
            label="gates.screen_repetitions",
            minimum=1,
        ),
        confirmation_repetitions=_required_int(
            raw_gates.get("confirmation_repetitions"),
            label="gates.confirmation_repetitions",
            minimum=1,
        ),
        coherency_repetitions=_required_int(
            raw_gates.get("coherency_repetitions"),
            label="gates.coherency_repetitions",
            minimum=2,
        ),
        startup_timeout_seconds=_required_number(
            raw_gates.get("startup_timeout_seconds"),
            label="gates.startup_timeout_seconds",
            minimum=1,
        ),
        request_timeout_seconds=_required_number(
            raw_gates.get("request_timeout_seconds"),
            label="gates.request_timeout_seconds",
            minimum=1,
        ),
        shutdown_timeout_seconds=_required_number(
            raw_gates.get("shutdown_timeout_seconds"),
            label="gates.shutdown_timeout_seconds",
            minimum=1,
        ),
        maximum_median_ttft_seconds=_required_number(
            raw_gates.get("maximum_median_ttft_seconds"),
            label="gates.maximum_median_ttft_seconds",
            minimum=0,
        ),
        minimum_screen_decode_ratio=_required_number(
            raw_gates.get("minimum_screen_decode_ratio"),
            label="gates.minimum_screen_decode_ratio",
            minimum=0,
        ),
        minimum_acceptance_ratio=_required_number(
            raw_gates.get("minimum_acceptance_ratio"),
            label="gates.minimum_acceptance_ratio",
            minimum=0,
        ),
        maximum_target_verify_ratio=_required_number(
            raw_gates.get("maximum_target_verify_ratio"),
            label="gates.maximum_target_verify_ratio",
            minimum=0,
        ),
        minimum_confirm_ci_ratio=_required_number(
            raw_gates.get("minimum_confirm_ci_ratio"),
            label="gates.minimum_confirm_ci_ratio",
            minimum=0,
        ),
        minimum_headroom_mib=_required_int(
            raw_gates.get("minimum_headroom_mib"),
            label="gates.minimum_headroom_mib",
            minimum=1,
        ),
        bootstrap_samples=_required_int(
            raw_gates.get("bootstrap_samples"),
            label="gates.bootstrap_samples",
            minimum=100,
        ),
    )
    server_endpoint = urllib.parse.urlsplit(gates.server_url)
    chat_endpoint = urllib.parse.urlsplit(gates.chat_url)
    derive_endpoint(gates.server_url, "/server_info")
    derive_endpoint(gates.chat_url, "/v1/chat/completions")
    if (server_endpoint.scheme, server_endpoint.netloc) != (
        chat_endpoint.scheme,
        chat_endpoint.netloc,
    ):
        raise CampaignError(
            "server-info and chat endpoints must share one local origin"
        )
    baseline = _parse_candidate(raw.get("baseline"), root=root, label="baseline")
    if baseline.policy != "baseline":
        raise CampaignError("baseline candidate must use baseline policy")
    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise CampaignError("candidates must be a non-empty list")
    candidates = tuple(
        _parse_candidate(item, root=root, label=f"candidates[{index}]")
        for index, item in enumerate(cast(list[object], raw_candidates))
    )
    identifiers = [baseline.identifier, *(item.identifier for item in candidates)]
    if len(identifiers) != len(set(identifiers)):
        raise CampaignError("candidate IDs must be unique")
    known = set(identifiers)
    priority_by_id = {item.identifier: item.priority for item in candidates}
    for candidate in (baseline, *candidates):
        if candidate.identifier in candidate.prerequisites or not set(
            candidate.prerequisites
        ).issubset(known):
            raise CampaignError(
                f"candidate {candidate.identifier} has invalid prerequisites"
            )
        overlap = (
            set(candidate.environment) & set(fixed_environment)
        ) - CANDIDATE_OVERRIDE_KEYS
        if overlap:
            raise CampaignError(
                f"candidate {candidate.identifier} overrides fixed keys: {sorted(overlap)}"
            )
        if not set(candidate.environment).issubset(controlled):
            raise CampaignError(
                f"candidate {candidate.identifier} has uncontrolled environment keys"
            )
        oscar_telemetry_overrides = (
            set(candidate.expected_server_info) & OSCAR_OWNED_SERVER_INFO_KEYS
        )
        if oscar_telemetry_overrides:
            raise CampaignError(
                f"candidate {candidate.identifier} overrides OSCAR-owned server "
                f"telemetry: {sorted(oscar_telemetry_overrides)}"
            )
        verify_policy = candidate.environment.get(
            "DSV4_DSPARK_FIXED_VERIFY_LEN",
            fixed_environment["DSV4_DSPARK_FIXED_VERIFY_LEN"],
        )
        if verify_policy not in {"2", "3", "4", "5", "6"}:
            raise CampaignError(
                f"candidate {candidate.identifier} has an invalid fixed verify tier"
            )
        cpuinfer_threads = candidate.environment.get(
            "DSV4_CPUINFER_THREADS",
            fixed_environment["DSV4_CPUINFER_THREADS"],
        )
        if cpuinfer_threads not in ADMITTED_CPUINFER_THREAD_COUNTS:
            raise CampaignError(
                f"candidate {candidate.identifier} has an invalid CPUInfer thread count"
            )
        if candidate.policy == "baseline" and cpuinfer_threads != "56":
            raise CampaignError("baseline candidate must use 56 CPUInfer threads")
        optimized_override_keys = set(KT_CPU_OPTIMIZED_ENVIRONMENT) & set(
            fixed_environment
        )
        if (
            set(candidate.environment) & optimized_override_keys
            and candidate.identifier != KT_CPU_OPTIMIZED_CANDIDATE_ID
        ):
            raise CampaignError(
                "only the exact g14 CPU-optimized candidate may override the "
                "coupled native serving tuple"
            )
        if "KT_MXFP4_N_BLOCK" in candidate.environment:
            raise CampaignError(
                "KT_MXFP4_N_BLOCK must be proved by the native binary, not an "
                "environment override"
            )
        scale_fold_mode = candidate.environment.get(
            "KT_MXFP4_AVX_SCALE_FOLD_MODE",
            fixed_environment.get("KT_MXFP4_AVX_SCALE_FOLD_MODE", "off"),
        )
        if scale_fold_mode not in {"off", KT_MXFP4_AVX_SCALE_FOLD_MODE}:
            raise CampaignError(
                f"candidate {candidate.identifier} has an unqualified MXFP4 "
                "AVX scale-fold mode"
            )
        if candidate.policy == "baseline" and scale_fold_mode != "off":
            raise CampaignError("baseline candidate must keep AVX scale folding off")
        if (
            candidate.identifier == KT_CPU_OPTIMIZED_CANDIDATE_ID
            and scale_fold_mode != KT_MXFP4_AVX_SCALE_FOLD_MODE
        ):
            raise CampaignError(
                "the g14 CPU-optimized candidate must select exact lut-v1 folding"
            )
        if scale_fold_mode == KT_MXFP4_AVX_SCALE_FOLD_MODE:
            scale_environment_mismatches = {
                key: {
                    "expected": expected,
                    "observed": candidate.environment.get(
                        key, fixed_environment.get(key)
                    ),
                }
                for key, expected in KT_CPU_OPTIMIZED_ENVIRONMENT.items()
                if candidate.environment.get(key, fixed_environment.get(key))
                != expected
            }
            if scale_environment_mismatches:
                raise CampaignError(
                    f"candidate {candidate.identifier} scale folding violates "
                    "the exact CPU-optimized tuple: "
                    f"{scale_environment_mismatches}"
                )
            optimized_identity_mismatches: dict[str, object] = {}
            if candidate.identifier != KT_CPU_OPTIMIZED_CANDIDATE_ID:
                optimized_identity_mismatches["id"] = candidate.identifier
            if candidate.priority != 10:
                optimized_identity_mismatches["priority"] = candidate.priority
            if candidate.prerequisites != (baseline.identifier,):
                optimized_identity_mismatches["prerequisites"] = list(
                    candidate.prerequisites
                )
            if candidate.policy != "speed":
                optimized_identity_mismatches["policy"] = candidate.policy
            if candidate.expected_plan != baseline.expected_plan:
                optimized_identity_mismatches["expected_plan"] = str(
                    candidate.expected_plan
                )
            if candidate.expected_plan_sha256 != baseline.expected_plan_sha256:
                optimized_identity_mismatches["expected_plan_sha256"] = (
                    candidate.expected_plan_sha256
                )
            if candidate.plan_materialized_at_launch:
                optimized_identity_mismatches["plan_materialized_at_launch"] = True
            if candidate.environment.get("DSV4_GPU_EXPERTS_MAX_PER_LAYER") != "14":
                optimized_identity_mismatches["DSV4_GPU_EXPERTS_MAX_PER_LAYER"] = (
                    candidate.environment.get("DSV4_GPU_EXPERTS_MAX_PER_LAYER")
                )
            optimized_server_contract = {
                "kt_task_queue_affinity_configured": True,
                "kt_single_numa_inline_dispatch_configured": True,
                "kt_mxfp4_avx_scale_fold_configured": True,
                "kt_mxfp4_avx_scale_fold_requested_mode": (
                    KT_MXFP4_AVX_SCALE_FOLD_MODE
                ),
                "kt_mxfp4_avx_scale_fold_expected_n_block": (
                    KT_MXFP4_AVX_SCALE_FOLD_N_BLOCK
                ),
            }
            for key, expected in optimized_server_contract.items():
                if candidate.expected_server_info.get(key) != expected:
                    optimized_identity_mismatches[key] = (
                        candidate.expected_server_info.get(key)
                    )
            artifacts_by_path = {
                artifact.path.resolve(): artifact.sha256
                for artifact in candidate.artifacts
            }
            required_artifacts = {
                Path(KT_CPU_OPTIMIZED_CANDIDATE_PATH).resolve(): (
                    KT_CPU_OPTIMIZED_CANDIDATE_SHA256
                ),
                (root / KT_CPU_OPTIMIZED_SELECTION_RECEIPT).resolve(): (
                    KT_CPU_OPTIMIZED_SELECTION_RECEIPT_SHA256
                ),
            }
            if any(
                artifacts_by_path.get(path) != digest
                for path, digest in required_artifacts.items()
            ):
                optimized_identity_mismatches["artifacts"] = artifacts_by_path
            if optimized_identity_mismatches:
                raise CampaignError(
                    "the g14 CPU-optimized candidate identity is not exact: "
                    f"{optimized_identity_mismatches}"
                )
        if (
            not candidate.plan_materialized_at_launch
            and candidate.expected_plan_sha256 is None
        ):
            raise CampaignError(
                f"candidate {candidate.identifier} must hash-bind its prebuilt plan"
            )
        if any(
            prerequisite != baseline.identifier
            and priority_by_id.get(prerequisite, candidate.priority)
            >= candidate.priority
            for prerequisite in candidate.prerequisites
        ):
            raise CampaignError(
                f"candidate {candidate.identifier} prerequisites must have lower priority"
            )
    if KT_CPU_OPTIMIZED_CANDIDATE_ID not in identifiers:
        raise CampaignError("manifest is missing the exact g14 CPU-optimized candidate")
    launcher = _resolve_path(raw.get("launcher"), root=root, label="launcher")
    expected_launcher = (root / REQUIRED_LOCAL_CAMPAIGN_LAUNCHER).resolve()
    if launcher != expected_launcher:
        raise CampaignError(
            "launcher must be the OSCAR-only local OpenCode wrapper: "
            f"{REQUIRED_LOCAL_CAMPAIGN_LAUNCHER}"
        )
    launcher_sha256 = _required_string(
        raw.get("launcher_sha256"), label="launcher_sha256"
    )
    if re.fullmatch(r"[0-9a-f]{64}", launcher_sha256) is None:
        raise CampaignError("launcher_sha256 is not lowercase SHA-256")
    source_artifacts = _parse_artifacts(
        raw.get("source_artifacts", []), root=root, label="source_artifacts"
    )
    source_artifact_kinds = {
        artifact.path.resolve(): artifact.kind for artifact in source_artifacts
    }
    invalid_launch_chain = [
        relative_path
        for relative_path in REQUIRED_LOCAL_LAUNCH_CHAIN
        if source_artifact_kinds.get((root / relative_path).resolve()) != "file"
    ]
    if invalid_launch_chain:
        raise CampaignError(
            "source_artifacts must hash-bind the complete local launch chain as "
            f"file artifacts: {invalid_launch_chain}"
        )
    return CampaignManifest(
        path=path.resolve(),
        launcher=launcher,
        launcher_sha256=launcher_sha256,
        fixed_environment=fixed_environment,
        controlled_environment_keys=controlled,
        oscar_contract=oscar_contract,
        source_artifacts=source_artifacts,
        gates=gates,
        baseline=baseline,
        candidates=tuple(sorted(candidates, key=lambda item: item.priority)),
    )


def verify_artifact(artifact: Artifact) -> dict[str, object]:
    if artifact.kind == "source_tree":
        actual = sha256_source_tree(artifact.path)
    else:
        if artifact.path.is_symlink() or not artifact.path.is_file():
            raise CampaignError(f"artifact is not a regular file: {artifact.path}")
        actual = sha256_file(artifact.path)
    if actual != artifact.sha256:
        raise CampaignError(f"artifact SHA-256 mismatch for {artifact.path}: {actual}")
    return {"path": str(artifact.path), "sha256": actual, "kind": artifact.kind}


def inspect_oscar_contract(contract: OscarContract) -> dict[str, object]:
    """Verify model-bound OSCAR files and derive exact server telemetry.

    The admission tool hashes the artifact, checkpoint fingerprint, complete
    checkpoint, and model config.  This independently verifies the immutable
    input files, the receipt file, its internal canonical digest, and all path
    bindings before using those values as the exact /server_info contract.
    """

    artifact_proofs = {
        "calibration_artifact": verify_artifact(contract.calibration_artifact),
        "checkpoint_fingerprint": verify_artifact(contract.checkpoint_fingerprint),
        "admission_receipt": verify_artifact(contract.admission_receipt),
    }
    try:
        loaded = cast(
            object,
            json.loads(contract.admission_receipt.path.read_text(encoding="utf-8")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(
            "OSCAR admission receipt is not valid UTF-8 JSON"
        ) from error
    receipt = _required_dict(loaded, label="OSCAR admission receipt")
    required_exact: dict[str, object] = {
        "format": "dsv4-oscar-int2-admission",
        "format_version": 1,
        "admitted": True,
        "model_id": contract.model_id,
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
        "artifact_file_sha256": contract.calibration_artifact.sha256,
        "checkpoint_fingerprint_sha256": contract.checkpoint_fingerprint.sha256,
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required_exact.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise CampaignError(f"OSCAR admission receipt mismatch: {mismatches}")

    expected_paths = {
        "artifact_path": contract.calibration_artifact.path,
        "checkpoint_fingerprint_path": contract.checkpoint_fingerprint.path,
    }
    for field, expected_path in expected_paths.items():
        raw_path = receipt.get(field)
        if (
            not isinstance(raw_path, str)
            or Path(raw_path).resolve() != expected_path.resolve()
        ):
            raise CampaignError(f"OSCAR admission receipt {field} mismatch")

    sha_fields = (
        "artifact_provenance_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "admission_sha256",
    )
    for field in sha_fields:
        value = receipt.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise CampaignError(f"OSCAR admission receipt {field} is not SHA-256")
    canonical_receipt = dict(receipt)
    admission_sha256 = cast(str, canonical_receipt.pop("admission_sha256"))
    computed_admission_sha256 = hashlib.sha256(
        json.dumps(
            canonical_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if admission_sha256 != computed_admission_sha256:
        raise CampaignError("OSCAR admission receipt canonical digest mismatch")

    expected_server_info = {
        **OSCAR_STATIC_SERVER_INFO,
        "dsv4_oscar_artifact_sha256": receipt["artifact_file_sha256"],
        "dsv4_oscar_model_config_sha256": receipt["config_sha256"],
        "dsv4_oscar_artifact_provenance_sha256": receipt["artifact_provenance_sha256"],
        "dsv4_oscar_checkpoint_sha256": receipt["checkpoint_sha256"],
        "dsv4_oscar_checkpoint_fingerprint_sha256": receipt[
            "checkpoint_fingerprint_sha256"
        ],
        "dsv4_oscar_admission_sha256": admission_sha256,
        "dsv4_oscar_admission_receipt_sha256": contract.admission_receipt.sha256,
        "dsv4_oscar_wo_a_absorption_state": {
            "enabled": True,
            "consumer_role": "target_compressed",
            "target_only": True,
            "applied": True,
            "apply_count": 1,
            "artifact_sha256": receipt["artifact_file_sha256"],
            "admission_sha256": admission_sha256,
            "expected_local_compressed_layer_ids": list(range(2, 43)),
            "absorbed_local_layer_ids": list(range(2, 43)),
            "runtime_restore_skipped_layer_ids": list(range(2, 43)),
            "all_local_target_compressed_layers_absorbed": True,
            "all_local_target_compressed_layers_skip_runtime_restore": True,
            "weight_dtype": "bfloat16",
            "head_layout": "per-head-nope448-rope64",
            "fold_orientation": "wo_a_nope@rotation",
            "rope_columns_unchanged": True,
        },
    }
    return {
        "artifacts": artifact_proofs,
        "admission": receipt,
        "expected_server_info": expected_server_info,
    }


def inspect_hybrid_plan(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"expected expert plan is not a regular file: {path}")
    before = sha256_file(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    after = sha256_file(path)
    if before != after:
        raise CampaignError("expert plan changed while being inspected")
    if not isinstance(loaded, dict):
        raise CampaignError("expert plan must contain a dictionary")
    masks = loaded.get("gpu_experts_mask_by_rank")
    if not isinstance(masks, torch.Tensor) or tuple(masks.shape[:2]) != (2, 43):
        raise CampaignError("expert plan GPU masks must have shape [2,43,experts]")
    counts = masks.to(dtype=torch.bool, device="cpu").sum(dim=2).to(torch.int64)
    return {
        "path": str(path.resolve()),
        "sha256": before,
        "format": loaded.get("format"),
        "placement_semantics_sha256": loaded.get("placement_semantics_sha256"),
        "gpu_rank_counts_by_layer": counts.tolist(),
        "minimum_gpu_width": int(counts.min()),
        "maximum_gpu_width": int(counts.max()),
        "total_gpu_expert_layers_by_rank": counts.sum(dim=1).tolist(),
    }


def validate_server_info(
    info: Mapping[str, object],
    candidate: Candidate,
    plan: Mapping[str, object],
    oscar_server_info: Mapping[str, object],
) -> dict[str, object]:
    hotspot.validate_server_contract(info)
    if set(oscar_server_info) != OSCAR_OWNED_SERVER_INFO_KEYS:
        raise CampaignError(
            "OSCAR server-info proof must contain the complete owned telemetry set"
        )
    required_exact: dict[str, object] = {
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "tp_size": 2,
        "ep_size": 2,
        "pp_size": 1,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "disable_prefill_cuda_graph": False,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "speculative_dspark_fixed_verify_len": int(
            candidate.environment.get("DSV4_DSPARK_FIXED_VERIFY_LEN", "4")
        ),
        "dsv4_sm86_small_batch_gemm_configured": True,
        "kt_draft_hybrid_expert_plan_format": "sglang_kt_hybrid_expert_shard_v1",
        "kt_draft_hybrid_gpu_rank_counts_by_layer": [[14] * 3, [14] * 3],
        "kt_draft_hybrid_min_gpu_experts_per_rank_per_layer": 14,
        "kt_draft_hybrid_max_gpu_experts_per_rank_per_layer": 14,
        "kt_draft_hybrid_total_gpu_expert_layers_by_rank": [42, 42],
        "kt_draft_hybrid_source_profile_sha256": (
            "9100d1ef47685c50bc2eb3e47c34a62e39070103626479eeb398c2f5e43e4425"
        ),
        "kt_draft_hybrid_source_ordering_sha256": (
            "3f893065bf9cc3686a6a4926bf210efc4e7cc2b2035495262db4911f92f22432"
        ),
        "kt_draft_hybrid_gpu_selection_strategy": "profile-hot",
    }
    required_exact.update(oscar_server_info)
    required_exact.update(candidate.expected_server_info)
    mismatches = {
        key: {"expected": expected, "observed": info.get(key)}
        for key, expected in required_exact.items()
        if info.get(key) != expected
    }
    if mismatches:
        raise CampaignError(f"server-info contract mismatch: {mismatches}")
    if info.get("kt_hybrid_expert_plan_sha256") != plan.get("sha256"):
        raise CampaignError("server did not bind the expected expert plan SHA-256")
    draft_sha256 = info.get("kt_draft_hybrid_expert_plan_sha256")
    if (
        not isinstance(draft_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", draft_sha256) is None
    ):
        raise CampaignError("server did not publish a hash-bound draft expert plan")
    published_counts = info.get("kt_hybrid_gpu_rank_counts_by_layer")
    if published_counts != plan.get("gpu_rank_counts_by_layer"):
        raise CampaignError("server plan counts do not match the hash-bound local plan")
    return validate_sm86_small_batch_server_telemetry(info)


def _phase_values(
    receipt: Mapping[str, object], field: str, *, flushed_only: bool = False
) -> list[float]:
    raw_phases = receipt.get("phases")
    if not isinstance(raw_phases, dict):
        raise CampaignError("hotspot receipt has no phases")
    values: list[float] = []
    for raw_phase in cast(dict[object, object], raw_phases).values():
        if not isinstance(raw_phase, dict):
            raise CampaignError("hotspot phase is malformed")
        phase = cast(dict[str, object], raw_phase)
        if flushed_only and phase.get("radix_flush_before") is not True:
            continue
        benchmark = phase.get("benchmark")
        if not isinstance(benchmark, dict):
            raise CampaignError("hotspot phase benchmark is malformed")
        value = cast(dict[str, object], benchmark).get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CampaignError(f"hotspot benchmark field {field} is not numeric")
        values.append(float(value))
    return values


def _validated_nvlink_counters(
    value: object,
    *,
    label: str,
    positive: bool,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} is not an object")
    counters = cast(dict[object, object], value)
    expected = hotspot.expected_nvlink_counter_keys()
    if frozenset(counters) != expected:
        raise CampaignError(f"{label} does not contain every local NVLink counter")
    validated: dict[str, int] = {}
    for raw_key, raw_value in counters.items():
        if not isinstance(raw_key, str):
            raise CampaignError(f"{label} contains a non-string counter key")
        if (
            not isinstance(raw_value, int)
            or isinstance(raw_value, bool)
            or raw_value < (1 if positive else 0)
        ):
            qualifier = "positive" if positive else "nonnegative"
            raise CampaignError(f"{label}.{raw_key} is not a {qualifier} integer")
        validated[raw_key] = raw_value
    return validated


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CampaignError(f"{label} is not a positive integer")
    return value


def _finite_positive_number(value: object, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise CampaignError(f"{label} is not finite and positive")
    return float(value)


def _validate_trace_metric(
    trace: Mapping[str, object],
    field: str,
    *,
    valid_counts: frozenset[int],
    phase_name: str,
) -> None:
    raw_summary = trace.get(field)
    if not isinstance(raw_summary, dict):
        raise CampaignError(f"hotspot phase {phase_name} {field} is malformed")
    summary = cast(dict[str, object], raw_summary)
    count = summary.get("count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count not in valid_counts
    ):
        raise CampaignError(f"hotspot phase {phase_name} {field} count is invalid")
    for statistic in ("mean", "p50", "p95"):
        _finite_positive_number(
            summary.get(statistic),
            label=f"hotspot phase {phase_name} {field}.{statistic}",
        )


def _validate_phase_trace(
    phase: Mapping[str, object],
    *,
    phase_name: str,
    completion_tokens: int,
    stream_events: int,
    verify_length: int,
) -> None:
    raw_components = phase.get("trace_components")
    if not isinstance(raw_components, list):
        raise CampaignError(f"hotspot phase {phase_name} trace components are invalid")
    untyped_components = cast(list[object], raw_components)
    if any(not isinstance(component, str) for component in untyped_components):
        raise CampaignError(f"hotspot phase {phase_name} trace components are invalid")
    components = cast(list[str], untyped_components)
    if (
        len(components) != len(set(components))
        or frozenset(components) != hotspot.REQUIRED_TRACE_COMPONENTS
    ):
        raise CampaignError(
            f"hotspot phase {phase_name} lacks the exact required trace components"
        )

    raw_trace = phase.get("trace")
    if not isinstance(raw_trace, dict):
        raise CampaignError(f"hotspot phase {phase_name} trace is malformed")
    trace = cast(dict[str, object], raw_trace)
    cycles = _positive_int(
        trace.get("cycle_count"), label=f"hotspot phase {phase_name} cycle_count"
    )
    if cycles != stream_events:
        raise CampaignError(
            f"hotspot phase {phase_name} trace cycles do not match stream events"
        )
    committed = _positive_int(
        trace.get("committed_tokens"),
        label=f"hotspot phase {phase_name} committed_tokens",
    )
    if not cycles <= committed <= verify_length * cycles:
        raise CampaignError(f"hotspot phase {phase_name} trace totals are impossible")
    if committed - completion_tokens not in {0, 1}:
        raise CampaignError(
            f"hotspot phase {phase_name} trace/output token totals do not reconcile"
        )
    if trace.get("request_observation_count") != cycles:
        raise CampaignError(
            f"hotspot phase {phase_name} request observation count is invalid"
        )
    trace_mean = _finite_positive_number(
        trace.get("mean_committed_tokens_per_cycle"),
        label=f"hotspot phase {phase_name} mean committed tokens",
    )
    if not math.isclose(trace_mean, committed / cycles, abs_tol=1e-6):
        raise CampaignError(
            f"hotspot phase {phase_name} mean committed tokens does not reconcile"
        )

    raw_acceptance = trace.get("acceptance_distribution")
    if not isinstance(raw_acceptance, dict) or not raw_acceptance:
        raise CampaignError(
            f"hotspot phase {phase_name} acceptance distribution is malformed"
        )
    acceptance_count = 0
    acceptance_tokens = 0
    for raw_length, raw_count in cast(dict[object, object], raw_acceptance).items():
        if (
            not isinstance(raw_length, str)
            or not raw_length.isdigit()
            or str(int(raw_length)) != raw_length
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} acceptance length is malformed"
            )
        length = int(raw_length)
        count = _positive_int(
            raw_count,
            label=f"hotspot phase {phase_name} acceptance count {raw_length}",
        )
        if not 1 <= length <= verify_length:
            raise CampaignError(
                f"hotspot phase {phase_name} acceptance length is out of range"
            )
        acceptance_count += count
        acceptance_tokens += length * count
    if acceptance_count != cycles or acceptance_tokens != committed:
        raise CampaignError(
            f"hotspot phase {phase_name} acceptance distribution does not reconcile"
        )

    expected_distribution = {str(verify_length): cycles}
    if trace.get("verify_len_distribution") != expected_distribution:
        raise CampaignError(
            f"hotspot phase {phase_name} verify-length distribution is invalid"
        )
    if trace.get("verify_graph_key_distribution") != expected_distribution:
        raise CampaignError(
            f"hotspot phase {phase_name} verify graph-key distribution is invalid"
        )
    _validate_trace_metric(
        trace,
        "draft_gpu_ms",
        valid_counts=frozenset({cycles}),
        phase_name=phase_name,
    )
    _validate_trace_metric(
        trace,
        "step_gpu_ms",
        valid_counts=frozenset({cycles}),
        phase_name=phase_name,
    )
    _validate_trace_metric(
        trace,
        "target_verify_gpu_ms",
        valid_counts=frozenset({cycles}),
        phase_name=phase_name,
    )
    _validate_trace_metric(
        trace,
        "step_cpu_ms",
        valid_counts=frozenset({cycles - 1, cycles}),
        phase_name=phase_name,
    )

    source_index = phase.get("trace_source_index")
    source_counts = phase.get("trace_records_by_source")
    typed_source_counts = (
        cast(list[object], source_counts) if isinstance(source_counts, list) else None
    )
    if (
        not isinstance(source_index, int)
        or isinstance(source_index, bool)
        or source_index < 0
        or typed_source_counts is None
        or source_index >= len(typed_source_counts)
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in typed_source_counts
        )
        or typed_source_counts[source_index] != cycles
    ):
        raise CampaignError(f"hotspot phase {phase_name} trace source is invalid")


def _validate_campaign_hotspot_receipt(
    receipt: Mapping[str, object],
    *,
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> dict[str, object]:
    """Validate campaign evidence and report natural-stop trajectory drift.

    The standalone hotspot benchmark keeps its stricter paired-trajectory
    eligibility rule.  A campaign may instead treat independently valid
    natural-stop responses as repeated samples, but only when trajectory
    mismatch is provably the sole reason the aggregate hotspot flag is false.
    """

    if receipt.get("accepted") is not True:
        raise CampaignError("hotspot receipt was not accepted")
    if receipt.get("receipt_version") != hotspot.RECEIPT_VERSION:
        raise CampaignError("hotspot receipt version is unsupported")
    if receipt.get("measurement_mode") != "trace":
        raise CampaignError("hotspot receipt did not use trace measurement mode")
    if receipt.get("verify_logits_diagnostic") is not False:
        raise CampaignError("hotspot receipt enabled verify-logit diagnostics")
    verify_policy = receipt.get("verify_policy")
    if (
        not isinstance(verify_policy, str)
        or not verify_policy.isdigit()
        or not 2 <= int(verify_policy) <= 6
    ):
        raise CampaignError("hotspot receipt did not use a fixed verify policy")
    verify_length = int(verify_policy)

    provenance = receipt.get("expert_plan_provenance")
    if not isinstance(provenance, dict):
        raise CampaignError("hotspot receipt has no expert-plan provenance")
    typed_provenance = cast(dict[str, object], provenance)
    expected_plan_path = typed_provenance.get("expected_plan_path")
    expected_plan_sha256 = typed_provenance.get("expected_plan_sha256")
    if (
        typed_provenance.get("binding") != "launcher-hash-and-kt-loader-validated"
        or not isinstance(expected_plan_path, str)
        or not expected_plan_path
        or not isinstance(expected_plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256) is None
    ):
        raise CampaignError("hotspot expert-plan provenance is malformed")

    raw_nvlink = receipt.get("nvlink_traffic")
    if not isinstance(raw_nvlink, dict):
        raise CampaignError("hotspot receipt has no NVLink traffic evidence")
    nvlink = cast(dict[str, object], raw_nvlink)
    before = _validated_nvlink_counters(
        nvlink.get("counters_before"),
        label="hotspot NVLink counters_before",
        positive=False,
    )
    after = _validated_nvlink_counters(
        nvlink.get("counters_after"),
        label="hotspot NVLink counters_after",
        positive=False,
    )
    deltas = _validated_nvlink_counters(
        nvlink.get("counter_deltas"),
        label="hotspot NVLink counter_deltas",
        positive=True,
    )
    if any(after[key] - before[key] != deltas[key] for key in deltas):
        raise CampaignError("hotspot NVLink counter deltas do not reconcile")

    raw_prompt_pair = receipt.get("prompt_pair")
    if not isinstance(raw_prompt_pair, dict):
        raise CampaignError("hotspot receipt has no prompt-pair evidence")
    prompt_pair = cast(dict[str, object], raw_prompt_pair)
    exact_input_sha256 = prompt_pair.get("exact_sha256")
    near_input_sha256 = prompt_pair.get("near_sha256")
    if (
        prompt_pair.get("input_tokens") != expected_input_tokens
        or not isinstance(exact_input_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", exact_input_sha256) is None
        or not isinstance(near_input_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", near_input_sha256) is None
        or exact_input_sha256 == near_input_sha256
    ):
        raise CampaignError("hotspot prompt-pair fingerprints are malformed")
    common_prefix_tokens = prompt_pair.get("common_prefix_tokens")
    common_suffix_tokens = prompt_pair.get("common_suffix_tokens")
    mutation_record_index = prompt_pair.get("mutation_record_index")
    if (
        not isinstance(common_prefix_tokens, int)
        or isinstance(common_prefix_tokens, bool)
        or not 0 < common_prefix_tokens < expected_input_tokens
        or not isinstance(common_suffix_tokens, int)
        or isinstance(common_suffix_tokens, bool)
        or common_suffix_tokens < 0
        or common_prefix_tokens + common_suffix_tokens >= expected_input_tokens
        or not isinstance(mutation_record_index, int)
        or isinstance(mutation_record_index, bool)
        or mutation_record_index < 0
    ):
        raise CampaignError("hotspot prompt-pair structure is malformed")
    prompt_fingerprint = {
        "input_tokens": expected_input_tokens,
        "exact_sha256": exact_input_sha256,
        "near_sha256": near_input_sha256,
        "common_prefix_tokens": common_prefix_tokens,
        "common_suffix_tokens": common_suffix_tokens,
        "mutation_record_index": mutation_record_index,
    }

    raw_phases = receipt.get("phases")
    if not isinstance(raw_phases, dict):
        raise CampaignError("hotspot receipt has no phases")
    phases = cast(dict[object, object], raw_phases)
    expected_phases = {phase.name: phase for phase in hotspot.PHASES}
    if set(phases) != set(expected_phases):
        raise CampaignError("hotspot receipt does not contain the five fixed phases")
    request_ids: set[str] = set()
    output_hashes_by_phase: dict[str, str] = {}
    completion_tokens_by_phase: dict[str, int] = {}
    cached_tokens_by_phase: dict[str, int] = {}
    for phase_name, raw_phase in phases.items():
        if not isinstance(phase_name, str) or not isinstance(raw_phase, dict):
            raise CampaignError("hotspot phase is malformed")
        phase = cast(dict[str, object], raw_phase)
        phase_contract = expected_phases[phase_name]
        if (
            phase.get("prompt_kind") != phase_contract.prompt_kind
            or phase.get("radix_flush_before") is not phase_contract.flush_before
        ):
            raise CampaignError(f"hotspot phase {phase_name} contract is invalid")
        benchmark = phase.get("benchmark")
        if not isinstance(benchmark, dict):
            raise CampaignError(f"hotspot phase {phase_name} benchmark is malformed")
        typed_benchmark = cast(dict[str, object], benchmark)
        semantic = typed_benchmark.get("semantic_validation")
        if (
            typed_benchmark.get("receipt_version") != hotspot.baseline.RECEIPT_VERSION
            or typed_benchmark.get("accepted") is not True
            or typed_benchmark.get("performance_claim_eligible") is not True
            or typed_benchmark.get("receipt_safety") != "validated_natural_stop"
            or typed_benchmark.get("ignore_eos") is not False
            or typed_benchmark.get("finish_reason") != "stop"
            or typed_benchmark.get("exact_requested_token_shape") is not False
            or typed_benchmark.get("requested_max_completion_tokens")
            != expected_output_tokens
            or typed_benchmark.get("terminal_token_scan") != "verified"
            or not isinstance(semantic, dict)
            or cast(dict[str, object], semantic).get("passed") is not True
            or cast(dict[str, object], semantic).get("issue_codes") != []
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} lacks eligible natural-stop semantics"
            )
        completion_tokens = _positive_int(
            typed_benchmark.get("completion_tokens"),
            label=f"hotspot phase {phase_name} completion_tokens",
        )
        if completion_tokens >= expected_output_tokens:
            raise CampaignError(
                f"hotspot phase {phase_name} did not stop below its output cap"
            )
        output_hash = typed_benchmark.get("output_sha256")
        input_hash = typed_benchmark.get("input_sha256")
        expected_input_hash = (
            exact_input_sha256
            if phase_contract.prompt_kind == "exact"
            else near_input_sha256
        )
        if (
            not isinstance(output_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", output_hash) is None
            or input_hash != expected_input_hash
            or typed_benchmark.get("input_tokens") != expected_input_tokens
            or typed_benchmark.get("server_prompt_tokens") != expected_input_tokens
            or typed_benchmark.get("http_status") != 200
            or typed_benchmark.get("saw_done") is not True
        ):
            raise CampaignError(f"hotspot phase {phase_name} I/O proof is malformed")
        request_id = typed_benchmark.get("request_id")
        if (
            not isinstance(request_id, str)
            or not request_id.endswith(f"-{phase_name}")
            or request_id in request_ids
        ):
            raise CampaignError(f"hotspot phase {phase_name} request ID is invalid")
        request_ids.add(request_id)
        for timing_field in (
            "elapsed_seconds",
            "time_to_first_token_seconds",
            "decode_seconds",
            "decode_tokens_per_second",
            "prefill_tokens_per_second",
            "total_tokens_per_second",
        ):
            _finite_positive_number(
                typed_benchmark.get(timing_field),
                label=f"hotspot phase {phase_name} {timing_field}",
            )
        elapsed_seconds = _finite_positive_number(
            typed_benchmark.get("elapsed_seconds"),
            label=f"hotspot phase {phase_name} elapsed_seconds",
        )
        ttft_seconds = _finite_positive_number(
            typed_benchmark.get("time_to_first_token_seconds"),
            label=f"hotspot phase {phase_name} time_to_first_token_seconds",
        )
        decode_seconds = _finite_positive_number(
            typed_benchmark.get("decode_seconds"),
            label=f"hotspot phase {phase_name} decode_seconds",
        )
        decode_rate = _finite_positive_number(
            typed_benchmark.get("decode_tokens_per_second"),
            label=f"hotspot phase {phase_name} decode_tokens_per_second",
        )
        first_stream_completion_tokens = typed_benchmark.get(
            "first_stream_completion_tokens"
        )
        stream_events = typed_benchmark.get("stream_events")
        event_count = typed_benchmark.get("event_count")
        if (
            not isinstance(first_stream_completion_tokens, int)
            or isinstance(first_stream_completion_tokens, bool)
            or not 0 < first_stream_completion_tokens < completion_tokens
            or not isinstance(stream_events, int)
            or isinstance(stream_events, bool)
            or stream_events <= 0
            or not isinstance(event_count, int)
            or isinstance(event_count, bool)
            or event_count < stream_events
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} stream accounting is invalid"
            )
        if ttft_seconds + decode_seconds > elapsed_seconds + 1e-3:
            raise CampaignError(
                f"hotspot phase {phase_name} timing intervals do not reconcile"
            )
        expected_decode_rate = (
            completion_tokens - first_stream_completion_tokens
        ) / decode_seconds
        if not math.isclose(
            decode_rate, expected_decode_rate, rel_tol=1e-6, abs_tol=1e-5
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} decode rate does not reconcile"
            )
        prefill_rate = _finite_positive_number(
            typed_benchmark.get("prefill_tokens_per_second"),
            label=f"hotspot phase {phase_name} prefill_tokens_per_second",
        )
        if not math.isclose(
            prefill_rate,
            expected_input_tokens / ttft_seconds,
            rel_tol=1e-6,
            abs_tol=1e-5,
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} prefill rate does not reconcile"
            )
        total_rate = _finite_positive_number(
            typed_benchmark.get("total_tokens_per_second"),
            label=f"hotspot phase {phase_name} total_tokens_per_second",
        )
        if not math.isclose(
            total_rate,
            (expected_input_tokens + completion_tokens) / elapsed_seconds,
            rel_tol=1e-6,
            abs_tol=1e-5,
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} total rate does not reconcile"
            )
        cached_tokens = typed_benchmark.get("server_cached_tokens")
        if (
            not isinstance(cached_tokens, int)
            or isinstance(cached_tokens, bool)
            or cached_tokens < 0
        ):
            raise CampaignError(
                f"hotspot phase {phase_name} cached-token count is invalid"
            )
        output_hashes_by_phase[phase_name] = output_hash
        completion_tokens_by_phase[phase_name] = completion_tokens
        cached_tokens_by_phase[phase_name] = cached_tokens
        recorder = phase.get("expert_recorder")
        if (
            not isinstance(recorder, dict)
            or cast(dict[str, object], recorder).get("enabled") is not False
        ):
            raise CampaignError(f"hotspot phase {phase_name} used expert recording")
        _validate_phase_trace(
            phase,
            phase_name=phase_name,
            completion_tokens=completion_tokens,
            stream_events=stream_events,
            verify_length=verify_length,
        )

    for phase_name in (
        "cold_first_exact",
        "warm_no_radix_near",
        "warm_no_radix_exact",
    ):
        if cached_tokens_by_phase[phase_name] != 0:
            raise CampaignError(f"hotspot phase {phase_name} violated its cache flush")
    if not 0 < cached_tokens_by_phase["radix_hot_exact"] <= expected_input_tokens:
        raise CampaignError("hotspot exact repeat did not prove radix reuse")
    if not 0 <= cached_tokens_by_phase["radix_hot_near"] <= common_prefix_tokens:
        raise CampaignError("hotspot near repeat cache count is invalid")

    raw_repeat = receipt.get("repeat_trajectories")
    if not isinstance(raw_repeat, dict):
        raise CampaignError("hotspot receipt has no repeat-trajectory evidence")
    repeat = cast(dict[str, object], raw_repeat)
    raw_groups = repeat.get("groups")
    typed_groups = (
        cast(dict[object, object], raw_groups) if isinstance(raw_groups, dict) else None
    )
    if typed_groups is None or set(typed_groups) != {"exact", "near"}:
        raise CampaignError("hotspot repeat-trajectory groups are malformed")
    expected_group_phases = {
        "exact": ["cold_first_exact", "radix_hot_exact", "warm_no_radix_exact"],
        "near": ["radix_hot_near", "warm_no_radix_near"],
    }
    group_comparability: list[bool] = []
    completion_spread_by_group: dict[str, dict[str, object]] = {}
    completion_tokens_by_group: dict[str, list[int]] = {}
    for group_name, phase_names in expected_group_phases.items():
        raw_group = typed_groups.get(group_name)
        if not isinstance(raw_group, dict):
            raise CampaignError(f"hotspot {group_name} trajectory group is malformed")
        group = cast(dict[str, object], raw_group)
        hashes = group.get("output_sha256")
        counts = group.get("completion_tokens")
        typed_unvalidated_hashes = (
            cast(list[object], hashes) if isinstance(hashes, list) else None
        )
        typed_unvalidated_counts = (
            cast(list[object], counts) if isinstance(counts, list) else None
        )
        if (
            group.get("phase_names") != phase_names
            or typed_unvalidated_hashes is None
            or len(typed_unvalidated_hashes) != len(phase_names)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in typed_unvalidated_hashes
            )
            or typed_unvalidated_counts is None
            or len(typed_unvalidated_counts) != len(phase_names)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in typed_unvalidated_counts
            )
        ):
            raise CampaignError(
                f"hotspot {group_name} trajectory evidence is malformed"
            )
        typed_hashes = cast(list[str], typed_unvalidated_hashes)
        typed_counts = cast(list[int], typed_unvalidated_counts)
        if typed_hashes != [output_hashes_by_phase[name] for name in phase_names]:
            raise CampaignError(
                f"hotspot {group_name} trajectory hashes do not match phase receipts"
            )
        if typed_counts != [completion_tokens_by_phase[name] for name in phase_names]:
            raise CampaignError(
                f"hotspot {group_name} trajectory counts do not match phase receipts"
            )
        hashes_equal = len(set(typed_hashes)) == 1
        counts_equal = len(set(typed_counts)) == 1
        comparable = hashes_equal and counts_equal
        if (
            group.get("hashes_equal") is not hashes_equal
            or group.get("token_counts_equal") is not counts_equal
            or group.get("paired_trajectory_comparable") is not comparable
        ):
            raise CampaignError(
                f"hotspot {group_name} trajectory summary does not reconcile"
            )
        group_comparability.append(comparable)
        minimum_count = min(typed_counts)
        maximum_count = max(typed_counts)
        within_natural_stop_band = (
            NATURAL_STOP_MAXIMUM_SPREAD_DENOMINATOR * (maximum_count - minimum_count)
            <= minimum_count
        )
        if not comparable and not within_natural_stop_band:
            raise CampaignError(
                f"hotspot {group_name} natural-stop trajectory spread exceeds 25%"
            )
        completion_tokens_by_group[group_name] = typed_counts
        completion_spread_by_group[group_name] = {
            "minimum_completion_tokens": minimum_count,
            "maximum_completion_tokens": maximum_count,
            "maximum_over_minimum_ratio": maximum_count / minimum_count,
            "within_twenty_five_percent": within_natural_stop_band,
            "paired_trajectory_comparable": comparable,
        }
    all_comparable = all(group_comparability)
    if repeat.get("all_paired_trajectories_comparable") is not all_comparable:
        raise CampaignError("hotspot aggregate trajectory summary does not reconcile")

    attribution = receipt.get("attribution")
    if not isinstance(attribution, dict):
        raise CampaignError("hotspot receipt has no attribution evidence")
    page_locality = cast(dict[str, object], attribution).get(
        "cpu_weight_page_cache_locality"
    )
    if not isinstance(page_locality, dict):
        raise CampaignError("hotspot receipt has no page-locality attribution")
    exact_comparable = group_comparability[0]
    typed_page_locality = cast(dict[str, object], page_locality)
    if (
        typed_page_locality.get("exact_output_trajectory_comparable")
        is not exact_comparable
        or typed_page_locality.get("eligible") is not exact_comparable
    ):
        raise CampaignError("hotspot page-locality eligibility is inconsistent")

    overall_eligible = receipt.get("performance_claim_eligible")
    if overall_eligible is True:
        if not all_comparable:
            raise CampaignError("eligible hotspot receipt has divergent trajectories")
        claim_mode = "strict_paired"
    elif overall_eligible is not False or all_comparable:
        raise CampaignError("hotspot receipt is ineligible for a non-trajectory reason")
    else:
        claim_mode = "unpaired_natural_stop_ensemble"
    return {
        "claim_mode": claim_mode,
        "trajectory_divergent": not all_comparable,
        "prompt_fingerprint": prompt_fingerprint,
        "completion_tokens_by_group": completion_tokens_by_group,
        "completion_spread_by_group": completion_spread_by_group,
    }


def summarize_hotspot_receipts(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_input_tokens: int,
    expected_output_tokens: int,
) -> dict[str, object]:
    if not receipts:
        raise CampaignError("no hotspot receipts to summarize")
    decode_rates: list[float] = []
    flushed_ttft: list[float] = []
    committed_tokens = 0
    cycle_count = 0
    weighted_target_verify = 0.0
    target_verify_count = 0
    receipt_decode_means: list[float] = []
    trajectory_divergent_receipt_count = 0
    prompt_fingerprint: dict[str, object] | None = None
    completion_tokens_by_group: dict[str, list[int]] = {"exact": [], "near": []}
    completion_spread_by_receipt: list[dict[str, object]] = []
    for receipt in receipts:
        validation = _validate_campaign_hotspot_receipt(
            receipt,
            expected_input_tokens=expected_input_tokens,
            expected_output_tokens=expected_output_tokens,
        )
        if validation["trajectory_divergent"] is True:
            trajectory_divergent_receipt_count += 1
        receipt_prompt_fingerprint = cast(
            dict[str, object], validation["prompt_fingerprint"]
        )
        if prompt_fingerprint is None:
            prompt_fingerprint = receipt_prompt_fingerprint
        elif receipt_prompt_fingerprint != prompt_fingerprint:
            raise CampaignError("hotspot receipts used different prompt fingerprints")
        receipt_completion_tokens = cast(
            dict[str, list[int]], validation["completion_tokens_by_group"]
        )
        for group_name, group_tokens in completion_tokens_by_group.items():
            group_tokens.extend(receipt_completion_tokens[group_name])
        completion_spread_by_receipt.append(
            cast(dict[str, object], validation["completion_spread_by_group"])
        )
        receipt_decode_rates = _phase_values(receipt, "decode_tokens_per_second")
        decode_rates.extend(receipt_decode_rates)
        receipt_decode_means.append(statistics.fmean(receipt_decode_rates))
        flushed_ttft.extend(
            _phase_values(receipt, "time_to_first_token_seconds", flushed_only=True)
        )
        raw_phases = cast(dict[str, object], receipt["phases"])
        for raw_phase in raw_phases.values():
            phase = cast(dict[str, object], raw_phase)
            trace = cast(dict[str, object], phase.get("trace"))
            phase_committed = trace.get("committed_tokens")
            phase_cycles = trace.get("cycle_count")
            target = trace.get("target_verify_gpu_ms")
            if (
                not isinstance(phase_committed, int)
                or isinstance(phase_committed, bool)
                or not isinstance(phase_cycles, int)
                or isinstance(phase_cycles, bool)
                or not isinstance(target, dict)
            ):
                raise CampaignError("hotspot trace is incomplete")
            target_mean = cast(dict[str, object], target).get("mean")
            target_count = cast(dict[str, object], target).get("count")
            if (
                not isinstance(target_mean, (int, float))
                or isinstance(target_mean, bool)
                or not isinstance(target_count, int)
                or isinstance(target_count, bool)
            ):
                raise CampaignError("target-verify trace is incomplete")
            committed_tokens += phase_committed
            cycle_count += phase_cycles
            weighted_target_verify += float(target_mean) * target_count
            target_verify_count += target_count
    if not decode_rates or not flushed_ttft or cycle_count <= 0:
        raise CampaignError("hotspot receipts contain no usable timing observations")
    if target_verify_count <= 0:
        raise CampaignError("hotspot receipts contain no target-verify observations")
    if len(decode_rates) != len(receipts) * len(hotspot.PHASES):
        raise CampaignError("hotspot phase observation count is not five per receipt")
    if prompt_fingerprint is None:
        raise CampaignError("hotspot receipts have no prompt fingerprint")
    overall_completion_tokens = [
        token
        for group_tokens in completion_tokens_by_group.values()
        for token in group_tokens
    ]
    return {
        "receipt_count": len(receipts),
        "phase_observation_count": len(decode_rates),
        "decode_rates": decode_rates,
        # Each five-phase receipt is the resampling unit. Treating its five
        # correlated exact/near phases as independent would make confidence
        # intervals spuriously narrow.
        "receipt_decode_means": receipt_decode_means,
        "mean_decode_tokens_per_second": statistics.fmean(decode_rates),
        "median_decode_tokens_per_second": statistics.median(decode_rates),
        "median_flushed_ttft_seconds": statistics.median(flushed_ttft),
        "maximum_flushed_ttft_seconds": max(flushed_ttft),
        "committed_tokens": committed_tokens,
        "cycle_count": cycle_count,
        "mean_committed_tokens_per_cycle": committed_tokens / cycle_count,
        "mean_target_verify_gpu_ms": weighted_target_verify / target_verify_count,
        "prompt_fingerprint": prompt_fingerprint,
        "completion_tokens_by_prompt_group": completion_tokens_by_group,
        "mean_completion_tokens_by_prompt_group": {
            group_name: statistics.fmean(group_tokens)
            for group_name, group_tokens in completion_tokens_by_group.items()
        },
        "mean_completion_tokens": statistics.fmean(overall_completion_tokens),
        "natural_stop_completion_spread_by_receipt": completion_spread_by_receipt,
        "trajectory_divergent_receipt_count": trajectory_divergent_receipt_count,
        "performance_claim_methodology": (
            "unpaired_natural_stop_ensemble"
            if trajectory_divergent_receipt_count
            else "strict_paired"
        ),
        "performance_claim_scope": (
            "whole_receipt_decode_ttft_only"
            if trajectory_divergent_receipt_count
            else "paired_phase_and_whole_receipt"
        ),
    }


def absolute_goal_state(
    summary: Mapping[str, object], gates: Gates
) -> dict[str, object]:
    """Report the user's absolute outcome separately from experiment admission.

    Relative screens remain useful even when the current machine is far below the
    requested outcome.  Consequently these fields are advisory for ``qualified``
    and ``promoted``; they become the explicit final-result gate instead of being
    hidden behind a baseline ratio.
    """

    mean_decode = _finite_positive_number(
        summary.get("mean_decode_tokens_per_second"),
        label="mean decode tokens per second",
    )
    median_decode = _finite_positive_number(
        summary.get("median_decode_tokens_per_second"),
        label="median decode tokens per second",
    )
    median_ttft = _finite_positive_number(
        summary.get("median_flushed_ttft_seconds"),
        label="median flushed TTFT",
    )
    maximum_ttft = _finite_positive_number(
        summary.get("maximum_flushed_ttft_seconds"),
        label="maximum flushed TTFT",
    )
    target_decode_met = mean_decode >= ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND
    stretch_decode_met = mean_decode >= ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND
    # The historical campaign gate applies to the median.  The user's wording is
    # "at most 7s", so the absolute outcome is deliberately stricter and checks
    # the slowest flushed observation while reporting both statistics.
    maximum_ttft_met = maximum_ttft <= gates.maximum_median_ttft_seconds
    advisories: list[str] = []
    if not target_decode_met:
        advisories.append("absolute_80_tps_goal_not_met")
    if not stretch_decode_met:
        advisories.append("absolute_90_tps_stretch_goal_not_met")
    if not maximum_ttft_met:
        advisories.append("absolute_7s_maximum_ttft_goal_not_met")
    return {
        "diagnostic_qualification_independent": True,
        "decode_metric": "five_phase_mean_decode_tokens_per_second",
        "ttft_metric": "maximum_flushed_ttft_seconds",
        "performance_claim_methodology": summary.get("performance_claim_methodology"),
        "observed_mean_decode_tokens_per_second": mean_decode,
        "observed_median_decode_tokens_per_second": median_decode,
        "observed_median_flushed_ttft_seconds": median_ttft,
        "observed_maximum_flushed_ttft_seconds": maximum_ttft,
        "target_decode_tokens_per_second": (ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND),
        "stretch_decode_tokens_per_second": (ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND),
        "maximum_ttft_seconds": gates.maximum_median_ttft_seconds,
        "target_decode_ratio": (mean_decode / ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND),
        "stretch_decode_ratio": (
            mean_decode / ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND
        ),
        "target_decode_shortfall_tokens_per_second": max(
            0.0, ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND - mean_decode
        ),
        "stretch_decode_shortfall_tokens_per_second": max(
            0.0, ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND - mean_decode
        ),
        "target_decode_met": target_decode_met,
        "stretch_decode_met": stretch_decode_met,
        "maximum_ttft_met": maximum_ttft_met,
        "target_configuration_goal_met": target_decode_met and maximum_ttft_met,
        "stretch_configuration_goal_met": stretch_decode_met and maximum_ttft_met,
        "advisories": advisories,
    }


def absolute_goal_contract(gates: Gates) -> dict[str, object]:
    return {
        "target_decode_tokens_per_second": (ABSOLUTE_TARGET_DECODE_TOKENS_PER_SECOND),
        "stretch_decode_tokens_per_second": (ABSOLUTE_STRETCH_DECODE_TOKENS_PER_SECOND),
        "maximum_ttft_seconds": gates.maximum_median_ttft_seconds,
        "decode_metric": "five_phase_mean_decode_tokens_per_second",
        "ttft_metric": "maximum_flushed_ttft_seconds",
        "diagnostic_qualification_independent": True,
    }


def bootstrap_mean_ratio_interval(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not candidate_values or not baseline_values:
        raise CampaignError("bootstrap inputs must not be empty")
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(samples):
        candidate_mean = statistics.fmean(
            generator.choice(candidate_values) for _ in candidate_values
        )
        baseline_mean = statistics.fmean(
            generator.choice(baseline_values) for _ in baseline_values
        )
        ratios.append(candidate_mean / baseline_mean)
    ratios.sort()
    lower = ratios[max(0, int(samples * 0.025) - 1)]
    upper = ratios[min(samples - 1, int(samples * 0.975))]
    return lower, upper


def compare_summaries(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    if candidate.get("prompt_fingerprint") != baseline.get("prompt_fingerprint"):
        raise CampaignError("candidate and baseline prompt fingerprints differ")
    candidate_decode = cast(list[float], candidate["receipt_decode_means"])
    baseline_decode = cast(list[float], baseline["receipt_decode_means"])
    lower, upper = bootstrap_mean_ratio_interval(
        candidate_decode,
        baseline_decode,
        samples=bootstrap_samples,
        seed=seed,
    )
    candidate_completion_by_group = cast(
        dict[str, object], candidate["mean_completion_tokens_by_prompt_group"]
    )
    baseline_completion_by_group = cast(
        dict[str, object], baseline["mean_completion_tokens_by_prompt_group"]
    )
    completion_ratios = {
        group_name: _finite_positive_number(
            candidate_completion_by_group.get(group_name),
            label=f"candidate {group_name} mean completion tokens",
        )
        / _finite_positive_number(
            baseline_completion_by_group.get(group_name),
            label=f"baseline {group_name} mean completion tokens",
        )
        for group_name in ("exact", "near")
    }
    completion_ratios["overall"] = _finite_positive_number(
        candidate.get("mean_completion_tokens"),
        label="candidate mean completion tokens",
    ) / _finite_positive_number(
        baseline.get("mean_completion_tokens"),
        label="baseline mean completion tokens",
    )
    return {
        "mean_decode_ratio": _finite_positive_number(
            candidate.get("mean_decode_tokens_per_second"),
            label="candidate mean decode rate",
        )
        / _finite_positive_number(
            baseline.get("mean_decode_tokens_per_second"),
            label="baseline mean decode rate",
        ),
        "decode_ratio_bootstrap_95_percent": [lower, upper],
        "acceptance_ratio": _finite_positive_number(
            candidate.get("mean_committed_tokens_per_cycle"),
            label="candidate mean committed tokens",
        )
        / _finite_positive_number(
            baseline.get("mean_committed_tokens_per_cycle"),
            label="baseline mean committed tokens",
        ),
        "target_verify_ratio": _finite_positive_number(
            candidate.get("mean_target_verify_gpu_ms"),
            label="candidate target-verify time",
        )
        / _finite_positive_number(
            baseline.get("mean_target_verify_gpu_ms"),
            label="baseline target-verify time",
        ),
        "flushed_ttft_ratio": _finite_positive_number(
            candidate.get("median_flushed_ttft_seconds"),
            label="candidate median flushed TTFT",
        )
        / _finite_positive_number(
            baseline.get("median_flushed_ttft_seconds"),
            label="baseline median flushed TTFT",
        ),
        "mean_completion_token_ratios": completion_ratios,
        "completion_length_fairness_band": [
            MINIMUM_COMPLETION_LENGTH_RATIO,
            MAXIMUM_COMPLETION_LENGTH_RATIO,
        ],
    }


def _result_path(work_directory: Path, candidate_id: str, stage: str) -> Path:
    return work_directory / candidate_id / f"{stage}.json"


def _load_result(
    path: Path, *, expected_manifest_sha256: str | None = None
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    loaded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict) or loaded.get("format") != RESULT_FORMAT:
        raise CampaignError(f"malformed campaign result: {path}")
    if (
        expected_manifest_sha256 is not None
        and loaded.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise CampaignError(f"stale campaign result from another manifest: {path}")
    return cast(dict[str, object], loaded)


def _combined_candidate_summary(
    work_directory: Path,
    candidate_id: str,
    *,
    expected_manifest_sha256: str,
    gates: Gates,
) -> dict[str, object] | None:
    receipts: list[Mapping[str, object]] = []
    for stage in ("baseline", "screen", "confirm"):
        result = _load_result(
            _result_path(work_directory, candidate_id, stage),
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if result is None:
            continue
        raw_receipts = result.get("hotspot_receipts")
        if not isinstance(raw_receipts, list):
            raise CampaignError("campaign result has no hotspot receipt list")
        receipts.extend(
            cast(dict[str, object], item)
            for item in cast(list[object], raw_receipts)
            if isinstance(item, dict)
        )
    return (
        summarize_hotspot_receipts(
            receipts,
            expected_input_tokens=gates.input_tokens,
            expected_output_tokens=gates.output_tokens,
        )
        if receipts
        else None
    )


def candidate_decision(
    candidate: Candidate,
    candidate_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    gates: Gates,
    *,
    confirmation: bool,
) -> dict[str, object]:
    seed = int.from_bytes(
        hashlib.sha256(candidate.identifier.encode("utf-8")).digest()[:8], "big"
    )
    comparison = compare_summaries(
        candidate_summary,
        baseline_summary,
        bootstrap_samples=gates.bootstrap_samples,
        seed=seed,
    )
    reasons: list[str] = []
    if (
        _finite_positive_number(
            candidate_summary.get("median_flushed_ttft_seconds"),
            label="candidate median flushed TTFT",
        )
        > gates.maximum_median_ttft_seconds
    ):
        reasons.append("median_ttft_above_gate")
    if (
        _finite_positive_number(
            comparison.get("mean_decode_ratio"), label="mean decode ratio"
        )
        < gates.minimum_screen_decode_ratio
    ):
        reasons.append("decode_ratio_below_screen_gate")
    if (
        _finite_positive_number(
            comparison.get("acceptance_ratio"), label="acceptance ratio"
        )
        < gates.minimum_acceptance_ratio
    ):
        reasons.append("acceptance_ratio_below_gate")
    if (
        _finite_positive_number(
            comparison.get("target_verify_ratio"), label="target-verify ratio"
        )
        > gates.maximum_target_verify_ratio
    ):
        reasons.append("target_verify_ratio_above_gate")
    completion_ratios = cast(
        dict[str, float], comparison["mean_completion_token_ratios"]
    )
    if any(
        not MINIMUM_COMPLETION_LENGTH_RATIO
        <= float(completion_ratios[group_name])
        <= MAXIMUM_COMPLETION_LENGTH_RATIO
        for group_name in ("exact", "near", "overall")
    ):
        reasons.append("completion_length_ratio_outside_fairness_band")
    interval = cast(list[float], comparison["decode_ratio_bootstrap_95_percent"])
    if confirmation and interval[0] < gates.minimum_confirm_ci_ratio:
        reasons.append("decode_ratio_confidence_lower_bound_below_gate")
    if candidate.policy == "speed" and confirmation and interval[0] <= 1.0:
        reasons.append("speed_candidate_not_significantly_faster")
    goals = absolute_goal_state(candidate_summary, gates)
    return {
        "promoted": not reasons,
        "confirmation": confirmation,
        "policy": candidate.policy,
        "reasons": reasons,
        "comparison": comparison,
        # Missing an absolute outcome is intentionally advisory here.  A screen
        # can still identify a better kernel or placement while the full 80/90
        # token/s target remains unmet.
        "absolute_goal_state": goals,
        "advisories": goals["advisories"],
    }


def summarize_campaign(
    manifest: CampaignManifest, work_directory: Path
) -> dict[str, object]:
    manifest_sha256 = sha256_file(manifest.path)
    baseline_result = _load_result(
        _result_path(work_directory, manifest.baseline.identifier, "baseline"),
        expected_manifest_sha256=manifest_sha256,
    )
    baseline_summary = _combined_candidate_summary(
        work_directory,
        manifest.baseline.identifier,
        expected_manifest_sha256=manifest_sha256,
        gates=manifest.gates,
    )
    baseline_goal_state = (
        absolute_goal_state(baseline_summary, manifest.gates)
        if baseline_summary is not None
        else None
    )
    candidate_rows: list[dict[str, object]] = []
    confirmed: set[str] = (
        {manifest.baseline.identifier}
        if baseline_result is not None
        and baseline_result.get("qualified") is True
        and baseline_summary is not None
        else set()
    )
    for candidate in manifest.candidates:
        screen_result = _load_result(
            _result_path(work_directory, candidate.identifier, "screen"),
            expected_manifest_sha256=manifest_sha256,
        )
        confirm_result = _load_result(
            _result_path(work_directory, candidate.identifier, "confirm"),
            expected_manifest_sha256=manifest_sha256,
        )
        summary = _combined_candidate_summary(
            work_directory,
            candidate.identifier,
            expected_manifest_sha256=manifest_sha256,
            gates=manifest.gates,
        )
        decision = None
        status = "waiting_prerequisites"
        if not set(candidate.prerequisites).issubset(confirmed):
            status = "waiting_prerequisites"
        elif screen_result is None:
            status = "pending_screen"
        elif screen_result.get("qualified") is not True:
            status = "rejected_screen_contract"
        elif summary is None or baseline_summary is None:
            status = "waiting_baseline"
        else:
            decision = candidate_decision(
                candidate,
                summary,
                baseline_summary,
                manifest.gates,
                confirmation=confirm_result is not None,
            )
            if confirm_result is None:
                status = (
                    "pending_confirmation"
                    if decision["promoted"] is True
                    else "rejected_screen_performance"
                )
            elif confirm_result.get("qualified") is not True:
                status = "rejected_confirmation_contract"
            elif decision["promoted"] is True:
                status = "confirmed"
                confirmed.add(candidate.identifier)
            else:
                status = "rejected_confirmation_performance"
        candidate_rows.append(
            {
                "id": candidate.identifier,
                "priority": candidate.priority,
                "description": candidate.description,
                "policy": candidate.policy,
                "prerequisites": list(candidate.prerequisites),
                "status": status,
                "predicted": candidate.predicted,
                "summary": summary,
                "absolute_goal_state": (
                    absolute_goal_state(summary, manifest.gates)
                    if summary is not None
                    else None
                ),
                "decision": decision,
            }
        )
    next_action: dict[str, str] | None = None
    if baseline_result is None or baseline_result.get("qualified") is not True:
        next_action = {"candidate": manifest.baseline.identifier, "stage": "baseline"}
    else:
        for row in candidate_rows:
            if row["status"] == "pending_screen":
                next_action = {"candidate": cast(str, row["id"]), "stage": "screen"}
                break
            if row["status"] == "pending_confirmation":
                next_action = {
                    "candidate": cast(str, row["id"]),
                    "stage": "confirm",
                }
                break
    return {
        "format": SUMMARY_FORMAT,
        "manifest": str(manifest.path),
        "manifest_sha256": manifest_sha256,
        "absolute_goal_contract": absolute_goal_contract(manifest.gates),
        "baseline": baseline_summary,
        "baseline_absolute_goal_state": baseline_goal_state,
        "candidates": candidate_rows,
        "next_action": next_action,
    }


def select_explicit_candidate_stage(
    manifest: CampaignManifest,
    work_directory: Path,
    candidate_id: str,
    stage: str,
) -> Candidate:
    """Validate a user-selected screen/confirmation without weakening gates."""
    if stage not in {"screen", "confirm"}:
        raise CampaignError("explicit candidate stage must be screen or confirm")
    candidates = {candidate.identifier: candidate for candidate in manifest.candidates}
    candidate = candidates.get(candidate_id)
    if candidate is None:
        raise CampaignError("explicit candidate ID is unknown or is the baseline")
    manifest_sha256 = sha256_file(manifest.path)
    baseline_result = _load_result(
        _result_path(work_directory, manifest.baseline.identifier, "baseline"),
        expected_manifest_sha256=manifest_sha256,
    )
    if baseline_result is None or baseline_result.get("qualified") is not True:
        raise CampaignError(
            "explicit candidate execution requires a qualified baseline"
        )
    summary = summarize_campaign(manifest, work_directory)
    rows = {
        cast(str, row["id"]): row
        for row in cast(list[dict[str, object]], summary["candidates"])
    }
    confirmed = {
        identifier
        for identifier, row in rows.items()
        if row.get("status") == "confirmed"
    }
    confirmed.add(manifest.baseline.identifier)
    missing = sorted(set(candidate.prerequisites) - confirmed)
    if missing:
        raise CampaignError(
            f"explicit candidate prerequisites are not confirmed: {missing}"
        )
    destination = _result_path(work_directory, candidate.identifier, stage)
    if destination.exists():
        raise CampaignError(f"refusing to overwrite campaign receipt: {destination}")
    if stage == "confirm":
        screen = _load_result(
            _result_path(work_directory, candidate.identifier, "screen"),
            expected_manifest_sha256=manifest_sha256,
        )
        if screen is None or screen.get("qualified") is not True:
            raise CampaignError(
                "explicit confirmation requires an existing qualified screen"
            )
    return candidate


def _query_gpu_inventory() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"index": fields[0], "name": fields[1], "uuid": fields[2]})
    if [row["index"] for row in rows[:2]] != ["0", "1"]:
        raise CampaignError("local GPUs 0 and 1 are not both present")
    return rows[:2]


def _query_compute_pids() -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_ids: list[int] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value or value == "N/A":
            continue
        if not value.isdecimal():
            raise CampaignError(
                f"nvidia-smi returned an invalid compute PID: {value!r}"
            )
        process_ids.append(int(value))
    return process_ids


def _query_free_memory_mib() -> list[int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in completed.stdout.splitlines()[:2]]
    if len(values) != 2:
        raise CampaignError("nvidia-smi did not report free memory for both GPUs")
    return values


def _http_status(url: str, timeout_seconds: float) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with cast(
            HttpStatusResponse,
            urllib.request.urlopen(request, timeout=timeout_seconds),
        ) as response:
            return response.status
    except urllib.error.HTTPError as error:
        try:
            return int(error.code)
        finally:
            error.close()


def _wait_for_server(
    url: str,
    timeout_seconds: float,
    *,
    process: ProcessStatus | None = None,
    poll_interval_seconds: float = 1.0,
) -> dict[str, object]:
    """Wait for SGLang's post-warmup state, then obtain server telemetry.

    ``/server_info`` is callable while the tokenizer manager is still in
    ``Starting``.  ``/health`` returns 503 in that state and becomes 200 only
    after the background generic warmup marks the server ``Up``.  Requiring the
    health transition closes the race without issuing a model request from the
    campaign controller.
    """

    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("server wait timeouts must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    health_url = derive_endpoint(url, "/health")
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise CampaignError(
                "launcher exited before the post-warmup health gate became ready"
            )
        remaining = deadline - time.monotonic()
        probe_timeout = min(10.0, max(0.1, remaining))
        try:
            health_status = _http_status(health_url, probe_timeout)
            if health_status != 200:
                last_error = CampaignError(
                    f"post-warmup health gate returned HTTP {health_status}"
                )
            else:
                info = hotspot.get_server_info(url, probe_timeout)
                if process is not None and process.poll() is not None:
                    raise CampaignError(
                        "launcher exited after the post-warmup health gate"
                    )
                return info
        except (
            urllib.error.URLError,
            TimeoutError,
            hotspot.HotspotBenchmarkError,
        ) as error:
            last_error = error
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    error_name = type(last_error).__name__ if last_error is not None else "timeout"
    raise CampaignError(
        f"server did not pass the post-warmup health gate: {error_name}"
    )


def _stop_owned_process(
    process: subprocess.Popen[bytes], timeout_seconds: float
) -> str:
    if process.poll() is not None:
        return "already_exited"
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_seconds)
        return "sigterm"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)
        return "sigkill_after_timeout"


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def execute_candidate_stage(
    manifest: CampaignManifest,
    candidate: Candidate,
    stage: str,
    work_directory: Path,
) -> dict[str, object]:
    if stage not in {"baseline", "screen", "confirm"}:
        raise CampaignError(f"unsupported campaign stage {stage}")
    if _query_compute_pids():
        raise CampaignError("GPU compute processes already exist; refusing to launch")
    gpu_inventory = _query_gpu_inventory()
    for artifact in (*manifest.source_artifacts, *candidate.artifacts):
        verify_artifact(artifact)
    oscar_proof_before_launch = inspect_oscar_contract(manifest.oscar_contract)
    if not manifest.launcher.is_file() or manifest.launcher.is_symlink():
        raise CampaignError("launcher must be a regular local file")
    if sha256_file(manifest.launcher) != manifest.launcher_sha256:
        raise CampaignError("launcher SHA-256 does not match the campaign manifest")
    if not candidate.plan_materialized_at_launch:
        plan_before_launch = inspect_hybrid_plan(candidate.expected_plan)
        if (
            candidate.expected_plan_sha256 is not None
            and plan_before_launch["sha256"] != candidate.expected_plan_sha256
        ):
            raise CampaignError("candidate plan does not match manifest SHA-256")

    stage_directory = work_directory / candidate.identifier
    stage_directory.mkdir(parents=True, exist_ok=True)
    log_path = stage_directory / f"{stage}.server.log"
    environment = os.environ.copy()
    # Eliminate hidden launch knobs, including ones a caller forgot to list in
    # the manifest. Every DSV4/SGLang/KT/NCCL setting admitted below is recorded.
    for key in tuple(environment):
        if SAFE_ENV_KEY.fullmatch(key):
            environment.pop(key)
    environment.update(manifest.fixed_environment)
    environment.update(candidate.environment)
    if not candidate.plan_materialized_at_launch:
        environment["DSV4_HYBRID_EXPERT_SHARD_PLAN"] = str(candidate.expected_plan)
    environment_config = {
        key: environment[key]
        for key in sorted(manifest.controlled_environment_keys)
        if key in environment
    }
    environment_sha256 = hashlib.sha256(
        json.dumps(environment_config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    repetitions = {
        "baseline": manifest.gates.baseline_repetitions,
        "screen": manifest.gates.screen_repetitions,
        "confirm": manifest.gates.confirmation_repetitions,
    }[stage]
    process: subprocess.Popen[bytes] | None = None
    shutdown_method = "not_started"
    primary_error: BaseException | None = None
    result: dict[str, object] = {
        "format": RESULT_FORMAT,
        "manifest_sha256": sha256_file(manifest.path),
        "candidate": candidate.identifier,
        "stage": stage,
        "qualified": False,
        "absolute_goal_contract": absolute_goal_contract(manifest.gates),
        "started_unix_seconds": time.time(),
        "launcher": {
            "path": str(manifest.launcher),
            "sha256": sha256_file(manifest.launcher),
        },
        "environment": environment_config,
        "environment_sha256": environment_sha256,
        "gpu_inventory": gpu_inventory,
        "oscar_contract": oscar_proof_before_launch,
        "hotspot_receipts": [],
    }
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                [str(manifest.launcher), "--launch"],
                cwd=manifest.launcher.parent.parent,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            info = _wait_for_server(
                manifest.gates.server_url,
                manifest.gates.startup_timeout_seconds,
                process=process,
            )
            if process.poll() is not None:
                raise CampaignError("launcher exited while server was becoming ready")
            oscar_proof_after_launch = inspect_oscar_contract(manifest.oscar_contract)
            if oscar_proof_after_launch != oscar_proof_before_launch:
                raise CampaignError("OSCAR contract changed during candidate launch")
            result["sm86_small_batch_gemm_log_proof"] = validate_sm86_small_batch_log(
                log_path
            )
            plan = inspect_hybrid_plan(candidate.expected_plan)
            if (
                candidate.expected_plan_sha256 is not None
                and plan["sha256"] != candidate.expected_plan_sha256
            ):
                raise CampaignError("materialized candidate plan SHA-256 mismatch")
            result["sm86_small_batch_gemm_server_proof"] = validate_server_info(
                info,
                candidate,
                plan,
                cast(
                    dict[str, object],
                    oscar_proof_after_launch["expected_server_info"],
                ),
            )
            scale_fold_worker_pids: list[int] | None = None
            scale_fold_mode = environment.get("KT_MXFP4_AVX_SCALE_FOLD_MODE", "off")
            if scale_fold_mode == KT_MXFP4_AVX_SCALE_FOLD_MODE:
                inline_proof = validate_kt_single_numa_inline_dispatch_server_telemetry(
                    info
                )
                inline_worker_pids = cast(list[int], inline_proof["worker_pids"])
                scale_fold_worker_pids = inline_worker_pids
                result["kt_single_numa_inline_dispatch_server_proof"] = inline_proof
                result["kt_mxfp4_avx_scale_fold_server_proof"] = (
                    validate_mxfp4_avx_scale_fold_server_telemetry(
                        info,
                        inline_worker_pids=inline_worker_pids,
                    )
                )
            result["plan"] = plan
            result["server_info"] = {
                key: info.get(key)
                for key in sorted(
                    {
                        *candidate.expected_server_info,
                        *OSCAR_OWNED_SERVER_INFO_KEYS,
                        "context_length",
                        "max_total_tokens",
                        "kv_cache_dtype",
                        "tp_size",
                        "ep_size",
                        "pp_size",
                        "kt_cpuinfer",
                        "disable_cuda_graph",
                        "disable_decode_cuda_graph",
                        "disable_prefill_cuda_graph",
                        "cuda_graph_backend_decode",
                        "cuda_graph_backend_prefill",
                        "kt_hybrid_expert_plan_sha256",
                        "kt_hybrid_expert_plan_format",
                        "kt_hybrid_placement_semantics_sha256",
                        "kt_hybrid_gpu_rank_counts_by_layer",
                        "kt_draft_hybrid_expert_plan_sha256",
                        "kt_draft_hybrid_expert_plan_format",
                        "kt_draft_hybrid_gpu_rank_counts_by_layer",
                        "kt_draft_hybrid_source_profile_sha256",
                        "kt_draft_hybrid_source_ordering_sha256",
                        "kt_draft_hybrid_gpu_selection_strategy",
                        "speculative_dspark_block_size",
                        "speculative_dspark_fixed_verify_len",
                        "dsv4_sm86_small_batch_gemm_configured",
                        "dsv4_sm86_small_batch_gemm_expected_worker_count",
                        "dsv4_sm86_small_batch_gemm_reporting_worker_count",
                        "dsv4_sm86_small_batch_gemm_active_worker_count",
                        "dsv4_sm86_small_batch_gemm_all_workers_active",
                        "dsv4_sm86_small_batch_gemm_worker_telemetry",
                        "kt_single_numa_inline_dispatch_configured",
                        "kt_single_numa_inline_dispatch_expected_worker_count",
                        "kt_single_numa_inline_dispatch_reporting_worker_count",
                        "kt_single_numa_inline_dispatch_active_worker_count",
                        "kt_single_numa_inline_dispatch_invalid_worker_count",
                        "kt_single_numa_inline_dispatch_duplicate_worker_count",
                        "kt_single_numa_inline_dispatch_rank_coverage_valid",
                        "kt_single_numa_inline_dispatch_ep2_topology_valid",
                        "kt_single_numa_inline_dispatch_all_workers_active",
                        "kt_single_numa_inline_dispatch_worker_telemetry",
                        "kt_mxfp4_avx_scale_fold_configured",
                        "kt_mxfp4_avx_scale_fold_requested_mode",
                        "kt_mxfp4_avx_scale_fold_expected_n_block",
                        "kt_mxfp4_avx_scale_fold_expected_worker_count",
                        "kt_mxfp4_avx_scale_fold_reporting_worker_count",
                        "kt_mxfp4_avx_scale_fold_active_worker_count",
                        "kt_mxfp4_avx_scale_fold_invalid_worker_count",
                        "kt_mxfp4_avx_scale_fold_duplicate_worker_count",
                        "kt_mxfp4_avx_scale_fold_rank_coverage_valid",
                        "kt_mxfp4_avx_scale_fold_ep2_topology_valid",
                        "kt_mxfp4_avx_scale_fold_all_workers_active",
                        "kt_mxfp4_avx_scale_fold_worker_telemetry",
                    }
                )
            }
            agents_context = manifest.gates.agents_path.read_text(encoding="utf-8")
            coherence = coherency.run_gate(
                completion_url=manifest.gates.chat_url,
                flush_url=derive_endpoint(
                    manifest.gates.chat_url, "/flush_cache", "timeout=30"
                ),
                model=str(manifest.gates.model_path),
                agents_context=agents_context,
                repetitions=manifest.gates.coherency_repetitions,
                maximum_tokens=256,
                tool_maximum_tokens=128,
                timeout_seconds=manifest.gates.request_timeout_seconds,
                flush_timeout_seconds=60.0,
                validate_tool_call=True,
            )
            result["coherency"] = coherence
            if coherence.get("coherent") is not True:
                raise CampaignError("strict semantic/tool coherency gate failed")
            if scale_fold_worker_pids is not None:
                # _wait_for_server returned only after SGLang's generic warmup;
                # the coherency requests above additionally guarantee a live
                # decode before parsing the PID-tagged native dispatch proof.
                # No request-time worker collective is introduced.
                result["kt_mxfp4_avx_scale_fold_dispatch_log_proof"] = (
                    validate_mxfp4_avx_scale_fold_dispatch_log(
                        log_path,
                        expected_worker_pids=scale_fold_worker_pids,
                    )
                )
            benchmark_receipts: list[dict[str, object]] = []
            for _ in range(repetitions):
                verify_policy = candidate.environment.get(
                    "DSV4_DSPARK_FIXED_VERIFY_LEN", "4"
                )
                benchmark_receipts.append(
                    hotspot.run_hotspot(
                        hotspot.HotspotArguments(
                            generate_url=derive_endpoint(
                                manifest.gates.server_url, "/generate"
                            ),
                            server_info_url=manifest.gates.server_url,
                            flush_url=derive_endpoint(
                                manifest.gates.server_url,
                                "/flush_cache",
                                "timeout=30",
                            ),
                            control_url=derive_endpoint(
                                manifest.gates.server_url, "/set_internal_state"
                            ),
                            hotspot_url=derive_endpoint(
                                manifest.gates.server_url, "/kt_expert_hotspot"
                            ),
                            model_path=manifest.gates.model_path,
                            input_tokens=manifest.gates.input_tokens,
                            output_tokens=manifest.gates.output_tokens,
                            timeout_seconds=manifest.gates.request_timeout_seconds,
                            flush_timeout_seconds=60.0,
                            progress_every=0,
                            ignore_eos=False,
                            near_prefix_ratio=0.70,
                            verify_policy=verify_policy,
                            hotspot_plan=None,
                            hotspot_generation=None,
                            hotspot_commit=False,
                            expert_recorder_directory=None,
                            expected_expert_plan=candidate.expected_plan,
                            require_nvlink_traffic=True,
                            require_trace=True,
                            output_file=None,
                        )
                    )
                )
            result["hotspot_receipts"] = benchmark_receipts
            summary = summarize_hotspot_receipts(
                benchmark_receipts,
                expected_input_tokens=manifest.gates.input_tokens,
                expected_output_tokens=manifest.gates.output_tokens,
            )
            result["summary"] = summary
            goals = absolute_goal_state(summary, manifest.gates)
            result["absolute_goal_state"] = goals
            result["advisories"] = goals["advisories"]
            if (
                _finite_positive_number(
                    summary.get("median_flushed_ttft_seconds"),
                    label="candidate median flushed TTFT",
                )
                > manifest.gates.maximum_median_ttft_seconds
            ):
                raise CampaignError("candidate median TTFT exceeds campaign gate")
            free_memory_mib = _query_free_memory_mib()
            result["post_graph_free_memory_mib"] = free_memory_mib
            if min(free_memory_mib) < manifest.gates.minimum_headroom_mib:
                raise CampaignError("candidate physical GPU headroom is below gate")
            result["qualified"] = True
    except Exception as error:  # noqa: BLE001 - safe receipt, then re-raise below
        primary_error = error
        result["error_type"] = type(error).__name__
        result["error_sha256"] = hashlib.sha256(str(error).encode()).hexdigest()
    finally:
        if process is not None:
            try:
                shutdown_method = _stop_owned_process(
                    process, manifest.gates.shutdown_timeout_seconds
                )
            except Exception as shutdown_error:  # noqa: BLE001 - preserve primary error
                result["shutdown_error_type"] = type(shutdown_error).__name__
                if primary_error is None:
                    primary_error = shutdown_error
        result["shutdown_method"] = shutdown_method
        result["finished_unix_seconds"] = time.time()
        residual_pids = _query_compute_pids()
        result["residual_compute_pids"] = residual_pids
        if residual_pids:
            result["qualified"] = False
            if primary_error is None:
                primary_error = CampaignError("model GPU processes remained after stop")
        _write_json_atomic(
            _result_path(work_directory, candidate.identifier, stage), result
        )
    if primary_error is not None:
        raise CampaignError(
            f"candidate {candidate.identifier}/{stage} failed; see safe receipt"
        ) from primary_error
    return result


def preflight_manifest(manifest: CampaignManifest) -> dict[str, object]:
    if not manifest.launcher.is_file() or manifest.launcher.is_symlink():
        raise CampaignError("launcher must be a regular local file")
    if not manifest.gates.agents_path.is_file():
        raise CampaignError("coherency context file is missing")
    if not manifest.gates.model_path.exists():
        raise CampaignError("model path is missing")
    actual_launcher_sha256 = sha256_file(manifest.launcher)
    if actual_launcher_sha256 != manifest.launcher_sha256:
        raise CampaignError("launcher SHA-256 does not match the campaign manifest")
    artifacts = [verify_artifact(item) for item in manifest.source_artifacts]
    oscar_contract = inspect_oscar_contract(manifest.oscar_contract)
    candidates: list[dict[str, object]] = []
    for candidate in (manifest.baseline, *manifest.candidates):
        candidate_artifacts = [verify_artifact(item) for item in candidate.artifacts]
        plan = None
        if candidate.expected_plan.is_file():
            plan = inspect_hybrid_plan(candidate.expected_plan)
            if (
                candidate.expected_plan_sha256 is not None
                and plan["sha256"] != candidate.expected_plan_sha256
            ):
                raise CampaignError(
                    f"candidate {candidate.identifier} plan SHA-256 mismatch"
                )
        elif not candidate.plan_materialized_at_launch:
            raise CampaignError(
                f"candidate {candidate.identifier} plan is missing: "
                f"{candidate.expected_plan}"
            )
        candidates.append(
            {
                "id": candidate.identifier,
                "environment_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            **manifest.fixed_environment,
                            **candidate.environment,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "artifacts": candidate_artifacts,
                "plan": plan,
                "plan_materialized_at_launch": candidate.plan_materialized_at_launch,
            }
        )
    return {
        "manifest": str(manifest.path),
        "manifest_sha256": sha256_file(manifest.path),
        "launcher": {
            "path": str(manifest.launcher),
            "sha256": actual_launcher_sha256,
        },
        "source_artifacts": artifacts,
        "oscar_contract": oscar_contract,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "dsv4_flash_candidate_campaign_v1.json",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/dsv4-candidate-campaign")
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    actions.add_argument("--execute-next", action="store_true")
    actions.add_argument("--execute-candidate", metavar="ID")
    parser.add_argument("--stage", choices=("screen", "confirm"))
    arguments = parser.parse_args()
    try:
        manifest = load_manifest(arguments.manifest)
        preflight = preflight_manifest(manifest)
        summary = summarize_campaign(manifest, arguments.work_dir)
        if arguments.execute_candidate is not None:
            if arguments.stage is None:
                raise CampaignError("--execute-candidate requires --stage")
            candidate = select_explicit_candidate_stage(
                manifest,
                arguments.work_dir,
                arguments.execute_candidate,
                arguments.stage,
            )
            execute_candidate_stage(
                manifest, candidate, arguments.stage, arguments.work_dir
            )
            summary = summarize_campaign(manifest, arguments.work_dir)
        elif arguments.stage is not None:
            raise CampaignError("--stage requires --execute-candidate")
        elif arguments.execute_next:
            action = summary.get("next_action")
            if not isinstance(action, dict):
                raise CampaignError("campaign has no eligible next action")
            candidate_id = cast(dict[str, object], action).get("candidate")
            stage = cast(dict[str, object], action).get("stage")
            all_candidates = {
                item.identifier: item
                for item in (manifest.baseline, *manifest.candidates)
            }
            if not isinstance(candidate_id, str) or candidate_id not in all_candidates:
                raise CampaignError("summary selected an unknown candidate")
            if not isinstance(stage, str):
                raise CampaignError("summary selected an invalid stage")
            execute_candidate_stage(
                manifest,
                all_candidates[candidate_id],
                stage,
                arguments.work_dir,
            )
            summary = summarize_campaign(manifest, arguments.work_dir)
        output = {"preflight": preflight, "summary": summary}
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except (CampaignError, OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "error_type": type(error).__name__,
                    "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
