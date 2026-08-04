#!/usr/bin/env python3
"""Stage an exact EP2 GPU union as a target-only PP2/EP1 expert plan.

A PP2 stage is TP1/EP1, so it must own the disjoint union of both source EP
ranks' GPU experts and offload the exact complement to its socket-local CPU
pool. The source may be either a uniform v1 plan or a padded/count-based
variable-width v2 plan. This helper validates it as a disjoint exact cover,
constructs the per-layer union, and publishes a content-addressed plan. The
command-line path additionally requires the qualified final EP confirmation
receipt and binds its digest into the transferred plan. Its sole stdout line is
the staged plan path suitable for
``DSV4_PP_HYBRID_EXPERT_SHARD_PLAN``.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import torch

PLAN_FORMAT: Final = "sglang_kt_hybrid_expert_shard_v1"
VARIABLE_PLAN_FORMAT: Final = "sglang_kt_hybrid_expert_shard_v2_variable"
TRANSFER_FORMAT: Final = "dsv4_ep2_winner_to_pp2_ep1_transfer_v5"
EP_CONFIRMATION_FORMAT: Final = "dsv4_candidate_campaign_result_v1"
EP_HOTSPOT_RECEIPT_VERSION: Final = 2
OSCAR_SPLIT_HISTORY_EXECUTION: Final = (
    "sm86-oscar-int2-split-history-fp32-online-v1"
)
OSCAR_SPLIT_HISTORY_SPLIT_MAP: Final = {
    "1": 16,
    "2": 16,
    "3": 8,
    "4": 4,
    "5": 4,
    "6": 4,
    "7": 4,
    "8": 2,
}
OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES: Final = 4_210_688
EP_HOTSPOT_PHASES: Final = frozenset(
    {
        "cold_first_exact",
        "radix_hot_exact",
        "radix_hot_near",
        "warm_no_radix_exact",
        "warm_no_radix_near",
    }
)
EP_FRESH_TTFT_PHASES: Final = frozenset(
    {"cold_first_exact", "warm_no_radix_exact", "warm_no_radix_near"}
)
QUALIFIED_NATIVE_ARTIFACT: Final = Path(
    "/var/lib/exo/experiments/dsv4-cpu-inline-scale-lut-n128-v1/"
    "lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so"
)
QUALIFIED_NATIVE_ARTIFACT_SHA256: Final = (
    "7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043"
)
QUALIFIED_LUT_HASH: Final = "06d1a83dbf20f545"
QUALIFIED_N_BLOCK: Final = 128
SOURCE_EP_SIZE: Final = 2
TARGET_EP_SIZE: Final = 1
NUM_LAYERS: Final = 43
NUM_EXPERTS: Final = 256
DEFAULT_CACHE_ROOT: Final = Path(
    "/var/lib/exo/cache/dsv4-flash-hybrid-pp2-opencode/transferred-ep2-winner-plans"
)
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)


class PlanTransferError(RuntimeError):
    """The source or staged placement failed closed validation."""


@dataclass(frozen=True, slots=True)
class SourcePlacement:
    plan_format: str
    gpu_masks: torch.Tensor
    cpu_expert_ids_padded_by_rank: torch.Tensor
    cpu_rank_counts_by_layer: torch.Tensor
    gpu_rank_counts_by_layer: torch.Tensor
    semantics_sha256: str


@dataclass(frozen=True, slots=True)
class TargetPlacement:
    plan_format: str
    gpu_masks: torch.Tensor
    cpu_expert_ids_padded_by_rank: torch.Tensor
    cpu_rank_counts_by_layer: torch.Tensor
    gpu_rank_counts_by_layer: torch.Tensor
    semantics_sha256: str


@dataclass(frozen=True, slots=True)
class EPConfirmation:
    receipt_path: Path
    receipt_sha256: str
    source_plan_path: Path
    source_plan_sha256: str
    source_placement_semantics_sha256: str
    coherency_receipt_path: Path | None = None
    coherency_receipt_sha256: str | None = None


class _ParsedArguments(Protocol):
    source_ep2_plan: Path
    source_ep_confirmation_receipt: Path
    source_ep_coherency_receipt: Path | None
    cache_root: Path
    expected_target_gpu_experts_per_layer: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PlanTransferError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _placement_semantics_sha256(
    gpu_masks: torch.Tensor,
    cpu_expert_ids_by_rank: Sequence[torch.Tensor],
) -> str:
    document = {
        "gpu_experts_mask_by_rank": gpu_masks.to(torch.uint8).tolist(),
        "cpu_expert_ids_by_rank": [
            expert_ids.to(torch.int64).tolist() for expert_ids in cpu_expert_ids_by_rank
        ],
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _variable_placement_semantics_sha256(
    gpu_masks: torch.Tensor,
    cpu_expert_ids_padded_by_rank: torch.Tensor,
    cpu_rank_counts_by_layer: torch.Tensor,
) -> str:
    document = {
        "gpu_experts_mask_by_rank": gpu_masks.to(torch.uint8).tolist(),
        "cpu_expert_ids_padded_by_rank": (
            cpu_expert_ids_padded_by_rank.to(torch.int64).tolist()
        ),
        "cpu_rank_counts_by_layer": cpu_rank_counts_by_layer.to(torch.int64).tolist(),
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest()


def _resolve_source_file(path: Path) -> Path:
    if not path.is_absolute():
        raise PlanTransferError(f"source EP2 plan path must be absolute: {path}")
    if path.is_symlink():
        raise PlanTransferError(f"source EP2 plan must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PlanTransferError(
            f"cannot resolve source EP2 plan {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise PlanTransferError(f"source EP2 plan is not a regular file: {resolved}")
    return resolved


def _resolve_confirmation_file(path: Path) -> Path:
    if not path.is_absolute():
        raise PlanTransferError(
            f"final EP confirmation receipt path must be absolute: {path}"
        )
    if path.is_symlink():
        raise PlanTransferError(
            f"final EP confirmation receipt must not be a symlink: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PlanTransferError(
            f"cannot resolve final EP confirmation receipt {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise PlanTransferError(
            f"final EP confirmation receipt is not a regular file: {resolved}"
        )
    return resolved


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PlanTransferError(f"final EP confirmation {label} is malformed")
    return cast(dict[str, object], value)


def _load_confirmation_json(path: Path, *, label: str) -> tuple[Path, str, dict[str, object]]:
    resolved = _resolve_confirmation_file(path)
    digest = sha256_file(resolved)
    try:
        raw = cast(object, json.loads(resolved.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanTransferError(f"{label} is not readable JSON") from error
    receipt = _require_mapping(raw, label=label)
    return resolved, digest, receipt


def _validate_direct_coherency_receipt(path: Path) -> tuple[Path, str]:
    resolved, digest, receipt = _load_confirmation_json(
        path, label="final EP coherency receipt"
    )
    forced_tool_call = _require_mapping(
        receipt.get("forced_tool_call"), label="final EP forced-tool proof"
    )
    semantic_runs = receipt.get("semantic_runs")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("cache_policy") != "flush_before_every_request"
        or receipt.get("coherent") is not True
        or receipt.get("deterministic_final_content") is not True
        or forced_tool_call.get("accepted") is not True
        or forced_tool_call.get("http_status") != 200
        or forced_tool_call.get("finish_reason") != "tool_calls"
        or forced_tool_call.get("saw_done") is not True
        or forced_tool_call.get("issue_codes") != []
        or not isinstance(semantic_runs, list)
        or len(semantic_runs) < 3
        or any(
            not isinstance(run, dict)
            or run.get("accepted") is not True
            or run.get("http_status") != 200
            or run.get("finish_reason") != "stop"
            or run.get("saw_done") is not True
            or run.get("issue_codes") != []
            or run.get("content_sha256") != run.get("expected_content_sha256")
            for run in semantic_runs
        )
    ):
        raise PlanTransferError(
            "final EP coherency receipt lacks deterministic text/tool-call proof"
        )
    if sha256_file(resolved) != digest:
        raise PlanTransferError("final EP coherency receipt changed while validating")
    return resolved, digest


def _validate_ep_nvlink_traffic(value: object) -> None:
    traffic = _require_mapping(value, label="final EP NVLink traffic proof")
    deltas = _require_mapping(
        traffic.get("counter_deltas"), label="final EP NVLink counter deltas"
    )
    expected_keys = {
        f"gpu{gpu}.link{link}.{direction}_kib"
        for gpu in range(2)
        for link in range(4)
        for direction in ("rx", "tx")
    }
    if set(deltas) != expected_keys or any(
        type(value) is not int or cast(int, value) <= 0 for value in deltas.values()
    ):
        raise PlanTransferError(
            "final EP receipt must prove positive traffic on all 16 NVLink counters"
        )
    for link in range(4):
        if (
            deltas[f"gpu0.link{link}.tx_kib"]
            != deltas[f"gpu1.link{link}.rx_kib"]
            or deltas[f"gpu1.link{link}.tx_kib"]
            != deltas[f"gpu0.link{link}.rx_kib"]
        ):
            raise PlanTransferError(
                "final EP NVLink counter deltas are not reciprocal by link"
            )


def _validate_ep_hotspot_phases(value: object) -> None:
    phases = _require_mapping(value, label="final EP hotspot phases")
    if set(phases) != EP_HOTSPOT_PHASES:
        raise PlanTransferError("final EP hotspot receipt must contain all five phases")
    for name, raw_phase in phases.items():
        phase = _require_mapping(raw_phase, label=f"final EP hotspot phase {name}")
        benchmark = _require_mapping(
            phase.get("benchmark"), label=f"final EP hotspot phase {name} benchmark"
        )
        semantic = _require_mapping(
            benchmark.get("semantic_validation"),
            label=f"final EP hotspot phase {name} semantic proof",
        )
        if (
            benchmark.get("accepted") is not True
            or benchmark.get("performance_claim_eligible") is not True
            or benchmark.get("http_status") != 200
            or benchmark.get("finish_reason") != "stop"
            or benchmark.get("saw_done") is not True
            or benchmark.get("input_tokens") != 2_694
            or semantic.get("enforced") is not True
            or semantic.get("passed") is not True
            or semantic.get("issue_codes") != []
        ):
            raise PlanTransferError(
                f"final EP hotspot phase {name} is not semantically/performance valid"
            )
        if name in EP_FRESH_TTFT_PHASES:
            ttft = benchmark.get("time_to_first_token_seconds")
            if (
                not isinstance(ttft, (int, float))
                or isinstance(ttft, bool)
                or not 0 < float(ttft) <= 7.0
                or phase.get("radix_flush_before") is not True
                or benchmark.get("server_cached_tokens") != 0
            ):
                raise PlanTransferError(
                    f"final EP hotspot phase {name} does not prove fresh TTFT <= 7s"
                )


def _load_direct_ep_confirmation(
    *,
    resolved_receipt: Path,
    receipt_sha256: str,
    receipt: dict[str, object],
    coherency_receipt: Path | None,
    expected_source_plan: Path | None,
) -> EPConfirmation:
    if coherency_receipt is None:
        raise PlanTransferError(
            "direct final EP hotspot receipt requires its separate coherency receipt"
        )
    resolved_coherency, coherency_sha256 = _validate_direct_coherency_receipt(
        coherency_receipt
    )
    if (
        receipt.get("receipt_version") != EP_HOTSPOT_RECEIPT_VERSION
        or receipt.get("accepted") is not True
        or receipt.get("measurement_mode") != "trace"
    ):
        raise PlanTransferError(
            "final EP hotspot receipt must be an accepted trace-mode v2 receipt"
        )
    _validate_ep_hotspot_phases(receipt.get("phases"))
    _validate_ep_nvlink_traffic(receipt.get("nvlink_traffic"))

    contract = _require_mapping(
        receipt.get("server_contract"), label="final EP server contract"
    )
    exact_contract = {
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 2,
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "kv_cache_dtype": "fp8_e4m3",
        "kt_cpuinfer": 56,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "disable_prefill_cuda_graph": False,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "breakable",
        "enable_p2p_check": True,
        "pre_warm_nccl": True,
        "dsv4_oscar_int2_split_history": True,
        "dsv4_oscar_int2_split_history_execution": (
            OSCAR_SPLIT_HISTORY_EXECUTION
        ),
        "dsv4_oscar_int2_split_history_split_map": (
            OSCAR_SPLIT_HISTORY_SPLIT_MAP
        ),
        "dsv4_oscar_int2_split_history_workspace_bytes": (
            OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
        ),
        "dsv4_oscar_int2_split_history_max_partial_rows": 32,
        "dsv4_oscar_int2_split_history_sink_owner": "stage2-exactly-once",
        "dsv4_oscar_int2_split_history_prefill_enabled": False,
        "dsv4_oscar_int2_split_history_fixed_address": True,
    }
    if any(contract.get(key) != expected for key, expected in exact_contract.items()):
        raise PlanTransferError(
            "final EP hotspot receipt violates the 524K graph-safe Oscar contract"
        )
    workers = _require_mapping(
        contract.get("dsv4_oscar_int2_split_history_workers"),
        label="final EP split-history workers",
    )
    worker_pids = workers.get("worker_pids")
    workspace_addresses = workers.get("workspace_addresses")
    if (
        workers.get("worker_count") != 2
        or workers.get("tp_pp_gpu_ranks") != [[0, 0, 0], [1, 0, 1]]
        or workers.get("workspace_bytes_per_worker")
        != OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
        or workers.get("fixed_address") is not True
        or not isinstance(worker_pids, list)
        or len(worker_pids) != 2
        or any(type(pid) is not int or pid <= 0 for pid in worker_pids)
        or len(set(cast(list[int], worker_pids))) != 2
        or not isinstance(workspace_addresses, list)
        or len(workspace_addresses) != 2
        or any(
            type(address) is not int or address <= 0
            for address in workspace_addresses
        )
    ):
        raise PlanTransferError(
            "final EP hotspot receipt does not prove both live split workspaces"
        )

    provenance = _require_mapping(
        receipt.get("expert_plan_provenance"), label="final EP plan provenance"
    )
    raw_plan_path = provenance.get("expected_plan_path")
    raw_plan_sha256 = provenance.get("expected_plan_sha256")
    if (
        provenance.get("binding") != "launcher-hash-and-kt-loader-validated"
        or not isinstance(raw_plan_path, str)
        or not raw_plan_path
        or not isinstance(raw_plan_sha256, str)
        or SHA256_PATTERN.fullmatch(raw_plan_sha256) is None
    ):
        raise PlanTransferError("final EP hotspot plan provenance is malformed")
    resolved_plan = _resolve_source_file(Path(raw_plan_path))
    if expected_source_plan is not None:
        resolved_expected = _resolve_source_file(expected_source_plan)
        if resolved_plan != resolved_expected:
            raise PlanTransferError(
                "source EP2 plan does not match the final hotspot receipt"
            )
    if sha256_file(resolved_plan) != raw_plan_sha256:
        raise PlanTransferError(
            "source EP2 plan digest does not match the final hotspot receipt"
        )
    try:
        raw_plan = torch.load(resolved_plan, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PlanTransferError("cannot inspect final EP hotspot plan") from error
    if not isinstance(raw_plan, dict):
        raise PlanTransferError("final EP hotspot plan is not a dictionary")
    raw_semantics = raw_plan.get("placement_semantics_sha256")
    if not isinstance(raw_semantics, str) or SHA256_PATTERN.fullmatch(raw_semantics) is None:
        raise PlanTransferError(
            "final EP hotspot plan lacks placement semantics provenance"
        )
    if (
        sha256_file(resolved_receipt) != receipt_sha256
        or sha256_file(resolved_coherency) != coherency_sha256
    ):
        raise PlanTransferError("final EP proof changed while validating")
    return EPConfirmation(
        receipt_path=resolved_receipt,
        receipt_sha256=receipt_sha256,
        source_plan_path=resolved_plan,
        source_plan_sha256=raw_plan_sha256,
        source_placement_semantics_sha256=raw_semantics,
        coherency_receipt_path=resolved_coherency,
        coherency_receipt_sha256=coherency_sha256,
    )


def load_ep_confirmation(
    path: Path,
    *,
    coherency_receipt: Path | None = None,
    expected_source_plan: Path | None = None,
) -> EPConfirmation:
    """Validate the exact qualified EP winner contract used by PP transfer."""

    resolved_receipt, receipt_sha256, receipt = _load_confirmation_json(
        path, label="final EP confirmation receipt"
    )
    if receipt.get("receipt_version") == EP_HOTSPOT_RECEIPT_VERSION:
        return _load_direct_ep_confirmation(
            resolved_receipt=resolved_receipt,
            receipt_sha256=receipt_sha256,
            receipt=receipt,
            coherency_receipt=coherency_receipt,
            expected_source_plan=expected_source_plan,
        )
    if coherency_receipt is not None:
        raise PlanTransferError(
            "separate coherency receipt is valid only with a direct hotspot receipt"
        )
    if (
        receipt.get("format") != EP_CONFIRMATION_FORMAT
        or receipt.get("stage") != "confirm"
        or receipt.get("qualified") is not True
        or receipt.get("shutdown_method") != "sigterm"
        or receipt.get("residual_compute_pids") != []
    ):
        raise PlanTransferError(
            "final EP confirmation must be a qualified, clean confirm-stage receipt"
        )

    coherency = _require_mapping(receipt.get("coherency"), label="coherency proof")
    forced_tool_call = _require_mapping(
        coherency.get("forced_tool_call"), label="forced-tool proof"
    )
    semantic_runs = coherency.get("semantic_runs")
    if (
        coherency.get("coherent") is not True
        or coherency.get("deterministic_final_content") is not True
        or forced_tool_call.get("accepted") is not True
        or not isinstance(semantic_runs, list)
        or len(semantic_runs) < 2
        or any(
            not isinstance(run, dict) or run.get("accepted") is not True
            for run in semantic_runs
        )
    ):
        raise PlanTransferError(
            "final EP confirmation lacks coherent text and tool-call proofs"
        )

    environment = _require_mapping(receipt.get("environment"), label="environment")
    exact_environment = {
        "DSV4_TENSOR_PARALLEL_SIZE": "2",
        "DSV4_PIPELINE_PARALLEL_SIZE": "1",
        "DSV4_EXPERT_PARALLEL_SIZE": "2",
        "DSV4_CONTEXT_LENGTH": "524288",
        "DSV4_MAX_TOTAL_TOKENS": "524288",
        "DSV4_KV_CACHE_DTYPE": "fp8_e4m3",
        "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE": "1",
        "SGLANG_DSV4_INT4_KV_STORAGE": "0",
        "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE": "0",
        "SGLANG_DSV4_SM86_C128_BF16_STORAGE": "0",
        "DSV4_CPUINFER_THREADS": "56",
        "KT_WORKER_SPIN_US": "1000",
        "KT_TASK_QUEUE_PIN_FIRST_CORE": "1",
        "KT_SINGLE_NUMA_INLINE_DISPATCH": "1",
        "KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1",
        "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
    }
    if any(environment.get(key) != value for key, value in exact_environment.items()):
        raise PlanTransferError(
            "final EP confirmation does not contain the qualified Oscar/CPU tuple"
        )
    if environment.get("DSV4_KT_CPU_OPTIMIZED_CANDIDATE") != str(
        QUALIFIED_NATIVE_ARTIFACT
    ):
        raise PlanTransferError(
            "final EP confirmation does not name the qualified native artifact"
        )

    inline_proof = _require_mapping(
        receipt.get("kt_single_numa_inline_dispatch_server_proof"),
        label="inline-dispatch proof",
    )
    scale_proof = _require_mapping(
        receipt.get("kt_mxfp4_avx_scale_fold_server_proof"),
        label="scale-fold proof",
    )
    if (
        inline_proof.get("kt_single_numa_inline_dispatch_all_workers_active")
        is not True
        or inline_proof.get("kt_single_numa_inline_dispatch_rank_coverage_valid")
        is not True
        or scale_proof.get("kt_mxfp4_avx_scale_fold_all_workers_active") is not True
        or scale_proof.get("kt_mxfp4_avx_scale_fold_rank_coverage_valid") is not True
        or scale_proof.get("kt_mxfp4_avx_scale_fold_expected_n_block")
        != QUALIFIED_N_BLOCK
        or scale_proof.get("kt_mxfp4_avx_scale_fold_requested_mode") != "lut-v1"
    ):
        raise PlanTransferError(
            "final EP confirmation lacks the qualified inline/N128 LUT proof"
        )
    scale_workers = scale_proof.get("workers")
    if (
        not isinstance(scale_workers, list)
        or len(scale_workers) != 2
        or any(
            not isinstance(worker, dict)
            or not isinstance(worker.get("telemetry"), dict)
            or worker["telemetry"].get("n_block") != QUALIFIED_N_BLOCK
            or worker["telemetry"].get("lut_hash") != QUALIFIED_LUT_HASH
            for worker in scale_workers
        )
    ):
        raise PlanTransferError(
            "final EP confirmation scale-fold workers do not prove the N128 LUT"
        )

    oscar_contract = _require_mapping(
        receipt.get("oscar_contract"), label="Oscar contract"
    )
    expected_server_info = _require_mapping(
        oscar_contract.get("expected_server_info"), label="Oscar server proof"
    )
    exact_oscar = {
        "dsv4_oscar_int2_kv_storage": True,
        "dsv4_kv_storage_mode": (
            "oscar_int2_asymmetric+protected_swa_bfloat16"
        ),
        "dsv4_c4_kv_bytes_per_token": 272,
        "dsv4_c128_kv_bytes_per_token": 272,
        "dsv4_c4_indexer_bytes_per_token": 40,
        "dsv4_int4_kv_storage": False,
        "dsv4_int4_c4_indexer_storage": False,
        "dsv4_sm86_c128_bf16_storage": False,
    }
    if any(expected_server_info.get(key) != value for key, value in exact_oscar.items()):
        raise PlanTransferError(
            "final EP confirmation does not prove physical Oscar INT2 storage"
        )

    plan = _require_mapping(receipt.get("plan"), label="plan provenance")
    raw_plan_path = plan.get("path")
    raw_plan_sha256 = plan.get("sha256")
    raw_semantics_sha256 = plan.get("placement_semantics_sha256")
    if not isinstance(raw_plan_path, str) or not raw_plan_path:
        raise PlanTransferError("final EP confirmation plan path is malformed")
    if not isinstance(raw_plan_sha256, str) or not SHA256_PATTERN.fullmatch(
        raw_plan_sha256
    ):
        raise PlanTransferError("final EP confirmation plan digest is malformed")
    if not isinstance(raw_semantics_sha256, str) or not SHA256_PATTERN.fullmatch(
        raw_semantics_sha256
    ):
        raise PlanTransferError(
            "final EP confirmation plan semantics digest is malformed"
        )
    resolved_plan = _resolve_source_file(Path(raw_plan_path))
    if expected_source_plan is not None:
        resolved_expected = _resolve_source_file(expected_source_plan)
        if resolved_plan != resolved_expected:
            raise PlanTransferError(
                "source EP2 plan does not match the final EP confirmation"
            )
    if sha256_file(resolved_plan) != raw_plan_sha256:
        raise PlanTransferError(
            "source EP2 plan digest does not match the final EP confirmation"
        )
    if sha256_file(resolved_receipt) != receipt_sha256:
        raise PlanTransferError("final EP confirmation changed while validating")
    return EPConfirmation(
        receipt_path=resolved_receipt,
        receipt_sha256=receipt_sha256,
        source_plan_path=resolved_plan,
        source_plan_sha256=raw_plan_sha256,
        source_placement_semantics_sha256=raw_semantics_sha256,
    )


def _prepare_cache_root(path: Path, source_plan: Path) -> Path:
    if not path.is_absolute():
        raise PlanTransferError(f"cache root path must be absolute: {path}")
    if path.is_symlink():
        raise PlanTransferError(f"cache root must not be a symlink: {path}")
    try:
        prospective = path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise PlanTransferError(f"cannot resolve cache root {path}: {error}") from error
    if prospective == source_plan:
        raise PlanTransferError("cache root must not be the source EP2 plan")
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PlanTransferError(f"cannot prepare cache root {path}: {error}") from error
    if not resolved.is_dir():
        raise PlanTransferError(f"cache root is not a directory: {resolved}")
    return resolved


def _load_plan(path: Path, description: str) -> dict[str, object]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PlanTransferError(f"cannot load {description} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise PlanTransferError(f"{description} must contain a dictionary: {path}")
    return cast(dict[str, object], loaded)


def _boolean_gpu_masks(
    value: object, *, expected_shape: tuple[int, ...]
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PlanTransferError("source plan GPU masks must be a tensor")
    if value.dtype == torch.bool:
        masks = value.to(device="cpu").contiguous()
    else:
        if value.is_floating_point() or value.is_complex():
            raise PlanTransferError(
                "source plan GPU masks must use boolean/integer values"
            )
        integer_masks = value.to(device="cpu", dtype=torch.int64).contiguous()
        if integer_masks.numel() and not bool(
            torch.all((integer_masks == 0) | (integer_masks == 1))
        ):
            raise PlanTransferError(
                "source plan GPU masks must contain only zero or one"
            )
        masks = integer_masks.to(torch.bool)
    if tuple(masks.shape) != expected_shape:
        raise PlanTransferError(
            f"source plan GPU mask shape must be {expected_shape}, got {tuple(masks.shape)}"
        )
    return masks


def _integer_cpu_shard(
    value: object,
    *,
    rank: int,
    expected_experts_per_layer: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PlanTransferError(f"source plan CPU shard rank {rank} must be a tensor")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise PlanTransferError(
            f"source plan CPU shard rank {rank} must use an integer dtype"
        )
    shard = value.to(device="cpu", dtype=torch.int64).contiguous()
    expected_shape = (NUM_LAYERS, expected_experts_per_layer)
    if tuple(shard.shape) != expected_shape:
        raise PlanTransferError(
            f"source plan CPU shard rank {rank} must have shape {expected_shape}, "
            f"got {tuple(shard.shape)}"
        )
    if shard.numel() and (int(shard.min()) < 0 or int(shard.max()) >= NUM_EXPERTS):
        raise PlanTransferError(
            f"source plan CPU shard rank {rank} contains an expert outside "
            f"[0, {NUM_EXPERTS})"
        )
    return shard


def _integer_tensor(
    value: object,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PlanTransferError(f"source plan {label} must be a tensor")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise PlanTransferError(f"source plan {label} must use an integer dtype")
    tensor = value.to(device="cpu", dtype=torch.int64).contiguous()
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise PlanTransferError(
            f"source plan {label} must have shape {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _validate_optional_integer_scalar(
    loaded: dict[str, object],
    *,
    key: str,
    expected: int,
) -> None:
    value = loaded.get(key)
    if value is None:
        return
    tensor = _integer_tensor(value, label=key, expected_shape=())
    observed = int(tensor.item())
    if observed != expected:
        raise PlanTransferError(f"source plan {key} must be {expected}, got {observed}")


def _validate_optional_rank_counts(
    loaded: dict[str, object],
    *,
    key: str,
    expected: tuple[int, int],
) -> None:
    value = loaded.get(key)
    if value is None:
        return
    tensor = _integer_tensor(
        value,
        label=key,
        expected_shape=(SOURCE_EP_SIZE,),
    )
    observed = cast(tuple[int, int], tuple(tensor.tolist()))
    if observed != expected:
        raise PlanTransferError(f"source plan {key} must be {expected}, got {observed}")


def _uniform_source_gpu_width(gpu_masks: torch.Tensor) -> int:
    gpu_counts = gpu_masks.sum(dim=2, dtype=torch.int64)
    width = int(gpu_counts[0, 0])
    if width <= 0 or not bool(torch.all(gpu_counts == width)):
        raise PlanTransferError(
            "source EP2 plan must assign one positive, uniform GPU expert count "
            "to both ranks in every layer"
        )
    if SOURCE_EP_SIZE * width > NUM_EXPERTS:
        raise PlanTransferError("source EP2 GPU expert union cannot exceed 256")
    return width


def _load_v1_cpu_placement(
    loaded: dict[str, object],
    *,
    gpu_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if (
        loaded.get("cpu_expert_ids_padded_by_rank") is not None
        or loaded.get("cpu_rank_counts_by_layer") is not None
        or loaded.get("gpu_rank_counts_by_layer") is not None
    ):
        raise PlanTransferError(
            "source v1 plan cannot contain variable-width v2 shard fields"
        )
    source_gpu_width = _uniform_source_gpu_width(gpu_masks)
    target_gpu_width = SOURCE_EP_SIZE * source_gpu_width
    source_cpu_width = (NUM_EXPERTS - target_gpu_width) // SOURCE_EP_SIZE
    raw_cpu_shards = loaded.get("cpu_expert_ids_by_rank")
    if not isinstance(raw_cpu_shards, (list, tuple)) or len(raw_cpu_shards) != 2:
        raise PlanTransferError("source EP2 plan must contain exactly two CPU shards")
    cpu_shards = (
        _integer_cpu_shard(
            raw_cpu_shards[0],
            rank=0,
            expected_experts_per_layer=source_cpu_width,
        ),
        _integer_cpu_shard(
            raw_cpu_shards[1],
            rank=1,
            expected_experts_per_layer=source_cpu_width,
        ),
    )
    _validate_optional_rank_counts(
        loaded,
        key="gpu_rank_counts",
        expected=(source_gpu_width, source_gpu_width),
    )
    _validate_optional_rank_counts(
        loaded,
        key="cpu_rank_counts",
        expected=(source_cpu_width, source_cpu_width),
    )
    _validate_optional_integer_scalar(
        loaded,
        key="gpu_union_expert_count",
        expected=target_gpu_width,
    )
    gpu_counts = gpu_masks.sum(dim=2, dtype=torch.int64)
    cpu_counts = torch.full(
        (SOURCE_EP_SIZE, NUM_LAYERS),
        source_cpu_width,
        dtype=torch.int64,
    )
    cpu_padded = torch.stack(cpu_shards, dim=0)
    semantics_sha256 = _placement_semantics_sha256(gpu_masks, cpu_shards)
    return cpu_padded, cpu_counts, gpu_counts, semantics_sha256


def _load_v2_cpu_placement(
    loaded: dict[str, object],
    *,
    gpu_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    if loaded.get("cpu_expert_ids_by_rank") is not None:
        raise PlanTransferError(
            "source variable-width v2 plan cannot contain v1 CPU shard fields"
        )
    cpu_padded = _integer_tensor(
        loaded.get("cpu_expert_ids_padded_by_rank"),
        label="cpu_expert_ids_padded_by_rank",
    )
    if cpu_padded.ndim != 3 or tuple(cpu_padded.shape[:2]) != (
        SOURCE_EP_SIZE,
        NUM_LAYERS,
    ):
        raise PlanTransferError(
            "source plan cpu_expert_ids_padded_by_rank must have shape "
            f"[{SOURCE_EP_SIZE}, {NUM_LAYERS}, max_cpu_experts]"
        )
    if cpu_padded.shape[2] > NUM_EXPERTS:
        raise PlanTransferError("source plan padded CPU shard width cannot exceed 256")
    cpu_counts = _integer_tensor(
        loaded.get("cpu_rank_counts_by_layer"),
        label="cpu_rank_counts_by_layer",
        expected_shape=(SOURCE_EP_SIZE, NUM_LAYERS),
    )
    if cpu_counts.numel() and (
        int(cpu_counts.min()) < 0 or int(cpu_counts.max()) > cpu_padded.shape[2]
    ):
        raise PlanTransferError(
            "source plan CPU counts exceed padded CPU shard storage"
        )
    for rank in range(SOURCE_EP_SIZE):
        for layer in range(NUM_LAYERS):
            count = int(cpu_counts[rank, layer])
            active = cpu_padded[rank, layer, :count]
            if active.numel() and (
                int(active.min()) < 0 or int(active.max()) >= NUM_EXPERTS
            ):
                raise PlanTransferError(
                    "source plan variable CPU shard contains an expert outside "
                    f"[0, {NUM_EXPERTS}) at rank {rank}, layer {layer}"
                )
            if bool(torch.any(cpu_padded[rank, layer, count:] != -1)):
                raise PlanTransferError(
                    "source plan variable CPU shard padding must be -1 at "
                    f"rank {rank}, layer {layer}"
                )

    gpu_counts = _integer_tensor(
        loaded.get("gpu_rank_counts_by_layer"),
        label="gpu_rank_counts_by_layer",
        expected_shape=(SOURCE_EP_SIZE, NUM_LAYERS),
    )
    actual_gpu_counts = gpu_masks.sum(dim=2, dtype=torch.int64)
    if not torch.equal(gpu_counts, actual_gpu_counts):
        raise PlanTransferError(
            "source plan GPU counts by layer do not match its masks"
        )
    _validate_optional_integer_scalar(
        loaded,
        key="min_gpu_experts_per_rank_per_layer",
        expected=int(gpu_counts.min()),
    )
    _validate_optional_integer_scalar(
        loaded,
        key="max_gpu_experts_per_rank_per_layer",
        expected=int(gpu_counts.max()),
    )
    rank_symmetric = loaded.get("rank_symmetric_widths")
    if rank_symmetric is not None:
        if not isinstance(rank_symmetric, bool):
            raise PlanTransferError("source plan rank_symmetric_widths must be boolean")
        observed_rank_symmetric = bool(torch.all(gpu_counts[0] == gpu_counts[1]))
        if rank_symmetric and not observed_rank_symmetric:
            raise PlanTransferError(
                "source plan rank_symmetric_widths does not match its GPU counts"
            )
    semantics_sha256 = _variable_placement_semantics_sha256(
        gpu_masks,
        cpu_padded,
        cpu_counts,
    )
    return cpu_padded, cpu_counts, gpu_counts, semantics_sha256


def _validate_exact_source_cover(
    *,
    gpu_masks: torch.Tensor,
    cpu_expert_ids_padded_by_rank: torch.Tensor,
    cpu_rank_counts_by_layer: torch.Tensor,
) -> None:
    assignment_counts = gpu_masks.sum(dim=0, dtype=torch.int64)
    for rank in range(SOURCE_EP_SIZE):
        for layer in range(NUM_LAYERS):
            count = int(cpu_rank_counts_by_layer[rank, layer])
            expert_ids = cpu_expert_ids_padded_by_rank[rank, layer, :count]
            assignment_counts[layer].scatter_add_(
                0,
                expert_ids,
                torch.ones_like(expert_ids, dtype=torch.int64),
            )
    if not bool(torch.all(assignment_counts == 1)):
        raise PlanTransferError(
            "source EP2 GPU/CPU placement must be a disjoint exact cover per layer"
        )


def load_source_placement(
    path: Path,
    *,
    expected_target_gpu_experts_per_layer: int | None = None,
) -> SourcePlacement:
    loaded = _load_plan(path, "source EP2 plan")
    source_format = loaded.get("format")
    if not isinstance(source_format, str) or source_format not in (
        PLAN_FORMAT,
        VARIABLE_PLAN_FORMAT,
    ):
        raise PlanTransferError(
            "source EP2 plan format must be "
            f"{PLAN_FORMAT!r} or {VARIABLE_PLAN_FORMAT!r}"
        )
    gpu_masks = _boolean_gpu_masks(
        loaded.get("gpu_experts_mask_by_rank"),
        expected_shape=(SOURCE_EP_SIZE, NUM_LAYERS, NUM_EXPERTS),
    )
    _validate_optional_integer_scalar(
        loaded,
        key="global_num_experts",
        expected=NUM_EXPERTS,
    )
    if source_format == PLAN_FORMAT:
        cpu_padded, cpu_counts, gpu_counts, semantics_sha256 = _load_v1_cpu_placement(
            loaded, gpu_masks=gpu_masks
        )
    else:
        cpu_padded, cpu_counts, gpu_counts, semantics_sha256 = _load_v2_cpu_placement(
            loaded, gpu_masks=gpu_masks
        )
    _validate_exact_source_cover(
        gpu_masks=gpu_masks,
        cpu_expert_ids_padded_by_rank=cpu_padded,
        cpu_rank_counts_by_layer=cpu_counts,
    )
    target_gpu_counts = gpu_counts.sum(dim=0)
    if int(target_gpu_counts.min()) <= 0:
        raise PlanTransferError(
            "source EP2 plan must assign a positive GPU expert union in every layer"
        )
    target_gpu_ceiling = int(target_gpu_counts.max())
    if (
        expected_target_gpu_experts_per_layer is not None
        and target_gpu_ceiling != expected_target_gpu_experts_per_layer
    ):
        raise PlanTransferError(
            "source EP2 GPU union ceiling does not match "
            "--expected-target-gpu-experts-per-layer: "
            f"expected {expected_target_gpu_experts_per_layer}, "
            f"got {target_gpu_ceiling}"
        )
    declared_semantics = loaded.get("placement_semantics_sha256")
    if declared_semantics is not None and (
        not isinstance(declared_semantics, str)
        or SHA256_PATTERN.fullmatch(declared_semantics) is None
        or declared_semantics != semantics_sha256
    ):
        raise PlanTransferError(
            "source EP2 placement_semantics_sha256 does not match its tensors"
        )
    return SourcePlacement(
        plan_format=source_format,
        gpu_masks=gpu_masks,
        cpu_expert_ids_padded_by_rank=cpu_padded,
        cpu_rank_counts_by_layer=cpu_counts,
        gpu_rank_counts_by_layer=gpu_counts,
        semantics_sha256=semantics_sha256,
    )


def transfer_placement(source: SourcePlacement) -> TargetPlacement:
    gpu_union = source.gpu_masks.any(dim=0)
    target_gpu_counts = gpu_union.sum(dim=1, dtype=torch.int64)
    if not torch.equal(
        target_gpu_counts,
        source.gpu_rank_counts_by_layer.sum(dim=0),
    ):
        raise PlanTransferError(
            "source EP2 GPU ranks must form a disjoint expert union per layer"
        )
    target_cpu_counts = NUM_EXPERTS - target_gpu_counts
    maximum_cpu_count = int(target_cpu_counts.max())
    cpu_padded = torch.full(
        (TARGET_EP_SIZE, NUM_LAYERS, maximum_cpu_count),
        -1,
        dtype=torch.int64,
    )
    for layer in range(NUM_LAYERS):
        cpu_complement = torch.where(~gpu_union[layer])[0]
        cpu_padded[0, layer, : cpu_complement.numel()] = cpu_complement
    target_gpu_masks = gpu_union.unsqueeze(0).contiguous()
    target_gpu_counts_by_layer = target_gpu_counts.unsqueeze(0).contiguous()
    target_cpu_counts_by_layer = target_cpu_counts.unsqueeze(0).contiguous()
    target_format = (
        VARIABLE_PLAN_FORMAT
        if source.plan_format == VARIABLE_PLAN_FORMAT
        else PLAN_FORMAT
    )
    if target_format == VARIABLE_PLAN_FORMAT:
        semantics_sha256 = _variable_placement_semantics_sha256(
            target_gpu_masks,
            cpu_padded,
            target_cpu_counts_by_layer,
        )
    else:
        if not bool(torch.all(target_cpu_counts == target_cpu_counts[0])):
            raise PlanTransferError(
                "uniform source unexpectedly produced a variable-width target"
            )
        semantics_sha256 = _placement_semantics_sha256(
            target_gpu_masks,
            (cpu_padded[0],),
        )
    return TargetPlacement(
        plan_format=target_format,
        gpu_masks=target_gpu_masks,
        cpu_expert_ids_padded_by_rank=cpu_padded,
        cpu_rank_counts_by_layer=target_cpu_counts_by_layer,
        gpu_rank_counts_by_layer=target_gpu_counts_by_layer,
        semantics_sha256=semantics_sha256,
    )


def _identity_document(
    *,
    source_plan: Path,
    source_plan_sha256: str,
    source_placement: SourcePlacement,
    ep_confirmation: EPConfirmation | None,
) -> dict[str, object]:
    source_gpu_counts = source_placement.gpu_rank_counts_by_layer
    source_cpu_counts = source_placement.cpu_rank_counts_by_layer
    target_gpu_counts = source_gpu_counts.sum(dim=0).unsqueeze(0)
    target_cpu_counts = NUM_EXPERTS - target_gpu_counts
    source_gpu_is_uniform = bool(
        torch.all(source_gpu_counts == source_gpu_counts[0, 0])
    )
    source_cpu_is_uniform = bool(
        torch.all(source_cpu_counts == source_cpu_counts[0, 0])
    )
    target_gpu_is_uniform = bool(
        torch.all(target_gpu_counts == target_gpu_counts[0, 0])
    )
    target_cpu_is_uniform = bool(
        torch.all(target_cpu_counts == target_cpu_counts[0, 0])
    )
    identity: dict[str, object] = {
        "format": TRANSFER_FORMAT,
        "source_ep2_plan": str(source_plan),
        "source_ep2_plan_sha256": source_plan_sha256,
        "source_plan_format": source_placement.plan_format,
        "target_plan_format": source_placement.plan_format,
        "source_ep2_placement_semantics_sha256": source_placement.semantics_sha256,
        "transfer_strategy": "union_ep2_gpu_residency_then_cpu_complement",
        "target_model_scope": "target_only",
        "target_speculative_decoding": False,
        "target_tensor_parallel_size": 1,
        "target_pipeline_parallel_size": 2,
        "target_expert_parallel_size": TARGET_EP_SIZE,
        "source_ep_size": SOURCE_EP_SIZE,
        "source_gpu_experts_per_rank": (
            int(source_gpu_counts[0, 0]) if source_gpu_is_uniform else None
        ),
        "source_cpu_experts_per_rank": (
            int(source_cpu_counts[0, 0]) if source_cpu_is_uniform else None
        ),
        "source_gpu_rank_counts_by_layer": source_gpu_counts.tolist(),
        "source_cpu_rank_counts_by_layer": source_cpu_counts.tolist(),
        "target_ep_size": TARGET_EP_SIZE,
        "target_gpu_experts_per_layer": (
            int(target_gpu_counts[0, 0]) if target_gpu_is_uniform else None
        ),
        "target_cpu_experts_per_layer": (
            int(target_cpu_counts[0, 0]) if target_cpu_is_uniform else None
        ),
        "target_gpu_rank_counts_by_layer": target_gpu_counts.tolist(),
        "target_cpu_rank_counts_by_layer": target_cpu_counts.tolist(),
        "target_min_gpu_experts_per_layer": int(target_gpu_counts.min()),
        "target_max_gpu_experts_per_layer": int(target_gpu_counts.max()),
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
    }
    if ep_confirmation is not None:
        identity["source_ep_confirmation_receipt"] = str(
            ep_confirmation.receipt_path
        )
        identity["source_ep_confirmation_receipt_sha256"] = (
            ep_confirmation.receipt_sha256
        )
        if (
            ep_confirmation.coherency_receipt_path is not None
            and ep_confirmation.coherency_receipt_sha256 is not None
        ):
            identity["source_ep_coherency_receipt"] = str(
                ep_confirmation.coherency_receipt_path
            )
            identity["source_ep_coherency_receipt_sha256"] = (
                ep_confirmation.coherency_receipt_sha256
            )
        identity["qualified_native_artifact"] = str(QUALIFIED_NATIVE_ARTIFACT)
        identity["qualified_native_artifact_sha256"] = (
            QUALIFIED_NATIVE_ARTIFACT_SHA256
        )
        identity["qualified_cpuinfer_threads"] = 56
        identity["qualified_worker_spin_us"] = 1000
        identity["qualified_task_queue_pin_first_core"] = True
        identity["qualified_single_numa_inline_dispatch"] = True
        identity["qualified_scale_fold_mode"] = "lut-v1"
        identity["qualified_scale_fold_n_block"] = QUALIFIED_N_BLOCK
        identity["qualified_scale_fold_lut_hash"] = QUALIFIED_LUT_HASH
        identity["kv_cache_public_carrier"] = "fp8_e4m3"
        identity["physical_kv_cache_storage"] = "oscar-int2-asymmetric"
        identity["oscar_split_history"] = True
        identity["oscar_split_history_execution"] = (
            OSCAR_SPLIT_HISTORY_EXECUTION
        )
        identity["oscar_split_history_split_map"] = (
            OSCAR_SPLIT_HISTORY_SPLIT_MAP
        )
        identity["oscar_split_history_workspace_bytes_per_worker"] = (
            OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
        )
        identity["oscar_split_history_worker_identities_ep"] = [
            [0, 0, 0],
            [1, 0, 1],
        ]
        identity["oscar_split_history_worker_identities_pp"] = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
        ]
        identity["context_length"] = 524_288
        identity["cuda_graph_decode"] = "full"
    return identity


def _target_plan_document(
    *,
    placement: TargetPlacement,
    identity: dict[str, object],
    identity_sha256: str,
) -> dict[str, object]:
    minimum_gpu_count = int(placement.gpu_rank_counts_by_layer.min())
    maximum_gpu_count = int(placement.gpu_rank_counts_by_layer.max())
    common: dict[str, object] = {
        "format": placement.plan_format,
        "gpu_experts_mask_by_rank": placement.gpu_masks,
        "global_num_experts": torch.tensor(NUM_EXPERTS, dtype=torch.int64),
        "gpu_selection_strategy": "transferred-ep2-winner-gpu-union",
        "min_gpu_experts_per_rank_per_layer": torch.tensor(
            minimum_gpu_count,
            dtype=torch.int64,
        ),
        "max_gpu_experts_per_rank_per_layer": torch.tensor(
            maximum_gpu_count,
            dtype=torch.int64,
        ),
        "placement_semantics_sha256": placement.semantics_sha256,
        "transfer_receipt": {
            **identity,
            "cache_identity_sha256": identity_sha256,
            "target_placement_semantics_sha256": placement.semantics_sha256,
            "status": "complete",
        },
    }
    if placement.plan_format == VARIABLE_PLAN_FORMAT:
        common.update(
            {
                "cpu_expert_ids_padded_by_rank": (
                    placement.cpu_expert_ids_padded_by_rank
                ),
                "cpu_rank_counts_by_layer": placement.cpu_rank_counts_by_layer,
                "gpu_rank_counts_by_layer": placement.gpu_rank_counts_by_layer,
            }
        )
        return common

    gpu_count = int(placement.gpu_rank_counts_by_layer[0, 0])
    cpu_count = int(placement.cpu_rank_counts_by_layer[0, 0])
    common.update(
        {
            "cpu_expert_ids_by_rank": [placement.cpu_expert_ids_padded_by_rank[0]],
            "gpu_rank_counts": torch.tensor([gpu_count], dtype=torch.int64),
            "cpu_rank_counts": torch.tensor([cpu_count], dtype=torch.int64),
            "gpu_union_expert_count": torch.tensor(gpu_count, dtype=torch.int64),
        }
    )
    return common


def _equal_plan_value(observed: object, expected: object) -> bool:
    if isinstance(expected, torch.Tensor):
        return (
            isinstance(observed, torch.Tensor)
            and observed.dtype == expected.dtype
            and torch.equal(observed, expected)
        )
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and all(
                _equal_plan_value(observed[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _equal_plan_value(observed_item, expected_item)
                for observed_item, expected_item in zip(observed, expected, strict=True)
            )
        )
    return type(observed) is type(expected) and observed == expected


def _validate_staged_plan(path: Path, expected: dict[str, object]) -> None:
    if not path.is_file() or path.is_symlink():
        raise PlanTransferError(f"staged PP2 plan is not a regular file: {path}")
    observed = _load_plan(path, "staged PP2 plan")
    if set(observed) != set(expected) or any(
        not _equal_plan_value(observed[key], value) for key, value in expected.items()
    ):
        raise PlanTransferError(f"staged PP2 plan verification failed: {path}")


@contextmanager
def _cache_lock(cache_root: Path) -> Iterator[None]:
    lock_path = cache_root / ".transfer.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as error:
        raise PlanTransferError(
            f"cannot open transfer lock {lock_path}: {error}"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OSError as error:
        raise PlanTransferError(
            f"cannot lock cache root {cache_root}: {error}"
        ) from error
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def stage_transferred_plan(
    *,
    source_ep2_plan: Path,
    source_ep_confirmation_receipt: Path | None = None,
    source_ep_coherency_receipt: Path | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    expected_target_gpu_experts_per_layer: int | None = None,
) -> Path:
    if expected_target_gpu_experts_per_layer is not None and (
        isinstance(expected_target_gpu_experts_per_layer, bool)
        or not 1 <= expected_target_gpu_experts_per_layer <= NUM_EXPERTS
    ):
        raise PlanTransferError(
            "expected target GPU expert ceiling must be an integer in [1, 256]"
        )
    ep_confirmation = (
        load_ep_confirmation(
            source_ep_confirmation_receipt,
            coherency_receipt=source_ep_coherency_receipt,
            expected_source_plan=source_ep2_plan,
        )
        if source_ep_confirmation_receipt is not None
        else None
    )
    resolved_source = _resolve_source_file(source_ep2_plan)
    source_sha256 = sha256_file(resolved_source)
    source_placement = load_source_placement(
        resolved_source,
        expected_target_gpu_experts_per_layer=(expected_target_gpu_experts_per_layer),
    )
    if (
        ep_confirmation is not None
        and source_placement.semantics_sha256
        != ep_confirmation.source_placement_semantics_sha256
    ):
        raise PlanTransferError(
            "source EP2 placement semantics do not match the final EP confirmation"
        )
    if sha256_file(resolved_source) != source_sha256:
        raise PlanTransferError("source EP2 plan changed while loading")
    target_placement = transfer_placement(source_placement)
    identity = _identity_document(
        source_plan=resolved_source,
        source_plan_sha256=source_sha256,
        source_placement=source_placement,
        ep_confirmation=ep_confirmation,
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    expected_plan = _target_plan_document(
        placement=target_placement,
        identity=identity,
        identity_sha256=identity_sha256,
    )
    resolved_cache_root = _prepare_cache_root(cache_root, resolved_source)
    minimum_gpu_count = int(target_placement.gpu_rank_counts_by_layer.min())
    maximum_gpu_count = int(target_placement.gpu_rank_counts_by_layer.max())
    if minimum_gpu_count == maximum_gpu_count:
        gpu_width_label = f"g{maximum_gpu_count}"
    else:
        gpu_width_label = f"g{minimum_gpu_count}to{maximum_gpu_count}"
    output_path = resolved_cache_root / (
        f"pp2-ep1-{gpu_width_label}-{identity_sha256}.pt"
    )

    with _cache_lock(resolved_cache_root):
        if output_path.exists() or output_path.is_symlink():
            _validate_staged_plan(output_path, expected_plan)
            return output_path

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.partial-",
            dir=resolved_cache_root,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        published = False
        try:
            torch.save(expected_plan, temporary_path)
            with temporary_path.open("rb") as staged_file:
                os.fsync(staged_file.fileno())
            _validate_staged_plan(temporary_path, expected_plan)
            try:
                os.rename(temporary_path, output_path)
            except OSError as error:
                raise PlanTransferError(
                    f"cannot atomically publish PP2 plan {output_path}: {error}"
                ) from error
            published = True
        finally:
            if not published and temporary_path.exists():
                temporary_path.unlink()
        _validate_staged_plan(output_path, expected_plan)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ep2-plan", type=Path, required=True)
    parser.add_argument(
        "--source-ep-confirmation-receipt",
        type=Path,
        required=True,
        help=(
            "qualified final EP confirm-stage receipt whose plan and Oscar/CPU "
            "contract must match the source"
        ),
    )
    parser.add_argument(
        "--source-ep-coherency-receipt",
        type=Path,
        help=(
            "strict deterministic text/tool-call receipt produced beside the "
            "authoritative final EP hotspot receipt"
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"content-addressed plan cache (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--expected-target-gpu-experts-per-layer",
        type=int,
        choices=range(1, NUM_EXPERTS + 1),
        metavar="MAX_COUNT",
        help=(
            "fail unless the maximum per-layer source EP2 rank union has this "
            "width; use the same admission ceiling as DSV4_GPU_EXPERTS_PER_LAYER "
            "in the PP2 launcher"
        ),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = cast(
        _ParsedArguments,
        cast(object, build_parser().parse_args(arguments)),
    )
    try:
        output_path = stage_transferred_plan(
            source_ep2_plan=parsed.source_ep2_plan,
            source_ep_confirmation_receipt=(
                parsed.source_ep_confirmation_receipt
            ),
            source_ep_coherency_receipt=parsed.source_ep_coherency_receipt,
            cache_root=parsed.cache_root,
            expected_target_gpu_experts_per_layer=(
                parsed.expected_target_gpu_experts_per_layer
            ),
        )
    except PlanTransferError as error:
        print(f"transfer_dsv4_ep2_plan_to_pp2: {error}", file=sys.stderr)
        return 2
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
