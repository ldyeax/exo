#!/usr/bin/env python3
"""Run a matched Marlin-only GLM-5.2 TP2 MTP quality gate.

The gate launches the same immutable hybrid checkpoint twice, sequentially:
first with MTP disabled and then with MTP enabled.  Both launches are pinned to
concurrency one, TBO/SLP/resident experts off, BF16 KV, compact Marlin
``kv_b_proj``, and original SGLang logprobs.  A representative native
``/generate`` workload captures teacher-forced perplexity, fixed top-k logits,
and deterministic coding/agent generations.  Each launch writes its own
receipt after verified cleanup; a final receipt compares the matched runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Final, Literal, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import httpx  # noqa: E402

from exo.shared.types.common import Host  # noqa: E402
from scripts import glm52_mtp_quality as quality  # noqa: E402
from scripts import run_sglang_kt_glm47_pp2_local_diagnostic as pp2  # noqa: E402
from scripts import run_sglang_kt_glm47_tp2_local_diagnostic as tp2  # noqa: E402
from scripts import run_sglang_kt_glm52_pp3_benchmark as glm52  # noqa: E402
from scripts import run_sglang_kt_glm52_tp2_local_benchmark as benchmark  # noqa: E402

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type QualityProfileName = Literal["thin", "representative"]
type QualityWorkloadRunner = Callable[
    [
        httpx.Client,
        Sequence[quality.TokenizedQualityCase],
        QualityCaptureContext,
    ],
    quality.QualityRun,
]

DEFAULT_RUNTIME_PYTHON: Final = (
    "/var/lib/exo/runtimes/glm52-osdi26-w8-overlay/dwagon/"
    "b1b05ea2a1b5b893c2ce5d2e3cd907bd100b57e963725936cb743ff5bf3a64e9/"
    "venv/bin/python"
)
DEFAULT_RUNTIME_INSTALL_RECEIPT: Final = (
    "/var/lib/exo/runtimes/glm52-osdi26-w8-overlay/dwagon/"
    "b1b05ea2a1b5b893c2ce5d2e3cd907bd100b57e963725936cb743ff5bf3a64e9/"
    "install-receipt.json"
)
DEFAULT_RUNTIME_INSTALL_RECEIPT_SHA256: Final = (
    "8ec4818a265d14c305e9cb0a7f086d72bb3037f1f29ab9ab8a4c0ff76ac88bae"
)
DEFAULT_SOURCE_DIRECTORY: Final = "/tmp/kvb-marlin-src"
DEFAULT_MODEL_PATH: Final = "/mnt/sanic/glm52-AMXINT4-W8A16-hybrid"
DEFAULT_KTRANSFORMERS_WEIGHT_PATH: Final = "/mnt/sanic/glm52-AMXINT4"
DEFAULT_SHARED_HOST_WEIGHTS_MANIFEST: Final = (
    "/var/lib/exo/shared-host-weights/glm52-amxint4-manifest.json"
)
DEFAULT_SHARED_HOST_WEIGHTS_CONTENT_ID: Final = (
    "3cfb9c32388cd021a725e60022ff312688f90cdfb72ac257f2851cffb5903a07"
)
DEFAULT_SHARED_HOST_WEIGHTS_STATE_DIRECTORY: Final = (
    "/var/lib/exo/shared-host-weights/glm52-amxint4-state"
)
DEFAULT_CONTEXT_LENGTH: Final = 1_024
DEFAULT_MAXIMUM_INPUT_TOKENS: Final = 512
DEFAULT_GENERATION_TOKENS: Final = 128
DEFAULT_STATIC_MEMORY_FRACTION: Final = 0.8
DEFAULT_MINIMUM_PRELAUNCH_FREE_VRAM_MIB: Final = 20_000
MODE_RESULT_FILENAME: Final = "glm52-tp2-mtp-quality-mode-result.json"
PAIRED_RESULT_FILENAME: Final = "glm52-tp2-mtp-quality-gate-result.json"
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_GENERATE_RESPONSE_MAXIMUM_BYTES: Final = 64 * 1024 * 1024
DEFAULT_GPU_OWNER_QUIESCENCE_TIMEOUT_SECONDS: Final = 15.0
DEFAULT_GPU_OWNER_QUIESCENCE_POLL_SECONDS: Final = 0.25
MTP_ON_ENDPOINT_PORT_OFFSET: Final = 20


class Glm52MtpQualityGateError(RuntimeError):
    """Raised when matched Marlin quality evidence is incomplete."""


def _validated_mode_endpoint_ports(
    config: benchmark.BenchmarkConfig,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return disjoint ``(distributed, service)`` ports for off/on modes."""

    mtp_off_ports = (config.distributed_port, config.service_port)
    mtp_on_ports = tuple(port + MTP_ON_ENDPOINT_PORT_OFFSET for port in mtp_off_ports)
    all_ports = (*mtp_off_ports, *mtp_on_ports)
    if (
        MTP_ON_ENDPOINT_PORT_OFFSET <= 0
        or any(not 1 <= port <= 65_535 for port in all_ports)
        or mtp_off_ports[0] == mtp_off_ports[1]
        or mtp_on_ports[0] == mtp_on_ports[1]
        or not set(mtp_off_ports).isdisjoint(mtp_on_ports)
    ):
        raise ValueError(
            "quality gate requires distinct, disjoint, in-range MTP mode endpoints"
        )
    return mtp_off_ports, cast(tuple[int, int], mtp_on_ports)


@dataclass(frozen=True, slots=True)
class QualityGateConfig:
    """Paired gate inputs plus the already-admitted TP2 launch contract."""

    benchmark_config: benchmark.BenchmarkConfig
    profile: quality.QualityProfile
    thresholds: quality.QualityThresholds

    def __post_init__(self) -> None:
        launch = self.benchmark_config
        _validated_mode_endpoint_ports(launch)
        if (
            launch.benchmark_concurrencies != (1,)
            or launch.kv_cache_dtype != "bfloat16"
            or launch.mla_kv_b_w8_backend != "marlin"
            or launch.enable_two_batch_overlap
            or launch.enable_amx_fine_grained_decode
            or launch.enable_stream_prefill
            or launch.enable_mtp
            or not launch.enable_shared_host_weights
            or launch.resident_gpu_expert_budget_total != 0
            or launch.capture_representative_routing
            or launch.benchmark_output_tokens < self.profile.generation_max_new_tokens
        ):
            raise ValueError(
                "quality gate requires c1, BF16 KV, Marlin, shared host weights, "
                "MTP/TBO/SLP/residency/routing off, and enough reserved output "
                "capacity in the paired base config"
            )


@dataclass(frozen=True, slots=True)
class QualityCaptureContext:
    run_id: str
    enable_mtp: bool
    profile: quality.QualityProfile
    run_contract: quality.RunContract
    marlin_census: quality.MarlinCensus


@dataclass(frozen=True, slots=True)
class QualityModeExecution:
    """One fully cleaned launch and its in-memory comparison evidence."""

    payload: JsonObject
    quality_run: quality.QualityRun | None


@dataclass(frozen=True, slots=True)
class QualityProcessSpec(benchmark.Glm52Tp2ProcessSpec):
    """TP2 launch with original unrounded logprobs made explicit."""

    @property
    def command(self) -> tuple[str, ...]:
        """Pin controls that SGLang otherwise derives differently by MTP mode."""

        inherited = super().command
        load_format = (
            () if "--load-format" in inherited else ("--load-format", "safetensors")
        )
        random_seed = (
            ()
            if "--random-seed" in inherited
            else ("--random-seed", str(quality.DEFAULT_SAMPLING_SEED))
        )
        return (
            *inherited,
            *load_format,
            *random_seed,
            "--disable-overlap-schedule",
        )

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        return (
            *super().environment,
            ("SGLANG_RETURN_ORIGINAL_LOGPROB", "1"),
        )

    def receipt(self) -> JsonObject:
        receipt = super().receipt()
        environment = cast(JsonObject, receipt["environment"])
        environment["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
        receipt["quality_logprob_contract"] = {
            "return_original_logprob": True,
            "environment_variable": "SGLANG_RETURN_ORIGINAL_LOGPROB",
            "environment_value": "1",
        }
        receipt["quality_execution_contract"] = {
            "server_random_seed": quality.DEFAULT_SAMPLING_SEED,
            "disable_overlap_schedule": True,
            "load_format": "safetensors",
        }
        return receipt


def build_quality_process_spec(
    config: benchmark.BenchmarkConfig,
    *,
    enable_mtp: bool,
) -> QualityProcessSpec:
    """Build the proven launch spec for one already-derived mode config."""

    return QualityProcessSpec(
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
        maximum_running_requests=1,
        kv_cache_dtype="bfloat16",
        mla_kv_b_w8_backend="marlin",
        enable_two_batch_overlap=False,
        enable_amx_fine_grained_decode=False,
        enable_stream_prefill=False,
        stream_prefill_token_threshold=config.stream_prefill_token_threshold,
        stream_prefill_experts_per_chunk=config.stream_prefill_experts_per_chunk,
        enable_mtp=enable_mtp,
        enable_shared_host_weights=True,
        shared_host_weights_manifest=config.shared_host_weights_manifest,
        shared_host_weights_content_id=config.shared_host_weights_content_id,
        shared_host_weights_state_directory=(
            config.shared_host_weights_state_directory
        ),
        chunked_prefill_size=config.chunked_prefill_size,
        resident_gpu_expert_budget_total=0,
        kt_gpu_experts_ratio=None,
        expert_placement_strategy="uniform",
        init_expert_location=None,
        init_expert_location_sha256=None,
        capture_representative_routing=False,
    )


def _mode_config(
    gate: QualityGateConfig,
    *,
    enable_mtp: bool,
) -> benchmark.BenchmarkConfig:
    base = gate.benchmark_config
    mode_name = "mtp-on" if enable_mtp else "mtp-off"
    mtp_off_ports, mtp_on_ports = _validated_mode_endpoint_ports(base)
    distributed_port, service_port = mtp_on_ports if enable_mtp else mtp_off_ports
    return replace(
        base,
        run_id=f"{base.run_id}-{mode_name}",
        result_directory=base.result_directory / mode_name,
        distributed_port=distributed_port,
        service_port=service_port,
        enable_mtp=enable_mtp,
    )


def _quality_run_receipt(run: quality.QualityRun) -> JsonObject:
    return cast(JsonObject, asdict(run))


def _comparison_receipt(
    comparison: quality.MatchedQualityComparison,
) -> JsonObject:
    return comparison.json_object()


def _write_json(path: Path, payload: JsonObject) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    tp2._fsync_directory(path.parent)


def _mode_result_path(config: benchmark.BenchmarkConfig) -> Path:
    return config.result_directory / MODE_RESULT_FILENAME


def _require_hybrid_marlin_contract(contract: JsonObject) -> None:
    if contract.get("compact_mla_kv_b_w8") is not True:
        raise Glm52MtpQualityGateError(
            "quality gate requires the compact hybrid kv_b W8 checkpoint"
        )
    manifest = contract.get("hybrid_checkpoint_manifest")
    if not isinstance(manifest, dict) or manifest.get("content_id") is None:
        raise Glm52MtpQualityGateError(
            "quality gate requires a content-addressed hybrid manifest"
        )


def _mode_runtime_contract(
    verified_mtp_on_contract: JsonObject,
    *,
    enable_mtp: bool,
) -> JsonObject:
    """Change only launch-intent fields after one immutable-artifact check."""

    contract = cast(JsonObject, deepcopy(verified_mtp_on_contract))
    raw_policy = contract.get("mtp_policy")
    if not isinstance(raw_policy, dict):
        raise Glm52MtpQualityGateError("verified contract lacks its MTP policy")
    policy = cast(JsonObject, raw_policy)
    expected_verified_policy: JsonObject = {
        "causal_layer_range": [0, benchmark.MODEL_LAYER_COUNT],
        "layer_78_is_nextn_predict": True,
        "layer_78_loaded": True,
        "layer_78_experts": "persistent_amxint4",
        "speculative_decoding_enabled": True,
    }
    if policy != expected_verified_policy:
        raise Glm52MtpQualityGateError(
            "one-time checkpoint verification did not use the MTP-on superset"
        )
    policy["layer_78_loaded"] = enable_mtp
    policy["layer_78_experts"] = "persistent_amxint4" if enable_mtp else "not_loaded"
    policy["speculative_decoding_enabled"] = enable_mtp
    contract["verification_derivation"] = {
        "artifact_and_runtime_verified_once_with_mtp_superset": True,
        "mode_specific_fields": [
            "mtp_policy.layer_78_loaded",
            "mtp_policy.layer_78_experts",
            "mtp_policy.speculative_decoding_enabled",
        ],
        "enable_mtp": enable_mtp,
    }
    return contract


def _tokenize_quality_cases(
    tokenizer: glm52.BenchmarkTokenizer,
    gate: QualityGateConfig,
) -> tuple[tuple[quality.TokenizedQualityCase, ...], JsonObject]:
    maximum_input_tokens = gate.benchmark_config.benchmark_input_tokens

    def encode_representative(text: str) -> Sequence[int]:
        rendered = glm52._render_chat_prompt(tokenizer, text)
        return tokenizer.encode(rendered, add_special_tokens=False)

    def encode_humaneval(text: str) -> Sequence[int]:
        instruction = (
            "Read the following Python function signature and docstring, then "
            "complete the function. Return only the implementation code.\n\n"
        )
        rendered = glm52._render_chat_prompt(tokenizer, instruction + text)
        return tokenizer.encode(rendered, add_special_tokens=False)

    ordered_cases = quality.tokenize_quality_cases(
        encode_representative,
        encode_humaneval,
        profile=gate.profile,
        maximum_input_tokens=maximum_input_tokens,
    )
    receipt: JsonObject = {
        "profile": cast(JsonObject, asdict(gate.profile)),
        "profile_content_sha256": gate.profile.content_sha256,
        "maximum_input_tokens": maximum_input_tokens,
        "humaneval_reference_content_sha256": (
            quality.HUMANEVAL_REFERENCE_CONTENT_SHA256
        ),
        "cases": [
            {
                "case_id": item.case_id,
                "category": item.category,
                "input_token_count": len(item.input_ids),
                "input_ids_sha256": quality.token_ids_sha256(item.input_ids),
                "teacher_forced_token_count": len(
                    item.teacher_forced_input_ids
                    if item.teacher_forced_input_ids is not None
                    else item.input_ids
                ),
                "teacher_forced_score_start_index": (
                    item.teacher_forced_score_start_index
                ),
                "teacher_forced_target_kind": item.teacher_forced_target_kind,
            }
            for item in ordered_cases
        ],
    }
    return ordered_cases, receipt


def _quality_run_contract(
    gate: QualityGateConfig,
    runtime_contract: JsonObject,
    *,
    enable_mtp: bool,
    workload_receipt: JsonObject,
) -> quality.RunContract:
    manifest = runtime_contract.get("hybrid_checkpoint_manifest")
    checkpoints = runtime_contract.get("checkpoints")
    runtime = runtime_contract.get("runtime")
    shared_weights = runtime_contract.get("shared_host_weights")
    if not all(
        isinstance(value, (dict, list))
        for value in (manifest, checkpoints, runtime, shared_weights)
    ):
        raise Glm52MtpQualityGateError(
            "runtime receipt lacks matched artifact identity"
        )
    common: JsonObject = {
        "topology": {
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 2,
            "maximum_running_requests": 1,
            "ordered_gpu_uuids": list(benchmark.ORDERED_GPU_UUIDS),
        },
        "execution": {
            "kv_cache_dtype": "bfloat16",
            "mla_kv_b_w8_backend": "marlin",
            "two_batch_overlap": False,
            "stream_prefill": False,
            "resident_gpu_expert_budget_total": 0,
            "shared_host_weights": True,
            "return_original_logprob": True,
            "server_random_seed": quality.DEFAULT_SAMPLING_SEED,
            "disable_overlap_schedule": True,
            "load_format": "safetensors",
        },
        "runtime": cast(JsonValue, runtime),
        "workload": workload_receipt,
    }
    artifact: JsonObject = {
        "hybrid_checkpoint_manifest": cast(JsonValue, manifest),
        "checkpoints": cast(JsonValue, checkpoints),
        "shared_host_weights": cast(JsonValue, shared_weights),
    }
    return quality.RunContract(
        common=common,
        artifact=artifact,
        runtime_controls=quality.MarlinRuntimeControls(
            configured_backend="marlin",
            environment_backend="marlin",
            return_original_logprob_environment="1",
        ),
        mtp=quality.MtpConfiguration(
            enabled=enable_mtp,
            speculative_algorithm="EAGLE" if enable_mtp else None,
            speculative_num_steps=1 if enable_mtp else None,
            speculative_eagle_topk=1 if enable_mtp else None,
            speculative_num_draft_tokens=2 if enable_mtp else None,
        ),
    )


def _marlin_census(
    attestation: JsonObject,
    *,
    enable_mtp: bool,
) -> quality.MarlinCensus:
    raw_observations = attestation.get("observations")
    if not isinstance(raw_observations, list) or len(raw_observations) != 2:
        raise Glm52MtpQualityGateError("Marlin attestation lacks both TP ranks")
    observations: list[quality.MarlinRankObservation] = []
    for raw in raw_observations:
        if not isinstance(raw, dict):
            raise Glm52MtpQualityGateError("Marlin rank receipt is invalid")
        observations.append(
            quality.MarlinRankObservation(
                tensor_parallel_rank=cast(
                    Literal[0, 1],
                    raw.get("tensor_parallel_rank"),
                ),
                backend=cast(Literal["marlin"], raw.get("backend")),
                module_count=cast(int, raw.get("module_count")),
                local_heads_per_module=cast(
                    int,
                    raw.get("local_heads_per_module"),
                ),
            )
        )
    return quality.MarlinCensus(
        mtp_enabled=enable_mtp,
        requested_backend=cast(
            Literal["marlin"],
            attestation.get("requested_backend"),
        ),
        expected_runtime_backend=cast(
            Literal["marlin"],
            attestation.get("expected_runtime_backend"),
        ),
        passed=cast(Literal[True], attestation.get("passed")),
        observations=cast(
            tuple[
                quality.MarlinRankObservation,
                quality.MarlinRankObservation,
            ],
            tuple(observations),
        ),
    )


def _capture_quality_workload(
    client: httpx.Client,
    cases: Sequence[quality.TokenizedQualityCase],
    context: QualityCaptureContext,
) -> quality.QualityRun:
    """Thin native client adapter; validation and comparison live in the core."""

    def generate(request: JsonObject) -> JsonObject:
        response = client.post("/generate", json=request)
        if len(response.content) > _GENERATE_RESPONSE_MAXIMUM_BYTES:
            raise Glm52MtpQualityGateError(
                "native generate response exceeds its safety bound"
            )
        if response.status_code != 200:
            raise Glm52MtpQualityGateError(
                f"native generate returned HTTP {response.status_code}: "
                f"{response.text[-1000:]}"
            )
        return glm52._strict_response_object(
            response,
            "quality native generate response",
        )

    return quality.capture_quality_run(
        generate,
        cases,
        run_id=context.run_id,
        contract=context.run_contract,
        backend_census=context.marlin_census,
        profile=context.profile,
    )


def _run_quality_mode(
    gate: QualityGateConfig,
    *,
    enable_mtp: bool,
    runtime_and_checkpoint_contract: JsonObject,
    cases: Sequence[quality.TokenizedQualityCase],
    workload_receipt: JsonObject,
    workload_runner: QualityWorkloadRunner = _capture_quality_workload,
) -> QualityModeExecution:
    """Launch, capture one quality half, and always perform owned cleanup."""

    config = _mode_config(gate, enable_mtp=enable_mtp)
    spec = build_quality_process_spec(config, enable_mtp=enable_mtp)
    lifecycle_config = benchmark._lifecycle_config(config)
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    started_at = benchmark._utc_now()
    started_monotonic = time.monotonic()
    owner_token = uuid.uuid4().hex
    process_spec = spec.receipt()
    prelaunch_snapshot: benchmark.GpuCapacitySnapshot | None = None
    prelaunch_gate: JsonObject | None = None
    postreadiness_snapshot: benchmark.GpuCapacitySnapshot | None = None
    postreadiness_gate: JsonObject | None = None
    server_capacity_gate: JsonObject | None = None
    marlin_attestation: JsonObject | None = None
    server_info: JsonObject | None = None
    quality_run: quality.QualityRun | None = None
    running: tp2.RunningParent | None = None
    readiness: tuple[JsonObject, ...] = ()
    readiness_ownership: JsonObject | None = None
    distributed_ownership: JsonObject | None = None
    launch_evidence: JsonObject | None = None
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []
    cleanup_complete = False
    journal_created = False
    journal_cleared = False
    partial_start: JsonObject | None = None
    partial_start_cleanup_verified: bool | None = None
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
            prelaunch_snapshot = benchmark.collect_gpu_capacity_snapshot()
            prelaunch_gate = benchmark.validate_prelaunch_gpu_capacity(
                prelaunch_snapshot,
                config,
            )
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
            marlin_attestation = benchmark.validate_compact_mla_backend_attestation(
                config.result_directory / "rank-0.log",
                spec,
            )
            postreadiness_snapshot = benchmark.collect_gpu_capacity_snapshot()
            postreadiness_gate = benchmark.validate_postreadiness_gpu_capacity(
                postreadiness_snapshot,
                config,
                running,
            )
            signal_state.checkpoint()

            with httpx.Client(
                base_url=f"http://{spec.service_endpoint}",
                timeout=config.request_timeout_seconds,
                headers={"Accept-Encoding": "identity"},
            ) as client:
                info_response = client.get("/server_info")
                info_response.raise_for_status()
                server_info = glm52._strict_response_object(
                    info_response,
                    "server_info",
                )
                server_capacity_gate = benchmark.validate_server_capacity(
                    server_info,
                    config,
                    spec,
                )
                distributed_ownership = (
                    tp2.observe_distributed_coordinator_listener_ownership(
                        spec,
                        running,
                    )
                )
                run_contract = _quality_run_contract(
                    gate,
                    runtime_and_checkpoint_contract,
                    enable_mtp=enable_mtp,
                    workload_receipt=workload_receipt,
                )
                census = _marlin_census(
                    marlin_attestation,
                    enable_mtp=enable_mtp,
                )
                quality_run = workload_runner(
                    client,
                    cases,
                    QualityCaptureContext(
                        run_id=config.run_id,
                        enable_mtp=enable_mtp,
                        profile=gate.profile,
                        run_contract=run_contract,
                        marlin_census=census,
                    ),
                )
                if running.process.poll() is not None:
                    raise Glm52MtpQualityGateError(
                        "TP2 parent exited during representative quality capture"
                    )
                signal_state.checkpoint()
        except BaseException as error:
            if isinstance(error, tp2.Tp2ParentLaunchError):
                launch_evidence = error.launch_evidence
                journal_created = journal_created or error.journal_created
                journal_cleared = journal_cleared or error.journal_cleared
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
            failure = error
        finally:
            signal_state.begin_cleanup()
            if running is not None:
                try:
                    cleanup_receipt = tp2.stop_local_parent(
                        running,
                        config.cleanup_timeout_seconds,
                    )
                    cleanup.append(
                        {
                            "rank": 0,
                            **cleanup_receipt.model_dump(mode="json"),
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
                except BaseException as error:
                    if failure is None:
                        failure = error
    finally:
        for managed_signal, previous_handler in previous_handlers.items():
            signal.signal(managed_signal, previous_handler)

    journal_path = tp2._ownership_journal_path(lifecycle_config)
    journal_retained = journal_path.exists()
    passed = (
        failure is None
        and running is not None
        and cleanup_complete
        and not journal_retained
        and prelaunch_gate is not None
        and postreadiness_gate is not None
        and server_capacity_gate is not None
        and marlin_attestation is not None
        and quality_run is not None
    )
    log_receipt = benchmark._log_receipt(config)
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm52_tp2_marlin_mtp_quality_mode",
        "status": "passed" if passed else "failed",
        "run_id": config.run_id,
        "mode": "mtp_on" if enable_mtp else "mtp_off",
        "started_at_utc": started_at,
        "completed_at_utc": benchmark._utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "matched_launch_contract": {
            "concurrency": 1,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "service_endpoint": str(spec.service_endpoint),
            "distributed_endpoint": str(spec.distributed_coordinator),
            "endpoint_policy": "deterministic_disjoint_pair_by_mtp_mode",
            "kv_cache_dtype": "bfloat16",
            "mla_kv_b_w8_backend": "marlin",
            "return_original_logprob": True,
            "enable_two_batch_overlap": False,
            "enable_stream_prefill": False,
            "resident_gpu_expert_budget_total": 0,
            "enable_mtp": enable_mtp,
        },
        "runtime_and_checkpoint_contract": runtime_and_checkpoint_contract,
        "process_spec": process_spec,
        "process_spec_sha256": benchmark._canonical_sha256(process_spec),
        "launch": (
            launch_evidence
            if launch_evidence is not None
            else {
                "process_created": False,
                "evidence_origin": "no_successful_popen",
            }
        ),
        "readiness": list(readiness),
        "readiness_ownership": readiness_ownership,
        "distributed_coordinator_ownership": distributed_ownership,
        "capacity_and_vram": {
            "prelaunch_snapshot": (
                None
                if prelaunch_snapshot is None
                else cast(JsonObject, asdict(prelaunch_snapshot))
            ),
            "prelaunch_gate": prelaunch_gate,
            "postreadiness_snapshot": (
                None
                if postreadiness_snapshot is None
                else cast(JsonObject, asdict(postreadiness_snapshot))
            ),
            "postreadiness_gate": postreadiness_gate,
            "server_capacity_gate": server_capacity_gate,
        },
        "server_info": server_info,
        "compact_mla_backend_attestation": marlin_attestation,
        "quality_run": (
            None if quality_run is None else _quality_run_receipt(quality_run)
        ),
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "ownership_journal": {
            "path": str(journal_path),
            "created": journal_created,
            "cleared_before_receipt": journal_cleared,
            "retained": journal_retained,
        },
        "logs": [] if log_receipt is None else [log_receipt],
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = benchmark._canonical_sha256(payload)
    _write_json(_mode_result_path(config), payload)
    return QualityModeExecution(payload=payload, quality_run=quality_run)


def _mode_cleanup_allows_next(mode: QualityModeExecution) -> bool:
    journal = mode.payload.get("ownership_journal")
    return (
        mode.payload.get("cleanup_complete") is True
        and isinstance(journal, dict)
        and journal.get("retained") is False
    )


def wait_for_pinned_gpu_owner_quiescence(
    ordered_gpu_uuids: tuple[str, ...],
    *,
    timeout_seconds: float = DEFAULT_GPU_OWNER_QUIESCENCE_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_GPU_OWNER_QUIESCENCE_POLL_SECONDS,
    required_clear_observations: int = 2,
    snapshot_collector: Callable[
        [], benchmark.GpuCapacitySnapshot
    ] = benchmark.collect_gpu_capacity_snapshot,
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> JsonObject:
    """Wait boundedly for teardown to release every pinned GPU compute owner."""

    if (
        not ordered_gpu_uuids
        or len(set(ordered_gpu_uuids)) != len(ordered_gpu_uuids)
        or timeout_seconds <= 0.0
        or poll_seconds <= 0.0
        or poll_seconds > timeout_seconds
        or required_clear_observations < 1
    ):
        raise ValueError("GPU owner quiescence policy is invalid")
    started_at_utc = benchmark._utc_now()
    started_monotonic = monotonic_clock()
    deadline = started_monotonic + timeout_seconds
    observations: list[JsonValue] = []
    pinned_gpu_uuid_set = set(ordered_gpu_uuids)
    failure_reason: str | None = None
    passed = False
    consecutive_clear_observations = 0
    while True:
        snapshot = snapshot_collector()
        observed_gpu_uuids = {device.uuid for device in snapshot.devices}
        missing_gpu_uuids = sorted(pinned_gpu_uuid_set - observed_gpu_uuids)
        pinned_owners = tuple(
            process
            for process in snapshot.compute_processes
            if process.gpu_uuid in pinned_gpu_uuid_set
        )
        observations.append(
            {
                "observed_at_utc": snapshot.observed_at_utc,
                "compute_process_stdout_sha256": (
                    snapshot.compute_process_stdout_sha256
                ),
                "missing_pinned_gpu_uuids": missing_gpu_uuids,
                "pinned_compute_owners": [
                    {
                        "gpu_uuid": process.gpu_uuid,
                        "pid": process.pid,
                        "used_memory_mib": process.used_memory_mib,
                    }
                    for process in pinned_owners
                ],
            }
        )
        if missing_gpu_uuids:
            failure_reason = "pinned_gpu_inventory_missing"
            break
        if not pinned_owners:
            consecutive_clear_observations += 1
            if consecutive_clear_observations >= required_clear_observations:
                passed = True
                break
        else:
            consecutive_clear_observations = 0
        now = monotonic_clock()
        remaining_seconds = deadline - now
        if remaining_seconds <= 0.0:
            failure_reason = "timeout_with_pinned_compute_owners"
            break
        sleeper(min(poll_seconds, remaining_seconds))

    completed_monotonic = monotonic_clock()
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "policy": "all_pinned_gpus_have_zero_compute_owners",
        "ordered_gpu_uuids": list(ordered_gpu_uuids),
        "timeout_seconds": timeout_seconds,
        "poll_seconds": poll_seconds,
        "required_consecutive_clear_observations": required_clear_observations,
        "started_at_utc": started_at_utc,
        "completed_at_utc": benchmark._utc_now(),
        "elapsed_seconds": completed_monotonic - started_monotonic,
        "observation_count": len(observations),
        "observations": observations,
        "failure_reason": failure_reason,
    }


def run_quality_gate(
    gate: QualityGateConfig,
    *,
    workload_runner: QualityWorkloadRunner = _capture_quality_workload,
) -> JsonObject:
    """Run both matched halves and write the paired fail-closed decision."""

    base = gate.benchmark_config
    mtp_off_ports, mtp_on_ports = _validated_mode_endpoint_ports(base)
    base.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    started_at = benchmark._utc_now()
    started_monotonic = time.monotonic()
    runtime_contract: JsonObject | None = None
    shared_prefault: JsonObject | None = None
    workload_receipt: JsonObject | None = None
    mode_results: list[QualityModeExecution] = []
    inter_mode_gpu_owner_quiescence: list[JsonValue] = []
    comparison: quality.MatchedQualityComparison | None = None
    failure: BaseException | None = None
    checkpoint_verified = False
    tokenizer_loaded = False
    shared_prefault_complete = False
    try:
        runtime_contract = benchmark.verify_runtime_and_checkpoint_contract(
            _mode_config(gate, enable_mtp=True)
        )
        _require_hybrid_marlin_contract(runtime_contract)
        checkpoint_verified = True
        # Prefault once with layer 78 included; both launches share this immutable
        # artifact and page cache, so repeating it would add no quality evidence.
        shared_prefault = benchmark.prefault_shared_host_weights(
            replace(
                _mode_config(gate, enable_mtp=True),
                result_directory=base.result_directory,
            )
        )
        shared_prefault_complete = True
        tokenizer = glm52._load_tokenizer(base.model_path)
        tokenizer_loaded = True
        cases, workload_receipt = _tokenize_quality_cases(tokenizer, gate)
        for enable_mtp in (False, True):
            mode_runtime_contract = _mode_runtime_contract(
                runtime_contract,
                enable_mtp=enable_mtp,
            )
            mode = _run_quality_mode(
                gate,
                enable_mtp=enable_mtp,
                runtime_and_checkpoint_contract=mode_runtime_contract,
                cases=cases,
                workload_receipt=workload_receipt,
                workload_runner=workload_runner,
            )
            mode_results.append(mode)
            if not _mode_cleanup_allows_next(mode):
                raise Glm52MtpQualityGateError(
                    "refusing the next launch because owned cleanup is incomplete"
                )
            if mode.payload.get("status") != "passed":
                raise Glm52MtpQualityGateError(
                    f"{mode.payload.get('mode')} quality capture failed; "
                    "fix its receipt before paying for the next model reload"
                )
            if not enable_mtp:
                transition_spec = build_quality_process_spec(
                    base,
                    enable_mtp=False,
                )
                quiescence = wait_for_pinned_gpu_owner_quiescence(
                    transition_spec.ordered_gpu_uuids
                )
                inter_mode_gpu_owner_quiescence.append(quiescence)
                if quiescence.get("status") != "passed":
                    raise Glm52MtpQualityGateError(
                        "refusing the MTP-on launch because pinned GPU compute "
                        "owners did not quiesce within the bounded teardown wait"
                    )
        off_run = mode_results[0].quality_run
        on_run = mode_results[1].quality_run
        if off_run is None or on_run is None:
            raise Glm52MtpQualityGateError(
                "both quality captures are required for comparison"
            )
        comparison = quality.compare_matched_runs(
            off_run,
            on_run,
            thresholds=gate.thresholds,
        )
    except BaseException as error:
        failure = error

    mode_payloads = [mode.payload for mode in mode_results]
    all_modes_passed = len(mode_payloads) == 2 and all(
        payload.get("status") == "passed" for payload in mode_payloads
    )
    comparison_receipt = None if comparison is None else _comparison_receipt(comparison)
    comparison_passed = (
        comparison_receipt is not None and comparison_receipt.get("status") == "passed"
    )
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm52_tp2_marlin_matched_mtp_quality_gate",
        "status": (
            "passed"
            if failure is None and all_modes_passed and comparison_passed
            else "failed"
        ),
        "run_id": base.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": benchmark._utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "execution_order": ["mtp_off", "mtp_on"],
        "mode_endpoint_plan": {
            "policy": "deterministic_disjoint_pair_by_mtp_mode",
            "mtp_on_port_offset": MTP_ON_ENDPOINT_PORT_OFFSET,
            "mtp_off": {
                "distributed": f"{base.dwagon_ip}:{mtp_off_ports[0]}",
                "service": f"{base.dwagon_ip}:{mtp_off_ports[1]}",
            },
            "mtp_on": {
                "distributed": f"{base.dwagon_ip}:{mtp_on_ports[0]}",
                "service": f"{base.dwagon_ip}:{mtp_on_ports[1]}",
            },
        },
        "checkpoint_verified_once_before_matched_launches": checkpoint_verified,
        "tokenizer_loaded_once_before_matched_launches": tokenizer_loaded,
        "shared_host_weights_prefaulted_once_with_mtp_layer": (
            shared_prefault_complete
        ),
        "runtime_and_checkpoint_contract": runtime_contract,
        "shared_host_weight_prefault": shared_prefault,
        "workload": workload_receipt,
        "inter_mode_gpu_owner_quiescence": inter_mode_gpu_owner_quiescence,
        "mode_receipts": [
            {
                "mode": mode.payload["mode"],
                "status": mode.payload["status"],
                "path": str(
                    _mode_result_path(
                        _mode_config(
                            gate,
                            enable_mtp=(mode.payload["mode"] == "mtp_on"),
                        )
                    )
                ),
                "receipt_content_sha256": mode.payload["receipt_content_sha256"],
                "cleanup_complete": mode.payload["cleanup_complete"],
            }
            for mode in mode_results
        ],
        "comparison": comparison_receipt,
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = benchmark._canonical_sha256(payload)
    _write_json(base.result_directory / PAIRED_RESULT_FILENAME, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = benchmark._parser()
    parser.description = __doc__
    parser.set_defaults(
        runtime_python=DEFAULT_RUNTIME_PYTHON,
        runtime_install_receipt=Path(DEFAULT_RUNTIME_INSTALL_RECEIPT),
        runtime_install_receipt_sha256=(DEFAULT_RUNTIME_INSTALL_RECEIPT_SHA256),
        local_source_directory=DEFAULT_SOURCE_DIRECTORY,
        model_path=DEFAULT_MODEL_PATH,
        ktransformers_weight_path=DEFAULT_KTRANSFORMERS_WEIGHT_PATH,
        benchmark_concurrencies=(1,),
        context_length=DEFAULT_CONTEXT_LENGTH,
        benchmark_input_tokens=DEFAULT_MAXIMUM_INPUT_TOKENS,
        benchmark_output_tokens=DEFAULT_GENERATION_TOKENS,
        maximum_total_tokens=None,
        kv_cache_dtype="bfloat16",
        mla_kv_b_w8_backend="marlin",
        enable_two_batch_overlap=False,
        enable_amx_fine_grained_decode=False,
        enable_stream_prefill=False,
        enable_mtp=False,
        enable_shared_host_weights=True,
        shared_host_weights_manifest=Path(DEFAULT_SHARED_HOST_WEIGHTS_MANIFEST),
        shared_host_weights_content_id=(DEFAULT_SHARED_HOST_WEIGHTS_CONTENT_ID),
        shared_host_weights_state_directory=Path(
            DEFAULT_SHARED_HOST_WEIGHTS_STATE_DIRECTORY
        ),
        capture_representative_routing=False,
        static_memory_fraction=DEFAULT_STATIC_MEMORY_FRACTION,
        minimum_prelaunch_free_vram_mib=(DEFAULT_MINIMUM_PRELAUNCH_FREE_VRAM_MIB),
    )
    parser.add_argument(
        "--quality-profile",
        choices=("thin", "representative"),
        default="thin",
        help=(
            "thin runs all eight coding/agent prompts plus two HumanEval "
            "references at 32 output tokens; representative runs all eight "
            "HumanEval references at 128 output tokens"
        ),
    )
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> QualityGateConfig:
    base = benchmark._config_from_arguments(arguments)
    profile_name = cast(QualityProfileName, arguments.quality_profile)
    profile = (
        quality.THIN_FIRST_RUN_PROFILE
        if profile_name == "thin"
        else quality.REPRESENTATIVE_QUALITY_GATE_PROFILE
    )
    return QualityGateConfig(
        benchmark_config=base,
        profile=profile,
        thresholds=quality.QualityThresholds(),
    )


def main() -> int:
    try:
        gate = _config_from_arguments(_parser().parse_args())
        payload = run_quality_gate(gate)
    except (
        Glm52MtpQualityGateError,
        benchmark.Glm52Tp2BenchmarkError,
        tp2.Tp2LocalDiagnosticError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as error:
        print(f"GLM-5.2 matched MTP quality gate failed: {error}", file=sys.stderr)
        return 1
    result_path = gate.benchmark_config.result_directory / PAIRED_RESULT_FILENAME
    print(
        json.dumps(
            {
                "status": payload["status"],
                "comparison": payload["comparison"],
                "result_path": str(result_path),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
