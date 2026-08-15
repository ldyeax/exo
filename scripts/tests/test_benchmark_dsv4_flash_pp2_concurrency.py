from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
import urllib.error
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self, cast
from urllib.parse import urlsplit
from urllib.request import Request

import pytest

from scripts import benchmark_dsv4_flash_pp2_concurrency as benchmark

OSCAR_HASHES = {
    "dsv4_oscar_artifact_sha256": "1" * 64,
    "dsv4_oscar_model_config_sha256": "2" * 64,
    "dsv4_oscar_artifact_provenance_sha256": "3" * 64,
    "dsv4_oscar_checkpoint_sha256": "4" * 64,
    "dsv4_oscar_checkpoint_fingerprint_sha256": "5" * 64,
    "dsv4_oscar_admission_sha256": "6" * 64,
    "dsv4_oscar_admission_receipt_sha256": "7" * 64,
}


def coherent_native_text(lane: int) -> str:
    contract = benchmark.NATIVE_CONTRACTS[lane]
    return " ".join(
        (
            (
                f"The controlled review selects {contract.profile} for the proposed "
                "inference runtime change."
            ),
            (
                f"Its measured baseline was {contract.baseline} tokens per second, "
                f"while the optimized result was {contract.optimized} tokens per "
                "second under the same workload."
            ),
            (
                f"The resulting improvement was {contract.improvement} tokens per "
                "second, but that number is evidence rather than automatic approval."
            ),
            (
                "Reviewers should compare repeated trials, preserve request ordering, "
                "and inspect generated language before accepting any timing claim."
            ),
            (
                f"A {contract.safeguards[0]} could make two simultaneous agents "
                "observe incorrect state even when their aggregate rate looks "
                "attractive."
            ),
            (
                f"The test must also prove {contract.safeguards[1]} because "
                "interrupted tool traffic is normal during an interactive coding "
                "session."
            ),
            (
                f"Finally, {contract.safeguards[2]} protects later requests from "
                "stale buffers, corrupted parser state, and misleading cache behavior."
            ),
            (
                "Distinct prompts and cache-cold starts make the comparison harder to "
                "game with memorized continuations or a single repeated token."
            ),
            (
                "Natural termination confirms that the model completed its answer "
                "instead of filling a forced token budget with meaningless material."
            ),
            (
                "Together these checks connect concurrency performance to behavior "
                "that an actual OpenCode client can safely use."
            ),
            contract.required_ending,
        )
    )


def oscar_worker(pp_rank: int, layer_ids: list[int]) -> dict[str, object]:
    absorption = {
        **benchmark.OSCAR_WO_A_EXACT_STATE,
        "artifact_sha256": OSCAR_HASHES["dsv4_oscar_artifact_sha256"],
        "admission_sha256": OSCAR_HASHES["dsv4_oscar_admission_sha256"],
        "expected_local_compressed_layer_ids": layer_ids,
        "absorbed_local_layer_ids": layer_ids,
        "runtime_restore_skipped_layer_ids": layer_ids,
    }
    return {
        "pid": 1_000 + pp_rank,
        "gpu_id": pp_rank,
        "tp_rank": 0,
        "pp_rank": pp_rank,
        "dp_rank": 0,
        **benchmark.OSCAR_STATIC_SERVER_INFO,
        **benchmark.OSCAR_SPLIT_HISTORY_SERVER_INFO,
        "dsv4_oscar_int2_split_history_workspace_address": 90_000 + pp_rank,
        **OSCAR_HASHES,
        "dsv4_oscar_wo_a_absorption_state": absorption,
    }


def oscar_provenance() -> benchmark.OscarProvenance:
    return benchmark.OscarProvenance(
        admission_receipt_path="/tmp/oscar-admission.json",
        admission_receipt_sha256=OSCAR_HASHES["dsv4_oscar_admission_receipt_sha256"],
        artifact_path="/tmp/oscar-calibration.pt",
        artifact_sha256=OSCAR_HASHES["dsv4_oscar_artifact_sha256"],
        admission_sha256=OSCAR_HASHES["dsv4_oscar_admission_sha256"],
    )


def launch_authorization_provenance() -> benchmark.LaunchAuthorizationProvenance:
    return benchmark.LaunchAuthorizationProvenance(
        authorization_receipt_path="/tmp/pp2-authorization.json",
        authorization_receipt_sha256="9" * 64,
        ordinal=1,
        run_role="transfer",
        ep_confirmation_receipt_sha256="8" * 64,
        ep_coherency_receipt_sha256="6" * 64,
    )


def cpu_worker(pp_rank: int, *, scale_fold: bool) -> dict[str, object]:
    telemetry: dict[str, object]
    if scale_fold:
        telemetry = {
            "requested_mode": "lut-v1",
            "n_block": 128,
            "lut_hash": "06d1a83dbf20f545",
            "zero_invalid_or_fallback_counts": True,
        }
    else:
        telemetry = {
            "required_worker_count": 56,
            "environment_enabled": True,
            "all_live_worker_affinities_exact": True,
            "all_worker_cpus_in_expected_numa": True,
        }
    return {
        "pid": 2_000 + pp_rank,
        "gpu_id": pp_rank,
        "tp_rank": 0,
        "pp_rank": pp_rank,
        "dp_rank": 0,
        "moe_ep_rank": 0,
        "moe_dp_rank": 0,
        "telemetry": telemetry,
        "validation_error": None,
    }


def cpu_proof(prefix: str, *, scale_fold: bool) -> dict[str, object]:
    return {
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
        f"{prefix}_worker_telemetry": [
            cpu_worker(0, scale_fold=scale_fold),
            cpu_worker(1, scale_fold=scale_fold),
        ],
    }


def pp2_server_info() -> dict[str, object]:
    pp0_worker = oscar_worker(0, list(range(2, 21)))
    pp1_worker = oscar_worker(1, list(range(21, 43)))
    return {
        "tp_size": 1,
        "pp_size": 2,
        "ep_size": 1,
        "context_length": 524_288,
        "max_total_tokens": 524_288,
        "kv_cache_dtype": "fp8_e4m3",
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_max_bs_decode": 2,
        "cuda_graph_bs_decode": [1, 2],
        "cuda_graph_backend_prefill": "disabled",
        "disable_overlap_schedule": True,
        "speculative_algorithm": None,
        "max_running_requests": 2,
        "pp_max_micro_batch_size": 1,
        "pp_async_batch_depth": 0,
        "chunked_prefill_size": 1024,
        "max_prefill_tokens": 1024,
        "kt_num_gpu_experts": 28,
        "kt_gpu_expert_admission_ceiling": 28,
        "kt_hybrid_expert_plan_format": "sglang_kt_hybrid_expert_shard_v1",
        "kt_hybrid_placement_semantics_sha256": "b" * 64,
        "kt_hybrid_gpu_rank_counts_by_layer": [[28] * 43],
        "kt_hybrid_min_gpu_experts_per_rank_per_layer": 28,
        "kt_hybrid_max_gpu_experts_per_rank_per_layer": 28,
        "kt_hybrid_total_gpu_expert_layers_by_rank": [28 * 43],
        "kt_cpuinfer": 56,
        "kt_threadpool_count": 2,
        "swa_full_tokens_ratio": 0.0048828125,
        "mem_fraction_static": 0.90,
        "disable_radix_cache": False,
        "dsv4_small_row_routing_configured": True,
        "dsv4_sm86_small_batch_gemm_configured": True,
        "kt_amx_fine_grained_decode_configured": True,
        "kt_mxfp4_amx_min_expert_tokens": 5,
        "kt_mxfp4_avx_tiled_min_expert_tokens": 2,
        "dsv4_int4_c4_indexer_storage": False,
        "dsv4_int4_kv_storage": False,
        "dsv4_sm86_c128_bf16_storage": False,
        "enable_p2p_check": True,
        "pre_warm_nccl": True,
        "kt_hybrid_expert_plan_sha256": "a" * 64,
        **cpu_proof("kt_single_numa_inline_dispatch", scale_fold=False),
        **cpu_proof("kt_mxfp4_avx_scale_fold", scale_fold=True),
        **benchmark.OSCAR_STATIC_SERVER_INFO,
        **benchmark.OSCAR_SPLIT_HISTORY_SERVER_INFO,
        "dsv4_oscar_int2_split_history_workspace_address": 90_000,
        **OSCAR_HASHES,
        "dsv4_oscar_wo_a_absorption_state": pp0_worker[
            "dsv4_oscar_wo_a_absorption_state"
        ],
        "internal_states": [
            {
                "dsv4_oscar_worker_telemetry": pp0_worker,
                "dsv4_oscar_worker_telemetry_workers": [pp0_worker, pp1_worker],
            }
        ],
    }


@dataclass(slots=True)
class FakeServerState:
    paths: list[str] = field(default_factory=list)
    native_payloads: dict[int, dict[str, Any]] = field(default_factory=dict)
    chat_payloads: dict[int, dict[str, Any]] = field(default_factory=dict)
    native_arrivals: dict[int, float] = field(default_factory=dict)
    chat_arrivals: dict[int, float] = field(default_factory=dict)
    degenerate_native_lane: int | None = None
    empty_native_lane: int | None = None
    empty_chat_lane: int | None = None
    wrong_tool_arguments_lane: int | None = None
    unexpected_tool_content_lane: int | None = None
    serialize_streams: bool = False
    server_info: dict[str, object] = field(default_factory=pp2_server_info)
    lock: threading.Lock = field(default_factory=threading.Lock)
    native_barrier: threading.Barrier = field(
        default_factory=lambda: threading.Barrier(2, timeout=3.0)
    )
    chat_barrier: threading.Barrier = field(
        default_factory=lambda: threading.Barrier(2, timeout=3.0)
    )
    stream_lock: threading.Lock = field(default_factory=threading.Lock)


class FakeResponse:
    status = 200

    def __init__(
        self,
        *,
        body: bytes = b"",
        lines: tuple[tuple[float, bytes], ...] = (),
        iteration_lock: threading.Lock | None = None,
    ) -> None:
        self.body = body
        self.lines = lines
        self.iteration_lock = iteration_lock

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        del unused

    def read(self) -> bytes:
        return self.body

    def __iter__(self):
        if self.iteration_lock is not None:
            self.iteration_lock.acquire()
        try:
            for delay_seconds, line in self.lines:
                if delay_seconds:
                    time.sleep(delay_seconds)
                yield line
        finally:
            if self.iteration_lock is not None:
                self.iteration_lock.release()


def sse_line(payload: object) -> bytes:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return b"data: " + encoded + b"\n"


class FakeUrlOpen:
    def __init__(self, state: FakeServerState) -> None:
        self.state = state

    def _payload(self, request: Request) -> dict[str, Any]:
        assert request.data is not None
        payload = json.loads(request.data)
        assert isinstance(payload, dict)
        return cast(dict[str, Any], payload)

    def _generate_response(self, request: Request) -> FakeResponse:
        payload = self._payload(request)
        input_ids = payload.get("input_ids")
        assert isinstance(input_ids, list)
        lane = {11: 0, 12: 1}[input_ids[0]]
        with self.state.lock:
            self.state.native_payloads[lane] = payload
            self.state.native_arrivals[lane] = time.perf_counter()
        self.state.native_barrier.wait()

        if self.state.empty_native_lane == lane:
            return FakeResponse(lines=((0.0, b"data: [DONE]\n"),))

        if self.state.degenerate_native_lane == lane:
            output_text = " a" * 180
            output_ids = [900 + lane] * 180
        else:
            output_text = coherent_native_text(lane)
            output_ids = [1_000 + lane * 500 + index for index in range(180)]

        return FakeResponse(
            iteration_lock=(
                self.state.stream_lock if self.state.serialize_streams else None
            ),
            lines=(
                (
                    0.0,
                    sse_line(
                        {
                            "text": output_text[:80],
                            "output_ids": output_ids[:1],
                            "meta_info": {
                                "prompt_tokens": benchmark.INPUT_TOKEN_COUNT,
                                "completion_tokens": 1,
                            },
                        }
                    ),
                ),
                (
                    0.01,
                    sse_line(
                        {
                            "text": output_text,
                            "output_ids": output_ids,
                            "meta_info": {
                                "prompt_tokens": benchmark.INPUT_TOKEN_COUNT,
                                "completion_tokens": len(output_ids),
                                "finish_reason": {"type": "stop"},
                            },
                        }
                    ),
                ),
                (0.0, b"data: [DONE]\n"),
            ),
        )

    def _chat_response(self, request: Request) -> FakeResponse:
        payload = self._payload(request)
        messages = payload.get("messages")
        assert isinstance(messages, list) and len(messages) == 2
        user_message = messages[1]
        assert isinstance(user_message, dict)
        content = user_message.get("content")
        assert isinstance(content, str)
        lane = 0 if "lane 0" in content else 1
        with self.state.lock:
            self.state.chat_payloads[lane] = payload
            self.state.chat_arrivals[lane] = time.perf_counter()
        self.state.chat_barrier.wait()

        if self.state.empty_chat_lane == lane:
            return FakeResponse(lines=((0.0, b"data: [DONE]\n"),))

        arguments = {
            "lane": lane,
            "marker": benchmark.TOOL_MARKER,
            "component": benchmark.TOOL_COMPONENTS[lane],
            "decision": benchmark.TOOL_DECISIONS[lane],
        }
        if self.state.wrong_tool_arguments_lane == lane:
            arguments["decision"] = "invented-decision"
        encoded_arguments = json.dumps(arguments, separators=(",", ":"))
        split_at = len(encoded_arguments) // 2
        first_delta: dict[str, object] = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{lane}",
                    "type": "function",
                    "function": {
                        "name": "record_benchmark_",
                        "arguments": encoded_arguments[:split_at],
                    },
                }
            ]
        }
        if self.state.unexpected_tool_content_lane == lane:
            first_delta["content"] = "SECRET_UNEXPECTED_PROSE"

        return FakeResponse(
            iteration_lock=(
                self.state.stream_lock if self.state.serialize_streams else None
            ),
            lines=(
                (
                    0.0,
                    sse_line(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": first_delta,
                                    "finish_reason": None,
                                }
                            ]
                        }
                    ),
                ),
                (
                    0.005,
                    sse_line(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "name": "marker",
                                                    "arguments": encoded_arguments[
                                                        split_at:
                                                    ],
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ]
                        }
                    ),
                ),
                (
                    0.0,
                    sse_line(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 80,
                                "completion_tokens": 24,
                            },
                        }
                    ),
                ),
                (0.0, b"data: [DONE]\n"),
            ),
        )

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        assert 0 < timeout <= 5.0
        path = urlsplit(request.full_url).path
        with self.state.lock:
            self.state.paths.append(path)
        if path == "/server_info":
            return FakeResponse(body=json.dumps(self.state.server_info).encode("utf-8"))
        if path == "/flush_cache":
            assert self._payload(request) == {}
            return FakeResponse(body=b'{"success":true}')
        if path == "/generate":
            return self._generate_response(request)
        if path == "/v1/chat/completions":
            return self._chat_response(request)
        raise AssertionError(f"unexpected fake endpoint: {path}")


def benchmark_config() -> benchmark.BenchmarkConfig:
    address = "http://fake-dsv4.invalid"
    return benchmark.BenchmarkConfig(
        generate_url=f"{address}/generate",
        chat_url=f"{address}/v1/chat/completions",
        flush_url=f"{address}/flush_cache",
        model="test-dsv4-model",
        request_timeout_seconds=5.0,
        flush_timeout_seconds=5.0,
        chat_max_tokens=64,
        expected_chunked_prefill_size=1024,
        expected_gpu_experts_per_layer=28,
        expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v1",
        expected_cpuinfer_threads=56,
    )


def native_prompts() -> tuple[list[int], list[int]]:
    return (
        [11] * benchmark.INPUT_TOKEN_COUNT,
        [12] * benchmark.INPUT_TOKEN_COUNT,
    )


def nvlink_counters(base: int) -> dict[str, int]:
    return {
        f"gpu{device}.link{link}.{direction}_kib": base
        for device in range(2)
        for link in range(4)
        for direction in ("tx", "rx")
    }


def test_flush_cache_accepts_current_sglang_plaintext_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        b"Cache flushed.\nPlease check backend logs for more details. "
        b"(When there are running or waiting requests, the operation will not "
        b"be performed.)\n"
    )

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert request.full_url == "http://fake-dsv4.invalid/flush_cache"
        assert 0 < timeout <= 5.0
        return FakeResponse(body=body)

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)

    observation = benchmark.flush_cache("http://fake-dsv4.invalid/flush_cache", 5.0)

    assert observation.status == 200
    assert observation.busy_retries == 0
    assert observation.response_sha256 == hashlib.sha256(body).hexdigest()


def test_flush_cache_retries_transient_busy_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        nonlocal calls
        assert request.full_url == "http://fake-dsv4.invalid/flush_cache"
        assert 0 < timeout <= 5.0
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "requests are still retiring",
                {},
                io.BytesIO(b"busy"),
            )
        return FakeResponse(body=b'{"success":true}')

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)

    observation = benchmark.flush_cache("http://fake-dsv4.invalid/flush_cache", 5.0)

    assert calls == 2
    assert observation.busy_retries == 1


def test_semantic_natural_eos_concurrency_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState()
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    provenance = benchmark.BenchmarkProvenance(
        run_label="fivefold-baseline",
        expert_plan_path="/tmp/pp2-fivefold.pt",
        expert_plan_sha256="a" * 64,
    )
    snapshots = iter((nvlink_counters(100), nvlink_counters(151)))
    report = benchmark.run_benchmark(
        replace(benchmark_config(), require_nvlink_traffic=True),
        native_prompts(),
        provenance=provenance,
        oscar_provenance=oscar_provenance(),
        launch_authorization=launch_authorization_provenance(),
        nvlink_snapshotter=lambda: next(snapshots),
    )

    assert report["schema_version"] == 4
    assert report["ok"] is True
    assert report["performance_claim_eligible"] is True
    configuration = cast(dict[str, Any], report["configuration"])
    assert configuration["launch_authorization"] == (
        launch_authorization_provenance().safe_receipt()
    )
    server_contract = cast(dict[str, Any], configuration["server_contract"])
    assert server_contract["tp_size"] == 1
    assert server_contract["pp_size"] == 2
    assert server_contract["ep_size"] == 1
    assert server_contract["max_total_tokens"] == 524_288
    assert server_contract["cuda_graph_max_bs_decode"] == 2
    assert server_contract["cuda_graph_bs_decode"] == [1, 2]
    assert server_contract["kt_hybrid_expert_plan_sha256"] == "a" * 64
    assert server_contract["kt_hybrid_expert_plan_format"] == (
        "sglang_kt_hybrid_expert_shard_v1"
    )
    assert server_contract["kt_hybrid_gpu_rank_counts_by_layer"] == [[28] * 43]
    assert server_contract["chunked_prefill_size"] == 1024
    assert server_contract["kt_num_gpu_experts"] == 28
    assert server_contract["dsv4_oscar_algorithm"] == "oscar-int2-asym-g64-v1"
    oscar_workers = cast(dict[str, Any], server_contract["oscar_pp_worker_contract"])
    assert oscar_workers["compressed_layer_union"] == list(range(2, 43))
    assert [
        worker["pp_rank"]
        for worker in cast(list[dict[str, Any]], oscar_workers["workers"])
    ] == [0, 1]
    assert oscar_workers["split_history"] == {
        "worker_identities": [[0, 0, 0, 0, 0], [0, 1, 0, 0, 1]],
        "workspace_bytes_per_worker": 4_210_688,
        "workspace_addresses_by_pp_rank": {"0": 90_000, "1": 90_001},
        "fixed_address": True,
    }
    assert configuration["benchmark_provenance"] == {
        "run_label": "fivefold-baseline",
        "expert_plan_path": "/tmp/pp2-fivefold.pt",
        "expert_plan_sha256": "a" * 64,
    }
    assert configuration["oscar_provenance"] == oscar_provenance().safe_receipt()
    native = cast(dict[str, Any], report["native_generate"])
    assert native["request_count"] == 2
    assert native["input_tokens_each"] == 2_694
    assert native["requested_max_output_tokens_each"] == 512
    assert native["ignore_eos"] is False
    assert native["aggregate_output_tokens"] == 360
    assert cast(float, native["aggregate_output_tokens_per_second"]) > 0
    assert cast(float, native["aggregate_decode_tokens_per_second"]) > 0
    assert cast(float, native["aggregate_requests_per_second"]) > 0
    assert cast(float, native["makespan_seconds"]) > 0
    assert cast(float, native["request_start_skew_seconds"]) < 0.1
    assert native["concurrent_start_confirmed"] is True
    assert native["concurrent_overlap_observed"] is True
    assert native["concurrent_output_overlap_observed"] is True
    assert native["semantic_success_count"] == 2
    assert native["all_semantically_valid"] is True
    assert native["all_naturally_terminated"] is True
    assert native["cross_lane_outputs_distinct"] is True
    assert native["all_decode_timing_valid"] is True
    assert native["all_streams_complete"] is True
    assert native["quality_gate_passed"] is True
    assert 0 < cast(float, native["decode_rate_jain_fairness"]) <= 1
    native_lanes = cast(list[dict[str, Any]], native["lanes"])
    assert [lane["lane"] for lane in native_lanes] == [0, 1]
    assert all(lane["completion_tokens"] == 180 for lane in native_lanes)
    assert all(lane["finish_reason"] == "stop" for lane in native_lanes)
    assert all(lane["output_hash_source"] == "output_ids" for lane in native_lanes)
    assert all(
        cast(dict[str, Any], lane["semantic_validation"])["passed"] is True
        for lane in native_lanes
    )

    chat = cast(dict[str, Any], report["openai_tool_calls"])
    assert chat["request_count"] == 2
    assert chat["structural_parser_success_count"] == 2
    assert chat["all_structurally_valid"] is True
    assert chat["all_completion_usage_valid"] is True
    assert chat["tool_arguments_distinct"] is True
    assert chat["concurrent_start_confirmed"] is True
    assert chat["concurrent_overlap_observed"] is True
    assert chat["concurrent_output_overlap_observed"] is True
    assert chat["quality_gate_passed"] is True
    assert cast(float, chat["aggregate_requests_per_second"]) > 0
    assert chat["aggregate_completion_tokens"] == 48
    chat_lanes = cast(list[dict[str, Any]], chat["lanes"])
    assert all(lane["structural_parser_success"] is True for lane in chat_lanes)
    assert all(lane["parser_issue_codes"] == [] for lane in chat_lanes)

    assert state.paths[0] == "/server_info"
    assert state.paths[1] == "/flush_cache"
    assert state.paths[2:4] == ["/generate", "/generate"]
    assert state.paths[4] == "/flush_cache"
    assert state.paths[5:7] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert set(state.native_payloads) == {0, 1}
    assert set(state.chat_payloads) == {0, 1}
    for payload in state.native_payloads.values():
        assert len(cast(list[int], payload["input_ids"])) == 2_694
        sampling = cast(dict[str, Any], payload["sampling_params"])
        assert sampling["max_new_tokens"] == 512
        assert sampling["ignore_eos"] is False
    for lane, payload in state.chat_payloads.items():
        assert payload["model"] == "test-dsv4-model"
        assert payload["parallel_tool_calls"] is False
        messages = cast(list[dict[str, str]], payload["messages"])
        assert [message["role"] for message in messages] == ["system", "user"]
        assert benchmark.TOOL_COMPONENTS[lane] in messages[1]["content"]
        tools = cast(list[dict[str, Any]], payload["tools"])
        function = cast(dict[str, Any], tools[0]["function"])
        assert function["name"] == benchmark.TOOL_NAME
    assert state.chat_payloads[0]["tools"] != state.chat_payloads[1]["tools"]

    encoded_report = json.dumps(report, sort_keys=True)
    assert "The controlled review selects" not in encoded_report
    assert "SECRET_UNEXPECTED_PROSE" not in encoded_report
    assert benchmark.TOOL_COMPONENTS[0] not in encoded_report


def test_live_concurrency_receipt_can_require_complete_nvlink_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState()
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))
    snapshots = iter((nvlink_counters(100), nvlink_counters(151)))

    report = benchmark.run_benchmark(
        benchmark.BenchmarkConfig(
            generate_url="http://fake-dsv4.invalid/generate",
            chat_url="http://fake-dsv4.invalid/v1/chat/completions",
            flush_url="http://fake-dsv4.invalid/flush_cache",
            model="test-dsv4-model",
            request_timeout_seconds=5.0,
            flush_timeout_seconds=5.0,
            chat_max_tokens=64,
            require_nvlink_traffic=True,
        ),
        native_prompts(),
        nvlink_snapshotter=lambda: next(snapshots),
    )

    assert report["ok"] is True
    assert report["performance_claim_eligible"] is False
    configuration = cast(dict[str, Any], report["configuration"])
    assert configuration["require_nvlink_traffic"] is True
    traffic = cast(dict[str, Any], report["nvlink_traffic"])
    assert len(cast(dict[str, int], traffic["counter_deltas"])) == 16
    assert set(cast(dict[str, int], traffic["counter_deltas"]).values()) == {51}


def test_serialized_output_windows_are_not_concurrency_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState(serialize_streams=True)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    report = benchmark.run_benchmark(benchmark_config(), native_prompts())

    assert report["ok"] is False
    assert report["performance_claim_eligible"] is False
    native = cast(dict[str, Any], report["native_generate"])
    assert native["concurrent_overlap_observed"] is True
    assert native["concurrent_output_overlap_observed"] is False
    assert native["quality_gate_passed"] is False
    chat = cast(dict[str, Any], report["openai_tool_calls"])
    assert chat["concurrent_overlap_observed"] is True
    assert chat["concurrent_output_overlap_observed"] is False
    assert chat["quality_gate_passed"] is False


@pytest.mark.parametrize(
    ("field_name", "bad_value", "issue_code"),
    (
        ("pp_size", 1, "pp_size_not_2"),
        ("tp_size", 2, "tp_size_not_1"),
        ("cuda_graph_backend_decode", "disabled", "decode_cuda_graph_backend_not_full"),
        ("max_total_tokens", 524_287, "max_total_tokens_not_524288"),
        (
            "cuda_graph_max_bs_decode",
            1,
            "decode_cuda_graph_max_batch_size_not_2",
        ),
        (
            "cuda_graph_bs_decode",
            [1],
            "decode_cuda_graph_batch_sizes_not_1_2",
        ),
        ("chunked_prefill_size", 512, "chunked_prefill_size_not_expected"),
        ("kt_num_gpu_experts", 26, "gpu_expert_count_not_expected"),
        (
            "dsv4_oscar_algorithm",
            "generic-int2",
            "oscar_static_dsv4_oscar_algorithm_mismatch",
        ),
        (
            "dsv4_kv_storage_mode",
            "fp8_e4m3",
            "oscar_static_dsv4_kv_storage_mode_mismatch",
        ),
        (
            "dsv4_oscar_int2_kv_storage",
            False,
            "oscar_static_dsv4_oscar_int2_kv_storage_mismatch",
        ),
        (
            "dsv4_int4_c4_indexer_storage",
            True,
            "dsv4_int4_c4_indexer_storage_not_disabled_for_oscar",
        ),
        (
            "dsv4_int4_kv_storage",
            True,
            "dsv4_int4_kv_storage_not_disabled_for_oscar",
        ),
        (
            "dsv4_sm86_c128_bf16_storage",
            True,
            "dsv4_sm86_c128_bf16_storage_not_disabled_for_oscar",
        ),
    ),
)
def test_server_contract_rejects_wrong_or_stale_live_configuration(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    bad_value: object,
    issue_code: str,
) -> None:
    state = FakeServerState()
    state.server_info[field_name] = bad_value
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    with pytest.raises(benchmark.BenchmarkError, match=issue_code):
        benchmark.run_benchmark(benchmark_config(), native_prompts())

    assert state.paths == ["/server_info"]


def test_server_contract_rejects_72_thread_oscar_transfer() -> None:
    server_info = pp2_server_info()
    server_info["kt_cpuinfer"] = 72

    with pytest.raises(
        benchmark.BenchmarkError,
        match="cpu_offload_threads_not_admitted",
    ):
        benchmark.validate_pp2_server_contract(
            server_info,
            expected_chunked_prefill_size=1024,
            expected_gpu_experts_per_layer=28,
            expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v1",
            expected_expert_plan_sha256="a" * 64,
            expected_cpuinfer_threads=56,
            expected_oscar_artifact_sha256=OSCAR_HASHES["dsv4_oscar_artifact_sha256"],
            expected_oscar_admission_sha256=OSCAR_HASHES["dsv4_oscar_admission_sha256"],
            expected_oscar_admission_receipt_sha256=OSCAR_HASHES[
                "dsv4_oscar_admission_receipt_sha256"
            ],
        )


@pytest.mark.parametrize(
    ("mutation", "issue_code"),
    (
        ("missing_stage", "oscar_pp_worker_records_not_exactly_2"),
        ("duplicate_pp_rank", "oscar_pp_worker_pp_rank_coverage_not_0_1"),
        ("hash_drift", "dsv4_oscar_artifact_sha256_not_top_level"),
        ("overlapping_layers", "oscar_pp_wo_a_stage_layers_overlap"),
        ("missing_layer", "oscar_pp_wo_a_layer_union_not_2_42"),
        (
            "non_oscar_worker",
            "dsv4_oscar_int2_kv_storage_mismatch",
        ),
        (
            "missing_split_workspace",
            "split_history_workspace_address_invalid",
        ),
        (
            "split_disabled_worker",
            "split_history_dsv4_oscar_int2_split_history_mismatch",
        ),
    ),
)
def test_server_contract_requires_two_exact_oscar_pp_workers(
    mutation: str,
    issue_code: str,
) -> None:
    server_info = pp2_server_info()
    internal_states = cast(list[dict[str, Any]], server_info["internal_states"])
    workers = cast(
        list[dict[str, Any]],
        internal_states[0]["dsv4_oscar_worker_telemetry_workers"],
    )
    if mutation == "missing_stage":
        workers.pop()
    elif mutation == "duplicate_pp_rank":
        workers[1]["pp_rank"] = 0
    elif mutation == "hash_drift":
        workers[1]["dsv4_oscar_artifact_sha256"] = "8" * 64
    elif mutation == "non_oscar_worker":
        workers[1]["dsv4_oscar_int2_kv_storage"] = False
    elif mutation == "missing_split_workspace":
        workers[1]["dsv4_oscar_int2_split_history_workspace_address"] = 0
    elif mutation == "split_disabled_worker":
        workers[1]["dsv4_oscar_int2_split_history"] = False
    else:
        absorption = cast(
            dict[str, Any], workers[1]["dsv4_oscar_wo_a_absorption_state"]
        )
        replacement = (
            list(range(20, 43))
            if mutation == "overlapping_layers"
            else list(range(22, 43))
        )
        for field in (
            "expected_local_compressed_layer_ids",
            "absorbed_local_layer_ids",
            "runtime_restore_skipped_layer_ids",
        ):
            absorption[field] = replacement

    with pytest.raises(benchmark.BenchmarkError, match=issue_code):
        benchmark.validate_pp2_server_contract(
            server_info,
            expected_chunked_prefill_size=1024,
            expected_gpu_experts_per_layer=28,
            expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v1",
            expected_expert_plan_sha256="a" * 64,
            expected_cpuinfer_threads=56,
        )


@pytest.mark.parametrize(
    ("prefix", "mutation", "issue_code"),
    (
        (
            "kt_single_numa_inline_dispatch",
            "identity",
            "pp2_cpu_inline_worker_identity_coverage_mismatch",
        ),
        (
            "kt_mxfp4_avx_scale_fold",
            "lut_hash",
            "pp2_cpu_scale_fold_worker_1_native_proof_mismatch",
        ),
    ),
)
def test_server_contract_requires_confirmed_cpu_tuple_on_pp_workers(
    prefix: str,
    mutation: str,
    issue_code: str,
) -> None:
    server_info = pp2_server_info()
    workers = cast(list[dict[str, Any]], server_info[f"{prefix}_worker_telemetry"])
    if mutation == "identity":
        workers[1]["gpu_id"] = 0
    else:
        telemetry = cast(dict[str, Any], workers[1]["telemetry"])
        telemetry["lut_hash"] = "0" * 16

    with pytest.raises(benchmark.BenchmarkError, match=issue_code):
        benchmark.validate_pp2_server_contract(
            server_info,
            expected_chunked_prefill_size=1024,
            expected_gpu_experts_per_layer=28,
            expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v1",
            expected_expert_plan_sha256="a" * 64,
            expected_cpuinfer_threads=56,
        )


def test_server_contract_binds_caller_plan_hash_to_loader_admitted_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState()
    state.server_info["kt_hybrid_expert_plan_sha256"] = "b" * 64
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))
    provenance = benchmark.BenchmarkProvenance(
        run_label="bound-plan",
        expert_plan_path="/tmp/plan.pt",
        expert_plan_sha256="a" * 64,
    )

    with pytest.raises(
        benchmark.BenchmarkError,
        match="loaded_expert_plan_sha256_not_expected",
    ):
        benchmark.run_benchmark(
            benchmark_config(), native_prompts(), provenance=provenance
        )

    assert state.paths == ["/server_info"]


def test_server_contract_accepts_hash_bound_variable_width_ep1_plan() -> None:
    server_info = pp2_server_info()
    counts = [28 + index % 8 for index in range(43)]
    server_info.update(
        {
            "pp_async_batch_depth": 1,
            "kt_num_gpu_experts": 35,
            "kt_gpu_expert_admission_ceiling": 35,
            "kt_hybrid_expert_plan_format": (
                "sglang_kt_hybrid_expert_shard_v2_variable"
            ),
            "kt_hybrid_gpu_rank_counts_by_layer": [counts],
            "kt_hybrid_min_gpu_experts_per_rank_per_layer": min(counts),
            "kt_hybrid_max_gpu_experts_per_rank_per_layer": max(counts),
            "kt_hybrid_total_gpu_expert_layers_by_rank": [sum(counts)],
        }
    )

    contract = benchmark.validate_pp2_server_contract(
        server_info,
        expected_chunked_prefill_size=1024,
        expected_gpu_experts_per_layer=35,
        expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v2_variable",
        expected_pp_async_batch_depth=1,
        expected_expert_plan_sha256="a" * 64,
    )

    assert contract["kt_gpu_expert_admission_ceiling"] == 35
    assert contract["kt_hybrid_gpu_rank_counts_by_layer"] == [counts]
    assert contract["kt_hybrid_min_gpu_experts_per_rank_per_layer"] == 28
    assert contract["kt_hybrid_max_gpu_experts_per_rank_per_layer"] == 35


@pytest.mark.parametrize(
    ("field_name", "bad_value", "issue_code"),
    (
        (
            "kt_hybrid_gpu_rank_counts_by_layer",
            [[28] * 42],
            "expert_plan_gpu_counts_not_ep1_x_43",
        ),
        (
            "kt_hybrid_min_gpu_experts_per_rank_per_layer",
            27,
            "expert_plan_minimum_gpu_count_mismatch",
        ),
        (
            "kt_gpu_expert_admission_ceiling",
            29,
            "gpu_expert_admission_ceiling_mismatch",
        ),
        (
            "kt_hybrid_expert_plan_format",
            "sglang_kt_hybrid_expert_shard_v0",
            "expert_plan_format_not_supported",
        ),
    ),
)
def test_server_contract_rejects_unbound_or_malformed_plan_geometry(
    field_name: str,
    bad_value: object,
    issue_code: str,
) -> None:
    server_info = pp2_server_info()
    server_info[field_name] = bad_value

    with pytest.raises(benchmark.BenchmarkError, match=issue_code):
        benchmark.validate_pp2_server_contract(
            server_info,
            expected_chunked_prefill_size=1024,
            expected_gpu_experts_per_layer=28,
            expected_expert_plan_format="sglang_kt_hybrid_expert_shard_v1",
            expected_expert_plan_sha256="a" * 64,
        )


def test_required_nvlink_traffic_fails_closed_when_any_link_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState()
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))
    before = nvlink_counters(100)
    after = nvlink_counters(151)
    after["gpu1.link3.rx_kib"] = before["gpu1.link3.rx_kib"]
    snapshots = iter((before, after))

    with pytest.raises(
        benchmark.BenchmarkError,
        match="did not prove complete NVLink traffic",
    ):
        benchmark.run_benchmark(
            benchmark.BenchmarkConfig(
                generate_url="http://fake-dsv4.invalid/generate",
                chat_url="http://fake-dsv4.invalid/v1/chat/completions",
                flush_url="http://fake-dsv4.invalid/flush_cache",
                model="test-dsv4-model",
                request_timeout_seconds=5.0,
                flush_timeout_seconds=5.0,
                chat_max_tokens=64,
                require_nvlink_traffic=True,
            ),
            native_prompts(),
            nvlink_snapshotter=lambda: next(snapshots),
        )


def test_benchmark_provenance_hashes_exact_absolute_plan(tmp_path: Path) -> None:
    plan = tmp_path / "pp2-fivefold.pt"
    content = b"content-addressed transferred plan"
    plan.write_bytes(content)

    provenance = benchmark.load_benchmark_provenance(
        run_label="fivefold-g13.baseline-1",
        expert_plan=plan,
    )

    assert provenance is not None
    assert provenance.run_label == "fivefold-g13.baseline-1"
    assert provenance.expert_plan_path == str(plan)
    assert provenance.expert_plan_sha256 == hashlib.sha256(content).hexdigest()


def test_oscar_provenance_rehashes_artifact_and_canonical_admission(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "calibration.pt"
    artifact.write_bytes(b"admitted OSCAR calibration")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt: dict[str, object] = {
        "format": "dsv4-oscar-int2-admission",
        "format_version": 1,
        "admitted": True,
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "artifact_path": str(artifact),
        "artifact_file_sha256": artifact_sha256,
        "artifact_provenance_sha256": "3" * 64,
        "checkpoint_path": "/tmp/dsv4-local-checkpoint-0731",
        "checkpoint_sha256": "4" * 64,
        "config_sha256": "2" * 64,
        "checkpoint_fingerprint_path": str(tmp_path / "fingerprint.json"),
        "checkpoint_fingerprint_sha256": "5" * 64,
        "validation_policy": "rehash-config-index-and-all-referenced-shards-v1",
    }
    admission_sha256 = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt["admission_sha256"] = admission_sha256
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    provenance = benchmark.load_oscar_provenance(receipt_path)

    assert provenance is not None
    assert provenance.artifact_path == str(artifact)
    assert provenance.artifact_sha256 == artifact_sha256
    assert provenance.admission_sha256 == admission_sha256
    assert (
        provenance.admission_receipt_sha256
        == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest does not match admission"):
        benchmark.load_oscar_provenance(receipt_path)


@pytest.mark.parametrize(
    ("run_label", "supply_plan"),
    (("baseline", False), (None, True)),
)
def test_benchmark_provenance_requires_label_and_plan_together(
    tmp_path: Path,
    run_label: str | None,
    supply_plan: bool,
) -> None:
    plan = tmp_path / "plan.pt"
    plan.write_bytes(b"plan")

    with pytest.raises(ValueError, match="must be supplied together"):
        benchmark.load_benchmark_provenance(
            run_label=run_label,
            expert_plan=plan if supply_plan else None,
        )


def test_benchmark_provenance_rejects_symlink_and_unsafe_label(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.pt"
    plan.write_bytes(b"plan")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(plan)

    with pytest.raises(ValueError, match="run label"):
        benchmark.load_benchmark_provenance(
            run_label="unsafe label",
            expert_plan=plan,
        )
    with pytest.raises(ValueError, match="must not be a symlink"):
        benchmark.load_benchmark_provenance(
            run_label="baseline",
            expert_plan=alias,
        )


def test_parse_args_accepts_receipt_provenance_pair(tmp_path: Path) -> None:
    plan = tmp_path / "plan.pt"
    parsed = benchmark.parse_args(
        [
            "--run-label",
            "fivefold-optimized",
            "--expert-plan",
            str(plan),
            "--require-nvlink-traffic",
            "--expected-chunked-prefill-size",
            "512",
            "--expected-gpu-experts-per-layer",
            "28",
            "--expected-expert-plan-format",
            "sglang_kt_hybrid_expert_shard_v2_variable",
            "--expected-pp-async-batch-depth",
            "1",
            "--expected-cpuinfer-threads",
            "56",
            "--oscar-admission-receipt",
            str(tmp_path / "admission.json"),
            "--launch-authorization-receipt",
            str(tmp_path / "authorization.json"),
        ]
    )

    assert parsed.run_label == "fivefold-optimized"
    assert parsed.expert_plan == plan
    assert parsed.require_nvlink_traffic is True
    assert parsed.expected_chunked_prefill_size == 512
    assert parsed.expected_gpu_experts_per_layer == 28
    assert parsed.expected_expert_plan_format == (
        "sglang_kt_hybrid_expert_shard_v2_variable"
    )
    assert parsed.expected_pp_async_batch_depth == 1
    assert parsed.expected_cpuinfer_threads == 56
    assert parsed.oscar_admission_receipt == tmp_path / "admission.json"
    assert parsed.launch_authorization_receipt == tmp_path / "authorization.json"


def test_load_launch_authorization_proves_receipt_bound_oscar_tuple(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "authorization.json"
    receipt.write_text(
        json.dumps(
            {
                "format": "dsv4_pp2_model_launch_authorization_v1",
                "ordinal": 1,
                "run_role": "transfer",
                "ep_winner": {
                    "receipt_sha256": "8" * 64,
                    "coherency_receipt_sha256": "6" * 64,
                },
                "configuration": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 2,
                    "expert_parallel_size": 1,
                    "context_length": 524_288,
                    "max_total_tokens": 524_288,
                    "decode_cuda_graph_backend": "full",
                    "kv_cache_public_carrier": "fp8_e4m3",
                    "physical_kv_cache_storage": "oscar-int2-asymmetric",
                    "oscar_split_history": True,
                    "oscar_split_history_execution": (
                        "sm86-oscar-int2-split-history-fp32-online-v1"
                    ),
                    "oscar_split_history_split_map": (
                        benchmark.OSCAR_SPLIT_HISTORY_SERVER_INFO[
                            "dsv4_oscar_int2_split_history_split_map"
                        ]
                    ),
                    "oscar_split_history_workspace_bytes_per_worker": 4_210_688,
                    "oscar_split_history_worker_identities": [
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1],
                    ],
                    "transferred_plan_sha256": "7" * 64,
                    "native_artifact_sha256": (
                        "7886a0e7cde36263ac57005aea572fd99"
                        "a401a8b3107f60d949d0dff97292043"
                    ),
                    "cpuinfer_threads": 56,
                    "worker_spin_us": 1000,
                    "task_queue_pin_first_core": True,
                    "single_numa_inline_dispatch": True,
                    "scale_fold_mode": "lut-v1",
                    "scale_fold_n_block": 128,
                    "scale_fold_lut_hash": "06d1a83dbf20f545",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = benchmark.load_launch_authorization_provenance(receipt)

    assert provenance.ordinal == 1
    assert provenance.run_role == "transfer"
    assert provenance.ep_confirmation_receipt_sha256 == "8" * 64
    assert provenance.ep_coherency_receipt_sha256 == "6" * 64
    assert (
        provenance.authorization_receipt_sha256
        == hashlib.sha256(receipt.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "removed_option",
    (
        "--expected-int4-c4-indexer-storage",
        "--expected-int4-kv-storage",
        "--expected-c128-bf16-storage",
    ),
)
def test_parse_args_rejects_non_oscar_storage_options(
    tmp_path: Path, removed_option: str
) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--oscar-admission-receipt",
                str(tmp_path / "admission.json"),
                "--launch-authorization-receipt",
                str(tmp_path / "authorization.json"),
                removed_option,
                "enabled",
            ]
        )


def test_parse_args_requires_oscar_admission_receipt() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([])


def test_parse_args_requires_launch_authorization_receipt() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--oscar-admission-receipt", "/tmp/oscar-admission.json"])


def test_parse_args_rejects_72_cpuinfer_threads() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--expected-cpuinfer-threads", "72"])


def test_parse_args_defaults_to_the_local_checkpoint_copy() -> None:
    parsed = benchmark.parse_args(
        [
            "--oscar-admission-receipt",
            "/tmp/oscar-admission.json",
            "--launch-authorization-receipt",
            "/tmp/pp2-authorization.json",
        ]
    )

    assert parsed.model_path == Path("/tmp/dsv4-local-checkpoint-0731")


def test_exact_prompt_builder_uses_varied_context_and_preserves_task() -> None:
    class StableWordTokenizer:
        def __init__(self) -> None:
            self.token_ids: dict[str, int] = {}
            self.seen_text: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            self.seen_text.append(text)
            tokens = re.findall(r"\S+", text)
            return [
                self.token_ids.setdefault(token, len(self.token_ids) + 1)
                for token in tokens
            ]

    def encode_messages(messages: list[dict[str, str]], *, thinking_mode: str) -> str:
        assert thinking_mode == "chat"
        return messages[0]["content"]

    tokenizer = StableWordTokenizer()
    lane_zero = benchmark.build_exact_prompt_ids(
        tokenizer, encode_messages, lane=0, target_tokens=320
    )
    lane_one = benchmark.build_exact_prompt_ids(
        tokenizer, encode_messages, lane=1, target_tokens=320
    )

    assert len(lane_zero) == 320
    assert len(lane_one) == 320
    assert lane_zero != lane_one
    combined_context = "\n".join(tokenizer.seen_text)
    assert "Inventory item R0-" in combined_context
    assert "Inventory item R1-" in combined_context
    assert "ORCHID-17" in combined_context
    assert "COBALT-29" in combined_context
    assert " a a a" not in combined_context


def test_filler_output_is_ineligible_even_when_stream_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState(degenerate_native_lane=1)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    report = benchmark.run_benchmark(benchmark_config(), native_prompts())

    assert report["ok"] is False
    assert report["performance_claim_eligible"] is False
    native = cast(dict[str, Any], report["native_generate"])
    assert native["quality_gate_passed"] is False
    assert native["all_semantically_valid"] is False
    lane = cast(list[dict[str, Any]], native["lanes"])[1]
    semantic = cast(dict[str, Any], lane["semantic_validation"])
    assert semantic["passed"] is False
    issues = cast(list[str], semantic["issue_codes"])
    assert "dominant_repeated_word" in issues
    assert "dominant_repeated_token" in issues
    assert "repeated_token_run" in issues


def test_decoded_output_ids_are_authoritative_for_semantic_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState()
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    report = benchmark.run_benchmark(
        benchmark_config(),
        native_prompts(),
        decode_output_ids=lambda _unused_ids: " a" * 180,
    )

    assert report["ok"] is False
    assert report["performance_claim_eligible"] is False
    native = cast(dict[str, Any], report["native_generate"])
    lanes = cast(list[dict[str, Any]], native["lanes"])
    assert all(lane["semantic_text_source"] == "decoded_output_ids" for lane in lanes)
    assert all(
        cast(dict[str, Any], lane["semantic_validation"])["passed"] is False
        for lane in lanes
    )


@pytest.mark.parametrize(
    ("fault_field", "issue_code"),
    (
        ("wrong_tool_arguments_lane", "tool_arguments_schema_mismatch"),
        ("unexpected_tool_content_lane", "unexpected_prose_content"),
    ),
)
def test_tool_call_semantic_or_content_failure_is_ineligible(
    monkeypatch: pytest.MonkeyPatch,
    fault_field: str,
    issue_code: str,
) -> None:
    state = FakeServerState()
    setattr(state, fault_field, 0)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    report = benchmark.run_benchmark(benchmark_config(), native_prompts())

    assert report["ok"] is False
    assert report["performance_claim_eligible"] is False
    chat = cast(dict[str, Any], report["openai_tool_calls"])
    assert chat["quality_gate_passed"] is False
    lane = cast(list[dict[str, Any]], chat["lanes"])[0]
    assert issue_code in cast(list[str], lane["parser_issue_codes"])


def test_empty_native_stream_fails_closed_without_output_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState(empty_native_lane=0)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    with pytest.raises(benchmark.BenchmarkError, match="lane 0 failed"):
        benchmark.run_benchmark(benchmark_config(), native_prompts())


def test_empty_chat_stream_fails_closed_without_unbound_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FakeServerState(empty_chat_lane=0)
    monkeypatch.setattr(benchmark.urllib.request, "urlopen", FakeUrlOpen(state))

    report = benchmark.run_benchmark(benchmark_config(), native_prompts())

    assert report["ok"] is False
    assert report["performance_claim_eligible"] is False
    chat = cast(dict[str, Any], report["openai_tool_calls"])
    assert chat["concurrent_output_overlap_observed"] is False
    assert chat["quality_gate_passed"] is False


def test_metrics_and_tool_structure_reject_wrong_results() -> None:
    assert benchmark.jains_fairness([10.0, 20.0]) == pytest.approx(0.9)
    assert benchmark.jains_fairness([0.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        benchmark.jains_fairness([1.0, -1.0])

    calls = {
        0: benchmark.ToolCallParts(
            identifier="call_0",
            kind="function",
            name_fragments=[benchmark.TOOL_NAME],
            argument_fragments=['{"lane":1,"marker":"wrong"}'],
        )
    }
    success, issues, arguments = benchmark._validate_tool_structure(
        lane=0,
        tool_calls=calls,
        finish_reason="tool_calls",
        saw_done=True,
        issue_codes=set(),
    )

    assert success is False
    assert issues == ("tool_arguments_schema_mismatch",)
    assert len(hashlib.sha256(arguments.encode("utf-8")).hexdigest()) == 64


def test_semantic_assessment_rejects_prompt_copy_and_repetition() -> None:
    good_text = coherent_native_text(0)
    good_ids = list(range(1_000, 1_180))
    good = benchmark.assess_native_output(
        0,
        good_text,
        prompt_ids=[11] * benchmark.INPUT_TOKEN_COUNT,
        output_ids=good_ids,
    )
    assert good.passed is True

    copied_prompt = list(range(900, 900 + benchmark.INPUT_TOKEN_COUNT))
    copied_ids = copied_prompt[:180]
    copied = benchmark.assess_native_output(
        0,
        good_text,
        prompt_ids=copied_prompt,
        output_ids=copied_ids,
    )
    assert copied.passed is False
    assert "prompt_copying" in copied.issue_codes
