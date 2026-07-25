from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import run_sglang_kt_glm47_tp2_local_diagnostic as tp2
from scripts import run_sglang_kt_glm52_tp2_local_benchmark as benchmark


def make_config(tmp_path: Path) -> benchmark.BenchmarkConfig:
    return benchmark.BenchmarkConfig(
        run_id="focused-test",
        result_directory=tmp_path / "result",
        runtime_python="/runtime/python",
        runtime_install_receipt=tmp_path / "install-receipt.json",
        runtime_install_receipt_sha256="0" * 64,
        local_source_directory="/runtime/source",
        model_path="/mnt/sanic/glm52",
        ktransformers_weight_path="/mnt/sanic/glm52-AMXINT4",
        dwagon_ip=benchmark.DEFAULT_DWAGON_IP,
        dwagon_socket_interface=benchmark.DEFAULT_DWAGON_SOCKET_INTERFACE,
        distributed_port=benchmark.DEFAULT_DISTRIBUTED_PORT,
        service_port=benchmark.DEFAULT_SERVICE_PORT,
        benchmark_concurrencies=benchmark.DEFAULT_BENCHMARK_CONCURRENCIES,
        context_length=benchmark.DEFAULT_CONTEXT_LENGTH,
        maximum_total_tokens=benchmark.DEFAULT_MAXIMUM_TOTAL_TOKENS,
        benchmark_input_tokens=benchmark.DEFAULT_BENCHMARK_INPUT_TOKENS,
        benchmark_output_tokens=benchmark.DEFAULT_BENCHMARK_OUTPUT_TOKENS,
        chunked_prefill_size=2_048,
        resident_gpu_expert_budget_total=0,
        kt_gpu_experts_ratio=None,
        expert_placement_strategy="uniform",
        init_expert_location=None,
        init_expert_location_sha256=None,
        kv_cache_dtype=benchmark.DEFAULT_KV_CACHE_DTYPE,
        enable_two_batch_overlap=False,
        enable_amx_fine_grained_decode=False,
        enable_stream_prefill=False,
        stream_prefill_token_threshold=4_096,
        stream_prefill_experts_per_chunk=4,
        enable_mtp=False,
        enable_shared_host_weights=False,
        shared_host_weights_manifest=None,
        shared_host_weights_content_id=None,
        shared_host_weights_state_directory=None,
        capture_representative_routing=False,
        static_memory_fraction=benchmark.DEFAULT_STATIC_MEMORY_FRACTION,
        minimum_prelaunch_free_vram_mib=(
            benchmark.DEFAULT_MINIMUM_PRELAUNCH_FREE_VRAM_MIB
        ),
        minimum_postreadiness_free_vram_mib=(
            benchmark.DEFAULT_MINIMUM_POSTREADINESS_FREE_VRAM_MIB
        ),
        readiness_timeout_seconds=10.0,
        request_timeout_seconds=10.0,
        cleanup_timeout_seconds=10.0,
    )


def make_snapshot(
    *,
    free_mib: int = 24_000,
    compute_processes: tuple[benchmark.GpuComputeProcess, ...] = (),
    rank_zero_pci_bus_id: str = "00000000:16:00.0",
) -> benchmark.GpuCapacitySnapshot:
    return benchmark.GpuCapacitySnapshot(
        observed_at_utc="2026-07-25T00:00:00.000000+00:00",
        inventory_command=("nvidia-smi",),
        inventory_stdout_sha256="1" * 64,
        compute_process_command=("nvidia-smi",),
        compute_process_stdout_sha256="2" * 64,
        devices=(
            benchmark.GpuMemoryDevice(
                uuid=benchmark.ORDERED_GPU_UUIDS[0],
                index=0,
                pci_bus_id=rank_zero_pci_bus_id,
                compute_capability="8.6",
                total_mib=24_576,
                free_mib=free_mib,
                used_mib=24_576 - free_mib,
            ),
            benchmark.GpuMemoryDevice(
                uuid=benchmark.ORDERED_GPU_UUIDS[1],
                index=1,
                pci_bus_id="00000000:d8:00.0",
                compute_capability="8.6",
                total_mib=24_576,
                free_mib=free_mib,
                used_mib=24_576 - free_mib,
            ),
        ),
        compute_processes=compute_processes,
    )


def make_server_info(
    config: benchmark.BenchmarkConfig,
    spec: benchmark.Glm52Tp2ProcessSpec,
) -> benchmark.JsonObject:
    return {
        "version": "installed-version",
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
        "kt_num_gpu_experts": 0,
        "kt_gpu_experts_ratio": None,
        "kt_max_deferred_experts_per_token": 0,
        "kt_expert_placement_strategy": "uniform",
        "init_expert_location": "trivial",
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
        "served_model_name": "GLM5.2",
        "tool_call_parser": "glm47",
        "reasoning_parser": "glm45",
        "trust_remote_code": True,
        "internal_states": [
            {
                "effective_max_running_requests_per_dp": (
                    spec.maximum_running_requests
                ),
                "max_total_tokens": config.maximum_total_tokens,
                "pp_max_micro_batch_size": spec.maximum_running_requests,
            }
        ],
    }


def test_process_spec_pins_fresh_topology_and_low_capacity(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = benchmark.build_process_spec(config)

    assert spec.ordered_gpu_uuids == (
        "GPU-a442b72e-6727-6322-ba5d-5a9512b79886",
        "GPU-63a7760a-6164-0758-9228-03dbf35d721c",
    )
    assert spec.cpu_cores == tuple(range(112))
    assert spec.memory_nodes == (0, 1)
    assert spec.environment[0] == (
        "CUDA_VISIBLE_DEVICES",
        ",".join(benchmark.ORDERED_GPU_UUIDS),
    )
    command = spec.command
    expected_pairs = {
        "--kt-method": "AMXINT4",
        "--kt-cpuinfer": "112",
        "--kt-threadpool-count": "2",
        "--kt-num-gpu-experts": "0",
        "--kt-max-deferred-experts-per-token": "0",
        "--pp-size": "1",
        "--tp-size": "2",
        "--max-total-tokens": "16000",
        "--max-running-requests": "2",
        "--kv-cache-dtype": "bfloat16",
    }
    for flag, expected_value in expected_pairs.items():
        assert command[command.index(flag) + 1] == expected_value
    numa_index = command.index("--kt-numa-nodes")
    assert command[numa_index + 1 : numa_index + 3] == ("0", "1")
    assert "--disable-radix-cache" in command
    assert "--disable-custom-all-reduce" in command
    assert "--enable-two-batch-overlap" not in command
    assert ("KT_AMX_FINE_GRAINED_DECODE", "1") not in spec.environment

    receipt = spec.receipt()
    assert receipt["model"]["kv_cache_dtype"] == "bfloat16"


def test_process_spec_enables_exact_paper_overlap_contract(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        enable_two_batch_overlap=True,
        enable_amx_fine_grained_decode=True,
    )
    spec = benchmark.build_process_spec(config)
    receipt = spec.receipt()

    assert "--enable-two-batch-overlap" in spec.command
    assert ("KT_AMX_FINE_GRAINED_DECODE", "1") in spec.environment
    assert receipt["paper_optimizations"] == {
        "two_batch_attention_moe_overlap": True,
        "fine_grained_amx_decode_dependencies": True,
        "bounded_stream_loading_prefill": False,
        "kt_backed_mtp": False,
        "zero_copy_shared_host_weights": False,
        "representative_route_capture": False,
    }
    assert receipt["gpu_workers"] == [
        {
            "tensor_parallel_rank": 0,
            "gpu_uuid": benchmark.ORDERED_GPU_UUIDS[0],
            "expected_pci_bus_suffix": ":16:00.0",
            "matching_numa_node": 0,
            "numa_local_physical_cpu_ids": list(range(56)),
        },
        {
            "tensor_parallel_rank": 1,
            "gpu_uuid": benchmark.ORDERED_GPU_UUIDS[1],
            "expected_pci_bus_suffix": ":d8:00.0",
            "matching_numa_node": 1,
            "numa_local_physical_cpu_ids": list(range(56, 112)),
        },
    ]


def test_process_spec_enables_stream_mtp_and_shared_mapping_contract(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "weights-manifest.json"
    state = tmp_path / "state"
    config = replace(
        make_config(tmp_path),
        enable_stream_prefill=True,
        enable_mtp=True,
        enable_shared_host_weights=True,
        shared_host_weights_manifest=manifest,
        shared_host_weights_content_id="a" * 64,
        shared_host_weights_state_directory=state,
    )
    spec = benchmark.build_process_spec(config)

    assert "--kt-stream-prefill" in spec.command
    assert (
        spec.command[spec.command.index("--kt-gpu-prefill-token-threshold") + 1]
        == "4096"
    )
    assert "--speculative-algorithm" in spec.command
    assert (
        spec.command[spec.command.index("--speculative-draft-model-path") + 1]
        == config.model_path
    )
    environment = dict(spec.environment)
    assert environment["KT_SHARED_HOST_WEIGHTS"] == "1"
    assert environment["KT_SHARED_HOST_WEIGHTS_MANIFEST"] == str(manifest)
    assert environment["KT_SHARED_HOST_WEIGHTS_CONTENT_ID"] == "a" * 64
    assert environment["KT_SHARED_HOST_WEIGHTS_STATE_DIR"] == str(state)
    assert spec.receipt()["paper_optimizations"] == {
        "two_batch_attention_moe_overlap": False,
        "fine_grained_amx_decode_dependencies": False,
        "bounded_stream_loading_prefill": True,
        "kt_backed_mtp": True,
        "zero_copy_shared_host_weights": True,
        "representative_route_capture": False,
    }


def test_speculative_stream_metrics_preserve_pinned_fields_and_timing() -> None:
    metrics = benchmark.parse_speculative_decoding_metrics(
        {
            "prompt_tokens": 7_744,
            "spec_accept_rate": 0.75,
            "spec_accept_length": 1.6,
            "spec_accept_token_num": 96,
            "spec_draft_token_num": 128,
            "spec_verify_ct": 80,
            "spec_accept_histogram": [10, 50, 20],
            "spec_target_verify_latency_ms": 7.5,
            "unrelated_server_field": "ignored",
        }
    )

    assert metrics is not None
    assert metrics.receipt() == {
        "spec_accept_rate": 0.75,
        "spec_accept_length": 1.6,
        "spec_accept_token_num": 96,
        "spec_draft_token_num": 128,
        "spec_verify_ct": 80,
        "spec_accept_histogram": [10, 50, 20],
        "target_verification_timings": {
            "spec_target_verify_latency_ms": 7.5,
        },
    }


def test_speculative_stream_metrics_merge_cumulative_events_per_request() -> None:
    recorder = benchmark.SpeculativeMetricsRecorder(concurrency=2)
    recorder.observe(
        0,
        {
            "spec_accept_rate": 0.5,
            "spec_accept_token_num": 1,
            "spec_draft_token_num": 2,
        },
    )
    recorder.observe(1, {"completion_tokens": 1})
    recorder.observe(
        0,
        {
            "spec_accept_rate": 0.75,
            "spec_accept_length": 1.5,
            "spec_accept_token_num": 3,
            "spec_draft_token_num": 4,
            "spec_verify_ct": 2,
            "spec_accept_histogram": [0, 1, 1],
        },
    )

    assert recorder.observation().receipt() == {
        "concurrency": 2,
        "requests": [
            {
                "request_index": 0,
                "stream_events_with_metrics": 2,
                "metrics": {
                    "spec_accept_rate": 0.75,
                    "spec_accept_length": 1.5,
                    "spec_accept_token_num": 3,
                    "spec_draft_token_num": 4,
                    "spec_verify_ct": 2,
                    "spec_accept_histogram": [0, 1, 1],
                    "target_verification_timings": {},
                },
            },
            {
                "request_index": 1,
                "stream_events_with_metrics": 0,
                "metrics": None,
            },
        ],
    }


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("spec_accept_rate", 1.1),
        ("spec_accept_length", float("nan")),
        ("spec_accept_token_num", True),
        ("spec_draft_token_num", -1),
        ("spec_verify_ct", 2.0),
        ("spec_accept_histogram", [0, False]),
        ("target_verify_time_ms", "fast"),
    ),
)
def test_speculative_stream_metrics_fail_closed_only_for_present_malformed_fields(
    field_name: str,
    value: benchmark.JsonValue,
) -> None:
    with pytest.raises(benchmark.Glm52Tp2BenchmarkError, match=field_name):
        benchmark.parse_speculative_decoding_metrics({field_name: value})

    assert (
        benchmark.parse_speculative_decoding_metrics(
            {"unrelated_optional_field": value}
        )
        is None
    )


def test_non_mtp_concurrency_path_does_not_install_metadata_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sentinel = cast(
        benchmark.glm52.BenchmarkCaseObservation,
        SimpleNamespace(concurrency=1),
    )

    def fake_run_concurrency_case(
        *_arguments: object,
        **keyword_arguments: object,
    ) -> benchmark.glm52.BenchmarkCaseObservation:
        calls.append(keyword_arguments)
        raw_observer = keyword_arguments.get("stream_meta_info_observer")
        if raw_observer is not None:
            observer = cast(
                Callable[[int, Mapping[str, benchmark.JsonValue]], None],
                raw_observer,
            )
            observer(0, {"spec_verify_ct": 4})
        return sentinel

    monkeypatch.setattr(
        benchmark.glm52,
        "run_concurrency_case",
        fake_run_concurrency_case,
    )
    client = cast(benchmark.httpx.Client, object())
    tokenizer = cast(benchmark.glm52.BenchmarkTokenizer, object())
    prompts = cast(tuple[benchmark.glm52.PreparedPrompt, ...], (object(),))

    assert benchmark._run_concurrency_case(client, tokenizer, prompts, 8, None) is (
        sentinel
    )
    assert calls == [{}]

    recorder = benchmark.SpeculativeMetricsRecorder(concurrency=1)
    assert (
        benchmark._run_concurrency_case(
            client,
            tokenizer,
            prompts,
            8,
            recorder,
        )
        is sentinel
    )
    assert set(calls[1]) == {"stream_meta_info_observer"}
    assert recorder.observation().requests[0].metrics is not None
    assert benchmark._speculative_decoding_receipt(False, [recorder]) == {}
    assert benchmark._speculative_decoding_receipt(True, [recorder]) == {
        "speculative_decoding_metrics": {
            "schema_version": 1,
            "source": "native_generate_sse_meta_info",
            "cases": [
                {
                    "concurrency": 1,
                    "requests": [
                        {
                            "request_index": 0,
                            "stream_events_with_metrics": 1,
                            "metrics": {
                                "spec_accept_rate": None,
                                "spec_accept_length": None,
                                "spec_accept_token_num": None,
                                "spec_draft_token_num": None,
                                "spec_verify_ct": 4,
                                "spec_accept_histogram": None,
                                "target_verification_timings": {},
                            },
                        }
                    ],
                }
            ],
        }
    }


def test_shared_host_weight_prefault_is_manifest_bound_and_numa_local(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensor_header = json.dumps(
        {
            "blk.3.ffn_down_exps.0.numa.0.weight": {
                "data_offsets": [0, 4],
                "dtype": "U8",
                "shape": [4],
            },
            "blk.3.ffn_down_exps.0.numa.1.weight": {
                "data_offsets": [4, 8],
                "dtype": "U8",
                "shape": [4],
            },
        },
        separators=(",", ":"),
    ).encode()
    tensor_header += b" " * (-len(tensor_header) % 8)
    safetensors_contents = (
        len(tensor_header).to_bytes(8, byteorder="little") + tensor_header + b"00001111"
    )
    file_contents = {
        "config.json": b'{"model_type":"glm_moe_dsa"}\n',
        "model-00001-of-00001.safetensors": safetensors_contents,
    }
    rows: list[benchmark.JsonValue] = []
    for relative_path in sorted(file_contents):
        contents = file_contents[relative_path]
        path = checkpoint / relative_path
        path.write_bytes(contents)
        path.chmod(0o400)
        rows.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size_bytes": len(contents),
            }
        )
    content_id = benchmark._canonical_sha256(
        {
            "files": rows,
            "kind": "kt_shared_host_weights_content",
            "schema_version": 1,
        }
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "content_id": content_id,
                "files": rows,
                "kind": "kt_shared_host_weights_manifest",
                "numa_nodes": [0, 1],
                "schema_version": 1,
            }
        )
    )
    config = replace(
        make_config(tmp_path),
        ktransformers_weight_path=str(checkpoint),
        enable_shared_host_weights=True,
        shared_host_weights_manifest=manifest,
        shared_host_weights_content_id=content_id,
        shared_host_weights_state_directory=tmp_path / "state",
    )
    observed_commands: list[tuple[str, ...]] = []

    def reader(
        command: tuple[str, ...],
        _timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        observed_commands.append(command)
        node = int(command[command.index("--numa-node") + 1])
        plan_path = Path(command[command.index("--plan") + 1])
        plan = cast(dict[str, object], json.loads(plan_path.read_bytes()))
        node_rows = cast(list[dict[str, object]], plan["nodes"])
        node_row = next(row for row in node_rows if row["numa_node"] == node)
        worker_receipt = {
            "schema_version": 1,
            "kind": "glm52_shared_host_weight_numa_prefault_result",
            "status": "completed",
            "numa_node": node,
            "file_count": node_row["file_count"],
            "extent_count": node_row["extent_count"],
            "tensor_count": node_row["tensor_count"],
            "completed_bytes": node_row["expected_bytes"],
            "elapsed_seconds": 0.01,
            "affinity_cpu_count": 1,
            "readahead_policy": "POSIX_FADV_RANDOM",
        }
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(worker_receipt),
            "",
        )

    config.result_directory.mkdir()
    receipt = benchmark.prefault_shared_host_weights(config, reader)

    assert receipt["completed_bytes"] == 8
    assert receipt["artifact_file_count"] == 2
    assert receipt["selected_file_count"] == 1
    assert receipt["worker_count"] == 2
    assert len(observed_commands) == 2
    assert {
        command[command.index("--numa-node") + 1] for command in observed_commands
    } == {"0", "1"}
    assert any("--membind=0" in command for command in observed_commands)
    assert any("--membind=1" in command for command in observed_commands)
    plan_receipt = cast(dict[str, object], receipt["prefault_plan"])
    direct_worker = benchmark._execute_prefault_worker_plan(
        plan_path=Path(cast(str, plan_receipt["path"])),
        expected_plan_sha256=cast(str, plan_receipt["sha256"]),
        numa_node=0,
    )
    assert direct_worker["completed_bytes"] == 4


def test_parser_defaults_to_exact_c1_c2_budget(tmp_path: Path) -> None:
    arguments = benchmark._parser().parse_args(
        [
            "--run-id",
            "default-capacity",
            "--result-directory",
            str(tmp_path / "result"),
        ]
    )
    config = benchmark._config_from_arguments(arguments)
    assert config.benchmark_concurrencies == (1, 2)
    assert (
        config.maximum_total_tokens
        == 2 * (7_744 + 128) + benchmark.DEFAULT_SCHEDULER_TOKEN_HEADROOM
        == 16_000
    )

    arguments.maximum_total_tokens = 20_000
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="must exactly fit",
    ):
        benchmark._config_from_arguments(arguments)


def test_parser_admits_exact_c1_only_budget(tmp_path: Path) -> None:
    arguments = benchmark._parser().parse_args(
        [
            "--run-id",
            "c1-capacity",
            "--result-directory",
            str(tmp_path / "result"),
            "--benchmark-concurrencies",
            "1",
        ]
    )

    config = benchmark._config_from_arguments(arguments)
    spec = benchmark.build_process_spec(config)

    assert config.benchmark_concurrencies == (1,)
    assert config.maximum_total_tokens == (
        7_744 + 128 + benchmark.DEFAULT_SCHEDULER_TOKEN_HEADROOM
    )
    assert spec.maximum_running_requests == 1
    assert spec.command[spec.command.index("--max-running-requests") + 1] == "1"
    assert spec.command[spec.command.index("--max-total-tokens") + 1] == str(
        config.maximum_total_tokens
    )
    server_info = make_server_info(config, spec)
    capacity = benchmark.validate_server_capacity(server_info, config, spec)
    assert capacity["scheduler_state"] == {
        "effective_max_running_requests_per_dp": 1,
        "max_total_tokens": config.maximum_total_tokens,
        "pp_max_micro_batch_size": 1,
    }


@pytest.mark.parametrize(
    "concurrencies",
    (("2", "1"), ("1", "1")),
)
def test_parser_rejects_unordered_or_duplicate_concurrencies(
    tmp_path: Path,
    concurrencies: tuple[str, ...],
) -> None:
    arguments = benchmark._parser().parse_args(
        [
            "--run-id",
            "invalid-capacity",
            "--result-directory",
            str(tmp_path / "result"),
            "--benchmark-concurrencies",
            *concurrencies,
        ]
    )

    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="ordered unique",
    ):
        benchmark._config_from_arguments(arguments)


def test_concurrency_admission_rejects_serialized_server_requests() -> None:
    concurrent_case = SimpleNamespace(
        concurrency=2,
        requests=(
            SimpleNamespace(
                request_index=0,
                observation=SimpleNamespace(
                    ttft_seconds=20.0,
                    end_to_end_seconds=100.0,
                ),
            ),
            SimpleNamespace(
                request_index=1,
                observation=SimpleNamespace(
                    ttft_seconds=30.0,
                    end_to_end_seconds=110.0,
                ),
            ),
        ),
    )
    evidence = benchmark.validate_concurrent_generation_admission(
        cast(benchmark.glm52.BenchmarkCaseObservation, concurrent_case)
    )
    assert evidence["overlap_seconds"] == 70.0

    serialized_case = SimpleNamespace(
        concurrency=2,
        requests=(
            concurrent_case.requests[0],
            SimpleNamespace(
                request_index=1,
                observation=SimpleNamespace(
                    ttft_seconds=120.0,
                    end_to_end_seconds=200.0,
                ),
            ),
        ),
    )
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="not server-resident together",
    ):
        benchmark.validate_concurrent_generation_admission(
            cast(benchmark.glm52.BenchmarkCaseObservation, serialized_case)
        )


def test_gpu_snapshot_parser_binds_inventory_and_compute_processes() -> None:
    inventory = "\n".join(
        (
            (
                f"{benchmark.ORDERED_GPU_UUIDS[0]}, 0, 00000000:16:00.0, "
                "8.6, 24576, 24000, 576"
            ),
            (
                f"{benchmark.ORDERED_GPU_UUIDS[1]}, 1, 00000000:D8:00.0, "
                "8.6, 24576, 23900, 676"
            ),
        )
    )
    applications = f"{benchmark.ORDERED_GPU_UUIDS[0]}, 1234, 22000\n"

    def command_runner(
        command: tuple[str, ...],
        _timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        stdout = (
            inventory if command == benchmark._GPU_INVENTORY_COMMAND else applications
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    snapshot = benchmark.collect_gpu_capacity_snapshot(command_runner)
    assert [device.free_mib for device in snapshot.devices] == [24_000, 23_900]
    assert snapshot.devices[1].pci_bus_id == "00000000:d8:00.0"
    assert snapshot.compute_processes == (
        benchmark.GpuComputeProcess(
            gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
            pid=1234,
            used_memory_mib=22_000,
        ),
    )
    assert (
        snapshot.inventory_stdout_sha256
        == hashlib.sha256(inventory.encode()).hexdigest()
    )


def test_prelaunch_capacity_gate_rejects_foreign_owner_and_old_pci(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    evidence = benchmark.validate_prelaunch_gpu_capacity(
        make_snapshot(),
        config,
    )
    assert evidence["passed"] is True

    occupied = make_snapshot(
        compute_processes=(
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[1],
                pid=321,
                used_memory_mib=100,
            ),
        )
    )
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="already have CUDA compute owners",
    ):
        benchmark.validate_prelaunch_gpu_capacity(occupied, config)

    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="expected PCI suffix",
    ):
        benchmark.validate_prelaunch_gpu_capacity(
            make_snapshot(rank_zero_pci_bus_id="00000000:d8:00.0"),
            config,
        )


def test_prelaunch_rejects_fp8_kv_on_sm86_before_model_load(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), kv_cache_dtype="fp8_e4m3")
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="fp8e4nv KV kernels require SM89",
    ):
        benchmark.validate_prelaunch_gpu_capacity(make_snapshot(), config)


def test_prelaunch_capacity_gate_uses_static_fraction_floor(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="requires at least 23442 MiB",
    ):
        benchmark.validate_prelaunch_gpu_capacity(
            make_snapshot(free_mib=22_000),
            config,
        )


def test_postreadiness_gate_requires_owned_cuda_processes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    snapshot = make_snapshot(
        free_mib=1_000,
        compute_processes=(
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
                pid=101,
                used_memory_mib=23_000,
            ),
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[1],
                pid=102,
                used_memory_mib=23_000,
            ),
        ),
    )
    owned = SimpleNamespace(
        process_group_id=77,
        pid=77,
        start_time_ticks=1_000,
        owner_token="owned-token",
    )
    running = cast(tp2.RunningParent, SimpleNamespace(owned=owned))

    def owned_identity(_pid: int) -> benchmark.ProcessIdentity:
        return benchmark.ProcessIdentity(
            parent_pid=77,
            process_group_id=77,
            session_id=77,
            start_time_ticks=1_001,
            environment_entries=frozenset({b"EXO_BENCHMARK_OWNER_TOKEN=owned-token"}),
        )

    evidence = benchmark.validate_postreadiness_gpu_capacity(
        snapshot,
        config,
        running,
        process_identity_reader=owned_identity,
    )
    assert evidence["passed"] is True

    def foreign_identity(_pid: int) -> benchmark.ProcessIdentity:
        return benchmark.ProcessIdentity(
            parent_pid=1,
            process_group_id=88,
            session_id=88,
            start_time_ticks=999,
            environment_entries=frozenset(),
        )

    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="not owned by this TP2 run",
    ):
        benchmark.validate_postreadiness_gpu_capacity(
            snapshot,
            config,
            running,
            process_identity_reader=foreign_identity,
        )


def test_postreadiness_gate_accepts_owned_spawn_session(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    snapshot = make_snapshot(
        free_mib=1_000,
        compute_processes=(
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
                pid=101,
                used_memory_mib=23_000,
            ),
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[1],
                pid=102,
                used_memory_mib=23_000,
            ),
        ),
    )
    running = cast(
        tp2.RunningParent,
        SimpleNamespace(
            owned=SimpleNamespace(
                process_group_id=77,
                pid=77,
                start_time_ticks=1_000,
                owner_token="owned-token",
            )
        ),
    )

    identities = {
        101: benchmark.ProcessIdentity(
            parent_pid=90,
            process_group_id=101,
            session_id=101,
            start_time_ticks=1_001,
            environment_entries=frozenset({b"EXO_BENCHMARK_OWNER_TOKEN=owned-token"}),
        ),
        102: benchmark.ProcessIdentity(
            # Some multiprocessing launchers can reparent a spawned CUDA
            # worker after startup. The inherited owner token plus a start
            # time after the owned parent remains a fail-closed proof.
            parent_pid=1,
            process_group_id=102,
            session_id=102,
            start_time_ticks=1_002,
            environment_entries=frozenset({b"EXO_BENCHMARK_OWNER_TOKEN=owned-token"}),
        ),
        90: benchmark.ProcessIdentity(
            parent_pid=77,
            process_group_id=90,
            session_id=90,
            start_time_ticks=1_000,
            environment_entries=frozenset({b"EXO_BENCHMARK_OWNER_TOKEN=owned-token"}),
        ),
    }

    evidence = benchmark.validate_postreadiness_gpu_capacity(
        snapshot,
        config,
        running,
        process_identity_reader=identities.__getitem__,
    )
    owned_processes = cast(list[dict[str, object]], evidence["owned_cuda_processes"])
    assert {item["ownership_proof"] for item in owned_processes} == {
        "ancestor_chain",
        "owner_token_and_start_time",
    }


def test_server_capacity_gate_is_exact(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    spec = benchmark.build_process_spec(config)
    server_info = make_server_info(config, spec)
    evidence = benchmark.validate_server_capacity(server_info, config, spec)
    assert evidence["passed"] is True
    assert evidence["scheduler_state_count"] == 1

    server_info["max_total_tokens"] = 20_000
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="max_total_tokens",
    ):
        benchmark.validate_server_capacity(server_info, config, spec)


def test_runtime_checkpoint_contract_records_78_plus_1_without_hashing_shards(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime = runtime_root / "python"
    runtime.write_text("#!/bin/sh\n")
    runtime.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir()
    install_receipt = runtime_root / "install-receipt.json"
    install_receipt.write_text('{"runtime": "installed"}\n')
    install_sha256 = hashlib.sha256(install_receipt.read_bytes()).hexdigest()
    checkpoints: list[Path] = []
    for name in ("model", "ktransformers"):
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "glm_moe_dsa",
                    "num_hidden_layers": 78,
                    "num_nextn_predict_layers": 1,
                }
            )
        )
        (checkpoint / "model.safetensors.index.json").write_text("{}")
        checkpoints.append(checkpoint)

    config = make_config(tmp_path)
    config = replace(
        config,
        runtime_python=str(runtime),
        runtime_install_receipt=install_receipt,
        runtime_install_receipt_sha256=install_sha256,
        local_source_directory=str(source),
        model_path=str(checkpoints[0]),
        ktransformers_weight_path=str(checkpoints[1]),
    )
    receipt = benchmark.verify_runtime_and_checkpoint_contract(config)
    assert receipt["mtp_policy"] == {
        "causal_layer_range": [0, 78],
        "layer_78_is_nextn_predict": True,
        "layer_78_loaded": False,
        "layer_78_experts": "not_loaded",
        "speculative_decoding_enabled": False,
    }


def test_prelaunch_failure_writes_receipt_without_starting_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    occupied = make_snapshot(
        compute_processes=(
            benchmark.GpuComputeProcess(
                gpu_uuid=benchmark.ORDERED_GPU_UUIDS[0],
                pid=444,
                used_memory_mib=1,
            ),
        )
    )
    started = False

    monkeypatch.setattr(
        benchmark,
        "verify_runtime_and_checkpoint_contract",
        lambda _config: {"verified": True},
    )
    monkeypatch.setattr(
        benchmark,
        "collect_gpu_capacity_snapshot",
        lambda: occupied,
    )

    def forbidden_start(
        _spec: tp2.Tp2LocalProcessSpec,
        _config: tp2.Tp2LocalDiagnosticConfig,
        _owner_token: str,
    ) -> tp2.RunningParent:
        nonlocal started
        started = True
        raise AssertionError("parent start must not be reached")

    monkeypatch.setattr(tp2, "start_local_parent", forbidden_start)
    with pytest.raises(
        benchmark.Glm52Tp2BenchmarkError,
        match="already have CUDA compute owners",
    ):
        benchmark.run_benchmark(config)
    assert started is False

    receipt_path = config.result_directory / "glm52-tp2-local-benchmark-result.json"
    payload = json.loads(receipt_path.read_text())
    assert payload["status"] == "failed"
    assert payload["started_parent_process_count"] == 0
    assert payload["capacity_and_vram"]["prelaunch_snapshot"] is not None
    assert payload["capacity_and_vram"]["prelaunch_gate"] is None
