#!/usr/bin/env python3
"""Run warm-resident GLM-5.2 BF16/AMXINT4 PP=3 concurrency benchmarks.

This is a focused engineering harness for the live dwagon -> dwagon -> fwuff
pipeline.  It starts all three SGLang-KTransformers ranks once, uses a short
semantic response as the only warm-up, and then measures deterministic
8K-input/128-output long-context cases at concurrency 1, 3, and 6 without
flushing caches or restarting the ranks.

The process ownership and remote cleanup protocol are reused from the proven
GLM-4.7 PP=3 diagnostic.  This script deliberately does not change Exo's
production launch schemas.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

import httpx  # noqa: E402

from exo.shared.types.common import Host, NodeId  # noqa: E402
from exo.shared.types.worker.sglang_kt import ResourceIndex  # noqa: E402
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
)
from scripts.run_sglang_kt_glm47_pp3_diagnostic import (  # noqa: E402
    Pp3DiagnosticConfig as LifecycleConfig,
)
from scripts.run_sglang_kt_glm47_pp3_diagnostic import (  # noqa: E402
    Pp3DiagnosticError,
    RunningStage,
    all_stages_alive,
    start_stage,
    stop_stage,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type EnvironmentVariable = tuple[str, str]
type StreamMetaInfoObserver = Callable[[Mapping[str, JsonValue]], None]
type IndexedStreamMetaInfoObserver = Callable[[int, Mapping[str, JsonValue]], None]

DWAGON_NODE_ID: Final = NodeId("dwagon")
FWUFF_NODE_ID: Final = NodeId("fwuff")

# Rank order is intentional. Rank 1 is the dwagon stage nearest the
# cross-host boundary and is placed beside the ConnectX-5 adapter on NUMA 0.
DWAGON_RANK_ZERO_GPU: Final = "GPU-63a7760a-6164-0758-9228-03dbf35d721c"
DWAGON_RANK_ONE_GPU: Final = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
FWUFF_RANK_TWO_GPU: Final = "GPU-93e47864-13c3-0211-f3a9-ccee1a00d618"
DWAGON_RANK_ZERO_CPUS: Final = tuple(range(56, 112))
DWAGON_RANK_ONE_CPUS: Final = tuple(range(56))
FWUFF_RANK_TWO_CPUS: Final = tuple(range(60))
DEFAULT_PIPELINE_LAYER_PARTITION: Final = (26, 28, 24)
MODEL_LAYER_COUNT: Final = 78

DEFAULT_DWAGON_RUNTIME: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/"
    "14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455/"
    "venv/bin/python"
)
DEFAULT_FWUFF_RUNTIME: Final = (
    "/var/lib/exo/runtimes/glm47-sglang-kt-overlay/fwuff/"
    "a4bbcbcdb9a65a4433fbeb88d65151a3c12354d425908265e6720f371f78a2cb/"
    "venv/bin/python"
)
DEFAULT_SOURCE_DIRECTORY: Final = "/var/lib/exo/sources/ktransformers-glm47-f9ca696"
DEFAULT_MODEL_PATH: Final = "/mnt/sanic/glm52"
DEFAULT_KTRANSFORMERS_WEIGHT_PATH: Final = "/mnt/sanic/glm52-AMXINT4"
DEFAULT_HCA_DEVICES: Final = ("mlx5_0:1",)
DEFAULT_DWAGON_IP: Final = "192.168.40.24"
DEFAULT_FWUFF_IP: Final = "192.168.40.93"
DEFAULT_DWAGON_SOCKET_INTERFACE: Final = "ens13f0np0"
DEFAULT_FWUFF_SOCKET_INTERFACE: Final = "ens17f0"
DEFAULT_CONTEXT_LENGTH: Final = 9_216
DEFAULT_BENCHMARK_INPUT_TOKENS: Final = 8_192
DEFAULT_BENCHMARK_OUTPUT_TOKENS: Final = 128
DEFAULT_BENCHMARK_CONCURRENCIES: Final = (1, 3, 6)
DEFAULT_SAMPLING_SEED: Final = 20_260_725
SERVED_MODEL_NAME: Final = "GLM5.2"
SEMANTIC_PROMPT: Final = "Reply with exactly EXO_GLM52_COHERENT and nothing else."
SEMANTIC_MARKER: Final = "EXO_GLM52_COHERENT"
_LONG_CONTEXT_SLOT: Final = "<EXO_LONG_CONTEXT_SLOT>"
_CONCURRENT_PROMPT_MARKERS: Final = (
    "Albatross",
    "Beryllium",
    "Coriander",
    "Dragonfly",
    "Eucalyptus",
    "Firebrick",
    "Gossamer",
    "Heliotrope",
    "Iridium",
)
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024


class Glm52Pp3BenchmarkError(RuntimeError):
    """Raised when the focused live benchmark cannot produce complete evidence."""


class BenchmarkTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: Literal[False],
        add_generation_prompt: Literal[True],
        enable_thinking: Literal[False],
    ) -> str: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: Literal[False],
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: Literal[True],
        clean_up_tokenization_spaces: Literal[False],
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    run_id: str
    result_directory: Path
    dwagon_runtime_python: str
    fwuff_runtime_python: str
    local_source_directory: str
    remote_source_directory: str
    model_path: str
    ktransformers_weight_path: str
    ssh_target: str
    dwagon_ip: str
    fwuff_ip: str
    dwagon_socket_interface: str
    fwuff_socket_interface: str
    distributed_port: int
    stage_ports: tuple[int, int, int]
    hca_devices: tuple[str, ...]
    pipeline_layer_partition: tuple[int, int, int]
    benchmark_concurrencies: tuple[int, ...]
    context_length: int
    maximum_total_tokens: int
    benchmark_input_tokens: int
    benchmark_output_tokens: int
    static_memory_fraction: float
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class EngineeringProcessSpec:
    pipeline_rank: ResourceIndex
    node_id: NodeId
    gpu_uuid: str
    service_endpoint: Host
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    executable: str
    arguments: tuple[str, ...]
    environment: tuple[EnvironmentVariable, ...]

    @property
    def target_profile(self) -> str:
        # The imported lifecycle uses this value only to select the already
        # proven PP=3 NCCL-environment isolation branch.
        return GLM_4_7_FLASH_PP3_DIAGNOSTIC_TARGET_PROFILE

    @property
    def unset_environment_variables(self) -> tuple[str, ...]:
        return ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF")

    @property
    def unset_environment_variable_prefixes(self) -> tuple[str, ...]:
        return ("NCCL_", "SGLANG_")

    @property
    def command(self) -> tuple[str, ...]:
        return (self.executable, *self.arguments)


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    input_ids: tuple[int, ...]
    input_ids_sha256: str


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    prompt_tokens: int
    completion_tokens: int
    total_client_seconds: float
    output_ids: tuple[int, ...]
    output_text: str
    coherent: bool


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    end_to_end_seconds: float
    ttft_seconds: float
    generation_window_seconds: float
    generation_tokens_per_second: float
    end_to_end_output_tokens_per_second: float
    first_event_output_tokens: int
    output_ids: tuple[int, ...]
    output_ids_sha256: str
    output_text: str
    server_output_text: str | None
    finish_reason: JsonValue
    stream_event_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkRequestObservation:
    request_index: int
    input_ids_sha256: str
    observation: BenchmarkObservation


@dataclass(frozen=True, slots=True)
class BenchmarkCaseObservation:
    concurrency: int
    zero_cached_tokens_required: bool
    case_wall_seconds: float
    total_prompt_tokens: int
    total_completion_tokens: int
    aggregate_output_tokens_per_second: float
    requests: tuple[BenchmarkRequestObservation, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    encoded = b"".join(token_id.to_bytes(4, "little") for token_id in token_ids)
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _status(message: str) -> None:
    print(f"[{_utc_now()}] {message}", file=sys.stderr, flush=True)


def _common_server_arguments(
    config: BenchmarkConfig,
    *,
    rank: int,
    service_ip: str,
    service_port: int,
    ktransformers_numa_node: int,
    cpu_count: int,
) -> tuple[str, ...]:
    return (
        "-m",
        "sglang.launch_server",
        "--model-path",
        config.model_path,
        "--kt-weight-path",
        config.ktransformers_weight_path,
        "--kt-method",
        "AMXINT4",
        "--kt-cpuinfer",
        str(cpu_count),
        "--kt-threadpool-count",
        "2",
        "--kt-numa-nodes",
        str(ktransformers_numa_node),
        str(ktransformers_numa_node),
        "--kt-num-gpu-experts",
        "0",
        "--kt-max-deferred-experts-per-token",
        "0",
        "--kt-expert-placement-strategy",
        "uniform",
        "--pp-size",
        "3",
        "--tp-size",
        "1",
        "--nnodes",
        "3",
        "--node-rank",
        str(rank),
        "--dist-init-addr",
        f"{config.dwagon_ip}:{config.distributed_port}",
        "--host",
        service_ip,
        "--port",
        str(service_port),
        "--context-length",
        str(config.context_length),
        "--max-total-tokens",
        str(config.maximum_total_tokens),
        "--mem-fraction-static",
        str(config.static_memory_fraction),
        "--max-running-requests",
        str(max(config.benchmark_concurrencies)),
        "--chunked-prefill-size",
        "2048",
        "--attention-backend",
        "flashinfer",
        "--kv-cache-dtype",
        "bfloat16",
        "--disable-cuda-graph",
        "--disable-custom-all-reduce",
        "--disable-shared-experts-fusion",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--watchdog-timeout",
        "3000",
        "--trust-remote-code",
    )


def _stage_environment(
    gpu_uuid: str,
    hca_devices: Sequence[str],
    pipeline_layer_partition: Sequence[int],
) -> tuple[EnvironmentVariable, ...]:
    hca_selection = f"={','.join(hca_devices)}"
    return (
        ("CUDA_VISIBLE_DEVICES", gpu_uuid),
        ("NCCL_NET", "IB"),
        ("NCCL_IB_HCA", hca_selection),
        ("NCCL_GIN_ENABLE", "0"),
        ("NCCL_GIN_TYPE", "0"),
        ("NCCL_NET_GDR_LEVEL", "LOC"),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
        (
            "SGLANG_PP_LAYER_PARTITION",
            ",".join(str(count) for count in pipeline_layer_partition),
        ),
    )


def build_process_specs(
    config: BenchmarkConfig,
) -> tuple[EngineeringProcessSpec, ...]:
    stage_inputs = (
        (
            DWAGON_NODE_ID,
            DWAGON_RANK_ZERO_GPU,
            DWAGON_RANK_ZERO_CPUS,
            1,
            config.dwagon_ip,
            config.dwagon_runtime_python,
        ),
        (
            DWAGON_NODE_ID,
            DWAGON_RANK_ONE_GPU,
            DWAGON_RANK_ONE_CPUS,
            0,
            config.dwagon_ip,
            config.dwagon_runtime_python,
        ),
        (
            FWUFF_NODE_ID,
            FWUFF_RANK_TWO_GPU,
            FWUFF_RANK_TWO_CPUS,
            0,
            config.fwuff_ip,
            config.fwuff_runtime_python,
        ),
    )
    return tuple(
        EngineeringProcessSpec(
            pipeline_rank=rank,
            node_id=node_id,
            gpu_uuid=gpu_uuid,
            service_endpoint=Host(ip=service_ip, port=config.stage_ports[rank]),
            cpu_cores=cast(tuple[ResourceIndex, ...], cpu_cores),
            memory_nodes=(cast(ResourceIndex, memory_node),),
            executable=executable,
            arguments=_common_server_arguments(
                config,
                rank=rank,
                service_ip=service_ip,
                service_port=config.stage_ports[rank],
                ktransformers_numa_node=memory_node,
                cpu_count=len(cpu_cores),
            ),
            environment=_stage_environment(
                gpu_uuid,
                config.hca_devices,
                config.pipeline_layer_partition,
            ),
        )
        for rank, (
            node_id,
            gpu_uuid,
            cpu_cores,
            memory_node,
            service_ip,
            executable,
        ) in enumerate(stage_inputs)
    )


def _lifecycle_config(config: BenchmarkConfig) -> LifecycleConfig:
    return LifecycleConfig(
        run_id=config.run_id,
        result_directory=config.result_directory,
        dwagon_runtime_python=config.dwagon_runtime_python,
        fwuff_runtime_python=config.fwuff_runtime_python,
        dwagon_model_path=config.model_path,
        fwuff_model_path=config.model_path,
        local_source_directory=config.local_source_directory,
        remote_source_directory=config.remote_source_directory,
        ssh_target=config.ssh_target,
        dwagon_ip=config.dwagon_ip,
        fwuff_ip=config.fwuff_ip,
        dwagon_socket_interface=config.dwagon_socket_interface,
        fwuff_socket_interface=config.fwuff_socket_interface,
        distributed_port=config.distributed_port,
        stage_ports=config.stage_ports,
        hca_devices=config.hca_devices,
        pipeline_layer_partition=config.pipeline_layer_partition,
        dwagon_stage_placement="pipeline-order",
        resident_gpu_experts=0,
        readiness_timeout_seconds=config.readiness_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        cleanup_timeout_seconds=config.cleanup_timeout_seconds,
        warmup_count=1,
        sample_count=1,
    )


def _as_lifecycle_spec(spec: EngineeringProcessSpec) -> SglangKtProcessLaunchSpec:
    # The lifecycle only consumes the structural properties implemented above.
    return cast(SglangKtProcessLaunchSpec, cast(object, spec))


def _load_tokenizer(model_path: str) -> BenchmarkTokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    return cast(BenchmarkTokenizer, tokenizer)


def _render_chat_prompt(tokenizer: BenchmarkTokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _prepare_semantic_prompt(tokenizer: BenchmarkTokenizer) -> PreparedPrompt:
    rendered = _render_chat_prompt(tokenizer, SEMANTIC_PROMPT)
    input_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    return PreparedPrompt(
        input_ids=input_ids,
        input_ids_sha256=_sha256_token_ids(input_ids),
    )


def _long_context_record(record_number: int) -> str:
    severity = ("low", "moderate", "high")[record_number % 3]
    subsystem = ("storage", "network", "scheduler", "inference")[record_number % 4]
    return (
        f"Record {record_number:04d}: At minute {record_number % 60:02d}, the "
        f"{subsystem} subsystem reported {severity} load. The operator retained "
        "the resident model, checked request ordering, and observed that pipeline "
        f"stage {record_number % 3} remained coherent. No cache eviction or "
        "process restart occurred. "
    )


def _prepare_long_context_prompt(
    tokenizer: BenchmarkTokenizer,
    token_count: int,
    variant_marker: str | None = None,
) -> PreparedPrompt:
    request_marker = "" if variant_marker is None else f"{variant_marker}. "
    user_content = (
        request_marker
        + "Read the incident timeline below. At the end, state the recurring "
        "operator action and whether the model was restarted. Keep the answer "
        "under 80 words.\n\n"
        + _LONG_CONTEXT_SLOT
        + "\n\nEnd of timeline. Answer the question now."
    )
    rendered = _render_chat_prompt(tokenizer, user_content)
    prefix, separator, suffix = rendered.partition(_LONG_CONTEXT_SLOT)
    if not separator:
        raise Glm52Pp3BenchmarkError("long-context template slot was not preserved")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    context_token_count = token_count - len(prefix_ids) - len(suffix_ids)
    if context_token_count <= 0:
        raise Glm52Pp3BenchmarkError("benchmark token count is smaller than its prompt")

    context_ids: list[int] = []
    record_number = 1
    while len(context_ids) < context_token_count:
        context_ids.extend(
            tokenizer.encode(
                _long_context_record(record_number),
                add_special_tokens=False,
            )
        )
        record_number += 1
    input_ids = tuple((*prefix_ids, *context_ids[:context_token_count], *suffix_ids))
    if len(input_ids) != token_count:
        raise Glm52Pp3BenchmarkError("long-context prompt has the wrong token count")
    return PreparedPrompt(
        input_ids=input_ids,
        input_ids_sha256=_sha256_token_ids(input_ids),
    )


def wait_for_all_stages(
    specs: Sequence[EngineeringProcessSpec],
    running: Sequence[RunningStage],
    timeout_seconds: float,
) -> list[JsonValue]:
    deadline = time.monotonic() + timeout_seconds
    pending = {spec.pipeline_rank: spec for spec in specs}
    observations: dict[int, JsonObject] = {}
    next_status = time.monotonic()
    while pending and time.monotonic() < deadline:
        if not all_stages_alive(running):
            return_codes = {stage.owned.rank: stage.process.poll() for stage in running}
            raise Glm52Pp3BenchmarkError(
                f"a pipeline rank exited before readiness: {return_codes}"
            )
        for rank, spec in tuple(pending.items()):
            started = time.monotonic()
            try:
                response = httpx.get(
                    f"http://{spec.service_endpoint}/health_generate",
                    timeout=2.0,
                )
                response.raise_for_status()
            except (httpx.HTTPError, OSError):
                continue
            observations[rank] = {
                "rank": rank,
                "endpoint": str(spec.service_endpoint),
                "status_code": response.status_code,
                "response_sha256": _sha256_bytes(response.content),
                "elapsed_seconds": time.monotonic() - started,
            }
            pending.pop(rank)
            _status(f"rank {rank} is ready at {spec.service_endpoint}")
        if pending:
            now = time.monotonic()
            if now >= next_status:
                _status(f"waiting for ranks {tuple(sorted(pending))}")
                next_status = now + 30.0
            time.sleep(0.25)
    if pending:
        raise Glm52Pp3BenchmarkError(
            f"pipeline readiness timed out for ranks {tuple(sorted(pending))}"
        )
    return [observations[rank] for rank in range(len(specs))]


def _strict_response_object(response: httpx.Response, description: str) -> JsonObject:
    try:
        payload = response.json()
    except ValueError as error:
        raise Glm52Pp3BenchmarkError(f"{description} is not JSON") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise Glm52Pp3BenchmarkError(f"{description} is not a JSON object")
    return cast(JsonObject, payload)


def _native_request(
    prepared: PreparedPrompt,
    *,
    max_new_tokens: int,
    stream: bool,
    ignore_eos: bool,
) -> JsonObject:
    return {
        "input_ids": list(prepared.input_ids),
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": ignore_eos,
            "sampling_seed": DEFAULT_SAMPLING_SEED,
        },
        "stream": stream,
        "return_logprob": False,
        "log_metrics": True,
    }


def run_semantic_warmup(
    client: httpx.Client,
    tokenizer: BenchmarkTokenizer,
    prepared: PreparedPrompt,
) -> SemanticObservation:
    started = time.perf_counter()
    response = client.post(
        "/generate",
        json=_native_request(
            prepared,
            max_new_tokens=24,
            stream=False,
            ignore_eos=False,
        ),
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise Glm52Pp3BenchmarkError(
            f"semantic warm-up returned HTTP {response.status_code}: "
            f"{response.text[-1000:]}"
        )
    payload = _strict_response_object(response, "semantic warm-up response")
    raw_output_ids = payload.get("output_ids")
    raw_text = payload.get("text")
    raw_meta = payload.get("meta_info")
    if (
        not isinstance(raw_output_ids, list)
        or not all(isinstance(item, int) for item in raw_output_ids)
        or not isinstance(raw_meta, dict)
    ):
        raise Glm52Pp3BenchmarkError("semantic warm-up response is incomplete")
    output_ids = tuple(cast(list[int], raw_output_ids))
    output_text = (
        raw_text
        if isinstance(raw_text, str)
        else tokenizer.decode(
            list(output_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    prompt_tokens = raw_meta.get("prompt_tokens")
    completion_tokens = raw_meta.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or not isinstance(completion_tokens, int)
        or prompt_tokens != len(prepared.input_ids)
        or completion_tokens != len(output_ids)
    ):
        raise Glm52Pp3BenchmarkError(
            "semantic warm-up returned inconsistent token counts"
        )
    coherent = SEMANTIC_MARKER in output_text
    if not coherent:
        raise Glm52Pp3BenchmarkError(
            f"semantic warm-up omitted {SEMANTIC_MARKER}: {output_text!r}"
        )
    return SemanticObservation(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_client_seconds=elapsed,
        output_ids=output_ids,
        output_text=output_text,
        coherent=coherent,
    )


def _parse_stream_event(line: str) -> JsonObject | None:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data: "):
        raise Glm52Pp3BenchmarkError("benchmark stream contains a non-SSE data line")
    encoded = line.removeprefix("data: ")
    if encoded == "[DONE]":
        return None
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise Glm52Pp3BenchmarkError(
            "benchmark stream contains invalid JSON"
        ) from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise Glm52Pp3BenchmarkError("benchmark stream event is not a JSON object")
    return cast(JsonObject, payload)


def run_long_context_benchmark(
    client: httpx.Client,
    tokenizer: BenchmarkTokenizer,
    prepared: PreparedPrompt,
    output_token_count: int,
    *,
    stream_meta_info_observer: StreamMetaInfoObserver | None = None,
) -> BenchmarkObservation:
    request = _native_request(
        prepared,
        max_new_tokens=output_token_count,
        stream=True,
        ignore_eos=True,
    )
    started = time.perf_counter()
    first_output_at: float | None = None
    last_output_at: float | None = None
    first_event_output_tokens: int | None = None
    output_ids: list[int] = []
    server_output_text: str | None = None
    prompt_tokens: int | None = None
    completion_tokens = 0
    cached_tokens = 0
    finish_reason: JsonValue = None
    event_count = 0
    saw_done = False
    with client.stream("POST", "/generate", json=request) as response:
        if response.status_code != 200:
            response.read()
            raise Glm52Pp3BenchmarkError(
                f"benchmark returned HTTP {response.status_code}: "
                f"{response.text[-1000:]}"
            )
        for line in response.iter_lines():
            received_at = time.perf_counter()
            if line == "data: [DONE]":
                saw_done = True
                continue
            event = _parse_stream_event(line)
            if event is None:
                continue
            event_count += 1
            raw_output_ids = event.get("output_ids")
            raw_meta = event.get("meta_info")
            if (
                not isinstance(raw_output_ids, list)
                or not all(isinstance(item, int) for item in raw_output_ids)
                or not isinstance(raw_meta, dict)
            ):
                raise Glm52Pp3BenchmarkError(
                    "benchmark stream event lacks token metadata"
                )
            event_output_ids = cast(list[int], raw_output_ids)
            raw_prompt_tokens = raw_meta.get("prompt_tokens")
            raw_completion_tokens = raw_meta.get("completion_tokens")
            raw_cached_tokens = raw_meta.get("cached_tokens", 0)
            if (
                not isinstance(raw_prompt_tokens, int)
                or not isinstance(raw_completion_tokens, int)
                or not isinstance(raw_cached_tokens, int)
            ):
                raise Glm52Pp3BenchmarkError(
                    "benchmark stream event has invalid token counts"
                )
            if prompt_tokens is None:
                prompt_tokens = raw_prompt_tokens
                cached_tokens = raw_cached_tokens
            elif (
                prompt_tokens != raw_prompt_tokens or cached_tokens != raw_cached_tokens
            ):
                raise Glm52Pp3BenchmarkError("benchmark stream token metadata changed")
            if stream_meta_info_observer is not None:
                stream_meta_info_observer(cast(Mapping[str, JsonValue], raw_meta))

            if event_output_ids:
                previous_count = len(output_ids)
                if raw_completion_tokens == len(event_output_ids):
                    if (
                        len(event_output_ids) <= previous_count
                        or event_output_ids[:previous_count] != output_ids
                    ):
                        raise Glm52Pp3BenchmarkError(
                            "cumulative benchmark output broke its prior prefix"
                        )
                    output_ids = list(event_output_ids)
                elif raw_completion_tokens == previous_count + len(event_output_ids):
                    output_ids.extend(event_output_ids)
                else:
                    raise Glm52Pp3BenchmarkError(
                        "benchmark output IDs disagree with completion count"
                    )
                if first_output_at is None:
                    first_output_at = received_at
                    first_event_output_tokens = len(output_ids)
                last_output_at = received_at
            completion_tokens = raw_completion_tokens
            raw_server_text = event.get("text")
            if isinstance(raw_server_text, str):
                server_output_text = raw_server_text
            if raw_meta.get("finish_reason") is not None:
                finish_reason = cast(JsonValue, raw_meta["finish_reason"])

    completed = time.perf_counter()
    if (
        not saw_done
        or prompt_tokens != len(prepared.input_ids)
        or completion_tokens != output_token_count
        or len(output_ids) != output_token_count
        or first_output_at is None
        or last_output_at is None
        or first_event_output_tokens is None
        or last_output_at <= first_output_at
    ):
        raise Glm52Pp3BenchmarkError(
            "benchmark stream did not provide complete exact-token timing evidence"
        )
    generation_window = last_output_at - first_output_at
    total = completed - started
    decoded = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return BenchmarkObservation(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        end_to_end_seconds=total,
        ttft_seconds=first_output_at - started,
        generation_window_seconds=generation_window,
        generation_tokens_per_second=(completion_tokens - first_event_output_tokens)
        / generation_window,
        end_to_end_output_tokens_per_second=completion_tokens / total,
        first_event_output_tokens=first_event_output_tokens,
        output_ids=tuple(output_ids),
        output_ids_sha256=_sha256_token_ids(output_ids),
        output_text=decoded,
        server_output_text=server_output_text,
        finish_reason=finish_reason,
        stream_event_count=event_count,
    )


def run_concurrency_case(
    client: httpx.Client,
    tokenizer: BenchmarkTokenizer,
    prepared_prompts: Sequence[PreparedPrompt],
    output_token_count: int,
    *,
    stream_meta_info_observer: IndexedStreamMetaInfoObserver | None = None,
) -> BenchmarkCaseObservation:
    concurrency = len(prepared_prompts)
    if concurrency <= 0:
        raise Glm52Pp3BenchmarkError("benchmark concurrency must be positive")

    release_times: list[float] = []

    def record_release() -> None:
        release_times.append(time.perf_counter())

    release_barrier = threading.Barrier(concurrency + 1, action=record_release)

    def run_request(
        request_index: int,
        prepared: PreparedPrompt,
    ) -> BenchmarkRequestObservation:
        try:
            release_barrier.wait(timeout=30.0)
        except threading.BrokenBarrierError as error:
            raise Glm52Pp3BenchmarkError(
                f"concurrency {concurrency} request release barrier broke"
            ) from error
        return BenchmarkRequestObservation(
            request_index=request_index,
            input_ids_sha256=prepared.input_ids_sha256,
            observation=run_long_context_benchmark(
                client,
                tokenizer,
                prepared,
                output_token_count,
                stream_meta_info_observer=(
                    None
                    if stream_meta_info_observer is None
                    else lambda meta_info: stream_meta_info_observer(
                        request_index,
                        meta_info,
                    )
                ),
            ),
        )

    observations: dict[int, BenchmarkRequestObservation] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix=f"glm52-c{concurrency}",
    ) as executor:
        futures = {
            request_index: executor.submit(run_request, request_index, prepared)
            for request_index, prepared in enumerate(prepared_prompts)
        }
        try:
            release_barrier.wait(timeout=30.0)
        except threading.BrokenBarrierError as error:
            for future in futures.values():
                future.cancel()
            raise Glm52Pp3BenchmarkError(
                f"concurrency {concurrency} controller release barrier broke"
            ) from error
        if len(release_times) != 1:
            raise Glm52Pp3BenchmarkError(
                f"concurrency {concurrency} release time was not recorded once"
            )
        for request_index, future in futures.items():
            try:
                observations[request_index] = future.result()
            except BaseException as error:
                raise Glm52Pp3BenchmarkError(
                    f"concurrency {concurrency} request {request_index} failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
        completed = time.perf_counter()

    ordered = tuple(observations[index] for index in range(concurrency))
    zero_cached_tokens_required = True
    if zero_cached_tokens_required:
        cached_by_request = {
            request.request_index: request.observation.cached_tokens
            for request in ordered
            if request.observation.cached_tokens != 0
        }
        if cached_by_request:
            raise Glm52Pp3BenchmarkError(
                f"concurrency {concurrency} requests reused cached prompt tokens: "
                f"{cached_by_request}"
            )
    total_prompt_tokens = sum(
        request.observation.prompt_tokens for request in ordered
    )
    total_completion_tokens = sum(
        request.observation.completion_tokens for request in ordered
    )
    case_wall_seconds = completed - release_times[0]
    if case_wall_seconds <= 0.0:
        raise Glm52Pp3BenchmarkError(
            f"concurrency {concurrency} case wall time is not positive"
        )
    return BenchmarkCaseObservation(
        concurrency=concurrency,
        zero_cached_tokens_required=zero_cached_tokens_required,
        case_wall_seconds=case_wall_seconds,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        aggregate_output_tokens_per_second=(
            total_completion_tokens / case_wall_seconds
        ),
        requests=ordered,
    )


def _process_receipt(running: RunningStage) -> JsonObject:
    owned = asdict(running.owned)
    owned["owner_token"] = _sha256_bytes(running.owned.owner_token.encode())
    return cast(JsonObject, owned)


def _pipeline_layer_ranges(
    partition: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    start = 0
    ranges: list[tuple[int, int]] = []
    for layer_count in partition:
        end = start + layer_count
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _spec_receipt(
    spec: EngineeringProcessSpec,
    pipeline_layer_partition: Sequence[int],
) -> JsonObject:
    layer_ranges = _pipeline_layer_ranges(pipeline_layer_partition)
    return {
        "pipeline_rank": spec.pipeline_rank,
        "layer_range": list(layer_ranges[spec.pipeline_rank]),
        "node_id": spec.node_id,
        "gpu_uuid": spec.gpu_uuid,
        "service_endpoint": str(spec.service_endpoint),
        "cpu_cores": list(spec.cpu_cores),
        "memory_nodes": list(spec.memory_nodes),
        "executable": spec.executable,
        "arguments": list(spec.arguments),
        "environment": dict(spec.environment),
    }


def _log_receipts(config: BenchmarkConfig) -> list[JsonValue]:
    receipts: list[JsonValue] = []
    for rank in range(3):
        path = config.result_directory / f"rank-{rank}.log"
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > _LOG_MAXIMUM_BYTES:
            raise Glm52Pp3BenchmarkError(
                f"rank {rank} log exceeds the engineering size bound"
            )
        receipts.append(
            {
                "rank": rank,
                "path": str(path),
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return receipts


def _write_result(config: BenchmarkConfig, payload: JsonObject) -> Path:
    path = config.result_directory / "glm52-pp3-benchmark-result.json"
    temporary = config.result_directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    return path


def _validate_server_capacity(
    server_info: JsonObject,
    config: BenchmarkConfig,
) -> None:
    expected_concurrency = max(config.benchmark_concurrencies)
    expected_micro_batch_size = math.ceil(expected_concurrency / 3)
    expected_top_level = {
        "context_length": config.context_length,
        "max_total_tokens": config.maximum_total_tokens,
        "max_total_num_tokens": config.maximum_total_tokens,
        "max_running_requests": expected_concurrency,
    }
    for field, expected in expected_top_level.items():
        if server_info.get(field) != expected:
            raise Glm52Pp3BenchmarkError(
                f"server capacity field {field} is {server_info.get(field)!r}, "
                f"expected {expected}"
            )

    raw_internal_states = server_info.get("internal_states")
    if not isinstance(raw_internal_states, list) or not raw_internal_states:
        raise Glm52Pp3BenchmarkError(
            "server_info did not expose scheduler internal states"
        )
    for state_index, raw_state in enumerate(raw_internal_states):
        if not isinstance(raw_state, dict):
            raise Glm52Pp3BenchmarkError(
                f"server scheduler state {state_index} is not an object"
            )
        expected_state = {
            "effective_max_running_requests_per_dp": expected_concurrency,
            "max_total_tokens": config.maximum_total_tokens,
            "pp_max_micro_batch_size": expected_micro_batch_size,
        }
        for field, expected in expected_state.items():
            if raw_state.get(field) != expected:
                raise Glm52Pp3BenchmarkError(
                    f"server scheduler state {state_index} field {field} is "
                    f"{raw_state.get(field)!r}, expected {expected}"
                )


def run_benchmark(config: BenchmarkConfig) -> JsonObject:
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    tokenizer = _load_tokenizer(config.model_path)
    semantic_prompt = _prepare_semantic_prompt(tokenizer)
    concurrent_request_count = sum(
        concurrency
        for concurrency in config.benchmark_concurrencies
        if concurrency > 1
    )
    if concurrent_request_count > len(_CONCURRENT_PROMPT_MARKERS):
        raise Glm52Pp3BenchmarkError(
            "not enough early-unique prompt markers for the requested "
            f"concurrency cases ({concurrent_request_count} required)"
        )
    marker_iterator = iter(_CONCURRENT_PROMPT_MARKERS)
    benchmark_prompts: dict[int, tuple[PreparedPrompt, ...]] = {}
    for concurrency in config.benchmark_concurrencies:
        if concurrency == 1:
            # Keep the single-stream request identical to the v1 benchmark.
            prompts = (
                _prepare_long_context_prompt(
                    tokenizer,
                    config.benchmark_input_tokens,
                ),
            )
        else:
            prompts = tuple(
                _prepare_long_context_prompt(
                    tokenizer,
                    config.benchmark_input_tokens,
                    next(marker_iterator),
                )
                for _request_index in range(concurrency)
            )
        benchmark_prompts[concurrency] = prompts
    specs = build_process_specs(config)
    lifecycle_config = _lifecycle_config(config)
    owner_token = uuid.uuid4().hex
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    running: list[RunningStage] = []
    readiness: list[JsonValue] = []
    server_info: JsonObject | None = None
    semantic: SemanticObservation | None = None
    benchmark_cases: list[BenchmarkCaseObservation] = []
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []

    try:
        for spec in specs:
            _status(
                f"starting rank {spec.pipeline_rank} on {spec.node_id} "
                f"GPU {spec.gpu_uuid}"
            )
            running.append(
                start_stage(
                    _as_lifecycle_spec(spec),
                    lifecycle_config,
                    owner_token,
                )
            )
        readiness = wait_for_all_stages(
            specs,
            running,
            config.readiness_timeout_seconds,
        )
        rank_zero_url = f"http://{specs[0].service_endpoint}"
        with httpx.Client(
            base_url=rank_zero_url,
            timeout=config.request_timeout_seconds,
            headers={"Accept-Encoding": "identity"},
        ) as client:
            info_response = client.get("/server_info")
            info_response.raise_for_status()
            server_info = _strict_response_object(info_response, "server_info")
            _validate_server_capacity(server_info, config)

            _status("running semantic coherency warm-up on the resident pipeline")
            semantic = run_semantic_warmup(
                client,
                tokenizer,
                semantic_prompt,
            )
            if not all_stages_alive(running):
                raise Glm52Pp3BenchmarkError(
                    "a pipeline rank exited during semantic warm-up"
                )
            _status(
                "semantic warm-up passed; starting the 8192-input/128-output "
                "concurrency cases without a cache flush or restart"
            )
            for concurrency in config.benchmark_concurrencies:
                _status(f"starting concurrency {concurrency} benchmark case")
                benchmark_case = run_concurrency_case(
                    client,
                    tokenizer,
                    benchmark_prompts[concurrency],
                    config.benchmark_output_tokens,
                )
                benchmark_cases.append(benchmark_case)
                if not all_stages_alive(running):
                    raise Glm52Pp3BenchmarkError(
                        "a pipeline rank exited during concurrency "
                        f"{concurrency}"
                    )
                _status(
                    f"concurrency {concurrency} complete: "
                    f"wall={benchmark_case.case_wall_seconds:.3f}s, "
                    "aggregate output="
                    f"{benchmark_case.aggregate_output_tokens_per_second:.3f} "
                    "tok/s"
                )
    except BaseException as error:
        failure = error
        _status(f"run failed: {type(error).__name__}: {error}")
    finally:
        for stage in reversed(running):
            receipt = stop_stage(stage, lifecycle_config)
            cleanup.append(
                {
                    "rank": stage.owned.rank,
                    **receipt.model_dump(mode="json"),
                }
            )
        cleanup.sort(key=lambda item: cast(int, cast(dict[str, object], item)["rank"]))

    cleanup_complete = len(cleanup) == len(running) and all(
        cast(dict[str, object], item)["ownership_verified"] is True
        and cast(dict[str, object], item)["terminated"] is True
        for item in cleanup
    )
    payload: JsonObject = {
        "schema_version": 2,
        "kind": "glm52_bf16_amxint4_pp3_engineering_benchmark",
        "status": (
            "passed"
            if (
                failure is None
                and len(benchmark_cases) == len(config.benchmark_concurrencies)
                and cleanup_complete
            )
            else "failed"
        ),
        "run_id": config.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "single_resident_cluster": True,
        "cache_flush_between_semantic_and_benchmark": False,
        "model": {
            "model_path": config.model_path,
            "ktransformers_weight_path": config.ktransformers_weight_path,
            "weight_dtype": "bfloat16",
            "expert_method": "AMXINT4",
            "kv_cache_dtype": "bfloat16",
            "resident_gpu_experts": 0,
        },
        "pipeline_layer_partition": list(config.pipeline_layer_partition),
        "benchmark_concurrencies": list(config.benchmark_concurrencies),
        "maximum_total_tokens": config.maximum_total_tokens,
        "hca_devices": list(config.hca_devices),
        "process_specs": [
            _spec_receipt(spec, config.pipeline_layer_partition) for spec in specs
        ],
        "processes": [_process_receipt(stage) for stage in running],
        "readiness": readiness,
        "server_info": server_info,
        "semantic_warmup": (
            None if semantic is None else cast(JsonObject, asdict(semantic))
        ),
        "benchmark_prompt": {
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
            "sampling_seed": DEFAULT_SAMPLING_SEED,
        },
        "benchmark_cases": [
            cast(JsonObject, asdict(benchmark_case))
            for benchmark_case in benchmark_cases
        ],
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "logs": _log_receipts(config),
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    result_path = _write_result(config, payload)
    if failure is not None:
        raise Glm52Pp3BenchmarkError(
            f"GLM-5.2 PP3 benchmark failed; evidence is in {result_path}: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    if len(benchmark_cases) != len(config.benchmark_concurrencies):
        raise Glm52Pp3BenchmarkError(
            "GLM-5.2 PP3 benchmark did not produce every concurrency case; "
            f"see {result_path}"
        )
    if not cleanup_complete:
        raise Glm52Pp3BenchmarkError(
            f"GLM-5.2 PP3 cleanup was incomplete; see {result_path}"
        )
    return payload


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _memory_fraction(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("memory fraction must be between 0 and 1")
    return value


def _parse_hca_devices(raw: str) -> tuple[str, ...]:
    devices = tuple(item.strip().removeprefix("=") for item in raw.split(","))
    if not devices or any(
        not device or ":" not in device or "," in device for device in devices
    ):
        raise argparse.ArgumentTypeError("HCA devices must be DEVICE:PORT entries")
    return devices


def _parse_positive_integer_list(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated list of integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all comma-separated values must be positive")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument(
        "--dwagon-runtime-python",
        default=DEFAULT_DWAGON_RUNTIME,
    )
    parser.add_argument(
        "--fwuff-runtime-python",
        default=DEFAULT_FWUFF_RUNTIME,
    )
    parser.add_argument(
        "--local-source-directory",
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument(
        "--remote-source-directory",
        default=DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--ktransformers-weight-path",
        default=DEFAULT_KTRANSFORMERS_WEIGHT_PATH,
    )
    parser.add_argument("--ssh-target", default="fwuff")
    parser.add_argument("--dwagon-ip", default=DEFAULT_DWAGON_IP)
    parser.add_argument("--fwuff-ip", default=DEFAULT_FWUFF_IP)
    parser.add_argument(
        "--dwagon-socket-interface",
        default=DEFAULT_DWAGON_SOCKET_INTERFACE,
    )
    parser.add_argument(
        "--fwuff-socket-interface",
        default=DEFAULT_FWUFF_SOCKET_INTERFACE,
    )
    parser.add_argument("--distributed-port", type=_positive_int, default=62500)
    parser.add_argument("--rank-zero-port", type=_positive_int, default=62510)
    parser.add_argument("--rank-one-port", type=_positive_int, default=62511)
    parser.add_argument("--rank-two-port", type=_positive_int, default=62512)
    parser.add_argument(
        "--hca-devices",
        type=_parse_hca_devices,
        default=DEFAULT_HCA_DEVICES,
        help="comma-separated NCCL DEVICE:PORT entries (default: mlx5_0:1)",
    )
    parser.add_argument(
        "--pipeline-layer-partition",
        type=_parse_positive_integer_list,
        default=DEFAULT_PIPELINE_LAYER_PARTITION,
        help="three comma-separated PP stage layer counts (default: 26,28,24)",
    )
    parser.add_argument(
        "--benchmark-concurrencies",
        type=_parse_positive_integer_list,
        default=DEFAULT_BENCHMARK_CONCURRENCIES,
        help="ordered comma-separated concurrency cases (default: 1,3,6)",
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
        "--max-total-tokens",
        dest="maximum_total_tokens",
        type=_positive_int,
        default=None,
        help=(
            "aggregate server KV-token capacity; defaults to the largest "
            "concurrency times input-plus-output tokens"
        ),
    )
    parser.add_argument(
        "--static-memory-fraction",
        type=_memory_fraction,
        default=0.90,
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
    result_directory = cast(Path, arguments.result_directory).resolve()
    ports = (
        cast(int, arguments.distributed_port),
        cast(int, arguments.rank_zero_port),
        cast(int, arguments.rank_one_port),
        cast(int, arguments.rank_two_port),
    )
    if len(set(ports)) != len(ports) or any(
        port <= 0 or port > 65_535 for port in ports
    ):
        raise Glm52Pp3BenchmarkError(
            "distributed and service ports must be distinct valid TCP ports"
        )
    run_id = cast(str, arguments.run_id)
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise Glm52Pp3BenchmarkError(
            "run_id must contain only safe identifier characters"
        )
    context_length = cast(int, arguments.context_length)
    benchmark_input_tokens = cast(int, arguments.benchmark_input_tokens)
    benchmark_output_tokens = cast(int, arguments.benchmark_output_tokens)
    raw_partition = cast(tuple[int, ...], arguments.pipeline_layer_partition)
    if len(raw_partition) != 3 or sum(raw_partition) != MODEL_LAYER_COUNT:
        raise Glm52Pp3BenchmarkError(
            "pipeline layer partition must contain three positive counts "
            f"that sum to {MODEL_LAYER_COUNT}"
        )
    pipeline_layer_partition = cast(tuple[int, int, int], raw_partition)
    benchmark_concurrencies = cast(
        tuple[int, ...],
        arguments.benchmark_concurrencies,
    )
    if len(set(benchmark_concurrencies)) != len(benchmark_concurrencies):
        raise Glm52Pp3BenchmarkError(
            "benchmark concurrency cases must be unique"
        )
    if benchmark_input_tokens + benchmark_output_tokens > context_length:
        raise Glm52Pp3BenchmarkError(
            "benchmark input plus output tokens exceed context length"
        )
    minimum_total_tokens = max(benchmark_concurrencies) * (
        benchmark_input_tokens + benchmark_output_tokens
    )
    raw_maximum_total_tokens = cast(int | None, arguments.maximum_total_tokens)
    maximum_total_tokens = (
        minimum_total_tokens
        if raw_maximum_total_tokens is None
        else raw_maximum_total_tokens
    )
    if maximum_total_tokens < minimum_total_tokens:
        raise Glm52Pp3BenchmarkError(
            "max total tokens must admit every exact input-plus-output token "
            "in the largest concurrency case "
            f"({minimum_total_tokens} tokens required)"
        )
    return BenchmarkConfig(
        run_id=run_id,
        result_directory=result_directory,
        dwagon_runtime_python=cast(str, arguments.dwagon_runtime_python),
        fwuff_runtime_python=cast(str, arguments.fwuff_runtime_python),
        local_source_directory=cast(str, arguments.local_source_directory),
        remote_source_directory=cast(str, arguments.remote_source_directory),
        model_path=cast(str, arguments.model_path),
        ktransformers_weight_path=cast(str, arguments.ktransformers_weight_path),
        ssh_target=cast(str, arguments.ssh_target),
        dwagon_ip=cast(str, arguments.dwagon_ip),
        fwuff_ip=cast(str, arguments.fwuff_ip),
        dwagon_socket_interface=cast(str, arguments.dwagon_socket_interface),
        fwuff_socket_interface=cast(str, arguments.fwuff_socket_interface),
        distributed_port=ports[0],
        stage_ports=cast(tuple[int, int, int], ports[1:]),
        hca_devices=cast(tuple[str, ...], arguments.hca_devices),
        pipeline_layer_partition=pipeline_layer_partition,
        benchmark_concurrencies=benchmark_concurrencies,
        context_length=context_length,
        maximum_total_tokens=maximum_total_tokens,
        benchmark_input_tokens=benchmark_input_tokens,
        benchmark_output_tokens=benchmark_output_tokens,
        static_memory_fraction=cast(float, arguments.static_memory_fraction),
        readiness_timeout_seconds=cast(float, arguments.readiness_timeout_seconds),
        request_timeout_seconds=cast(float, arguments.request_timeout_seconds),
        cleanup_timeout_seconds=cast(float, arguments.cleanup_timeout_seconds),
    )


def main() -> int:
    try:
        config = _config_from_arguments(_parser().parse_args())
        payload = run_benchmark(config)
    except (
        Glm52Pp3BenchmarkError,
        Pp3DiagnosticError,
        httpx.HTTPError,
        OSError,
        ValueError,
    ) as error:
        print(f"GLM-5.2 PP3 benchmark failed: {error}", file=sys.stderr)
        return 1
    raw_cases = cast(list[object], payload["benchmark_cases"])
    case_summaries: list[JsonObject] = []
    for raw_case in raw_cases:
        case = cast(dict[str, object], raw_case)
        raw_requests = cast(Sequence[object], case["requests"])
        request_summaries: list[JsonValue] = []
        for raw_request in raw_requests:
            request = cast(dict[str, object], raw_request)
            observation = cast(dict[str, object], request["observation"])
            request_summaries.append(
                {
                    "request_index": cast(int, request["request_index"]),
                    "prompt_tokens": cast(int, observation["prompt_tokens"]),
                    "completion_tokens": cast(int, observation["completion_tokens"]),
                    "ttft_seconds": cast(float, observation["ttft_seconds"]),
                    "generation_tokens_per_second": cast(
                        float,
                        observation["generation_tokens_per_second"],
                    ),
                    "end_to_end_seconds": cast(
                        float,
                        observation["end_to_end_seconds"],
                    ),
                }
            )
        case_summaries.append(
            {
                "concurrency": cast(int, case["concurrency"]),
                "case_wall_seconds": cast(float, case["case_wall_seconds"]),
                "aggregate_output_tokens_per_second": cast(
                    float,
                    case["aggregate_output_tokens_per_second"],
                ),
                "requests": request_summaries,
            }
        )
    print(
        json.dumps(
            {
                "pipeline_layer_partition": payload["pipeline_layer_partition"],
                "cases": case_summaries,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    print(config.result_directory / "glm52-pp3-benchmark-result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
