#!/usr/bin/env python3
"""Persistent TP1 GLM-5.2 MTP draft probe for ``fwuff``.

This is the deliberately small rung between the EDR wire-floor probe and a
full SGLang scheduler integration.  It keeps the layer-78 NextN model, draft
KV, and AMX expert state resident on fwuff.  The target side sends accepted
token IDs plus BF16 target hidden rows; fwuff returns greedy proposal IDs and
component timings.

The request identity/round/action contract follows the useful part of the
SPECTRE remote-drafter protocol (SGLang PR #22272), while carrying target
hidden states required by native GLM MTP.  It does not claim asynchronous
rollback or target/verifier integration yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Literal, TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Action = Literal["OPEN", "ADVANCE", "FINISH", "ABORT"]

SCHEMA_VERSION: Final = 2
MAGIC: Final = b"GLM52DR1"
FRAME_PREFIX: Final = struct.Struct("!8sII")
HIDDEN_SIZE: Final = 6_144
BF16_BYTES: Final = 2
HIDDEN_ROW_BYTES: Final = HIDDEN_SIZE * BF16_BYTES
MAXIMUM_HEADER_BYTES: Final = 64 * 1024
MAXIMUM_HIDDEN_ROWS_PER_REQUEST: Final = 512
# Keep the wire/runtime bounded while admitting the next fixed-depth
# experiments above the original K4 baseline. The target's top-k-one tree and
# page allocator are depth-generic; eight remains tiny relative to the 8K
# draft KV reservation and keeps decode advances below the prefill batching
# threshold.
MAXIMUM_DRAFT_DEPTH: Final = 8
# Batch only prefill-sized committed bursts. At the current bound, decode
# verification can return at most nine rows; retaining DECODE below the
# threshold preserves the measured K4 trajectory and future fixed-depth runs.
MINIMUM_BATCHED_COMMITTED_ROWS: Final = 64
MAXIMUM_RAW_TIMING_SAMPLES: Final = 64
DEFAULT_BIND_ADDRESS: Final = "10.44.0.2"
DEFAULT_REMOTE_ADDRESS: Final = "10.44.0.2"
DEFAULT_PORT: Final = 18_680
DEFAULT_MODEL_PATH: Final = "/mnt/sanic/glm52-AMXINT4-W8A16-hybrid"
DEFAULT_KT_WEIGHT_PATH: Final = "/mnt/sanic/glm52-AMXINT4"
DEFAULT_CONTEXT_LENGTH: Final = 8_192
DEFAULT_MAXIMUM_TOTAL_TOKENS: Final = 8_192
DEFAULT_RANDOM_SEED: Final = 20_260_725
REMOTE_HEADER_WEIGHTS: Final = frozenset(
    {
        "model.embed_tokens.weight",
        "lm_head.qweight",
        "lm_head.scales",
    }
)


class RemoteDraftError(RuntimeError):
    """Expected fail-closed remote-draft error."""


@dataclass(frozen=True, slots=True)
class DraftRequest:
    request_id: str
    round_id: int
    action: Action
    token_ids: tuple[int, ...]
    hidden_rows: int
    payload_bytes: int
    draft_depth: int = 1

    @classmethod
    def from_header(cls, header: Mapping[str, object]) -> "DraftRequest":
        if header.get("schema_version") != SCHEMA_VERSION:
            raise RemoteDraftError("unsupported remote-draft schema version")
        request_id = header.get("request_id")
        round_id = header.get("round_id")
        action = header.get("action")
        raw_token_ids = header.get("token_ids")
        hidden_rows = header.get("hidden_rows")
        payload_bytes = header.get("payload_bytes")
        draft_depth = header.get("draft_depth")
        if not isinstance(request_id, str) or not request_id:
            raise RemoteDraftError("request_id must be a non-empty string")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
            raise RemoteDraftError("round_id must be a nonnegative integer")
        if action not in {"OPEN", "ADVANCE", "FINISH", "ABORT"}:
            raise RemoteDraftError("action is not admitted")
        if not isinstance(raw_token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in raw_token_ids
        ):
            raise RemoteDraftError("token_ids must be nonnegative integers")
        if (
            isinstance(hidden_rows, bool)
            or not isinstance(hidden_rows, int)
            or hidden_rows < 0
            or hidden_rows > MAXIMUM_HIDDEN_ROWS_PER_REQUEST
        ):
            raise RemoteDraftError("hidden_rows is outside the admitted range")
        if (
            isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes < 0
        ):
            raise RemoteDraftError("payload_bytes must be nonnegative")
        if (
            isinstance(draft_depth, bool)
            or not isinstance(draft_depth, int)
            or not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH
        ):
            raise RemoteDraftError(f"draft_depth must be in [1, {MAXIMUM_DRAFT_DEPTH}]")

        token_ids = tuple(cast(list[int], raw_token_ids))
        expected_payload_bytes = hidden_rows * HIDDEN_ROW_BYTES
        if payload_bytes != expected_payload_bytes:
            raise RemoteDraftError(
                f"payload has {payload_bytes} bytes; expected {expected_payload_bytes}"
            )
        if action in {"OPEN", "ADVANCE"}:
            if hidden_rows == 0 or len(token_ids) != hidden_rows:
                raise RemoteDraftError(
                    "OPEN/ADVANCE require one accepted token per hidden row"
                )
        elif hidden_rows != 0 or token_ids:
            raise RemoteDraftError("FINISH/ABORT must not carry model payload")
        if action == "OPEN" and round_id != 0:
            raise RemoteDraftError("OPEN must use round_id zero")
        return cls(
            request_id=request_id,
            round_id=round_id,
            action=cast(Action, action),
            token_ids=token_ids,
            hidden_rows=hidden_rows,
            payload_bytes=payload_bytes,
            draft_depth=draft_depth,
        )

    def to_header(self) -> JsonObject:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "round_id": self.round_id,
            "action": self.action,
            "token_ids": list(self.token_ids),
            "hidden_rows": self.hidden_rows,
            "payload_dtype": "BF16",
            "payload_bytes": self.payload_bytes,
            "draft_depth": self.draft_depth,
        }


@dataclass(frozen=True, slots=True)
class ForwardTiming:
    sequence_length: int
    h2d_milliseconds: float
    allocator_wall_milliseconds: float
    allocator_cuda_milliseconds: float
    metadata_wall_milliseconds: float
    metadata_cuda_milliseconds: float
    model_milliseconds: float
    proposal_milliseconds: float
    total_milliseconds: float
    # One timing sample describes one model forward.  Committed prefill can
    # cover multiple shifted token/target-hidden pairs in one ordinary EXTEND;
    # decode and tentative proposal forwards always cover one row.
    row_count: int = 1


@dataclass(frozen=True, slots=True)
class ForwardResult:
    proposal_id: int
    hidden_state: Any
    timing: ForwardTiming


@dataclass(frozen=True, slots=True)
class CommittedExtendLayout:
    prefix_length: int
    row_count: int
    sequence_length: int


@dataclass(frozen=True, slots=True)
class DraftResponse:
    request_id: str
    round_id: int
    status: Literal["ok", "error"]
    proposal_ids: tuple[int, ...] = ()
    sequence_length: int = 0
    timings: tuple[ForwardTiming, ...] = ()
    committed_forward_count: int = 0
    tentative_forward_count: int = 0
    error: str | None = None

    def to_header(self) -> JsonObject:
        value: JsonObject = {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "round_id": self.round_id,
            "status": self.status,
            "proposal_ids": list(self.proposal_ids),
            "sequence_length": self.sequence_length,
            "timings": [cast(JsonValue, asdict(timing)) for timing in self.timings],
            "committed_forward_count": self.committed_forward_count,
            "tentative_forward_count": self.tentative_forward_count,
        }
        if self.error is not None:
            value["error"] = self.error
        if (
            len(self.timings) > MAXIMUM_RAW_TIMING_SAMPLES
            or len(_encode_header_bytes(value)) > MAXIMUM_HEADER_BYTES
        ):
            value["timings"] = []
            value["timing_encoding"] = "summary-v1"
            value["timing_summary"] = cast(
                JsonValue,
                summarize_timing_groups(
                    self.timings,
                    committed_forward_count=self.committed_forward_count,
                    tentative_forward_count=self.tentative_forward_count,
                ),
            )
        return value


@dataclass(slots=True)
class SequenceState:
    request_id: str
    last_round_id: int
    draft_depth: int
    sequence_length: int = 0

    def admit(self, request: DraftRequest) -> None:
        if request.request_id != self.request_id:
            raise RemoteDraftError("another request already owns the draft slot")
        if request.action == "OPEN":
            raise RemoteDraftError("request is already open")
        if request.draft_depth != self.draft_depth:
            raise RemoteDraftError("draft_depth cannot change while a request is open")
        if request.round_id != self.last_round_id + 1:
            raise RemoteDraftError(
                f"round_id {request.round_id} does not follow {self.last_round_id}"
            )
        self.last_round_id = request.round_id


def _encode_header_bytes(header: Mapping[str, object]) -> bytes:
    return json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def encode_frame(header: Mapping[str, object], payload: bytes = b"") -> bytes:
    header_bytes = _encode_header_bytes(header)
    if not header_bytes or len(header_bytes) > MAXIMUM_HEADER_BYTES:
        raise RemoteDraftError("frame header size is outside the admitted range")
    return (
        FRAME_PREFIX.pack(MAGIC, len(header_bytes), len(payload))
        + header_bytes
        + payload
    )


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise EOFError("peer closed the connection")
        offset += received
    return bytes(result)


def receive_frame_header(
    connection: socket.socket,
) -> tuple[JsonObject, int]:
    prefix = _recv_exact(connection, FRAME_PREFIX.size)
    magic, header_size, payload_size = FRAME_PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise RemoteDraftError("invalid remote-draft frame magic")
    if header_size == 0 or header_size > MAXIMUM_HEADER_BYTES:
        raise RemoteDraftError("invalid remote-draft header size")
    if payload_size > MAXIMUM_HIDDEN_ROWS_PER_REQUEST * HIDDEN_ROW_BYTES:
        raise RemoteDraftError("remote-draft payload exceeds the admitted maximum")
    try:
        header = json.loads(_recv_exact(connection, header_size))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteDraftError("remote-draft header is not valid JSON") from error
    if not isinstance(header, dict):
        raise RemoteDraftError("remote-draft header must be a JSON object")
    return cast(JsonObject, header), payload_size


def send_frame(
    connection: socket.socket,
    header: Mapping[str, object],
    payload: bytes = b"",
) -> None:
    connection.sendall(encode_frame(header, payload))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_timings(timings: Sequence[ForwardTiming]) -> JsonObject:
    if not timings:
        raise ValueError("cannot summarize no timings")
    if any(timing.row_count <= 0 for timing in timings):
        raise ValueError("timing row counts must be positive")
    output: JsonObject = {
        "iterations": len(timings),
        "rows": sum(timing.row_count for timing in timings),
    }
    for field_name in (
        "h2d_milliseconds",
        "allocator_wall_milliseconds",
        "allocator_cuda_milliseconds",
        "metadata_wall_milliseconds",
        "metadata_cuda_milliseconds",
        "model_milliseconds",
        "proposal_milliseconds",
        "total_milliseconds",
    ):
        values = [float(getattr(timing, field_name)) for timing in timings]
        output[field_name] = {
            "minimum": min(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "maximum": max(values),
            "mean": sum(values) / len(values),
        }
    return output


def summarize_timing_groups(
    timings: Sequence[ForwardTiming],
    *,
    committed_forward_count: int,
    tentative_forward_count: int,
) -> JsonObject:
    if (
        committed_forward_count < 0
        or tentative_forward_count < 0
        or committed_forward_count + tentative_forward_count != len(timings)
    ):
        raise ValueError("timing group counts do not match timing samples")
    committed = timings[:committed_forward_count]
    tentative = timings[
        committed_forward_count : committed_forward_count + tentative_forward_count
    ]
    return {
        "all": cast(JsonValue, summarize_timings(timings)),
        "committed": (
            cast(JsonValue, summarize_timings(committed)) if committed else None
        ),
        "tentative": (
            cast(JsonValue, summarize_timings(tentative)) if tentative else None
        ),
        "first_sequence_length": timings[0].sequence_length,
        "last_sequence_length": timings[-1].sequence_length,
    }


def timing_sample(timing: ForwardTiming) -> JsonObject:
    value = cast(JsonObject, asdict(timing))
    accounted = (
        timing.h2d_milliseconds
        + timing.allocator_cuda_milliseconds
        + timing.metadata_cuda_milliseconds
        + timing.model_milliseconds
        + timing.proposal_milliseconds
    )
    value["unaccounted_milliseconds"] = max(
        0.0,
        timing.total_milliseconds - accounted,
    )
    stage_values = {
        "h2d": timing.h2d_milliseconds,
        "allocator_wall": timing.allocator_wall_milliseconds,
        "allocator_cuda": timing.allocator_cuda_milliseconds,
        "metadata_wall": timing.metadata_wall_milliseconds,
        "metadata_cuda": timing.metadata_cuda_milliseconds,
        "model": timing.model_milliseconds,
        "proposal": timing.proposal_milliseconds,
        "unaccounted": cast(float, value["unaccounted_milliseconds"]),
    }
    value["largest_stage"] = max(stage_values, key=stage_values.__getitem__)
    return value


def _parse_forward_timing(value: Mapping[str, object]) -> ForwardTiming:
    fields: dict[str, int | float] = {}
    for name in ForwardTiming.__dataclass_fields__:
        try:
            raw_value = value.get(name, 1) if name == "row_count" else value[name]
            fields[name] = (
                int(raw_value)
                if name in {"sequence_length", "row_count"}
                else float(raw_value)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RemoteDraftError(f"invalid forward timing field: {name}") from error
    timing = ForwardTiming(**fields)
    if timing.row_count <= 0:
        raise RemoteDraftError("forward timing row_count must be positive")
    return timing


def _install_remote_weight_loader() -> None:
    """Patch only this process to load the standalone embed/head plus layer 78.

    Patch 0009 supplies the fail-closed 38-tensor selector but intentionally
    leaves the local TP2 loader unchanged.  This external harness therefore
    installs a process-local loader seam; the immutable runtime remains
    untouched.
    """

    from sglang.srt.layers.moe.utils import get_kt_ep_weight_layer_index
    from sglang.srt.model_loader.loader import (
        SAFE_WEIGHTS_INDEX_NAME,
        DefaultModelLoader,
    )
    from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
    from sglang.srt.speculative.kt_mtp import (
        select_glm52_remote_mtp_weights,
    )

    if getattr(DefaultModelLoader, "_glm52_remote_draft_patch", False):
        return

    original_filter = DefaultModelLoader._get_glm52_kt_mtp_weight_filter
    original_load_weights = DeepseekV3ForCausalLMNextN.load_weights

    def remote_filter(
        *,
        source: Any,
        hf_folder: str,
        hf_weights_files: list[str],
        use_safetensors: bool,
    ) -> tuple[list[str], set[str]] | None:
        model_config = source.model_config
        physical_layer_index = get_kt_ep_weight_layer_index(0)
        if (
            model_config is None
            or not model_config.is_draft_model
            or physical_layer_index == 0
        ):
            return original_filter(
                source=source,
                hf_folder=hf_folder,
                hf_weights_files=hf_weights_files,
                use_safetensors=use_safetensors,
            )
        if (
            getattr(model_config.hf_config, "model_type", None) != "glm_moe_dsa"
            or physical_layer_index != 78
            or not use_safetensors
        ):
            raise RemoteDraftError(
                "remote loader only admits a safetensors GLM-5.2 layer-78 draft"
            )
        index_path = Path(hf_folder) / SAFE_WEIGHTS_INDEX_NAME
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise RemoteDraftError(
                f"cannot load remote draft index {index_path}"
            ) from error
        allowed_names, required_shards = select_glm52_remote_mtp_weights(
            weight_map,
            physical_layer_index,
        )
        selected_files = [
            path
            for path in hf_weights_files
            if os.path.basename(path) in required_shards
        ]
        selected_shards = {os.path.basename(path) for path in selected_files}
        if selected_shards != required_shards:
            raise RemoteDraftError(
                "remote draft loader cannot resolve all selected checkpoint shards"
            )
        return selected_files, allowed_names

    def remote_load_weights(
        model: Any,
        weights: Iterable[tuple[str, Any]],
    ) -> None:
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        parameters = dict(model.named_parameters())
        loaded_headers: set[str] = set()

        def layer_weights() -> Iterable[tuple[str, Any]]:
            for name, tensor in weights:
                if name not in REMOTE_HEADER_WEIGHTS:
                    yield name, tensor
                    continue
                parameter = parameters.get(name)
                if parameter is None:
                    raise RemoteDraftError(
                        f"standalone draft parameter is missing: {name}"
                    )
                weight_loader = getattr(
                    parameter,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(parameter, tensor)
                loaded_headers.add(name)

        original_load_weights(model, layer_weights())
        if loaded_headers != REMOTE_HEADER_WEIGHTS:
            missing = sorted(REMOTE_HEADER_WEIGHTS - loaded_headers)
            raise RemoteDraftError(
                f"standalone draft did not load required header weights: {missing}"
            )

    DefaultModelLoader._get_glm52_kt_mtp_weight_filter = staticmethod(remote_filter)
    DeepseekV3ForCausalLMNextN.load_weights = remote_load_weights
    DefaultModelLoader._glm52_remote_draft_patch = True


def _build_server_args(
    *,
    model_path: str,
    kt_weight_path: str,
    context_length: int,
    maximum_total_tokens: int,
    memory_fraction_static: float,
) -> Any:
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        skip_tokenizer_init=True,
        load_format="safetensors",
        trust_remote_code=True,
        context_length=context_length,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        mem_fraction_static=memory_fraction_static,
        max_running_requests=1,
        max_total_tokens=maximum_total_tokens,
        chunked_prefill_size=min(2_048, maximum_total_tokens),
        max_prefill_tokens=maximum_total_tokens,
        device="cuda",
        tp_size=1,
        pp_size=1,
        ep_size=1,
        random_seed=DEFAULT_RANDOM_SEED,
        attention_backend="flashinfer",
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=model_path,
        speculative_draft_load_format="safetensors",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=2,
        speculative_moe_a2a_backend="none",
        moe_a2a_backend="none",
        kt_weight_path=kt_weight_path,
        kt_method="AMXINT4",
        kt_cpuinfer=60,
        kt_threadpool_count=2,
        kt_numa_nodes=[0, 0],
        kt_num_gpu_experts=0,
        kt_max_deferred_experts_per_token=0,
        disable_shared_experts_fusion=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        enable_dp_attention=False,
        enable_eplb=False,
        record_kt_gpu_expert_distribution=False,
        kt_enable_dynamic_expert_update=False,
        kt_stream_prefill=False,
        init_expert_location="trivial",
    )


class RemoteDraftRuntime:
    """One persistent TP1 layer-78 worker and one active draft sequence."""

    def __init__(
        self,
        *,
        model_path: str,
        kt_weight_path: str,
        context_length: int,
        maximum_total_tokens: int,
        memory_fraction_static: float,
        maximum_rows_per_request: int,
        preallocate_draft_kv: bool = True,
    ) -> None:
        if maximum_rows_per_request <= 0:
            raise RemoteDraftError("maximum_rows_per_request must be positive")
        configured_kv_b_backend = os.environ.get("SGLANG_MLA_KV_B_W8_BACKEND")
        if configured_kv_b_backend not in {None, "marlin"}:
            raise RemoteDraftError(
                "fwuff remote draft requires SGLANG_MLA_KV_B_W8_BACKEND=marlin"
            )
        os.environ["SGLANG_MLA_KV_B_W8_BACKEND"] = "marlin"
        _install_remote_weight_loader()

        import torch
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.layers.dp_attention import initialize_dp_attention
        from sglang.srt.layers.moe.utils import (
            speculative_kt_ep_context,
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.speculative.kt_mtp import admit_glm52_kt_remote_draft
        from transformers import AutoConfig

        self.torch = torch
        self.server_args = _build_server_args(
            model_path=model_path,
            kt_weight_path=kt_weight_path,
            context_length=context_length,
            maximum_total_tokens=maximum_total_tokens,
            memory_fraction_static=memory_fraction_static,
        )
        # Ordinarily the colocated target runner fills these two draft-only
        # fields and initializes the process groups.  This standalone worker
        # has no target by design, so establish the exact TP1 equivalents.
        self.server_args.draft_runner_cache_size = maximum_total_tokens
        self.server_args.max_num_reqs = 1
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.admission = admit_glm52_kt_remote_draft(
            self.server_args,
            hf_config,
        )
        if not self.admission.enabled or self.admission.physical_layer_index != 78:
            raise RemoteDraftError(
                f"remote draft admission is disabled: {self.admission.reason}"
            )
        torch.cuda.set_device(0)
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29680",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            attention_data_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            attention_context_model_parallel_size=1,
            moe_data_model_parallel_size=1,
        )
        draft_model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=model_path,
            is_draft_model=True,
        )
        initialize_dp_attention(
            server_args=self.server_args,
            model_config=draft_model_config,
        )
        self._contexts: tuple[Callable[[], Any], ...] = (
            speculative_moe_backend_context,
            speculative_moe_a2a_backend_context,
            lambda: speculative_kt_ep_context(
                enabled=True,
                physical_layer_index=78,
            ),
        )
        with (
            self._contexts[0](),
            self._contexts[1](),
            self._contexts[2](),
        ):
            self.worker = TpModelWorker(
                server_args=self.server_args,
                gpu_id=0,
                tp_rank=0,
                pp_rank=0,
                dp_rank=None,
                moe_ep_rank=0,
                attn_cp_rank=0,
                moe_dp_rank=0,
                nccl_port=29_680,
                is_draft_worker=True,
            )
        self.model_runner = self.worker.model_runner
        self.device = self.model_runner.device
        self.maximum_rows_per_request = maximum_rows_per_request
        maximum_payload_bytes = maximum_rows_per_request * HIDDEN_ROW_BYTES
        self.pinned_payload = torch.empty(
            maximum_payload_bytes,
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.pinned_payload_view = memoryview(self.pinned_payload.numpy())
        self.pinned_input_ids = torch.empty(
            (maximum_rows_per_request,),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.device_hidden = torch.empty(
            (maximum_rows_per_request, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.input_ids = torch.empty(
            (1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.extend_input_ids = torch.empty(
            (maximum_rows_per_request,),
            dtype=torch.int64,
            device=self.device,
        )
        self.request_pool_indices = torch.empty(
            (1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.sequence_lengths = torch.empty(
            (1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.sequence_lengths_cpu = torch.empty((1,), dtype=torch.int64)
        self.empty_last_cache_location = torch.full(
            (1,),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self._h2d_start = torch.cuda.Event(enable_timing=True)
        self._h2d_end = torch.cuda.Event(enable_timing=True)
        self._allocator_start = torch.cuda.Event(enable_timing=True)
        self._allocator_end = torch.cuda.Event(enable_timing=True)
        self._metadata_start = torch.cuda.Event(enable_timing=True)
        self._metadata_end = torch.cuda.Event(enable_timing=True)
        self._model_start = torch.cuda.Event(enable_timing=True)
        self._model_end = torch.cuda.Event(enable_timing=True)
        self.preallocate_draft_kv = preallocate_draft_kv
        self.reserved_cache_locations: Any | None = None
        self.sequence_state: SequenceState | None = None
        self.request_pool_index: int | None = None

    def payload_view(self, payload_bytes: int) -> memoryview:
        if payload_bytes > len(self.pinned_payload_view):
            raise RemoteDraftError("payload exceeds the pinned receive buffer")
        return self.pinned_payload_view[:payload_bytes]

    def reset_sequence(self, request_id: str, draft_depth: int = 1) -> None:
        self.model_runner.req_to_token_pool.clear()
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator.clear()
        if not self.model_runner.req_to_token_pool.free_slots:
            raise RemoteDraftError("draft request pool has no free slot")
        self.request_pool_index = self.model_runner.req_to_token_pool.free_slots.pop(0)
        self.request_pool_indices[0] = self.request_pool_index
        self.reserved_cache_locations = None
        if self.preallocate_draft_kv:
            reservable_tokens = allocator.available_size()
            reserved = allocator.alloc(reservable_tokens)
            if reserved is None or len(reserved) != reservable_tokens:
                raise RemoteDraftError(
                    "cannot reserve the single-request draft KV pool"
                )
            request_mapping = self.model_runner.req_to_token_pool.req_to_token[
                self.request_pool_index
            ]
            usable_tokens = min(len(request_mapping), len(reserved))
            request_mapping[:usable_tokens].copy_(reserved[:usable_tokens])
            self.reserved_cache_locations = reserved[:usable_tokens]
        self.sequence_state = SequenceState(
            request_id=request_id,
            last_round_id=0,
            draft_depth=draft_depth,
        )

    def finish_sequence(self) -> None:
        self.sequence_state = None
        self.request_pool_index = None
        self.reserved_cache_locations = None
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    def _prepare_committed_extend(
        self,
        row_count: int,
    ) -> tuple[CommittedExtendLayout, Any]:
        if self.sequence_state is None or self.request_pool_index is None:
            raise RemoteDraftError("no draft sequence is open")
        if row_count <= 1:
            raise RemoteDraftError("batched draft EXTEND requires at least two rows")
        if row_count > self.maximum_rows_per_request:
            raise RemoteDraftError("draft EXTEND exceeds the pinned row capacity")
        if self.reserved_cache_locations is None:
            raise RemoteDraftError(
                "batched draft EXTEND requires the full preallocated KV mapping"
            )
        prefix_length = self.sequence_state.sequence_length
        new_sequence_length = prefix_length + row_count
        if new_sequence_length > self.server_args.context_length:
            raise RemoteDraftError("draft sequence reached its context limit")
        if new_sequence_length > len(self.reserved_cache_locations):
            raise RemoteDraftError("reserved draft KV pool is exhausted")
        out_cache_loc = self.reserved_cache_locations[prefix_length:new_sequence_length]
        if len(out_cache_loc) != row_count:
            raise RemoteDraftError("reserved draft KV mapping is incomplete")
        return (
            CommittedExtendLayout(
                prefix_length=prefix_length,
                row_count=row_count,
                sequence_length=new_sequence_length,
            ),
            out_cache_loc,
        )

    def _forward_one(self, token_id: int, hidden_row: Any) -> ForwardResult:
        import torch
        from sglang.srt.managers.schedule_batch import ModelWorkerBatch
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if self.sequence_state is None or self.request_pool_index is None:
            raise RemoteDraftError("no draft sequence is open")
        if self.sequence_state.sequence_length >= self.server_args.context_length:
            raise RemoteDraftError("draft sequence reached its context limit")

        total_start = time.perf_counter()
        self._h2d_start.record()
        if hidden_row.device.type == "cpu":
            self.device_hidden[0].copy_(hidden_row, non_blocking=True)
            model_hidden_states = self.device_hidden[:1]
        else:
            model_hidden_states = hidden_row.reshape(1, HIDDEN_SIZE)
        self._h2d_end.record()

        new_sequence_length = self.sequence_state.sequence_length + 1
        self.input_ids[0] = token_id
        self.sequence_lengths[0] = new_sequence_length
        self.sequence_lengths_cpu[0] = new_sequence_length
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator_wall_start = time.perf_counter()
        self._allocator_start.record()
        if self.reserved_cache_locations is not None:
            if new_sequence_length > len(self.reserved_cache_locations):
                raise RemoteDraftError("reserved draft KV pool is exhausted")
            out_cache_loc = self.reserved_cache_locations[
                new_sequence_length - 1 : new_sequence_length
            ]
        elif allocator.page_size == 1:
            out_cache_loc = allocator.alloc(1)
        else:
            prefix_length = new_sequence_length - 1
            if prefix_length == 0:
                last_cache_location = self.empty_last_cache_location
            else:
                last_cache_location = self.model_runner.req_to_token_pool.req_to_token[
                    self.request_pool_index,
                    prefix_length - 1,
                ].reshape(1)
            out_cache_loc = allocator.alloc_decode(
                seq_lens=self.sequence_lengths,
                seq_lens_cpu=self.sequence_lengths_cpu,
                last_loc=last_cache_location,
            )
        self._allocator_end.record()
        allocator_wall_milliseconds = (
            time.perf_counter() - allocator_wall_start
        ) * 1_000
        if out_cache_loc is None:
            raise RemoteDraftError("draft KV pool is exhausted")
        self.model_runner.req_to_token_pool.req_to_token[
            self.request_pool_index,
            new_sequence_length - 1,
        ] = out_cache_loc[0]
        worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.DECODE,
            input_ids=self.input_ids,
            req_pool_indices=self.request_pool_indices,
            seq_lens=self.sequence_lengths,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=self.sequence_lengths_cpu,
            seq_lens_sum=new_sequence_length,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=ForwardMode.DECODE,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            reqs=[SimpleNamespace(rid=self.sequence_state.request_id)],
        )
        forward_batch = ForwardBatch.init_new(worker_batch, self.model_runner)
        metadata_wall_start = time.perf_counter()
        self._metadata_start.record()
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        self._metadata_end.record()
        metadata_wall_milliseconds = (time.perf_counter() - metadata_wall_start) * 1_000
        forward_batch.spec_info = SimpleNamespace(
            hidden_states=model_hidden_states,
        )

        self._model_start.record()
        with (
            self._contexts[0](),
            self._contexts[1](),
            self._contexts[2](),
            torch.inference_mode(),
        ):
            logits_output = self.model_runner.forward(
                forward_batch,
                skip_attn_backend_init=True,
            ).logits_output
        self._model_end.record()
        self._model_end.synchronize()
        proposal_start = time.perf_counter()
        proposal_id = int(
            torch.argmax(logits_output.next_token_logits[0], dim=-1).item()
        )
        proposal_milliseconds = (time.perf_counter() - proposal_start) * 1_000
        self.sequence_state.sequence_length = new_sequence_length
        return ForwardResult(
            proposal_id=proposal_id,
            hidden_state=logits_output.hidden_states[0],
            timing=ForwardTiming(
                sequence_length=new_sequence_length,
                h2d_milliseconds=self._h2d_start.elapsed_time(self._h2d_end),
                allocator_wall_milliseconds=allocator_wall_milliseconds,
                allocator_cuda_milliseconds=self._allocator_start.elapsed_time(
                    self._allocator_end
                ),
                metadata_wall_milliseconds=metadata_wall_milliseconds,
                metadata_cuda_milliseconds=self._metadata_start.elapsed_time(
                    self._metadata_end
                ),
                model_milliseconds=self._model_start.elapsed_time(self._model_end),
                proposal_milliseconds=proposal_milliseconds,
                total_milliseconds=(time.perf_counter() - total_start) * 1_000,
                row_count=1,
            ),
        )

    def _forward_many(
        self,
        token_ids: Sequence[int],
        hidden_rows: Any,
    ) -> ForwardResult:
        """Commit shifted MTP rows with one ordinary draft-model EXTEND.

        ``token_ids[i]`` and ``hidden_rows[i]`` remain paired exactly as they
        were in the serial DECODE path.  The sole request already owns a full
        preallocated page-64 mapping, so EXTEND only selects the committed
        prefix slice; it never replaces or incrementally rebuilds that map.
        """

        import torch
        from sglang.srt.managers.schedule_batch import ModelWorkerBatch
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        row_count = len(token_ids)
        layout, out_cache_loc = self._prepare_committed_extend(row_count)
        state = cast(SequenceState, self.sequence_state)
        request_mapping = self.model_runner.req_to_token_pool.req_to_token[
            self.request_pool_index
        ]
        committed_mapping = request_mapping[
            layout.prefix_length : layout.sequence_length
        ]
        if not torch.equal(committed_mapping, out_cache_loc):
            raise RemoteDraftError(
                "draft EXTEND cache slice differs from the preallocated request map"
            )
        prefix_mapping_before = request_mapping[: layout.prefix_length].clone()
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator_available_before = allocator.available_size()

        total_start = time.perf_counter()
        self._h2d_start.record()
        # Populate the fixed pinned ID staging buffer without allocating a
        # device tensor.  Hidden rows already reside in the pinned wire buffer.
        for index, token_id in enumerate(token_ids):
            self.pinned_input_ids[index] = token_id
        self.extend_input_ids[:row_count].copy_(
            self.pinned_input_ids[:row_count],
            non_blocking=True,
        )
        self.device_hidden[:row_count].copy_(hidden_rows, non_blocking=True)
        self._h2d_end.record()

        self.sequence_lengths[0] = layout.sequence_length
        self.sequence_lengths_cpu[0] = layout.sequence_length
        allocator_wall_start = time.perf_counter()
        self._allocator_start.record()
        self._allocator_end.record()
        allocator_wall_milliseconds = (
            time.perf_counter() - allocator_wall_start
        ) * 1_000

        worker_batch = ModelWorkerBatch(
            forward_mode=ForwardMode.EXTEND,
            input_ids=self.extend_input_ids[:row_count],
            req_pool_indices=self.request_pool_indices,
            seq_lens=self.sequence_lengths,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=self.sequence_lengths_cpu,
            seq_lens_sum=layout.sequence_length,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=row_count,
            extend_seq_lens=[row_count],
            extend_prefix_lens=[layout.prefix_length],
            extend_logprob_start_lens=[0],
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            reqs=[SimpleNamespace(rid=state.request_id)],
        )
        forward_batch = ForwardBatch.init_new(worker_batch, self.model_runner)
        metadata_wall_start = time.perf_counter()
        self._metadata_start.record()
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        self._metadata_end.record()
        metadata_wall_milliseconds = (time.perf_counter() - metadata_wall_start) * 1_000
        forward_batch.spec_info = SimpleNamespace(
            hidden_states=self.device_hidden[:row_count],
        )

        self._model_start.record()
        with (
            self._contexts[0](),
            self._contexts[1](),
            self._contexts[2](),
            torch.inference_mode(),
        ):
            logits_output = self.model_runner.forward(
                forward_batch,
                skip_attn_backend_init=True,
            ).logits_output
        self._model_end.record()
        self._model_end.synchronize()
        if allocator.available_size() != allocator_available_before:
            raise RemoteDraftError(
                "draft EXTEND mutated the KV allocator free-page state"
            )
        if not torch.equal(
            request_mapping[: layout.prefix_length],
            prefix_mapping_before,
        ):
            raise RemoteDraftError("draft EXTEND mutated the committed prefix mapping")
        if (
            logits_output.next_token_logits.shape[0] != 1
            or logits_output.hidden_states.shape[0] != 1
        ):
            raise RemoteDraftError(
                "draft EXTEND did not return exactly the last committed row"
            )
        proposal_start = time.perf_counter()
        proposal_id = int(
            torch.argmax(logits_output.next_token_logits[0], dim=-1).item()
        )
        proposal_milliseconds = (time.perf_counter() - proposal_start) * 1_000
        state.sequence_length = layout.sequence_length
        return ForwardResult(
            proposal_id=proposal_id,
            hidden_state=logits_output.hidden_states[0],
            timing=ForwardTiming(
                sequence_length=layout.sequence_length,
                h2d_milliseconds=self._h2d_start.elapsed_time(self._h2d_end),
                allocator_wall_milliseconds=allocator_wall_milliseconds,
                allocator_cuda_milliseconds=self._allocator_start.elapsed_time(
                    self._allocator_end
                ),
                metadata_wall_milliseconds=metadata_wall_milliseconds,
                metadata_cuda_milliseconds=self._metadata_start.elapsed_time(
                    self._metadata_end
                ),
                model_milliseconds=self._model_start.elapsed_time(self._model_end),
                proposal_milliseconds=proposal_milliseconds,
                total_milliseconds=(time.perf_counter() - total_start) * 1_000,
                row_count=row_count,
            ),
        )

    def _propose_chain(
        self,
        first_result: ForwardResult,
        draft_depth: int,
    ) -> tuple[tuple[int, ...], tuple[ForwardTiming, ...]]:
        if self.sequence_state is None or self.request_pool_index is None:
            raise RemoteDraftError("no draft sequence is open")
        if not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH:
            raise RemoteDraftError("draft depth is outside the admitted range")

        committed_length = self.sequence_state.sequence_length
        allocator = self.model_runner.token_to_kv_pool_allocator
        allocator_state = (
            None
            if self.reserved_cache_locations is not None
            else allocator.backup_state()
        )
        proposal_ids = [first_result.proposal_id]
        tentative_timings: list[ForwardTiming] = []
        hidden_state = first_result.hidden_state
        try:
            for _ in range(1, draft_depth):
                result = self._forward_one(proposal_ids[-1], hidden_state)
                proposal_ids.append(result.proposal_id)
                tentative_timings.append(result.timing)
                hidden_state = result.hidden_state
        finally:
            self.sequence_state.sequence_length = committed_length
            self.sequence_lengths[0] = committed_length
            self.sequence_lengths_cpu[0] = committed_length
            if allocator_state is not None:
                allocator.restore_state(allocator_state)
                tentative_end = min(
                    committed_length + draft_depth - 1,
                    self.model_runner.req_to_token_pool.req_to_token.shape[1],
                )
                self.model_runner.req_to_token_pool.req_to_token[
                    self.request_pool_index,
                    committed_length:tentative_end,
                ].zero_()
        return tuple(proposal_ids), tuple(tentative_timings)

    def process(self, request: DraftRequest) -> DraftResponse:
        if request.hidden_rows > self.maximum_rows_per_request:
            raise RemoteDraftError("request exceeds the runtime's pinned row capacity")
        opened_here = False
        previous_round_id: int | None = None
        previous_sequence_length: int | None = None
        allocator_state: Any | None = None
        if request.action == "OPEN":
            if self.sequence_state is not None:
                raise RemoteDraftError("another draft request is already open")
            self.reset_sequence(request.request_id, request.draft_depth)
            opened_here = True
        else:
            if self.sequence_state is None:
                raise RemoteDraftError("request has no open draft state")
            previous_round_id = self.sequence_state.last_round_id
            previous_sequence_length = self.sequence_state.sequence_length
            self.sequence_state.admit(request)

        if request.action in {"FINISH", "ABORT"}:
            sequence_length = cast(SequenceState, self.sequence_state).sequence_length
            self.finish_sequence()
            return DraftResponse(
                request_id=request.request_id,
                round_id=request.round_id,
                status="ok",
                sequence_length=sequence_length,
            )

        action_start_length = cast(SequenceState, self.sequence_state).sequence_length
        if previous_sequence_length is None:
            previous_sequence_length = action_start_length

        try:
            if self.reserved_cache_locations is None:
                allocator_state = (
                    self.model_runner.token_to_kv_pool_allocator.backup_state()
                )
            hidden_rows = (
                self.pinned_payload[: request.payload_bytes]
                .view(self.torch.bfloat16)
                .view(request.hidden_rows, HIDDEN_SIZE)
            )
            committed_timings: list[ForwardTiming] = []
            if (
                request.hidden_rows >= MINIMUM_BATCHED_COMMITTED_ROWS
                and self.reserved_cache_locations is not None
            ):
                # Use EXTEND for the bulk prefix, then preserve the historical
                # DECODE kernel semantics at the committed/proposal boundary.
                # The latter is where small EXTEND-vs-DECODE numerical drift
                # otherwise enters every tentative chain.
                bulk_result = self._forward_many(
                    request.token_ids[:-1],
                    hidden_rows[:-1],
                )
                final_result = self._forward_one(
                    request.token_ids[-1],
                    hidden_rows[-1],
                )
                committed_timings.extend((bulk_result.timing, final_result.timing))
            else:
                final_result: ForwardResult | None = None
                for token_id, hidden_row in zip(
                    request.token_ids,
                    hidden_rows,
                    strict=True,
                ):
                    final_result = self._forward_one(token_id, hidden_row)
                    committed_timings.append(final_result.timing)
                if final_result is None:
                    raise RemoteDraftError(
                        "model action did not carry any accepted rows"
                    )
            proposal_ids, tentative_timings = self._propose_chain(
                final_result,
                request.draft_depth,
            )
            return DraftResponse(
                request_id=request.request_id,
                round_id=request.round_id,
                status="ok",
                proposal_ids=proposal_ids,
                sequence_length=cast(
                    SequenceState,
                    self.sequence_state,
                ).sequence_length,
                timings=(*committed_timings, *tentative_timings),
                committed_forward_count=len(committed_timings),
                tentative_forward_count=len(tentative_timings),
            )
        except Exception:
            if opened_here:
                self.finish_sequence()
            else:
                state = cast(SequenceState, self.sequence_state)
                state.last_round_id = cast(int, previous_round_id)
                state.sequence_length = previous_sequence_length
                self.sequence_lengths[0] = previous_sequence_length
                self.sequence_lengths_cpu[0] = previous_sequence_length
                if allocator_state is not None:
                    self.model_runner.token_to_kv_pool_allocator.restore_state(
                        allocator_state
                    )
                    if self.request_pool_index is not None:
                        mapping_end = min(
                            previous_sequence_length
                            + request.hidden_rows
                            + request.draft_depth
                            - 1,
                            self.model_runner.req_to_token_pool.req_to_token.shape[1],
                        )
                        self.model_runner.req_to_token_pool.req_to_token[
                            self.request_pool_index,
                            previous_sequence_length:mapping_end,
                        ].zero_()
            raise


def _recv_into(connection: socket.socket, output: memoryview) -> None:
    offset = 0
    while offset < len(output):
        received = connection.recv_into(output[offset:])
        if received == 0:
            raise EOFError("peer closed the connection during payload receive")
        offset += received


def _serve_connection(
    runtime: RemoteDraftRuntime,
    connection: socket.socket,
    peer_address: str,
) -> None:
    disconnect_reason = "peer_eof"
    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                header, wire_payload_bytes = receive_frame_header(connection)
            except EOFError:
                return
            request_id = str(header.get("request_id", "unknown"))
            round_id = (
                header.get("round_id")
                if isinstance(header.get("round_id"), int)
                else -1
            )
            try:
                request = DraftRequest.from_header(header)
                if wire_payload_bytes != request.payload_bytes:
                    raise RemoteDraftError("frame and request payload sizes disagree")
                _recv_into(
                    connection,
                    runtime.payload_view(request.payload_bytes),
                )
                response = runtime.process(request)
            except Exception as error:
                response = DraftResponse(
                    request_id=request_id,
                    round_id=cast(int, round_id),
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                )
            response_header = response.to_header()
            send_frame(connection, response_header)
            print(
                json.dumps(
                    {
                        "event": "GLM52_FWUFF_REMOTE_DRAFT_RESPONSE",
                        "peer": peer_address,
                        **response_header,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    except Exception as error:
        disconnect_reason = f"{type(error).__name__}: {error}"
        print(
            json.dumps(
                {
                    "event": "GLM52_FWUFF_REMOTE_DRAFT_CONNECTION_ERROR",
                    "peer": peer_address,
                    "error": disconnect_reason,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        sequence_state = runtime.sequence_state
        if sequence_state is not None:
            request_id = sequence_state.request_id
            sequence_length = sequence_state.sequence_length
            runtime.finish_sequence()
            print(
                json.dumps(
                    {
                        "event": "GLM52_FWUFF_REMOTE_DRAFT_DISCONNECT_CLEANUP",
                        "peer": peer_address,
                        "request_id": request_id,
                        "sequence_length": sequence_length,
                        "reason": disconnect_reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def serve(runtime: RemoteDraftRuntime, bind_address: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_address, port))
        listener.listen(1)
        print(
            json.dumps(
                {
                    "event": "GLM52_FWUFF_REMOTE_DRAFT_READY",
                    "bind_address": bind_address,
                    "port": port,
                    "admission": asdict(runtime.admission),
                    "maximum_draft_depth": MAXIMUM_DRAFT_DEPTH,
                    "draft_kv_allocation": (
                        "single_request_full_pool_reservation"
                        if runtime.preallocate_draft_kv
                        else "incremental_paged_alloc_decode"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while True:
            connection, peer = listener.accept()
            with connection:
                _serve_connection(runtime, connection, peer[0])


def _receive_response(connection: socket.socket) -> JsonObject:
    header, payload_bytes = receive_frame_header(connection)
    if payload_bytes:
        _recv_exact(connection, payload_bytes)
        raise RemoteDraftError("draft response unexpectedly carried a payload")
    return header


def run_client(
    *,
    address: str,
    port: int,
    iterations: int,
    warmup_iterations: int,
    request_id: str,
    draft_depth: int,
) -> JsonObject:
    if (
        iterations <= 0
        or warmup_iterations < 0
        or not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH
    ):
        raise RemoteDraftError("client iteration counts are invalid")
    payload = bytes(HIDDEN_ROW_BYTES)
    round_trip_milliseconds: list[float] = []
    remote_timings: list[ForwardTiming] = []
    remote_commit_timings: list[ForwardTiming] = []
    remote_tentative_timings: list[ForwardTiming] = []
    proposal_chains: list[list[int]] = []
    raw_samples: list[JsonObject] = []
    with socket.create_connection((address, port), timeout=30.0) as connection:
        connection.settimeout(300.0)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        total_iterations = warmup_iterations + iterations
        for index in range(total_iterations):
            request = DraftRequest(
                request_id=request_id,
                round_id=index,
                action="OPEN" if index == 0 else "ADVANCE",
                token_ids=(index % 1_000,),
                hidden_rows=1,
                payload_bytes=HIDDEN_ROW_BYTES,
                draft_depth=draft_depth,
            )
            start = time.perf_counter()
            send_frame(connection, request.to_header(), payload)
            response = _receive_response(connection)
            elapsed_milliseconds = (time.perf_counter() - start) * 1_000
            if response.get("status") != "ok":
                raise RemoteDraftError(
                    f"fwuff draft rejected round {index}: {response.get('error')}"
                )
            raw_proposals = response.get("proposal_ids")
            raw_timings = response.get("timings")
            if (
                not isinstance(raw_proposals, list)
                or len(raw_proposals) != draft_depth
                or not isinstance(raw_timings, list)
                or len(raw_timings) != draft_depth
                or any(not isinstance(item, dict) for item in raw_timings)
                or response.get("committed_forward_count") != 1
                or response.get("tentative_forward_count") != draft_depth - 1
            ):
                raise RemoteDraftError("fwuff returned an invalid draft response")
            parsed_timings = [
                _parse_forward_timing(cast(dict[str, object], raw_timing))
                for raw_timing in raw_timings
            ]
            if index >= warmup_iterations:
                round_trip_milliseconds.append(elapsed_milliseconds)
                proposal_chains.append([int(value) for value in raw_proposals])
                remote_timings.extend(parsed_timings)
                remote_commit_timings.append(parsed_timings[0])
                remote_tentative_timings.extend(parsed_timings[1:])
                raw_samples.append(
                    {
                        "round_id": index,
                        "sequence_length": int(response.get("sequence_length", -1)),
                        "round_trip_milliseconds": elapsed_milliseconds,
                        "remote_timings": [
                            timing_sample(timing) for timing in parsed_timings
                        ],
                    }
                )
        finish_request = DraftRequest(
            request_id=request_id,
            round_id=total_iterations,
            action="FINISH",
            token_ids=(),
            hidden_rows=0,
            payload_bytes=0,
            draft_depth=draft_depth,
        )
        send_frame(connection, finish_request.to_header())
        finish_response = _receive_response(connection)
        if finish_response.get("status") != "ok":
            raise RemoteDraftError("fwuff rejected FINISH")

    return {
        "kind": "glm52_fwuff_remote_draft_client_benchmark",
        "schema_version": SCHEMA_VERSION,
        "transport": {
            "address": address,
            "port": port,
            "payload_bytes_per_advance": HIDDEN_ROW_BYTES,
            "tcp_nodelay": True,
            "target_hidden_dtype": "BF16",
        },
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "draft_depth": draft_depth,
        "round_trip_milliseconds": {
            "minimum": min(round_trip_milliseconds),
            "p50": percentile(round_trip_milliseconds, 0.50),
            "p95": percentile(round_trip_milliseconds, 0.95),
            "p99": percentile(round_trip_milliseconds, 0.99),
            "maximum": max(round_trip_milliseconds),
            "mean": sum(round_trip_milliseconds) / len(round_trip_milliseconds),
        },
        "fwuff_component_timings": summarize_timings(remote_timings),
        "fwuff_commit_timings": summarize_timings(remote_commit_timings),
        "fwuff_tentative_timings": (
            summarize_timings(remote_tentative_timings)
            if remote_tentative_timings
            else None
        ),
        "raw_samples": raw_samples,
        "proposal_chains": proposal_chains,
        "scope": (
            "real persistent fwuff layer-78 GPU+AMX draft with pinned-host "
            "H2D and EDR TCP; synthetic zero target-hidden rows; no target verify"
        ),
    }


def run_microbenchmark(
    runtime: RemoteDraftRuntime,
    *,
    iterations: int,
    warmup_iterations: int,
    draft_depth: int,
) -> JsonObject:
    if (
        iterations <= 0
        or warmup_iterations < 0
        or not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH
    ):
        raise RemoteDraftError("microbenchmark iteration/depth values are invalid")
    runtime.pinned_payload[:HIDDEN_ROW_BYTES].zero_()
    measured: list[ForwardTiming] = []
    committed: list[ForwardTiming] = []
    tentative: list[ForwardTiming] = []
    cycle_milliseconds: list[float] = []
    raw_samples: list[JsonObject] = []
    proposal_chains: list[list[int]] = []
    for index in range(warmup_iterations + iterations):
        request = DraftRequest(
            request_id="fwuff-isolated-microbenchmark",
            round_id=index,
            action="OPEN" if index == 0 else "ADVANCE",
            token_ids=(index % 1_000,),
            hidden_rows=1,
            payload_bytes=HIDDEN_ROW_BYTES,
            draft_depth=draft_depth,
        )
        cycle_start = time.perf_counter()
        response = runtime.process(request)
        elapsed_milliseconds = (time.perf_counter() - cycle_start) * 1_000
        if index >= warmup_iterations:
            response_timings = list(response.timings)
            measured.extend(response_timings)
            committed.extend(response_timings[: response.committed_forward_count])
            tentative.extend(response_timings[response.committed_forward_count :])
            cycle_milliseconds.append(elapsed_milliseconds)
            proposal_chains.append(list(response.proposal_ids))
            raw_samples.append(
                {
                    "iteration_index": index - warmup_iterations,
                    "round_id": index,
                    "sequence_length": response.sequence_length,
                    "cycle_milliseconds": elapsed_milliseconds,
                    "forward_timings": [
                        timing_sample(timing) for timing in response_timings
                    ],
                }
            )
    sequence_length = cast(SequenceState, runtime.sequence_state).sequence_length
    runtime.finish_sequence()
    return {
        "kind": "glm52_fwuff_isolated_remote_draft_microbenchmark",
        "schema_version": SCHEMA_VERSION,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": iterations,
        "draft_depth": draft_depth,
        "sequence_length": sequence_length,
        "timings": summarize_timings(measured),
        "commit_timings": summarize_timings(committed),
        "tentative_timings": (summarize_timings(tentative) if tentative else None),
        "cycle_milliseconds": {
            "minimum": min(cycle_milliseconds),
            "p50": percentile(cycle_milliseconds, 0.50),
            "p95": percentile(cycle_milliseconds, 0.95),
            "p99": percentile(cycle_milliseconds, 0.99),
            "maximum": max(cycle_milliseconds),
            "mean": sum(cycle_milliseconds) / len(cycle_milliseconds),
        },
        "raw_samples": raw_samples,
        "proposal_chains": proposal_chains,
        "draft_kv_allocation": (
            "single_request_full_pool_reservation"
            if runtime.preallocate_draft_kv
            else "incremental_paged_alloc_decode"
        ),
        "admission": cast(JsonValue, asdict(runtime.admission)),
        "scope": (
            "real persistent fwuff layer-78 GPU+AMX draft and greedy head; "
            "synthetic zero target-hidden rows; no EDR or target verify"
        ),
    }


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--kt-weight-path", default=DEFAULT_KT_WEIGHT_PATH)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument(
        "--maximum-total-tokens",
        type=int,
        default=DEFAULT_MAXIMUM_TOTAL_TOKENS,
    )
    parser.add_argument("--memory-fraction-static", type=float, default=0.80)
    parser.add_argument(
        "--maximum-rows-per-request",
        type=int,
        default=MAXIMUM_HIDDEN_ROWS_PER_REQUEST,
    )
    parser.add_argument(
        "--incremental-draft-kv",
        action="store_true",
        help=(
            "allocate draft KV pages on demand instead of reserving the sole "
            "request's full pool at OPEN"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    microbenchmark = subparsers.add_parser("microbenchmark")
    _add_runtime_arguments(microbenchmark)
    microbenchmark.add_argument("--iterations", type=int, default=32)
    microbenchmark.add_argument("--warmup-iterations", type=int, default=4)
    microbenchmark.add_argument(
        "--draft-depth",
        type=int,
        choices=range(1, MAXIMUM_DRAFT_DEPTH + 1),
        default=1,
    )
    microbenchmark.add_argument("--output", type=Path)

    server = subparsers.add_parser("serve")
    _add_runtime_arguments(server)
    server.add_argument("--bind-address", default=DEFAULT_BIND_ADDRESS)
    server.add_argument("--port", type=int, default=DEFAULT_PORT)

    client = subparsers.add_parser("client")
    client.add_argument("--address", default=DEFAULT_REMOTE_ADDRESS)
    client.add_argument("--port", type=int, default=DEFAULT_PORT)
    client.add_argument("--iterations", type=int, default=64)
    client.add_argument("--warmup-iterations", type=int, default=8)
    client.add_argument("--request-id", default="fwuff-edr-draft-probe")
    client.add_argument(
        "--draft-depth",
        type=int,
        choices=range(1, MAXIMUM_DRAFT_DEPTH + 1),
        default=1,
    )
    client.add_argument("--output", type=Path)
    return parser


def _write_or_print(value: JsonObject, output: Path | None) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "client":
        result = run_client(
            address=parsed.address,
            port=parsed.port,
            iterations=parsed.iterations,
            warmup_iterations=parsed.warmup_iterations,
            request_id=parsed.request_id,
            draft_depth=parsed.draft_depth,
        )
        _write_or_print(result, parsed.output)
        return 0

    runtime = RemoteDraftRuntime(
        model_path=parsed.model_path,
        kt_weight_path=parsed.kt_weight_path,
        context_length=parsed.context_length,
        maximum_total_tokens=parsed.maximum_total_tokens,
        memory_fraction_static=parsed.memory_fraction_static,
        maximum_rows_per_request=parsed.maximum_rows_per_request,
        preallocate_draft_kv=not parsed.incremental_draft_kv,
    )
    if parsed.command == "serve":
        serve(runtime, parsed.bind_address, parsed.port)
        return 0
    result = run_microbenchmark(
        runtime,
        iterations=parsed.iterations,
        warmup_iterations=parsed.warmup_iterations,
        draft_depth=parsed.draft_depth,
    )
    _write_or_print(result, parsed.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
