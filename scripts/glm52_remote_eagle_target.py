#!/usr/bin/env python3
"""Feature-gated TP2 target bridge for the persistent ``fwuff`` MTP service.

The bridge replaces only the local EAGLE draft worker.  Target verification,
sampling, penalties, grammar handling, and target KV ownership remain in the
pinned SGLang worker.  TP rank zero owns the EDR socket; candidate chains are
broadcast through the existing target TP group before both ranks verify them.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from scripts.glm52_fwuff_remote_draft import (
    DEFAULT_PORT,
    DEFAULT_REMOTE_ADDRESS,
    HIDDEN_ROW_BYTES,
    HIDDEN_SIZE,
    MAXIMUM_DRAFT_DEPTH,
    DraftRequest,
    _receive_response,
    send_frame,
)

ENABLE_ENVIRONMENT_VARIABLE: Final = "EXO_GLM52_REMOTE_EAGLE"
ADDRESS_ENVIRONMENT_VARIABLE: Final = "EXO_GLM52_REMOTE_EAGLE_ADDRESS"
PORT_ENVIRONMENT_VARIABLE: Final = "EXO_GLM52_REMOTE_EAGLE_PORT"
DEPTH_ENVIRONMENT_VARIABLE: Final = "EXO_GLM52_REMOTE_EAGLE_DEPTH"
MAXIMUM_ROWS_PER_MESSAGE_ENVIRONMENT_VARIABLE: Final = (
    "EXO_GLM52_REMOTE_EAGLE_MAXIMUM_ROWS_PER_MESSAGE"
)
DEFAULT_MAXIMUM_ROWS_PER_MESSAGE: Final = 512

logger = logging.getLogger(__name__)


class RemoteEagleTargetError(RuntimeError):
    """Fail-closed target-side remote EAGLE error."""


@dataclass(frozen=True, slots=True)
class RemoteEagleConfiguration:
    address: str
    port: int
    draft_depth: int
    maximum_rows_per_message: int

    @classmethod
    def from_environment(cls) -> "RemoteEagleConfiguration":
        if os.environ.get(ENABLE_ENVIRONMENT_VARIABLE) != "1":
            raise RemoteEagleTargetError(
                f"{ENABLE_ENVIRONMENT_VARIABLE} must be exactly 1"
            )
        address = os.environ.get(
            ADDRESS_ENVIRONMENT_VARIABLE,
            DEFAULT_REMOTE_ADDRESS,
        )
        try:
            port = int(os.environ.get(PORT_ENVIRONMENT_VARIABLE, str(DEFAULT_PORT)))
            draft_depth = int(os.environ.get(DEPTH_ENVIRONMENT_VARIABLE, "2"))
            maximum_rows = int(
                os.environ.get(
                    MAXIMUM_ROWS_PER_MESSAGE_ENVIRONMENT_VARIABLE,
                    str(DEFAULT_MAXIMUM_ROWS_PER_MESSAGE),
                )
            )
        except ValueError as error:
            raise RemoteEagleTargetError(
                "remote EAGLE integer environment value is invalid"
            ) from error
        if not address:
            raise RemoteEagleTargetError("remote EAGLE address is empty")
        if not 1 <= port <= 65_535:
            raise RemoteEagleTargetError("remote EAGLE port is invalid")
        if not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH:
            raise RemoteEagleTargetError(
                f"remote EAGLE depth must be in [1, {MAXIMUM_DRAFT_DEPTH}]"
            )
        if not 1 <= maximum_rows <= DEFAULT_MAXIMUM_ROWS_PER_MESSAGE:
            raise RemoteEagleTargetError(
                "remote EAGLE maximum rows per message is invalid"
            )
        return cls(
            address=address,
            port=port,
            draft_depth=draft_depth,
            maximum_rows_per_message=maximum_rows,
        )


def topk1_linear_tree_indices(
    draft_depth: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the exact pinned EAGLE top-k-1 static-chain tree inputs."""

    if not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH:
        raise RemoteEagleTargetError("draft depth is outside the admitted range")
    parent_indices = () if draft_depth == 1 else tuple(range(-1, draft_depth - 1))
    return parent_indices, tuple(range(draft_depth))


def shifted_prefill_token_ids(
    input_ids: Sequence[int],
    next_token_id: int,
) -> tuple[int, ...]:
    """Mirror EAGLE's prefill draft input ``[prompt[1:], sampled]``."""

    if not input_ids:
        raise RemoteEagleTargetError("prefill input cannot be empty")
    if next_token_id < 0 or any(token_id < 0 for token_id in input_ids):
        raise RemoteEagleTargetError("token ids must be nonnegative")
    return (*input_ids[1:], next_token_id)


def _uint8_payload_memoryview(
    byte_storage: Any,
    payload_bytes: int,
) -> memoryview:
    """Expose only byte-backed CPU storage to the socket framing code."""

    if payload_bytes < 0:
        raise RemoteEagleTargetError("payload byte count cannot be negative")
    payload = memoryview(byte_storage[:payload_bytes].numpy())
    if payload.format != "B" or payload.itemsize != 1:
        raise RemoteEagleTargetError(
            "remote EAGLE payload storage must expose unsigned bytes"
        )
    if len(payload) != payload_bytes:
        raise RemoteEagleTargetError("remote EAGLE payload byte count changed")
    return payload


def _validate_response(
    response: Mapping[str, object],
    *,
    request_id: str,
    round_id: int,
    draft_depth: int,
) -> tuple[int, ...]:
    if response.get("status") != "ok":
        raise RemoteEagleTargetError(
            f"fwuff rejected round {round_id}: {response.get('error')}"
        )
    if response.get("request_id") != request_id or response.get("round_id") != round_id:
        raise RemoteEagleTargetError("fwuff response identity does not match request")
    raw_proposals = response.get("proposal_ids")
    if (
        not isinstance(raw_proposals, list)
        or len(raw_proposals) != draft_depth
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in raw_proposals
        )
    ):
        raise RemoteEagleTargetError("fwuff returned an invalid candidate chain")
    return tuple(cast(list[int], raw_proposals))


class RemoteDraftConnection:
    """One request-owned persistent TCP connection to ``fwuff``."""

    def __init__(self, configuration: RemoteEagleConfiguration) -> None:
        self.configuration = configuration
        self.connection = socket.create_connection(
            (configuration.address, configuration.port),
            timeout=30.0,
        )
        self.connection.settimeout(300.0)
        self.connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        self.request_id: str | None = None
        self.next_round_id = 0

    def close(self) -> None:
        self.connection.close()

    def _exchange(
        self,
        *,
        request_id: str,
        action: str,
        token_ids: Sequence[int],
        payload: memoryview,
    ) -> tuple[int, ...]:
        request = DraftRequest(
            request_id=request_id,
            round_id=self.next_round_id,
            action=cast(Any, action),
            token_ids=tuple(token_ids),
            hidden_rows=len(token_ids),
            payload_bytes=len(payload),
            draft_depth=self.configuration.draft_depth,
        )
        send_frame(self.connection, request.to_header(), payload)
        response = _receive_response(self.connection)
        proposals = _validate_response(
            response,
            request_id=request_id,
            round_id=self.next_round_id,
            draft_depth=self.configuration.draft_depth,
        )
        self.next_round_id += 1
        return proposals

    def open_or_advance(
        self,
        *,
        request_id: str,
        token_ids: Sequence[int],
        payload: memoryview,
    ) -> tuple[int, ...]:
        if not token_ids or len(payload) != len(token_ids) * HIDDEN_ROW_BYTES:
            raise RemoteEagleTargetError(
                "each accepted token requires exactly one BF16 hidden row"
            )
        if len(token_ids) > self.configuration.maximum_rows_per_message:
            raise RemoteEagleTargetError("remote EAGLE message has too many rows")
        if self.request_id is None:
            self.request_id = request_id
            action = "OPEN"
        elif self.request_id == request_id:
            action = "ADVANCE"
        else:
            # The c1 scheduler is serialized, but SGLang health probes can
            # complete without a final empty decode callback.  Fail closed on
            # that stale request before admitting the next serialized request.
            self.finish(abort=True)
            self.request_id = request_id
            action = "OPEN"
        return self._exchange(
            request_id=request_id,
            action=action,
            token_ids=token_ids,
            payload=payload,
        )

    def finish(self, *, abort: bool = False) -> None:
        if self.request_id is None:
            return
        request = DraftRequest(
            request_id=self.request_id,
            round_id=self.next_round_id,
            action="ABORT" if abort else "FINISH",
            token_ids=(),
            hidden_rows=0,
            payload_bytes=0,
            draft_depth=self.configuration.draft_depth,
        )
        send_frame(self.connection, request.to_header())
        response = _receive_response(self.connection)
        _validate_response(
            response,
            request_id=self.request_id,
            round_id=self.next_round_id,
            draft_depth=0,
        )
        self.request_id = None
        self.next_round_id = 0


def build_topk1_verify_input(
    *,
    verified_id: Any,
    proposal_ids: Any,
    sequence_lengths: Any,
    sequence_lengths_cpu: Any,
) -> Any:
    """Reconstruct native ``EagleVerifyInput`` for a linear remote chain."""

    import torch
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sglang.srt.speculative.eagle_info import EagleVerifyInput
    from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient

    if verified_id.ndim != 1 or verified_id.numel() != 1:
        raise RemoteEagleTargetError("the first bridge admits exactly one request")
    if proposal_ids.ndim != 2 or proposal_ids.shape[0] != 1:
        raise RemoteEagleTargetError("candidate tensor must have shape [1, K]")
    draft_depth = int(proposal_ids.shape[1])
    parent_indices, score_indices = topk1_linear_tree_indices(draft_depth)
    parent_list = torch.tensor(
        [parent_indices],
        dtype=torch.int64,
        device=proposal_ids.device,
    )
    top_scores_index = torch.tensor(
        [score_indices],
        dtype=torch.int64,
        device=proposal_ids.device,
    )
    (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        verified_id,
        parent_list,
        top_scores_index,
        proposal_ids,
        sequence_lengths,
        int(sequence_lengths.sum().item()),
        1,
        draft_depth,
        draft_depth + 1,
    )
    return EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=tree_mask,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        retrive_cum_len=None,
        spec_steps=draft_depth,
        topk=1,
        draft_token_num=draft_depth + 1,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=int(sequence_lengths.sum().item()),
        seq_lens_cpu=sequence_lengths_cpu,
    )


def _validate_target_server_args(server_args: Any) -> int:
    mismatches: list[str] = []
    draft_depth = int(server_args.speculative_num_steps)
    expected = {
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 1,
        "speculative_algorithm": "EAGLE",
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": draft_depth + 1,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
        "max_running_requests": 1,
    }
    for name, expected_value in expected.items():
        actual = getattr(server_args, name, None)
        if actual != expected_value:
            mismatches.append(f"{name}={actual!r} expected {expected_value!r}")
    if not 1 <= draft_depth <= MAXIMUM_DRAFT_DEPTH:
        mismatches.append(f"speculative_num_steps={draft_depth!r}")
    if mismatches:
        raise RemoteEagleTargetError(
            "remote EAGLE target admission rejected: " + "; ".join(mismatches)
        )
    return draft_depth


def create_remote_eagle_worker_class() -> type[Any]:
    """Create the worker lazily so importing this module stays CUDA-free."""

    import torch
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import (
        EagleDraftInput,
        EagleVerifyInput,
    )
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

    class RemoteEAGLEWorker(EAGLEWorker):
        def __init__(
            self,
            server_args: Any,
            gpu_id: int,
            tp_rank: int,
            dp_rank: int | None,
            moe_ep_rank: int,
            attn_cp_rank: int,
            moe_dp_rank: int,
            nccl_port: int,
            target_worker: Any,
        ) -> None:
            del dp_rank, moe_ep_rank, attn_cp_rank, moe_dp_rank, nccl_port
            self.server_args = server_args
            self.topk = 1
            self.speculative_num_steps = _validate_target_server_args(server_args)
            self.speculative_num_draft_tokens = self.speculative_num_steps + 1
            self.enable_nan_detection = server_args.enable_nan_detection
            self.gpu_id = gpu_id
            self.tp_rank = tp_rank
            self.device = server_args.device
            self.target_worker = target_worker
            self.page_size = server_args.page_size
            self._model_runner = target_worker.model_runner
            self.model_config = target_worker.model_config
            (
                self.req_to_token_pool,
                self.token_to_kv_pool_allocator,
            ) = target_worker.get_memory_pool()
            # Keep the native EAGLE decode allocator contract even though the
            # draft forward itself runs on fwuff.  The inherited top-k-1
            # preprocess uses these as dummy kernel arguments while it shadows
            # paged cache locations and restores the allocator state.
            self.num_new_pages_per_topk = torch.empty(
                (),
                dtype=torch.int64,
                device=self.device,
            )
            self.extend_lens = torch.empty(
                (),
                dtype=torch.int64,
                device=self.device,
            )
            self._configuration = RemoteEagleConfiguration.from_environment()
            if self._configuration.draft_depth != self.speculative_num_steps:
                raise RemoteEagleTargetError(
                    "wire depth differs from target speculative_num_steps"
                )
            self._connection = (
                RemoteDraftConnection(self._configuration)
                if get_tp_group().rank_in_group == 0
                else None
            )
            self._pinned_payload = (
                torch.empty(
                    self._configuration.maximum_rows_per_message * HIDDEN_ROW_BYTES,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                if self._connection is not None
                else None
            )
            self._pinned_hidden = (
                self._pinned_payload.view(torch.bfloat16).reshape(
                    self._configuration.maximum_rows_per_message,
                    HIDDEN_SIZE,
                )
                if self._pinned_payload is not None
                else None
            )
            self._prefill_coherency_validated = False
            self._active_request_id: str | None = None
            self._proposal_ids: Any | None = None

        def _hidden_layout_error(
            self,
            hidden_states: Any,
            expected_rows: int,
        ) -> str | None:
            if not isinstance(hidden_states, torch.Tensor):
                return "target hidden states are not a tensor"
            if hidden_states.dtype != torch.bfloat16:
                return (
                    "target hidden dtype is "
                    f"{hidden_states.dtype}, expected torch.bfloat16"
                )
            if hidden_states.ndim != 2:
                return f"target hidden rank is {hidden_states.ndim}, expected rank two"
            if int(hidden_states.shape[1]) != HIDDEN_SIZE:
                return (
                    "target hidden width is "
                    f"{hidden_states.shape[1]}, expected {HIDDEN_SIZE}"
                )
            if int(hidden_states.shape[0]) != expected_rows:
                return (
                    "target hidden row count is "
                    f"{hidden_states.shape[0]}, expected {expected_rows}"
                )
            if not bool(torch.isfinite(hidden_states).all().item()):
                return "target hidden states contain non-finite values"
            return None

        def _validate_prefill_coherency(
            self,
            hidden_states: Any,
            expected_rows: int,
        ) -> None:
            """Prove the FULL target hidden output is coherent across TP ranks."""

            if self._prefill_coherency_validated:
                return
            tp_group = get_tp_group()
            local_error = self._hidden_layout_error(
                hidden_states,
                expected_rows,
            )
            layout_failure = torch.tensor(
                [1 if local_error is not None else 0],
                dtype=torch.int32,
                device=self.device,
            )
            layout_failure = tp_group.all_reduce(layout_failure)
            if int(layout_failure.item()) != 0:
                detail = (
                    local_error
                    if local_error is not None
                    else "the peer TP rank rejected its hidden layout"
                )
                raise RemoteEagleTargetError(
                    "remote EAGLE prefill layout gate failed: " + detail
                )

            rank_zero_hidden = (
                hidden_states.detach().clone()
                if tp_group.rank_in_group == 0
                else torch.empty_like(hidden_states)
            )
            tp_group.broadcast(rank_zero_hidden, src=0)
            exact_match = torch.equal(hidden_states, rank_zero_hidden)
            mismatch = torch.tensor(
                [0 if exact_match else 1],
                dtype=torch.int32,
                device=self.device,
            )
            mismatch = tp_group.all_reduce(mismatch)
            if int(mismatch.item()) != 0:
                raise RemoteEagleTargetError(
                    "remote EAGLE prefill FULL hidden rows differ across TP ranks"
                )
            self._prefill_coherency_validated = True
            if tp_group.rank_in_group == 0:
                logger.info(
                    "EXO_REMOTE_EAGLE_PREFILL_COHERENCY_OK "
                    "rows=%d width=%d dtype=bfloat16 finite=true exact=true",
                    expected_rows,
                    HIDDEN_SIZE,
                )

        def _broadcast_remote_result(
            self,
            coordinator_proposals: tuple[int, ...] | None,
            coordinator_error: Exception | None,
        ) -> Any:
            tp_group = get_tp_group()
            status = torch.tensor(
                [1 if coordinator_error is not None else 0],
                dtype=torch.int32,
                device=self.device,
            )
            if tp_group.rank_in_group != 0:
                status.zero_()
            tp_group.broadcast(status, src=0)
            if tp_group.rank_in_group == 0 and coordinator_proposals is not None:
                proposals = torch.tensor(
                    [coordinator_proposals],
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                proposals = torch.empty(
                    (1, self.speculative_num_steps),
                    dtype=torch.int64,
                    device=self.device,
                )
            tp_group.broadcast(proposals, src=0)
            if int(status.item()) != 0:
                if coordinator_error is not None:
                    raise RemoteEagleTargetError(
                        f"fwuff coordinator failed: {coordinator_error}"
                    ) from coordinator_error
                raise RemoteEagleTargetError("fwuff coordinator failed on TP rank zero")
            return proposals

        def _exchange_hidden_rows(
            self,
            request_id: str,
            token_ids: Any,
            hidden_states: Any,
        ) -> Any:
            coordinator_proposals: tuple[int, ...] | None = None
            coordinator_error: Exception | None = None
            if self._connection is not None:
                assert self._pinned_hidden is not None
                assert self._pinned_payload is not None
                try:
                    token_ids_cpu = token_ids.to(
                        device="cpu",
                        dtype=torch.int64,
                    ).tolist()
                    layout_error = self._hidden_layout_error(
                        hidden_states,
                        len(token_ids_cpu),
                    )
                    if layout_error is not None:
                        raise RemoteEagleTargetError(layout_error)
                    for start in range(
                        0,
                        len(token_ids_cpu),
                        self._configuration.maximum_rows_per_message,
                    ):
                        stop = min(
                            start + self._configuration.maximum_rows_per_message,
                            len(token_ids_cpu),
                        )
                        rows = stop - start
                        self._pinned_hidden[:rows].copy_(
                            hidden_states[start:stop],
                            non_blocking=True,
                        )
                        torch.cuda.current_stream().synchronize()
                        payload = _uint8_payload_memoryview(
                            self._pinned_payload,
                            rows * HIDDEN_ROW_BYTES,
                        )
                        coordinator_proposals = self._connection.open_or_advance(
                            request_id=request_id,
                            token_ids=token_ids_cpu[start:stop],
                            payload=payload,
                        )
                except Exception as error:
                    coordinator_error = error
            proposals = self._broadcast_remote_result(
                coordinator_proposals,
                coordinator_error,
            )
            self._active_request_id = request_id
            self._proposal_ids = proposals
            return proposals

        def _remote_prefill(
            self,
            batch: Any,
            hidden_states: Any,
            next_token_ids: Any,
        ) -> None:
            if batch.batch_size() != 1:
                raise RemoteEagleTargetError(
                    "the first remote bridge admits concurrency one only"
                )
            request_id = str(batch.reqs[0].rid)
            input_ids = batch.input_ids
            shifted_ids = torch.cat((input_ids[1:], next_token_ids[:1]))
            if shifted_ids.shape[0] != hidden_states.shape[0]:
                raise RemoteEagleTargetError(
                    "prefill shifted IDs and target hidden rows differ"
                )
            self._validate_prefill_coherency(
                hidden_states,
                int(shifted_ids.shape[0]),
            )
            proposals = self._exchange_hidden_rows(
                request_id,
                shifted_ids,
                hidden_states,
            )
            batch.spec_info = EagleDraftInput(
                verified_id=next_token_ids[:1],
                hidden_states=hidden_states[-1:],
            )
            batch.spec_info.remote_proposal_ids = proposals

        def draft(self, batch: Any) -> Any:
            if batch.forward_mode.is_idle():
                return EagleVerifyInput.create_idle_input(
                    self.topk,
                    self.speculative_num_steps,
                    self.speculative_num_draft_tokens,
                )
            if batch.batch_size() != 1 or self._proposal_ids is None:
                raise RemoteEagleTargetError(
                    "remote candidate state is unavailable for decode"
                )
            # Preserve native EAGLE's paged-cache shadow transaction before
            # target verification.  With top-k one this performs no draft
            # model compute: it derives the last partial-page location,
            # temporarily allocates the K draft slots, records their request
            # mappings, and restores the shared allocator.  Omitting this
            # transaction corrupts the large-page lifecycle when decode
            # crosses a page boundary.
            super()._draft_preprocess_decode(batch)
            draft_input = batch.spec_info
            anchor = draft_input.verified_id[-1:].to(torch.int64)
            return build_topk1_verify_input(
                verified_id=anchor,
                proposal_ids=self._proposal_ids,
                sequence_lengths=batch.seq_lens,
                sequence_lengths_cpu=batch.seq_lens_cpu,
            )

        def _advance_after_verify(self, batch: Any) -> None:
            draft_input = batch.spec_info
            if draft_input.verified_id.numel() == 0:
                self._finish_remote(abort=False)
                return
            if self._active_request_id is None:
                raise RemoteEagleTargetError("remote request identity is absent")
            proposals = self._exchange_hidden_rows(
                self._active_request_id,
                draft_input.verified_id,
                draft_input.hidden_states,
            )
            draft_input.remote_proposal_ids = proposals

        def _finish_remote(self, *, abort: bool) -> None:
            coordinator_error: Exception | None = None
            if self._connection is not None:
                try:
                    self._connection.finish(abort=abort)
                except Exception as error:
                    coordinator_error = error
            self._broadcast_remote_result(
                tuple(0 for _ in range(self.speculative_num_steps)),
                coordinator_error,
            )
            self._active_request_id = None
            self._proposal_ids = None

        def forward_batch_generation(self, batch: Any) -> Any:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                logits_output, next_token_ids, _ = self.forward_target_extend(batch)
                self._remote_prefill(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                )
                return GenerationBatchResult(
                    logits_output=logits_output,
                    next_token_ids=next_token_ids,
                    num_accepted_tokens=0,
                    can_run_cuda_graph=False,
                )

            spec_info = self.draft(batch)
            logits_output, verify_output, _, can_run_cuda_graph = self.verify(
                batch,
                spec_info,
            )
            self._advance_after_verify(batch)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=(verify_output.accept_length_per_req_cpu),
                can_run_cuda_graph=can_run_cuda_graph,
            )

        def clear_cache_pool(self) -> None:
            if self._active_request_id is not None:
                self._finish_remote(abort=True)

    return RemoteEAGLEWorker


def install_remote_eagle_worker() -> None:
    """Install the worker selector only under the explicit feature gate."""

    if os.environ.get(ENABLE_ENVIRONMENT_VARIABLE) != "1":
        return
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if getattr(SpeculativeAlgorithm, "_exo_remote_eagle_installed", False):
        return
    original_create_worker = SpeculativeAlgorithm.create_worker
    remote_worker_class = create_remote_eagle_worker_class()

    def create_worker(algorithm: Any, server_args: Any) -> Any:
        if algorithm.is_eagle():
            if not server_args.disable_overlap_schedule:
                raise RemoteEagleTargetError(
                    "the first remote bridge requires the non-overlap EAGLE worker"
                )
            return remote_worker_class
        return original_create_worker(algorithm, server_args)

    SpeculativeAlgorithm.create_worker = create_worker
    SpeculativeAlgorithm._exo_remote_eagle_installed = True
