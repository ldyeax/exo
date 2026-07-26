from __future__ import annotations

import json
import socket
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest

from scripts import glm52_fwuff_remote_draft as draft


class BufferConnection:
    def __init__(self, payload: bytes):
        self.payload = bytearray(payload)
        self.offset = 0

    def recv_into(self, output: memoryview) -> int:
        remaining = len(self.payload) - self.offset
        size = min(len(output), remaining)
        if size == 0:
            return 0
        output[:size] = self.payload[self.offset : self.offset + size]
        self.offset += size
        return size


class ScriptedConnection(BufferConnection):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.sent = bytearray()

    def setsockopt(self, level: int, option: int, value: int) -> None:
        del level, option, value

    def sendall(self, payload: bytes) -> None:
        self.sent.extend(payload)


class FakeRuntime:
    def __init__(self) -> None:
        self.payload = bytearray(draft.HIDDEN_ROW_BYTES)
        self.sequence_state: draft.SequenceState | None = None
        self.finish_count = 0

    def payload_view(self, payload_bytes: int) -> memoryview:
        return memoryview(self.payload)[:payload_bytes]

    def process(self, value: draft.DraftRequest) -> draft.DraftResponse:
        self.sequence_state = draft.SequenceState(
            request_id=value.request_id,
            last_round_id=value.round_id,
            draft_depth=value.draft_depth,
            sequence_length=value.hidden_rows,
        )
        return draft.DraftResponse(
            request_id=value.request_id,
            round_id=value.round_id,
            status="ok",
            proposal_ids=(7,),
            sequence_length=value.hidden_rows,
            committed_forward_count=value.hidden_rows,
        )

    def finish_sequence(self) -> None:
        self.finish_count += 1
        self.sequence_state = None


class FakeHiddenPayload:
    def __init__(self) -> None:
        self.rows = 0

    def __getitem__(self, key: object) -> "FakeHiddenPayload":
        del key
        return self

    def view(self, *shape: object) -> "FakeHiddenPayload | tuple[str, ...]":
        if len(shape) == 1:
            return self
        self.rows = int(cast(int, shape[0]))
        return tuple(f"hidden-{index}" for index in range(self.rows))


def request(
    *,
    action: draft.Action = "OPEN",
    round_id: int = 0,
    rows: int = 1,
    draft_depth: int = 1,
) -> draft.DraftRequest:
    return draft.DraftRequest(
        request_id="request-1",
        round_id=round_id,
        action=action,
        token_ids=tuple(range(rows)),
        hidden_rows=rows,
        payload_bytes=rows * draft.HIDDEN_ROW_BYTES,
        draft_depth=draft_depth,
    )


def test_request_round_trip_and_exact_hidden_payload_contract():
    value = request(rows=2)
    assert draft.DraftRequest.from_header(value.to_header()) == value
    invalid = value.to_header()
    invalid["payload_bytes"] = 1
    with pytest.raises(draft.RemoteDraftError, match="expected"):
        draft.DraftRequest.from_header(invalid)


@pytest.mark.parametrize("action", ["FINISH", "ABORT"])
def test_terminal_actions_forbid_model_payload(action: draft.Action):
    invalid = request(action=action).to_header()
    with pytest.raises(draft.RemoteDraftError, match="must not carry"):
        draft.DraftRequest.from_header(invalid)


def test_open_requires_round_zero_and_advance_requires_rows():
    invalid_open = request(round_id=1).to_header()
    with pytest.raises(draft.RemoteDraftError, match="round_id zero"):
        draft.DraftRequest.from_header(invalid_open)

    invalid_advance = request(action="ADVANCE", round_id=1, rows=0).to_header()
    with pytest.raises(draft.RemoteDraftError, match="require one accepted"):
        draft.DraftRequest.from_header(invalid_advance)


def test_sequence_state_rejects_stale_and_foreign_rounds():
    state = draft.SequenceState(
        request_id="request-1",
        last_round_id=0,
        draft_depth=1,
    )
    state.admit(request(action="ADVANCE", round_id=1))
    assert state.last_round_id == 1
    with pytest.raises(draft.RemoteDraftError, match="does not follow"):
        state.admit(request(action="ADVANCE", round_id=1))
    foreign = request(action="ADVANCE", round_id=2).to_header()
    foreign["request_id"] = "request-2"
    with pytest.raises(draft.RemoteDraftError, match="another request"):
        state.admit(draft.DraftRequest.from_header(foreign))


def test_request_depth_is_bounded_and_stable_for_open_sequence():
    assert (
        draft.DraftRequest.from_header(
            request(draft_depth=draft.MAXIMUM_DRAFT_DEPTH).to_header()
        ).draft_depth
        == draft.MAXIMUM_DRAFT_DEPTH
    )
    invalid = request().to_header()
    invalid["draft_depth"] = draft.MAXIMUM_DRAFT_DEPTH + 1
    with pytest.raises(draft.RemoteDraftError, match="draft_depth"):
        draft.DraftRequest.from_header(invalid)

    state = draft.SequenceState(
        request_id="request-1",
        last_round_id=0,
        draft_depth=2,
    )
    with pytest.raises(draft.RemoteDraftError, match="cannot change"):
        state.admit(request(action="ADVANCE", round_id=1, draft_depth=3))


def test_frame_transport_preserves_json_and_payload():
    value = request()
    payload = bytes(draft.HIDDEN_ROW_BYTES)
    connection = BufferConnection(draft.encode_frame(value.to_header(), payload))
    header, payload_bytes = draft.receive_frame_header(connection)
    assert draft.DraftRequest.from_header(header) == value
    assert payload_bytes == len(payload)
    assert draft._recv_exact(connection, payload_bytes) == payload


def test_frame_rejects_wrong_magic():
    header = json.dumps(request().to_header()).encode()
    connection = BufferConnection(
        draft.FRAME_PREFIX.pack(b"WRONG!!!", len(header), 0) + header
    )
    with pytest.raises(draft.RemoteDraftError, match="magic"):
        draft.receive_frame_header(connection)


def test_timing_summary_uses_nearest_rank_percentiles():
    values = tuple(
        draft.ForwardTiming(
            sequence_length=index,
            h2d_milliseconds=float(index),
            allocator_wall_milliseconds=float(index + 2),
            allocator_cuda_milliseconds=float(index + 3),
            metadata_wall_milliseconds=float(index + 4),
            metadata_cuda_milliseconds=float(index + 5),
            model_milliseconds=float(index + 10),
            proposal_milliseconds=float(index + 20),
            total_milliseconds=float(index + 30),
        )
        for index in range(1, 5)
    )
    summary = draft.summarize_timings(values)
    assert summary["iterations"] == 4
    assert summary["rows"] == 4
    assert summary["h2d_milliseconds"] == {
        "minimum": 1.0,
        "p50": 2.0,
        "p95": 4.0,
        "p99": 4.0,
        "maximum": 4.0,
        "mean": 2.5,
    }


def test_large_response_compacts_timings_but_small_response_stays_raw():
    values = tuple(
        draft.ForwardTiming(
            sequence_length=index + 1,
            h2d_milliseconds=0.01,
            allocator_wall_milliseconds=0.02,
            allocator_cuda_milliseconds=0.01,
            metadata_wall_milliseconds=0.20,
            metadata_cuda_milliseconds=0.18,
            model_milliseconds=3.75,
            proposal_milliseconds=0.06,
            total_milliseconds=4.40,
        )
        for index in range(draft.MAXIMUM_RAW_TIMING_SAMPLES + 1)
    )
    small_header = draft.DraftResponse(
        request_id="small",
        round_id=0,
        status="ok",
        timings=values[:4],
        committed_forward_count=1,
        tentative_forward_count=3,
    ).to_header()
    assert len(cast(list[object], small_header["timings"])) == 4
    assert "timing_encoding" not in small_header

    large_header = draft.DraftResponse(
        request_id="large",
        round_id=0,
        status="ok",
        timings=values,
        committed_forward_count=len(values) - 3,
        tentative_forward_count=3,
    ).to_header()
    assert large_header["timings"] == []
    assert large_header["timing_encoding"] == "summary-v1"
    timing_summary = cast(dict[str, Any], large_header["timing_summary"])
    assert cast(dict[str, object], timing_summary["all"])["iterations"] == len(values)
    assert cast(dict[str, object], timing_summary["all"])["rows"] == len(values)
    assert cast(dict[str, object], timing_summary["committed"])["iterations"] == (
        len(values) - 3
    )
    assert cast(dict[str, object], timing_summary["tentative"])["iterations"] == 3
    assert len(draft.encode_frame(large_header)) < draft.MAXIMUM_HEADER_BYTES


def test_timing_parser_preserves_schema_two_legacy_samples_and_batch_rows():
    legacy = {
        key: value
        for key, value in cast(
            dict[str, object],
            draft.DraftResponse(
                request_id="legacy",
                round_id=0,
                status="ok",
                timings=(
                    draft.ForwardTiming(
                        sequence_length=1,
                        h2d_milliseconds=0.01,
                        allocator_wall_milliseconds=0.02,
                        allocator_cuda_milliseconds=0.01,
                        metadata_wall_milliseconds=0.20,
                        metadata_cuda_milliseconds=0.18,
                        model_milliseconds=3.75,
                        proposal_milliseconds=0.06,
                        total_milliseconds=4.40,
                    ),
                ),
                committed_forward_count=1,
            ).to_header()["timings"][0],
        ).items()
        if key != "row_count"
    }
    assert draft._parse_forward_timing(legacy).row_count == 1

    batched = dict(legacy, sequence_length=65, row_count=65)
    parsed = draft._parse_forward_timing(batched)
    assert parsed.sequence_length == 65
    assert parsed.row_count == 65
    assert draft.summarize_timings((parsed,))["rows"] == 65


@pytest.mark.parametrize("row_count", [2, 5, 64, 65])
def test_committed_extend_layout_uses_exact_full_pool_slice(row_count: int):
    runtime = object.__new__(draft.RemoteDraftRuntime)
    runtime.sequence_state = draft.SequenceState(
        request_id="request-1",
        last_round_id=0,
        draft_depth=4,
        sequence_length=0,
    )
    runtime.request_pool_index = 0
    runtime.maximum_rows_per_request = 128
    runtime.server_args = SimpleNamespace(context_length=256)
    reserved = list(range(256))
    runtime.reserved_cache_locations = reserved

    layout, selected = runtime._prepare_committed_extend(row_count)

    assert layout == draft.CommittedExtendLayout(
        prefix_length=0,
        row_count=row_count,
        sequence_length=row_count,
    )
    assert selected == reserved[:row_count]
    assert runtime.reserved_cache_locations is reserved
    assert reserved == list(range(256))


@pytest.mark.parametrize("row_count", [2, 5])
def test_committed_extend_layout_crosses_page_64_without_remapping(
    row_count: int,
):
    runtime = object.__new__(draft.RemoteDraftRuntime)
    runtime.sequence_state = draft.SequenceState(
        request_id="request-1",
        last_round_id=1,
        draft_depth=4,
        sequence_length=63,
    )
    runtime.request_pool_index = 0
    runtime.maximum_rows_per_request = 128
    runtime.server_args = SimpleNamespace(context_length=256)
    reserved = list(range(1_000, 1_256))
    runtime.reserved_cache_locations = reserved

    layout, selected = runtime._prepare_committed_extend(row_count)

    assert layout.prefix_length == 63
    assert layout.sequence_length == 63 + row_count
    assert selected == reserved[63 : 63 + row_count]
    assert runtime.reserved_cache_locations is reserved


@pytest.mark.parametrize("row_count", [64, 65])
def test_process_batches_bulk_rows_and_decodes_final_boundary(row_count: int):
    runtime = object.__new__(draft.RemoteDraftRuntime)
    runtime.maximum_rows_per_request = 128
    runtime.pinned_payload = FakeHiddenPayload()
    runtime.torch = SimpleNamespace(bfloat16="bfloat16")
    runtime.sequence_state = None
    runtime.request_pool_index = None
    runtime.reserved_cache_locations = list(range(256))
    runtime.sequence_lengths = [0]
    runtime.sequence_lengths_cpu = [0]
    captured_bulk: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    captured_boundary: list[tuple[int, str]] = []

    def reset_sequence(
        self: draft.RemoteDraftRuntime,
        request_id: str,
        draft_depth: int,
    ) -> None:
        self.sequence_state = draft.SequenceState(
            request_id=request_id,
            last_round_id=0,
            draft_depth=draft_depth,
        )
        self.request_pool_index = 0

    def forward_many(
        self: draft.RemoteDraftRuntime,
        token_ids: tuple[int, ...],
        hidden_rows: tuple[str, ...],
    ) -> draft.ForwardResult:
        captured_bulk.append((tuple(token_ids), tuple(hidden_rows)))
        state = cast(draft.SequenceState, self.sequence_state)
        state.sequence_length += len(token_ids)
        return draft.ForwardResult(
            proposal_id=token_ids[-1] + 100,
            hidden_state=f"result-{hidden_rows[-1]}",
            timing=draft.ForwardTiming(
                sequence_length=state.sequence_length,
                h2d_milliseconds=0.1,
                allocator_wall_milliseconds=0.0,
                allocator_cuda_milliseconds=0.0,
                metadata_wall_milliseconds=0.1,
                metadata_cuda_milliseconds=0.1,
                model_milliseconds=1.0,
                proposal_milliseconds=0.1,
                total_milliseconds=1.3,
                row_count=len(token_ids),
            ),
        )

    def forward_one(
        self: draft.RemoteDraftRuntime,
        token_id: int,
        hidden_row: str,
    ) -> draft.ForwardResult:
        captured_boundary.append((token_id, hidden_row))
        state = cast(draft.SequenceState, self.sequence_state)
        state.sequence_length += 1
        return draft.ForwardResult(
            proposal_id=token_id + 100,
            hidden_state=f"result-{hidden_row}",
            timing=draft.ForwardTiming(
                sequence_length=state.sequence_length,
                h2d_milliseconds=0.1,
                allocator_wall_milliseconds=0.0,
                allocator_cuda_milliseconds=0.0,
                metadata_wall_milliseconds=0.1,
                metadata_cuda_milliseconds=0.1,
                model_milliseconds=1.0,
                proposal_milliseconds=0.1,
                total_milliseconds=1.3,
            ),
        )

    def propose_chain(
        self: draft.RemoteDraftRuntime,
        first_result: draft.ForwardResult,
        draft_depth: int,
    ) -> tuple[tuple[int, ...], tuple[draft.ForwardTiming, ...]]:
        del self
        assert draft_depth == 1
        assert first_result.hidden_state == f"result-hidden-{row_count - 1}"
        return (first_result.proposal_id,), ()

    runtime.reset_sequence = MethodType(reset_sequence, runtime)
    runtime._forward_many = MethodType(forward_many, runtime)
    runtime._forward_one = MethodType(forward_one, runtime)
    runtime._propose_chain = MethodType(propose_chain, runtime)

    token_ids = tuple(range(10, 10 + row_count))
    response = runtime.process(
        draft.DraftRequest(
            request_id="request-1",
            round_id=0,
            action="OPEN",
            token_ids=token_ids,
            hidden_rows=row_count,
            payload_bytes=row_count * draft.HIDDEN_ROW_BYTES,
            draft_depth=1,
        )
    )

    assert captured_bulk == [
        (
            token_ids[:-1],
            tuple(f"hidden-{index}" for index in range(row_count - 1)),
        )
    ]
    assert captured_boundary == [(token_ids[-1], f"hidden-{row_count - 1}")]
    assert response.proposal_ids == (token_ids[-1] + 100,)
    assert response.sequence_length == row_count
    assert response.committed_forward_count == 2
    assert response.tentative_forward_count == 0
    assert len(response.timings) == 2
    assert [timing.row_count for timing in response.timings] == [
        row_count - 1,
        1,
    ]
    assert sum(timing.row_count for timing in response.timings) == row_count


@pytest.mark.parametrize("row_count", [2, 5])
def test_process_keeps_small_committed_bursts_on_exact_serial_path(
    row_count: int,
):
    runtime = object.__new__(draft.RemoteDraftRuntime)
    runtime.maximum_rows_per_request = 128
    runtime.pinned_payload = FakeHiddenPayload()
    runtime.torch = SimpleNamespace(bfloat16="bfloat16")
    runtime.sequence_state = None
    runtime.request_pool_index = None
    runtime.reserved_cache_locations = list(range(256))
    runtime.sequence_lengths = [0]
    runtime.sequence_lengths_cpu = [0]
    captured: list[tuple[int, str]] = []

    def reset_sequence(
        self: draft.RemoteDraftRuntime,
        request_id: str,
        draft_depth: int,
    ) -> None:
        self.sequence_state = draft.SequenceState(
            request_id=request_id,
            last_round_id=0,
            draft_depth=draft_depth,
        )
        self.request_pool_index = 0

    def forbid_batch(*args: object, **kwargs: object) -> draft.ForwardResult:
        del args, kwargs
        raise AssertionError("small committed input must retain serial DECODE")

    def forward_one(
        self: draft.RemoteDraftRuntime,
        token_id: int,
        hidden_row: str,
    ) -> draft.ForwardResult:
        captured.append((token_id, hidden_row))
        state = cast(draft.SequenceState, self.sequence_state)
        state.sequence_length += 1
        return draft.ForwardResult(
            proposal_id=token_id + 100,
            hidden_state=f"result-{hidden_row}",
            timing=draft.ForwardTiming(
                sequence_length=state.sequence_length,
                h2d_milliseconds=0.1,
                allocator_wall_milliseconds=0.0,
                allocator_cuda_milliseconds=0.0,
                metadata_wall_milliseconds=0.1,
                metadata_cuda_milliseconds=0.1,
                model_milliseconds=1.0,
                proposal_milliseconds=0.1,
                total_milliseconds=1.3,
            ),
        )

    def propose_chain(
        self: draft.RemoteDraftRuntime,
        first_result: draft.ForwardResult,
        draft_depth: int,
    ) -> tuple[tuple[int, ...], tuple[draft.ForwardTiming, ...]]:
        del self
        assert draft_depth == 1
        assert first_result.hidden_state == f"result-hidden-{row_count - 1}"
        return (first_result.proposal_id,), ()

    runtime.reset_sequence = MethodType(reset_sequence, runtime)
    runtime._forward_many = MethodType(forbid_batch, runtime)
    runtime._forward_one = MethodType(forward_one, runtime)
    runtime._propose_chain = MethodType(propose_chain, runtime)

    token_ids = tuple(range(10, 10 + row_count))
    response = runtime.process(
        draft.DraftRequest(
            request_id="request-1",
            round_id=0,
            action="OPEN",
            token_ids=token_ids,
            hidden_rows=row_count,
            payload_bytes=row_count * draft.HIDDEN_ROW_BYTES,
            draft_depth=1,
        )
    )

    assert captured == [
        (token_id, f"hidden-{index}") for index, token_id in enumerate(token_ids)
    ]
    assert response.proposal_ids == (token_ids[-1] + 100,)
    assert response.sequence_length == row_count
    assert response.committed_forward_count == row_count
    assert [timing.row_count for timing in response.timings] == [1] * row_count


def test_failed_batched_advance_restores_round_and_committed_length():
    runtime = object.__new__(draft.RemoteDraftRuntime)
    runtime.maximum_rows_per_request = 128
    runtime.pinned_payload = FakeHiddenPayload()
    runtime.torch = SimpleNamespace(bfloat16="bfloat16")
    state = draft.SequenceState(
        request_id="request-1",
        last_round_id=0,
        draft_depth=1,
        sequence_length=64,
    )
    runtime.sequence_state = state
    runtime.request_pool_index = 0
    reserved = list(range(256))
    runtime.reserved_cache_locations = reserved
    runtime.sequence_lengths = [64]
    runtime.sequence_lengths_cpu = [64]

    def failing_forward_many(
        self: draft.RemoteDraftRuntime,
        token_ids: tuple[int, ...],
        hidden_rows: tuple[str, ...],
    ) -> draft.ForwardResult:
        del hidden_rows
        cast(draft.SequenceState, self.sequence_state).sequence_length += len(token_ids)
        raise RuntimeError("synthetic batched forward failure")

    runtime._forward_many = MethodType(failing_forward_many, runtime)

    with pytest.raises(RuntimeError, match="synthetic batched"):
        runtime.process(
            draft.DraftRequest(
                request_id="request-1",
                round_id=1,
                action="ADVANCE",
                token_ids=tuple(range(64)),
                hidden_rows=64,
                payload_bytes=64 * draft.HIDDEN_ROW_BYTES,
                draft_depth=1,
            )
        )

    assert state.last_round_id == 0
    assert state.sequence_length == 64
    assert runtime.sequence_lengths == [64]
    assert runtime.sequence_lengths_cpu == [64]
    assert runtime.reserved_cache_locations is reserved
    assert reserved == list(range(256))


def test_connection_eof_clears_unfinished_sequence(capsys: pytest.CaptureFixture[str]):
    open_request = request()
    connection = ScriptedConnection(
        draft.encode_frame(
            open_request.to_header(),
            bytes(draft.HIDDEN_ROW_BYTES),
        )
    )
    runtime = FakeRuntime()
    draft._serve_connection(
        cast(draft.RemoteDraftRuntime, runtime),
        cast(socket.socket, connection),
        "test-peer",
    )
    assert runtime.finish_count == 1
    assert runtime.sequence_state is None

    response, payload_bytes = draft.receive_frame_header(
        cast(socket.socket, BufferConnection(bytes(connection.sent)))
    )
    assert payload_bytes == 0
    assert response["status"] == "ok"
    output = capsys.readouterr().out
    assert "GLM52_FWUFF_REMOTE_DRAFT_DISCONNECT_CLEANUP" in output
    assert open_request.request_id in output


def test_parser_keeps_client_free_of_runtime_requirements():
    parsed = draft.build_parser().parse_args(
        ["client", "--iterations", "7", "--warmup-iterations", "2"]
    )
    assert parsed.command == "client"
    assert parsed.iterations == 7
    assert parsed.draft_depth == 1
    assert not hasattr(parsed, "model_path")
