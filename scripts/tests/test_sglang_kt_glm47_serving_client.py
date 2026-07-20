import json
from collections.abc import Callable, Iterator

import httpx
import pytest

from exo.shared.types.common import NodeId
from exo.worker.sglang_kt.serving_benchmark_receipt import (
    GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
    SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES,
    calculate_sglang_kt_token_ids_sha256,
)
from scripts.sglang_kt_glm47_serving_client import (
    Glm47NativeServingClient,
    Glm47ServingClientError,
    NativeGenerateRequest,
    NativeSamplingParameters,
    NativeSanityGenerateRequest,
    NativeSanitySamplingParameters,
    PreparedSanityRequest,
    build_deterministic_glm47_input_ids,
    build_glm47_server_info_identity,
    prepare_glm47_serving_workload,
    run_glm47_serving_invocation,
    run_glm47_serving_sanity,
    run_glm47_serving_workload,
)

PINNED_SERVER_INFO_FIXTURE: dict[str, object] = {
    "version": "0.0.0.dev0",
    "model_path": "/mnt/sanic/models/zai-org/GLM-4.7-Flash",
    "tokenizer_path": "/mnt/sanic/models/zai-org/GLM-4.7-Flash",
    "tp_size": 1,
    "pp_size": 1,
    "nnodes": 1,
    "node_rank": 0,
    "disable_radix_cache": True,
    "internal_states": [{"avail_req_size": 1024, "max_total_num_tokens": 32768}],
}


class StepClock:
    def __init__(self, step_ns: int = 1_000_000_000) -> None:
        self._value = 0
        self._step_ns = step_ns

    def __call__(self) -> int:
        value = self._value
        self._value += self._step_ns
        return value


SANITY_INPUT_IDS = (
    154822,
    154824,
    154827,
    20795,
    448,
    6896,
    4063,
    46,
    62674,
    3333,
    8374,
    323,
    4302,
    770,
    13,
    154828,
    154842,
)
SANITY_OUTPUT_IDS = (3257, 46, 62674, 3333, 8374)


class FakeSanityTokenizer:
    chat_template = "unused by a prepared request"

    def __init__(self, decoded_text: str = "EXO_SANITY_OK") -> None:
        self._decoded_text = decoded_text

    def apply_chat_template(
        self,
        _conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "unused"

    def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(SANITY_INPUT_IDS)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert token_ids == list(SANITY_OUTPUT_IDS)
        assert skip_special_tokens and not clean_up_tokenization_spaces
        return self._decoded_text


def _prepared_sanity(
    decoded_text: str = "EXO_SANITY_OK",
) -> PreparedSanityRequest:
    return PreparedSanityRequest(
        tokenizer=FakeSanityTokenizer(decoded_text),
        tokenizer_class="TokenizersBackend",
        chat_template_sha256=(
            "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"
        ),
        rendered_prompt_sha256=(
            "62acda2056933064acbc3211ff3474871d746256bc57e3cd195032e9507b7604"
        ),
        native_request=NativeSanityGenerateRequest(
            input_ids=SANITY_INPUT_IDS,
            sampling_params=NativeSanitySamplingParameters(
                max_new_tokens=16,
                temperature=0.0,
                ignore_eos=False,
                sampling_seed=20_260_719,
            ),
            stream=False,
            return_logprob=False,
            log_metrics=False,
        ),
    )


class GuardedOversizedStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.chunks_read = 0

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(64):
            self.chunks_read += 1
            if self.chunks_read > 17:
                raise AssertionError("client consumed data after proving line overflow")
            yield b"x" * (16 * 1024)


class TrickledUnterminatedStream(httpx.SyncByteStream):
    def __init__(self) -> None:
        self.chunks_read = 0

    def __iter__(self) -> Iterator[bytes]:
        for _ in range(64):
            self.chunks_read += 1
            yield b"x"


def _event(
    output_ids: list[int],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    finished: bool = False,
    marker: int | None = None,
) -> bytes:
    finish_reason = (
        {"type": "length", "length": completion_tokens} if finished else None
    )
    payload: dict[str, object] = {
        "text": "",
        "output_ids": output_ids,
        "meta_info": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "finish_reason": finish_reason,
        },
    }
    if marker is not None:
        payload["event_marker"] = marker
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _stream_response(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> bytes:
    chunks = [
        _event(
            [100 + index],
            prompt_tokens=prompt_tokens,
            completion_tokens=index + 1,
            cached_tokens=cached_tokens,
            finished=index + 1 == completion_tokens,
        )
        for index in range(completion_tokens)
    ]
    return b"".join((*chunks, b"data: [DONE]\n\n"))


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: Callable[[], int] | None = None,
    deadline_clock: Callable[[], int] | None = None,
    timeout_seconds: float = 30.0,
) -> Glm47NativeServingClient:
    return Glm47NativeServingClient(
        "http://127.0.0.1:62075",
        timeout_seconds=timeout_seconds,
        transport=httpx.MockTransport(handler),
        clock_ns=clock or StepClock(),
        deadline_clock_ns=deadline_clock or StepClock(step_ns=1_000_000),
    )


def _request(max_new_tokens: int) -> NativeGenerateRequest:
    return NativeGenerateRequest(
        input_ids=(10, 11, 12),
        sampling_params=NativeSamplingParameters(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            ignore_eos=True,
            sampling_seed=GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
        ),
        stream=True,
        return_logprob=False,
        log_metrics=True,
    )


def test_deterministic_workloads_have_canonical_dimensions_and_hashes() -> None:
    first = build_deterministic_glm47_input_ids("prefill")
    second = build_deterministic_glm47_input_ids("prefill")
    decode = build_deterministic_glm47_input_ids("decode")
    assert first == second
    assert len(first) == 1_024
    assert len(decode) == 128
    assert all(100 <= token_id < 154_880 for token_id in first)
    assert calculate_sglang_kt_token_ids_sha256(first) == (
        GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256
    )
    assert calculate_sglang_kt_token_ids_sha256(decode) == (
        GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256
    )

    workload = prepare_glm47_serving_workload("prefill")
    assert workload.native_request.input_ids == first
    assert workload.native_request.model_dump(mode="json") == {
        "input_ids": list(first),
        "sampling_params": {
            "max_new_tokens": 32,
            "temperature": 0.0,
            "ignore_eos": True,
            "sampling_seed": 20_260_719,
        },
        "stream": True,
        "return_logprob": False,
        "log_metrics": True,
    }
    assert workload.receipt_request.input_ids_sha256 == (
        calculate_sglang_kt_token_ids_sha256(first)
    )
    assert workload.receipt_request.return_logprob is False
    assert workload.receipt_request.log_metrics is True


def test_health_flush_server_info_and_stream_generate() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health_generate":
            return httpx.Response(200, content=b"")
        if request.url.path == "/flush_cache":
            return httpx.Response(200, text="Cache flushed.\n")
        if request.url.path == "/server_info":
            return httpx.Response(200, json=PINNED_SERVER_INFO_FIXTURE)
        assert request.url.path == "/generate"
        return httpx.Response(
            200,
            content=_stream_response(prompt_tokens=3, completion_tokens=3),
            headers={"content-type": "text/event-stream"},
        )

    request = _request(3)
    with _client(handler, clock=StepClock()) as client:
        assert client.health_generate().status_code == 200
        assert client.flush_cache().status_code == 200
        server_info = client.server_info()
        server_identity = build_glm47_server_info_identity(
            server_info,
            node_id=NodeId("dwagon"),
            host="127.0.0.1",
            port=62075,
        )
        result = client.generate(request)

    assert server_identity.version == "0.0.0.dev0"
    assert server_identity.canonical_response_sha256 == (
        server_info.canonical_response_sha256
    )
    assert server_identity.model_path == PINNED_SERVER_INFO_FIXTURE["model_path"]
    assert server_identity.node_rank == 0
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 3
    assert result.cached_tokens == 0
    assert result.stream_line_count == 8
    assert result.stream_event_count == 3
    assert result.output_bearing_event_count == 3
    assert result.first_stream_event_output_tokens == 1
    assert result.client_observed_ttft_seconds == 1.0
    assert result.client_observed_generation_window_seconds == 4.0
    assert result.client_observed_decode_tokens_per_second == 0.5
    posted = NativeGenerateRequest.model_validate_json(requests[-1].content)
    assert posted.input_ids == (10, 11, 12)
    assert posted.sampling_params.sampling_seed == GLM_4_7_FLASH_SERVING_SAMPLING_SEED
    assert requests[-1].headers["accept-encoding"] == "identity"


def test_workload_runner_flushes_every_warmup_and_sample() -> None:
    methods_and_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods_and_paths.append((request.method, request.url.path))
        if request.url.path == "/flush_cache":
            return httpx.Response(200, text="Cache flushed.\n")
        raw_request = NativeGenerateRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            content=_stream_response(
                prompt_tokens=len(raw_request.input_ids),
                completion_tokens=raw_request.sampling_params.max_new_tokens,
            ),
            headers={"content-type": "text/event-stream"},
        )

    with _client(handler, clock=StepClock(step_ns=1_000_000)) as client:
        evidence = run_glm47_serving_workload(
            client,
            prepare_glm47_serving_workload("prefill"),
        )

    assert len(evidence.warmups) == 2
    assert len(evidence.samples) == 3
    assert all(item.cached_tokens == 0 for item in evidence.samples)
    assert methods_and_paths.count(("POST", "/flush_cache")) == 5
    assert methods_and_paths.count(("POST", "/generate")) == 5


def test_public_invocation_helper_flushes_then_generates() -> None:
    methods_and_paths: list[tuple[str, str]] = []
    workload = prepare_glm47_serving_workload("prefill")

    def handler(request: httpx.Request) -> httpx.Response:
        methods_and_paths.append((request.method, request.url.path))
        if request.url.path == "/flush_cache":
            return httpx.Response(200, text="Cache flushed.\n")
        return httpx.Response(
            200,
            content=_stream_response(
                prompt_tokens=len(workload.native_request.input_ids),
                completion_tokens=workload.native_request.sampling_params.max_new_tokens,
            ),
            headers={"content-type": "text/event-stream"},
        )

    with _client(handler, clock=StepClock(step_ns=1_000_000)) as client:
        evidence = run_glm47_serving_invocation(client, workload, 7)

    assert methods_and_paths == [("POST", "/flush_cache"), ("POST", "/generate")]
    assert evidence.ordinal == 7
    assert evidence.input_ids_sha256 == workload.receipt_request.input_ids_sha256
    assert evidence.output_bearing_event_count == 32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "0.6.3.post1"),
        ("model_path", "relative/model"),
        ("tp_size", 2),
        ("pp_size", 2),
        ("nnodes", 2),
        ("node_rank", 1),
        ("disable_radix_cache", False),
    ],
)
def test_server_info_identity_rejects_unpinned_rank_zero_fields(
    field: str,
    value: object,
) -> None:
    response = dict(PINNED_SERVER_INFO_FIXTURE)
    response[field] = value

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with _client(handler) as client:
        observation = client.server_info()
    with pytest.raises(Glm47ServingClientError, match="pinned local SGLang"):
        build_glm47_server_info_identity(
            observation,
            node_id=NodeId("dwagon"),
            host="127.0.0.1",
            port=62075,
        )


def test_server_info_identity_rejects_missing_required_field() -> None:
    response = dict(PINNED_SERVER_INFO_FIXTURE)
    del response["node_rank"]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    with _client(handler) as client:
        observation = client.server_info()
    with pytest.raises(Glm47ServingClientError, match="pinned local SGLang"):
        build_glm47_server_info_identity(
            observation,
            node_id=NodeId("dwagon"),
            host="127.0.0.1",
            port=62075,
        )


def test_server_info_identity_rejects_response_changed_after_fetch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=PINNED_SERVER_INFO_FIXTURE)

    with _client(handler) as client:
        observation = client.server_info()
    changed_response = dict(observation.response)
    changed_response["tokenizer_path"] = "/different/model"
    changed_observation = observation.model_copy(
        update={"response": changed_response},
    )
    with pytest.raises(Glm47ServingClientError, match="hash is not bound"):
        build_glm47_server_info_identity(
            changed_observation,
            node_id=NodeId("dwagon"),
            host="127.0.0.1",
            port=62075,
        )


def test_generate_rejects_cache_metadata_changes() -> None:
    body = b"".join(
        (
            _event([100], prompt_tokens=3, completion_tokens=1, cached_tokens=0),
            _event(
                [101],
                prompt_tokens=3,
                completion_tokens=2,
                cached_tokens=1,
                finished=True,
            ),
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    request = _request(2)
    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="metadata changed"),
    ):
        client.generate(request)


def test_generate_rates_multitoken_first_event_from_first_chunk_size() -> None:
    body = b"".join(
        (
            _event([100, 101], prompt_tokens=3, completion_tokens=2),
            _event(
                [102],
                prompt_tokens=3,
                completion_tokens=3,
                finished=True,
            ),
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    request = _request(3)
    with _client(handler, clock=StepClock()) as client:
        result = client.generate(request)
    assert result.first_stream_event_output_tokens == 2
    assert result.client_observed_generation_window_seconds == 2.0
    assert result.client_observed_decode_tokens_per_second == 0.5


def test_generate_accepts_and_hashes_cumulative_output_ids() -> None:
    body = b"".join(
        (
            _event([100], prompt_tokens=3, completion_tokens=1),
            _event([100, 101], prompt_tokens=3, completion_tokens=2),
            _event(
                [100, 101, 102],
                prompt_tokens=3,
                completion_tokens=3,
                finished=True,
            ),
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    with _client(handler, clock=StepClock()) as client:
        result = client.generate(_request(3))

    assert result.output_ids_sha256 == calculate_sglang_kt_token_ids_sha256(
        (100, 101, 102)
    )
    assert result.output_bearing_event_count == 3


def test_generate_rejects_cumulative_output_with_changed_prefix() -> None:
    body = b"".join(
        (
            _event([100], prompt_tokens=3, completion_tokens=1),
            _event(
                [999, 101],
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
            ),
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="prior prefix"),
    ):
        client.generate(_request(2))


def test_generate_rejects_duplicate_json_event() -> None:
    event = _event([100], prompt_tokens=3, completion_tokens=1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=event + event + b"data: [DONE]\n\n")

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="duplicate event"),
    ):
        client.generate(_request(2))


def test_generate_rejects_duration_overflow_while_streaming() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_stream_response(prompt_tokens=3, completion_tokens=2),
        )

    with (
        _client(handler, timeout_seconds=1.5) as client,
        pytest.raises(Glm47ServingClientError, match="duration bound"),
    ):
        client.generate(_request(2))


def test_generate_rejects_stream_line_count_overflow() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b": keepalive\n" * 17)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="line-count bound"),
    ):
        client.generate(_request(2))


def test_generate_rejects_oversized_stream_line() -> None:
    oversized_line = b":" + b"x" * SGLANG_KT_SERVING_MAXIMUM_SSE_LINE_BYTES

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_line + b"\n")

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="oversized line"),
    ):
        client.generate(_request(2))


def test_generate_rejects_unterminated_line_incrementally() -> None:
    stream = GuardedOversizedStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="oversized line"),
    ):
        client.generate(_request(2))
    assert stream.chunks_read == 17


def test_generate_checks_total_deadline_between_trickled_chunks() -> None:
    stream = TrickledUnterminatedStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    with (
        _client(
            handler,
            deadline_clock=StepClock(),
            timeout_seconds=3.5,
        ) as client,
        pytest.raises(Glm47ServingClientError, match="duration bound"),
    ):
        client.generate(_request(2))
    assert stream.chunks_read == 1


@pytest.mark.parametrize(
    "final_event",
    [
        _event(
            [101],
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
        ).replace(b'"type":"length"', b'"type":"stop"'),
        _event(
            [101],
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
        ).replace(b'"length":2', b'"length":1'),
        _event(
            [101],
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
        ).replace(b'"length":2}', b'"length":2,"extra":true}'),
    ],
)
def test_generate_rejects_noncanonical_finish_reason(final_event: bytes) -> None:
    body = b"".join(
        (
            _event([100], prompt_tokens=3, completion_tokens=1),
            final_event,
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError),
    ):
        client.generate(_request(2))


def test_generate_rejects_stream_event_count_overflow() -> None:
    body = b"".join(
        _event(
            [],
            prompt_tokens=3,
            completion_tokens=0,
            marker=index,
        ).replace(b"\n\n", b"\n")
        for index in range(11)
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="event-count bound"),
    ):
        client.generate(_request(2))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            _event([], prompt_tokens=3, completion_tokens=1, marker=1),
            "advanced without output IDs",
        ),
        (
            _event([100, 101, 102], prompt_tokens=3, completion_tokens=3),
            "completion exceeded",
        ),
    ],
)
def test_generate_rejects_completion_progress_overflow(
    body: bytes,
    message: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body + b"data: [DONE]\n\n")

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match=message),
    ):
        client.generate(_request(2))


def test_generate_requires_two_output_bearing_events() -> None:
    body = b"".join(
        (
            _event(
                [100, 101],
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
            ),
            b"data: [DONE]\n\n",
        )
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="complete timing"),
    ):
        client.generate(_request(2))


@pytest.mark.parametrize(
    "body",
    [
        _stream_response(prompt_tokens=3, completion_tokens=2) + b"data: [DONE]\n\n",
        b"".join(
            (
                _event([100], prompt_tokens=3, completion_tokens=1),
                _event(
                    [101],
                    prompt_tokens=3,
                    completion_tokens=2,
                    finished=True,
                ),
                b"data: [DONE]\n\n",
                _event(
                    [102],
                    prompt_tokens=3,
                    completion_tokens=3,
                    finished=True,
                ),
            )
        ),
    ],
)
def test_generate_rejects_duplicate_or_post_done_data(body: bytes) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    request = _request(2)
    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match=r"after \[DONE\]"),
    ):
        client.generate(request)


def test_generate_rejects_stream_without_done_marker() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_event(
                [100, 101],
                prompt_tokens=3,
                completion_tokens=2,
                finished=True,
            ),
            headers={"content-type": "text/event-stream"},
        )

    request = _request(2)
    with (
        _client(handler) as client,
        pytest.raises(Glm47ServingClientError, match="incomplete"),
    ):
        client.generate(request)


def test_sanity_generation_records_marker_and_flushes_before_return() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/flush_cache":
            return httpx.Response(200, text="Cache flushed.\n")
        assert request.url.path == "/generate"
        posted = NativeSanityGenerateRequest.model_validate_json(request.content)
        assert posted.input_ids == SANITY_INPUT_IDS
        assert not posted.stream
        assert not posted.log_metrics
        return httpx.Response(
            200,
            json={
                "text": "EXO_SANITY_OK",
                "output_ids": list(SANITY_OUTPUT_IDS),
                "meta_info": {
                    "prompt_tokens": len(SANITY_INPUT_IDS),
                    "completion_tokens": len(SANITY_OUTPUT_IDS),
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 154813},
                },
            },
        )

    with _client(handler, clock=StepClock()) as client:
        evidence = run_glm47_serving_sanity(
            client,
            "/models/glm47",
            prepared_request=_prepared_sanity(),
        )

    assert paths == ["/generate", "/flush_cache"]
    assert evidence.output_ids == SANITY_OUTPUT_IDS
    assert evidence.server_output_text == "EXO_SANITY_OK"
    assert evidence.locally_decoded_output_text == "EXO_SANITY_OK"
    assert evidence.post_sanity_cache_flush_status_code == 200


@pytest.mark.parametrize(
    ("server_text", "decoded_text"),
    [
        ("WRONG", "EXO_SANITY_OK"),
        ("EXO_SANITY_OK", "WRONG"),
    ],
)
def test_sanity_generation_rejects_incoherent_marker(
    server_text: str,
    decoded_text: str,
) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/flush_cache":
            return httpx.Response(200, text="Cache flushed.\n")
        return httpx.Response(
            200,
            json={
                "text": server_text,
                "output_ids": list(SANITY_OUTPUT_IDS),
                "meta_info": {
                    "prompt_tokens": len(SANITY_INPUT_IDS),
                    "completion_tokens": len(SANITY_OUTPUT_IDS),
                    "finish_reason": {"type": "stop", "matched": 154813},
                },
            },
        )

    with (
        _client(handler, clock=StepClock()) as client,
        pytest.raises(Glm47ServingClientError, match="coherent marker"),
    ):
        run_glm47_serving_sanity(
            client,
            "/models/glm47",
            prepared_request=_prepared_sanity(decoded_text),
        )
    assert paths == ["/generate", "/flush_cache"]
