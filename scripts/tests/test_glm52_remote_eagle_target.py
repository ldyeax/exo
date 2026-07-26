from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import glm52_remote_eagle_target as target


@pytest.mark.parametrize(
    ("depth", "parents", "scores"),
    (
        (1, (), (0,)),
        (2, (-1, 0), (0, 1)),
        (3, (-1, 0, 1), (0, 1, 2)),
        (4, (-1, 0, 1, 2), (0, 1, 2, 3)),
        (5, (-1, 0, 1, 2, 3), (0, 1, 2, 3, 4)),
        (
            8,
            (-1, 0, 1, 2, 3, 4, 5, 6),
            (0, 1, 2, 3, 4, 5, 6, 7),
        ),
    ),
)
def test_topk1_linear_tree_matches_pinned_eagle_mapping(
    depth: int,
    parents: tuple[int, ...],
    scores: tuple[int, ...],
) -> None:
    assert target.topk1_linear_tree_indices(depth) == (parents, scores)


def test_shifted_prefill_ids_mirror_native_eagle_prefill() -> None:
    assert target.shifted_prefill_token_ids((10, 11, 12), 99) == (11, 12, 99)
    assert target.shifted_prefill_token_ids((10,), 99) == (99,)
    with pytest.raises(target.RemoteEagleTargetError, match="cannot be empty"):
        target.shifted_prefill_token_ids((), 99)


def test_payload_memoryview_uses_uint8_backing_not_bfloat16_numpy() -> None:
    class FakeUint8Tensor:
        def __init__(self, values: bytearray) -> None:
            self.values = values

        def __getitem__(self, item: slice) -> "FakeUint8Tensor":
            return FakeUint8Tensor(self.values[item])

        def numpy(self) -> bytearray:
            return self.values

    payload = target._uint8_payload_memoryview(
        FakeUint8Tensor(bytearray((1, 2, 3, 4))),
        3,
    )
    assert payload.format == "B"
    assert payload.tobytes() == b"\x01\x02\x03"


def test_configuration_is_explicit_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(target.ENABLE_ENVIRONMENT_VARIABLE, raising=False)
    with pytest.raises(target.RemoteEagleTargetError, match="exactly 1"):
        target.RemoteEagleConfiguration.from_environment()

    monkeypatch.setenv(target.ENABLE_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setenv(target.DEPTH_ENVIRONMENT_VARIABLE, "4")
    config = target.RemoteEagleConfiguration.from_environment()
    assert config.address == "10.44.0.2"
    assert config.port == 18_680
    assert config.draft_depth == 4
    assert config.maximum_rows_per_message == 512

    monkeypatch.setenv(target.DEPTH_ENVIRONMENT_VARIABLE, "9")
    with pytest.raises(target.RemoteEagleTargetError, match="depth"):
        target.RemoteEagleConfiguration.from_environment()


def test_response_requires_exact_identity_and_chain_depth() -> None:
    response: dict[str, object] = {
        "status": "ok",
        "request_id": "request-1",
        "round_id": 7,
        "proposal_ids": [1, 2, 3],
    }
    assert target._validate_response(
        response,
        request_id="request-1",
        round_id=7,
        draft_depth=3,
    ) == (1, 2, 3)
    response["round_id"] = 6
    with pytest.raises(target.RemoteEagleTargetError, match="identity"):
        target._validate_response(
            response,
            request_id="request-1",
            round_id=7,
            draft_depth=3,
        )


def test_target_admission_binds_tp2_c1_and_native_verify_depth() -> None:
    args = SimpleNamespace(
        tp_size=2,
        pp_size=1,
        ep_size=1,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=4,
        speculative_num_draft_tokens=5,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        max_running_requests=1,
    )
    assert target._validate_target_server_args(args) == 4
    args.speculative_eagle_topk = 2
    with pytest.raises(target.RemoteEagleTargetError, match="topk"):
        target._validate_target_server_args(args)


def test_target_admission_requires_disabled_radix_cache() -> None:
    args = SimpleNamespace(
        tp_size=2,
        pp_size=1,
        ep_size=1,
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=4,
        speculative_num_draft_tokens=5,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        disable_radix_cache=False,
        max_running_requests=1,
    )
    with pytest.raises(target.RemoteEagleTargetError, match="disable_radix_cache"):
        target._validate_target_server_args(args)
