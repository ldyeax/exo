import multiprocessing as mp
import sys
import threading
import time
from multiprocessing.synchronize import Event

import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    fail_after,
)
from loguru import logger

from exo.utils.channels import ErrorOverride, MpReceiver, MpSender, channel, mp_channel


class CustomClosedResourceError(ClosedResourceError):
    pass


class CustomBrokenResourceError(BrokenResourceError):
    pass


class CustomEndOfStream(EndOfStream):
    pass


class CustomWouldBlock(WouldBlock):
    pass


ERROR_OVERRIDE = ErrorOverride(
    closed_resource_error=CustomClosedResourceError,
    broken_resource_error=CustomBrokenResourceError,
    end_of_stream=CustomEndOfStream,
    would_block=CustomWouldBlock,
)


def foo(recv: MpReceiver[str]):
    expected = ["hi", "hi 2", "bye"]
    with recv as r:
        for item in r:
            assert item == expected.pop(0)


def bar(send: MpSender[str]):
    logger.warning("hi")
    send.send("hi")
    time.sleep(0.1)
    logger.warning("hi 2")
    send.send("hi 2")
    time.sleep(0.1)
    logger.warning("bye")
    send.send("bye")
    time.sleep(0.1)
    send.close()


def flush_before_holding_gil(
    send: MpSender[str],
    gil_hold_started: Event,
    close_allowed: Event,
) -> None:
    for message in ("running", "connecting", "acknowledged"):
        send.send(message)
    send.flush(timeout_seconds=2)
    gil_hold_started.set()

    previous_switch_interval = sys.getswitchinterval()
    try:
        sys.setswitchinterval(10)
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            pass
    finally:
        sys.setswitchinterval(previous_switch_interval)

    send.send("after gil hold")
    close_allowed.wait()
    send.close()
    send.join()


@pytest.mark.anyio
async def test_channel_ipc():
    with fail_after(0.5):
        s, r = mp_channel[str]()
        p1 = mp.Process(target=foo, args=(r,))
        p2 = mp.Process(target=bar, args=(s,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()


def test_mp_channel_flush_precedes_gil_holding_work() -> None:
    send, receive = mp_channel[str]()
    gil_hold_started = mp.Event()
    close_allowed = mp.Event()
    process = mp.Process(
        target=flush_before_holding_gil,
        args=(send, gil_hold_started, close_allowed),
    )
    final_messages: list[str] = []
    lifecycle_received = threading.Event()
    consume_flush_allowed = threading.Event()

    def collect_messages() -> None:
        final_messages.extend(receive.receive() for _ in range(3))
        lifecycle_received.set()
        consume_flush_allowed.wait()
        final_messages.append(receive.receive())

    final_receiver = threading.Thread(
        target=collect_messages,
        daemon=True,
    )
    process.start()
    final_receiver.start()
    try:
        assert lifecycle_received.wait(timeout=2)
        assert final_messages == ["running", "connecting", "acknowledged"]
        assert not gil_hold_started.is_set()

        consume_flush_allowed.set()
        assert gil_hold_started.wait(timeout=2)
        final_receiver.join(timeout=2)
        assert final_messages == [
            "running",
            "connecting",
            "acknowledged",
            "after gil hold",
        ]
    finally:
        consume_flush_allowed.set()
        close_allowed.set()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)

    assert process.exitcode == 0


def test_mp_channel_flush_times_out_and_closes_channel() -> None:
    send, _receive = mp_channel[str]()

    with pytest.raises(TimeoutError, match="flush timed out"):
        send.flush(timeout_seconds=0.01)

    with pytest.raises(ClosedResourceError):
        send.send("unreachable")

    send.close()
    send.join()


def test_mp_channel_flush_timeout_bounds_marker_enqueue() -> None:
    send, _receive = mp_channel[str](1)
    send.send("fill bounded queue")

    with pytest.raises(TimeoutError, match="flush timed out"):
        send.flush(timeout_seconds=0.01)

    send.close()
    send.join()


def test_mp_channel_blocking_receive_translates_closed_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send, receive = mp_channel[str]()

    def force_would_block() -> str:
        raise WouldBlock

    monkeypatch.setattr(receive, "receive_nowait", force_would_block)
    receive.close()

    with pytest.raises(ClosedResourceError):
        receive.receive()

    send.close()
    send.join()


def test_channel_error_override_replaces_sync_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)

    with pytest.raises(CustomWouldBlock) as would_block_info:
        send.send_nowait(1)
    assert type(would_block_info.value.__cause__) is WouldBlock

    recv.close()
    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        send.send_nowait(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send.close()
    with pytest.raises(CustomClosedResourceError) as closed_resource_info:
        send.send_nowait(1)
    assert type(closed_resource_info.value.__cause__) is ClosedResourceError


@pytest.mark.anyio
async def test_channel_error_override_replaces_async_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    recv.close()

    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        await send.send(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send, recv = channel[int](error_override_config=ERROR_OVERRIDE)
    send.close()
    with pytest.raises(CustomEndOfStream) as end_of_stream_info:
        await recv.receive()
    assert type(end_of_stream_info.value.__cause__) is EndOfStream


@pytest.mark.anyio
async def test_channel_error_override_is_preserved_by_clones():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    send_clone = send.clone()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await send_clone.send(1)

    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    cloned_send = recv.clone_sender()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await cloned_send.send(1)
