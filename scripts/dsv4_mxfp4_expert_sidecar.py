#!/usr/bin/env python3
"""Serve profile-selected native MXFP4 expert shards over the IB interface."""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass

import torch
from kt_kernel import KTMoEWrapper

_REQUEST_HEADER = struct.Struct("!4sIIII")
_RESPONSE_HEADER = struct.Struct("!4sII")
_REQUEST_MAGIC = b"KTR1"
_RESPONSE_MAGIC = b"KTO1"

logger = logging.getLogger("dsv4_mxfp4_expert_sidecar")


def _recv_exact(connection: socket.socket, size: int) -> bytearray:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        received = connection.recv_into(view[offset:])
        if received == 0:
            raise EOFError
        offset += received
    return data


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    topk: int
    swiglu_limit: float


class ExpertShard:
    def __init__(
        self,
        *,
        layer_idx: int,
        expert_ids: torch.Tensor,
        model_path: str,
        shape: ModelShape,
        processor_count: int,
        numa_node: int,
        max_tokens: int,
    ):
        self.layer_idx = layer_idx
        self.shape = shape
        self.max_tokens = max_tokens
        self.expert_ids = expert_ids.to(dtype=torch.int64, device="cpu")
        self.global_to_local = torch.full(
            (int(self.expert_ids.max().item()) + 1,),
            -1,
            dtype=torch.int64,
        )
        self.global_to_local[self.expert_ids] = torch.arange(
            self.expert_ids.numel(), dtype=torch.int64
        )
        self.wrapper = KTMoEWrapper(
            layer_idx=layer_idx,
            num_experts=self.expert_ids.numel(),
            num_experts_per_tok=shape.topk,
            hidden_size=shape.hidden_size,
            moe_intermediate_size=shape.intermediate_size,
            gpu_experts_mask=torch.zeros(self.expert_ids.numel(), dtype=torch.bool),
            cpuinfer_threads=processor_count,
            threadpool_count=1,
            weight_path=model_path,
            chunked_prefill_size=max_tokens,
            method="MXFP4",
            numa_nodes=[numa_node],
            swiglu_limit=shape.swiglu_limit,
        )
        self.wrapper.weight_expert_ids = self.expert_ids
        self.wrapper.load_weights(
            torch.arange(self.expert_ids.numel(), dtype=torch.int64)
        )

    def validate_request(self, *, batch_size: int, hidden_size: int, topk: int) -> None:
        if not 0 < batch_size <= self.max_tokens:
            raise ValueError(
                f"invalid sidecar batch size {batch_size}; "
                f"expected 1..{self.max_tokens}"
            )
        if hidden_size != self.shape.hidden_size:
            raise ValueError(
                f"invalid sidecar hidden size {hidden_size}; "
                f"expected {self.shape.hidden_size}"
            )
        if topk != self.shape.topk:
            raise ValueError(
                f"invalid sidecar top-k {topk}; expected {self.shape.topk}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        in_range = (global_topk_ids >= 0) & (
            global_topk_ids < self.global_to_local.numel()
        )
        safe_ids = torch.where(in_range, global_topk_ids, 0)
        local_ids = torch.where(
            in_range,
            self.global_to_local[safe_ids],
            -1,
        ).contiguous()
        output = torch.zeros_like(hidden_states)
        batch_size = torch.tensor([hidden_states.shape[0]], dtype=torch.int32)
        self.wrapper.cpu_infer.submit(
            self.wrapper.moe.forward_task(
                batch_size.data_ptr(),
                local_ids.shape[-1],
                local_ids.data_ptr(),
                topk_weights.data_ptr(),
                hidden_states.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        self.wrapper.cpu_infer.sync()
        return output


class SidecarServer:
    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        shards: dict[int, ExpertShard],
    ):
        self.bind_host = bind_host
        self.port = port
        self.shards = shards
        self.compute_lock = threading.Lock()
        self.request_counts = {layer_idx: 0 for layer_idx in shards}
        self.token_counts = {layer_idx: 0 for layer_idx in shards}

    def serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.bind_host, self.port))
        listener.listen(4)
        logger.info(
            "ready on %s:%d with %d layer shards",
            self.bind_host,
            self.port,
            len(self.shards),
        )
        while True:
            connection, peer = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("client connected: %s", peer)
            threading.Thread(
                target=self._serve_connection,
                args=(connection, peer),
                daemon=True,
            ).start()

    def _serve_connection(
        self, connection: socket.socket, peer: tuple[str, int]
    ) -> None:
        with connection:
            try:
                while True:
                    header = _recv_exact(connection, _REQUEST_HEADER.size)
                    magic, layer_idx, batch_size, hidden_size, topk = (
                        _REQUEST_HEADER.unpack(header)
                    )
                    if magic != _REQUEST_MAGIC:
                        raise ValueError("invalid request magic")
                    shard = self.shards.get(layer_idx)
                    if shard is None:
                        raise ValueError(
                            f"sidecar has no native expert shard for layer {layer_idx}"
                        )
                    shard.validate_request(
                        batch_size=batch_size,
                        hidden_size=hidden_size,
                        topk=topk,
                    )
                    hidden_payload = _recv_exact(
                        connection, batch_size * hidden_size * 2
                    )
                    ids_payload = _recv_exact(connection, batch_size * topk * 8)
                    weights_payload = _recv_exact(connection, batch_size * topk * 4)
                    hidden_states = torch.frombuffer(
                        hidden_payload, dtype=torch.bfloat16
                    ).reshape(batch_size, hidden_size)
                    topk_ids = torch.frombuffer(ids_payload, dtype=torch.int64).reshape(
                        batch_size, topk
                    )
                    topk_weights = torch.frombuffer(
                        weights_payload, dtype=torch.float32
                    ).reshape(batch_size, topk)
                    with self.compute_lock:
                        start_time = time.perf_counter()
                        output = shard.forward(hidden_states, topk_ids, topk_weights)
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self.request_counts[layer_idx] += 1
                        self.token_counts[layer_idx] += batch_size
                        request_count = self.request_counts[layer_idx]
                        if request_count == 1 or request_count % 128 == 0:
                            logger.info(
                                "served layer=%d requests=%d tokens=%d "
                                "last_batch=%d compute_ms=%.3f",
                                layer_idx,
                                request_count,
                                self.token_counts[layer_idx],
                                batch_size,
                                elapsed_ms,
                            )
                    payload = output.contiguous().view(torch.uint8).numpy()
                    connection.sendall(
                        _RESPONSE_HEADER.pack(_RESPONSE_MAGIC, 0, int(payload.nbytes))
                    )
                    connection.sendall(memoryview(payload))
            except EOFError:
                logger.info("client disconnected: %s", peer)
            except Exception as error:
                logger.exception("client request failed: %s", peer)
                message = str(error).encode("utf-8")
                try:
                    connection.sendall(
                        _RESPONSE_HEADER.pack(_RESPONSE_MAGIC, 1, len(message))
                    )
                    connection.sendall(message)
                except OSError:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--bind-host", default="10.44.0.2")
    parser.add_argument("--port", type=int, default=29561)
    parser.add_argument("--processor-count", type=int, default=60)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Exclusive global layer index; defaults to all plan rows.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Remapping a decode request touches only a few dozen scalar IDs. Letting
    # PyTorch fan these tiny operations out through its process-wide OpenMP
    # pool consumed more cycles than the native expert kernel in live
    # profiling, and contended with the explicitly pinned KT worker pool.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args = parse_args()
    plan = torch.load(args.plan, map_location="cpu", weights_only=True)
    remote_expert_ids = plan["remote_expert_ids"]
    if not isinstance(remote_expert_ids, torch.Tensor):
        remote_expert_ids = torch.as_tensor(remote_expert_ids, dtype=torch.int64)
    if remote_expert_ids.ndim != 2:
        raise ValueError("remote_expert_ids must have shape [layers, experts]")
    layer_end = remote_expert_ids.shape[0] if args.layer_end is None else args.layer_end
    if not 0 <= args.layer_start < layer_end <= remote_expert_ids.shape[0]:
        raise ValueError(
            "sidecar layer range must satisfy "
            f"0 <= start < end <= {remote_expert_ids.shape[0]}, got "
            f"[{args.layer_start}, {layer_end})"
        )
    shape = ModelShape(
        hidden_size=int(plan["hidden_size"]),
        intermediate_size=int(plan["intermediate_size"]),
        topk=int(plan["topk"]),
        swiglu_limit=float(plan["swiglu_limit"]),
    )
    shards = {
        layer_idx: ExpertShard(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
            model_path=args.model,
            shape=shape,
            processor_count=args.processor_count,
            numa_node=args.numa_node,
            max_tokens=args.max_tokens,
        )
        for layer_idx, expert_ids in enumerate(remote_expert_ids)
        if args.layer_start <= layer_idx < layer_end and expert_ids.numel() > 0
    }
    SidecarServer(
        bind_host=args.bind_host,
        port=args.port,
        shards=shards,
    ).serve()


if __name__ == "__main__":
    main()
