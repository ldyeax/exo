"""Run two-rank MLX NCCL collectives across a range of payload sizes."""

import json
import os
import platform
import time
from importlib.metadata import version

import mlx.core as mx

REQUIRED_ENVIRONMENT = (
    "MLX_RANK",
    "MLX_WORLD_SIZE",
    "NCCL_HOST_IP",
    "NCCL_PORT",
    "CUDA_VISIBLE_DEVICES",
    "NCCL_NET",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "NCCL_GIN_ENABLE",
    "NCCL_GIN_TYPE",
)
BENCHMARK_CASES = (
    ("2KiB", 1024, 1000),
    ("1MiB", 524288, 200),
    ("64MiB", 33554432, 32),
)


def main() -> None:
    missing = [name for name in REQUIRED_ENVIRONMENT if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing environment variables: {', '.join(missing)}")
    if os.environ["NCCL_NET"].upper() != "IB":
        raise RuntimeError("NCCL_NET must be IB; socket fallback is not allowed")
    if os.environ["NCCL_GIN_ENABLE"] != "0" or os.environ["NCCL_GIN_TYPE"] != "0":
        raise RuntimeError("ConnectX-3 requires NCCL GIN to be disabled")

    rank = int(os.environ["MLX_RANK"])
    world_size = int(os.environ["MLX_WORLD_SIZE"])
    if world_size != 2 or rank not in (0, 1):
        raise RuntimeError(
            f"this smoke test requires ranks 0 and 1, got rank={rank}, "
            f"world_size={world_size}"
        )
    if not mx.cuda.is_available():
        raise RuntimeError("MLX CUDA backend is unavailable")

    mx.set_default_device(mx.gpu)
    probe = mx.arange(1024, dtype=mx.float32)
    mx.eval(probe)
    mx.synchronize()

    group = mx.distributed.init(backend="nccl", strict=True)
    if group.rank() != rank or group.size() != world_size:
        raise RuntimeError(
            f"NCCL group mismatch: rank={group.rank()}, size={group.size()}"
        )

    def barrier() -> None:
        token = mx.distributed.all_sum(mx.array([1], dtype=mx.int32), group=group)
        mx.eval(token)
        if int(token.item()) != world_size:
            raise RuntimeError(f"barrier reduction returned {token.item()}")

    barrier()
    reduced = mx.distributed.all_sum(
        mx.full((1024,), rank + 1, dtype=mx.bfloat16), group=group
    )
    mx.eval(reduced)
    if not bool(mx.all(reduced == 3).item()):
        raise RuntimeError("all_sum correctness failure")

    source = mx.full((1024,), rank + 1, dtype=mx.bfloat16)
    gathered = mx.distributed.all_gather(source, group=group)
    mx.eval(gathered)
    if gathered.shape != (2048,):
        raise RuntimeError(f"all_gather shape failure: {gathered.shape}")
    if not bool(mx.all(gathered[:1024] == 1).item()):
        raise RuntimeError("all_gather rank-0 payload failure")
    if not bool(mx.all(gathered[1024:] == 2).item()):
        raise RuntimeError("all_gather rank-1 payload failure")
    barrier()

    for label, elements, repeats in BENCHMARK_CASES:
        payload = mx.full((elements,), rank + 1, dtype=mx.bfloat16)
        mx.eval(payload)
        for _ in range(3):
            output = mx.distributed.all_sum(payload, group=group)
            mx.eval(output)
        mx.synchronize()
        barrier()

        started = time.perf_counter()
        for _ in range(repeats):
            output = mx.distributed.all_sum(payload, group=group)
            mx.eval(output)
        mx.synchronize()
        elapsed = time.perf_counter() - started
        maximum = mx.distributed.all_max(mx.array([elapsed]), group=group)
        mx.eval(maximum)
        max_elapsed = float(maximum.item())
        if not bool(mx.all(output == 3).item()):
            raise RuntimeError(f"{label} all_sum correctness failure")
        barrier()

        if rank == 0:
            payload_bits = elements * 2 * 8
            algorithm_gbps = repeats * payload_bits / max_elapsed / 1_000_000_000
            bus_factor = 2 * (world_size - 1) / world_size
            print(
                json.dumps(
                    {
                        "case": label,
                        "repeats": repeats,
                        "latency_us": max_elapsed * 1_000_000 / repeats,
                        "allreduce_algorithm_gbps": algorithm_gbps,
                        "allreduce_bus_gbps": algorithm_gbps * bus_factor,
                    }
                ),
                flush=True,
            )
        del payload, output
        mx.clear_cache()

    barrier()
    print(
        json.dumps(
            {
                "status": "PASS",
                "host": platform.node(),
                "rank": rank,
                "mlx": version("mlx"),
                "mlx_cuda": version("mlx-cuda-13"),
                "nccl": version("nvidia-nccl-cu13"),
                "device": str(mx.default_device()),
                "thread_local_stream_api": hasattr(mx, "new_thread_local_stream"),
                "hca": os.environ["NCCL_IB_HCA"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
