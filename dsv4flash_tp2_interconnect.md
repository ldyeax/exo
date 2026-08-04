# DeepSeek V4 Flash TP2 interconnect qualification

This note defines the local-only transport gate for the two RTX 3090
DeepSeek-V4-Flash TP2/EP2 path. It does not qualify fwuff, load model weights,
or replace the semantic, tool-call, 524K KV-quality, TTFT, or decode gates in
`dsv4flash_opencode.md`.

## Why the old `NS` result was misleading

`nvidia-smi topo -m` reports the two 3090s as `NV4`. An earlier check recorded
`nvidia-smi topo -p2p p` as `NS` and treated that as potentially missing general
peer access. The installed `nvidia-smi topo --help` defines `p` more narrowly:
it is the **PCIe-only** P2P capability. On this host the capability matrices are:

| Capability | GPU0 to GPU1 | GPU1 to GPU0 |
|---|---:|---:|
| `-p2p n` (NVLink) | `OK` | `OK` |
| `-p2p r` (peer read) | `OK` | `OK` |
| `-p2p w` (peer write) | `OK` | `OK` |
| `-p2p p` (PCIe only) | `NS` | `NS` |

The PCIe-only result does not contradict usable NVLink. Qualification now
requires the first three matrices, the CUDA peer API and exact peer copies,
NCCL route evidence, and attributed NVLink counter deltas.

## Fail-closed harness

Run the harness only when both GPUs may be borrowed for a short microbenchmark:

```bash
runtime=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python
receipt=/tmp/dsv4-tp2-interconnect-$(date +%Y%m%d-%H%M%S).json
"$runtime" scripts/qualify_dsv4_tp2_interconnect.py \
  --output "$receipt" \
  --python "$runtime" \
  --devices 0,1
```

The harness refuses to start CUDA work if either GPU has a compute process. It
then requires all of the following:

- exactly two RTX 3090/SM86 devices and an `NV4` topology relationship;
- `OK` for NVIDIA's NVLink, peer-read, and peer-write matrices;
- four active 14.062 GB/s links per GPU;
- bidirectional `torch.cuda.can_device_access_peer` and exact 4 MiB peer copies;
- a correct two-rank BF16 NCCL all-reduce at 8, 16, 24, 40, 48, and 64 KiB,
  plus a 1 MiB control shape;
- successful 48 KiB all-reduce capture and replay in a CUDA graph;
- NCCL log routes in both directions through `P2P/*`, with no `SHM` or `NET/*`
  data-channel fallback;
- a positive payload-counter delta on every link after the NCCL workload;
- at most 100 microseconds for ordinary and graph-replayed 48 KiB all-reduce,
  and at least 10 GB/s bus bandwidth at 1 MiB.

The child environment removes inherited NCCL policy, exposes only GPU 0 and 1,
disables IB, confines socket bootstrap to loopback, requests `NVL` P2P, and
retains SHM only so an unexpected fallback can be observed and rejected. Raw
NCCL logs, worker results, peer-copy evidence, and their SHA-256 digests are
stored in `<receipt>.artifacts`. The receipt is atomic and never overwritten.

Validate an existing fresh receipt without launching CUDA kernels:

```bash
"$runtime" scripts/qualify_dsv4_tp2_interconnect.py \
  --validate-receipt "$receipt" \
  --maximum-receipt-age-seconds 300 \
  --devices 0,1
```

Validation checks age, qualifier hash, GPU UUIDs, all mandatory passing gates,
both NCCL-log hashes, and that the GPUs have not become busy.

## NCCL diagnostic OpenCode launch

The diagnostic launcher is deliberately separate from the accepted fast
launcher:

```bash
scripts/dsv4_flash_hybrid_ep2_dwagon_opencode_nvlink_diagnostic.sh --launch
```

Before model preparation it runs the qualification harness. Immediately before
the final `sglang.launch_server` exec, its Python shim validates that receipt
again. A failed, missing, stale, changed, or wrong-GPU receipt blocks launch.

The diagnostic path also enables SGLang's actual P2P check and NCCL prewarm,
forces NCCL instead of SGLang custom all-reduce, and emits per-process transport
logs. It is therefore an NCCL diagnostic/A-B path, not a claim that NCCL is
faster than the accepted custom collective.

The wrapper pins and rejects attempts to weaken these accepted serving
constraints:

- TP2/EP2 on local GPUs 0 and 1, PP1, one running request;
- 524,288 context and total-token pool with `fp8_e4m3` KV;
- the 2,560-slot SWA reserve and 1,024-token chunk;
- full decode graphs and safe breakable prefill graphs at 256/512/1,024;
- eager whole-attention handling at the breakable boundary;
- checkpoint DSpark block size 5 with speculation enabled.

Running the wrapper without `--launch` only prepares the existing model path;
it does not borrow the GPUs or produce a transport receipt.

## Live result on 2026-08-03

The final implementation produced and then revalidated
`/tmp/dsv4-tp2-interconnect-live-20260803-5.json`. No model was launched.

| Gate | Result |
|---|---:|
| CUDA peer access/copy | both directions, exact |
| NCCL transport | `P2P/CUMEM` both directions, 4 channels, no fallback |
| Ordinary 48 KiB all-reduce | 42.936 microseconds |
| CUDA-graph 48 KiB all-reduce | 20.712 microseconds |
| 1 MiB bus bandwidth | 20.102 GB/s |
| Aggregate endpoint/link counter delta | 6,662,872 KiB |
| Links with positive attributed delta | all 4 links on both GPUs, TX and RX |

The aggregate counter value intentionally counts both endpoints and both
directions; it is transport attribution evidence, not an application goodput
number. The latency and bandwidth rows are the comparable performance metrics.
Two immediately preceding successful repetitions measured 25.673-25.840
microseconds ordinary and 10.233-10.284 microseconds graph-replayed at 48 KiB;
the final receipt deliberately records the slower successful observation.

This result rules out a current NCCL host-staging fallback as the explanation
for the TP2 plateau. It does not measure SGLang's custom all-reduce or the
per-layer CPU-expert barrier. The next controlled model-level comparison is the
accepted custom-all-reduce launcher versus this NCCL-forced diagnostic, using
speculative-cycle time and accepted tokens per cycle separately.
