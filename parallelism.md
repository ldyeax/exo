# Parallelism Findings for Dwagon and Fwuff

Date: 2026-07-19

This document summarizes the discussion and source inspection concerning tensor
parallelism (TP), pipeline parallelism (PP), expert parallelism (EP), CPU/GPU
offload, NUMA placement, NVLink, and InfiniBand for the dwagon/fwuff Exo work.

## Executive Summary

- Exo currently models TP and PP as mutually exclusive sharding modes. It does
  not have a first-class TP+PP topology.
- The active MLX/NCCL implementation supports multi-host TP. TP=2 and TP=3 have
  already completed correctness proofs over InfiniBand.
- The experimental SGLang/KTransformers path has a PP=3, TP=1 launch-plan
  contract. It is not yet an active Exo `Instance` and has not completed an
  end-to-end three-stage model run.
- A SGLang/KTransformers stage is designed to combine one GPU with assigned CPU
  cores and NUMA memory. Routed MoE experts can be divided between resident GPU
  experts and AMX CPU experts.
- Mixed CPU/GPU expert execution is proven for GLM-4.7-Flash at PP=1. It is a
  correctness result, not a serving-throughput result, and has not yet been
  proven inside PP=3.
- Exo has no first-class distributed EP. KTransformers' `kt_ep` wrapper is local
  CPU/GPU expert routing within one stage, not cross-host expert placement.
- The preferred capacity topology remains three heterogeneous PP stages, each
  pairing a GPU with a CPU/NUMA domain. Separating all CPUs into one stage and
  all GPUs into another would increase communication and leave one resource
  class waiting during single-request decode.
- Dwagon's two RTX 3090s have an NV4/NVLink bridge. It can accelerate local
  GPU-to-GPU traffic and is particularly valuable for local TP=2. It does not
  combine the two VRAM pools automatically.
- Current network traffic is deliberately host-staged. The launch profile sets
  `NCCL_NET_GDR_LEVEL=LOC`, which disables GPUDirect RDMA.
- Fwuff's RTX 3090 reports both the GPUDirect RDMA and DMA-BUF CUDA capability
  attributes as zero. GPUDirect on the 3090s should remain an unsupported A/B
  experiment, even after the ConnectX-5 upgrade.

## What Has Been Explored

### Tensor parallelism

The MLX/NCCL work has exercised TP across Linux CUDA nodes:

- TP=2 across dwagon and fwuff.
- TP=3 across dwagon's two RTX 3090s and fwuff's RTX 3090.
- Models tested include SmolLM2, Llama 3.2 3B, Llama 3.1 8B, GPT-OSS 20B, and
  GLM-4.7-Flash.
- These tests proved model correctness, NCCL initialization, and InfiniBand
  transport selection. Several runs also proved payload on both QDR rails.

The active `MlxNcclInstance` is deliberately restricted to full-layer Tensor
shards on CUDA GPUs. It does not implement the point-to-point `send`/`recv`
operations required by MLX pipeline execution.

### Pipeline parallelism

The SGLang/KTransformers design expresses the intended GLM-5.2 topology as:

```text
PP=3, TP=1

rank 0: dwagon GPU 0 + CPU/NUMA domain, layers 0-29
rank 1: dwagon GPU 1 + CPU/NUMA domain, layers 30-57
rank 2: fwuff GPU   + CPU/NUMA domain, layers 58-77
```

The immutable launch-plan and process-spec types exist, including model,
runtime, GPU, CPU, NUMA, HCA, endpoint, and layer bindings. The launcher emits
`--pp-size 3`, `--tp-size 1`, and the explicit `30,28,20` layer partition.

This remains a pre-admission integration path. The SGLang/KTransformers types
are outside the active Exo `Instance` union until the runtime fixes, multi-stage
validation, external-process lifecycle, and API proxy are complete.

### Expert parallelism

Exo does not currently model a distributed expert placement such as:

```text
expert IDs 0-63 -> dwagon CPU/GPU domains
expert IDs 64-127 -> fwuff CPU/GPU domain
```

KTransformers uses a wrapper named `kt_ep`, but its current role is local
heterogeneous expert execution:

- A configurable subset of experts is resident on the stage GPU.
- Other experts remain in host memory and execute through CPUInfer/AMX.
- Selected CPU and GPU expert results are merged within the same stage.

That is expert offload and local expert routing, not cross-host EP. True EP
would require expert ownership metadata, token dispatch and return collectives,
load balancing, fault handling, and integration with PP/TP groups.

## Why TP and PP Do Not Compose Yet

The mathematical observation is correct: a PP stage could internally use TP,
and the next stage only needs to receive the correct activation. The missing
piece is that Exo currently treats a stage as one rank, not as a distributed
group of ranks.

The current limitations are:

1. `Sharding` is an enum with either `Tensor` or `Pipeline`.
2. Shard metadata is a union of tensor metadata or pipeline metadata rather
   than a two-dimensional coordinate.
3. MLX model loading invokes either tensor auto-parallelization or pipeline
   auto-parallelization, not both.
4. The MLX distributed setup creates one flat global group.
5. TP+PP needs separate TP subgroups and a PP group whose members are stage
   groups rather than individual ranks.
6. Placement, downloads, lifecycle, failure handling, and API routing all
   assume the existing one-dimensional rank model.

A conventional uniform topology needs a rectangular rank grid. For example:

```text
PP=2 x TP=2 = 4 GPU ranks

stage 0: GPU ranks 0,1 in TP
stage 1: GPU ranks 2,3 in TP
```

The present three-GPU cluster cannot form a uniform topology with both PP and
TP greater than one. Its factors are only PP=3/TP=1 or PP=1/TP=3. A topology
with dwagon using TP=2 and fwuff using TP=1 would have unequal stage widths and
would require a larger redesign, including activation conversion between
differently sharded and replicated representations.

## Treating NUMA Domains as Logical Resources

It is useful to view the cluster as three compute domains:

```text
domain 0: dwagon GPU 0 + dwagon NUMA 0 CPU/memory
domain 1: dwagon GPU 1 + dwagon NUMA 1 CPU/memory
domain 2: fwuff GPU     + fwuff CPU/memory
```

The SGLang/KTransformers launch-plan types can represent two logical stages on
the same physical node. Each stage has its own GPU UUID, service endpoint,
disjoint CPU set, and selected NUMA memory nodes.

There is an important physical-locality qualification. Current enumeration
places both dwagon GPUs on NUMA 0, while the ConnectX-3 HCA is on NUMA 1. A
stage using GPU 1 with CPU expert weights in NUMA 1 must move GPU/CPU expert
traffic across UPI. The logical resource model does not remove that hardware
cost.

The ideal arrangement is:

- GPU 0, its CPU expert memory, and stage-0 worker cores local to NUMA 0.
- GPU 1, its CPU expert memory, the inter-host HCA, and stage-1 worker cores
  local to NUMA 1.
- Preserve NVLink between the two dwagon GPUs if the slot geometry permits it.

If GPU-to-CPU NUMA locality and NVLink cannot both be preserved, measure both
arrangements. CPU expert weight traffic is likely more frequent than a PP
boundary activation, while NVLink is much more important for local TP=2.

## CPU and GPU Work Within a PP Stage

The experimental `SglangKtStageSpec` assigns every stage:

- One GPU UUID.
- CPU cores and NUMA memory nodes.
- CPUInfer thread and thread-pool counts.
- A KTransformers weight method.
- A resident GPU expert count.
- Optional HCA devices.

The corresponding launcher passes `--kt-cpuinfer`, `--kt-threadpool-count`,
`--kt-numa-nodes`, and `--kt-num-gpu-experts`.

The intended division is approximately:

```text
GPU: attention, routing, norms, dense/shared work, resident experts
CPU: nonresident routed experts using AMX
GPU: merge routed-expert output and continue
```

This does not mean arbitrary Transformer layers can be placed entirely on a
CPU stage. The AMX integration currently targets MoE expert MLP execution.

GLM-4.7-Flash has passed real PP=1 model-validation runs with:

- Zero resident GPU experts as a CPU-routed control.
- One resident GPU expert per routed layer.
- Four resident GPU experts per routed layer.
- Deterministic CPU and GPU components, merge evidence, and real extend/decode
  forwards.

These runs have `performance_comparable=false`. They prove correct execution,
not useful overlap or serving speed. PP=3 mixed execution remains unproven.

## Performance of Heterogeneous PP Stages

Different stage hardware is valid for correctness, but Exo does not currently
auto-balance it. Layer ranges must be selected using measured wall time, memory
capacity, and legal model boundaries.

For stage service times `T0`, `T1`, and `T2`:

```text
single-sequence decode latency ~= T0 + T1 + T2 + communication
steady multi-request throughput ~= 1 / max(T0, T1, T2)
stage i utilization under a full pipeline ~= Ti / max(T0, T1, T2)
```

For one autoregressive sequence, token `n+1` cannot start until token `n` has
passed through every stage and has been sampled. PP therefore cannot overlap
successive tokens from that sequence. Prefill chunks and multiple independent
requests can fill the pipeline, but batch-one decode remains serial.

Within a hybrid stage, ideal overlap would give approximately:

```text
stage time ~= common GPU work
             + max(CPU expert branch, GPU expert branch)
             + transfer/synchronization/merge overhead
```

Actual performance may be worse because routed-expert selection is irregular,
CPU and GPU branches must synchronize, and data crosses PCIe or UPI. The slower
branch controls the merge point.

The `30,28,20` GLM-5.2 partition is a memory-feasible initial configuration,
not a measured performance optimum. Balancing must consider:

- Attention and indexer cost per layer.
- CPU expert execution and host-memory bandwidth.
- GPU-resident expert execution.
- CPU/GPU dispatch and merge transfers.
- NUMA locality.
- PP boundary latency.
- Prompt-dependent expert frequency.
- Legal GLM IndexShare/NSA stage boundaries.

## Why an All-CPU Stage Followed by an All-GPU Stage Is Unattractive

The discussed alternative was:

```text
PP stage 0: dwagon NUMA 0 + dwagon NUMA 1 + fwuff CPU in CPU-TP3
PP stage 1: all three RTX 3090s in GPU-TP3
```

It has several problems:

- During batch-one decode, the GPU group waits for the CPU stage, then the CPU
  group waits for the GPU stage.
- A distributed CPU TP backend and AMX tensor-sharded kernels do not exist in
  the current stack.
- A full CPU GLM stage would need CPU implementations for attention, routing,
  normalization, embeddings/head where applicable, and model-specific
  attention/indexer operations, not only expert MLPs.
- CPU TP would add collectives across UPI and InfiniBand during every layer.
- GPU TP would add its own per-layer collectives, followed by a PP activation
  transfer between the two groups.
- GLM-4.7-Flash is unfriendly to equal TP=3: hidden size 2048, 20 attention
  heads, and 64 routed experts are not divisible by three.
- GLM-5.2 has 256 routed experts, also not divisible by three.

Pairing each GPU with nearby CPU/AMX resources allows the two resource classes
to cooperate inside the same layer and is a better fit for the existing code.

## Pipeline Idleness

Using three CPU/GPU domains as three PP stages is capacity-efficient but not
single-request compute-efficient. With perfectly balanced stages, each stage
is active for roughly one third of batch-one decode time. If one stage is much
slower, the other two are active for an even smaller fraction.

This is not caused by heterogeneous hardware alone. It is a consequence of
autoregressive PP. Heterogeneity makes it worse when stage service times are
not balanced.

Ways to improve utilization include:

- Multiple concurrent requests.
- Chunked or microbatched prefill.
- Better layer partitioning.
- CPU/GPU overlap within each stage.
- More effective resident-expert placement.
- Future TP or EP where communication and model dimensions make it worthwhile.

## NVLink on Dwagon

Dwagon's two RTX 3090s were previously observed as an active `NV4` pair. GA102
provides four NVLinks with a theoretical total of 56.25 GB/s in each direction,
or 112.5 GB/s bidirectional.

For the proposed PP ordering:

```text
dwagon GPU 0 -- NVLink --> dwagon GPU 1 -- InfiniBand --> fwuff GPU
```

NCCL should use CUDA P2P/IPC over NVLink for the local boundary and `NET/IB`
for the remote boundary. This must be verified in NCCL topology logs and with
an interconnect benchmark. At the time of the latest read-only check, dwagon's
`nvidia-smi` could not communicate with its driver, so the earlier `NV4` state
could not be reconfirmed live.

NVLink's value depends on the parallelism mode:

- PP transfers an activation at each stage boundary. This is relatively small,
  so NVLink mainly reduces local boundary latency and avoids host staging.
- TP communicates partial tensors through collectives repeatedly in almost
  every layer. NVLink is therefore substantially more important for local TP=2.
- A three-rank TP group spanning both hosts still crosses InfiniBand. One local
  NVLink edge does not remove the cross-host collective bottleneck.
- NVLink does not create a transparent 48 GB VRAM pool. The runtime must still
  shard weights, activations, and cache explicitly.

`NCCL_NET_GDR_LEVEL=LOC` controls GPU-to-NIC GPUDirect RDMA. It does not disable
local GPU-to-GPU NVLink P2P.

## InfiniBand and GPUDirect RDMA

The current admitted network baseline is intentionally host-staged:

```text
send:    GPU -> pinned host memory -> HCA -> InfiniBand
receive: InfiniBand -> HCA -> pinned host memory -> GPU
```

The GLM-5.2 launch environment sets:

```text
NCCL_NET=IB
NCCL_GIN_ENABLE=0
NCCL_GIN_TYPE=0
NCCL_NET_GDR_LEVEL=LOC
```

`LOC` means GPUDirect RDMA is always disabled. It does not disable normal NCCL
over verbs or GPU-to-GPU NVLink.

Read-only hardware checks found:

- `nvidia-peermem` kernel modules are installed on both systems but not loaded.
- Fwuff's RTX 3090 reports
  `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` (116) as zero.
- It also reports `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` (124) as zero.
- Dwagon's current ConnectX-3 is NUMA 1 while both GPUs are NUMA 0, requiring a
  cross-socket peer path.
- Fwuff's GPU and HCA both report NUMA 0, but the GPU capability result still
  rejects the supported GPUDirect paths.

The ConnectX-3 HCA generation is technically new enough for `nvidia-peermem`.
The primary limitation is the GeForce RTX 3090 support boundary and platform
topology, not simply the current HCA generation.

The planned ConnectX-5 cards improve bandwidth, latency, software support, and
the HCA side of GPUDirect/GIN. They do not change the 3090 CUDA capability
result. Treat GPUDirect as an unsupported experiment, not a design dependency.

GPUDirect RDMA also does not mean that the CPU disappears. It permits the HCA
DMA engine to read and write GPU memory without a host-memory bounce buffer.
The CPU still establishes connections, posts or coordinates work, handles
completion, and provides CUDA memory ordering. GPU-initiated networking such as
GIN is a separate capability with additional hardware and runtime requirements.

For PP, GPUDirect may reduce boundary latency and host-memory traffic, but the
activation messages are relatively small. It would be more consequential for
cross-host TP or future EP because those modes communicate large tensors more
frequently.

## Recommended Near-Term Direction

1. Complete the SGLang/KTransformers lifecycle and API integration without
   broadening the topology beyond PP=3/TP=1.
2. Prove PP=3 initialization and short-forward parity before loading GLM-5.2.
3. Run each stage as a GPU plus its assigned CPU/AMX expert resources.
4. Start from the legal `30,28,20` split and zero deferred experts.
5. Sweep resident GPU experts per stage and measure CPU/GPU branch timing,
   transfer volume, and merge stalls.
6. Compare concurrency one against enough independent requests to fill all PP
   stages.
7. Confirm the local dwagon PP boundary uses NVLink P2P and the remote boundary
   uses InfiniBand.
8. Preserve local TP=2 over dwagon NVLink as a latency-oriented comparison for
   models that fit and have TP-compatible dimensions.
9. Keep host-staged InfiniBand as the production baseline after the ConnectX-5
   upgrade.
10. Run GPUDirect only as a controlled A/B experiment requiring explicit NCCL
    `GDRDMA` evidence, correct output, clean HCA counters, stability, and a
    repeatable performance win.

## Future TP+PP Work

If TP+PP becomes a priority, the clean design should introduce:

- A two-dimensional rank coordinate `(pipeline_rank, tensor_rank)`.
- A stage specification containing one or more tensor ranks.
- TP subgroup communicators and PP neighbor communication between stage groups.
- Nested shard metadata that represents both layer range and tensor slice.
- Model-specific divisibility and uneven-sharding validation.
- Placement that accounts for NVLink, PCIe/NUMA locality, InfiniBand, VRAM,
  host RAM, KV cache, and CPU expert capacity.
- Stage-group lifecycle and failure handling rather than one-rank stages.
- Tests for activation representation at PP boundaries and stage-local TP roots.

A uniform PP=2/TP=2 hardware proof needs four GPU ranks. With only three GPUs,
software work can be unit-tested and simulated, but a representative hardware
validation requires a fourth GPU or another GPU node. Unequal TP widths should
be considered a separate, later feature.

## Source Pointers

- `src/exo/shared/types/worker/shards.py`: mutually exclusive TP/PP metadata.
- `src/exo/worker/engines/mlx/auto_parallel.py`: separate tensor and pipeline
  auto-parallel paths.
- `src/exo/shared/types/worker/sglang_kt.py`: per-stage GPU, CPU, NUMA, expert,
  and HCA resource contract.
- `src/exo/worker/sglang_kt/launch_spec.py`: fixed PP size, TP=1, KTransformers
  arguments, and host-staged NCCL environment.
- `FWUFFYDWAGON.md`: full implementation plan, hardware topology, and runtime
  admission status.
- `test_results.md`: TP proofs and GLM-4.7 CPU/GPU hybrid correctness receipts.

External technical references:

- NVIDIA NCCL GPU Direct troubleshooting:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html>
- NVIDIA NCCL environment variables:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
- NVIDIA CUDA GPUDirect RDMA documentation:
  <https://docs.nvidia.com/cuda/gpudirect-rdma/>
- NVIDIA GA102 architecture whitepaper:
  <https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf>
