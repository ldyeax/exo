# OSDI'26 CPU-GPU Hybrid MoE Reference

## Source and scope

This note summarizes:

- Wenxin Wang, Yule Hou, Yu Ji, Peng Qu, and Youhui Zhang,
  [“Achieving Cloud-Grade SLOs for Local Mixture-of-Experts Inference through
  CPU-GPU Hybrid Design”](https://arxiv.org/html/2606.10493), accepted to
  OSDI '26.
- What the paper actually demonstrates.
- How its ideas map to the GLM-5.2 work on `dwagon` and `fwuff`.

The paper's measurements are not GLM-5.2 measurements and should not be quoted
as expected performance for this cluster. Its primary models are intact FP8
DeepSeek-R1/DeepSeek-V3-class models and Kimi-K2, plus a Q4_K_M DeepSeek-R1
control. Its hardware is also substantially newer than ours. The value of the
paper is its execution design for the same broad regime: very large sparse MoE
models, low concurrency, large host DRAM, dual-socket CPUs, and one or two
consumer GPUs with insufficient VRAM for the whole model.

## Published hardware and results

The primary node in the paper has:

- Two AMD EPYC 9355 CPUs.
- 24 DDR5-6400 memory channels and 1.15 TB of DRAM.
- 1,228 GB/s theoretical aggregate DRAM bandwidth.
- Two RTX 5090 GPUs with 32 GB VRAM each.
- PCIe-attached GPUs; the design explicitly targets commodity interconnects
  rather than a datacenter NVLink fabric.

The headline results reported by the authors include:

- Up to 1,200 prompt tokens/s with single-GPU stream-loading prefill.
- More than 1,800 prompt tokens/s with two-GPU distributed stream-loading
  prefill, enough to put a 45K prompt under a 30-second TTFT target on their
  platform.
- 28 decode tokens/s on INT4 DeepSeek-R1 and 21.5 decode tokens/s on the intact
  FP8 model after CPU-kernel and synchronization work.
- Approximately 50% more aggregate throughput from a two-batch
  attention/MoE-overlap schedule.
- Less than 15% latency impact for concurrent prefill and decode in the
  intra-node disaggregated design under the paper's tested conditions.

These are phase-specific engine figures, not end-to-end OpenAI-request rates.
They must not be compared directly with our end-to-end
`output_tokens / (last_completion - first_submission)` benchmark metric.

## Core design

### 1. Keep decode's sparse experts on the CPU

For small-batch decode, routed-expert matrix-vector work is memory-bandwidth
bound. The system keeps dense layers such as attention on GPU and most routed
experts in host DRAM. This is the same fundamental split used by
KTransformers.

The paper argues that the important CPU metric at concurrency 1-4 is effective
DRAM bandwidth, not peak matrix FLOPs. It estimates CPU MoE work at about 60%
of decode time in its engine and expects decode to scale approximately with
effective DRAM bandwidth on lower-end CPUs.

Its CPU changes are more granular than simply adding threads:

- Slice each activated expert's gate/up projection along the output dimension
  to expose more parallel tasks.
- Distribute the eight activated experts across cores inside each NUMA node.
- Use per-expert barriers instead of global barriers between gate/up and down.
- Fuse activation conversion and quantization operations into the surrounding
  projection kernels.
- Minimize synchronization and cross-socket traffic.

The reported optimized FP8 GEMV path expands FP8 directly into BF16 vector
lanes, performs `vdpbf16ps`, and applies scales after accumulation. With further
tiling and scale-block work, the authors report 947 GB/s on their dual-socket
platform. This is an AVX-512 implementation; it does not imply native CPU FP8
instructions.

### 2. Move long-prompt expert compute to the GPU by streaming weights

KTransformers' AMX prefill computes routed experts on CPU. That becomes
compute-bound on long prompts even though the same CPU path is attractive for
batch-one decode.

The paper uses a separate stream-loading prefill (SLP) mode:

1. The authoritative expert weights stay in host DRAM.
2. A loader thread and CUDA stream stage the next sub-layer's weights into a
   reusable GPU ring buffer.
3. A model thread/stream computes with weights whose ready event has fired.
4. An unloader thread/stream recycles the buffer after the model signals that a
   weight is no longer in use.
5. Transfer and GPU compute overlap at sub-layer granularity.

The manual ring buffer avoids tens of thousands of `cudaMalloc`/`cudaFree`
operations. For the paper's models, a one-layer expert ring is 11.3-16.9 GB in
single-GPU mode or 5.6-8.5 GB per GPU in two-GPU mode. At very long context,
where compute rather than transfer dominates, the implementation can shrink to
a two-slot ping-pong buffer.

This is materially different from keeping a handful of frequently used experts
resident on GPU. Resident experts accelerate decode by avoiding CPU work for
hot routes. SLP streams all needed experts for a long prefill to expose dense
GEMM work to the GPU.

### 3. Use communication-lean two-GPU prefill parallelism

Distributed SLP combines striped context parallelism with a small-scale expert
parallel scheme called SmallEP.

Instead of standard expert-parallel dispatch and combine:

- All ranks gather the unsorted hidden states.
- Each rank repeats gating/sorting locally.
- Each rank retains only tokens routed to its local experts.
- It computes a local weighted partial sum.
- Ranks exchange the already reduced hidden-size results and finish the local
  reduction.

The redundant gate/sort work is intentionally traded for less communication.
The paper reports about 50% less MoE communication and 1.64x prefill throughput
over one-GPU SLP for its two-GPU setting. This design is specifically motivated
by small EP groups and PCIe-class links; it is not ordinary TP or PP.

### 4. Interleave two decode requests instead of only batching them

Plain batching at concurrency two can activate nearly twice as many CPU
experts, increasing memory traffic and worsening each request's TPOT. At the
same time, a conventional hybrid layer alternates between GPU attention and CPU
MoE work, leaving one device class idle.

The paper creates two execution threads and CUDA streams. While request A runs
attention on the GPU, request B runs MoE on the CPU, then they exchange roles.
The authors call this dual-batch attention-MoE overlap. It is a pipeline across
device classes within the same layer sequence, not pipeline parallelism across
contiguous layer ranges.

This is the paper's most directly relevant concurrency-two idea. Ordinary
continuous batching alone cannot create the same overlap if the current
KTransformers wrapper serializes its GPU and CPU phases.

### 5. Disaggregate prefill and decode inside one node

For mixed workloads, one GPU can perform stream-loading prefill while the other
serves decode or short chunked prefills. The processes share one authoritative
host-memory copy of the weights through a zero-copy ring-buffer interface, so
disaggregation does not duplicate the model in DRAM.

The paper's example policy is:

- Use chunked prefill for short prompts.
- Use distributed SLP for an isolated long prompt.
- Use one-GPU SLP when decodes are active, leaving the other GPU isolated for
  decode.

This is a latency-isolation policy. It does not make a single decode request
faster by itself.

## Mapping to dwagon and fwuff

### Where the match is strong

`dwagon` is architecturally close to the paper's target:

- Dual-socket, two-NUMA CPU with 112 physical cores.
- AMX and AVX-512/BF16 support.
- 768 GB host DRAM, enough for the complete precomputed AMXINT4 expert set plus
  the runtime's loaded dense tensors and working memory. The full BF16
  checkpoint is much larger on disk and is not claimed to fit in DRAM.
- Two RTX 3090 GPUs with 24 GB each and local NVLink.

The post-slot-change topology was re-read on 2026-07-25 rather than copied from
the older slot notes. `GPU-a442...` at `16:00.0` is NUMA 0,
`GPU-63a7...` at `d8:00.0` is NUMA 1, and the pair still reports `NV4`.
Both the ConnectX-3 (`mlx4_0`) and ConnectX-5 (`mlx5_0`) are `NODE`-distance
from the NUMA-0 GPU and `SYS`-distance from the NUMA-1 GPU. Fwuff's
`GPU-93e4...` remains on its sole NUMA node.

The existing SGLang/KTransformers path already implements the first-level
hybrid split: attention/dense work on GPU and routed experts in CPU DRAM. Our
AMXINT4 files are pre-quantized and loaded from disk; startup is not remaking
the quantization.

The paper reinforces that `dwagon` should be treated as one dual-socket
bandwidth machine for latency-oriented decode. Dividing it into two serial PP
stages prevents a single token from using both NUMA domains concurrently and
adds a stage boundary. It is capacity-efficient, but it is not the natural
batch-one topology.

### Important differences

- RTX 3090 is SM86 and has no native FP8 Tensor Core path. The paper's RTX 5090
  SLP results cannot be projected onto our GPUs by FLOP count or VRAM alone.
- Each 3090 has 24 GB rather than 32 GB. A GLM-5.2 one-layer expert ring must be
  measured; it may leave too little space for dense weights, attention
  workspace, and KV.
- The paper's AMD node has 24 DDR5 channels and 1,228 GB/s theoretical
  bandwidth. Our actual sustained local and cross-socket bandwidth must be
  measured. Core count alone is not a useful scaling proxy.
- `fwuff` has only 256 GB RAM and cannot hold the full current AMXINT4
  checkpoint alone. It cannot simply become an independent full-model replica.
- Our third GPU is across InfiniBand and host-staged on the RTX 3090 path.
  The new ConnectX-5 link is very fast for activations and artifacts, but a
  cross-host serial PP stage still adds dependency latency. More link bandwidth
  cannot remove that dependency.
- The paper presents a purpose-built engine. Its SLP, SmallEP, shared-weight
  prefill/decode disaggregation, fine-grained CPU barriers, and two-stream
  attention/MoE overlap are not features we can assume exist in the installed
  SGLang-KTransformers runtime.

## Recommended low-concurrency path

### Available or close to available

1. **Make a dwagon-only PP=1/TP=2 latency baseline.**
   Keep the entire CPU expert checkpoint in dwagon's RAM, give KTransformers
   both NUMA pools and physical-core sets, and use the two local 3090s/NVLink
   for dense tensor parallelism. This removes the two serial PP boundaries and
   the cross-host dependency. It is the highest-priority concurrency-one
   comparison with the completed 26/28/24 PP=3 run.

2. **Sweep only low-risk KTransformers scheduling knobs while resident.**
   Compare deferred experts `0`, `1`, and `2`; then compare zero resident GPU
   experts with the largest safe hot-expert budget. Record CPU/GPU branch
   timing, per-NUMA memory bandwidth, VRAM headroom, TTFT, and decode-window
   TPS. KTransformers documents `1-4` deferred experts as the intended tuning
   range, but higher values can affect quality, so preserve the coherency gate.

3. **Profile and place hot experts.**
   Record GLM-5.2 routing on representative coding/agent prompts, then compare
   uniform, frequency-based, and dynamic placement. Upstream KTransformers
   reports that frequency and dynamic placement help most when only a small
   fraction of experts fits in VRAM. Those published percentages are from a
   different model and four RTX 4090s; only the direction is transferable.

4. **Tune long-prefill chunking separately from decode.**
   The completed benchmark used 2K chunks and disabled dynamic chunking. Test
   4K, 8K, and the full 8K prompt while the server remains resident. Newer
   SGLang pipeline code can overlap chunks of the same request across stages,
   but our current fork must first prove that its dynamic/mixed-chunk path is
   compatible with KT and SM86.

5. **Evaluate GLM-5.2's MTP layer.**
   Z.ai reports up to 20% greater MTP acceptance length in GLM-5.2, and current
   SGLang exposes `NEXTN`/EAGLE speculative decoding. This is potentially the
   largest decode lever at concurrency one. It requires an Ampere-compatible
   proof, acceptance-length telemetry, and an exact-output/coherency check.
   The rank-view stager currently excludes layer 78 because the measured
   non-speculative run did not use it.

### Runtime work suggested by the paper

6. **Implement a two-request attention/MoE overlap schedule.**
   This is preferable to merely increasing SGLang's batch size for concurrency
   two. The target is to overlap GPU attention for one microbatch with CPU AMX
   experts for the other, with two explicit streams and bounded synchronization.

7. **Port fine-grained CPU scheduling before changing quantization again.**
   Split each selected expert projection across more tasks, replace global
   barriers with per-expert dependencies, fuse conversions, and keep tasks and
   weights NUMA-local. Measure sustained DRAM bandwidth per socket. This attacks
   the likely batch-one decode bottleneck without reducing the number of
   selected experts or model quality.

8. **Prototype stream-loading prefill for 8K and longer prompts.**
   Keep the AMXINT4 CPU decode path, but make long prefill a separate execution
   mode that streams an expert layer from host RAM into reusable GPU buffers.
   Start on the two local dwagon GPUs. Add the remote fwuff GPU only after local
   SLP demonstrates that PCIe transfer can be hidden behind 3090 compute.

9. **Treat SmallEP/context parallel prefill as a later remote-GPU use.**
   The third GPU may help prefill if the exchange transfers reduced hidden
   states rather than running a serial third of the layers. The new multi-rail
   network is well suited to this experiment. It requires a new execution
   backend; current PP=3 does not provide SmallEP.

10. **Use shared resident weights for lifecycle and future disaggregation.**
    Rank-specific checkpoint views and the controller-owned resident-lifecycle
    primitives now provide the building blocks for avoiding repeated discovery
    and reloads; the benchmark launcher still needs to consume them before this
    becomes the default path. A future SLP/decode split should share one
    host-resident expert allocation rather than start two independent 400+ GB
    model processes.

## Practical interpretation of the completed 26/28/24 run

The PP=3 run reached about 7 output tokens/s in the server's rolling
decode-only log at concurrency six. Its end-to-end aggregate was lower because
that benchmark intentionally included uncached 8K prefills in the denominator.
At concurrency one, the measured decode window was about 1.52 tokens/s.

That shape is consistent with the paper's diagnosis:

- PP can fill its three stages when several requests are available.
- A single request must traverse every stage serially for every token.
- CPU and GPU phases inside each stage are not fully overlapped.
- Increasing link bandwidth helps communication but does not fill otherwise
  idle stages at concurrency one.

Therefore the immediate low-concurrency objective should not be another PP
partition sweep. It should be a resident dwagon-only TP=2 baseline followed by
hot-expert/deferred-expert and MTP experiments. The OSDI-style two-batch overlap
is the preferred longer-term concurrency-two design; SLP is the preferred
long-context prefill design.

## Related upstream references

- [OSDI '26 paper, full HTML](https://arxiv.org/html/2606.10493)
- [KTransformers KT-Kernel parameters and NUMA guidance](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- [KTransformers GLM-5 launch tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5-Tutorial.md)
- [KTransformers expert placement tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)
- [SGLang pipeline-parallelism design](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pipeline_parallelism.md)
- [GLM-5.2 FP8 model card and IndexShare/MTP notes](https://huggingface.co/zai-org/GLM-5.2-FP8)
