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

As of 2026-07-25, the arXiv paper does not link a public implementation or
artifact. SLP, SmallEP, the FP8 kernel, and the dual-batch scheduler therefore
need either a later upstream release or an independent implementation; they
are not components we can install from the paper today.

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
AMXINT4 expert files are pre-quantized and loaded from disk. The newer hybrid
checkpoint also stores the large GPU-resident linear matrices in an
Ampere-friendly weight-only INT8 representation. Neither conversion is remade
at launch, and the compact GPU weights are not expanded back to BF16 while
loading.

The paper reinforces that `dwagon` should be treated as one dual-socket
bandwidth machine for latency-oriented decode. Dividing it into two serial PP
stages prevents a single token from using both NUMA domains concurrently and
adds a stage boundary. It is capacity-efficient, but it is not the natural
batch-one topology.

### Important differences

- RTX 3090 is SM86 and has no native FP8 Tensor Core path. The paper's RTX 5090
  SLP results cannot be projected onto our GPUs by FLOP count or VRAM alone.
  This does not exclude weight-only INT8: Marlin supports the current SM86
  path. Earlier Triton validation is retained as historical evidence, but all
  forward quality and performance work is Marlin-only.
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
  attention/MoE overlap are not upstream SGLang-KTransformers features. The
  sprint implementations documented below require the exact final overlay and
  retain the measured and unmeasured boundaries described for each path.

## Recommended low-concurrency path

### Available or close to available

1. **Make a dwagon-only PP=1/TP=2 latency baseline.**
   Keep the entire CPU expert checkpoint in dwagon's RAM, give KTransformers
   both NUMA pools and physical-core sets, and use the two local 3090s/NVLink
   for dense tensor parallelism. This removes the two serial PP boundaries and
   the cross-host dependency. It is the highest-priority concurrency-one
   comparison with the completed 26/28/24 PP=3 run.

2. **Sweep exact execution knobs while resident.**
   Keep deferred experts at `0`, then compare zero resident GPU experts with
   the smallest safe hot-expert budget. Record CPU/GPU branch timing, per-NUMA
   memory bandwidth, VRAM headroom, TTFT, and decode-window TPS. A GPU-resident
   expert can introduce quantization-path numerical differences, so retain the
   fixed-input coherency check.

   Deferred experts are not merely an exact scheduling optimization in the
   current KT implementation. Lower-score routed experts from layer `L` are
   computed asynchronously and their contribution is added to layer `L+1`'s
   MoE result, after it has missed layer `L+1` attention and routing. Values
   `1-4` therefore change the network. Treat them as a separate approximation
   experiment requiring fixed-logit/perplexity and representative task-quality
   gates; the short semantic warm-up alone is insufficient.

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
   but our current PP fork disables its dynamic/mixed-chunk overlap path. A
   synchronized port must prove compatibility with KT and SM86 before this is
   an available knob.

5. **Evaluate GLM-5.2's MTP layer.**
   Z.ai reports up to 20% greater MTP acceptance length in GLM-5.2, and current
   SGLang exposes `NEXTN`/EAGLE speculative decoding. The sprint implemented
   the required PP=1/TP=2 KTransformers path: layer 78 uses the persistent
   AMXINT4 artifact instead of loading its roughly 18 GiB BF16 expert body onto
   each GPU. Construction-time target embedding/LM-head sharing removed the
   original transient VRAM spike. The compact-W8 loader now also preserves
   ownership of those exact shared modules: it does not run GPTQ/Marlin
   post-processing on the target `lm_head` a second time, and it validates
   module identity before admitting the draft.

   The resulting 7,744-input/128-output c1 run passed semantic coherency and
   reached 1.7778 end-to-end output tokens/s, with a 1.90 mean acceptance
   length and 0.95 acceptance rate. Its output-token hash differed from the
   non-speculative baseline, so the result is coherent but is not claimed to
   be bit-identical. A later compact-W8 240-input/16-output TP2+MTP smoke also
   passed, including exact semantic-marker, ownership, VRAM, speculative
   acceptance, and cleanup gates. That short smoke validates integration; it
   is not the representative quality gate required before adopting MTP for
   low-concurrency c1.

### Runtime work suggested by the paper

6. **Implement a two-request attention/MoE overlap schedule.**
   The sprint implemented a KTransformers-specific two-batch schedule with
   separate task contexts. A controlled exact-extent-prefault A/B found no
   benefit on this machine: c2 was 1.8371 tokens/s with TBO and 1.8445 tokens/s
   without it. Output hashes matched across the two cases. TBO is therefore
   disabled for the measured c1/c2 policy while the branch remains available
   for instrumentation and the planned c3-on benchmark.

7. **Port fine-grained CPU scheduling before changing quantization again.**
   Split each selected expert projection across more tasks, replace global
   barriers with per-expert dependencies, fuse conversions, and keep tasks and
   weights NUMA-local. The implemented per-expert AMX dependency path is
   bit-exact against the staged path in focused tests and has run on the full
   model. Per-socket bandwidth attribution remains useful follow-up work.

8. **Prototype stream-loading prefill for 8K and longer prompts.**
   Keep the AMXINT4 CPU decode path, but make long prefill a separate execution
   mode that streams an expert layer from host RAM into reusable GPU buffers.
   The bounded two-slot implementation now exists, but the live dwagon trials
   did not establish a viable path: the 8K/four-expert ring failed closed on
   transient VRAM, while a 2K/two-expert run admitted its 144 MiB ring but made
   no first-chunk progress for seven minutes and was managed-interrupted.
   Do not add the remote fwuff GPU until this local path progresses and wins.

9. **Treat SmallEP/context parallel prefill as a later remote-GPU use.**
   The third GPU may help prefill if the exchange transfers reduced hidden
   states rather than running a serial third of the layers. The new multi-rail
   network is well suited to this experiment. End-to-end SmallEP model
   integration is implemented and covered by focused tests, but it has not run
   live on GLM-5.2 or over the remote transport. Current PP=3 remains a
   different algorithm.

10. **Use shared resident weights for lifecycle and future disaggregation.**
    Immutable shared mappings are live in the benchmark launcher. An executable
    dual-process prefill/decode runtime has also been implemented with
    identity, lease, and fail-closed resource checks and passes focused tests.
    A live process-pair run is still required before claiming latency isolation
    or throughput improvement.

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

The resident dwagon-only TP=2 work confirms that another PP partition sweep is
not the immediate low-concurrency priority. KT-backed MTP is now the strongest
measured c1 lever, while the exact TBO A/B says to leave two-batch overlap off
for c1/c2 on this hardware; c3 remains a planned on-policy measurement.
Deferred experts still belong in a separate quality-gated approximation lane.
SLP remains the architecturally preferred
long-context direction from the paper, but the current implementation is not a
viable production path until its first-chunk stall is resolved.

## 2026-07-25 implementation sprint

The first implementation and measurement pass is complete. The original
measured source identities are KTransformers
`d063aeb7a9c73db36aa87b9203c5eb440215bcd9` and its SGLang submodule
`cc6d4cfaec98a7c96adc2baae7a2816ed1c55e66`. The native build ID is
`5f4e39bc24bf41dcb2c7252fdb7aa38614093241316086244058879511c7f03a`;
the installed overlay ID is
`a1b2d826bbfd7788a693cfc5d12ce09fce7578010c45f7242bb4b031dabc0f6e`.
The install receipt is:

`/var/lib/exo/runtimes/glm52-osdi26-sglang-kt-overlay/dwagon/a1b2d826bbfd7788a693cfc5d12ce09fce7578010c45f7242bb4b031dabc0f6e/install-receipt.json`

The pre-W8 completed combined source is KTransformers
`098740a24a29c25972d1249b157840bf156f2849` with SGLang
`1218b2f8965b5a27c9d9004ff9324373414be322`. Its completed exact-source native
build ID is
`3c62024df78a94d1720e2ea3381fde9a563e6727cfc80b1a8b8d67980c19c71b`;
the build receipt has SHA-256
`6847dcee34829de83619002c00fb4ad9d7995c96ae39474168330e3d98635c0b`:

`/var/lib/exo/runtimes/glm52-osdi26-final/dwagon/3c62024df78a94d1720e2ea3381fde9a563e6727cfc80b1a8b8d67980c19c71b/build-receipt.json`

The installed overlay ID is
`a91a25bb520bd8735273669600e46cbf271749301bfa56e18b02e18f42cc25a5`;
its receipt has SHA-256
`9144511127687a2ae006b87c0895f5bcd3f15221df88d9dd8bcb63fd2e77431c`:

`/var/lib/exo/runtimes/glm52-osdi26-final-overlay/dwagon/a91a25bb520bd8735273669600e46cbf271749301bfa56e18b02e18f42cc25a5/install-receipt.json`

The current compact-W8 extension is KTransformers
`2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a` with SGLang
`f3f6ccfbbcdd5ef5e650a74eecc1a233e07cec34`. Its native build ID is
`e1044999f40b59643145bd6221580e4535e8e0f827888972d0ad7ac25a9a80e6`;
the build receipt has SHA-256
`62e5c3ddb06f51a523ba4e682e628e2386d0edb5efeb282b44009374955a3b4f`:

`/tmp/exo-kvb-w8-runtime-build-v6/dwagon/e1044999f40b59643145bd6221580e4535e8e0f827888972d0ad7ac25a9a80e6/build-receipt.json`

The installed compact-W8 overlay ID is
`b1b05ea2a1b5b893c2ce5d2e3cd907bd100b57e963725936cb743ff5bf3a64e9`;
its receipt has SHA-256
`8ec4818a265d14c305e9cb0a7f086d72bb3037f1f29ab9ab8a4c0ff76ac88bae`:

`/var/lib/exo/runtimes/glm52-osdi26-w8-overlay/dwagon/b1b05ea2a1b5b893c2ce5d2e3cd907bd100b57e963725936cb743ff5bf3a64e9/install-receipt.json`

The source is now retained directly in this repository as a local nested
submodule checkout. `vendor/ktransformers` is pinned to
`45a3a797658140dcf8426cd3f2b2c6c969f8f5d8`; its code-bearing patch terminal
remains `2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a`. The integration-only commit
advances `vendor/ktransformers/third_party/sglang` from the installed
eight-patch source above to the ninth-patch fwuff terminal
`720b40b2783b1a515134f4ab9fe820931cfbee36` and changes its URL to the local
durable source repository. The corresponding local bare repositories are
`/var/lib/exo/sources/ktransformers-glm52-osdi26-patched.git` and
`/var/lib/exo/sources/sglang-glm52-osdi26-patched.git`. They are not public
remotes; the mail patches remain the portable reconstruction authority.

The installed compact-W8 suite passed all 78 focused tests, including the
Marlin `kv_b_proj` path, its earlier Triton cross-check, and the MTP
shared-module ownership regressions.
The repository-side materializer and benchmark-receipt suite passed 45 focused
tests, including full-size 64-head serialization, immutable expert/file-table
attestation, CLI backend propagation, hybrid-manifest binding, and fail-closed
per-rank runtime-backend censuses.

The pre-W8 final overlay imports the native extension, P/D runtime, SmallEP, SLP
contract, and MTP modules without a GPU; the P/D command-line entry point also
passes its `--help` smoke check. Those are build/install checks, not live
SmallEP or P/D measurements.

Both original commits are retained on durable local branch
`exo/glm52-osdi26`, and the final commits are retained on
`exo/glm52-osdi26-final`, in the persistent source repositories under
`/var/lib/exo/sources`. The exact pre-W8 final checkout is:

`/var/lib/exo/sources/ktransformers-glm52-osdi26-final`

The pre-W8 final incremental bundles and their hashed manifest are under:

`/mnt/sanic/exo-runtime-source-bundles/glm52-osdi26-final-20260725`

The KTransformers bundle SHA-256 is
`8675c8df5dc158926b6b2309606e9321486b7f24ae2955250427fcc19a74a135`,
the SGLang bundle SHA-256 is
`7889af41436892659974752831d0c460f58c561eca45d6a1cab2b3fd5e8a6c39`,
and the bundle-manifest SHA-256 is
`dd7c16a83af602089f3ff7e625b4f1762a3ba75c40acb9cb27df309686cf512f`.
Both bundle chains reconstruct and verify to the final commits on `fwuff`.

The new peer-artifact layer also transferred the source bundles, build
receipt, and three wheels to `fwuff` as one authenticated, hash-verified,
storage-only snapshot. The cold 8,433,277-byte transfer completed in 0.347
seconds; its receipt SHA-256 is
`f5b0bc9ba9fa87c4e8f59678139a9fcb5518c8bbc3649c1b731f1d018e3e3165`.
The dwagon-side receipt mirror is:

`/var/lib/exo/peer-artifact-deployment/final-source-runtime-20260725/fwuff-five-link-transfer-receipt.json`

No wheel was installed remotely, and the dwagon-native `kt-kernel` wheel is
explicitly storage-only on `fwuff`. Each file was smaller than the configured
64 MiB striping chunk, so this small transfer used EDR only; the earlier 16 MiB
cold proof remains the evidence that all five configured rails can carry
payload concurrently.

Across the measured runtime and completed combined source, the sprint now
contains:

- A per-expert AMX dependency path with a bit-exact staged-versus-fine-grained
  test.
- A KTransformers-specific two-batch attention/CPU-MoE schedule.
- A bounded two-slot GLM expert-streaming ring and executable two-rank SmallEP
  collective, now integrated into the model path.
- A GLM-5.2 NextN path that keeps layer 78 experts in AMXINT4.
- An immutable hybrid GPU checkpoint whose ordinary large linears and all 79
  MLA `kv_b_proj` matrices remain compact W8 through loading and execution.
- Direct SM86 `kv_b_proj` execution through Marlin, including the
  metadata-planning and shared-module ownership fixes required by live MTP.
  The earlier Triton path remains only as historical cross-check evidence.
- Immutable file-backed shared AMXINT4 mappings with generation/lease
  ownership.
- An executable intra-node prefill/decode runtime with fail-closed resource,
  identity, and shared-weight lease checks.

The original installed runtime passed 42 focused SGLang tests covering TBO,
MTP, stream-prefill, and SmallEP. The installed AMX extension also passed two
bit-exact fine-grained decode cases and seven shared-mapping lifecycle tests.
The later construction-time MTP sharing, one-expert SLP mode, SmallEP model
integration, and executable P/D path have their own focused tests in the final
source. These are implementation acceptance checks, not substitutes for live
GLM-5.2 measurements.

### Persistent hybrid quantization and shared mappings

The routed-expert and MTP-expert AMXINT4 conversion remains persistent. It is
not recomputed at model launch. The immutable checkpoint at
`/mnt/sanic/glm52-AMXINT4` has content ID
`3cfb9c32388cd021a725e60022ff312688f90cdfb72ac257f2851cffb5903a07`
and manifest:

`/var/lib/exo/shared-host-weights/glm52-amxint4-manifest.json`

The GPU-side companion is the immutable checkpoint:

`/mnt/sanic/glm52-AMXINT4-W8A16-hybrid`

Its content ID is
`2d176d81a2808d097519890b51ed8a4a2df66dbc0563b6771deaf3aa59700025`,
and its `hybrid-checkpoint-manifest.json` has SHA-256
`692cc7a5ae4b2d4c5136ef8ddbd602966904eccb7e540ff2f4b641459069d14a`.
The five safetensor shards contain 20,056,714,112 payload bytes and 2,052
tensors. The conversion quantized 598 ordinary large linear matrices and all
79 MLA `kv_b_proj` matrices offline while preserving 540 source tensors,
including norms and sensitive scalars, in BF16 or FP32. Activations and the KV
cache remain BF16. Routed experts are omitted from this companion artifact and
continue to come from the authoritative AMXINT4 checkpoint, including layer 78
for MTP.

For ordinary linears, GPTQ-Marlin consumes a compact W8 representation. Each
MLA `kv_b_proj` is serialized as per-head packed GPTQ words and scales. The
Marlin backend performs a compact-to-compact repack into its kernel layout and
never materializes a persistent BF16 weight during load. The focused launcher
now accepts only `SGLANG_MLA_KV_B_W8_BACKEND=marlin`; its CLI rejects both
`auto` and `triton`, and the runtime census rejects any Triton marker.

The important integration fix was earlier than the kernel call: FlashInfer
must see the compact MLA configuration before metadata planning, and all
cached MLA flags must be synchronized when any compact `kv_b_proj` module is
active. Residual ragged-MHA planning now fails closed instead of silently
producing incoherent output. The installed Triton implementation and receipt
remain reproducibility evidence for the completed investigation, not an
admitted backend for subsequent gates.

MTP required a second ownership fix. The draft model receives the exact target
embedding and `lm_head` at construction. Its loader skips quantization
post-processing only for those identity-matched shared modules, then validates
that they are still the target objects. This prevents a second Marlin
post-process from consuming an already-processed shared `lm_head` while
retaining normal post-processing for draft-owned weights.

All AMX manifest files have their write bits removed. The direct expert loader
maps the safetensor storage read-only instead of allocating another roughly
checkpoint-sized anonymous `BufferB` copy. The manifest, inode, content,
ordered-NUMA, process-generation, and lease identities are checked before
attachment.

Cold file faults caused the first shared-mapping c1 diagnostic to spend
158.60 seconds before its first token. The passed benchmark therefore
prefaulted the resident artifact before starting SGLang and did not evict it
between coherency and timing. That removed the cold-I/O artifact from the timed
case.

The first prefault implementation read all 406,340,251,216 manifest bytes. It
was sufficient to establish a warm benchmark, but it read unrelated data and
did not prove NUMA-local placement. It has since been replaced with
safetensor-extent-aware workers:

- NUMA 0: 115,200 causal-expert extents and 181,823,078,400 bytes.
- NUMA 1: 115,200 causal-expert extents and 181,823,078,400 bytes.
- Each worker runs under strict `--cpunodebind=N --membind=N`.
- `POSIX_FADV_RANDOM` prevents readahead from first-touching the alternating
  other-node extent.
- Layer 78 is included only for an MTP run; the unrelated 35.19 GiB final
  shard is skipped.

The real extent helper completed both node plans in 27.396 seconds against the
warm cache, versus 76.620 seconds for the whole-manifest reader. A future cold
run will establish the intended per-node first touch; rereading already-cached
pages does not migrate them and is not claimed as placement proof.

### Passed PP=1/TP=2 benchmark

The final receipt is:

`/var/lib/exo/benchmarks/glm52-tp2-osdi26-prefault-tbo-fine-routing-20260725T083300Z/glm52-tp2-local-benchmark-result.json`

It passed the exact GPU/NUMA, VRAM, process-ownership, server-capacity,
semantic, concurrency-residency, cleanup, and routing-capture gates. The
semantic warm-up returned `EXO_GLM52_COHERENT` in 7.122 seconds. All three
timed-request output-token hashes exactly match the original PP=1/TP=2
baseline.

| Case | Original resident baseline | OSDI runtime, warm shared weights | Change |
| --- | ---: | ---: | ---: |
| c1 end-to-end aggregate | 1.1799 tok/s | 1.2069 tok/s | +2.29% |
| c1 wall time | 108.484 s | 106.060 s | -2.23% |
| c1 TTFT | 25.839 s | 24.867 s | -3.76% |
| c1 decode window | 1.5367 tok/s | 1.5642 tok/s | +1.79% |
| c2 end-to-end aggregate | 1.8760 tok/s | 1.8371 tok/s | -2.07% |
| c2 wall time | 136.462 s | 139.347 s | +2.11% |
| c2 concurrent-generation overlap | 86.816 s | 87.791 s | +1.12% |

The c1 result supports retaining shared mappings and fine-grained AMX as the
default low-concurrency direction. A subsequent exact A/B used the same
extent-prefault policy and outputs:

| Case | TBO on | TBO off | Off versus on |
| --- | ---: | ---: | ---: |
| c1 end-to-end aggregate | 1.206868 tok/s | 1.209928 tok/s | +0.25% |
| c1 wall time | 106.060 s | 105.791 s | -0.25% |
| c2 end-to-end aggregate | 1.837142 tok/s | 1.844473 tok/s | +0.40% |
| c2 wall time | 139.347 s | 138.793 s | -0.40% |

The TBO-off receipt is:

`/var/lib/exo/benchmarks/glm52-tp2-osdi26-prefault-tbooff-fine-20260725T051200Z/glm52-tp2-local-benchmark-result.json`

Both cases passed semantic coherency, and their timed output-token hashes
match. The small result is decisive only for configuration policy, not a broad
performance claim: TBO has no measured benefit in the tested c1/c2 cases and
is default-off there. Its branch remains available for the planned c3-on
benchmark, profiling, or future hardware.

### Matched c1 capacity and live MTP

The launcher now permits a c1-only capacity rather than forcing the c2 token
pool. At `max_total_tokens=8128`, the matched non-speculative run reached
1.194548 end-to-end tokens/s, 107.154 seconds wall time, 25.886 seconds TTFT,
and 1.56278 decode-window tokens/s:

`/var/lib/exo/benchmarks/glm52-tp2-osdi26-c1-8128-fine-shared-20260725T092500Z/glm52-tp2-local-benchmark-result.json`

The first live MTP attempt loaded layer 78 through persistent AMXINT4, but
failed before readiness when draft construction allocated duplicate embedding
and output-head modules. It needed another approximately 908 MiB while only
253-277 MiB was free. The fix shares the exact target embedding and head during
draft construction, instead of constructing duplicates and replacing them
afterward.

The fixed run passed semantic coherency and the VRAM/ownership gates:

`/var/lib/exo/benchmarks/glm52-tp2-osdi26-c1-mtp-sharefix-20260725T155200Z/glm52-tp2-local-benchmark-result.json`

It produced:

- 1.777819 end-to-end output tokens/s and 71.9983 seconds wall time.
- 25.4846 seconds TTFT and 2.73047 decode-window tokens/s.
- Mean speculative acceptance length 1.90 and acceptance rate 0.95.
- 887 MiB and 975 MiB free after readiness on the two GPUs.

This is a substantial c1 result, but not an exact-output A/B. The deterministic
prompt's output hash differed from the non-speculative baseline even though the
semantic coherency gate passed. Quality and perplexity checks are therefore
still required before making MTP the unconditional default.

### Compact-W8 TP2+MTP integration gate

The final compact-W8 source and receipt-bound overlay passed two earlier
explicit-backend TP2+MTP integration smokes:

- Triton:
  `/var/lib/exo/benchmarks/glm52-w8-explicit-triton-attested-gate-20260725/glm52-tp2-local-benchmark-result.json`
  (receipt content SHA-256
  `796648e829800b0e0db272b6aeff44fd36e730eba17d2e8cd55e55ee9a66cfb9`).
- Marlin:
  `/var/lib/exo/benchmarks/glm52-w8-explicit-marlin-attested-gate-20260725/glm52-tp2-local-benchmark-result.json`
  (receipt content SHA-256
  `e69ca5479ce408441239868a5ef9f8536fdfb6d7e24c3a344d915728e7e8cec5`).

Each historical receipt binds the backend in the top-level configuration,
process specification, and exact pre-`Popen` environment. Before accepting
timing, the launcher also binds hybrid manifest SHA-256
`692cc7a5ae4b2d4c5136ef8ddbd602966904eccb7e540ff2f4b641459069d14a`
and content ID
`2d176d81a2808d097519890b51ed8a4a2df66dbc0563b6771deaf3aa59700025`,
then requires the merged rank log to attest exactly 79 compact modules with 32
local heads on each TP rank and no conflicting backend markers. Both historical
backend censuses passed.

Both semantic warm-ups returned exact `EXO_GLM52_COHERENT` markers in nine
tokens. Both 240-input/16-output cases reproduced output-token SHA-256
`6fe701f56abd403cf97996be6d050eb6c0868dfa521b870e05583a541f259cd1`,
matching the MTP-off control at
`/var/lib/exo/benchmarks/glm52-w8-final-target-gate-20260725/glm52-tp2-local-benchmark-result.json`.
Native speculative metrics recorded eight accepted draft tokens out of eight
in both cases. TBO was off, the KV cache remained BF16, all process-ownership
checks passed, and managed cleanup released both GPUs and all four ports.

| Explicit `kv_b_proj` backend | Aggregate output | TTFT | Generation window | Free MiB by rank |
| --- | ---: | ---: | ---: | ---: |
| Triton | 1.704140 tok/s | 3.47423 s | 2.53663 tok/s | 11,831 / 11,919 |
| Marlin | 1.837885 tok/s | 2.87814 s | 2.57460 tok/s | 11,905 / 11,993 |

This established end-to-end backend parity for that integration case, not a
statistically controlled backend comparison. Marlin is now the only admitted
backend for new quality and benchmark receipts. The first forced-Marlin receipt at
`/var/lib/exo/benchmarks/glm52-w8-forced-marlin-mtp-gate-20260725/glm52-tp2-local-benchmark-result.json`
included a 221.8-second cold build of the SM86 BF16
`moe_wna16_marlin` specialization in tvm-ffi's shared cache.
The rank logs show that compilation finished immediately before the first
prefill; it was compiler latency, not a collective deadlock. A production
runtime should prebuild and attest the required BF16 SM86 specializations in a
content-addressed `TVM_FFI_CACHE_DIR` rather than relying on a host-global
first-request build.

The earlier MTP-off control used implicit `auto` selection and lacks a compact
backend census, so it is not admissible as the matched control for the new
Marlin-only quality gate. That control must be rerun with 78 Marlin modules per
TP rank; MTP-on must attest 79.

The 240/16 case is deliberately an integration, coherency, capacity, and
ownership smoke. Its high 100% acceptance on only eight draft tokens is not a
representative acceptance estimate, and neither it nor the exact semantic
marker substitutes for fixed-logit/perplexity and representative task-quality
gates. Those quality gates remain required before using TP2+MTP as the default
low-concurrency c1 path. The large remaining VRAM budget is the intended
enabler for a larger SLP ring and makes TP1 P/D capacity plausible, but this
smoke validates neither of those later modes.

### Fwuff remote K4 speculative decoding

The first integrated remote-draft path is now operational. Dwagon keeps all 78
target layers in TP2 over its two NVLinked RTX 3090s. TP rank zero sends
accepted token IDs and BF16 target-hidden rows through persistent pinned host
buffers over the dedicated EDR link. Fwuff keeps only layer 78 and its draft KV
resident: its RTX 3090 runs attention, dense, embedding, and head work while
its 60 AMX cores run the existing AMXINT4 MTP experts. Fwuff returns a linear
top-k-one candidate chain; both dwagon ranks reconstruct the same tree and use
SGLang's native target-only verifier. The measured paired bridge admits c1,
TP2, radix-cache off, CUDA graphs off, overlap off, and target-only acceptance
thresholds of 1.0. Paired target evidence remains K4-only. The current
source/protocol admits a fixed K1–K8 per request, with only the isolated K5
wire/runtime smoke below beyond K4.

The speed-first ladder selected K4:

- Isolated resident fwuff p50 EDR round trips were 9.908, 14.288, and
  18.715 ms for K2, K3, and K4. At K4 the per-forward model p50/p95 was
  3.747/3.816 ms.
- Raw `TCP_NODELAY` was the best synchronous transport. Its 12 KiB p50/p95
  was 0.512/0.730 ms; the two-slot transport reached 3,220 messages/s but was
  not needed to prove the synchronous path.
- The first 240-input/16-output K4 smoke reproduced the exact local-control
  output SHA-256
  `6fe701f56abd403cf97996be6d050eb6c0868dfa521b870e05583a541f259cd1`.
  It reached 4.51690 generation-window and 2.17495 end-to-end tokens/s. Against
  the existing exact Marlin MTP control's 2.57460 and 1.83805 tokens/s, those
  are 75.44% and 18.33% improvements. No new dwagon-only control was run.

The first longer attempt exposed a bridge lifecycle bug after six successful
K4 rounds. The target crossed sequence 256 and asynchronously reported a CUDA
illegal access; fwuff had returned every response successfully and remained
healthy. DeepSeek DSA forces page size 64, so page size one was not a valid
workaround. The remote worker had skipped native EAGLE's top-k-one cache-shadow
transaction before verification:
`get_last_loc_large_page_size_top_k_1`, backed-up paged allocation,
`assign_draft_cache_locs`, then allocator restore. Restoring that transaction
without running local draft compute fixed the fault. The hardened service also
compacts large timing responses and clears unfinished draft state on peer EOF.

The fixed paired 240-input/128-output K4 run completed through multiple page
boundaries:

`/var/lib/exo/benchmarks/glm52-fwuff-remote-eagle-k4-paired-smoke-20260726/attempt-06-native-page-bookkeeping/client-240x128.json`

The immutable receipt file SHA-256 is
`8982a2e7aa4562b6b53a73d3df18092c78f2c8b6eb46c55f847389ca176568bd`;
its self-canonicalized content SHA-256 is
`361d8d6437296ee8f63ecc3bf3e523592fc8bb56fcf0b25aa0179e7958854601`.
It records:

- 19.0310 seconds end to end, 3.22329 seconds TTFT, 8.03440
  generation-window tokens/s, and 6.72587 end-to-end tokens/s.
- 94 of 136 draft tokens accepted, 0.691176 acceptance rate, 3.76471 mean
  emitted tokens per verify, and histogram `[6, 4, 1, 4, 19]` for zero through
  four accepted draft tokens across 34 verifies.
- Full-output SHA-256
  `a6597ed9d0a561c0045e80270556d4ce21a603dbf3fcc9e1b83633dccf09a817`.
  Its first 16 tokens exactly match the prior local Marlin control and the
  semantic first sentence is coherent. There is no matched 128-token
  dwagon-only oracle, so this is not promoted to the still-pending
  representative quality gate.
- A clean fwuff `FINISH` at sequence 363, zero remote errors, 78 compact
  Marlin `kv_b_proj` modules per target rank, no target fault marker, and
  ownership-clean target teardown. Fwuff remains resident and clean for the
  next paired iteration.

Fwuff's decode forwards remained small relative to target verification:
model p50/p95 was 3.813/3.936 ms and total p50/p95 was 4.478/4.625 ms over 222
committed and tentative forwards. A K4 chain therefore costs about 18 ms on
fwuff while the observed end-to-end verify cycle averaged roughly 465 ms.
Transport-only double buffering can recover only a small fraction of the
remaining time; deeper/adaptive trees or reducing target verification cost
have more leverage.

The paired 250 ms telemetry is preserved below the same receipt directory.
During the exact request window, dwagon GPU0 averaged 9.03% utilization
(8% median) and GPU1 94.22% (100% median), an 85.2-point mean gap. The
correlation was weakly negative at -0.120, with no sustained anti-phase
transfer. Fwuff averaged 5.58% at the NVML sampling resolution and was nonzero
in 28.9% of samples; its approximately 4 ms bursts are heavily aliased. Fwuff
was active during four of the six GPU0-idle frames, so it filled a small
cluster-wide gap without filling GPU0 itself. Remote speculation therefore
improved c1 materially while the target TP imbalance remains a separate
scheduling/spin problem.

#### Boundary-canonicalized batched prefill

Fwuff now batches only prefill-sized committed bursts. The request keeps its
full preallocated page-64 KV mapping; ordinary `EXTEND` consumes the first
`N-1` shifted token/target-hidden pairs, and the final committed row uses the
historical `DECODE` path. Bursts below 64 rows remain entirely serial, so
accepted decode blocks cannot change the measured K4 trajectory. The response
timing records the batched row count explicitly, and failed actions restore
the logical round, GPU/CPU sequence lengths, allocator state, and incremental
mapping.

This policy came from two measured iterations rather than an up-front sweep:

- Pure batching made the 128+112 committed prefill work about 1.006 seconds
  faster and reduced TTFT from 3.22329 to 2.11392 seconds. It also batched
  twelve five-row decode advances, however, and the `EXTEND` boundary hidden
  state entered the recursive tentative chain. Acceptance fell from 0.691176
  to 0.440217, generation fell to 6.21012 tokens/s, and end-to-end time rose to
  22.5651 seconds. The exact first 16 output tokens and all lifecycle checks
  still passed. This result is retained as a diagnostic, not a winning
  configuration.
- Serializing `N<64` and canonicalizing the final prefill row through
  `DECODE` removed both perturbations. The exact N=2/5/64/65 gate passed:
  N=2/5 used serial forwards, while N=64/65 reported row counts
  `[N-1, 1, 1, 1, 1]` for bulk EXTEND, boundary DECODE, and three tentative
  forwards. One 128+112 warmup then completed the committed work in
  47.0510 and 30.7971 ms and finished cleanly at sequence 240.

The final paired 240-input/128-output result is:

`/var/lib/exo/benchmarks/glm52-fwuff-remote-eagle-k4-batched-prefill-20260726/attempt-02-boundary-decode-240x128/client-240x128.json`

Its immutable file SHA-256 is
`0bbb22d2bf3a285480021bc8e7a19262565ce4cdc93b400f1e0d5b7355dcd79e`;
its self-canonicalized content SHA-256 is
`d29d134df1c16aa5c98f1b89b744870150b32f7cfe713c8b5a0885668ca9ae98`.
The frozen fwuff component/comparison receipt is
`/var/lib/exo/benchmarks/glm52-fwuff-remote-eagle-k4-batched-prefill-20260726/attempt-02-boundary-decode-240x128/fwuff-component-comparison.json`;
its file/content SHA-256 values are
`d052fa41efda1067185c1ef04399c5ffe5f03d6657fb73077af766ca46c9d8a9`
and
`4a24d360faeac33d322706790b6b436d11f8bea06de8ee3583b3631b77247961`.
It binds all 33 fwuff responses, 30 decode advances, contiguous rounds 0–32,
and the clean `FINISH` at sequence 365 to the frozen target and service
sources.
Against the serial-prefill paired baseline it records:

- TTFT 2.13279 seconds, down 1.09050 seconds or 33.83%.
- 8.64701 generation-window tokens/s, up 7.62%, and 7.60963 end-to-end
  tokens/s, up 13.14%.
- 16.8208 seconds end to end, down 2.21019 seconds or 11.61%.
- 96/124 accepted draft tokens, 0.774194 acceptance, 4.12903 mean emitted
  tokens per verify, and histogram `[3, 2, 3, 4, 19]` across 31 verifies.
  These improve on the serial-prefill baseline's 0.691176, 3.76471, and
  34 verifies.
- The exact historical 16-token control prefix passes. The full output SHA-256
  is `9f8916313367d0fb71647c43ee0d3a3863c41c59aa81a50a2dd04bf1b2d8db3e`
  and shares 27 tokens with the earlier 128-token paired output before the
  target-verified path diverges. The new continuation remains coherent and
  follows the prompt's record structure, while the older continuation became
  a repeated “Sigh” loop. This is evidence of a sound target-verified path,
  not a claim of bitwise full-output equivalence or completion of the still
  pending representative quality gate.
- Fwuff returned a clean `FINISH` at sequence 365; the target recorded no
  fault marker, both ranks retained 78 compact Marlin `kv_b_proj` modules, and
  owned target processes were cleaned. Fwuff source SHA-256 is
  `c7fda766479d8b8a5cfc9a32660e8d1ca75c503f73168def029fd22a805698b6`
  and its service remains resident and clean.

The focused gate and warmup receipts are
`/var/lib/exo/benchmarks/glm52-fwuff-boundary-prefill-equivalence-20260726T021000Z/receipt.json`
and
`/var/lib/exo/benchmarks/glm52-fwuff-boundary-prefill-warmup-20260726T021100Z/receipt.json`.
No additional dwagon-only model run was used. A concurrent Ollama model briefly
occupied fwuff VRAM during the failed pure-batch iteration, then expired
naturally; the final gate and paired measurement ran with PID 2821961 as the
only fwuff GPU process and about 19.85 GiB free before target connection.

Here `K` is the number of future tokens recursively proposed by fwuff for one
dwagon target-verification pass: K4 verifies four draft tokens and can emit
one through five tokens; K5 verifies five and can emit one through six. It is
not a model or quantization version. The K4 acceptance histogram
`[3, 2, 3, 4, 19]` means 19/31 verifies accepted all four proposals. Its
measured survival through proposal positions one through four was 0.903,
0.839, 0.742, and 0.613. A fwuff tentative forward cost 4.480 ms mean
(4.473 ms p50, 4.589 ms p95), so K5 is the next decisive fixed-depth test;
K6 is not yet justified without measuring fifth-token acceptance and target
verifier scaling.

The source-level depth bound is now eight rather than four, while a request
retains one fixed depth from `OPEN` through `FINISH`. The top-k-one tree,
fwuff tentative rollback, TP broadcast, and inherited page allocator are
depth-generic in source; target-verified evidence remains K4.
Focused Ruff/compile checks and 36 protocol tests passed. Fwuff service PID
2826982 loaded source SHA-256
`7cbc255a411449a1e7bb6ff668adcfc0bbee17d16af2f713551a5b0cfd8bff09`
and advertised maximum depth eight. The committed source is AST-identical
after formatting; its SHA-256 is
`934e86988c03d491dffe67f8f522c58a82b0d6bcbd1cbedf78bac7fc841bc364`.
A two-round K5 admission smoke returned two chains of exactly five
tokens, followed by a clean `FINISH` response at round 2 and sequence 2. Its
frozen receipt is
`/var/lib/exo/benchmarks/glm52-fwuff-isolated-draft-depth8-20260726T022731Z/attempt-01/fwuff-k5-admission-smoke.json`
with SHA-256
`d3bcfb3c19d92497c3f491b5e971749b7a0d7f39425db9c9328e47fa636aa5c3`.
The first round included 951.7 ms of one-time compilation; the next round
completed in 23.825 ms over EDR. Two synthetic rounds are an admission check,
not a K5 performance result. No paired K5 target benchmark was run before
this checkpoint.

The next speculative-decoding leverage is therefore one matched fwuff+TP2 K5
240/128 run against the frozen K4 receipt. Retain K5 only if quality/lifecycle
pass and generation throughput improves by at least 3%; run K6 only if the
fifth-token conditional acceptance is at least 0.80 and the inferred marginal
target row costs at most 25 ms. Per-round adaptive depth is later work because
the current wire state, TP tensor shapes, page shadow transaction, and SGLang
accounting intentionally remain launch-fixed. Any asynchronous lookahead must
retain versioned draft KV and exact rollback. The separate GPU0/GPU1 target
imbalance also remains a larger compute-scheduling target than fwuff's
now-sub-100-ms committed prefill contribution.

### Live SLP attempts

Local SLP did not produce a benchmark result worth retaining:

- The 8K/four-expert case failed closed because its 288 MiB device ring plus
  the 512 MiB safety margin exceeded transient free VRAM.
- The first 2K/one-expert attempt exposed an inherited
  `host_buffer_experts >= 2` guard. That guard belonged to the legacy
  per-expert double-buffer mode; SLP already uses two ring slots. The final
  source explicitly admits a one-expert ring slot while retaining the old
  requirement for legacy mode.
- The 2K/two-expert case admitted a 144 MiB device ring and 144 MiB pinned host
  ring per rank, but produced no first chunk after seven minutes. It was
  managed-interrupted at 629.156 seconds and all owned processes were cleaned
  up.

These are fail-closed or managed failures, not performance samples. SLP is
currently nonviable on this runtime until the first-chunk stall is diagnosed.

The same resident server also captured eight deterministic coding/agent
prompts with exact routing (`max_deferred_experts_per_token=0`, no dynamic
updates). The materialized frequency input is:

`/var/lib/exo/benchmarks/glm52-tp2-osdi26-prefault-tbo-fine-routing-20260725T083300Z/routing/glm52-routing-frequency-f12495d2d88068ae4db7eb76cca5edc09df28b614a8d1bf6203086bd6161a002.pt`

At the current 16K-token c2 capacity and VRAM floor, no resident expert fits
the fail-closed budget. The profile is immediately useful for a matched
short-context residency A/B, not as permission to overcommit the measured
16K lane.

### Remaining live measurements

Persistent hybrid quantization, compact Marlin kernels, the TP2+MTP
integration smoke, the paired fwuff K4 path, boundary-canonicalized fwuff
prefill, and the original c1/c2 TBO isolation are complete. SmallEP model
integration and the executable shared-weight P/D process path are
focused-tested but not live. Under the current instruction not to run more
dwagon-only tests, the remaining order is:

1. Treat fwuff prefill and EDR transport as measured rather than continuing to
   optimize them. Run the prepared fixed-K5 paired 240/128 gate against K4
   without changing rollback or the serial `N<64` decode path. Admit K6 only
   under the acceptance and marginal-verifier-cost stop gate above; implement
   per-round adaptive depth only after the fixed-depth evidence warrants its
   wider protocol and accounting changes.
2. Keep the representative local TP2 MTP-on/MTP-off quality gate pending until
   another dwagon-only campaign is authorized. It still requires 79/78 Marlin
   modules per rank plus fixed-logit, perplexity, and representative
   coding/agent tasks; the remote 128-token run does not replace it.
3. Once that path and the other runtime pieces are stable, benchmark the
   intended scheduler policy: TBO off for low-concurrency c1 and TBO on for c3.
   Keep the earlier c2 exact A/B as historical evidence rather than projecting
   it onto c3.
4. Instrument SLP's AMXINT4 export, pinned-host preparation, H2D transfer,
   ready-event and recycle-event waits, and GPU compute separately. Prove local
   first-chunk progress before another chunk-size sweep.
5. Validate TP2 coherency and performance before increasing the local SLP ring
   size. Exercise integrated SmallEP only once that local SLP path is viable;
   then add `fwuff` as a communication-lean prefill participant exchanging
   reduced hidden-size results, not as another serial PP stage.
6. Attempt TP1 prefill/decode only after the larger local SLP ring is viable.
   Verify both P/D leases bind the same immutable content and generation before
   measuring latency isolation.
7. Use the captured route profile for the short-context 15-expert
   uniform-versus-frequency A/B. Keep the 16K exact lane at zero resident
   experts.

## Upstream integration audit

This audit was last refreshed on 2026-07-25 against KTransformers
`a8062bfa7e1060ce5855b5f1ad6aa6b116678307` and SGLang
`f5155d960286db25952217f343ee0d3c358f7f77`. The sprint runtime is based on
older pinned revisions plus local patches, so an upstream feature is not
automatically present in the installed environment. The paper and its
[USENIX entry](https://www.usenix.org/conference/osdi26/presentation/wang-wenxin)
still do not link an artifact.

The compact-MLA audit also checked
[official Triton](https://github.com/triton-lang/triton) and SGLang's related
absorbed-MLA LoRA Triton work in
[PR #25001](https://github.com/sgl-project/sglang/pull/25001) and
[PR #27087](https://github.com/sgl-project/sglang/pull/27087). Those PRs
implement and tune LoRA correction kernels rather than compact W8 base-weight
execution, so they were shape and launch references, not drop-in INT8 support.
The installed official Triton release already targets SM86, so a platform fork
would add maintenance risk without unlocking this kernel. The examined
[AppMana SGLang fork](https://github.com/AppMana/forks-sglang/blob/deepseek_v4_ampere/python/sglang/srt/layers/quantization/dsv4_int.py)
expands its INT8 weights into a persistent BF16 parameter; that defeats this
artifact's memory objective and was not integrated.

### Operational versus scaffold state

Here, **operational** means that the end-to-end code path exists in the
completed sprint source or has produced a recorded run. It does not mean that
it has already been benchmarked on every GLM-5.2 topology. **Scaffold** means
that a launcher, contract, or useful upstream primitive exists but the claimed
optimization is not implemented end to end.

| Area | Current sprint state | Evidence and boundary |
| --- | --- | --- |
| 26/28/24 PP=3, AMXINT4 | **Operational and measured** | The coherency-gated concurrency 1/3/6 run is the current baseline. Its approximately 7 tokens/s server log at concurrency six is decode-window throughput, not the lower end-to-end aggregate that includes uncached prefill. |
| Dwagon-only PP=1/TP=2 | **Operational and measured** | The passed 7,744-input/128-output c1/c2 receipt records fixed GPU/NUMA ordering, exact capacity, a semantic warm-up, zero cached prompt tokens, simultaneous c2 residency, and cleanup. Warm c1 reached 1.2069 output tok/s end to end and 1.5642 tok/s in its decode window. |
| Persistent GPU W8 and compact MLA `kv_b_proj` | **Operational, tested, and live** | The immutable hybrid artifact keeps 598 ordinary linears and 79 `kv_b_proj` matrices compact while AMXINT4 remains authoritative for routed and MTP experts. Marlin uses a compact-to-compact repack with no BF16 weight expansion. The installed suite passed its Marlin tests and an earlier Triton cross-check. Historical explicit-Triton and explicit-Marlin TP2+MTP receipts bind the immutable hybrid manifest and a machine-checked 79-module-per-rank runtime census. New launches fail closed on Marlin; the matched Marlin MTP-off quality control still needs a fresh 78-module-per-rank receipt. |
| Routing-aware GPU residency | **Trace and planner operational; residency A/B pending** | Merged KTransformers [PR #1796](https://github.com/kvcache-ai/ktransformers/pull/1796) supplies arbitrary per-layer masks. The sprint captured an exact eight-prompt GLM-5.2 route trace and materialized a hashed frequency input. The 16K lane has no safe resident budget; the admitted next experiment is the matched short-context 15-expert A/B. Candidate logic in [PR #2064](https://github.com/kvcache-ai/ktransformers/pull/2064) and co-activation-aware [PR #2093](https://github.com/kvcache-ai/ktransformers/pull/2093) remains useful policy work. |
| Long-prefill expert streaming | **Implemented and live-attempted; currently nonviable** | Local SGLang commit `0fbf63c2b` adds a bounded two-slot GLM ring, BF16 export from persistent AMXINT4, explicit ready/recycle events, and the reused-slot safety rule. Final commit `1218b2f89` makes the one-expert SLP slot distinct from the legacy double-buffer contract. The 8K/four-expert trial failed closed on VRAM; 2K/two-expert admitted a 144 MiB ring but made no first-chunk progress and was managed-interrupted. No SLP speedup is claimed. The [upstream slot fix](https://github.com/kvcache-ai/sglang/commit/b3356b6c46c137a3bec1c67974f180092d0fcc92) remains a required donor. |
| Fine-grained AMX decode | **Implemented, bit-exact, and live** | Local KTransformers commit `13e73b5` replaces phase-wide decode joins with per-expert dependencies behind `KT_AMX_FINE_GRAINED_DECODE=1`. The installed native extension passes staged-versus-fine bit equality at two thread counts. It was enabled in the passed benchmark; c1 decode-window throughput improved 1.79%, while normal run-to-run variation still requires another A/B before assigning all of that delta to the scheduler. |
| Two-request attention/CPU-MoE overlap | **Implemented and live; off for measured c1/c2** | Local SGLang commit `e44e4f614` adapts split/recombine and yield sequencing to independent KT task contexts without a DeepEP backend. Correctness tests pass and the c2 benchmark proved concurrent generation. In the controlled A/B, c2 was 1.837142 tokens/s on versus 1.844473 off, with matching output hashes. TBO therefore remains off for c1/c2; the requested c3-on case is pending. Generic SGLang [TBO PR #4068](https://github.com/sgl-project/sglang/pull/4068) and unified KT [PR #12834](https://github.com/sgl-project/sglang/pull/12834) were design donors, not drop-in implementations of this schedule. |
| GLM-5.2 MTP/IndexShare | **KT-backed, tested, and live measured; representative quality pending** | Layer 78 uses persistent KTransformers AMXINT4 experts. Construction-time sharing avoids duplicate target embedding/head allocation, and exact-identity loader ownership prevents a second Marlin post-process of the shared `lm_head`. The earlier long c1 run reached 1.777819 end-to-end and 2.73047 decode-window tokens/s, with 1.90 acceptance length and 0.95 acceptance rate. The final compact-W8 240/16 smoke passed with eight of eight draft tokens accepted, but that sample is not a quality or representative acceptance result. The stable upstream reland remains [PR #30839](https://github.com/sgl-project/sglang/pull/30839); [PR #30992](https://github.com/sgl-project/sglang/pull/30992) is relevant only with context parallelism, and reverted [PR #29787](https://github.com/sgl-project/sglang/pull/29787) must not be ported alone. |
| Fwuff remote MTP drafting | **Operational and paired-live; representative quality pending** | A persistent fwuff layer-78 GPU+AMX service supplies K4 chains to dwagon's native TP2 verifier over pinned-host EDR. Boundary-canonicalized batched prefill reduced the fixed 240/128 run to 2.13279 seconds TTFT and 16.8208 seconds end to end, with 8.64701 generation-window and 7.60963 end-to-end tokens/s, 96/124 accepted drafts, and 4.12903 emitted tokens per verify. This is 13.14% more end-to-end throughput than serial fwuff prefill. The first 16 tokens exactly match the prior local Marlin control; the coherent full continuation remains target-verified but has no matched 128-token local oracle. The next fwuff lever is adaptive/deeper versioned drafting or target-verification scheduling, not more prefill work. |
| Shared host weights and P/D disaggregation | **KT sharing live; executable P/D runtime tested, not live** | Local KTransformers commit `f227a26` adds immutable direct file mappings with manifest/generation/lease ownership. The full model ran through this path without a checkpoint-sized anonymous expert copy, and exact NUMA-bound extent prefault is live-checked. The final SGLang source implements the separate prefill/decode process path with fail-closed identity, capacity, and lease checks. Focused tests pass, but a live process pair has not yet been launched. OffloaderV2 [PR #8034](https://github.com/sgl-project/sglang/pull/8034) was a lifecycle donor. |
| SmallEP | **End-to-end model integration implemented and tested; not live** | The sprint implements two-rank all-gather, repeated local routing, complete disjoint expert export, local weighted partials, hidden-size partial exchange, payload accounting, and model-path integration. Reference and integration tests pass. It has not run on full GLM-5.2 or the remote `fwuff` transport, so no distributed-prefill speedup is claimed. DeepEP, context parallelism, and PP=3 remain different algorithms. |

### Integration decisions

- Retain the immutable hybrid checkpoint and direct compact execution as the
  GPU-body baseline. Do not add a load-time BF16 expansion or an alternate
  mutable quantization cache.
- Require Marlin for all new compact `kv_b_proj` launches. Reject `auto`,
  Triton, missing Marlin module counts, or conflicting runtime markers before
  accepting quality or timing evidence.
- Run the representative TP2+MTP quality gate before treating the compact
  integration smoke as an operational default. Compare fixed logits,
  perplexity, and representative tasks against a matched MTP-off control.
- Once the runtime pieces are stable, benchmark the intended scheduler policy:
  TBO off for c1 and TBO on for c3. The earlier exact c2 A/B remains useful
  evidence, but it does not answer the c3 question.
- Diagnose the SLP first-chunk stall before another threshold or chunk sweep.
  Instrument AMXINT4 export, pinned-host preparation, H2D, ready/recycle-event
  waits, and GPU compute separately. Preserve the two-slot reuse-safety
  contract and fail-closed VRAM admission.
- Keep the implemented AMX dependency graph opt-in until another resident A/B
  separates its effect from normal variation. Preserve exact equality with the
  staged path and keep its tasks NUMA-local.
- Retain the measured KT-backed MTP path and construction-time target-module
  sharing plus identity-scoped loader ownership. Do not regress to loading
  layer 78's roughly 18 GiB BF16 expert body onto each 3090 or to
  post-processing the shared `lm_head` twice.
- Retain fwuff as the c1 K4 drafter rather than another target-model PP stage.
  Batch large shifted prefill transfers through page-correct draft `EXTEND`
  before pursuing transport-only overlap. Preserve native target page-shadow
  bookkeeping, exact thresholds, TP hidden coherency, disconnect rollback, and
  the first-16-token oracle.
- Retain the immutable shared mapping and exact-extent prefault as the
  authoritative allocation path. First validate TP2 coherency/performance,
  then increase the local SLP ring. Attempt TP1 P/D only afterward; keep its
  task queues and scratch private and require both leases to bind the same
  content/generation.
- Exercise integrated SmallEP only after local SLP is viable. Add `fwuff` as a
  communication-lean prefill participant, not another serial PP stage.
- Treat [HybriMoE](https://github.com/PKU-SEC-Lab/HybriMoE) and SGLang's
  experimental [Paged Experts PR #29971](https://github.com/sgl-project/sglang/pull/29971)
  as policy and buffer-design donors only. Their model formats, cache-miss
  behavior, and runtime assumptions do not match GLM-5.2 AMXINT4 on this
  cluster.
- Keep deferred experts out of the exact-performance lane: upstream
  [commit `dd4377b`](https://github.com/kvcache-ai/ktransformers/commit/dd4377b60bfb2fbfb5492ab0c7a758ab2b6cac1c)
  changes layer semantics rather than merely rescheduling exact work.

The resident baseline, persistent hybrid W8 artifact, direct compact
Marlin path, shared allocation, AMX dependency graph, KT TBO branch,
measured KT-backed MTP path, bounded SLP ring, SmallEP model integration, and
executable shared-weight P/D path now exist. The final TP2+MTP integration
smoke is complete, but the representative quality gate is not. The remaining
sequence is TP2 quality/performance validation, the c1-off/c3-on TBO
measurement, instrumented local SLP viability, a larger local SLP ring,
integrated SmallEP and communication-lean `fwuff` participation, and only then
TP1 P/D.

## Related upstream references

- [OSDI '26 paper, full HTML](https://arxiv.org/html/2606.10493)
- [KTransformers KT-Kernel parameters and NUMA guidance](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
- [KTransformers GLM-5 launch tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5-Tutorial.md)
- [KTransformers expert placement tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)
- [SGLang pipeline-parallelism design](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pipeline_parallelism.md)
- [Official Triton repository](https://github.com/triton-lang/triton)
- [SGLang absorbed-MLA `kv_b_proj` LoRA PR #25001](https://github.com/sgl-project/sglang/pull/25001)
- [SGLang LoRA-B Triton optimization PR #27087](https://github.com/sgl-project/sglang/pull/27087)
- [GLM-5.2 FP8 model card and IndexShare/MTP notes](https://huggingface.co/zai-org/GLM-5.2-FP8)
