# DeepSeek V4 Flash dynamic expert-hotspot implementation and results

Date: 2026-08-04

Status: graph-safe transaction and Oscar split-history decoder implemented;
coherent g14-p28 EP2 result qualified, autonomous controller remains opt-in

## Executive summary

DeepSeek V4 Flash can change which routed experts deserve GPU residency while
the server is running. The correct unit proved to be an individual rank-owned
routed expert, not a complete Transformer layer. The implementation now:

1. retains immutable full CPU shadows for every rank-owned target expert;
2. copies MXFP4 weights and UE8M0 scales into existing fixed-address GPU slots;
3. validates all ranks at a quiescent request boundary;
4. updates the logical/physical tables and GPU/CPU masks in place as one
   generation; and
5. rolls every rank back to the previous generation if validation or commit
   fails.

CUDA graph addresses, shapes, expert counts, and the 524,288-token Oscar INT2
cache remain unchanged across a swap. A one-slot transaction, explicit rollback,
full g13-p26 transaction, repeated semantic prompts, forced tool calls, and
captured-graph replay all passed. The best held-out result was then frozen as
the repository-owned **g14-p28** startup manifest with **fixed verify 4**. The
live controller remains opt-in: ordinary serving materializes the qualified
placement directly and does not pay for full CPU shadows or an update pause.

This is a real target-tail improvement, but it did not reach 80 or 90 token/s.
After the later CPU and Oscar-attention work, the final coherent TP2/EP2
reference is **44.690 mean / 44.735 median HTTP decode token/s**, with
**5.542 s median / 5.624 s maximum** fresh TTFT. The 34.420-token/s Oscar
baseline and 36.05-token/s pre-Oscar result below remain historical controls
for the placement and kernel decisions.

## Why dynamic residency is relevant

The starting TP2/EP2 configuration had 13 GPU-resident and 115 socket-local
CPU experts per rank and layer. Its coherent route profile placed only 27.9508%
of recorded target expert activations on the GPUs. The accepted g14-p28
placement now uses 14 GPU and 114 CPU experts per rank and layer. The remaining
routed work still passes through AVX/AMX CPU experts and the CPU/GPU merge
barrier in every MoE layer.

The detailed route capture also shows that the CPU workload is dominated by
small expert batches:

| CPU expert rows per call | Share of CPU calls |
|---:|---:|
| 1 | 73.43% |
| 2-4 | 25.48% |
| 5-6 | 1.09% |

This makes expert identity important. Moving a frequently selected expert from
CPU to GPU can eliminate many tiny CPU calls, weight reads, and scheduling
tails. A profile-specific historical selection of all 26 available union experts was
estimated to cover 39.2479% of the observed profile rather than 27.9508%,
without adding GPU slots. That estimate is workload-specific and must not be
treated as a production result; previous exact-prompt profiles have overfit.

The experiment confirmed that identity matters: the g13-p26 live transaction
reduced weighted target verification by 12.95% and improved trace committed
throughput by 12.88% against its same-process rollback. Expanding and refining
that selection to g14-p28 produced a further 2.48% target-verification decrease
and roughly 1.3% trace/HTTP throughput gain in the instrumented comparison,
while retaining the 4-GiB allocator-headroom gate. Dynamic policy remains
useful for future workload shifts, but the current production choice is a
frozen, reproducible startup ordering.

## Existing mechanisms and their limits

### Expert distribution recording

SGLang already records logical and physical expert counts through
[`ExpertDistributionRecorder`](vendor/sglang/python/sglang/srt/eplb/expert_distribution.py).
It supports bounded rolling statistics and is already used to build the static
DeepSeek V4 plans. This is sufficient as the observation substrate, although
the hotspot policy also needs per-rank CPU-wait and row-size costs.

### EPLB

SGLang's
[`EPLBManager`](vendor/sglang/python/sglang/srt/eplb/eplb_manager.py#L30)
periodically computes a new expert layout and updates expert weights. It is
primarily a GPU-rank load-balancer, not a GPU-versus-CPU residency cache.

The KTransformers hybrid shard path explicitly rejects EPLB and redundant
experts because its CPU ownership and compact physical mappings are static:
[`kt_ep_wrapper.py`](vendor/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py#L2843).

### KTransformers dynamic expert update

The wrapper contains an updater that:

- selects experts from observed routes;
- copies their weights into GPU slots;
- broadcasts the selection;
- updates masks and logical-to-GPU tables; and
- uses in-place CUDA tensor updates to preserve captured addresses.

The graph-safe mapping operation is already demonstrated in
[`_update_gpu_experts_from_batch`](vendor/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py#L4447).

The inherited path was not usable for this deployment:

- hybrid sharding rejects dynamic expert updates at construction time;
- it selects from one current batch rather than a stable historical window;
- it depends on a temporary full-GPU layer used by the prefill fallback;
- the V4 MXFP4 path explicitly skips it because the fallback copier assumes
  INT4-style weight names and layouts; and
- compact CPU storage does not automatically retain an immediately usable
  shadow of every GPU-resident expert.

The original V4 MXFP4 skip remains documented in
[`kt_ep_wrapper.py`](vendor/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py#L4153).
The new hotspot path does not re-enable that incompatible fallback. It uses a
separate V4-aware transaction with rank-owned shadows, fixed-slot copies, and
idle-boundary coordination.

## Implemented and measured result

The implementation exposes an idle-only control endpoint and a fail-closed
multi-rank transaction. Dry-run validates the entire proposed generation
without mutation. Commit synchronizes staging reuse, copies packed MXFP4 weight
bytes and scale bytes exactly, verifies ownership and slot contents, and only
then updates the fixed-address mapping tensors. Rollback restores the previous
generation on both ranks. The scale copier accepts the pre-use
`float8_e8m0` representation and the post-use `uint8` representation because
the installed grouped-MoE kernel canonicalizes those scale tensors in place;
the transaction bit-copies the payload rather than numerically converting it.

The staged qualification was:

| Generation | Change | Result |
|---:|---|---|
| 1 | one slot on each rank | committed; graph/semantic/tool pass |
| 2 | explicit rollback | restored baseline coherently |
| 3 | g13-p26, 354 rank-0 and 364 rank-1 swaps | committed; target verify -12.95% |
| startup | frozen g14-p28, 14 slots/rank/layer | accepted final placement |

The one-slot move was intentionally a correctness proof and had no measurable
speed win. The g13 transaction copied roughly 9.60 GB across both ranks in a
bounded between-request pause and improved same-process trace throughput by
12.88%. A strict exact-answer repeat and a forced-tool request passed after the
transaction and again after the g14 placement. Captured graph tests confirmed
that replay observed the new bytes while all pointers remained stable.

The final placement is persisted in
[`dsv4_flash_opencode_g14_p28_frozen_plan.json`](scripts/data/dsv4_flash_opencode_g14_p28_frozen_plan.json).
The manifest records 43 layers, two ownership ranks, 14 GPU experts per rank,
114 CPU experts per rank, the selected IDs, ownership hashes, and a placement
semantics hash. The launcher materializes it into the runtime plan and falls
back to the profile builder only when the operator explicitly changes geometry
or profile inputs.

Placement provenance is now fail-closed end to end. The launcher computes the
materialized plan's SHA-256 before model startup. Every KTransformers worker
compares the file to that admitted digest before loading and verifies that the
file did not change during load. `/server_info` publishes the loader-admitted
digest, and `benchmark_dsv4_flash_hotspot.py --expected-expert-plan` hashes an
absolute, regular, non-symlink plan and rejects any mismatch. Future
claim-eligible hotspot runs must pass the materialized startup plan explicitly:

```bash
--expected-expert-plan \
  /var/lib/exo/cache/dsv4-flash-hybrid-ep2/opencode-g14-p28-frozen.pt
```

This binding was added after the historical receipts in this document. Those
receipts remain valid semantic/timing evidence with launcher and log
provenance, but they predate the loader-admitted digest field and must not be
retroactively described as cryptographically plan-bound.

### Qualified runtime foundation

Hotspot qualification used the local 48-shard checkpoint at
`/tmp/dsv4-local-checkpoint-0731`; no NFS model path and no fwuff device were
involved. The old packed-E4M3 qualification described later in this document
has been superseded. Current serving accepts only calibrated, model-bound
Oscar INT2: C4/C128 history is 272 bytes/token, the C4 scorer is 40
bytes/token, and the protected SWA reserve remains BF16. The public
`fp8_e4m3` spelling is only SGLang's raw-byte carrier. The launcher and
campaign reject generic FP8, generic INT4, and selective-BF16 history, and
require Oscar writer, C4 scorer, sparse-attention, speculative, CUDA-graph,
artifact-hash, and admission-receipt telemetry on both ranks.

The OpenCode coherency repair is equally important to interpreting hotspot
results. Breakable prefill replay now uses live `ForwardBatch` metadata, and
the whole DSV4 attention module remains eager inside the surrounding prefill
graph so padded replay rows cannot write KV slot zero. Target verification and
DSpark decode remain captured. Repeated long semantic answers and exact forced
tool calls passed after both commit and rollback; the placement speedup is not
being measured on the earlier garbled-output path.

### Repeated-prompt qualification

Each qualification receipt used five phases: cold exact, radix-hot exact,
radix-hot near, cache-flushed warm exact, and cache-flushed warm near. This
executes the same 2,694-token long prompt three times while a controlled near
variant appears twice. The near prompt retains a 69.67% common prefix but
reported zero server cache reuse; the exact hot phase reused 2,560 tokens.
Repeating the complete receipt detects both prompt-specific overfit and
same-process page/graph warmth. Natural output trajectories varied, so paired
cycle attribution is not claimed even though each phase independently passed
natural-stop and semantic gates.

Three instrumented g14 receipts consistently beat the preceding g13 placement.
Two clean, instrumentation-free receipts then produced 36.05 pooled HTTP decode
token/s, about 4.79 s median uncached TTFT, and a qualified 0.611 s exact
radix-hit TTFT. Post-capture headroom was 4.15 GiB in the SGLang allocator and
3.73 GiB/GPU by physical idle memory.

### NVLink evidence

The hotspot campaigns also ruled out an idle second GPU or silent host-memory
fallback. A fresh P2P matrix was true in every direction. NCCL labeled the peer
edge `NVL[48.0]`, reported direct P2P, and selected `P2P/IPC` for every channel
without SHM/NET data paths. Around each real five-phase campaign, all 16
physical counters (two GPUs, four links, Tx and Rx) advanced; clean TP2 receipts
advanced the least-active counter by about 2.66 million KiB. This proves active
bidirectional NVLink-class P2P during the model window. It does not identify
every byte with a particular collective or support a derived application
bandwidth claim.

## Implemented scope and future policy

### Implemented unit: expert identity within a fixed layer/rank budget

The transaction keeps the configured GPU-slot count fixed in every target
layer on each rank and changes only which logical experts occupy those slots.
The accepted plan uses 14 slots; the one-slot and g13 experiments used their
launch-time fixed budgets. This preserves:

- GPU allocation size;
- grouped-MoE kernel geometry;
- CUDA-graph tensor addresses and shapes;
- TP2/EP2 topology;
- rank-local CPU pools; and
- the 524,288-token cache and graph-memory budget.

Restrict swaps to experts already owned by the same EP rank. Moving CPU
ownership between ranks is a separate EPLB problem and would add weight
transport, dispatcher, recovery, and rank-failure complexity.

### Later unit: variable slots per layer

A later allocator could give more slots to layers where GPU residency saves
more wall time and fewer slots to cold layers. This requires a stable paged
weight arena or fixed maximum slot arrays with indirection. Ordinary per-layer
tensor resizing would invalidate graph assumptions and is not suitable.

### Complete layers remain out of scope

Moving a whole layer changes dense and attention weights, KV ownership,
collectives, pipeline boundaries, and captured graphs. Whole-layer movement
should remain a coarse drain, repartition, recapture, and requalification
operation. Routed experts are independently addressable and provide nearly all
of the desired adaptability at much lower risk.

## Hotspot policy

Permanent cumulative frequency is not sufficient. It becomes anchored to old
traffic and responds poorly when the workload changes. Use both recent and
longer-term history, similar to Window-TinyLFU or a decayed LFU cache.

For every layer, rank, and logical expert, maintain:

- exponentially decayed call count;
- row histogram for `M=1..6`;
- estimated CPU bytes and CPU execution time;
- estimated GPU execution time;
- contribution to per-layer CPU wait and rank tail;
- last promotion/demotion epoch; and
- current residency generation.

A useful initial score is:

```text
predicted_saving[layer, expert] =
    EMA(sum_M calls[M] * (cpu_cost[layer, expert, M]
                         - gpu_cost[layer, expert, M]))
  + predicted_rank_tail_reduction
  - amortized_promotion_cost
```

The selected hot set maximizes predicted cycle-time reduction, not route count.
This distinction matters because:

- a frequent `M=1` CPU expert can be expensive because of repeated weight
  streaming and task overhead;
- some tiny GPU expert shapes are also inefficient;
- removing work from the faster CPU rank may not reduce the layer barrier; and
- the slower CPU/GPU branch controls the merge point.

### Stability controls

The policy should include:

- a minimum observation window, initially 512-2,048 verification cycles;
- promotion hysteresis over the current victim's score;
- minimum residency and cooldown epochs;
- a maximum of one or two swaps per layer and epoch;
- a churn budget in bytes and milliseconds;
- a minimum predicted amortization horizon; and
- a fail-safe static-plan mode.

The offline profile remains the cold-start plan. Online statistics refine it
rather than beginning from an arbitrary ordering.

## MXFP4 weight ownership and movement

The current hybrid CPU wrapper stores a compact shard, generally the complement
of the GPU mask. Arbitrary eviction requires the evicted GPU expert to become a
valid CPU expert immediately. There are two viable designs.

### Implemented: immutable CPU shadow for every target expert

Keep all target MXFP4 expert weights addressable through immutable mapped
checkpoint storage on their owning NUMA node. The active CPU mask determines
which experts are dispatched to CPU, while GPU slots contain promoted copies.

Advantages:

- demotion needs no checkpoint reload;
- rollback is immediate;
- source weights have stable identity and hashes; and
- swapping GPU residency does not change host ownership.

The current implementation retains full rank-owned CPU shadows only when the
live controller is enabled. The frozen production plan leaves the controller
off, avoiding this host-memory cost. A later always-on policy should still
prefer direct file mappings or the persistent KTransformers representation
over a second anonymous checkpoint-sized copy.

### Alternative: on-demand mapped reload

Reload a demoted expert from the mapped checkpoint before committing a swap.
This uses less persistent host-side indexing but increases promotion latency and
failure surface. A partially loaded expert must never become routable.

### GPU representation

The direct V4 MXFP4 copier targets an existing GPU expert slot. It understands
the `DeepSeekMxfp4MoEMethod` packed tensors and scales rather than falling
through the inherited INT4 copier. It does not load a temporary 256-expert GPU
layer.

An expert is approximately 12.75 MiB of packed weights per layer in the current
geometry. Replacing one slot in every one of 43 layers is therefore roughly
548 MiB of GPU writes per rank, before formatting overhead. This is acceptable
only when amortized over a substantial serving window and kept off the request
critical path where possible.

## CUDA-graph-safe update protocol

CUDA graphs capture addresses, not the semantic identity of bytes stored at
those addresses. Expert data and mapping values may therefore change if all
buffers retain their original address, shape, stride, and dtype and no replay
can observe a partial update.

### Implemented protocol: quiescent in-place swap

With one running request, the implemented safe protocol is:

1. Observe routes during the request without modifying placement.
2. At request completion, stop admitting a new batch briefly.
3. Confirm no target-verify, draft, or prefill graph is in flight.
4. Ensure the victim has a valid CPU shadow.
5. Copy and format the promoted MXFP4 expert into the victim's existing GPU
   slot.
6. Verify the copied slot with a checksum or deterministic probe.
7. Update the CPU ownership mask, GPU mask, logical-to-slot mapping, and reverse
   mapping in place.
8. All-gather the new generation and require agreement across ranks.
9. Resume admission.

This needs no additional GPU slot. Its cost is a bounded between-request pause,
which must be measured against the predicted savings.

### Later implementation: background staging

True background promotion requires an inactive slot or stable paged arena so a
request never executes partially copied weights. One extra expert slot in every
layer costs roughly 0.535 GiB per GPU. That may fit the current headroom, but it
must preserve the hard graph-memory reserve and is not necessary for the first
proof.

### Atomic generation contract

Every placement has a monotonic generation containing:

- logical-to-GPU slot tables;
- GPU masks;
- CPU-active masks and compact mappings;
- per-rank ownership hashes; and
- hashes of promoted GPU slot contents.

All ranks must commit the same generation at a safe boundary. Failure before
commit leaves the previous generation active. Failure after any rank commits
must fail closed, drain serving, and restore the previous plan rather than
allow ranks to route different expert identities.

## Numerical and serving correctness

CPU and GPU expert implementations consume the same quantized weights but can
accumulate differently. A placement change can therefore perturb target logits
near a greedy decision boundary and change speculative acceptance or generated
tokens.

The production policy should freeze placement for the lifetime of every active
request. With future concurrency, an epoch transition must drain all requests
unless the runtime retains both old and new generations, which is unnecessary
for the initial implementation.

Admission tests must cover:

- CPU-only, GPU-only, and mixed reconstruction for each promoted expert;
- eager versus captured target verification;
- all six verification rows and every graph tier;
- prefill, decode, DSpark verification, and cache reuse;
- rank agreement and forced rollback;
- varied semantic OpenCode requests;
- exact forced tool calls;
- long-context retrieval at the admitted context tiers; and
- speculative acceptance and cycle-time distributions before and after swaps.

## Performance instrumentation

The cache policy needs costs that correspond to the real critical path. Record:

- GPU hit rate by calls and routed rows;
- CPU calls avoided by row size;
- estimated expert-weight bytes avoided;
- CPU submit and wait time by layer and rank;
- GPU expert time by layer;
- TP/EP collective time;
- per-layer rank skew and merge-tail time;
- promotion bytes, copy/format time, and pause time;
- hot-set churn and residency duration;
- complete speculative-cycle time;
- committed tokens per cycle; and
- semantic and speculative-acceptance changes.

The policy should use these measurements to update its cost table. It should
not infer savings solely from GPU utilization or global route frequency.

## Implementation status and remaining sequence

### Stage 0: replay-only solver -- complete

- Feed existing route captures into an online EMA/Window-TinyLFU simulator.
- Compare its selected hot sets with the static g13 plan.
- Estimate CPU calls, rows, bytes, rank tails, and swap traffic.
- Evaluate on held-out prompts to reject profile overfitting.

This stage produced the g13-p26 and g14-p28 candidates. Held-out repeated
prompts, rather than the capture prompt alone, selected the winner.

### Stage 1: manual graph-safe MXFP4 slot swap -- complete

- Retain or map a CPU shadow for both the promoted and evicted experts.
- Add a V4 MXFP4 copy/format routine targeting one existing GPU slot.
- Update mapping tensors and KT CPU masks in place at a quiescent boundary.
- Prove rollback, graph replay, and exact ownership on both ranks.

A one-slot commit and explicit rollback passed before the full-layer plan was
attempted.

### Stage 2: target-only online hot set -- transaction complete, policy manual

- Start from the frozen target g14 plan.
- Observe for a fixed epoch.
- Permit bounded rank-local swaps between requests.
- Keep all three draft stages static; draft coverage is already approximately
  98.44% and draft time is a small part of the speculative cycle.

### Stage 3: autonomous cost-aware controller -- pending

- Replace route frequency with measured CPU/GPU and barrier savings.
- Add hysteresis, churn budgets, and automatic rollback on regression.
- Persist only aggregate statistics and validated hot-set metadata, not prompt
  contents.

### Stage 4: optional inter-layer budget and background promotion -- pending

- Introduce a stable paged GPU expert arena or reserved staging slots.
- Allocate slots across layers according to marginal cycle savings.
- Copy candidates asynchronously and commit completed generations atomically.

This stage is optional and should follow proof that fixed-width identity swaps
produce material gains.

## Admission gates

A dynamic policy should replace the static launcher default only if it meets all
of the following over representative held-out workloads:

- zero incomplete or mismatched placement generations;
- exact structural and semantic tool-call success;
- no degeneration, prompt copying, or stream-protocol failures;
- no material speculative-acceptance regression;
- at least 4 GiB of post-graph SGLang allocator headroom;
- bounded update pause and churn;
- at least 5% lower median target-verification time;
- no material TTFT regression; and
- repeatable improvement across multiple prompt classes rather than one hot
  trace.

The first useful milestone is not fully autonomous migration. It is one
coherent, CUDA-graph-safe, rank-synchronized MXFP4 expert swap that lowers
measured target verification while preserving the 524K context and current
semantic gates.

That milestone is complete. The g13 transaction exceeded the 5% target-verify
gate, preserved speculative acceptance within the 2% allowance, stayed below
7 s TTFT on the 2,694-token shape, passed semantic/tool checks, and rolled back
cleanly. The final g14 placement retained 4.15 GiB of allocator headroom and
won the repeated held-out sweep. Fully autonomous epoch selection remains
below the admission line because its stability across broader prompt classes
and long contexts has not yet been demonstrated.

## Expected value and limitations

Dynamic hot sets remove launch-time guesswork and can track workload phases at
the same VRAM footprint. The observed profile suggests a credible improvement
in GPU route coverage, and avoiding dominant `M=1` CPU calls may reduce the
per-layer tail more than the raw activation percentage implies.

It was not independently a path from approximately 30 to 80-90 token/s. The GPU
expert kernel is also inefficient at tiny shapes, TP/EP collectives remain, and
speculative acceptance is a separate multiplier. Dynamic residency should be
developed alongside:

- grouped or persistent small-row MXFP4 GPU execution;
- a faster grouped CPU `M=1-4` path;
- measured NCCL/NVLink qualification; and
- improved, semantically qualified speculative acceptance.

The accompanying small-row work moved SM86 routing metadata to a graph-safe GPU
kernel (roughly 13-21 microseconds in isolation). A grouped CPU M1-4 candidate
measured only 0.94-0.96x the installed path and was rejected. Custom all-reduce
v2 remained enabled, and the NVLink checks above prove that the second 3090 was
active; communication fallback is not the explanation for the remaining
plateau.

The later CPU-tail pass found a materially stronger path. Reusing the pinned
`TaskQueue` thread as logical worker zero removes the standalone NUMA
distributor; a same-binary checkpoint screen selected 56 physical workers with
a 1,000-us spin window and measured a 2.2203x fixed-4 route-weighted gain.
Seventy-two workers were slower and are rejected. On top of that winner, the
OCP-safe MXFP4 LUT scale fold passed two alternating checkpoint-backed screens:
weighted latency fell 20.84% and 22.91%, every high-weight M1--M4 arm improved,
all outputs were bit-identical, and all 72 test buffers reported zero unsafe,
NaN, rejected, fallback, or invalid-mode counts. These remain offline kernel
results until the complete Oscar-only model campaign proves end-to-end decode,
TTFT, coherency, graphs, memory, and NVLink together.

At the accepted g14 placement, the instrumented mean is 2.3029 committed tokens
per roughly 65.26 ms cycle, or 35.29 trace token/s; clean HTTP receipts pool to
36.05 token/s. Reaching 80 at that acceptance requires a 28.79 ms cycle, and 90
requires 25.59 ms. Weighted target verification alone is 58.48 ms. The next
large gain must therefore remove target-side CPU expert streaming and rank-tail
latency or fuse substantially more verification work. Better acceptance helps,
but cannot repair a verification stage already larger than the entire target
cycle budget.

The principal future advantage of the live controller is that kernel and
memory improvements can benefit experts important to the current workload,
rather than experts that happened to be hot in one offline trace. Its frozen
manifest already captures that benefit for the qualified OpenCode workload
without introducing online churn into production serving.

## Post-warmup repeated-prompt result

The final combined CPU candidate was run through the five-phase exact/near
hotspot protocol after its independent screen and confirmation. The receipt is
`/tmp/dsv4-oscar-int2/kt-combined-nondeep-v2-hotspot.json`. All semantic phases
passed and averaged 39.839 token/s. Exact radix reuse was real (2,560 cached
tokens and 4.953 s TTFT saved), but OS page-cache locality was not eligible for
paired attribution because natural-stop output trajectories changed. This is
the expected way to use repeated long prompts here: flushed exact and near
controls distinguish radix reuse, compilation, and page warmth without
pretending that different greedy trajectories have identical expert routes.

The run also closes two earlier bottleneck hypotheses. First, all 16 local
NVLink RX/TX counters advanced by approximately 2.619 GiB, proving both GPUs
and all four links were carrying payload. Second, 178 sampled small-token
hybrid observations averaged only 0.045 ms of CPU wait. The remaining
54--56 ms target-verification stage is therefore not explained by a dead link
or a blocking CPU queue. Attention and resident-GPU small-batch work are the
next critical-path candidates.

For Oscar attention specifically, H64 with an 8-head program block gives only
8 CTAs for single-token decode and 40 for the observed five-row verification
shape. The next candidate retains the immutable Oscar INT2 cache and uses a
two-stage split-history attention kernel: several CTAs produce sink-free FP32
online-softmax partials, followed by one ordered combine which owns the sink.
Its workspace is fixed at 4,210,688 bytes, independent of the 524K context,
and prefill remains on the monolithic path. Qualification measures the C4 and
C128 layer families independently before enabling the feature in a model run.

## Oscar split-history implementation and final result

That candidate is now complete. The runtime uses a fixed-address,
CUDA-graph-safe two-stage Oscar decoder with the split map
`{1:16, 2:16, 3:8, 4:4, 5:4, 6:4, 7:4, 8:2}`. It never changes the physical
cache: C4 and C128 remain calibrated asymmetric Oscar INT2, the C4 scorer
remains Oscar INT2, and protected recent SWA remains BF16. The SGLang
`fp8_e4m3` option is only the external byte-carrier spelling. Generic FP8,
generic INT4, and selective-C128 BF16 are disabled by the production launcher.

The graph microbenchmark at
`/tmp/dsv4-oscar-int2/oscar-split-decode-bench-v1.json` passed every parity
gate. One-row C4-512 improved 4.655x and full C128-4096 improved 9.174x;
five-row verification improved 2.490x and 2.664x respectively. The weighted
41-layer kernel estimate improved 3.342x for decode and 2.239x for the observed
five-row verification shape.

The independent model confirmation at
`/tmp/dsv4-oscar-int2/split-history-ep2-confirm-hotspot.json` produced
**44.690 mean / 44.735 median token/s**, **5.542 s median / 5.624 s maximum**
fresh TTFT, **47.635 ms** mean target verification, and **2.402 committed
tokens/cycle**. Every natural-stop output passed the semantic contract, the
separate 42K OpenCode semantic/tool receipt was coherent, and all 16 NVLink
counters moved. This promotes split history over the 39.196-token/s combined
CPU artifact while preserving its inline-dispatch/N128-LUT work.

The result also sharpens the bottleneck diagnosis. The attention change removed
about 6.87 ms from mean target verification and raised model decode 14.02%, but
the remaining 47.6 ms verification stage still exceeds the roughly 30 ms full
cycle required for 80 token/s at current acceptance. NVLink traffic is active,
CPU queue wait is about 0.045 ms, and more static expert residency already
regressed end-to-end acceptance. Future work must reduce resident target GPU
small-batch work or change the verification/acceptance trade, not merely add
cache capacity or CPU workers.

Exactly two PP2 concurrency launches then tested transferability. Swapping the
layer partition from 21/22 to 22/21 improved aggregate native decode from
27.844 to 28.606 token/s, but the second semantic lane reproducibly omitted
`cache isolation` in both launch configurations. PP2 is therefore rejected;
the hotspot result promoted here is the coherent EP2 configuration only.
