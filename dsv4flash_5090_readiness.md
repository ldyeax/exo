# DeepSeek V4 Flash: RTX 5090 readiness track

Status: design and fail-closed preflight schema v4 are ready; no RTX 5090 is
installed or qualified. The base OpenCode launcher keeps its two-RTX-3090
defaults (`8.6`, `8.6`, and NVLink), while a proposed mixed launcher passes the
v4 advisory source gates only when the static parser finds FlashInfer
`8.6 12.0`, Torch `8.6;12.0`, and NCCL auto (an explicitly empty DSV4 P2P
level). This is not shell evaluation or runtime proof. With the current
runtime, the executable PP3 admission target is target-only:
speculative decoding and overlap scheduling are disabled, while target decode
CUDA graphs remain enabled. PP-aware DSpark is future runtime work, not part of
the initial 5090 admission path. The mixed launcher is additionally blocked on
a native SM120 implementation of the admitted Oscar INT2 cache contract. This
is a KV-cache blocker, not a model-weight-format blocker: the branch already
contains an SM120 FlashInfer CUTLASS MXFP8-by-MXFP4 expert path, while the
current Oscar writer, C4 scorer, and sparse-attention kernels fail closed on
anything other than exact SM86. Generic FP8 is not an allowed 5090 fallback.
See [the OpenCode optimization log](dsv4flash_opencode.md) for the measured
two-RTX-3090 baseline and PP2 concurrency campaign.

## Decision summary

Do not add the 5090 to the current TP2/EP2 world merely by changing
`CUDA_VISIBLE_DEVICES`. The RTX 5090 is compute capability 12.0, has 32 GB of
GDDR7 and PCIe Gen 5, but has no NVLink. NVIDIA specifies 575 W total graphics
power and a 1000 W required system power for the reference configuration. The
Founders Edition installation guide also calls for three free expansion slots
and cable clearance. These are admission constraints, not tuning suggestions.
[NVIDIA RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

The safest integration sequence is:

1. Keep the validated 3090+3090 TP2/EP2 service unchanged while qualifying the
   5090, its power/thermal envelope, its SM120 kernels, and its PCIe paths.
2. Treat a three-GPU PP3 topology as the first all-card candidate, not TP3 or
   EP3. Use TP1/EP1 and start by measuring the 5090 as the last target-model
   stage, which owns the final normalization/output path and can absorb a
   deliberately smaller layer slice. This is a target-only placement seed, not
   a claim about future draft-model ownership.
3. Promote PP3 only if it beats the unchanged TP2 baseline while preserving the
   exact 524K Oscar INT2 memory, graph, coherency, and quality gates. A mixed
   5090+3090 TP2 run is a secondary transport experiment, not the default
   topology.

## Facts established now

### Host topology

Static PCI and NUMA inspection on 2026-08-02 found:

| Device/slot | PCI address | NUMA | Maximum capability | Current negotiated link |
|---|---:|---:|---:|---:|
| RTX 3090, slot 1 | `0000:16:00.0` | 0 | device PCIe 4.0 x16; slot PCIe 5.0 capable | inventory only; not a 5090 gate |
| RTX 3090, slot 3 | `0000:d8:00.0` | 1 | device PCIe 4.0 x16; slot PCIe 5.0 capable | inventory only; not a 5090 gate |
| Empty slot 8 path | `0000:4a:00.0` | 0 | slot PCIe 5.0 x16 capable | unavailable until card installation |

The two sockets each expose 56 physical AMX-capable cores. Slot 8 is the
electrically attractive 5090 candidate because its upstream root port is local
to NUMA node 0 and advertises 32 GT/s. Physical three-slot clearance, power
connectors, PSU rail capacity, and cooling have not been proven. Do not install
the card until those manual checks pass.

Preflight v4 does not conflate slot capability with the installed card's live
link. It records `max_link_width`/`max_link_speed_gt_s` and
`current_link_width`/`current_link_speed_gt_s` separately. The automatic
`rtx5090-pcie-max-link` gate requires maximum x16 at 32 GT/s, and the independent
`rtx5090-pcie-current-link` gate requires the installed card to be negotiated
at x16 and 32 GT/s during admission. Because the checker deliberately launches
no CUDA work, collect the current-link inventory during or immediately after an
active bandwidth workload. An idle power-managed link may downtrain and is not
by itself a hardware rejection; the retained active-load receipt is the
authority. A card that remains downtrained under load blocks admission.

Current DSV4 runtime facts are:

- PyTorch `2.9.1+cu128`, reporting CUDA 12.8;
- FlashInfer `0.6.9` and Triton `3.5.1`;
- `/opt/cuda` is CUDA toolkit 13.1;
- the base launcher retains `FLASHINFER_CUDA_ARCH_LIST=8.6`,
  `TORCH_CUDA_ARCH_LIST=8.6`, and `NCCL_P2P_LEVEL=NVL` as its unchanged
  two-3090 defaults;
- those settings have validated `DSV4_FLASHINFER_CUDA_ARCH_LIST`,
  `DSV4_TORCH_CUDA_ARCH_LIST`, and `DSV4_NCCL_P2P_LEVEL` overrides. An
  explicitly empty NCCL override unsets `NCCL_P2P_LEVEL` and lets NCCL choose
  from the detected PCIe topology; and
- preflight v4 statically parses the launcher source and requires the defaults
  it can recognize, not comments or mere variable-name mentions, to be
  `8.6 12.0`, `8.6;12.0`, and empty/NCCL-auto respectively. This check is
  advisory: it does not execute shell control flow and can be fooled by an
  unreachable assignment or a later dynamic assignment.

The base defaults alone cannot produce an SM120 FlashInfer/Torch JIT binary,
and the base NCCL setting only permits the NVLink path while NVIDIA states that
the 5090 has no NVLink. A future mixed-generation launcher must therefore make
both architectures and NCCL auto-detection its intended defaults; passing
overrides manually to the current two-GPU launcher is not a qualified 5090
entrypoint. NVIDIA lists the RTX 5090 and RTX 3090 as compute capabilities 12.0
and 8.6 respectively.
[NVIDIA CUDA GPU table](https://developer.nvidia.com/cuda/gpus)

CUDA 12.8 is the first toolkit generation with Blackwell support. This local
FlashInfer version further requires CUDA 12.9 or newer to emit its SM120 family
target. NVIDIA recommends native cubins or retained PTX and documents
`CUDA_FORCE_PTX_JIT=1` as the compatibility test for binaries that rely on PTX.
[NVIDIA Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.9.0/blackwell-compatibility-guide/index.html)

Do not add SM120 packages to the hash-frozen EP2 runtime. The current Python
search path can expose distributions from more than one content-addressed
runtime, so package metadata alone is not adequate provenance. Build a new,
isolated mixed-SM86/SM120 runtime and record `sys.executable`, the complete
`sys.path`, each imported module's resolved `__file__`, distribution
path/version, Python ABI, driver version, absolute `nvcc` path/version,
SGLang/KTransformers source revisions and tree hashes, and hashes of loaded
native extensions. Admission must reject cross-runtime imports, duplicate
providers, or a module origin outside the new runtime and pinned source trees.

### Why TP3 and EP3 are rejected

This checkpoint has hidden size 4096, 64 attention heads, and 256 routed
experts. Three does not divide 4096 or 256. The current launcher also accepts
only TP1/TP2 and EP1/EP2, and SGLang requires EP to divide TP in this path.
Making a three-way tensor or expert split work would require padding or a new
uneven collective implementation, with a collective on every applicable
layer over a non-NVLink path. That is unnecessary implementation risk.

### Why PP3 is only a candidate

The local DeepSeek V4 target model supports pipeline layer partitioning and
PP-local DSV4 KV slices. The current SGLang server contract nevertheless has a
hard `pp_size > 1` guard: overlap scheduling must be disabled and
`speculative_algorithm` must be `None`. Therefore today's executable PP3 path
is target-only. Target decode CUDA graphs may remain enabled, but there are no
DSpark proposal/verification graphs or acceptance metrics in this path.

Removing only that guard would not make speculative PP correct. The current
DSpark worker assumes target embeddings and the LM head are available where it
runs, while target PP deliberately splits those endpoints across ranks. A
future PP-aware speculative design must define draft ownership, embedding and
LM-head access, proposal transport, verification coordination, and graph
capture across stages. It must not assume that a complete draft model belongs
on the last stage before those mechanics are implemented and measured.

The current branch can select a socket-local KTransformers pool by PP rank for
a uniform-width TP1 pipeline, closing the earlier two-stage NUMA-selection
gap. That is insufficient when two PP processes share NUMA node 0: the native
worker pool and task queue currently select cores from the start of a NUMA
node, so the two processes can collide even if both name the correct node. PP3
is still not launcher-ready: it needs per-rank physical core lists (including a
distinct task-queue core), an asymmetric GPU-expert budget while
`--kt-num-gpu-experts` remains scalar, and qualified mixed-SM86/SM120 compiler
and NCCL paths. These gaps must be fixed and unit-tested before a PP3 launcher
is added.

Remote fwuff execution is outside the current scope. The 5090 admission plan
must stand on this host's two CPU sockets and three local GPUs; it may not
depend on fwuff CPU offload, networking, or hidden remote state. Future remote
offload would require a new explicit authorization and a separate campaign.

## Initial PP3 proposal

The initial physical/rank order and placement seed should be:

| PP rank | GPU / locality | Half-open target layers | Initial cache layout | Initial physical CPU set |
|---:|---|---:|---|---|
| 0 | node-1 RTX 3090 (`d8:00.0`) | `[0,18)` = 18 | layers 0/1 BF16, then 8 C4 + 8 C128 | `56-111` |
| 1 | node-0 RTX 3090 (`16:00.0`) | `[18,35)` = 17 | 9 C4 + 8 C128 | `0-39` |
| 2 | node-0 RTX 5090 (`4a:00.0`) | `[35,43)` = 8 | 4 C4 + 4 C128; final norm/head owner | `40-55` |

The CPU sets are a commissioning seed, not a supported current launcher
configuration. They require new per-rank core-list/offset plumbing and runtime
telemetry proving disjoint physical cores, no SMT siblings, the intended NUMA
node, and a distinct task-queue CPU. Whole-process `numactl` binding alone does
not prove that the native worker threads stay disjoint.

This creates one cross-socket transition, followed by a same-socket PCIe
transition into the 5090. Start with a conservative `18,17,8` target-layer
partition. It is a measurement seed, not an accepted balance: tune the split
from per-stage target-forward latency and the final normalization/output cost
on rank 2. Revisit both partition and rank order if a PP-aware speculative path
is implemented later.

The first functional launch must retain the OpenCode constraints. Oscar
asymmetric INT2 is **KV-only**: it compresses target-layer historical KV and
C4 scorer rows. Routed-expert and other model weights remain the checkpoint's
MXFP4 representation (`--kt-method MXFP4`); target layers 0/1 and the protected
recent SWA region remain BF16. The `fp8_e4m3` string below is only SGLang's
external raw-byte carrier spelling and must resolve to the admitted Oscar
layouts in every stage-local pool. INT2 model-weight conversion, generic FP8 KV
fallback, BF16 historical fallback, identity rotation, or a missing admitted
artifact is a blocker.

```text
context length             524288
maximum total tokens       524288
KV carrier / physical      fp8_e4m3 / Oscar INT2 history + protected BF16 SWA
SWA full-token ratio       0.0048828125 (2560 slots at 524288)
chunked-prefill size       1024
target decode CUDA graph   enabled, batch size 1
speculative decoding       disabled (required when PP size is greater than 1)
overlap scheduling         disabled (required when PP size is greater than 1)
DSpark/speculative graphs  not applicable in the target-only admission path
prefill CUDA graph         disabled
maximum running requests   1
ragged verification        not applicable while speculative decoding is off
DSpark block size          not applicable while speculative decoding is off
multi-stream MoE overlap   disabled until requalified
```

The 2,560-slot reserve is the OpenCode readiness default. It clears the
2,304-slot admission floor for a 1,024-token chunk, a 128-token sliding window,
and one 256-token page. A 2,048-token chunk would require 4,352 slots and does
not fit this reserve; the former 1,024-slot reserve clamps chunked prefill to
512 tokens. Because the larger reserve consumes more memory, do not carry
forward the old headroom number without a fresh measurement.

Use a new architecture-specific cache root, for example
`/var/lib/exo/cache/dsv4-flash-opencode-pp3-sm86-sm120-v1`. Never reuse the
accepted SM86 Triton, FlashInfer, TorchInductor, graph, or model cache. Within
the new root, separate JIT and graph artifacts by PP rank, GPU UUID, compute
capability, compiler/package fingerprint, and source-tree hash; rank 0/1 and
rank 2 must not race on one generic cache key. The isolated mixed process
environment must compile both targets:

```text
FLASHINFER_CUDA_ARCH_LIST="8.6 12.0"
TORCH_CUDA_ARCH_LIST="8.6;12.0"
```

The pinned PyTorch compiler helper was statically checked and emits separate
`sm_86` and `sm_120` cubins for the latter value. FlashInfer normalizes 12.0 to
the SM120 family target when the CUDA compiler is at least 12.9.

Do not set `NCCL_P2P_LEVEL=NVL` for a group containing the 5090. Begin with the
NCCL topology default, capture `NCCL_DEBUG=INFO`, and accept a rank order only
after P2P correctness and bandwidth tests. NVIDIA recommends
`nvidia-smi topo -p2p p` for the capability matrix and `nvbandwidth` for actual
bandwidth; P2P may be unavailable or poor because of PCIe topology, ACS, IOMMU,
or the driver even when the nominal link is fast.
[NVIDIA NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)

Test all three GPU pairs, not only the two PP edges, and retain NCCL
`INIT,GRAPH,COLL` logs plus CUDA peer-access and bandwidth/latency receipts.
Peer access involving the 5090 is not assumed: NCCL SHM/host staging can be a
correct commissioning result if the product gates still pass, but its transport
must be explicit. `NCCL_P2P_DISABLE=1` is a diagnostic A/B only, never a hidden
production default. A mixed 3090+5090 TP2 test is transport diagnostics only;
its per-layer collectives and asymmetric memory make it a poor first placement
when direct P2P is unavailable.

## Memory policy

The qualified two-3090 Oscar TP2/EP2 baseline physically stores 272-byte
compressed shared-history rows, 40-byte C4 scorer rows, and 1,024-byte protected
BF16 SWA rows. It retains **4,619 MiB of physical post-graph free memory on each
GPU** while fitting the exact 524K pool, the 2,560-slot SWA reserve
(`0.0048828125`), and resident g14 target/draft placements. The older 3.73,
4.15, 4.83, and 5.57 GiB figures describe pre-Oscar or obsolete-reserve builds
and are historical evidence only. PP changes target-layer and draft-model
residency, so the 4,619 MiB result is not a PP3 promise.
Re-measure every rank before using any of the 5090's additional 8 GB:

- leave the accepted EP2 launcher chain, runtime, cache root, frozen plan,
  Oscar artifact/admission files, and native overlay byte-for-byte unchanged;
  materialize a new content-addressed PP3/EP1 plan from the exact union of the
  frozen two-rank placement and retain the original semantic/provenance hashes;
- start every owned layer at transferred GPU-expert width 28. The final 5090
  stage also owns normalization/output weights and graph pools, so its extra
  memory is not assumed to be free capacity;

- initial PP3 qualification must record and justify its free-memory floor on
  every rank after target decode graph capture; do not inherit either TP2
  headroom result or claim a PP3 floor until the proposed configuration measures
  it;
- record free, allocated, reserved, and peak memory separately on every rank
  before model load, after weights, after KV allocation, after graph capture,
  and after a 524K admission probe;
- do not increase the 5090 GPU-expert count until those measurements exist;
- increase GPU experts in one-expert-per-layer steps and keep at least 4 GiB
  hard headroom after the worst accepted run;
- allocate stable, rank-local Oscar workspaces before graph capture and record
  memory after each of three exact-context repetitions; no graph pool or JIT
  artifact may be shared across CUDA contexts or architectures;
- use runtime-reported pool bytes rather than extrapolating from the 3090,
  because PP changes the number of target KV layers per rank. The initial PP3
  admission path has no resident draft model.

Oscar cross-architecture parity remains a quality blocker. A future native
SM120 consumer must use the same model-bound rotations, per-row clipping,
adjacent-four unsigned INT2 packing, exact RoPE tail, affine scale/zero fields,
and protected-SWA ownership as SM86. It must pass byte-layout, decode, C4 top-k,
prefill, verification, graph-replay, and held-out parity tests against the
admitted artifact. No generic FP8 path is accepted merely because allocation
succeeds.

An input containing all 524,288 tokens leaves no token available for generation
when `max_total_tokens` is exactly 524,288. Use a separate exact allocator probe
or a declared 524,287-input-plus-one-output end-to-end case; do not silently
raise the maximum or call an allocation-only probe a quality result.

## Required implementation before PP3

1. The base launcher's FlashInfer/Torch architecture lists and NCCL topology
   policy are validated overrides, with the original SM86/NVL behavior
   preserved by default and covered by tests. A proposed PP3 wrapper must make
   `8.6 12.0`, `8.6;12.0`, and an explicitly empty NCCL override its intended
   defaults. Preflight v4's static parser rejects comments, variable mentions,
   dynamic forms it cannot parse, SM86-only defaults, and an NVL default, but a
   source-check pass is advisory rather than runtime evidence. Do not change
   the base launcher's defaults or the accepted EP2/PP2 entrypoints.
2. Add a separate mixed-GPU launcher only after the remaining contracts below
   exist, and capture the detected transport in every benchmark receipt. The
   obsolete `DSV4_RTX5090_PP3_CONTRACT_V1=1` marker is neither necessary nor
   sufficient. Before hardware admission, replace the advisory source parse
   with a side-effect-free `--print-contract` mode or declarative manifest that
   reports the same contract the launcher will actually execute. It must include
   GPU UUID/PCI identity, half-open layer ranges and counts, Oscar physical mode
   and artifact/admission/provenance hashes, exact context/reserve/chunk/graph
   settings, per-rank expert-plan width/hash, physical CPU lists, kernel backend,
   cache paths, and the prohibition on remote endpoints.
3. Complete PP-rank-aware KTransformers configuration. Retain the tested
   PP-rank NUMA selection for uniform TP1 pipelines, then add per-rank GPU
   expert width, CPUInfer thread count, explicit physical worker-core lists,
   a task-queue core, and shard-plan validation. Reject list lengths that do not
   exactly match PP size, overlapping core sets, SMT siblings, wrong-NUMA cores,
   or more than 56 physical cores on either socket. Runtime telemetry must prove
   the effective affinity, not merely echo requested values.
4. Extend the hybrid plan/runtime contract so the global layer row may have a
   different GPU width on the 5090-owned layers. The current scalar
   `--kt-num-gpu-experts` equality check rejects this correctly.
5. Allocate CPU cores without oversubscription. There are three PP processes
   but only two CPU sockets. Prefer more resident target experts on rank 2; do
   not silently give rank 1 and rank 2 the same 56-core AMX pool.
6. Reject every remote-expert plan and endpoint in the initial PP3 launcher.
   All CPU offload must remain rank-local and NUMA-bound on this host.
7. Keep all ranks on the portable Triton MXFP4 weight path for the first
   functional PP3 launch so one global backend cannot select conflicting
   per-rank behavior. Do not describe SM120 MXFP4 weight execution as missing:
   the branch already auto-selects a FlashInfer CUTLASS MXFP8-by-MXFP4 method on
   SM120, separately from the SM100-only TRT-LLM FP4 path. Qualify that native
   SM120 path with shipped-binary, numerical, memory, and performance A/B
   receipts before allowing rank-aware backend selection.
8. Port the Oscar C4/C128 writers, once-per-query C4 scorer, mixed protected-SWA
   plus compressed-history attention, and absorbed output-rotation contract to
   SM120. Retain architecture-neutral byte storage, but create separate
   content-addressed SM120 kernel caches. Dispatch on each tensor's exact device
   capability after rank/device binding: `(8,6)` selects the frozen SM86 path,
   `(12,0)` selects the new SM120 path, and every unknown capability fails
   closed. Admission must reject a mixed stage if any target layer reports
   generic FP8, BF16 history, identity rotation, or a missing artifact hash.
9. Make the first PP3 launcher explicitly target-only: set PP3, disable overlap
   scheduling, disable speculative decoding, keep target decode graphs enabled,
   and fail preflight if any speculative argument is present.
10. Add that launcher only after steps 1-9 have unit tests and the read-only
   preflight has no automatic blockers. Track PP-aware DSpark under a separate
   later implementation and admission plan.

## Admission and benchmark gates

### Installation and software

- PSU, independent cable/connector, chassis clearance, and sustained cooling
  are signed off for both 3090s plus a 575 W 5090.
- The 5090 enumerates as SM120 with at least 31 GiB visible and PCIe 5.0 x16
  maximum capability on its intended slot.
- CUDA toolkit is at least 12.9, Torch's CUDA build is at least 12.8, and the
  FlashInfer package is at least 0.6.9. These are automatic version gates, not
  runtime proof.
- The mixed runtime is a new immutable build. Its receipt proves imported module
  origins and distribution ownership, source/runtime/native-extension hashes,
  Python ABI, driver and compiler identity, generated `sm_86` plus `sm_120` or
  `sm_120f` targets, and rank/architecture-separated caches. No import resolves
  through the frozen EP2 runtime or an unrelated environment.
- Separate manual receipts prove an empty-cache Torch SM120 compile **and
  execution** and an empty-cache Triton SM120 compile **and execution** against
  the detected RTX 5090 UUID. FlashInfer's SM120 MXFP4 MoE method and every
  Oscar writer/scorer/attention consumer must also compile and execute before a
  model launch is accepted; `cuobjdump --list-elf` or equivalent must prove the
  expected native objects rather than relying on environment strings.
- `CUDA_FORCE_PTX_JIT=1` smoke tests pass where PTX fallback is claimed, then
  the variable is removed for performance runs.

### Transport

- Run `nvidia-smi topo -m`, `nvidia-smi topo -p2p p`, NVIDIA `nvbandwidth`,
  and NCCL tests for every proposed rank pair.
- Measure 8 KiB, 64 KiB, and 1 MiB collectives/transfers; decode is sensitive
  to latency, so bulk PCIe bandwidth alone is insufficient.
- The NCCL log must match the transport recorded in the benchmark receipt. No
  run may claim NVLink for an edge involving the 5090.
- Capture current PCIe width/speed during or immediately after the active test.
  A downtrained idle sample is not an acceptance receipt; a link that remains
  below PCIe 5.0 x16 under load is a blocker.

### Correctness and long-context quality

- Retain the cache-flushed 42,125-byte OpenCode gate: three exact semantic
  responses with deterministic final content plus one exact forced tool call.
  Also retain malformed-stream, degeneration, prompt-copy, and runaway checks.
- Compare target logits and semantically validated natural-EOS output against
  the qualified TP2 run on the exact 2,694-token prompt. Forced 512-token
  filler output is explicitly forbidden. DSpark acceptance is not a metric for
  the target-only PP3 path; add an acceptance gate only to a future PP-aware
  speculative admission plan.
- Run Oscar-KV retrieval and agentic tests at 64K, 128K, 256K, and 524K. At
  overlapping lengths, compare against BF16 KV and the admitted SM86 Oscar
  output. At 524K, qualify the exact artifact-bound rotation, clipping,
  scale/zero, packing, and protected-SWA contract used by SM120 and require no
  material task-accuracy regression. Generic FP8 or an identity/default
  transform is not an accepted control.
- A successful allocation is not a 524K quality pass.
- Before model-level quality, require SM120 unit receipts for byte-identical
  C4/C128 packing, scale/zero and rotation handling, clipping, negative padding,
  page tails, C4 top-k/logits, decode/extend/prefill, prefix reuse, eager-versus-
  graph output, and graph replay while logical lengths change.

### CUDA graph and memory

- Target decode graph capture must complete for every rank without eager
  fallback or capture warnings. Prefill graphs remain disabled. There is no
  DSpark/speculative graph capture in the current PP3 path.
- Record the post-capture headroom under the 2,560-slot reserve and establish a
  measured qualification floor before promotion; retain 4 GiB as the hard
  floor after any later expert expansion.
- A 524K request must be admitted, complete, and release its allocations
  without a monotonic memory increase across three repetitions.

### Staged qualification and performance

Do not make the 90-token/s objective obscure whether the hardware and runtime
are correct. Promotion proceeds through three distinct decisions:

1. **Commissioning-qualified, not production:** hardware identity, isolated
   toolchain/native objects, transport, CPU affinity, exact Oscar parity and
   quality, exact 524K pool, graph replay, memory headroom, and leak gates pass.
   This status changes nothing about the hash-frozen EP2 service.
2. **Experimental production candidate:** compare the PP3 result with both the
   unchanged EP2 receipt and a target-only two-3090 PP2 control using the same
   prompt/context/graph policy. After two warmups and at least five measured
   runs, require median useful natural-EOS decode throughput of at least 46.972
   token/s (5% above the final coherent 44.735 median), no accepted run below
   44.735 token/s, and p95 TTFT no greater than 7.0 s. A candidate that merely
   fits or regresses the baseline is rejected.
3. **Program stretch exit:** retain at least 90.0 median decode token/s and no
   accepted run below 85 token/s as the optimization target. A correct PP3
   system may be commissioned without claiming this performance milestone.

Every performance stage also reports p50/p95 inter-token latency, per-stage
target-forward latency, CPU expert route fraction and effective affinity,
GPU/CPU utilization, clocks, power, temperature, throttling reasons, measured
transport, and per-rank memory. Do not report a DSpark acceptance length for a
target-only run. A future PP-aware speculative candidate must add acceptance
length and per-stage proposal/verification latency under a separate plan.

## Read-only preflight

After installation, run the admission checker with the DSV4 runtime Python:

```bash
/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/dsv4_flash_rtx5090_preflight.py --json
```

That default intentionally inspects the checked-in two-3090 base launcher, so
its SM86/NVL defaults block mixed-hardware admission. Evaluate a proposed PP3
entrypoint explicitly:

```bash
/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/dsv4_flash_rtx5090_preflight.py \
  --launcher /path/to/proposed-dsv4-pp3-launcher.sh --json
```

The checker launches no CUDA kernels and changes no state. Schema
`dsv4-rtx5090-preflight-v4` inventories GPU index, UUID, PCI identity, memory,
compute capability, exact PCI/NUMA placement, separate maximum and current
PCIe links, toolchain floors, topology/P2P output, and the candidate launcher's
statically recognizable mixed-architecture and NCCL defaults. Its JSON report
includes the PCI-address-derived rank order; explicit half-open target-layer
ranges `[0,18)`, `[18,35)`, and `[35,43)`; counts `18,17,8`; and a fixed 524K
commissioning profile (`TP1/PP3/EP1`, target-only, Oscar INT2 through the
external E4M3 byte carrier, 2,560 protected-SWA slots, full decode graph,
disabled prefill graph, one running request, mixed SM86/SM120 cache root). CUDA
indexes are reported observations, never rank identity. Tests require the
ranges to be contiguous, non-overlapping, and to cover every layer in `[0,43)`
exactly once.

The automatic gates cover unique identities; exact 2x3090 + 1x5090 inventory;
fixed PCI/NUMA locations; PCI-derived rank resolution; SM120 identity and
memory; separate maximum and current PCIe 5.0 x16 links; retained 3090 memory;
CUDA/Torch/FlashInfer version floors; source-parsed mixed architecture defaults;
and a source-parsed NCCL-auto default. A legacy contract-marker string does not
satisfy any of those gates. The two source gates are advisory because v4 is a
small assignment parser, not a shell evaluator; unreachable or subsequently
overridden assignments can fool it. Installed-hardware admission remains
fail-closed because the five manual gates below cannot be self-attested.

Required hardening is a side-effect-free launcher `--print-contract` mode or a
declarative manifest consumed by both launcher and preflight. Until one exists,
the source-gate details must never be described as proof of runtime-effective
values.

Five gates deliberately remain manual: empty-cache Torch SM120 compile/runtime,
empty-cache Triton SM120 compile/runtime, measured PCIe P2P bandwidth,
power/cooling, and `oscar-int2-sm120-parity-quality`. The last gate requires
installed-GPU receipts for the KV-only writer, C4 writer/scorer, mixed protected
BF16-SWA plus compressed-history decode/extend/prefill paths, CUDA graph replay,
artifact parity, and task quality through 524K. The command returns nonzero
whenever an automatic blocker or manual gate remains. It does not accept a
self-attestation flag: those gates stay manual until a separate artifact
validator is implemented. A JSON result can therefore have no automatic
blocker while still reporting `automatic_blocked: false`,
`manual_pending: true`, `admitted: false`, and `blocked: true`.

### Pre-install result on 2026-08-02 and corrected interconnect audit

The checker was run on `dwagon` before any hardware change. It exited 1, which
is the expected admission result while the requested card is absent.

| Gate | Observed result | Status |
|---|---|---|
| Required GPU inventory | 0 RTX 5090; 2 RTX 3090 | Block: no 5090 installed |
| Existing GPU identity | both RTX 3090s report SM86 and 24,576 MiB | Pass |
| Existing GPU NUMA locality | GPU 0 on NUMA 0; GPU 1 on NUMA 1 | Pass |
| System CUDA toolkit | 13.1, above the 12.9 SM120 compilation floor | Pass |
| Torch CUDA build | 12.8 | Pass |
| FlashInfer | 0.6.9 | Pass inventory/version gate |
| RTX 5090 maximum/current PCIe | unavailable because the card is absent | Specific gates deferred; inventory gate already blocks |
| Base-launcher source-parsed architecture defaults | FlashInfer `8.6`; Torch `8.6` | Advisory gate blocks mixed hardware |
| Base-launcher source-parsed NCCL default | `NVL` | Advisory gate blocks; a mixed launcher must intend NCCL auto |
| Current two-3090 interconnect | fresh CUDA P2P matrix true in all directions; NCCL selected direct `NVL`/`P2P/IPC`; all 16 physical Tx/Rx link counters advanced during model windows | Pass for the existing 3090 pair; says nothing about a future 5090 edge |
| Torch SM120 empty-cache compile/runtime | no installed-card receipt | Manual gate |
| Triton SM120 empty-cache compile/runtime | no installed-card receipt | Manual gate |
| Power, clearance, cooling | not machine-verifiable | Manual gate |
| Measured P2P/NCCL bandwidth | not run by read-only checker | Manual gate |
| `oscar-int2-sm120-parity-quality` | no installed-card parity/quality receipt through 524K | Manual gate |

This result validates that schema v4 fails closed for missing hardware,
slot/NUMA identity, SM86-only/NVL source-parsed launcher defaults, and every
unresolved manual gate while recognizing the usable local toolchain. The base
launcher's override capability is tested separately; neither that capability
nor a static source-parser pass proves future runtime defaults. Package
versions do not prove SM120 execution, which is why Torch and Triton have
separate manual runtime gates. The early `topo -p2p p` `NS` observation is
superseded for the existing
3090 pair by the later three-layer audit: a fresh all-true CUDA peer matrix,
direct NCCL NVLink/P2P-IPC selection without SHM/NET fallback, and positive
traffic on every physical Tx/Rx counter during real model work. That evidence
must not be carried over to the future 5090, which has no NVLink and retains its
separate manual PCIe-P2P bandwidth gate.

The v4 checker has 19 focused tests, and the base-launcher mixed-architecture /
NCCL-auto override contract has 5 more. These 24 tests cover fail-closed source
parsing, separate maximum/current PCIe blockers, hardware identity/location,
rank stability across CUDA-index changes, exact half-open range counts and
coverage, the Oscar-INT2 SM120 gate identity/contract, toolchain floors, JSON
status, and unchanged two-3090 defaults. They do not replace the five
installed-hardware manual gates.

## 2026-08-04 reference update after split-history qualification

Future 5090 comparisons must use the final coherent two-3090 EP2 reference,
not the historical 34.392-token/s Oscar baseline. The new confirmation is
**44.690 mean / 44.735 median decode token/s** with **5.542 s median / 5.624 s
maximum** fresh TTFT. It uses exact 524K limits, full decode/speculative CUDA
graphs, breakable prefill graphs, both 3090s over verified NVLink, CPU offload,
and the mandatory Oscar INT2 split-history implementation.

The future SM120 admission gate must now prove parity for both stages of that
implementation: sink-free FP32 partial generation and deterministic ordered
combination with one sink owner, including fixed-address CUDA-graph replay.
The current SM86 split map and 4,210,688-byte worker arena are reference inputs,
not assumed SM120 optima. Generic FP8, generic INT4, or BF16 history is still
not an acceptable fallback on the 5090. Only the protected SWA reserve remains
BF16 under the Oscar contract.

The exactly-two PP2 experiment does not establish a new comparison baseline.
Its optimized 22/21 partition reached 28.606 aggregate native token/s under two
concurrent requests but failed a reproducible semantic marker and exceeded the
7 s concurrent TTFT target. A future PP3 run must beat the coherent 44.735
single-request EP2 median for production promotion and must independently pass
all semantic/tool, concurrency, Oscar, graph, memory, and PCIe-P2P gates.
