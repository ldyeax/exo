# DeepSeek V4 Flash OpenCode optimization campaign

Started: 2026-08-03; last updated: 2026-08-04

Host: `dwagon`

Checkpoint: `/tmp/dsv4-local-checkpoint-0731`

## Current campaign status

Serving qualification is now **OSCAR-INT2 only**. The EP2 and PP2 launchers
will not start a serving run without a calibrated, model-bound Oscar artifact
and admission receipt, and they reject the earlier raw-E4M3, symmetric-INT4,
and selective-BF16 history-cache alternatives. `fp8_e4m3` is retained only as
SGLang's public raw-byte carrier.

The final coherent TP2/EP2 candidate combines the CPU inline-dispatch/N128-LUT
artifact with the fixed-address Oscar split-history decoder. Its independent
confirmation measured **44.690 mean / 44.735 median decode token/s**,
**5.542 s median / 5.624 s maximum** fresh TTFT, and **2.402 committed tokens
per speculative cycle**. Mean target verification is **47.635 ms**. The exact
524K pool, full decode/speculative CUDA graphs, breakable prefill graphs, two
3090s, CPU offload, semantic output, and forced OpenCode tool call all passed.
The <=7 s short-prompt TTFT gate passes, but the 80 and 90 token/s decode goals
do not. A realistic 42K OpenCode system context still takes about 22--24 s to
first output.

The independent NVLink preflight proved peer access in both directions, NCCL
P2P/CUMEM transport without SHM/NET fallback, a 10.322 microsecond 48 KiB graph
all-reduce, and 20.161 GB/s at 1 MiB. Every one of the 16 per-link counters
also advanced during each model receipt (minimum delta 2,586,151 KiB). NVLink
is active and is not the present single-request limiter; target verification
and its socket-local CPU expert tail are.

All speed and latency results below the next heading are retained as historical
pre-OSCAR controls. They must not be presented as the final cache candidate.
The current implementation, calibration evidence, and remaining admission
work are recorded in [the OSCAR hard-switch section](#2026-08-03-oscar-int2-hard-switch).

## Pre-OSCAR control status

The last pre-OSCAR local TP2/EP2 handoff was coherent and reproducible, but the
80 and 90 token/s decode goals were not achieved. That control used the
frozen **g14-p28** placement (14 GPU and 114 socket-local CPU experts per rank
and layer), compact **fixed verify 4**, breakable prefill plus full
decode/speculative CUDA graphs, the SM86 byte-packed E4M3 cache, and the full
524,288-token pool. Two
clean five-phase receipts produced a pooled **36.05 HTTP decode token/s** and
about **4.79 s median uncached TTFT**. A qualified exact radix hit reused
2,560 of 2,694 prompt tokens and reached **0.611 s TTFT**. The clean launch
retained **4.15 GiB** of SGLang allocator headroom after graph capture and
**3.73 GiB** of physical idle memory per GPU.

Each five-phase receipt contains a cold exact prompt, a same-process radix-hot
exact repeat, a controlled near prompt, then cache-flushed warm exact and near
controls. Across two clean receipts that is six executions of the exact long
prompt and four executions of its near variant. Every phase stopped naturally
and passed semantic, repetition, prompt-copy, token-ID, and stream-protocol
gates. Natural generations did not follow identical token trajectories, so
the receipt-level paired-attribution flag correctly remains false; the pooled
HTTP rate is a diagnostic aggregate of individually accepted runs, not an
artificial fixed-output benchmark.

### Greedy trajectory drift: bounded diagnostic and current hypothesis

`temperature=0` plus `sampling_seed=0` does not make this execution path
bitwise invariant. In greedy mode the sampler goes directly to
`torch.argmax` (`vendor/sglang/python/sglang/srt/layers/sampler.py:126-133`),
so the seed is not consulted. The DSpark draft and acceptance paths are also
greedy and broadcast rank-0 token choices; this prevents two TP ranks from
committing different tokens within one request, but it does not force rank 0's
upstream floating-point inputs to be identical across independent requests.

The strongest concrete **hypothesis**, not a proven root cause, is C4 selection
order. The production launcher explicitly sets `SGLANG_OPT_USE_TOPK_V2=0`
(`scripts/dsv4_flash_0731_tp2_dwagon.sh:415`). With indexer capture and HiSparse
disabled in the qualified server, `dsv4/indexer.py:1120-1176` selects the v1
top-k transform. That CUDA kernel uses `atomicAdd(&s_counter, 1)` to assign
qualifying C4 indices to output slots
(`kernels/jit/csrc/deepseek_v4/topk_v1.cuh:125-151,181-220`). The selected set
can therefore be stable while its array order follows atomic arrival order;
an exact-score boundary tie could also affect which tied index fills the last
slot. Oscar passes this array directly as `extra_indices`
(`deepseek_v4_backend.py:1861-1904`). Its sparse-attention kernel visits those
indices in array order and performs an online FP32 max/sum/accumulator update
for each tile (`oscar_int2_decode.py:110-295`). A permutation is mathematically
equivalent but changes floating-point association, and a later near-tie
argmax or expert-routing boundary can amplify the small difference into a
different natural-stop trajectory.

This chain is evidence for a targeted experiment, not evidence that the v1
kernel is the cause. The five-phase benchmark also compares fresh full-prefill
and radix-reuse paths, which need not be bitwise identical, while the observed
fresh-versus-fresh drift means cache-path mixing is not the whole explanation.
The GPU MoE small-row router, split-K scratch reduction, and CPU AMX merge paths
were inspected and use fixed ownership/reduction order for a fixed shape, so
they are lower-priority suspects. CARv2, graph replay, and speculative verify
remain one-variable A/B candidates after C4 selection order is isolated.

The hotspot harness now has an opt-in bounded probe for that isolation:

```text
--deterministic-repeat-count 3 --deterministic-repeat-output-tokens 128
```

It runs after the five performance phases, flushes the radix cache before every
identical prompt, forces the same output-token count, and compares redacted
token-ID hashes when the server supplies complete IDs (decoded-text hash plus
count otherwise). Counts are restricted to 2-5 repeats and 2-256 output tokens.
The nested receipt is always labeled `diagnostic_only`,
`performance_claim_eligible=false`, and `coherency_claim_eligible=false`; its
timings and semantic assessment cannot satisfy either campaign gate. Production
top-k ordering remains unchanged until this probe and a deterministic-selector
A/B establish causality and cost.

The pre-OSCAR control launcher materialized the repository-owned
[`g14-p28` manifest](scripts/data/dsv4_flash_opencode_g14_p28_frozen_plan.json)
and runs the deterministic `dsv4_opencode_2694` compilation warmup internally
before ASGI readiness. It exercises the two 1,024-token chunks, the 646-token
remainder, and decode without exposing warmup output. No NFS checkpoint is
needed: the TP and PP entrypoints, benchmark, and coherency validator default
to the local 48-shard checkpoint at `/tmp/dsv4-local-checkpoint-0731`.

The harder cold OpenCode gate also passed on the g14-p28 placement: repeated
42,125-byte system-prompt requests returned the exact expected facts, and the
forced-tool request returned the exact schema and arguments with
`finish_reason=tool_calls`. Historical packed-build timings were 18.1072,
18.3685, and 18.5085 s TTFT for three semantic requests and 23.4939 s for the
tool request. A transient
`flush_cache` HTTP 400 while a prior flush was still draining is now retried
within the timeout; all other non-success statuses remain fatal. The <=7 s
target therefore applies to the qualified 2,694-token performance shape, not
a cache-cold 42 KB OpenCode system context.

The earlier 90-93 token/s result remains permanently retracted. It was 512
copies of the prompt filler token under `ignore_eos`, not useful model output. No
number in this document is called valid unless its own output gate passed.

All accepted model runs used both local RTX 3090s plus socket-local CPU
offload. No work was run on fwuff, and fwuff's GPU was never included in CUDA
or NCCL.

## Pre-OSCAR TP2/EP2 launcher record

The production entrypoint remains:

```bash
scripts/dsv4_flash_hybrid_ep2_dwagon_opencode.sh --launch
```

It no longer launches the packed-FP8 control described in this section. It now
fails closed until the Oscar artifact and admission files described below are
present. The following table records the historical control contract for
comparison only.

The wrapper injects the official OpenCode-facing model name and parsers:

```text
served model       deepseek-v4-flash
reasoning parser   deepseek-v4
tool parser        deepseekv4
thinking default   true
API                http://127.0.0.1:30010/v1
```

The effective runtime contract is:

| Area | Qualified value |
|---|---|
| Parallelism | TP2 + EP2 on local CUDA devices 0 and 1 |
| Target placement | frozen g14-p28: 14 GPU + 114 socket-local CPU experts per rank/layer |
| Draft placement | 14 GPU + 114 CPU experts per rank on each of 3 DSpark stages |
| CPU execution | one 56-thread pool per rank, NUMA 0/1 |
| Context / total pool | 524,288 / 524,288 tokens |
| KV request / physical storage | `fp8_e4m3`; raw E4M3FN bytes with software BF16 decode on exact SM86 |
| SWA reserve | ratio `0.0048828125`, or 2,560 slots |
| Prefill | 1,024-token chunks; breakable graph tiers 256/512/1,024 |
| Prefill attention boundary | whole attention module eager inside breakable replay |
| Decode/speculation | full target-verify and draft CUDA graphs, BS1 |
| Verification policy | compact fixed tier 4; graph tiers 1-6 retained as rollback |
| Startup warmup | internal `dsv4_opencode_2694` pre-readiness compile request plus generic warmup |
| DSpark geometry | block 5, target layers 40/41/42, Markov rank 256 |
| Model MoE overlap | enabled |
| Draft numerics | BF16 LM head; FP32 Markov projection |
| CPU small-row dispatch | AMX at 5+ routed rows; pinned AVX path at 2-4 rows |
| Request policy | one running request; radix cache retained |

Startup proves the important invariants rather than inferring them from flags:

- the target plan is materialized from the repository-owned g14-p28 manifest,
  while explicit geometry/profile overrides deliberately use the profile
  builder;
- draft shared-expert checkpoint coverage is logged as 18/18 on both ranks;
- requested and effective KV dtypes are logged separately;
- target prefill, target verify, and draft graph captures must all finish;
- the 2,694-token compile warmup and generic warmup must finish before ASGI
  startup completes;
- the runtime must report both `context_len=524288` and
  `max_total_num_tokens=524288`.

The clean g14-p28 launch retained **4.15 GiB** of SGLang allocator headroom
after target-prefill, target-verify, and draft-verify capture, and
`nvidia-smi` showed **3.73 GiB/GPU** physically free while idle. For historical
comparison, the original BF16-cache launch reported 3.28 GiB of allocator
headroom and the earlier g13 packed launch reported 4.83 GiB. The extra g14
slot costs memory, but preserves the 4-GiB allocator admission floor. SGLang
allocator availability and `nvidia-smi` free memory measure different scopes
and must not be conflated.

## Why the original OpenCode output was corrupt

The original handoff used a partial breakable-prefill graph boundary. A
515-token request replayed in a 1,024-token bucket while Python closures still
held capture-time metadata for 1,024 rows. The 509 padded cache locations were
zero, so replay repeatedly wrote KV slot 0. Short warm prompts and hash-only
tests did not expose it.

The repair has two parts:

1. DSV4 break bridges use the live replay `ForwardBatch` rather than stale
   capture metadata.
2. The production boundary keeps the entire DSV4 attention module eager
   inside the surrounding breakable graph:

   ```text
   DSV4_PREFILL_GRAPH_BACKEND=breakable
   DSV4_EAGER_ATTN_MODULE_IN_BCG=1
   DSV4_CAPTURE_ATTN_IN_BCG=0
   ```

This retains prefill CUDA graphs for the surrounding model work while keeping
KV-addressing-sensitive attention out of replay. Target verification and the
DSpark draft remain fully captured.

An independent 2x DGX Spark report found a different cold-agent failure around
speculative placeholders on the final chunk. Its direct patch is vLLM-specific,
but it supports the same operational lesson: cold, cache-busted, multi-chunk
agent prompts are mandatory; a warm chat smoke test is not a coherency gate.

## DSpark correctness repairs

The official checkpoint geometry is block 5 with target layers 40, 41, and 42,
Markov rank 256, and a 128-token sliding window. The former local block-8
override changed the drafter's trained semi-autoregressive geometry and is not
a valid optimization. The launcher now fails back to block 5 unless the user
explicitly owns an experiment.

The draft loader is also fail-closed. An independent vLLM deployment reported
that silently missing shared-expert tensors reduced mean accepted length from
4.01 to 2.28 and mean decode from 55.4 to 32.7 token/s. The local SGLang loader
already mapped those tensors, but it previously tolerated incomplete mapped
destinations. It now requires all 18 draft shared-expert tensors and every
mapped `mtp.*` destination. Draft CUDA-graph/eager parity was exact on both TP
ranks for the first eight steps.

## Placement, hotspot, and CPU-kernel work

The accepted target placement is the reviewable
[`dsv4_flash_opencode_g14_p28_frozen_plan.json`](scripts/data/dsv4_flash_opencode_g14_p28_frozen_plan.json)
manifest. It pins 14 rank-owned GPU experts in each of 43 layers and records
ownership and semantic hashes; the launcher materializes the corresponding
binary plan at startup. It was selected from coherent, cache-flushed OpenCode
decode traces by optimizing the critical rank/layer CPU tail, not merely total
route frequency. The earlier repository profile
[`dsv4_flash_opencode_target_distinct_decode_calls_sparse.json`](scripts/data/dsv4_flash_opencode_target_distinct_decode_calls_sparse.json)
remains the fallback for explicit geometry experiments. Its historical g13
selection covered 27.9508% of recorded target activations. The draft profile is
[`scripts/data/dsv4_flash_draft_distinct_decode_calls_sparse.json`](scripts/data/dsv4_flash_draft_distinct_decode_calls_sparse.json)
and covers 98.4409% of recorded draft activations across the two disjoint rank
sets.

The live hotspot implementation retains full immutable rank-owned CPU shadows,
copies V4 MXFP4 weights directly into fixed-address GPU slots, and commits
GPU/CPU masks and logical mappings only at an idle boundary. All ranks validate
the same generation before commit; failed updates roll back to the prior
generation. A one-slot transaction and its rollback passed graph replay,
semantic, and forced-tool checks. The full g13-p26 transaction moved 354 slots
on rank 0 and 364 on rank 1, then lowered measured target verification by
12.95% and raised trace committed throughput by 12.88% relative to the
same-process rollback baseline. The g14-p28 placement subsequently delivered a
smaller but repeatable gain and became the frozen launch-time default. The live
controller stays opt-in because a frozen startup plan avoids CPU-shadow memory
and update-pause costs during ordinary serving.

The installed KTransformers package remains the rollback point. The launcher
uses [`scripts/stage_dsv4_kt_avx_tail_overlay.py`](scripts/stage_dsv4_kt_avx_tail_overlay.py)
to copy a complete package into a content-addressed overlay and accepts only
the pinned candidate hash. The AVX candidate improved isolated routed-expert
microbenchmarks by 41.5% at two rows and 67.4% at three rows; those sizes cover
35.4% of recorded routed CPU rows. AMX remains faster from five rows upward.
An explicit `DSV4_KTRANSFORMERS_SOURCE` always wins, and
`DSV4_STAGE_KT_AVX_TAIL_OVERLAY=0` is the deliberate opt-out.

The remaining low-level suggestions were tested rather than assumed. The SM86
small-row routing metadata path now runs as a graph-safe GPU kernel and measured
roughly 13-21 microseconds per isolated call. A grouped CPU M1-4 MXFP4 candidate
reached only 0.94-0.96x the installed path and was rejected. Custom all-reduce
v2 remains enabled. KTransformers timing telemetry was repaired so asynchronous
CUDA events are not queried during graph capture; production leaves that timing
off after qualification. These changes removed Python/capture artifacts from
the measurement, but none displaced target verification as the dominant tail.

The SM86 MXFP4 small-batch opt-in now fails closed. The pinned
`make_default_opt_flags_nvidia` signature, its thirteen-positional-argument
call contract, and the exact non-persistent dispatch constraint are checked;
an incompatible package upgrade or target dispatch aborts instead of silently
using the package heuristic. Focused CUDA-graph parity covers every local width
E14-E22, logical row count 1-6, and dynamic live-route count 0-6. The selected
kernel remains block-N 128, split-K 2, four stages, and four warps for both W13
and W2.

Each scheduler records specialization selections made during graph compilation
and capture. Because graph replay does not re-enter Python, rank-local counters
are TP-all-gathered over the CPU process group before the single scheduler
control reply is returned. `/server_info` exposes
`dsv4_sm86_small_batch_gemm_worker_telemetry`, the reporting/active/expected
worker counts, and `dsv4_sm86_small_batch_gemm_all_workers_active`. The last
field is true only when distinct rank records cover the configured
`tp_size * pp_size * dp_size` topology and every record installed and selected
the specialization. A lone TP0 report therefore cannot qualify a TP2 run.

## Last clean pre-OSCAR TP2/EP2 qualification

That clean control launch disabled benchmark-only expert timing, recorder, hotspot
shadow, and NCCL debug instrumentation while retaining the frozen g14-p28
placement, fixed verify 4, FP8 cache, all CUDA graphs, P2P checking, and NCCL
pre-warm. Two complete five-phase receipts produced:

| Metric | Result |
|---|---:|
| Accepted semantic phases | **10/10** |
| Pooled HTTP decode | **36.05 token/s** |
| Median uncached TTFT | **about 4.79 s** |
| Qualified exact radix-hit TTFT | **0.611 s** |
| Exact radix reuse | 2,560 / 2,694 tokens |
| Post-graph allocator headroom | **4.15 GiB/GPU** |
| Physical idle headroom | **3.73 GiB/GPU** |
| Strict repeated semantic / forced tool | **pass / pass** |

The five phases are `cold_first_exact`, `radix_hot_exact`,
`radix_hot_near`, `warm_no_radix_exact`, and `warm_no_radix_near`. The exact
prompt appears three times per receipt; the near prompt preserves a controlled
69.67% common prefix but changes the task sufficiently that the server reported
zero cache reuse. Cache flushes before the two warm controls distinguish radix
reuse from ordinary process, graph, and host-page warmth. Repeating the entire
sequence twice guards against choosing a single lucky output trajectory.

The clean HTTP-only receipts intentionally contain no per-cycle timing hooks.
The instrumented qualification immediately before the clean launch measured
g14-p28 at 35.2917 trace committed token/s, 2.3029 mean accepted tokens per
step, and 58.4825 ms weighted target verification. The mean full cycle implied
by those values is about 65.26 ms. At the current acceptance, 80 token/s would
require a 28.79 ms cycle and 90 token/s a 25.59 ms cycle. Target verification
alone is already more than twice either budget, so the plateau is the
target-side CPU-expert/verification tail rather than an inactive second GPU or
the DSpark draft. Perfect six-token acceptance would be about 91.9 token/s at
the present cycle, but the observed acceptance is 2.30; acceptance tuning alone
cannot credibly supply that ideal.

### NVLink qualification during model work

NVLink was checked at three independent layers rather than inferred from a
topology diagram:

1. A fresh runtime P2P probe reported all four directed peer-access entries
   true (`0->0`, `0->1`, `1->0`, and `1->1`), with cache SHA-256
   `9415bf51669aa98993e6c5342be7b7960bafdd62341ebce4637cc8c384c24823`.
2. NCCL was launched with `NCCL_P2P_LEVEL=NVL`; its topology labeled the peer
   edge `NVL[48.0]`, reported `isAllDirectP2p 1`, and selected `P2P/IPC` for
   every channel in both directions. No SHM or NET data-channel route appeared.
3. Hardware counters were sampled around each real five-phase model campaign.
   All 16 counters (two GPUs, four links, Tx and Rx) advanced; the clean TP2
   receipts advanced the least-active counter by about 2.66 million KiB.

This proves that the benchmark process had functioning bidirectional GPU P2P,
that NCCL selected the direct NVLink-class path instead of SHM/NET fallback,
and that all four physical lanes carried traffic during the model window. It
does **not** prove that every counter byte belonged to one named collective or
justify deriving an application bandwidth from those deltas.

## Historical TP2/EP2 qualifications

The tables below are retained as historical, controlled milestones. They are
superseded by the clean g14-p28/fixed4 qualification above, but preserve the
evidence that separated the coherency, cache-storage, warmup, and placement
changes.

The historical BF16-cache no-override restart produced:

| Cache-cold sample | Completion | TTFT (s) | Decode (token/s) | Gate |
|---:|---:|---:|---:|---|
| 1 | 232, natural stop | 5.190474 | 29.165350 | pass |
| 2 | 232, natural stop | 4.856548 | 31.957486 | pass |
| 3 | 244, natural stop | 4.936766 | 30.176400 | pass |
| **Median** | | **4.936766** | **30.176400** | **pass** |

Each input was exactly 2,694 tokenizer IDs. The prompt contained a varied
repository-style context and an exact semantic task; it did not use filler or
`ignore_eos`. The benchmark decoded returned token IDs and rejected missing
facts, missing required ending, degeneration, prompt copying, terminal-token
misuse, incomplete streaming, or a non-natural finish.

The strict OpenCode receipt was:

| Gate | Result |
|---|---|
| System prompt | 42,125 bytes; cache flushed before every request |
| Semantic runs | 3/3 exact; deterministic content and reasoning hashes |
| Semantic TTFT | 18.8565 / 19.0303 / 19.0037 s |
| Forced tool | exact schema/arguments, no prose, `tool_calls` finish |
| Forced-tool TTFT | 19.8464 s |
| Overall | `coherent=true` |

### SM86 packed-E4M3 qualification

The same historical g13 launcher was then qualified with the
architecture-neutral cache implementation active. Startup logged all of the
following on both ranks:

```text
requested=fp8_e4m3, effective=fp8_e4m3, device=SM86
DeepSeek V4 KV cache on SM86 uses raw E4M3FN bytes with software decode
swa_size=2560 c4_size=131072 c128_size=4096
context_len=524288 max_total_num_tokens=524288 available_gpu_mem=4.83 GB
```

All three prefill tiers (256/512/1,024), the six-token target-verification
graph, and the five-token draft-verification graph captured successfully. The
custom 2,694-token compilation warmup then ran before readiness. The strict
semantic/tool gate passed before performance was measured.

| Cache-cold packed sample | Completion | TTFT (s) | Decode (token/s) | Gate |
|---:|---:|---:|---:|---|
| 1, first external request | 229, natural stop | 4.821314 | 29.479782 | pass |
| 2 | 243, natural stop | 4.830498 | 26.921200 | pass |
| 3 | 233, natural stop | 4.827004 | 24.719720 | pass |
| **Median** | | **4.827004** | **26.921200** | **pass** |

An earlier packed qualification without the shape-specific pre-readiness
warmup took 8.4198 s on its first external request and about 4.88 s on the
next two. It remains useful evidence for the warmup fix, but it is superseded
by the table above.

The end-to-end token/s difference versus BF16 includes content-dependent
speculative acceptance and is larger than the isolated packed-kernel costs.
Three samples are not enough to attribute that acceptance difference to cache
quantization. The safe conclusion is that packed E4M3 is a memory and
context-capacity win, not a decode-speed win.

### Optimization ledger

All rows below are semantic natural-stop runs. Single-sample intermediates are
diagnostic, not final claims.

| Controlled configuration | Decode (token/s) | TTFT (s) | Decision |
|---|---:|---:|---|
| Repaired block-5 baseline, BF16 draft numerics | 16.8998 | 6.4231 | correctness baseline |
| FP32 LM head + FP32 Markov | 19.3464 | 5.7841 | separate the two numerics |
| Model MoE overlap + FP32 Markov only | 21.3581 | 5.7421 | keep overlap |
| AMX crossover 5 | 21.0195 | 5.4085 | keep for row-size dispatch |
| AVX 2/3-row tail + coherent g12 | 26.6255-30.1182 | 5.0419-5.2567 | keep AVX overlay |
| Coherent g13, FP32 Markov | median 30.4848 | median 4.9424 | keep g13 |
| Same g13, BF16 Markov | median 26.8786 | median 4.9421 | reject; acceptance regressed |
| Exact BF16-cache launcher baseline | **median 30.1764** | **median 4.9368** | comparison baseline |
| Historical packed-E4M3 launcher | **median 26.9212** | **median 4.8270** | superseded 524K milestone |
| Last clean pre-OSCAR g14-p28/fixed4 | **pooled 36.05** | **about 4.79** | historical control |

The following structural estimate is the historical g13 result. Its trace
measured 84.1715 ms per complete speculative step with a
mean accepted length of 2.5606. Even a perfect six-token commit every cycle
would be about 71.3 token/s. At current acceptance, 90 token/s would require a
28.45 ms cycle; even at perfect acceptance it requires less than 66.67 ms.
The primary bottleneck is target verification and CPU expert streaming, not
the three-stage draft, which measured about 6.47 ms of the cycle.

The pre-OSCAR g14 trace above improves that cycle to about 65.26 ms, but still
misses both goals at observed acceptance. Further work toward 80-90 should
therefore prioritize target-side expert
residency/streaming, concurrent CPU/GPU expert execution, and verify kernels.
Acceptance-only tuning cannot close the gap on this hardware.

Two packed-cache kernel changes survived parity and graph-replay testing:

- the C4 scorer uses a shape-stable persistent strided grid. At batch 6 with
  704 live tokens in a 131,072-token capacity it reduced launch programs from
  1,758 to 384 and measured 0.010240 ms versus 0.032768 ms for the former
  grouped grid under the same `triton.testing.do_bench` protocol. At the full
  131K capacity it was 3.6% slower, so the direct one-page grid remains for
  64 pages or fewer;
- sparse attention now loads each key's seven UE8M0 scales once and broadcasts
  them across the seven 64-value groups. Sustained-clock isolated C4 shapes
  improved by about 8.5% and C128 shapes by about 7%, with bit-exact output
  and exact CUDA-graph replay.

These are kernel-local results, not end-to-end decode claims. Natural-output
length and speculative acceptance varied enough that the final server median
did not improve over the earlier packed receipt.

## SM86 E4M3-as-storage implementation

The former SM89 gate is now split into two capabilities: native FP8 compute
still requires SM89+, while exact SM86 can retain E4M3 as architecture-neutral
`uint8` storage and decode it into BF16 tensor-core operands. Other pre-SM89
capabilities continue to fall back to BF16. This keeps server-argument,
planner, allocator, writer, and consumer decisions consistent.

Each logical main-cache token is:

| Region | Bytes | Representation |
|---|---:|---|
| no-PE latent | 448 | raw E4M3FN bytes |
| RoPE tail | 128 | 64 BF16 values |
| scales | 8 | seven UE8M0 group scales plus one pad byte |
| **Total** | **584** | architecture-neutral byte layout |

Allocator page padding makes SWA P128 and C4 P64 about 585 bytes/token; a
two-token C128 page is 864 bytes/token. The planner charges these physical
page sizes instead of assuming the logical 584-byte record.

The implementation covers the fused main/SWA writer, C4 and C128 compressor
writers, DSpark writes, debug dequantization, the C4 indexer, sparse prefill
fallback, sparse decode, speculative verification, and all current CUDA-graph
tiers. The hot consumers never pass a native `torch.float8` pointer to Triton
on Ampere. They load raw bytes, use an exact graph-stable E4M3FN decode table,
and feed BF16 HMMA. Tests cover all 256 E4M3FN encodings, page sizes P2/P64/P128,
negative locations, padding canaries, SWA/C4/C128 parity, indexer boundaries,
and real capture/replay. The final focused qualification passed **33 CPU
dtype/planner/layout tests** and **19 SM86 CUDA tests**. The CUDA suite includes
production 64-head sparse attention, a full 131,072-token C4 scorer, graph
replay with live-length changes, and a writer-to-scorer chain captured in one
CUDA graph.

This applies the storage part of the FP8-on-Ampere blog, but not its literal
integer-MMA path. The blog's own 4,096-square experiment measured the custom
IMMA kernel at 2.914 ms versus 2.267 ms for decode plus FP16 matmul. Local
direct bit decoding was likewise slower than the 256-entry LUT. A second Q/K
integer quantization boundary would also need top-512 and acceptance parity,
so IMMA remains a later experiment rather than the coherent default.

## 524K context and KV-cache truth

The launcher now physically retains `fp8_e4m3` on RTX 3090/SM86. The specialized
writers compute and persist seven per-token UE8M0 scales; the generic warning
that an unspecified checkpoint-level FP8 scale defaults to 1.0 does not fully
describe this DSV4 byte layout. It must not be treated as a quality waiver:
retrieval, reasoning, and agent/tool behavior still need direct qualification
at long context.

The SWA reserve is now 2,560 slots, not the earlier 1,024. With a 256-token
page, 128-token sliding window, and 1,024-token prefill chunk, the admission
floor is 2,304 slots. The old reserve silently limited the practical chunk to
512. The final runtime allocated the full 524,288-token logical pool with the
2,560-slot reserve and all requested graphs.

Allocation is not long-context quality. Retrieval, reasoning, and agentic tool
use at 64K, 128K, 256K, and 524K remain unqualified. Scale calibration,
BF16-overlap comparisons, prefix-cache reuse, and concurrent slot-reuse tests
remain mandatory before relying on the full window on either SM86 or a future
RTX 5090.

## Pre-OSCAR INT4 side track

The requested "OSCAR INT4" has two distinct upstream ideas. OSCAR itself is an
**INT2** KV-cache method with offline covariance-aware rotations, clipping,
and protected BF16 sink/recent windows. Its main implementation targets
full-attention models; DeepSeek-style MLA is not supported there, and its MLA
branch is experimental. The closest published INT4 design is SAW-INT4:
token-wise packed INT4 plus block-diagonal Hadamard rotation. Its released
kernel is also MHA-only and its published setup is H100.

RTX 3090 does support S4/U4 integer Tensor Core MMA, but that does not make
BF16-query by INT4-cache attention an integer MMA. Both integer operands and
scale/zero corrections would be required; SAW's own Triton attention instead
unpacks nibbles and computes in floating point.

The SM86 implementation is now end-to-end rather than an isolated prototype.
Two independent, exact-SM86, opt-in switches preserve the external
`fp8_e4m3` carrier contract while replacing the affected physical pools:

- `SGLANG_DSV4_INT4_KV_STORAGE=1` packs the shared latent used by SWA, C4,
  and C128 attention;
- `SGLANG_DSV4_INT4_C4_INDEXER_STORAGE=1` packs the independent C4
  scorer/indexer pool.

Both switches fail closed on another compute capability or an incompatible
carrier/backend combination. The latent layout is:

| Region | Logical bytes/token | Representation |
|---|---:|---|
| no-PE latent | 224 | 448 signed INT4 nibbles, symmetric per-64 groups |
| RoPE tail | 128 | 64 exact BF16 values |
| scales | 14 | seven BF16 scales |
| **Logical / aligned physical** | **366 / 368** | two untouched pad bytes |

The fused Triton and CUDA-JIT writers support contiguous or scattered writes,
skip negative padding locations, preserve padding canaries, and reuse
caller-owned outputs. Packing and fused dequantization are connected through
the main/SWA writer, both compressor implementations, the C4 indexer, C4/C128
sparse prefill and decode, DSpark writes, speculative verification, debug
dequantization, slot reuse, and CUDA-graph capture/replay.

The independent C4 scorer pool uses the following production layout. Each
P64 page stores 4,096 packed value bytes plus 512 bytes of BF16 per-32 scales:
4,608 bytes/page, or 72 bytes/token, versus 8,448 bytes/page for the packed
FP8 C4 pool. That is a 45.45% C4-capacity reduction. Its persistent Triton
kernel unpacks signed nibbles, decodes the BF16 query path, performs BF16 HMMA,
and replays in a real CUDA graph. The compressor/writer, allocator, indexer,
and model dispatch now all select it under the independent flag.

At the current 131,072-entry C4 capacity, including the allocator's extra
page, that layout recovers **157.5769 MiB/GPU**. Latent INT4 independently
recovers **634.2607 MiB/GPU** at the full 524,288-token allocation; enabling
both recovers **791.8376 MiB/GPU**. The campaign reinvests those exact byte
savings into hash-bound expert-residency plans rather than claiming that INT4
attention itself is faster.

Steady-clock `triton.testing.do_bench` comparisons were shape-dependent:

| C4 scorer shape | INT4 (ms) | FP8 (ms) | BF16 (ms) | INT4 versus FP8 |
|---|---:|---:|---:|---:|
| B1, full 131K | 0.074752 | 0.069632 | 4.819968 | 7.35% slower |
| B1, full 524K indexer stress | 0.235520 | 0.241664 | 19.241983 | 2.54% faster |

The combined kernel/graph/pool suite passes **86 focused tests**. It covers all
four flag combinations, capability and incompatibility gates, both writer
families, negative locations, page edges, slot reuse, C4/C128 attention,
attention sinks, five-row speculative geometry, real CUDA capture/replay, and
exact allocator accounting. The source-level implementation is therefore
complete; the remaining question is promotion, not plumbing.

Promotion is intentionally conservative. Production-shape latent INT4 sparse
decode is still 1.23-1.25x the byte-FP8 latency, so it is a memory-for-residency
experiment rather than a faster attention kernel. Symmetric per-64 latent
quantization is outlier-sensitive because DeepSeek V4 shares the latent as K
and V. The C4 path is lower risk—its normalized H128 rotation precedes per-32
scoring—but still needs full-model top-512 selection recall and long-context
quality. Short OpenCode/tool coherence, throughput, graph safety, and NVLink
traffic are tested in the controlled campaign; retrieval and agent quality at
64K, 128K, 256K, and 524K remain separate admission gates.

## Pre-OSCAR pipeline-parallel concurrency history

Every run in this section predates the Oscar hard switch. They remain useful
scheduler and harness evidence, but they do not count toward the exactly two
Oscar PP2 runs requested after an EP2 winner is admitted.

The PP launcher is
[`scripts/dsv4_flash_hybrid_pp2_dwagon_opencode.sh`](scripts/dsv4_flash_hybrid_pp2_dwagon_opencode.sh).
It transfers the frozen g14-p28 winner into TP1/PP2/EP1. Each stage owns the
union of both EP ranks: 28 GPU and 228 socket-local CPU experts per layer. Run
1 used the recommended 21/22 layer partition, async depth 0, microbatch size 1,
and 1,024-token prefill chunks.

Local SGLang rejects speculative decoding and overlap scheduling when PP is
greater than one, so this path is target-only. DSpark is absent. The corrected
launcher captures full target-decode graphs for scheduler batch sizes 1 and 2.
Prefill is deliberately eager because local DeepSeek-V4 PP prefill-graph
capture pins C4-indexer
scratch and OOMs at the required 524K allocation. Thus the PP experiments keep
CUDA graphs for decode, but cannot be graph-equivalent to TP2/EP2 prefill.
All PP attempts retained packed FP8 cache, a 2,560-slot SWA reserve, the full
524,288-token pool, two request slots, and socket-local CPU offload.

The synchronized harness uses two distinct 2,694-token semantic prompts with
natural EOS, then two distinct forced tool calls. It verifies overlap, output
facts, non-degeneration, exact tool schemas, usage, finish reasons, and stream
completion. During live use it exposed and fixed a harness bug: SGLang's
successful `/flush_cache` response is plain text, while the fake test server
used JSON. Both known success contracts are now accepted; every other response
fails closed. The failed pre-fix attempt sent no model request and is not a
benchmark run.

### Pre-OSCAR PP run 1: 21/22, depth 0, chunk 1,024

| Metric | Result |
|---|---:|
| Raw aggregate decode | **28.372989 token/s** |
| Native lane TTFT | 9.766561 / 5.666195 s |
| Native makespan | 20.575437 s |
| Native semantic success | **1/2**; one required marker omitted |
| Natural stop / stream / nondegeneration | **2/2 pass** |
| Tool calls | **2/2 exact and structurally valid** |
| Tool TTFT | 9.568140 / 9.596337 s |
| Tool makespan | 12.648053 s |
| NVLink counter gate | **all 16 positive** |
| Performance claim eligible | **no** |

The two native request lifetimes overlapped, stopped naturally, and remained
nondegenerate, but later server-log review showed BS1 serialization and the
original receipt did not prove overlapping output windows. One otherwise
coherent lane omitted one of seven required semantic
markers, so the 28.373 token/s result is scheduling telemetry, not an accepted
performance claim or a genuine two-lane throughput result. Tool parsing and
arguments passed on both lanes. The all-16 counter gate also passed, although
the PP traffic pattern is directional and its smallest counter delta was only
2 KiB.

### Diagnostic run 2a: chunk 512 with a missing BS2 graph

The first chunk-512 attempt retained the 21/22 partition, async depth 0,
microbatch size 1, packed cache, and eager PP prefill, but exposed that the BS2
decode graph had not actually been captured.

| Metric | Result |
|---|---:|
| Raw concurrent aggregate decode | **8.396 token/s** |
| Change from run 1 | **-70.4%** |
| Native lane TTFT | 11.385 / 6.202 s |
| Native TTFT spread / makespan | 5.183 / 53.486 s |
| Native semantic success | **1/2**; same marker class omitted |
| Natural stop / other semantic gates | **pass** |
| Tool calls | **2/2 exact and structurally valid** |
| Tool TTFT | 11.450 / 11.480 s |
| Tool makespan | 14.538 s |
| NVLink counter gate | **all 16 positive** |
| Performance claim eligible | **no** |

Chunk 512 made head-of-line behavior and throughput substantially worse and is
not independently interpretable in this run: the raw 8.396 token/s collapse is
a diagnostic for the missing scheduler graph, not the optimized second run.
The same native lane again missed semantic marker 5, while natural termination,
the remaining semantic checks, exact tool structure, and the all-16 NVLink gate
passed.

### Corrected run 2b: chunk 512 with BS1/BS2 decode graphs

Run 2b corrected the graph contract. The live receipt records g28, chunk 512,
full decode graph batch sizes `[1,2]`, FP8 cache, PP2, and both context and total
pool at 524,288. Scheduler logs on both ranks recorded `#running-req: 2` and
`cuda graph: True` during the model window.

| Metric | Result |
|---|---:|
| Raw concurrent aggregate decode / output | **23.881235 / 17.874588 token/s** |
| Native lane decode | 12.721239 / 17.358894 token/s |
| Native TTFT p50 / max | 8.366965 / 10.811133 s |
| Native makespan / output overlap | 23.217319 / **11.118220 s** |
| Native semantic success | **1/2**; marker 5 omitted on one lane |
| Natural stop / stream / nondegeneration | **2/2 pass** |
| Tool calls | **2/2 exact and structurally valid** |
| Tool TTFT p50 | 10.173240 s |
| Tool makespan / output overlap | 13.305596 / **3.030141 s** |
| NVLink counter gate | **all 16 positive**, 2-59,128 KiB |
| Performance claim eligible | **no** |

Run 2b improved raw aggregate decode by 184.44% over the broken-graph run 2a.
It was 15.83% below run 1's raw rate, but unlike run 1 it proved simultaneous
native and tool output windows, so the two rates answer different scheduling
questions. The repeated marker omission still makes `ok=false` and
`performance_claim_eligible=false`; exact tools and genuine overlap do not
waive the native semantic gate.

The checked-in launcher now captures decode BS1 and BS2 but conservatively
retains **21/22, depth 0, microbatch 1, and chunk 1,024**. Chunk 512 is not
promoted on the present evidence. A fresh chunk-1,024 run with the corrected
BS1/BS2 graph set would be required before claiming the best coherent concurrent
throughput for that combined configuration.

The run-2b receipt records this caller-supplied plan path and SHA-256:

```text
/tmp/dsv4-final-pp2.xNWXuS/run2b-cache/transferred-ep2-winner-plans/pp2-ep1-g28-6e9abd43ea3004d4f944d73b296374d2789540c35d45431b7a58871c99d08417.pt
cfac27ea82d3a2825f12332c6f16b6212f04a0ff684360cf1b95e4b0ba385b83
```

Run 2b predates the live loader-digest field. Its historical provenance is
therefore based on that caller-supplied file, the launcher's transfer contract,
and server/log evidence for g28; it must not be retroactively described as
cryptographically plan-bound. The current launcher now hashes the transferred
plan, KTransformers enforces the digest and immutability during load,
`/server_info` exposes the admitted hash, and the PP harness compares it with
the SHA-256 of `--expert-plan`. Future receipts fail if the caller's file and
the plan admitted by the live workers differ.

Reproduce the corrected concurrency gate while the same absolute transferred
plan exists with the now-live digest binding. The current harness is Oscar-only:
it requires an admitted receipt, treats `fp8_e4m3` only as the public raw-byte
carrier, and rejects generic FP8/BF16 history, generic INT4/C4, and selective
C128 compatibility storage before a result can be emitted:

```bash
/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/benchmark_dsv4_flash_pp2_concurrency.py \
  --server-info-url http://127.0.0.1:30010/server_info \
  --request-timeout 900 \
  --run-label pp2-g28-p21-22-a0-m1-c512-bs12-run2b \
  --expert-plan /tmp/dsv4-final-pp2.xNWXuS/run2b-cache/transferred-ep2-winner-plans/pp2-ep1-g28-6e9abd43ea3004d4f944d73b296374d2789540c35d45431b7a58871c99d08417.pt \
  --require-nvlink-traffic \
  --expected-chunked-prefill-size 512 \
  --expected-gpu-experts-per-layer 28 \
  --oscar-admission-receipt "${DSV4_OSCAR_ADMISSION_RECEIPT_PATH:?set admitted OSCAR receipt}" \
  --output-file /tmp/dsv4-final-pp2.xNWXuS/run2b-reproduction.json
```

For historical context, the prior packed g13-era PP runs reported 28.7454
token/s for 21/22 and 28.3101 token/s for a 22/21 split, with 2/2 native and
2/2 tool gates in each receipt. Those placements and harness conditions are
superseded and are not mixed into the g14 comparison. The pre-OSCAR PP2 work
proved real output overlap for agents and tool calls, but neither receipt
passed the full native semantic gate. Before the hard switch, TP2/EP2 was the
only configuration meeting the <=7 s uncached 2,694-token TTFT target.

## fwuff decision

No fwuff CPU offload was added. The local trace shows the target verify path is
the bottleneck, so remote AMX remains technically relevant, but there was no
strong evidence that an InfiniBand request/response plus remote expert work
would beat two socket-local 56-thread pools at the small routed row counts in
this workload. Adding an unproven remote tier would also enlarge the
coherency/failure surface. fwuff's GPU remained untouched throughout.

Remote AMX should be reconsidered only after a request-batched transport
prototype demonstrates end-to-end verify-cycle savings under the same strict
semantic gate. Its CUDA device list must remain empty even then.

## RTX 5090 preparation

The future-hardware track is documented in
[`dsv4flash_5090_readiness.md`](dsv4flash_5090_readiness.md) and checked by
[`scripts/dsv4_flash_rtx5090_preflight.py`](scripts/dsv4_flash_rtx5090_preflight.py).
The read-only preflight correctly blocks today because no 5090 is installed.

The initial candidate is target-only PP3, not TP3/EP3: neither hidden size
4096 nor 256 experts divides by three, the 5090 has no NVLink, and local SGLang
does not yet support DSpark over PP. Admission requires SM86+SM120 builds,
non-NVL NCCL topology qualification, asymmetric per-rank expert budgets,
non-oversubscribed CPU pools, measured PCIe/P2P latency, 524K Oscar-INT2
parity/quality, and fresh semantic/tool qualification. Generic FP8 is not a
fallback. The existing two-3090 launcher remains unchanged until those gates
pass.

## GitHub scheduled-workflow audit

The machine-readable
[`github_fork_scheduled_workflow_audit_2026-08-03.json`](scripts/data/github_fork_scheduled_workflow_audit_2026-08-03.json)
receipt records the final audit at **2026-08-03T19:51:12Z**. Across all **15
forks visible to the authenticated `ldyeax` account**, it found **35
schedule-bearing workflows**. Every one is recorded as `disabled_manually`,
with **0 queued** and **0 in-progress** schedule-triggered runs. The receipt
lists every repository, workflow ID, path, and final state, so this conclusion
does not depend on the narrative summary alone.

The scope limitation is explicit: all 15 visible forks were public, and the
credential cannot prove anything about repositories invisible to that account
or forks owned by another account or organization. The receipt is a timestamped
audit snapshot, not continuous enforcement.

## Pre-OSCAR reproduction record

The commands in this section record the historical packed-cache qualification
procedure. The current launcher additionally requires the Oscar artifact and
admission receipt, and none of these commands can create or waive them.

Run the semantic performance probe only after a successful cache flush:

```bash
curl --fail --silent --show-error -X POST \
  -H 'Content-Type: application/json' -d '{}' \
  'http://127.0.0.1:30010/flush_cache?timeout=30'

/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/benchmark_dsv4_flash_128k.py
```

The repeated cache-state/NVLink control qualification used:

```bash
/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/benchmark_dsv4_flash_hotspot.py \
  --verify-policy 4 --http-only --require-nvlink-traffic \
  --expected-expert-plan /var/lib/exo/cache/dsv4-flash-hybrid-ep2/opencode-g14-p28-frozen.pt \
  --output-file /tmp/dsv4-hotspot-final.json
```

`--expected-expert-plan` is mandatory for a future claim-eligible hotspot
receipt. The launcher hashes the materialized startup plan, KTransformers
rejects a digest mismatch or mutation while loading, `/server_info` exposes the
loader-admitted digest, and the harness hashes the absolute non-symlink file and
requires equality. This binds the measured server to the placement file rather
than trusting a path or run label. The command above deliberately remains the
clean HTTP diagnostic; a claim-eligible receipt must also use instrumented
trace mode and satisfy its trajectory and no-recorder gates instead of passing
`--http-only`.

Run the strict OpenCode gate after every graph, kernel, placement, parser, or
weight-loader change:

```bash
/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python \
  scripts/validate_dsv4_flash_coherency.py \
  --repetitions 3 --validate-tool-call
```

Run the PP concurrency gate only against the PP launcher, using the complete
run-2b command in the PP section above. Do not omit `--run-label`, the absolute
`--expert-plan`, `--require-nvlink-traffic`, or the expected chunk/expert gates;
a bare benchmark invocation does not record enough provenance for comparison.

The pre-OSCAR focused validation ledger was:

| Scope | Result |
|---|---:|
| DSV4 cache dtype/planner/layout CPU tests | 33 passed |
| SM86 packed-FP8 CUDA tests | 19 passed |
| SM86 INT4 storage + C4 PoC tests | 15 passed |
| Hotspot benchmark/controller tests | 27 passed |
| TP/PP OpenCode launcher tests | 55 passed |
| PP concurrency benchmark tests | 30 passed |
| Coherency-validator tests | 12 passed |
| Base launcher + semantic benchmark tests | 20 passed |
| RTX 5090 preflight-v3 and launcher-override tests | 21 passed (16 + 5) |
| KTransformers placement/offload wrapper tests | 45 passed |
| `dsv4_opencode_2694` warmup tests | 2 passed |

Relevant modules also pass `py_compile`, both launchers pass `bash -n`, and
both repository diffs pass `git diff --check`. The repository-wide `uv` check
set could not be installed in this host environment because its lock includes
an architecture-specific MLX dependency; that limitation is not hidden by the
focused results above.

Acceptance rules:

- never use filler, repeated-token output, or forced `ignore_eos` output as a
  performance claim;
- require natural termination, complete output IDs/SSE, semantic facts,
  non-degeneration, and exact forced-tool structure;
- flush before every cold semantic request and every synchronized group;
- record requested and effective KV dtype, actual pool sizes, graph backends,
  and post-capture memory from the live server;
- keep the 524K allocation separate from long-context quality qualification;
- keep fwuff's GPU absent unless the user explicitly changes that scope.

## 2026-08-03 OSCAR-INT2 hard switch

Status: the fail-closed runtime and calibration method are implemented. The
real long-prompt capture, source-ABI-valid statistics, calibrated artifact,
model-bound admission, complete two-GPU model load, exact 524K allocation,
CUDA-graph replay, coherency gate, and three-repetition EP2 baseline have all
passed. Performance optimization remains in progress because the coherent
baseline is 34.420 token/s, not the requested 80/90 token/s.

The cache campaign is **OSCAR-only**. Raw E4M3, signed symmetric INT4, and
generic asymmetric INT2 remain useful controls, but none is an eligible KV
candidate or performance configuration. The serving launchers enforce this at
process startup:

- `SGLANG_DSV4_OSCAR_INT2_KV_STORAGE=1`, exact SM86, the public
  `fp8_e4m3` byte-carrier spelling, a 524,288-token context and total pool, and
  enabled decode CUDA graphs are mandatory. EP2 additionally requires
  graph-backed DSpark drafting and target verification.
- Generic DSV4 INT4, independent INT4 C4, selective-BF16 C128, HiSparse, and
  calibration-capture mode conflict with admitted serving and are rejected.
- The artifact and checkpoint fingerprint must be readable absolute regular
  files and not symlinks. Before launch, `admit` rehashes the complete local
  checkpoint and config and writes a model-bound receipt; workers revalidate
  the artifact, checkpoint, config, fingerprint, receipt, algorithm ABI, exact
  layer coverage, orthogonality, held-out metrics, and non-identity rotations.
- The normal launcher defaults are now populated at
  `/var/lib/exo/cache/dsv4-flash-hybrid-ep2/oscar-int2`: the artifact hash is
  `b7439a92abde1db65673264422eaa1b09762d48243dc98e54789d72939504ed4`
  and the fingerprint hash is
  `ec313105239a17db2487343729769d7ebcc36f4540f16464e8045a5a81436f15`.
  A fresh persistent admission receipt was generated by rehashing the live
  local checkpoint; no copied `/tmp` admission is trusted.
- The lower-level EP2 launcher permits a non-OSCAR process only for an explicit
  unquantized calibration capture. In that mode the Oscar storage flag is off
  and artifact/admission paths must be unset. Such a bootstrap is never a
  serving or performance candidate.
- PP2 requires the same EP2 artifact and admission contract and rejects every
  non-Oscar cache flag. Its exactly two concurrency runs remain gated on an
  admitted, coherent EP2 winner.

### Target and draft cache ownership

Target and speculative workers no longer inherit the same physical compression
topology:

| Worker role | Physical topology | Calibration residency |
|---|---|---|
| EP2 target | Checkpoint ratios retained exactly: layers 0/1 are ratio 0; layers 2..42 contain 21 C4 and 20 C128 layers | All 41 shared-latent maps and exactly the 21 C4 scorer maps are required and retained on the worker GPU |
| DSpark draft | Every local NextN ratio is forced to 0; positive protected SWA capacity; zero C4/C128 cache and state capacity | Full artifact and model admission are validated on CPU, then no target rotation or C4 map is retained |

Any draft compressed capacity, target missing capacity, missing/extra artifact
layer, or draft attempt to access a shared/C4 calibration is a startup/runtime
error. The draft is therefore an Oscar-admitted protected-SWA consumer, not a
silent generic-cache fallback. Target layers 0 and 1 similarly remain their
native ratio-0 BF16 path; they do not receive an identity map disguised as
calibration.

The exact physical layouts are:

| Pool | Codes | Exact/metadata fields | Physical bytes |
|---|---:|---|---:|
| Protected SWA | none | 448 BF16 no-PE values plus 64 exact BF16 RoPE values | **1,024 per row** |
| C4/C128 compressed shared history | 112 B for 448 adjacent-4 asymmetric U2 codes | 128 B exact BF16 RoPE + 28 B interleaved BF16 scale/zero + 4 B untouched alignment | **272 per compressed row** |
| Independent C4 scorer/indexer | 32 B for 128 adjacent-4 asymmetric U2 codes | one FP32 scale/zero pair (8 B) | **40 per row**, or 2,560 B per P64 page |

For compressed target layers, the recent SWA no-PE values are kept in rotated
BF16 while the RoPE tail remains unchanged. Historical rows use the 272-byte
Oscar format. Writers fuse rotation, per-row calibrated clipping, affine U2
quantization, packing, and scatter into caller-owned storage. Attention and
the C4 scorer fuse unpack/dequantization into their consumers and do not create
a cache-sized BF16 workspace. The same address-stable interfaces cover C4,
C128, prefill, decode, target verification, draft extension, negative padded
locations, prefix reuse, and CUDA-graph replay.

### DeepSeek-V4 rotation and objective

DeepSeek V4 needs a model-specific Oscar formulation. Its cached row is a
shared K/V MLA latent: 448 non-positional channels plus 64 RoPE channels. Two
independent Oscar K and V rotations would therefore be incoherent. The
artifact instead carries one covariance-trained orthogonal `R[448,448]` for
each compressed layer. The runtime contract is:

```text
cache_nope = latent_nope @ R
query_nope = query_nope @ R
output_nope = rotated_attention_output_nope @ R.T
rope64 = exact BF16, unchanged
```

The corrected 448-dimensional construction is
`R448 = U_V/SST @ Pbr @ blockdiag(H64 x 7)`. Placing the balancing permutation
before the seven independent H64 blocks distributes high-energy
eigendirections across quantization groups before each group is mixed. The
earlier `U @ blockdiag(H64 x 7) @ Pbr` order only permuted already isolated
mixtures and is retired. This is a true 448-dimensional orthogonal transform;
there is no cropped 512-wide approximation.

The C4 scorer is a complete power-of-two domain and retains the official
ordering: `R128 = U_K/QQT @ H128 @ Pbr`. Its map, clipping policy, and physical
pool are independent of the shared-latent map.

The shared rotation, clip sweep, and unrotated comparison are fit from Oscar's
normalized attention-weighted value covariance, the V/SST objective. Because
DeepSeek V4 shares the stored latent between K and V, admission remains stricter
than the fit: held-out reconstruction is scored equally against query QQT and
value SST sensitivity. Thus moving from an equal QQT/SST spectral fit to the
official V/SST fit is a source-faithfulness correction, not a weakened quality
gate. C4 continues to fit and validate against its query QQT objective.

### Real long-prompt capture

The deterministic corpus is
[`dsv4_oscar_int2_prompts_long.json`](scripts/data/dsv4_oscar_int2_prompts_long.json),
SHA-256
`faada3ce6083b110c513f2370586503140c89d3741da328bfafffc1a6e501434`.
It contains 12 content-addressed prompts: eight train and four held out, with no
text crossing the split. The corpus contains 13,172 checkpoint-tokenizer tokens;
individual prompts range from 1,069 to 1,127 tokens.

The calibration-only TP2 capture completed all 12 requests. The runtime emitted
424 non-empty raw tensor-state files totaling 177,609,208 bytes. Finalization
retained 338 referenced tensor files in 86 layer/split chunks plus the manifest.
The finalized capture manifest SHA-256 is
`7f8722559caa56cffb250ddf335919057891a6092bb2f15fcada5a9917ae6f82`.
It is bound to checkpoint fingerprint SHA-256
`ec313105239a17db2487343729769d7ebcc36f4540f16464e8045a5a81436f15`,
checkpoint SHA-256
`2f21b2a5cb30a5200b9cb18ca3c97b7e54c322b42827ad489ffc106d66240035`,
and config SHA-256
`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`.

The first compressed-history-only statistics reduction had file SHA-256
`50a059a0a31d1a201714ee68029ab961950a70577c2fdd3c5fc36049d3e774cb`
and canonical tree SHA-256
`f22128308f886f239978cae51187efe1f881489feff6ac8a995a5eaa92ab1af1`.
It covered exactly layers 2..42 and excluded protected SWA rows. Aggregate
compressed-latent rows were 22,784 train and 11,392 held out. Every C4 layer
had 1,024/512 train/held-out rows and every C128 layer had 64/32, all above the
immutable 32/16 minima. These counts matter: the earlier short capture had
only 8/4 C128 rows and could not qualify a rotation.

That statistics file is diagnostic evidence, not the final artifact. The
algorithm/source metadata was subsequently frozen around the official V/SST
objective, so the updated loader intentionally rejects statistics from the old
ABI. Regeneration from the same provenance-bound capture produced
`statistics-sst-v2.pt`, file SHA-256
`59a13ae0dc354489f68e1c22a65289e09602a50e54cf94b1ea94ef0758cc61e1`
and canonical statistics SHA-256
`c3a4b4f2bc90850346eb00ba943a52b74c950371805663d2f6f8b547c19e65e0`.
It preserves the exact 41-layer coverage and 22,784/11,392 sample-row counts.

### Objective-gate evidence

The initial equal QQT/SST spectral objective passed 61 of 62 domains: all 21
C4 maps and 40 of 41 shared maps. Layer 41 alone failed the held-out improvement
gate. Its train joint error improved from `0.0009313186` unrotated to
`0.0008606442` rotated (**+7.5886%**), but held out regressed from `0.0008573615`
to `0.0008911339` (**-3.9391%**). Orthogonality
(`1.1887329565e-08`) and the absolute error gate passed; the improvement gate
alone rejected it. All C4 domains passed, with minimum held-out improvement
**+27.8439%**.

A source-faithful diagnostic scan then fit the shared rotation, clipping, and
unrotated baseline on normalized attention-weighted V/SST covariance while
retaining the equal held-out QQT/SST admission metric. It passed **41/41**
shared layers with the original gates: minimum improvement **+20.3748%** at
layer 40, median **+78.2372%**, mean **+68.2712%**, and maximum held-out joint
error `0.042418677`. Layer 41 improved from a V/SST-trained unrotated
`0.000851335` to `0.000130243` (**+84.7013%**). This scan validates the
principled objective choice; it is not itself an admitted serving artifact.

### Admitted artifact

The source-ABI-valid calibration produced artifact format v2 with the frozen
algorithm
`oscar-dsv4-compressed-history-shared-v-sst-u-pbr-h64x7-c4-k-qqt-u-h128-pbr`.
Its file SHA-256 is
`b7439a92abde1db65673264422eaa1b09762d48243dc98e54789d72939504ed4`
and its canonical provenance-tree SHA-256 is
`fab4ac1a7daa8547272e7fc90e8d4c29b5f35cc94c9d37c6792ddf9d5cf5ed86`.

All **41/41** shared maps pass: minimum held-out improvement **+20.3748%**,
median **+78.2372%**, and maximum held-out joint error `0.042418677`. All
**21/21** C4 maps pass: minimum improvement **+27.8439%**, median **+54.6270%**,
and maximum held-out joint error `0.046637440`. This is exact required
coverage; layers 0/1 are deliberately absent rather than supplied with identity
maps.

Full-checkpoint admission returned `admitted=true` using policy
`rehash-config-index-and-all-referenced-shards-v1`. The receipt file SHA-256 is
`f0f0501b917bf0242dd2fea632d7d5d8158861d36198781da97504d6e5d98323`
and its canonical admission SHA-256 is
`51bfae6e89bce7d37376554da1ce584280e7f736c89bc3a28c2c4743a0c13fe9`.
It binds the artifact to the checkpoint, config, and fingerprint hashes listed
above. That offline proof is now supplemented by the model-level serving
receipt described below; neither proof alone substitutes for the other.

### Measured EP2 baseline and remaining optimization

The complete g14 EP2 baseline receipt is
`/tmp/dsv4-oscar-int2/campaign-v1/g14-oscar-int2-baseline/baseline.json`,
SHA-256
`e4bf83a681f658ef5aa039ac79dde899c5cf6f5bab20a2ebf3d48ce4d2469d23`.
It proves the artifact/admission hashes and target/draft worker roles, the
1,024/272/40-byte physical layouts, the full 524,288-token allocation,
target/draft CUDA graphs, strict semantic and forced-tool coherence, both local
GPUs, both SM86 small-batch workers, and positive payload deltas on all local
NVLink counters. Natural-stop outputs followed different valid trajectories,
so the campaign reports an unpaired five-phase ensemble rather than making a
paired page-locality claim.

| Oscar-only configuration | Mean decode | Median decode | Maximum flushed TTFT | Mean target verify | Physical free/GPU | Decision |
|---|---:|---:|---:|---:|---:|---|
| g14 baseline, 3 receipts | **34.420** | **34.392** | **5.303 s** | **64.797 ms** | **4,619 MiB** | coherent reference |
| +86 expert slots/rank | 34.233 | 33.925 | 5.189 s | 61.845 ms | 3,497 MiB | reject: lower acceptance erased verify gain |
| +172 expert slots/rank | 33.462 | 33.289 | 5.362 s | 62.592 ms | 2,377 MiB minimum | reject: slower and less headroom |

The residency screens are important negative results. More GPU-resident
experts did shorten target verification by 3.4--4.6%, but committed tokens per
cycle fell from 2.446 to 2.330 and 2.300, so end-to-end decode regressed. The
campaign therefore does not spend another full launch on k86/k172
confirmation.

At the observed 2.446 committed tokens per cycle, 80 and 90 token/s require
complete cycles of at most **30.57 ms** and **27.17 ms**. Even perfect fixed-4
acceptance would require at most 50.0 and 44.44 ms, respectively, while the
baseline target verification alone is 64.797 ms. The next candidates therefore
remove work from that stage rather than merely increasing cache capacity:

- skip masked non-boundary Oscar C4/C128 writer work during graph replay;
- rotate each C4 query once into caller-owned stable workspace instead of once
  per active page program;
- absorb the shared-latent inverse output rotation into each admitted target
  layer's output projection, eliminating the runtime restore kernel;
- replace the separate CPUInfer distributor with the pinned `TaskQueue` as
  logical worker zero. A same-binary checkpoint sweep selected 56 workers and
  a 1,000-us spin window: the fixed-4 route-weighted median fell from
  2,665.340 us to 1,200.465 us (**2.2203x**) with identical M1--M6 output
  hashes. Both tested 72-worker policies were slower and are rejected;
- evaluate an opt-in MXFP4 E8M0 scale fold only after whole-buffer admission
  and numerical parity. The full checkpoint contains 8,657,043,456 scale
  bytes, all in the safe 118--126 domain. The first two alternating N-block-128
  LUT pairs improved fixed-4 route-weighted latency by 20.84% and 22.91%; every
  M1--M4 arm improved, all outputs were bit-identical, and all admission and
  fallback counters were clean. The N-block sweep and full model run remain
  separate gates;
- A/B one transactional host `performance` governor/EPP policy around a full
  campaign stage, with exact restoration evidence; and
- publish exact rank-local inline-dispatch and all-worker affinity readback so
  a serving result cannot claim the optimization from launcher intent alone.

Long-context Oscar quality remains a separate gate beyond allocation and the
short coherency suite. After a coherent EP2 winner is selected, and not before,
exactly two PP2 concurrency runs will be performed: one direct transfer run and
one optimization run informed by the first.

## 2026-08-04: combined EP2 winner and post-warmup bottleneck trace

The CPU-inline/N128-LUT combination is now a confirmed Oscar-only EP2 winner.
The clean confirmation receipt is
`/tmp/dsv4-oscar-int2/campaign-optimized-v1/g14-oscar-int2-cpu-inline-scale-lut-n128/confirm.json`.
Across its 15 natural-stop phases it measured **39.196 token/s mean** and
**38.821 token/s median**, with **5.592 s median** and **5.691 s maximum**
flushed TTFT. Mean target verification was **54.503 ms**, committed tokens per
cycle were **2.374**, and both GPUs retained **4,619 MiB** of physical free
memory after graph capture. The screen and confirmation together improve mean
decode by **1.1212x** over the same-campaign baseline; the whole-receipt
bootstrap 95% interval is **[1.0850, 1.1580]**. All semantic, OpenCode tool,
Oscar admission, graph, memory, worker, and interconnect gates passed.

The exact native artifact is
`/var/lib/exo/experiments/dsv4-cpu-inline-scale-lut-n128-v1/lib/kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so`,
SHA-256
`7886a0e7cde36263ac57005aea572fd99a401a8b3107f60d949d0dff97292043`.
Its serving tuple is CPUInfer 56, worker spin 1,000 us, task-queue pinning on,
single-NUMA inline dispatch on, `lut-v1`, and compile-time N-block 128. The
launcher now fails closed on that complete tuple rather than treating the
shared-object path as sufficient provenance.

A separate five-phase, repeated 2,694-token hotspot trace is frozen at
`/tmp/dsv4-oscar-int2/kt-combined-nondeep-v2-hotspot.json`. All five outputs
passed the semantic contract and averaged **39.839 token/s**, **55.247 ms**
target verification, **6.247 ms** draft work, and **2.445 committed tokens per
cycle**. Natural-stop trajectories differed, so this receipt is diagnostic and
does not make a paired page-locality claim. Exact radix reuse saved 4.953 s of
TTFT, while the near prompt did not reuse the mutated middle region.

NVLink is conclusively active. Every RX and TX counter on all four links of
both 3090s advanced during the trace; individual deltas were between
**2,618,773 and 2,618,966 KiB**. The topology is NV4 with CUDA peer access and
CUMEM enabled. A separately measured 48 KiB graph all-reduce is about 10.3 us,
and even a pessimistic accounting leaves collective time below 0.9 ms per
cycle. Inter-GPU fallback is therefore not the decode plateau.

The non-deep hybrid timing stream contained 178 small-token observations.
Their mean CPU wait was only **0.045 ms**; the wrapper's CPU wait is already
hidden and cannot supply the roughly 31 ms/cycle still needed for 80 token/s.
At the confirmation acceptance, 80 token/s requires a cycle no longer than
29.675 ms and target verification no longer than about 23.148 ms. Perfect
fixed-4 acceptance at the current cycle time would still cap decode near
65.5 token/s. The next source candidate therefore attacks the target GPU
hotpath: the H64 Oscar decoder currently launches only 8 CTAs at T=1 and 40
at the observed T=5 verify shape on an 82-SM GPU, while its compiled kernel
uses roughly 189 registers/thread and 40 KiB of dynamic shared memory.

The implementation under qualification splits only Oscar compressed history
into multiple contiguous ranges, emits sink-free FP32 online-softmax partials,
and combines them in deterministic split order with one sink owner. It uses a
fixed, backend-owned 4,210,688-byte arena whose address survives CUDA-graph
capture. Large prefill and unsupported shapes keep the existing monolithic
kernel. C4 and C128 are benchmarked separately before a full launch so the
short C128 history cannot be regressed merely to improve the 512-row C4 path.

## Online references and how they affected this work

- [Official DeepSeek V4 Flash DSpark configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/config.json): block 5, target layers 40/41/42, Markov rank 256, SWA 128.
- [Official DeepSeek V4 Flash DSpark model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark): the current vLLM recipe recommends greedy DSpark with seven speculative tokens on a 4x GB300 system. That is a starting geometry, not a transferable optimum: the local fixed-4 sweep won on two 3090s because each extra target-verification row is dominated by the CPU expert tail.
- [Official SGLang DeepSeek V4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4): `deepseek-v4` reasoning parser and `deepseekv4` tool parser.
- [SGLang DSpark integration report](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/): compact verification is mainly a high-concurrency win; at batch 1 it tied static in their study, and zero-overhead scheduling/overlap is the relevant implementation direction.
- [DSpark paper](https://arxiv.org/abs/2607.05147): 60-85% relative per-user improvement over MTP-1 at matched throughput, not an absolute two-3090 speed promise.
- [FP8 as storage with IMMA on Ampere](https://amohan.dev/blog/2026/fp8-as-storage-imma-ampere/): raw-byte storage is portable to SM86, but its own IMMA result did not beat decode plus FP16 matmul; local DSV4 therefore decodes to BF16 HMMA.
- [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): MXFP4 uses E2M1 elements in groups of 32 with a shared E8M0 power-of-two scale. This is the contract behind the opt-in CPU scale-fold work; unsafe scale encodings are rejected for the whole loaded buffer rather than mixed into a hot-loop fallback.
- [NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/): confirms S4/U4 IMMA support, while not implying that mixed BF16-query/INT4-cache attention can use it directly.
- [OSCAR project](https://oscar-quantize.github.io/) and [official repository](https://github.com/FutureMLS-Lab/OSCAR): OSCAR is calibrated INT2 KV, and its main path does not support DeepSeek-style MLA.
- [SAW-INT4 paper](https://arxiv.org/abs/2604.19157) and [official repository](https://github.com/togethercomputer/saw-int4): motivated the isolated signed-nibble and H64 proof, but its released serving kernel is MHA-only.
- [Independent shared-expert loader report](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/DSPARK-SHARED-EXPERT-FIX.md): missing draft shared experts can preserve fluent output while destroying acceptance and speed.
- [Independent cold-agent garble report](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/AGENT_GARBLE_FIX.md): cold, cache-busted agent contexts are required for qualification.
- [Independent 2x DGX Spark results](https://forums.developer.nvidia.com/t/deepseek-v4-flash-dspark-on-2x-dgx-spark-gb10-big-single-stream-speed-boost-60-67-tok-s-1m-context-now-with-concurrency/374846): useful evidence for content-dependent acceptance, not a directly comparable SM86 benchmark.

## 2026-08-04: final Oscar split-history EP2 result

The split-history kernel is implemented and is now mandatory in the production
OpenCode launcher, `scripts/dsv4_flash_hybrid_ep2_dwagon_opencode.sh`. The
launcher admits no generic FP8, generic INT4, or selective-BF16 history cache.
Its public `fp8_e4m3` value is only SGLang's raw-byte carrier; the physical C4
and C128 history is calibrated asymmetric Oscar INT2 at 272 bytes/token, the
Oscar C4 scorer is 40 bytes/token, and only the protected 1,024-token SWA
reserve remains BF16 at 1,024 bytes/token.

The SM86 implementation splits only compressed history. Stage 1 produces
sink-free FP32 online-softmax partials, and deterministic stage 2 combines
them in split order and owns the sink exactly once. The split map is
`{1:16, 2:16, 3:8, 4:4, 5:4, 6:4, 7:4, 8:2}` and each worker owns a fixed
4,210,688-byte workspace whose address is stable through graph replay. Large
prefill remains on the existing monolithic path. EP2 keeps breakable prefill
graphs and full decode/speculative graphs, exact 524,288 context and token
limits, CPUInfer 56, both 3090s, and the confirmed inline-dispatch/N128-LUT
native artifact.

The CUDA-graph microbenchmark receipt is
`/tmp/dsv4-oscar-int2/oscar-split-decode-bench-v1.json`, SHA-256
`0851764cba5a6b13d9bf5a201c6df34be9c8cd0a146701f8c444e069f82df2cf`.
All six shapes passed numerical parity. At one token, C4-512 improved
0.176888 to 0.038000 ms (**4.655x**) and full C128-4096 improved 1.146568 to
0.124976 ms (**9.174x**). At five verification rows, C4-512 improved 0.175456
to 0.070464 ms (**2.490x**) and full C128-4096 improved 1.161092 to 0.435820
ms (**2.664x**). The route-weighted 41-layer estimate improved 3.342x at one
row and 2.239x at five rows; these are kernel measurements, not model-rate
claims.

The independent confirmation receipt is
`/tmp/dsv4-oscar-int2/split-history-ep2-confirm-hotspot.json`, SHA-256
`7c0342777fafec5e1ab270011675af6bae09b4ffd3007390f2e7013e71a0d7c4`.
Its five natural-stop decode rates were 43.528, 47.434, 45.986, 41.765, and
44.735 token/s: **44.690 mean** and **44.735 median**. Fresh 2,694-token TTFT
was **5.542 s median** and **5.624 s maximum**. Mean phase target verification
was **47.635 ms** with **2.402 committed tokens/cycle**. This is 14.02% faster
than the 39.196-token/s combined CPU confirmation and 29.84% faster than the
original 34.420-token/s Oscar baseline. Every phase passed semantic validation
and every one of the 16 NVLink counters advanced by about 2.621 GiB. The
top-level paired-locality attribution flag remains false because natural-stop
trajectories changed; it must not be used to claim a causal page-warmth speedup.

The separate OpenCode receipt is
`/tmp/dsv4-oscar-int2/split-history-ep2-confirm-coherency.json`, SHA-256
`763cd28ffb4eb21008d8cc63692d4a201dfaef70ef9a8cc263780b805063c471`.
All semantic repeats and the exact forced tool call passed, with deterministic
final content. Its realistic 42,125-byte system prompt took about 22--24 s to
first output, so the <=7 s result applies to the controlled 2,694-token
qualification prompt, not a full 42K OpenCode context.

The final local result therefore passes the Oscar-only, coherence, CUDA-graph,
524K, two-GPU, CPU-offload, and short-prompt TTFT gates, but it does **not**
reach the 80 token/s objective or 90 token/s stretch objective. Target
verification remains the bottleneck. CPU queue wait and NVLink have both been
measured and excluded as primary causes.

## Exactly two PP2 concurrency launches

The EP winner was transferred into PP2/EP1/TP1 with the same Oscar artifact,
physical cache, split-history kernel, native CPU artifact, 56 CPU workers,
exact 524K limits, and full BS1/BS2 decode CUDA graphs. Prefill graphs were
disabled for PP memory safety. Both runs issued two synchronized native
requests and two synchronized OpenAI tool-call requests, required natural
termination and output overlap, and measured all 16 NVLink directions.

| PP2 launch | Layer split | Native aggregate decode | Native output throughput | Native TTFT p50/max | Tool throughput | Qualification |
|---|---:|---:|---:|---:|---:|---|
| 1, direct transfer | 21/22 | 27.844 token/s | 20.265 token/s | 8.234/10.514 s | 18.270 token/s | reject |
| 2, optimized | 22/21 | **28.606 token/s** | **20.960 token/s** | 8.214/10.544 s | **21.251 token/s** | reject |

Run 2 changed exactly one knob: it moved one target layer from the stage that
also owns the output head to the other stage. Native aggregate decode improved
2.74%, output throughput 3.43%, and tool throughput 16.31%. Both tool lanes in
both launches were structurally exact, all streams completed naturally with
real overlap, and all 16 NVLink deltas were positive. However, lane 1 omitted
the required phrase `cache isolation` in run 1, an identical deterministic
repeat, and run 2 (`missing_semantic_marker_5`). Both benchmark receipts are
therefore `ok=false` and `performance_claim_eligible=false`; PP2 also misses
the <=7 s concurrent TTFT target and is not a production candidate.

The run-1 receipt is
`/tmp/dsv4-oscar-int2/pp2-transfer-run1.json`, SHA-256
`f405249899bd2caad34a8d05b99ee05afdd2ae135386d610fcf54cd6d5141c52`.
The run-2 receipt is `/tmp/dsv4-oscar-int2/pp2-optimized-run2.json`, SHA-256
`ea26c8bf748c7993bae18ffb0c645986711f3f68337c5cdd63fa57d52bfc192b`.
The canonical ledger is
`/var/lib/exo/cache/dsv4-flash-hybrid-pp2-opencode/pp2-two-launch-ledger.json`,
SHA-256
`a73185001d79717996f628cd26fd4d770aa9816e0be0ee789f14f8c80887f3b8`;
it records exactly two authorizations and is exhausted. Both host-policy
receipts independently prove performance-policy application and exact
restoration across all 224 CPUs. Their top-level `accepted=false` values are
solely the expected result of terminating the serving children with signal 9
after evidence collection, not a claim of graceful server exit.

A final adversarial audit also closed two controller gaps without another model
run. The PP2 entrypoint now pins the canonical cache, ledger, and role-specific
authorization paths, rejects a supplied old authorization, and replaces its
public recursive-shim switch with an inherited, unlinked, mode-0600 one-shot
capability. Before ledger authorization, the shim revalidates the effective
Oscar-only environment, exact 524K limits, PP topology, graph contract, and
final CLI, including duplicate and alias rejection. The focused launcher suite
has 44 passing tests covering namespace rotation, direct shim entry, reusable
authorization, and post-token CLI/environment tampering.

No third PP2 model launch is permitted through this campaign entrypoint. This
is deliberately not a host-wide security boundary: a root operator can invoke
SGLang or a generic lower-level launcher directly. The coherent EP2
split-history launcher remains the handoff configuration.

## 2026-08-04: graph-safe timing, fused-path trials, and final verify sweep

This pass kept the production constraints fixed: local EP2/TP2 on both RTX
3090s, CPUInfer offload, PP1, exact 524,288 context and token limits, Oscar
INT2 physical history, breakable prefill CUDA graphs, and full target/draft
verification graphs. No PP2 or remote-host run was performed.

### Internal graph-safe timing

`SGLANG_DSV4_INTERNAL_TIMING=1` now enables category timing for both target and
draft execution without allocating, synchronizing, or reading events inside a
captured graph. Persistent CUDA events are recorded into fixed slots during
replay and resolved only at the DSpark cycle snapshot boundary. The categories
are attention/indexer, routed MoE, shared MoE, projections/norms, collectives,
and final head. Timing is diagnostic-only and remains disabled by default
because the extra event recording reduced observed decode to roughly 38--39
token/s.

The diagnostic receipt is
`/tmp/dsv4-oscar-int2/timing-v1/hotspot.json`, SHA-256
`9a78d2d07cb4574f1cfc4376927d02f21f69dd022a354f183ecc2db046389d4f`.
A representative cold fixed-4 cycle attributed the following mean GPU time:

| Target category | Mean GPU time |
|---|---:|
| Routed MoE | 32.278 ms |
| Projections/norms | 10.002 ms |
| Collectives | 7.584 ms |
| Attention/indexer | 5.288 ms |
| Shared MoE | 1.971 ms |
| Final head | 0.726 ms |

The draft adds 3.894 ms of routed MoE and less than 2.1 ms across its other
categories. Target routed MoE is therefore the largest remaining measured
component, not Oscar attention, CPU queue wait, or the final head.

### Fused T≈5 MoE trial

`SGLANG_V4_MXFP4_FUSED_T5_MOE=1` enables an experimental small-row path. It
interleaves gate/up W13 rows and E8M0 scales, applies the activation in the W13
kernel, keeps W2 scatter/gamma/merge caller-owned, and fuses the KT
logical-to-local route mask/remap into the small routing kernel. The production
`Mxfp4TritonKernelsMoEMethod` adapter was fixed to carry these capabilities;
previously only the unused `DeepSeekMxfp4MoEMethod` adapter supplied them.
Live two-rank proof after full graph capture recorded 46 conversions per rank,
1,342 fused applies per rank after warmup, and 826 KT-fused routing applies per
rank. Minimal coherence passed.

The isolated receipts are
`/tmp/dsv4-oscar-int2/moe-t5-baseline-bench-v1.json` and
`/tmp/dsv4-oscar-int2/moe-t5-fused-bench-v1.json`, SHA-256
`9eb40b819954533d1db9ca360c4b659324f44b2326960a2ce0ea50757b5f5d9d`
and
`6973ff6c388698f5e1ad8b902b2d7515d887cfd7336769531eb8eca3e71b908c`.
At T=5/E=14, one-, two-, and three-route microcases improved by 7.5%, 2.4%,
and 2.8%, respectively. The full-model candidate nevertheless regressed: the
receipt
`/tmp/dsv4-oscar-int2/fused-t5-c4-v1/production-fixed4-v2.json`, SHA-256
`1adc9f4ec1be93ae3ea9010bf76070a493f190880e810ca5bca279a0984f25d3`,
measured **41.069 committed token/s** from 1,138 committed tokens over
27,709.387 ms of whole GPU cycles. That is 7.3% below the reconstructed
44.319-token/s coherent split-history reference. The path remains opt-in and
is not a production default.

### C4/Oscar fusion trial

`SGLANG_DSV4_OSCAR_FUSED_C4_PIPELINE=1` enables an experimental C4 pipeline.
Exact global top-k remains a synchronization boundary: no exact implementation
can feed the final attention consumer until every active C4 page has
contributed to top-k. The selected hybrid therefore bypasses RoPE and scoring
entirely when the static sequence length is already at or below top-k, while
retaining the original narrow RoPE plus Oscar scorer for longer history. The
short path improved the whole C4-plus-top-k CUDA graph by about 2.25--2.58x,
or only 0.0073--0.0092 ms per affected layer. A fused long-history RoPE kernel
was 7--9% slower and was rejected.

The rejected long-RoPE and selected-hybrid micro receipts are
`/tmp/dsv4-oscar-int2/c4-fused-rope-rejected-bench-v1.json` and
`/tmp/dsv4-oscar-int2/c4-hybrid-pipeline-bench-v1.json`, SHA-256
`b82ccd0767c157531cc980e779db09efc54cfd637e69223d29cd349f8716b3a5`
and
`e1b6327282aed996f4f90c758caa47f3ead21c1a85c848b0f55e01dc6d1ef0d4`.
The 2,694-token production workload cannot use the <=512-token bypass. Its
C4-only receipt,
`/tmp/dsv4-oscar-int2/c4-only-v1/production-fixed4.json`, SHA-256
`17f5e59972f7c6cb8b87b3a4f5e1e02c0f65ed16bffe83c31a19841c31c65c31`,
measured **42.494 committed token/s**, 4.1% below the same split-history
reference. This path also remains opt-in and disabled in production.

### Whole-cycle DSpark verify-length sweep

The final sweep used the uninstrumented, unfused Oscar split-history baseline.
Every tier used the same 2,694-token exact/near prompt pair, a 256-token cap,
greedy sampling, natural EOS validation, and five phases. The selection metric
is the requested aggregate: total committed tokens divided by the sum of all
whole GPU cycle time, never the mean of per-phase rates.

| Verify length | Committed tokens | Cycles | Whole GPU time | Objective | Max flushed TTFT |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,110 | 645 | 31,144.008 ms | 35.641 token/s | 5.198 s |
| 3 | 1,133 | 534 | 28,334.897 ms | 39.986 token/s | 5.209 s |
| **4** | **1,156** | **476** | **26,887.047 ms** | **42.995 token/s** | **5.203 s** |
| 5 | 1,130 | 462 | 27,574.997 ms | 40.979 token/s | 5.189 s |
| 6 | 1,135 | 454 | 29,767.433 ms | 38.129 token/s | 5.196 s |

The tier receipts are
`/tmp/dsv4-oscar-int2/final-baseline-v1/verify-{2,3,4,5,6}.json`; their
respective SHA-256 values are
`a158191e1c0c9682474d5ad1936d5643be71bc078c353c9ab6e29d00a5d587da`,
`99785d54aa3f256ac2676ff6992071e919e83997e654cfe32abc9fb5dceed090`,
`5377d32266796bab48b6d5e69aae96e3f21e85dd955eae15b7bb43b5544573b2`,
`39efa6cb347f639acfcc4e8b0f1aed9522d6b8a243aa1676e184c91d5c8fdeb5`,
and
`989c031e27829fe4de9e9e3bfb8926088e4e007cfb17d078bcbc502b5609406e`.
Fixed-4 remains the launcher default. Its five HTTP decode rates average
43.425 token/s, and its controlled fresh-prompt TTFT remains below 7 s. The
canonical selected receipt is
`/tmp/dsv4-oscar-int2/final-baseline-v1/production-selected.json`, SHA-256
`5377d32266796bab48b6d5e69aae96e3f21e85dd955eae15b7bb43b5544573b2`.

### Final OpenCode follow-up qualification

The final retained server exposed 524,288 context and total tokens, FP8 as the
public byte carrier, the admitted Oscar INT2 split-history execution on both
ranks, EP2/TP2/PP1, full decode/speculative graphs, breakable prefill graphs,
and 5.06 GB free after graph capture. Both experimental fused flags and
internal timing were disabled.

The realistic receipt is
`/tmp/dsv4-oscar-int2/final-baseline-v1/opencode-followups.json`, SHA-256
`88ecc3c85e9bd3567420f9aa5c58ae1462f7b46ab366f995d242f2e10daeed8d`.
Three separately cache-flushed 42,125-byte OpenCode semantic requests produced
the exact expected content and reasoning hashes. The standalone forced tool
call passed. A retained-context sequence then passed its initial semantic
answer, forced follow-up tool call, and exact post-tool-result continuation;
the latter two reached first output in 2.180 and 2.183 s. The full 42K initial
requests took roughly 20.45--21.91 s to first output, so they do not contradict
the <=7 s controlled 2,694-token TTFT claim.

The requested 80/90-token/s goal is not met. The graph-safe trace now localizes
the plateau: roughly 32 ms/cycle remains in target routed MoE, followed by
about 10 ms of projections/norms and 7.6 ms of collectives. The two new fused
ideas are retained as tested opt-ins but rejected as defaults because the
model-scale receipts, not their microbenchmarks, regressed the whole-cycle
objective.

## 2026-08-14: prediction-acceptance audit

### Interpret the current rate against the cap

SGLang's logged `spec_accept_rate` is a strict generated-draft yield, not the
fraction of the scheduled verify window that succeeded. For this launch,
`speculative_num_draft_tokens=6`, so every round generates five draft tokens
and one target bonus token. The implementation computes

```text
correct draft tokens / (verify rounds * 5)
```

while `spec_accept_length` includes the bonus token. Fixed verify length 4 also
includes that bonus slot, so it can commit at most three correct drafts per
round. The logged rate therefore has a hard ceiling of `3 / 5 = 0.60` even if
every draft token that is allowed into the verify window is correct.

Under that definition, a logged rate of 0.34--0.44 means approximately
1.70--2.20 correct drafts and an acceptance length of 2.70--3.20 tokens per
round. Relative to the three draft positions the fixed-4 policy can use, that
is roughly **57--73% scheduled-window utilization**, not 34--44%. End-of-stream
rounds can perturb the request-level identity slightly, but not the metric's
meaning or ceiling.

This also reconciles the local number with one useful informal comparison. An
independent, patched 2x DGX Spark deployment reported unconditional
per-position survival of `0.826/0.725/0.572/0.471/0.399`, which sums to a 0.599
full-five-draft yield. If only its first three positions could be committed,
the same data would appear as `(0.826 + 0.725 + 0.572) / 5 = 0.425` in the
current strict logger. The report also measured 68.7% on code but only 33.7% on
prose reasoning, confirming that traffic mix can move the headline more than a
small kernel change.

Future receipts must report all of these separately:

1. generated-draft yield: correct drafts divided by all five generated drafts;
2. scheduled-window utilization: correct drafts divided by the sum of
   `verify_len - 1` over rounds;
3. untrimmed block survival and per-position conditional survival from a
   diagnostic `cap-accept` run;
4. mean useful committed tokens per whole GPU cycle; and
5. useful decode token/s, TTFT, and end-to-end latency.

There is no universal 50% break-even threshold. Whether speculation pays is a
hardware- and workload-specific comparison between target-only token time and
the complete draft-plus-verify cycle. The missing paired experiment is planned
in [`dsv4f_no_dspark.md`](dsv4f_no_dspark.md).

### Longer verification was not the coherency failure

The retracted 70+ token/s campaign combined three separate problems: the
partial prefill graph wrote padded rows repeatedly into KV slot 0; the harness
used repeated filler, forced 512 tokens with `ignore_eos`, hid the generated
text, and lacked a semantic gate; and the campaign changed the checkpoint's
trained block-5 geometry to block 6 or 8. Target verification normally protects
token correctness, but it could not make that corrupted prefill state or the
non-semantic benchmark into a valid performance result.

The repaired semantic sweep has already tested verify lengths 2 through 6.
Lengths 5 and 6 were coherent; they simply lost to length 4 on whole-cycle
throughput by 4.7% and 11.3%, respectively. Thus increasing the current cap
from 4 to 5 or 6 needs only a launcher change, graph/receipt recapture, and full
qualification--not an architectural rewrite--but the existing evidence says
not to do it. Six is the natural maximum for the checkpoint's trained block of
five drafts plus one bonus. Going beyond six would require a trained wider
drafter or a multi-block/chained-draft runtime, new confidence calibration, new
graphs, and complete requalification. The old untrained block-size override is
not an acceptable shortcut.

### Ranked next experiments

1. **Measure target-only break-even first.** Run the fail-closed A/B/A plan in
   `dsv4f_no_dspark.md` after the live server is released. Keep Oscar,
   TP2/EP2/PP1, g14-p28, graphs, cache, prompts, and sampling identical, and do
   not spend the freed draft memory. Retain DSpark only for a repeatable useful
   throughput win, provisionally at least 5%.
2. **Capture representative acceptance traces.** Add per-position survival,
   first-mismatch position, draft confidence, target top-1 margin, cap reason,
   prompt class, and cache state to diagnostic-only receipts. Use cache-busted
   OpenCode code, tool, and reasoning traces rather than filler. Run full-block
   `cap-accept` diagnostics so fixed-4 does not hide positions four and five.
3. **Calibrate the mechanism already designed for this problem.** Fit STS on
   those traces, refresh the SPS cost table on the admitted Oscar topology, and
   compare fixed-4 with confidence-scheduled compact verification. Optimize
   useful whole-cycle token/s rather than the displayed acceptance percentage.
   This can avoid verification waste but does not, by itself, make draft logits
   more accurate.
4. **Audit prediction quality before changing precision.** The loader already
   fails closed on all 18 draft shared-expert tensors and every mapped `mtp.*`
   destination, and graph/eager draft parity is exact. Preserve those gates and
   add a reference trace that records the first draft/target disagreement. FP32
   Markov already improved acceptance and remains selected; the earlier FP32 LM
   head arm did not beat FP32-Markov-only, so do not repeat it without evidence
   that BF16 head rounding changes target agreement.
5. **Improve the drafter if raw yield remains the limiter.** Distill or fine
   tune the DSpark sequential/Markov and confidence heads on target logits from
   representative OpenCode traces, weighting later positions where survival
   decays. This is the credible route to a materially higher five-position raw
   yield; merely exposing more verification slots cannot improve predictions.
6. **Test a separate prompt-lookup arm for code.** Repeated source context can
   favor n-gram/prompt speculative decoding with much lower draft cost. Treat it
   as a target-only-plus-prompt-lookup comparison, not as an assumed composition
   with DSpark.

Do not prioritize wider fixed verification, tree/top-k branching, more GPU
experts, or the two rejected fused kernels. The current trace attributes about
49--50 ms/cycle to target verification, with routed target MoE dominant; these
changes add or enlarge precisely the expensive work, and the model-scale sweeps
already regressed.

No `fwuff` GPU experiment was run for this audit. That host cannot reproduce
the admitted dwagon Oscar artifact, target placement, and TP2 topology, so a
small throughput run would not answer the break-even question. It becomes
useful only for a draft-only, bounded-memory trace replay after identical input
IDs, predictor weights, and target-reference logits have been captured; such a
run must preflight free memory and leave existing processes resident.

Research basis:

- [DSpark paper](https://arxiv.org/abs/2607.05147): semi-autoregressive draft
  dependency modeling plus confidence-scheduled, load-aware verification based
  on estimated prefix survival and engine cost.
- [SpecDec++](https://arxiv.org/abs/2405.19715): a trained acceptance head and
  threshold-based adaptive candidate length improved its tested workloads by
  7.2--11.1% over fixed-length speculative decoding.
- [SGLang's DSpark integration report](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/):
  compact ragged verification, STS, SPS cost fitting, `cap-accept`
  observability, and the warning that dynamic trimming was mainly a
  high-concurrency win in its initial measurements.
- [Independent 2x DGX Spark field report](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/README.md):
  fixing missing shared-expert weights raised acceptance from 25.7% to 60.2%,
  while patched acceptance varied from 68.7% for code to 33.7% for prose
  reasoning. This is diagnostic evidence, not a hardware-comparable benchmark.
- [Original speculative-decoding paper](https://arxiv.org/abs/2211.17192):
  target verification preserves the target distribution; speedup depends on
  the cost and agreement of the approximation, not on an acceptance-rate rule
  of thumb.
