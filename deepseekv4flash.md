# DeepSeek V4 Flash audit, integration, and dwagon handoff

Date: 2026-08-02

Local host: `dwagon`

Reference host: `fwuff`
Model: `deepseek-ai/DeepSeek-V4-Flash-0731`

## Final result

The 2026-08-02 continuation reached and exceeded the requested local target.
The recommended configuration is now TP2/EP2 across both RTX 3090s, with 12
target experts resident on each GPU and the complementary 116 experts per rank
served by two socket-local 56-thread AMX pools. The three DSpark draft stages
remain CPU-offloaded. This is genuine dual-GPU execution plus CPU expert
offload; it is not a one-GPU result reported under a two-GPU launch.

On the private fwuff workload, repeated fresh-cache runs processed exactly
2,694 input and 512 output tokens at 12.94 and 12.96 ms mean TPOT. That is
**77.22 decode tok/s on average**, with a 77.16 tok/s minimum, 6.55 s mean
TTFT, and 6.80-token mean DSpark acceptance. Against fwuff's recorded 31.088
tok/s and 25.0995 s TTFT on the same prompt and token shape, dwagon is 2.48x
faster in decode and reaches the first token 3.83x sooner.

| Runtime / configuration | TTFT | Mean TPOT | Decode rate | DSpark acceptance |
|---|---:|---:|---:|---:|
| fwuff best recorded run, private prompt | 25.0995 s | 32.1666 ms | 31.088 tok/s | 5.45 |
| dwagon August 1 TP1 baseline, same private prompt | 7.8882 s | 32.6061 ms | 30.669 tok/s | 5.375 |
| dwagon TP2/EP2 hybrid, private prompt, run 1 | 6.5880 s | 12.94 ms | 77.280 tok/s | 6.80 |
| dwagon TP2/EP2 hybrid, private prompt, run 2 | 6.5044 s | 12.96 ms | 77.160 tok/s | 6.80 |

The public fixed-shape prompt independently produced 75.129, 76.952, and
81.264 decode tok/s after cache flushes: 77.782 mean and 76.952 median, with
4.404 s mean TTFT. It has the same 2,694/512 shape but different content, so it
is supporting reproducibility evidence rather than the direct fwuff comparison.

The key performance unlock was also a correctness fix. A target MoE side
stream handed a tensor to the consumer stream without recording that consumer,
allowing allocator reuse while work was still pending. Recording the consumer
stream closes that lifetime bug. A second overlap path remained timing-sensitive,
so the hybrid launcher now defaults multi-stream MoE overlap off. With that
safe path, the blended route placement passed 20/20 deterministic short-output
runs. Raising the verified DSpark block from five to six then moved the stable
decode range above 70 tok/s; this also passed 20/20. Every speculative block is
still checked by the target model.

The earlier audit and parity work follows for provenance.

The fwuff deployment was audited read-only at its moved canonical location,
`/mnt/sanic/projects/deploy-dsv4-general-release`. No file in Kassie's tree
was changed. Transient accesses caused by the move from `/home/kassie` were
retried.

The authoritative performance reference is the workload that was actually
recorded on fwuff: **2,694 input tokens and 512 output tokens**. The server had
128K capacity, but no record showed a 128K active prompt benchmark. Therefore,
128K is now only an optional capacity setting and is not the default benchmark
or launch shape.

The fastest coherent configuration in the original August 1 audit was one GPU,
TP1, the fwuff hot12 expert ordering, one 60-thread CPU pool on NUMA node 0,
two same-layer deferred experts, BF16 KV cache, static DSpark verification,
DSpark block size 5, full decode graph, and 256/512/1024/2048
breakable-prefill graph tiers.

That original audit measured:

| Runtime | TTFT | Mean TPOT | Decode rate | DSpark acceptance |
|---|---:|---:|---:|---:|
| fwuff best recorded run | 25.0995 s | 32.1666 ms | 31.088 tok/s | 5.45 |
| dwagon exact-parity source/config | 7.8882 s | 32.6061 ms | 30.669 tok/s | 5.375 |
| dwagon cumulative generic hot12 | 8.1336 s | 32.7127 ms | 30.569 tok/s | 5.375 |

Those numbers established source/configuration parity before the TP2/EP2 work.
The report from memory that fwuff decoded at about 31 tok/s was accurate, but it
described this short active context, not a 128K prompt.

## Reference evidence

The fwuff metrics store contained 14 recorded rows, eight with 512 generated
tokens. Its best matching row had the values shown above. The exact benchmark
dataset had one JSONL record containing two conversation turns:

- input/output shape: 2,694 / 512 tokens;
- SHA256: `654ed3f540221597965a6f6f9d5486f379d5f70323c17c405f1bb94b5b07934e`;
- local temporary copy used for parity: `/tmp/fwuff-agentic-coding.jsonl`.

The dataset content is intentionally not published in the public repository.
The hash and shape are sufficient to identify the reference without exposing
the conversation.

The remote deployment inventory was:

| Component | Revision/state | Relevant role |
|---|---|---|
| `ktransformers` | `a8062bfa7e1060ce5855b5f1ad6aa6b116678307` | upstream 0.6.4 native MXFP4 and general fixes |
| `sglang-official` | `e1964da451ef9fbec04b326c729916281f90809b` | official 2026-07-31 DSV4 fixes and recipes |
| `sglang-kvcache-ai` | `04653fa88f5ce4ea83632c4ad436343db6bc8324` plus compatibility edits | KT integration and optional fallbacks |
| `sglang-deploy` | official base plus 24 tracked edits and 8 untracked assets | the serving SM86/BF16/MXFP4 deployment |

The remote tracked deployment patch was 1,773 lines with SHA256
`bb6b1e0dd0a699c5fbb540d7b7bc87cd53dfae1d276bfafc0770cd103816190b`.
The eight untracked assets included three source files and five benchmark or
profile files. The source assets covered BF16 sparse decode and MXFP4/Triton
integration.

## Model contract

| Field | Pinned value |
|---|---|
| local path | `/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731` |
| Hugging Face revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| architecture | `DeepseekV4ForCausalLM` |
| layers / routed experts / top-k | 43 / 256 / 6 |
| DSpark block size | checkpoint/default baseline 5; validated hybrid launch 6 |
| model maximum position length | 1,048,576 |
| checkpoint | BF16 activations, native MXFP4 routed experts, FP8 dense weights |
| indexed tensor bytes | 166,878,536,440 |
| safetensors shards | 48 |

The preparation helper verifies the revision, architecture, dimensions, shard
index, every indexed shard, total tensor bytes, and absence of partial shard
files before writing an expert plan.

## Cumulative source integration

The direct root SGLang dependency is now `vendor/sglang`, public branch
`exo/dsv4-cumulative-0801`, commit
`e2f179cb64714077f9d2ebe0d7ca5fd16923184c`. Its history combines:

- the exact fwuff SM86 release tree imported on top of official SGLang;
- the previously accumulated DSV4/DSpark branch;
- packed-cache and Ampere sparse-attention kernels;
- native MXFP4 routed experts and compact logical routing;
- static and compact ragged verification support;
- breakable prefill/decode graph support and graph-lifetime fixes;
- pipeline-parallel route profiling and deadlock-safe per-rank dumps;
- remote expert sidecars, multiple remote tiers, and CPU expert sharding;
- streamed and prefetched model loading improvements;
- DSV4 reasoning-parser completeness fixes;
- explicit KT NUMA placement and graph-metadata relinking;
- focused regression tests for the merged compatibility paths.

The merge deliberately kept the newer official/fwuff implementations at
conflict points, then restored non-conflicting capabilities from the older
accumulated branch through their current interfaces. Obsolete duplicate kernel
paths were removed where the same implementations had moved under the current
`kernels/ops` tree.

The reusable release audit patch remains at
`scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch`, SHA256
`6f35e2e14cbc29d05332c239db9334cb77c4d9bccf9c233cb80c3f30e4652032`.
The launcher accepts either the legacy patched source or the cumulative
in-tree implementation.

### SGLang enhancement decisions

| Enhancement | Result |
|---|---|
| SM86 all-BF16 cache/indexer/attention path | Imported as a complete coherent path. The separate packed cache remains available rather than mixing layouts. |
| Native MXFP4 GPU experts | Kept with compact routes, caller-owned outputs, portable Ampere kernels, and in-place reduction. |
| Arbitrary hot-expert placement | Kept explicit masks, profiles, ranked initialization, compact CPU complements, remote tiers, and draft controls. |
| Zero-GPU-expert execution | Kept the direct CPU/remote path without a zero-width GPU call. |
| Decode graph KT buffers | Fixed physical-token sizing and retained graph-owned buffers. |
| Breakable prefill buffers | Retained one pinned buffer set per 256/512/1024/2048 capture tier. |
| TP eager-break synchronization | Added the capture-only barrier after segment teardown. |
| Graph metadata refresh | Preserves captured tensor identities and relinks indexer/C4 metadata to live core buffers. |
| Route recording under PP/DSpark | Draft-only selections are excluded; per-rank PP dumps avoid world-collective deadlocks. |
| DSpark graph memory guard | Replaced the fixed 1 GiB cutoff with the measured conservative guard. |
| RoPE and top-k compatibility | Accepted singleton-head RoPE and a correctly shaped int32 top-k fallback. |
| Ampere compilation | Added SM90 guards, smaller metadata allocation, portable sparse attention, and CUDA-JIT norm fallback. |
| RunAI/streamed loading | Kept streamed views, clone/lifetime protection, prefetch coordination, and loader tests. |
| Expert ordering | Imported the complete 43-by-256 hottest-first fwuff ordering as data. |

### Changes not forced into the active path

The original deployment patch was not copied blindly. The audit found a
misplaced HIP LUT fragment, calls to missing BF16 pool methods, an unused BF16
store-mode calculation, and a cache/indexer mode combination that was unsafe
unless the full layout conversion was selected. Those individual defective
fragments were excluded or replaced by the coherent newer implementation.

Forced Marlin MXFP4 on SM86 was rejected because the remote experiment yielded
invalid all-zero IDs. Hopper-only cluster kernels, ROCm/AITER and NPU changes,
and DSA/DeepEP refactors are retained only where generally compatible; none is
enabled by the dwagon launch.

## Deferred experts

KTransformers is now at public branch `exo/glm52-osdi26-patched`, commit
`373539da61a45d1ceda56a783b8621d5a28bd551`.

The previous deferred-expert scheduler was incorrect: it placed a tail task in
the following ring slot, allowed the current layer to return before the tail
finished, and then accumulated the stale contribution into the next layer's
MoE output. That violates transformer layer ordering and makes the result
timing-dependent.

The fix keeps the high-score/tail split but submits both tasks to the current
layer's output. The immediate task initializes the output, the tail task adds
to it, and synchronization drains both before the layer returns. Invalid or
unowned expert IDs use a fixed extra sentinel slot, avoiding both accidental
expert-zero aliases and data-dependent tensor shapes during CUDA graph
capture.

Four focused tests pass, including sentinel routing, invalid protected routes,
same-layer task ordering, and real CUDA graph capture/replay. The validated
August 1 hot12 benchmark used defer2 with these exact same-layer semantics.
The accepted August 2 TP2/EP2 configuration uses defer0 and disjoint CPU expert
shards, avoiding the centralized stage that made the earlier TP2 runs slow.

## Optimization campaign

Coherency validation was used as warm-up. Measurements were accepted only when
the deterministic validation passed; output content was not logged in this
report.

### Parity baseline

The closest fwuff-equivalent run used one RTX 3090, hot12, defer2, one NUMA
node, one 60-thread KT pool, BF16 KV cache, static verification, block size 5,
and the same prefill tiers. It reached 30.669 decode tok/s with 7.888 s TTFT and
5.375 acceptance. This establishes code/setup parity with fwuff's 31.088
tok/s best row.

### Three significant full-system iterations

| Iteration | Change | Coherency/warm-up | Authoritative workload | Decision |
|---|---|---|---|---|
| 1 | TP2, both GPUs, hot48, defer2 | Short validation passed | No 512-token response after 6m29s; bounded below about 1.3 response tok/s end-to-end | Reject: centralized CPU experts serialized both ranks. |
| 2 | TP2 hot48 with deferred tail disabled | Short validation passed and improved | No full response after 3m17s | Reject: removing the second CPU task helped but did not remove the central CPU/TP synchronization bottleneck. |
| 3 | TP1 hot12 spread across both NUMA nodes, 112 threads and two pools | Failed deterministic coherency; acceptance collapsed to 1.0 | Invalid result | Reject: KT pools are not interchangeable socket aggregation. |

Additional controlled checks:

- hot16 on one GPU served but failed deterministic coherency; hot12 remained
  the largest validated placement in the original TP1 campaign;
- compact ragged verification failed coherency on SM86; static verification is
  retained;
- a decode-specific route ordering regressed to 28.19 and then 25.37 tok/s and
  changed the coherent result; the generic fwuff ordering is retained;
- an early cumulative sample measured 39.4148 ms TPOT (25.371 tok/s) with only
  4.4375 acceptance; the August 1 hot12/static selection restored 32.7127 ms
  and 5.375 acceptance.

### August 2 TP2/EP2 hybrid campaign

The new path removes the centralized CPU-expert bottleneck from the August 1
TP2 experiments. Expert parallelism gives each rank a disjoint target shard:
12 GPU experts and 116 CPU experts per layer, with rank 0 bound to NUMA 0 and
rank 1 bound to NUMA 1. Both ranks still participate in tensor-parallel dense
layers and attention. The three draft stages use 128 CPU experts per rank.
Idle GPU memory after capture was 17,455 MiB on each 24,576 MiB RTX 3090.

Route placement is built from exact distinct decode calls, not aggregate token
counts alone. Per-pass recorder traces from both replicated ranks are checked
for agreement and reduced to a 43-by-256 target profile. For every layer the
GPU union first takes the 12 hottest experts from that live trace, then fills
the remaining 12 union slots from the older 32K decode profile. The plan is
disjoint and exhaustive across GPU and CPU owners. On the recorded trace it
covers 73.36% of distinct expert calls and 87.67% of routed expert instances.

The placement campaign deliberately rejected regressions:

| Change | Fresh 2,694/512 decode result | Decision |
|---|---:|---|
| TP2/EP2 legacy target hot12 placement | 30.420 tok/s | coherent baseline; CPU/TP synchronization still dominant |
| exact-call-only target placement | 28.947 tok/s | reject; low-frequency coverage was too narrow |
| 16 GPU experts per rank | 30.098 tok/s | reject; extra residency did not repay GPU work |
| caller-owned in-place MoE output | 23.53-25.31 tok/s | reject and keep default off |
| lower AMX threshold, smaller GPU sets, eager/full attention variants | slower, invalid, or OOM | reject |
| blended hot-prefix/fill placement with block 5 and safe streams | 69.24-78.10 tok/s | coherent; established the main speedup |
| same placement with verified block 6 | 75.13-81.26 tok/s | accept; 77.78 tok/s three-run mean |

The first blended launches exposed an intermittent stream-lifetime failure:
the same short greedy request produced multiple hashes and occasional runaway
outputs. Serializing only the KT CPU stream did not fix it. The target shared
expert path allocated on an alternate CUDA stream, waited before consuming on
the main stream, but did not register the main stream as a tensor user. The
implementation now performs producer-to-consumer synchronization followed by
`Tensor.record_stream(consumer)` in normal and DeepEP joins. PyTorch documents
that contract in [Tensor.record_stream](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
and its [CUDA semantics note](https://docs.pytorch.org/docs/main/notes/cuda.html).

That allocator fix alone did not eliminate every full-model failure, indicating
a second overlap-sensitive path. The DSV4 MoE dual-stream branch is therefore
gated by `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`, and the hybrid launcher defaults
it to `0`. This combination passed 20/20 exact short-output runs at block 5 and
again at block 6. An explicit opt-in remains available for future isolation,
but it is not a supported performance setting today.

The 512-token public greedy runs completed cleanly but did not have identical
output SHA256 values across repeats. That is consistent with long-horizon
numerical sensitivity near greedy decision boundaries, but it means the
20/20 short gate must not be described as bitwise long-generation determinism.
The safe-stream configuration eliminated the malformed/runaway behavior seen
with overlap; a separate long-horizon reference-logit comparison remains useful
future correctness work.

Block 6 intentionally differs from the checkpoint's advertised block 5. The
server warns about that gamma mismatch, but speculative correctness is retained
because all seven proposed/continuation positions are verified by the target
model. It reduced public benchmark stream events from roughly 88-90 to 77-79
and raised private-prompt acceptance to 6.80. The repeated private result was
77.16-77.28 decode tok/s, safely above the requested 70 tok/s.

Draft-hot placement was implemented as a separate, default-off three-stage
hybrid plan. Its profile covered 97.75% of exact draft calls and 98.90% of draft
routes at about 459 MiB per GPU, but it failed the earlier overlap-enabled
coherency gate. It was not mixed into the accepted result; draft experts remain
CPU-offloaded until that path receives a fresh safe-stream validation.

### Custom-kernel and parallelism findings

A one-expert Any4/TinyGEMM proof was promising: its two GEMMs measured about
0.031 and 0.019 ms and the full expert about 0.053 ms versus about 0.766 ms for
the existing eager Triton path, with cosine similarity 1.0. That did not survive
the real resident/graph shape: 12 separately launched expert kernels took about
0.405 ms under a CUDA graph versus about 0.373 ms for grouped Triton. The next
kernel step must therefore be grouped/persistent rather than one launch per
expert. Any4 is CC-BY-NC-4.0, so its code was not vendored into this Apache
repository.

This conclusion matches the primary implementation guidance: CUTLASS uses a
[persistent grouped scheduler](https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/grouped_scheduler.html),
and CUDA graphs reduce launch overhead but do not merge independent kernels
([NVIDIA CUDA graph launch analysis](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)).
NCCL operations remain graph-capturable per the
[NCCL CUDA Graph guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html),
so TP/EP communication is not inherently incompatible with the accepted full
decode graph.

PP was retained as a profiling and multi-host option, but it is a poor fit for
this single-request two-GPU decode target because it adds bubbles and does not
solve per-layer expert latency. TP alone had already failed on the centralized
CPU queue. TP2 plus EP2 is the useful local decomposition: dense work spans both
GPUs while CPU expert ownership and AMX pools are socket-local.

### Interpretation of the August 1 utilization bursts

The roughly one-second CPU/GPU-on and one-second-off pattern is consistent
with DSpark blocks and breakable graph segments alternating with a centralized
KT CPU-expert stage and TP barriers. Rank 0 owns the heavy CPU work; rank 1
often reaches the barrier first. Consequently one GPU can remain saturated
while the other fluctuates around partial utilization during each burst.

Two identical GPUs do not double this workload automatically. TP2 partitions
the dense GPU work, but the August 1 launch added communication around one
centralized CPU MoE queue. The accepted August 2 configuration implements the
remedy anticipated by that finding: EP2 distributes CPU ownership across two
socket-local queues, then route-aware GPU placement and safe graph execution
make the second GPU useful during decode.

## Fastest prepared local launch

The hybrid launcher now defaults to the accepted local configuration:

- TP2 and EP2 over physical GPUs 0 and 1;
- 12 target GPU experts plus 116 socket-local CPU experts on each rank;
- one 56-thread AMX pool on each of NUMA nodes 0 and 1;
- all three DSpark draft expert shards on CPU;
- exact-call hot12 plus 32K-profile fill placement;
- safe single-stream MoE execution and caller-owned in-place output disabled;
- BF16 KV cache, static verified DSpark block 6;
- full decode graph and breakable prefill tiers 256/512/1024/2048;
- 8,192-token context/capacity, one running request, and 2,048-token chunks.

The persisted route-count profile is
`/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v4-agentic-distinct-decode-calls.pt`,
SHA256 `e0d907f549e1df6658cf92e6755f30efc975e2049e979823d0cf0c927e4b0e35`.
It contains expert-call counts and recorder provenance, not prompt text. If it
must be regenerated from a fresh two-rank `per_pass` recording:

```bash
scripts/build_dsv4_decode_call_profile.py \
  --rank-profiles "$RECORDER_RANK0" "$RECORDER_RANK1" \
  --profile-topology replicated-per-pass \
  --decode-routes-per-layer 36 \
  --output /var/lib/exo/profiles/dsv4-native-mxfp4/flash-v4-agentic-distinct-decode-calls.pt
```

Preparation validates inputs and writes the blended EP2 plan without starting
the model:

```bash
scripts/dsv4_flash_hybrid_ep2_dwagon.sh
```

Launch when desired:

```bash
scripts/dsv4_flash_hybrid_ep2_dwagon.sh --launch
```

Gate a changed launch with 20 fresh short requests before measuring it:

```bash
scripts/validate_dsv4_flash_coherency.py \
  --repetitions 20 --flush-cache-between-runs
```

The public-shape benchmark helper defaults to 2,694 input and 512 output
tokens and does not print generated text:

```bash
scripts/benchmark_dsv4_flash_128k.py
```

For the direct private comparison, use SGLang's custom dataset benchmark with
`/tmp/fwuff-agentic-coding.jsonl`, one prompt, `--sharegpt-output-len 512`, no
warm-up requests, temperature zero, and thinking enabled. Do not publish the
dataset contents; its identifying SHA256 remains
`654ed3f540221597965a6f6f9d5486f379d5f70323c17c405f1bb94b5b07934e`.

For capacity testing only, override both limits, for example:

```bash
DSV4_CONTEXT_LENGTH=128000 DSV4_MAX_TOTAL_TOKENS=128000 \
  scripts/dsv4_flash_hybrid_ep2_dwagon.sh --launch
```

That is not the authoritative performance comparison.

## Repository and submodule graph

All cumulative forks are public:

| Repository | Branch / commit | Role |
|---|---|---|
| `https://github.com/ldyeax/exo` | `agent/linux-cuda-nccl` | root integration and launch handoff |
| `https://github.com/ldyeax/exo_sglang` | `exo/dsv4-cumulative-0801` at `e2f179cb6`; historical `exo/dsv4-flash-0731` at `ad67dbbc0`; GLM bundle at `73e877ac5` | cumulative SGLang lines |
| `https://github.com/ldyeax/exo_ktransformers` | `exo/glm52-osdi26-patched` at `373539da6` | cumulative KTransformers and deferred-expert fix |
| `https://github.com/ldyeax/exo_llama_cpp` | `exo/kimi-k3-cumulative` at `651092c60` | cumulative llama.cpp/Kimi work |

Root exo directly tracks all three public forks. KTransformers' active and
archived llama declarations also target `ldyeax/exo_llama_cpp`; the active
commit is the same `651092c60` used by root exo, replacing the unrelated old
shallow snapshot. KTransformers' nested SGLang targets the public
`ldyeax/exo_sglang` GLM-compatible branch. It is intentionally not repointed
to the divergent DSV4 branch: their merge base predates both accumulated lines,
and forcing the pointer would discard its GLM compatibility. Root exo carries
the DSV4 cumulative branch directly instead.

Unmodified pybind11 and custom FlashInfer dependencies continue to use their
upstream repositories.

## Validation record

- Root exo: BasedPyright reports 0 errors/warnings/notes; repository Ruff
  passes; Nix formatting is at a fixed point; pytest reports 1,169 passed,
  6 skipped, and 193 slow tests deselected. Independent vendor trees now use
  their own lint/test policies instead of being recursively collected by exo.
- Route-profile, shard-plan, and launcher tests: 46 passed.
- Current focused SGLang DSV4/KT suite: 52 passed, including 33 KT wrapper,
  eight BCG/stream tests, DSpark synchronization/acceptance, caller-owned MoE
  output, breakable-graph buffers, frozen draft KV, and SWA cache coverage.
- DSV4 reasoning parser: 102 tests passed; only counts were exposed during the
  safety review.
- Changed SGLang stream files compile, their focused tests/lint pass, and vendor
  whitespace checks pass. The inherited production file retains three existing
  unused local assignments outside this change.
- KTransformers current-source deferred-expert tests: 5 passed and one CUDA
  graph test skipped when the final CPU-only test process had no CUDA access.
- Block-5 blended placement passed 20/20 deterministic short requests;
  block-6 passed a separate 20/20 gate with the same exact hash.
- Three public fixed-shape block-6 runs were all above 75 tok/s. Two private
  fwuff-prompt runs were 77.16-77.28 tok/s with 6.80 acceptance.
- Model contract, expert ordering, SPS data, shell syntax, and preparation
  checks pass. Default preparation rebuilt a semantically identical persisted
  GPU/CPU plan from the persisted profile.
- SPS table SHA256:
  `117b07418b934dd4d3546ab7c32789dfcf1a8648ff6e2ca312076ebd88ff08d2`.
- CUDA 13.1/13.3 is available under `/opt/cuda`; Nix 2.35.1 and the required
  build/runtime dependencies were installed system-wide rather than hidden in
  project-local workarounds.
- At handoff, all campaign servers are stopped, GPU compute-process enumeration
  is empty, and the benchmark CPU policy holder verified restoration of all 224
  cpufreq policies, turbo/pstate state, RAPL limits, and temperature thresholds.

The August 2 measurements were taken from the current root and vendor working
trees described here. A future change to placement, DSpark gamma, stream
overlap, graph capture, or KT output ownership must begin with the 20-run
coherency gate before its performance result is accepted.
