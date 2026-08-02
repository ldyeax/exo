# DeepSeek V4 Flash audit, integration, and dwagon handoff

Date: 2026-08-01

Local host: `dwagon`

Reference host: `fwuff`
Model: `deepseek-ai/DeepSeek-V4-Flash-0731`

## Final result

The fwuff deployment was audited read-only at its moved canonical location,
`/mnt/sanic/projects/deploy-dsv4-general-release`. No file in Kassie's tree
was changed. Transient accesses caused by the move from `/home/kassie` were
retried.

The authoritative performance reference is the workload that was actually
recorded on fwuff: **2,694 input tokens and 512 output tokens**. The server had
128K capacity, but no record showed a 128K active prompt benchmark. Therefore,
128K is now only an optional capacity setting and is not the default benchmark
or launch shape.

The fastest coherent local configuration found was one GPU, TP1, the fwuff
hot12 expert ordering, one 60-thread CPU pool on NUMA node 0, two same-layer
deferred experts, BF16 KV cache, static DSpark verification, DSpark block size
5, full decode graph, and 256/512/1024/2048 breakable-prefill graph tiers.

On the same 2,694/512 workload:

| Runtime | TTFT | Mean TPOT | Decode rate | DSpark acceptance |
|---|---:|---:|---:|---:|
| fwuff best recorded run | 25.0995 s | 32.1666 ms | 31.088 tok/s | 5.45 |
| dwagon exact-parity source/config | 7.8882 s | 32.6061 ms | 30.669 tok/s | 5.375 |
| dwagon cumulative generic hot12 | 8.1336 s | 32.7127 ms | 30.569 tok/s | 5.375 |

The exact local parity result is 1.37% behind fwuff's best decode TPOT, and the
reusable cumulative configuration is 1.70% behind. Dwagon's TTFT is much
lower. The report from memory that fwuff decoded at about 31 tok/s was accurate,
but it described this short active context, not a 128K prompt.

All model servers were stopped after testing. No model is left running.

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
| DSpark block size | 5 |
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
hot12 benchmark used defer2 with these exact same-layer semantics. TP2 still
performs better with defer0 than defer2, but both TP2 configurations were far
behind TP1 because the CPU expert stage remains centralized.

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

- hot16 on one GPU served but failed deterministic coherency; hot12 remains the
  largest validated fwuff placement on this checkpoint and hardware;
- compact ragged verification failed coherency on SM86; static verification is
  retained;
- a decode-specific route ordering regressed to 28.19 and then 25.37 tok/s and
  changed the coherent result; the generic fwuff ordering is retained;
- an early cumulative sample measured 39.4148 ms TPOT (25.371 tok/s) with only
  4.4375 acceptance; the final hot12/static selection restored 32.7127 ms and
  5.375 acceptance.

### Interpretation of the observed utilization bursts

The roughly one-second CPU/GPU-on and one-second-off pattern is consistent
with DSpark blocks and breakable graph segments alternating with a centralized
KT CPU-expert stage and TP barriers. Rank 0 owns the heavy CPU work; rank 1
often reaches the barrier first. Consequently one GPU can remain saturated
while the other fluctuates around partial utilization during each burst.

Two identical GPUs do not double this workload automatically. TP2 partitions
the dense GPU work, but it also adds communication and synchronization around
a CPU MoE stage that was not expert-parallelized across ranks. Dwagon's extra
cores and second GPU therefore help TTFT and capacity, but the tested TP2
shape makes decode slower, not faster. Future multi-GPU work should distribute
CPU expert ownership or use the implemented remote/CPU-shard tiers instead of
adding more tensor-parallel ranks around one centralized expert queue.

## Fastest prepared local launch

The launcher defaults now reproduce the authoritative workload's best expected
serving shape, not the unrelated 128K capacity target:

- local-only TP1 on physical GPU 0;
- root `vendor/sglang` cumulative source and `vendor/ktransformers`;
- 8,192-token context and token capacity, overridable explicitly;
- hot12 from the complete fwuff 43-by-256 ordering;
- one 60-thread AMX pool bound to NUMA node 0;
- defer2 with the corrected same-layer behavior;
- BF16 KV cache, static DSpark block-5 verification;
- full decode graph and breakable prefill tiers 256/512/1024/2048;
- CUDA JIT normalization on SM86;
- one running request and 2,048-token prefill chunks.

Preparation only; this validates inputs and writes the hot12 plan without
starting the model:

```bash
scripts/dsv4_flash_fwuff_parity.sh
```

Launch when desired:

```bash
scripts/dsv4_flash_fwuff_parity.sh --launch
```

Then run coherency warm-up before collecting a speed result:

```bash
scripts/validate_dsv4_flash_coherency.py
```

The public-shape benchmark helper defaults to 2,694 input and 512 output
tokens and does not print generated text:

```bash
scripts/benchmark_dsv4_flash_fwuff_baseline.sh
```

For capacity testing only, override both limits, for example:

```bash
DSV4_CONTEXT_LENGTH=128000 DSV4_MAX_TOTAL_TOKENS=128000 \
  scripts/dsv4_flash_fwuff_parity.sh --launch
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
- SGLang cumulative merge: 46 DSV4/runtime tests passed, 8 platform skips, and
  10 subtests passed.
- SGLang KT wrapper: 16 tests passed.
- DSV4 reasoning parser: 102 tests passed; only counts were exposed during the
  safety review.
- All changed SGLang Python files compiled; focused lint passed. The inherited
  upstream files retain their established late-import and lambda conventions.
- KTransformers deferred experts: 4 tests passed with direct GPU access,
  including CUDA graph capture/replay.
- Local exact parity and cumulative hot12 runs passed deterministic coherency.
- Model contract, expert ordering, SPS data, shell syntax, and preparation
  checks pass.
- SPS table SHA256:
  `117b07418b934dd4d3546ab7c32789dfcf1a8648ff6e2ca312076ebd88ff08d2`.
- CUDA 13.1/13.3 is available under `/opt/cuda`; Nix 2.35.1 and the required
  build/runtime dependencies were installed system-wide rather than hidden in
  project-local workarounds.
- At handoff, GPU compute-process enumeration is empty.

The full-model performance measurements were taken on the validated
`f829ed5dd` parent of the final cumulative SGLang merge. The final merge keeps
that implementation as its first parent and adds the older accumulated feature
line. Its affected compatibility paths have focused regression coverage, but
the next launch should still begin with the coherency helper before accepting
any new measurement.
