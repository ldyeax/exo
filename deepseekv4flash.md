# DeepSeek V4 Flash release audit and dwagon launch handoff

Date: 2026-08-01
Local host: `dwagon`
Remote reference host: `fwuff`
Status: source integration and launch preparation complete; model not launched

## Outcome

The `deploy-dsv4-general-release` deployment was audited read-only at its
canonical moved location:

`/mnt/sanic/projects/deploy-dsv4-general-release`

No file under Kassie's deployment tree was changed. The sibling
`deploy-dsv4-dwagon` tree was byte-compared only as a fallback and carried the
same custom SGLang delta.

The local accumulated forks already contained most of the important DSV4 work
and, in several areas, were more advanced than the release tree. The selected
missing correctness, capture-safety, loading, parser, and Ampere-compatibility
changes are recorded in:

- `scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch`
- `scripts/patches/dsv4-flash/README.md`

The fastest expected conservative dwagon launch is prepared at:

- `scripts/dsv4_flash_0731_tp2_dwagon.sh`
- `scripts/prepare_dsv4_flash_0731.py`

Running the shell script without arguments performs preparation only. It has
already completed successfully and generated the validated hot48 logical mask
at `/var/lib/exo/cache/dsv4-flash-0731-release/agentic-hot48-mask.pt`. It did
not start SGLang or load model weights.

## Provenance and inventory

### Remote release tree

| Component | Revision/state | Relevant contents |
|---|---|---|
| `ktransformers` | clean `a8062bfa7e1060ce5855b5f1ad6aa6b116678307` | upstream 0.6.4, native MXFP4 and current general fixes |
| `sglang-official` | clean `e1964da451ef9fbec04b326c729916281f90809b` | official 2026-07-31 DSV4 recipes and fixes |
| `sglang-kvcache-ai` | `04653fa88f5ce4ea83632c4ad436343db6bc8324` plus two compatibility edits | KT integration history and optional operator fallbacks |
| `sglang-deploy` | base `e1964da451ef9fbec04b326c729916281f90809b`, 24 tracked edits and 8 untracked assets | working SM86 BF16 cache stack, MXFP4 path, graph fixes, profiles |

The remote tracked SGLang patch is 1,773 lines with SHA256:

`bb6b1e0dd0a699c5fbb540d7b7bc87cd53dfae1d276bfafc0770cd103816190b`

The eight untracked assets omitted by ordinary `git diff` were three source
files and five benchmark/profile files. They were audited separately. The
three source files are the BF16 decode kernel and two MXFP4/Triton integration
modules. The local accumulated fork already has equivalent or more advanced
MXFP4 implementations.

### Model contract

| Field | Pinned value |
|---|---|
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| local path | `/mnt/sanic/llm_models/DeepSeek-V4-Flash-0731` |
| Hugging Face revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| architecture | `DeepseekV4ForCausalLM` |
| layers / routed experts / top-k | 43 / 256 / 6 |
| DSpark block size | 5 |
| maximum position length | 1,048,576 |
| checkpoint dtype/quantization | BF16 activations, native MXFP4 routed experts, FP8 dense weights |
| indexed tensor bytes | 166,878,536,440 |
| safetensors shards | 48 |

The preparation helper validates all of these values, the revision metadata,
every indexed shard, and the absence of partial shard files before emitting a
GPU-expert mask.

### Local accumulated sources

The accepted runtime source remains
`/var/lib/exo/sources/sglang-dspark-30261`, based on SGLang commit
`6cc9352dfe6c5c013750e72b39c127870ef5b54f` plus the accumulated DSV4/DSpark,
pipeline-parallel, compact-ragged, remote-sidecar, packed-cache, and workspace
changes. Those changes are committed as `40e43604a` on
`exo/dsv4-flash-0731`.

The tracked KTransformers checkout is `vendor/ktransformers`, branch
`exo/glm52-osdi26-patched`, now at `2521adb`. Its nested SGLang checkout is at
`73e877ac5` on `bundle/glm52-fwuff-sglang`.

### Public fork publication

| Repository | Published branches | Purpose |
|---|---|---|
| `https://github.com/ldyeax/exo_sglang` | `bundle/glm52-fwuff-sglang` at `73e877ac5`; `exo/dsv4-flash-0731` at `40e43604a` | accumulated reusable SGLang changes and the exact audited 0731 runtime integration |
| `https://github.com/ldyeax/exo_ktransformers` | `exo/glm52-osdi26-patched` at `2521adb` | accumulated KTransformers changes and public nested SGLang pointer |

Both repositories are public GitHub forks of their `kvcache-ai` upstreams.
The exo and KTransformers `.gitmodules` files now use these public HTTPS URLs;
unmodified third-party submodules continue to use their upstream repositories.

The final reusable release patch has SHA256:

`6f35e2e14cbc29d05332c239db9334cb77c4d9bccf9c233cb80c3f30e4652032`

## Enhancement comparison and decisions

### SGLang deployment patch

| Enhancement | Local comparison | Decision |
|---|---|---|
| SM86 all-BF16 DSV4 KV/cache/indexer stack | Local packed FP8/UE8M0 cache is 584 bytes per token versus remote 1024 bytes, with accepted SM86 LUT dequant, TileLang indexer, and portable sparse attention | Keep local packed cache. Do not partially import a layout-wide BF16 conversion. |
| Native MXFP4 GPU experts | Local version adds compact logical routes, caller-owned GEMM outputs, shared DSV4 MLP workspace, and in-place reduction | Keep local implementation. |
| Arbitrary hot-expert placement | Local wrapper supports explicit logical masks, profile selection, compact CPU complements, remote tiers, and draft controls | Keep local wrapper; import the 0731 route ordering as data. |
| Zero-GPU-expert execution | Already has a direct CPU/remote path without a zero-width GPU invocation | Already present. |
| KT decode graph buffers | Local registration used request counts while KT allocates by physical tokens; fine verify tiers could replace pinned buffers still referenced by earlier graphs | Fixed: register physical fine-ragged tiers or request width multiplied by tokens per request. |
| KT prefill graph buffers | Missing before enabling breakable prefill graphs | Fixed: retain one pinned buffer set per 256/512/1024/2048 capture tier. |
| Breakable graph TP synchronization | Local TP2 eager breaks could begin after variable per-rank segment teardown without a new barrier | Fixed with a capture-only TP barrier after segment teardown. |
| Expert route recorder | Draft selections could be recorded without a target-layer scope | Fixed; draft-only selections are ignored. |
| DSpark draft graph memory guard | Fixed 1 GiB cutoff disabled a measured ~70 MiB graph on tight placements | Reduced to a conservative 0.25 GiB cutoff. |
| 4D singleton-head RoPE input | Remote accepted this shape, local expected 3D | Imported as a low-risk compatibility fix. |
| Top-k-v2 raw-index output | Optional output could be absent or incompatible | Imported a correctly shaped int32 fallback. |
| Top-k-v2 Ampere build guards | Cluster kernels and 128 KiB metadata allocation are Hopper-oriented | Imported SM90 compile guards and a 64 KiB metadata cap; launch still keeps top-k-v2 disabled on SM86. |
| Optional `sgl_kernel` import fallbacks | Installed runtime exports the referenced operators, and the local FP4 path uses its JIT module | Not needed for the pinned runtime; documented for future version skew. |
| Route-ranked placement | Remote hot12 covered 36.23% of its measured agentic demand | Imported the complete 43×256 hottest-first ordering. The local TP2 plan selects the first 48 logical IDs per layer. |

### Remote patch defects deliberately excluded

The release patch was not copied wholesale. The audit found:

1. A misplaced HIP-path LUT fragment references values that are not defined in
   that branch.
2. The BF16 indexer pool exposes calls whose corresponding pool methods are
   absent.
3. One BF16 compressor path computes a store-mode value that is not consumed.
4. The SM86 BF16 path can select an incompatible FP4-indexer input form unless
   the full mode combination is rejected.
5. The BF16 conversion changes pool sizes, page strides, storage types, gather,
   scatter, prefill, decode, and model writes together. A partial port would be
   unsafe even where individual hunks look useful.

Forced Marlin for MXFP4 on SM86 is also excluded: the remote experiment
produced all-zero output IDs. The launch retains native packed MXFP4 weights
with the portable Triton routed-expert path; dense FP8 remains on its validated
path.

### Official SGLang changes referenced by the release tree

Imported or adapted:

- DeepSeek V4-specific strict-thinking parsing plus complete stream-finalization
  behavior.
- Streaming FP8 projection pairing, RunAI streamed-view cloning, and the paired
  async-load lifetime guard. This removes whole-generator materialization and
  prevents reuse races if RunAI is selected later.
- FP16 FE8M0 Marlin scale dequantization used by the referenced KT SGLang pin.
- The TP barrier for breakable CUDA-graph eager boundaries.
- Small completeness fixes for TBO forward-batch construction and prefill
  delayer state negotiation.

Reviewed and deferred:

- ROCm/AITER MQA preshuffle and AMD/NPU fixes: not used on SM86 CUDA.
- SM90 FP8 MegaMoE and cluster-only work: wrong architecture.
- target verify-mask changes: superseded by the local device-side compact
  ragged DSpark verifier.
- RunAI transport selection and fastsafetensors/GDS policy: the prepared launch
  uses ordinary local-path safetensors plus KT native expert loading.
- DSA/DeepEP and distributed attention refactors: not enabled in the local
  TP2 launch.
- sparse-window cached-prefix fixes: radix caching remains explicitly disabled
  in the prepared single-request launch.

### KTransformers changes

Already present locally before this audit:

- native checkpoint MXFP4 loading;
- AVX512/AVX2/AMX MXFP4 paths;
- four-token and four-row prefill work;
- multipool/NUMA support;
- dynamic and profile-guided GPU experts;
- memory ownership, leak, and use-after-free fixes;
- native 2604B asymmetric SwiGLU clamping;
- compact CPU expert storage when GPU or remote tiers own experts.

Imported from the current remote/upstream history:

- compressed-tensors RAWINT4 int32-to-byte normalization;
- correct bind-based port availability checks;
- loopback-only archived balance-server scheduler sockets;
- AVX-VNNI-256 per-expert RAWINT4 weight/scale pointer loading;
- the referenced FP16 FE8M0 Marlin specialization in nested SGLang.

Large Kimi-K2 RAWINT4 matmul/SFT/LoRA work, SYCL-only changes, and unrelated
training features were reviewed but not pulled into the DSV4 launch path.

## Benchmarks and interpretation

### Remote validated 0731, one RTX 3090

Best measured remote agentic configuration used hot12, defer2, one CPU pool,
BF16 cache, DSpark block 5, and 128K capacity:

| Workload | Result |
|---|---|
| 2,694 input / 512 output | 25.10 s TTFT, about 107.3 input tok/s |
| decode | 32.17 ms/token, 31.09 tok/s |
| DSpark acceptance | 5.45 |
| free memory after load | about 0.84 GiB |

Other remote findings:

- hot8: 28.49 decode tok/s;
- static expert IDs 0–7: 17.53 decode tok/s;
- hot11 was numerically invalid and is discarded;
- defer1, defer3, decode-only placement, two pools on one NUMA node, and FP16
  activations all regressed or were invalid;
- 500K/no-GPU-expert mode retained about 102 input tok/s but only 4.05 decode
  tok/s.

### Local historical TP2 baseline, earlier checkpoint revision

The accepted local TP2 hot48 DSpark run measured:

| Workload | Result |
|---|---|
| 32,768 input | 531.75 fresh input tok/s |
| 4,096 output | 48.03 ms/token, 20.82 tok/s |
| DSpark acceptance | 4.78 |

These numbers are not directly comparable: model revision, prompt length,
source tree, GPU count, cache format, placement profile, and output length all
differ. The remote single-GPU result is the stronger short-context decode
measurement; the local TP2 result is the only dwagon-validated long-prefill
configuration and uses both 3090s plus both NUMA nodes.

The old local activation profile was collected on revision `62af8...` and is
not used by the new launcher. The imported ordering is bound to the 0731 model,
but only its first 12 experts per layer were directly benchmarked remotely.
Selecting the first 48 per layer is a reasoned TP2 extrapolation and must be
re-profiled after the first authorized model run. The retained TP2 SPS table is
also a hardware-specific baseline, not a fresh 0731 calibration.

## Prepared fastest expected dwagon launch

Primary choice: TP2 hot48 on both local RTX 3090s.

Key settings:

- compute remains local to dwagon; model files are read from the selected local
  mount path;
- prefers `/mnt/sanic-edr` when already mounted, otherwise uses `/mnt/sanic`;
- TP2 over NVLink, `NCCL_P2P_LEVEL=NVL`;
- 104 physical CPU workers, two thread pools, explicitly mapped to NUMA nodes
  0 and 1;
- native MXFP4 experts, explicit 43×256 logical hot48 mask, two deferred CPU
  experts per token, and no draft GPU experts;
- compact DSpark verification, block 5, graph-tier alignment, and the measured
  fine-tier TP2 SPS table;
- breakable decode graph at batch size 1 and breakable prefill tiers
  256/512/1024/2048;
- packed SM86 cache, top-k-v2 disabled, portable Triton MXFP4;
- 65,536 default context to retain hot48 memory headroom; override with
  `DSV4_CONTEXT_LENGTH` only after measuring memory;
- one running request, 2,048-token prefill chunks, radix cache disabled.

Preparation only:

```bash
scripts/dsv4_flash_0731_tp2_dwagon.sh
```

Future authorized launch:

```bash
scripts/dsv4_flash_0731_tp2_dwagon.sh --launch
```

The launch form refuses to continue unless two GPUs are visible, each has at
least 22,000 MiB free, port 30010 is unused, the model contract passes, and the
audited source patch is installed. On the final preparation snapshot both
RTX 3090s were idle with 24,123 MiB free, but the model was not launched.

The host has 112 physical cores across two NUMA nodes and approximately 755 GiB
RAM. The model cannot be copied to the root filesystem, which had only about
28 GiB free. `/mnt/sanic` is NFS over Ethernet; the existing EDR/IPoIB mount
path remains the preferred cold-load source when it is already available. No
network or mount configuration was changed during this audit.

## Validation completed

- A fresh GitHub checkout of the published exo branch fetched KTransformers
  `2521adb0b6b2fc128ed6db533ed9cd5bbb1d5e6c` and nested SGLang
  `73e877ac5bf60030b8aec16b1c0c890cb65b890c` through the new public HTTPS
  submodule URLs.
- exo type checking: 0 errors, 0 warnings, 0 notes.
- exo-owned tests excluding the optional image suite: 1,169 passed, 5 skipped,
  193 deselected. The image suite could not collect because torch is not
  installed in the root exo environment.
- DSV4 preparation helper: 3 passed, 1 optional torch test skipped in the root
  environment; the same plan was then generated and validated with the pinned
  DSV4 torch runtime.
- DeepSeek V4 parser focused test: passed.
- RunAI streaming loader focused suite: passed.
- KTransformers port and Python RAWINT4 loader/backend tests: 8 passed.
- AVX-VNNI per-expert native equivalence test requires rebuilding the KT native
  extension; the installed pre-audit extension aborts on the new ABI and is
  therefore not claimed as validated here. It is unrelated to native MXFP4.
- shell syntax, source patch reverse-check, model revision/shards/size, expert
  mask shape/count, and `git diff --check`: passed.
- No model process, benchmark request, or warmup was run.

Repository-wide Ruff traverses the accumulated vendored SGLang,
KTransformers, and llama.cpp trees and reports 17,285 pre-existing findings;
the new preparation helper and its test pass focused Ruff. Unrestricted pytest
likewise enters vendor test trees and stops during collection on missing
optional pybind11 and Mooncake test modules. `nix fmt` was requested as required
by the repository instructions but Nix is not installed on dwagon. No automatic
formatting rewrite was applied.

## Remaining performance validation after an authorized launch

1. Re-record 0731 routes on the intended agentic workload and compare hot12,
   hot32, and hot48 on TP2.
2. Re-profile the TP2 compact-ragged SPS table after the final kernels are
   built.
3. A/B breakable prefill graphs against eager prefill at 2K, 32K, and 64K.
4. Confirm exact-output repeatability before accepting a speed result.
5. Measure TP2 hot48 against the remote-style single-GPU hot12 fallback on the
   same prompt and output length.
6. Sweep defer0/defer1/defer2 only after the route profile is current.

Until those measurements exist, “fastest expected” means the strongest
evidence-backed launch configuration, not a new benchmark claim.
# Cumulative fork and dependency graph update (2026-08-01)

The audited runtime sources are now published as public forks and referenced by
the local cumulative graph:

- `ldyeax/exo_sglang`, branch `bundle/glm52-fwuff-sglang`, commit
  `73e877ac5` (with the DeepSeek V4 integration branch
  `exo/dsv4-flash-0731` at `40e43604a`).
- `ldyeax/exo_ktransformers`, branch `exo/glm52-osdi26-patched`, commit
  `f38772417`.
- `ldyeax/exo_llama_cpp`, branch `exo/kimi-k3-cumulative`, commit
  `651092c60`.

KTransformers no longer pins its active llama dependency to the unrelated
ancient shallow snapshot. It uses the cumulative llama fork and contains the
small compatibility layer required by modern ggml. Its active and archived
llama submodule declarations point to the same fork/branch; its SGLang
submodule points to `exo_sglang`. Exo also tracks `exo_llama_cpp` directly at
`vendor/llama.cpp`, preventing the llama work from being siloed under
KTransformers.

System validation used CUDA 13.1/13.3 from `/opt/cuda`, system-wide ccache
4.13.5, and system-wide NCCL 2.30.7 installed under `/usr/local`. The cumulative
llama CUDA/NCCL targets and focused DFlash/RPC/backend tests passed, and
KTransformers built successfully in both CPU-only and SM86 CUDA modes against
the new llama revision. No model launch was performed.
