# DeepSeek V4 Pro notes

- Use DSpark.

## Local modified inference sources

The OSDI26 GLM-5.2 runtime sources are checked out inside this repository as
nested local submodules:

- `vendor/ktransformers` is pinned to integration commit
  `45a3a797658140dcf8426cd3f2b2c6c969f8f5d8`. The code-bearing
  KTransformers patch terminal is
  `2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a`; the integration commit only
  changes the nested SGLang URL and advances its gitlink.
- `vendor/ktransformers/third_party/sglang` is pinned to
  `720b40b2783b1a515134f4ab9fe820931cfbee36`, which contains all nine
  recorded SGLang patches.

These are local-only submodules for now. Their URLs point to durable bare
repositories under `/var/lib/exo/sources`; replace those URLs when remote
forks are created.

The KTransformers history adds the AMXINT4 expert runtime used by the hybrid
path, fine-grained AMX decode dependencies, BF16 expert staging/export for SLP
and SmallEP, immutable shared host-weight mappings and leases for P/D, and
pins the integrated SGLang runtime.

The SGLang history adds GLM-5.2 TP/PP integration, two-batch attention/CPU-MoE
overlap, layer-78 MTP with persistent AMXINT4 experts, bounded SLP and
SmallEP/P-D plumbing, direct compact W8A16 MLA `kv_b_proj` execution on
Ampere, coherent FlashInfer metadata, shared Marlin module ownership, and the
standalone `fwuff` TP1 MTP drafter. New compact launches are Marlin-only;
the retained Triton source is historical cross-check code, not the supported
policy.

The checkpoint layout keeps routed and MTP experts in AMXINT4, large
GPU-resident matrices in compact weight-only INT8, and norms, sensitive
scalars, activations, and the initial KV cache in BF16. Do not expand compact
weights back to BF16 at load.

## 2026-07-25 — Native MXFP4 deployment work

### Contract and starting state

- Target: serve DeepSeek V4 Pro with its native MXFP4 expert weights, DSpark
  multi-token prediction, full hardware use, and an InfiniBand-assisted
  `fwuff` expert tier. The weights must not be converted to AMXINT or further
  quantized.
- Performance acceptance: at least 20 token/s single-stream decode and at
  least 500 token/s prefill, with real serving profiles for 1 user at 1M
  context and 4 users at 256K, including 32K fresh-prefill and 4K decode
  workloads.
- Bring-up order: prove the same native-MXFP4 execution path on the already
  staged DeepSeek V4 Flash checkpoint before attempting the approximately
  800–850 GB Pro checkpoint.
- Starting checkout: branch `agent/linux-cuda-nccl`, clean worktree, two
  commits ahead of `origin/agent/linux-cuda-nccl`.
- The pre-existing local KTransformers/SGLang integration is an AMXINT4/INT8
  GLM-5.2 path. It is useful reference code but **does not satisfy** this
  task's native-MXFP4 format constraint and must not be reported as such.

### Documentation/coherency note

- `infiniband_card_limitation.md` describes the original fwuff ConnectX-3
  OEM-QDR personality and its approximately 32 Gb/s card-wide ceiling.
  `infiniband_cards.md` records the later cross-flash to generic FDR firmware
  and measured approximately 52.9 Gb/s dual-port same-direction payload.
  Fresh hardware/firmware and counter receipts are required before relying on
  either historical state.

### Work log

- Read the repository agent/rules files, the existing task journal, the local
  hardware/network deployment notes, the OSDI hybrid-MoE work, the
  InfiniBand/speculation notes, and the KTransformers V4-Flash/AMX guidance.
- Fresh `dwagon` receipt: 112 physical cores/224 threads across two NUMA
  nodes, AMX BF16/INT8 plus AVX-512 BF16/VNNI/FP16, 755 GiB RAM with
  approximately 730 GiB available, two idle 24 GiB RTX 3090/SM86 GPUs on
  separate NUMA nodes with NV4 active, and no model process using either GPU.
- Fresh `fwuff` receipt: 60 physical cores/120 threads on one NUMA node with
  the same AMX/AVX-512 features, 247 GiB RAM with approximately 221 GiB
  available, one RTX 3090/SM86, and `/mnt/sanic` on 5.6 TiB RAID0 with
  approximately 1.3 TiB free.
- Fresh fabric receipt: ConnectX-5 `mlx5_0` is ACTIVE at 100 Gb/s on both
  hosts; both ConnectX-3 `mlx4_0` ports are ACTIVE at 40 Gb/s. IPoIB addresses
  `10.44.0.1/2`, `10.44.1.1/2`, and `10.44.2.1/2` remain configured.
- Resource-coherency caveat: fwuff still has the prior GLM-5.2 remote-draft
  service resident (about 4.3 GiB VRAM) and an active but model-empty Ollama
  daemon. They must be cleanly released under the deployment lease before a
  V4 benchmark can claim full-hardware ownership.
- Artifact inventory: V4 Pro is not present under `/mnt/sanic`; V4 Flash is
  present at `/mnt/sanic/llm_models/DeepSeek-V4-Flash`, revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`, with 46 indexed shards,
  69,187 tensors, and 159,609,485,896 indexed bytes. Its config specifies
  `expert_dtype=fp4`; the model card identifies this as FP4 expert plus FP8
  non-expert mixed precision.
- Initial source-history discovery found KTransformers PR/merge `#1970`
  (`041bdfc`, native V4-Flash MXFP4), native AVX-512 MXFP4 PR `#2006`
  (`f077244`), AVX2 MXFP4 dispatch PR `#2015` (`ef6c47f`), and kvcache-ai
  SGLang fork Ampere V4-Flash PR `#58` (`37eecb4e9`). Exact current upstream
  PR states and patch contents still need to be bound before integration.
- Remaining: upstream PR audit, runtime/source integration, native MXFP4
  kernel verification (including executed AMX and AVX-512 evidence), model
  integrity/coherency tests, Flash deployment and benchmarks, Pro download,
  distributed expert-tier implementation, full deployment, and acceptance
  benchmarks.

### Upstream source audit and pinned artifacts

- KTransformers upstream `main` was fetched at
  `a8062bfa7e1060ce5855b5f1ad6aa6b116678307` (release 0.6.4). Native
  DeepSeek-V4 MXFP4 support is merged through PRs `#1950`, `#1957`, `#1970`,
  `#1980`, `#2006`, and `#2015`. PR `#2085` remains open and avoids redundant
  host copies of GPU-assigned experts; it is relevant to the Pro memory
  budget but is not yet an upstream-stable dependency.
- The current merged KTransformers native-MXFP4 CPU path is AVX-512/AVX2
  despite the `amx/fp4-moe.hpp` filename and wrapper class names. The true
  AMX-tile proposal in closed PRs `#2010`/`#2014` was deliberately removed
  before `#2015`: applying one scale per 32-value MXFP4 group forced a
  tile-zero/dot/store cycle per group and was slower than the direct AVX-512
  kernel. That removed implementation will not be misrepresented or simply
  resurrected.
- KTransformers PR `#2015` reported 574.7 token/s at 2K and 805.3 token/s at
  4K prefill, but only 18.9–19.3 token/s decode on its single-RTX-5090 test.
  This is useful direction, not acceptance evidence for this SM86 system.
- kvcache-ai/SGLang `main` was fetched at
  `04653fa88f5ce4ea83632c4ad436343db6bc8324`. Its merged PR `#58`
  (`37eecb4e9`) is the required DeepSeek-V4 Ampere/SM86 attention and cache
  support. PR `#62` adds the compatible Ampere Marlin path for FP8
  non-expert weights while explicitly leaving native MXFP4 experts on their
  MXFP4 path. No DSpark implementation exists in this fork.
- sgl-project/SGLang `main` was fetched at
  `2cbddb842d67b7d16f04c5a7856a0ff9bddc7767`. The original DSpark PR
  `#29538` closed unmerged; generalized DSpark PR `#30261` merged at
  `6cc935c` and subsequent fixes are present on current `main`. No separate
  sgl-project PR supersedes kvcache-ai PR `#58` for the SM86 native-MXFP4
  KTransformers deployment.
- Exact checkpoint pins and sizes from the Hugging Face API:
  Flash-DSpark is revision `62af8fffb2f7030cac4de2f0169f5b8d1101b646`
  (about 166.9 GB); Pro-DSpark is revision
  `7c09739fd136abfb70a49ec334157f65f45b52cd` (about 892.7 GB).
  Both declare `expert_dtype=fp4`, BF16 activations, FP8 non-expert
  quantization, one MTP layer, and a 1,048,576-token maximum position.
  Pro-DSpark has 61 layers, 384 routed experts, hidden size 7168, and routed
  expert width 3072.
- A pinned, resumable `exo-dsv4-download.service` was started on `fwuff`.
  It stages Flash-DSpark first into
  `/mnt/sanic/llm_models/DeepSeek-V4-Flash-DSpark`, then Pro-DSpark into
  `/mnt/sanic/llm_models/DeepSeek-V4-Pro-DSpark`. Starting free space was
  1,319,853,957,120 bytes, sufficient for both exact artifacts with roughly
  260 GB remaining.
- The user confirmed that the prior fwuff session is over and authorized
  stale-process cleanup. The GLM-5.2 remote drafter (PID 2826982, 4,276 MiB
  GPU memory) was terminated and the model-empty `ollama.service` was
  stopped. The DeepSeek download service remained active. A post-cleanup
  receipt showed no GPU compute processes and approximately 229 GiB host
  memory available on fwuff.

### Kernel direction

- Decode will retain and optimize the direct AVX-512 BF16-dot MXFP4 kernel,
  which matches the format's small-M execution shape.
- Prefill needs a genuinely different high-M path to satisfy the AMX
  requirement without repeating the rejected per-group tile cycle. The
  implementation direction is bounded, transient MXFP4 dequantization with
  fused UE8M0 scaling into pipelined BF16 weight tiles, allowing AMX BF16
  accumulation across useful K/M blocks before storing. MXFP4 remains the
  authoritative persisted weight format; no converted weight checkpoint or
  persistent secondary expert copy is permitted. Numerical results must be
  checked against the existing native-MXFP4 reference path before serving.

### First native-MXFP4 AMX kernel

- Added a 32-token by 32-output high-M kernel to
  `kt-kernel/operators/amx/fp4-moe.hpp`. Each K=32 group is decoded directly
  from the authoritative nibble-packed E2M1 weights, fused with its existing
  scale into bounded BF16 scratch tiles, VNNI-transposed, and consumed by four
  `TDPBF16PS` operations. Four FP32 AMX accumulators stay resident across the
  entire K dimension and are stored once. No persistent dequantized expert
  copy or alternate checkpoint is created.
- Small-M execution remains the direct AVX-512 BF16-dot MXFP4 path. Runtime
  dispatch defaults to AMX at eight tokens assigned to an individual expert;
  `KT_MXFP4_AMX_MIN_EXPERT_TOKENS` permits a measured threshold override.
  Separate atomic dispatch counters were added so executed backends can be
  attested.
- Added `test_mxfp4_amx_prefill.cpp`. Synthetic native-MXFP4 parity against
  the existing AVX-512 reference passed at M=8, 17, 32, 47, and 64. Relative
  L2 was at most `8.91e-9`; maximum absolute difference was
  `5.9604645e-8`. Disassembly of the executed binary contains `LDTILECFG`,
  `TILELOADD`, `TDPBF16PS`, `TILESTORED`, and `TILEZERO`.
- Added `benchmark_mxfp4_amx_prefill.cpp`. A core-pinned single-thread
  microbenchmark for one 256-output expert partition measured the following
  kernel crossover on dwagon:

  | Expert M | Flash K=4096 speedup | Pro K=7168 speedup |
  | ---: | ---: | ---: |
  | 1 | 0.674x | not measured |
  | 4 | 0.912x | not measured |
  | 8 | 1.813x | 1.811x |
  | 32 | 7.400x | 7.461x |
  | 128 | 7.361x | 7.306x |

  At M=32–128 the AMX path sustained approximately 153–156 GFLOP/s versus
  approximately 21 GFLOP/s for the current AVX-512 high-M path. This supports
  the conservative default crossover of eight while preserving AVX-512
  decode/small-expert performance.

### Flash-DSpark artifact coherency

- Flash-DSpark download completed at exact revision
  `62af8fffb2f7030cac4de2f0169f5b8d1101b646`; the per-file Hugging Face
  metadata records the same revision.
- Validated every safetensors header and data extent and cross-checked every
  key against `model.safetensors.index.json`: 48/48 shards, 72,317 unique
  tensors, 166,886,535,336 shard bytes, and 166,878,536,440 indexed tensor
  bytes. No `.incomplete` files remain.
- The staging service advanced to Pro-DSpark automatically; the first
  approximately 102 GiB was present when Flash validation completed.

### Built runtime and checkpoint-level kernel coherency

- Created an isolated copy-on-write runtime at
  `/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv`;
  prior GLM runtimes remain untouched.
- Built the modified KT-Kernel extension in Release mode with native CPU
  targeting, AMX BF16/INT8 tiles, AVX-512 BF16/VNNI/VBMI, and no CPU-feature
  fallback substitution. The installed runtime selects the AMX variant and
  exports `AMXFP4_KGroup_MOE`. Static disassembly contains 76 `TDPBF16PS`
  sites and 695 AVX-512 `VDPBF16PS` sites.
- Ran layer 1, one real Flash-DSpark expert, and all gate/up/down projections
  through the full KTransformers MoE wrapper at Q=8. The upstream Torch
  reference assertion reports a misleading 44.137% relative-mean error
  because its reference activations produce near-zero outputs. This is not an
  AMX regression: forcing the exact request to AVX-512 yields identical mean
  and maximum errors and the same complete BF16 output SHA-256,
  `2a355ed039db34b946bb95cfbc3f6bb4cf58913404bfb1c7cc541ee5e0cfc9ca`.
  The executed AMX and forced-AVX checkpoint outputs are therefore
  byte-for-byte identical.

### First full Flash-DSpark bring-up

- Started an isolated, transient TP=2 SGLang/KTransformers service on dwagon
  using the exact Flash-DSpark checkpoint as both the model and
  KTransformers weight source. The first conservative baseline places zero
  experts on GPU so that native CPU MXFP4 execution can be validated before
  the Ampere GPU-expert patch changes the execution mix. It assigns 104 CPU
  inference threads across both NUMA nodes and enables the new
  `KT_MXFP4_AMX_MIN_EXPERT_TOKENS=8` crossover.
- Both SM86 RTX 3090 ranks initialized and each held approximately 8.2 GiB.
  SGLang recognized all 43 MoE layers and 256 native MXFP4 experts per layer.
  All 48 checkpoint shards loaded without error in 2 minutes 5 seconds.
  Post-load CPU initialization was still active at this checkpoint; host
  memory availability remained approximately 725 GiB.
- The pinned Pro-DSpark download reached approximately 707 GiB. The fwuff
  RAID still had approximately 376 GiB free, which leaves sufficient margin
  for the checkpoint's approximately 893 GB final size.

### Pro-DSpark download completion and artifact coherency

- The pinned Pro-DSpark download completed at revision
  `7c09739fd136abfb70a49ec334157f65f45b52cd`. The finished directory occupies
  892,762,513,519 bytes including metadata and contains no `.incomplete`
  files.
- Parsed and validated every native safetensors header and data extent, then
  cross-checked every tensor against `model.safetensors.index.json`: 66/66
  shards, 149,782 header tensors, 149,782 index tensors,
  892,744,322,880 shard bytes, and 892,727,580,904 indexed tensor bytes.
  There are zero missing or extra tensor mappings and zero malformed or
  truncated shard extents. The index's declared total size is exactly
  892,727,580,904 bytes.

### Flash runtime integration findings

- Full TP=2 load now completes consistently. The GPU/non-expert rank loads in
  approximately 26–32 seconds; constructing all 43 dual-NUMA native-MXFP4 CPU
  expert layers takes approximately 106–115 seconds. Model memory settles at
  approximately 7.5 GiB per GPU before the compressed-attention cache pool is
  allocated.
- DeepSeek V4 compressed-attention sizing at the initial 16K bring-up setting
  reports capacity for 647,424 full tokens per rank with FP8 KV cache and
  FP32 recurrent state. This is a useful receipt, not yet the required 1M
  serving validation.
- Repaired a TileLang/TVM-FFI 0.1.11 compatibility failure by making
  `TVMDerivedObject._inst` a real weak-referenceable slot and routing its
  assignments through `object.__setattr__`. A standalone real V4 MHC
  `hc_split_sinkhorn` graph subsequently compiled and passed with finite
  output, row/column errors near `1e-6`, and deterministic output SHA-256
  `b6d6...`.
- SGLang's JIT wrappers redundantly compiled kernels already exported by its
  matching `sgl_kernel` wheel. The independently embedded TileLang CUDA
  runtime stubs could not see PyTorch's locally scoped CUDA runtime on SM86.
  The GPTQ-Marlin repack, GPTQ-Marlin GEMM, and five Hadamard transforms now
  prefer their exact AOT `sgl_kernel` operators and retain JIT fallback when
  those exports are unavailable. Repack is bit-exact against the CPU
  reference; the AOT Hadamard is exact against the direct wheel call, retains
  norm to `1e-5`, and has FP16 involution error below `9.77e-4`.
- The remaining V4-specific JIT modules need their own CUDA stub, so the
  runtime's TileLang stub source now accepts `TILELANG_LIBCUDART_PATH` and
  explicitly `dlopen`s that ABI-matching runtime when `RTLD_DEFAULT` and
  `RTLD_NEXT` cannot find it. The exact previously failing
  `topk_transform_512` V4 kernel was compiled and executed standalone on
  SM86 after this fix; all 512 output and raw indices were valid and unique.
- The KT extension had initially been built CPU-only, hiding
  `submit_with_cuda_stream` and `sync_with_cuda_stream` even though current
  source implements both. It was rebuilt with native AMX/AVX-512 CPU flags,
  CUDA stream integration, and SM86 targeting, then installed into the
  isolated runtime. The source and installed binaries have matching SHA-256
  `63863bde94e2ebdf7af929188dbd3f23c647a32e13a2e05ddd8a7d7666c8af9b`;
  both stream methods are exported. Disassembly still attests both AMX
  `TDPBF16PS` and AVX-512 `VDPBF16PS`.
- Graph capture reached native CPU MoE execution and logged the first real
  AVX-512 decode dispatch at M=1. Earlier launches were intentionally failed
  and drained after exposing the loader/API mismatches; neither GPU retains a
  stale process.
- Bring-up v10 is running with the fixed embedded-stub loader, CUDA-enabled
  KT extension, native MXFP4 experts, and the eight-token AMX crossover.

### First live Flash service and native-path baseline

- A stale V4 top-k JIT cache entry predated the CUDA-loader repair, because
  the TileLang cache key does not include its runtime-stub source. Quarantined
  that single cache directory and rebuilt the module. The rebuilt SM86 module
  directly links the installed CUDA 13 runtime; the deployment preloads the
  same runtime globally so TileLang's device API can resolve it in spawned
  tensor-parallel ranks.
- Bring-up v11 completed: both TP ranks captured batch-size-one CUDA graphs in
  approximately 34 seconds, registered 87 graph addresses, and the SGLang
  HTTP service became ready on `127.0.0.1:30000`. `/health`,
  `/get_model_info`, and `/generate` all passed. The served architecture is
  `DeepseekV4ForCausalLM` from the exact Flash-DSpark model directory.
- Repeated temperature-zero inference produced identical text and identical
  output-token IDs. The output-ID SHA-256 was
  `2e0994ba929172237f847779fac35378b6371ec5cffa20e53582df2fc6780a36`
  on both cold and warm requests.
- Added a narrow `bench_serving` tokenizer fix: when a new, unregistered model
  architecture explicitly ships a generic `PreTrainedTokenizerFast`, the
  benchmark loads that self-contained tokenizer without first requiring
  `AutoConfig` support for the model. This makes SGLang's required serving
  benchmark usable with the V4 checkpoint and does not alter server
  inference.
- Live native-kernel receipts now cover every required CPU execution class:
  first AVX-512 decode at expert M=1, first AVX-512 low-occupancy prefill at
  M=3, and first AMX prefill at M=10 with the crossover set to eight.
- Zero-GPU-expert baseline, after one warmup request:

  | Served workload | Result |
  | --- | ---: |
  | 512 input / 64 output, TTFT | 2,258.52 ms |
  | 512 input / 64 output, mean TPOT | 60.92 ms (16.41 token/s) |
  | 2,048 input / 8 output, TTFT | 4,879.90 ms |
  | 2,048 input / 8 output, input throughput | 380.21 token/s |
  | Warm deterministic 18 input / 32 output, E2E | 2,169.59 ms |

  These are coherent baselines, not acceptance results: decode remains below
  20 token/s and prefill remains below 500 token/s.
- Bring-up v12 moves 32 native checkpoint experts per layer onto the two SM86
  GPUs and enables runtime hot-expert replacement, while leaving the other
  224 experts in authoritative native MXFP4 form on the dual-NUMA AMX/AVX-512
  tier. The 32-expert choice uses approximately 9.20 GB of sharded expert
  storage per GPU and is intended to exploit the otherwise idle GPUs without
  hiding CPU-kernel performance.
- V12 became ready with 206,336 full-token cache slots per TP rank. Its first
  performance pass, using 2,048-token prefill chunks, measured:

  | Served workload | Result |
  | --- | ---: |
  | 512 input / 64 output, mean TPOT | 43.26 ms (23.12 token/s) |
  | 512 input / 64 output, TTFT | 2,073.48 ms |
  | 2,048 input / 8 output, input throughput | 410.01 token/s |
  | 2,048 input / 8 output, mean TPOT | 58.58 ms (17.07 token/s) |
  | 4,096 input / 8 output, input throughput | 428.21 token/s |
  | 4,096 input / 8 output, mean TPOT | 48.42 ms (20.65 token/s) |

  This crosses the decode target at the 512- and 4,096-token checkpoints but
  is not accepted: prefill is still below 500 token/s.
- The post-benchmark coherency gate rejected v12. Four repeated greedy
  requests were internally deterministic, but their output-ID SHA-256 was
  `b5102498f4c463177db497e8ddd8beaa93f226d884febbb3f6a2d25ee58f0df0`
  rather than the zero-GPU-expert baseline
  `2e0994ba929172237f847779fac35378b6371ec5cffa20e53582df2fc6780a36`;
  the generated answer changed materially. The GPU-expert performance numbers
  therefore remain diagnostic only until static SM86 expert execution and the
  dynamic expert remapper are isolated.
- V13 disabled dynamic expert replacement while retaining 32 static GPU
  experts per layer. It produced the same divergent output hash as v12, so
  the remapper is exonerated and the difference is isolated to using the SM86
  GPU expert compute path. A 4,096-token AMX chunk reduced
  4,096-input / 8-output TTFT to 7,918.52 ms, equivalent to 517.28 input
  tokens/s before the first token; SGLang's whole-request input-throughput
  metric was 493.39 token/s and TPOT was 47.14 ms (21.21 token/s). These
  results are diagnostic only because the coherency gate failed.
- A stricter layer-level numerical oracle then showed that whole-model token
  bit-equality was the wrong acceptance criterion for mixing CPU and GPU
  reductions. The portable GPU path's individual GEMM1, SwiGLU, and GEMM2
  stages were bit-exact against direct native-MXFP4 dequantized computation.
  On real Flash layer-1 checkpoint weights, six routed experts, and token
  counts 1, 8, and 32, SM86 GPU versus AVX-512/AMX CPU output had maximum
  absolute error `0.0009765625` and relative L2 error at most
  `0.00318981`. This validates weights, E2M1/UE8M0 scale interpretation,
  routing, SwiGLU clamp, and output reduction. Added a CUDA regression test
  for the portable native-MXFP4 MoE path; it passes on SM86. Whole-model
  acceptance will therefore require this bounded layer error, deterministic
  serving, and semantic probes rather than cross-backend output-ID identity.
- V14 advertises the checkpoint's full 1,048,576-token context while exposing
  the current TP2/GPU32 cache limit honestly: 204,032 resident full-token
  slots per rank. OpenAI chat-template probes passed (`2 + 2` returned exactly
  `4`; the MXFP4 explanation was correct), and three repeated greedy chat
  requests produced identical content SHA-256
  `697679bc881123d0ce333b42ac72f15ea840227fef894bf25403d94e12f77921`.
  The earlier divergent raw `/generate` text was a continuation of an
  untemplated base string, not a valid chat semantic test.
- V14's required 32,768-fresh-token / 64-output served benchmark completed
  after a warmup request. TTFT was 66,065.20 ms (495.99 fresh token/s),
  whole-request input throughput was 471.47 token/s, and TPOT was 53.62 ms
  (18.65 token/s). This is close but still below both strict acceptance
  thresholds at the required fresh-context length, so the next run will
  profile the CPU/GPU overlap and replace uniform GPU experts with measured
  hot-expert placement.

### Profile-guided native-MXFP4 placement

- V15 increased static SM86 residency from 32 to 36 experts per layer and
  enabled the upstream expert-distribution recorder after adding the missing
  V4 layer context. OpenAI chat probes remained deterministic and returned
  exactly `4` for the arithmetic probe. A matched 4,096/64 benchmark regressed
  to 361.49 whole-request input token/s and 54.34 ms TPOT; its usable KV pool
  also fell to 120,320 tokens.
- V15 then failed a 32K warmup coherently with a CUDA OOM in the MHC post
  workspace: GPU 0 had only 96.94 MiB free when another 256 MiB was required.
  This establishes 36 experts/layer as overpacked on 24 GiB SM86 rather than
  treating the interrupted HTTP stream as a serving result. The failed unit
  and its orphan rank were drained, returning both GPUs to 1 MiB.
- The V15 recorder dump is valid: 104 populated forward rows over 43 layers
  and 256 logical experts. A compact 4K profile was saved as
  `/var/lib/exo/profiles/dsv4-native-mxfp4/flash-v15-4k-hot-experts.pt`.
  Thirty-six hot experts cover approximately 71–81% of routed calls in
  sampled layers, showing that the previous trivial expert IDs left most
  GPU locality unused.
- V16 returned to the safe 32-expert-equivalent budget (1,376 expert slots
  globally) and used KTransformers' frequency strategy with that measured
  native-MXFP4 profile. The placement varies by layer rather than wasting an
  equal quota: zero slots on the three non-routed front layers and 24–50
  slots on sampled routed layers. It retained 200,448 resident full-token KV
  slots and passed repeated OpenAI arithmetic probes.
- After one-time Triton shape compilation, V16's matched 4,096/64 result was
  511.18 whole-request input token/s, 5,097.55 ms TTFT, and 45.54 ms TPOT
  (21.96 token/s). This is the first coherent single-stream run to cross both
  requested throughput thresholds at once.
- V16's required 32,768-fresh-token / 64-output run measured 626.68
  whole-request input token/s, 48,713.66 ms TTFT (672.68 fresh token/s by
  TTFT), and 55.73 ms TPOT (17.94 token/s). Prefill now clears the target,
  while long-context decode remains 11.5% short.
- The bounded 512-row recorder avoided the V15 profiling-memory mistake and
  cleanly separated 16 prefill chunks (1,966,080 routed calls each) from 96
  decode steps (480 calls each). The old 4K placement covers only 50.71% of
  the observed 32K decode calls. A decode-ranked placement with the identical
  1,376-slot memory budget covers 75.93% and also improves 32K-prefill
  coverage from 57.81% to 65.80%. Compact decode and prefill profiles are in
  `/var/lib/exo/profiles/dsv4-native-mxfp4/`.
- V17 loaded the decode-ranked profile without increasing GPU expert storage.
  It retained 204,288 resident full-token slots and passed two repeated
  chat-template arithmetic probes. Its required 32,768/64 benchmark measured
  678.96 whole-request input token/s, 45,196.31 ms TTFT (725.0 fresh token/s
  by TTFT), and 47.87 ms TPOT: **20.89 token/s single-stream decode**. This is
  the first coherent required-length Flash result to clear both 500 prefill
  token/s and 20 decode token/s.

### Pipeline coherency work

- V4's model class advertised PP arguments but embedded and normalized on
  every rank, always produced logits, and exposed no stage range to the
  model runner. It now follows SGLang's PP proxy contract: only the first
  stage embeds and expands MHC channels, intermediate stages pass
  `hidden_states`, and only the last stage performs CP gather, HC collapse,
  final normalization, and logits. Non-local embedding, HC, norm, and head
  weights are skipped during checkpoint load.
- DeepSeek V4 compressed-cache sizing and allocation are now stage-local.
  The memory calculator slices the checkpoint compression schedule to
  `[start_layer, end_layer)`, the SWA pool maps global IDs to local buffers,
  and compressed KV/state mappings retain global layer IDs while allocating
  only local c4/c128 layers. A synthetic SM86 arithmetic/mapping check passed:
  the six-layer full schedule costs 2,096 bytes/full-token versus 1,224 for
  the selected three-layer stage, with exact local compressed IDs.
- The exact patched 207.5 MB source tree was transferred to fwuff over
  `10.44.0.1 ↔ 10.44.0.2` InfiniBand at 416 MB/s. A native fwuff overlay was
  built with AMX, AVX-512 BF16/VNNI/VBMI, CUDA stream integration, and SM86
  code. Its extension SHA-256 is
  `a8e7fb4d56b8f66047cf4830096eb59d60d88428ea740d3291773563ad10469b`;
  both CUDA stream APIs are exported, and disassembly finds 835 AMX/AVX-512
  BF16 dot-product instructions.
- The first live PP2 attempts exposed four independent upstream assumptions,
  each at a clean initialization boundary: weight accounting treated every
  norm as final-stage-only; post-load hooks walked placeholder layers;
  `PPProxyTensors` was imported only for type checking; and CUDA-graph proxy
  buffers assumed a two-dimensional hidden state. The loader and hooks are
  now stage-local, the proxy type is imported at runtime, and V4 declares its
  authentic `[token, hc_mult, hidden_size]` MHC pipeline tensor shape.
- fwuff is one 60-core/120-thread NUMA node. Its native AMX pool now uses one
  60-physical-core subpool on NUMA 0 rather than two subpools or 112 logical
  CPU IDs; the latter produced explicit out-of-topology core warnings.
- Both PP stages loaded only their assigned layers (dwagon 0–20, fwuff
  21–42), including native MXFP4 experts, and the exact compressed-cache
  calculator proved capacity above one million tokens. The generic dense-KV
  pre-clamp was removed for DSV4, after which both stages allocated exactly
  1,048,576 full-token slots. PP0 uses 9,748.8 bytes/full-token and PP1
  11,070.4; the resulting common pools are 104,704 SWA, 262,144 c4, 8,192
  c128, 6,544 c4-state, and 104,704 c128-state slots.
- dwagon captured the shaped V4 CUDA graph in 5.94 seconds with 4.54 GiB
  still available. fwuff then isolated a CUDA 12.4 source-portability defect:
  its compiler rejected C99 indexed array initializers accepted by CUDA 13.
  The fused norm/RoPE dispatch table now uses equivalent ordered C++
  initialization. A direct fp32/head-512/rope-64 SM86 JIT build passed on
  fwuff before the coordinated relaunch.
- The subsequent PP1 graph reached the first real cache store and exposed one
  remaining global-layer assumption in the SWA setter. Reads, ordinary writes,
  and fused writes now share a checked global-to-stage-local mapper. Focused
  tests cover the `[21, 23)` synthetic stage and both write paths; they pass,
  and the patched source has the same SHA-256
  (`2c9f96db498f6fc0aa041d916743e8d6602005b96f7a833b50ab5001be91e30c`)
  on both hosts.
- The repaired 21/22 PP2 deployment initialized coherently, allocated exactly
  1,048,576 full-token slots on both ranks, captured both CUDA graphs, and
  returned exactly `4` for two deterministic arithmetic probes. The required
  32,768/64 served benchmark then measured 769.33 whole-request input token/s,
  38,646.68 ms TTFT (847.9 fresh token/s), and 62.46 ms TPOT
  (16.01 token/s). Prefill clears the target, but decode is slower than V17.
- NCCL confirms `NET/IB` on `mlx5_0:1` with OOB on `ibs5`/`ibs2`; the rings
  use InfiniBand RDMA but report GDR disabled, so Linux netdev byte counters
  do not represent payload traffic. The current equal layer split makes the
  60-core fwuff rank carry 22 layers while the 104-core dwagon pool carries
  21. The next deployment uses SGLang's supported explicit PP partition,
  shifts layers toward dwagon, and spends fwuff's released GPU memory on a
  larger hot-expert set.
- V18 used a 26/17 split, 4/16 GPU expert budgets, and loaded exactly layers
  0–25 / 26–42. Both stages again allocated the full 1,048,576-token pools;
  after graph capture dwagon retained 1.87 GiB and fwuff 5.86 GiB. Two short
  probes returned exactly `4`, but a 32K request coherently failed on PP0:
  PyTorch's unfused MHC-post expression materialized a 256 MiB contraction
  temporary with only 131 MiB physically free. The configuration is rejected
  and has no benchmark result.
- The existing TileLang MHC-post kernel initially failed on this TileLang
  build because an optional register-usage pass was serialized into `ptxas`
  as the literal `ir.IntImm(...)`. Removing that hint from the MHC kernels
  preserves their launch bounds and makes the fused post compile on both
  CUDA 13/dwagon and CUDA 12/fwuff for SM86. Both hosts match the PyTorch
  oracle with 0.015625 max absolute BF16 error and 5.02e-08 mean error. At the
  real 4,096-token `[4096,8,4096]` shape, peak incremental allocation is
  exactly the 268,435,456-byte output, eliminating the failed expression's
  additional 256 MiB temporary. The synchronized source SHA-256 is
  `f0fbd250a07c9a491d35135d7cba6a7171b762cae3cdbeaf75314e6e8d33c189`.
- V19 enabled only the validated fused MHC-post path with the same 26/17 and
  4/16 hot-expert split. It retained exact 1M capacity, passed two repeated
  arithmetic probes, and completed the formerly failing 32,768/64 request:
  541.46 whole-request input token/s, 56,756.02 ms TTFT (577.3 fresh
  token/s), and 59.42 ms TPOT (16.83 token/s).
- Mellanox hardware counters, rather than Linux netdev counters, measured
  1,087,035,488 bytes from dwagon and 1,087,048,784 bytes received by fwuff
  over the full profiled request, only 14.67 MiB/s including client setup.
  This matches the four-channel BF16 MHC activation volume and is far below
  the 100 Gb/s link. Both AMX CPU pools reached their configured core counts;
  communication is not the active limit. V20 therefore keeps the safe PP0
  memory layout and uses fwuff's 5.86 GiB post-graph reserve to raise its
  native-MXFP4 hot-expert budget from 16 to 32.
- V20's 32-expert fwuff rank loaded coherently and still had 2.56 GiB after
  graph capture, but it exposed a heterogeneous-PP capacity defect before
  benchmarking. SGLang globally reduced raw free bytes; fwuff's larger
  resident set supplied the byte minimum, which PP0 then divided by its
  larger 12,281.6 bytes/token and incorrectly advertised 947,200 tokens.
- DeepSeek V4 profiling now computes each stage's local token capacity first
  and all-reduces the token count, after which every stage allocates the
  common count using its own bytes/token. A focused two-rank synthetic test
  proves 2,048 local tokens reduce to a simulated 1,024-token remote
  capacity, and all three PP cache/capacity tests pass. The synchronized
  profiler SHA-256 is
  `6550f941a998821fd63f6bffc5f90b8caaffe745a5a85ea4e5e7ca3604546ba4`.
- V21 proved the corrected collective live with asymmetric free memory: PP0
  profiled 1,210,624 tokens, PP1 profiled 1,362,688, and both allocated the
  configured 1,048,576 after reducing token capacity. Its 32K/64 result was
  543.84 input token/s, 56,381.91 ms TTFT, and 61.23 ms TPOT
  (16.33 token/s). Doubling fwuff residency from 16 to 32 regressed decode
  from V19's 16.83 token/s, so 32 is rejected and the default returns to 16.
- Upstream DSpark commit `6cc9352df` cannot be cherry-picked onto the pinned
  KTransformers SGLang tree safely: an isolated no-commit transplant produced
  43 core-runtime conflicts spanning the scheduler, attention backends, graph
  runners, memory pools, server arguments, and the locally repaired V4 model.
  Copying only the newly added DSpark files also fails closed on prerequisite
  modules introduced between the fork's February merge base and July main.
- A clean checkout at `6cc9352df` imports its KTransformers wrapper, V4 target,
  V4 DSpark draft, and confidence-scheduled DSpark worker successfully against
  the existing native `kt_kernel` runtime. This establishes a lower-risk
  integration route: keep the proven PP target runtime, build a July DSpark
  runtime separately, and port only the required SM86/native-MXFP4 and pipeline
  fixes. Upstream currently rejects DSpark when `pp_size != 1`, so that
  restriction is real implementation work rather than a launch-argument issue.
- Exact Pro checkpoint accounting found 788.87 GiB of target-layer tensors,
  3.45 GiB of fixed tensors, and 39.10 GiB of bundled three-layer DSpark
  tensors. Every target layer is about 12.92–12.95 GiB packed. The first safe
  heterogeneous PP boundary is therefore `44,17`: 569.00 GiB of layer tensors
  on dwagon and 219.86 GiB on fwuff. A compute-balanced partition would exceed
  fwuff's 247 GiB. Pro PP2 V1 was launched at that boundary with zero GPU
  experts to measure authentic residency before selecting hot experts; both
  ranks completed the 66-shard scan without an OOM and proceeded into
  stage-local native-MXFP4 expert finalization.
- Pro PP2 V1 then established that packed checkpoint size was not a valid
  residency estimate for the previous loader. fwuff completed layers 44–57
  and was finalizing layer 58 when the kernel OOM killer terminated its
  245,751,828-kB anonymous working set (237.3 GiB peak). The old path expanded
  each one-byte UE8M0 scale to BF16 in Python, staged another BF16 tensor for
  tensor-parallel loading, and persisted it as FP32 in every CPU expert
  buffer. Across all 61 Pro layers, the persistent FP32 scale view alone
  costs roughly 183 GiB.
- The native MXFP4 buffer now persists the checkpoint's UE8M0 scale as one
  byte and reconstructs its exact FP32 power-of-two exponent at the AVX-512
  or AMX point of use. GPU hot-expert export converts directly from UE8M0 to
  BF16, preserving the Ampere kernel contract without changing or requantizing
  weights. The Python loader also retains the original byte tensor, removing
  the expanded staging copy. This is a lossless representation fix expected
  to save roughly 3 GiB per Pro layer across Python staging plus persistent
  buffers.
- Coherency was checked on both hosts after rebuilding their separate
  AMX/AVX512 extensions. A synthetic native-scale AMX-vs-AVX512 oracle passes
  at 8, 17, 32, 47, and 64 expert tokens with maximum relative L2 below
  9e-9 and maximum absolute error below 6e-8. A real Flash layer-1 expert
  matches an independent Torch MXFP4 reference to 0.391% mean-relative after
  BF16 MoE arithmetic. The checkpoint loader returns contiguous `torch.uint8`
  scales at exactly one byte per element; fwuff and dwagon runtime extensions
  match their respective rebuilt artifacts byte-for-byte.
- Pro PP2 V2 proved the memory breakthrough at full scale: fwuff finalized all
  17 assigned layers, crossed the exact V1 layer-58 failure point, and ended
  weight loading with zero cgroup OOM events. The 44-layer dwagon stage instead
  exposed a separate GPU topology limit: its single 3090 reached 23.46 GiB
  while post-processing dense FP8 weights at layer 26, and a 96 MiB Marlin
  repack failed. This was a CUDA OOM, not a recurrence of CPU scale expansion.
- Pro PP3 V3 treats dwagon as two logical launch nodes, one per 3090 and NUMA
  domain, plus fwuff as rank 2. SGLang formed the three-rank distributed group
  correctly and loaded the `22,22,17` partition completely. Each stage
  captured its decode graph; the two 22-layer local stages retained 10.30 and
  11.94 GiB immediately after weight loading, and fwuff retained 12.56 GiB.
  Native MXFP4 CPU residency remained within all hosts while every available
  GPU participated.
- V3's limiting first stage profiled 973,568 full-token slots at 11,104
  bytes/token, below the exact 1,048,576 deployment contract. More importantly,
  its 50 MiB post-graph margin let startup succeed but not the first pipeline
  request: ProcessGroupNCCL lazily created the local PP edge and failed a
  10 MiB CUDA channel allocation. The profiler now preinitializes each forward
  PP device edge before measuring compressed-cache capacity, so persistent
  communication buffers are accounted rather than discovered on live traffic.
  Four focused pipeline cache/capacity/communicator tests pass.
- Pro PP3 V4 shifts one layer from the fixed-weight-heavy first stage to the
  middle 3090 (`21,23,17`). Based on the measured stage footprints, removing
  that layer reduces both PP0 resident dense weight and per-token cache bytes;
  PP1's former 1.79 GiB post-graph reserve covers its added layer and the
  remaining 75,008 tokens. The relaunch also uses a fresh rendezvous port and
  the preinitialized NCCL profile path.
- Pro PP3 V4 completed all 61 target layers and is the first coherent served
  Pro deployment. The three stages preinitialized their real NCCL pipeline
  edges before compressed-cache profiling, then captured their decode graphs
  with 0.21, 0.03, and 4.10 GiB reported GPU memory available. The common
  cache is 1,037,056 tokens at full 1,048,576 configured context, so the
  current target-only layout is still 11,520 tokens short of the strict
  single-user capacity contract.
- Two end-to-end OpenAI requests returned exactly `4` for the same
  deterministic arithmetic probe. The first cold request took 14.22 seconds;
  the repeated warm request took 2.36 seconds and left all three ranks active.
  This proves native-MXFP4 execution across both dwagon NUMA domains and the
  InfiniBand-connected fwuff stage, including the formerly lazy local NCCL
  edge. It is a functional baseline, not a performance result: target-only
  generation remains far below the 20 token/s requirement and DSpark is not
  yet attached.
- A subsequent 128-input/32-output native benchmark probe failed before it
  could establish a throughput baseline. PP1 had only 13 MiB physically free
  after the real pipeline communicators were active, and the unfused
  MHC-pre expression attempted a 14 MiB FP32 `flatten` allocation. PP1 raised
  a CUDA OOM and the other ranks exited coherently. The two short probes remain
  valid functional evidence, but V4 is rejected as a robust deployment. As
  with the earlier MHC-post failure, the next fix is to remove the avoidable
  FP32 temporary in a fused kernel rather than hide it with a shorter context.
- The existing TileLang MHC-pre design is now admitted for this hardware after
  an independent Torch-oracle check at 1, 17, 256, and 4,096 tokens. Maximum
  BF16 output error is 0.001953125, mean error is at most 5.31e-6, and the
  post/combination coefficients remain within 2.54e-4/1.47e-4. At the real
  4,096-token prefill chunk it runs in 0.865 ms and peaks at 59,457,536
  incremental bytes, eliminating the approximately 448 MiB FP32 flattened
  activation. Both fused MHC-pre and fused MHC-post are now enabled in every
  Flash and Pro deployment script.

### Native-MXFP4 DSpark bring-up on SM86

- Before resuming DSpark work, the journal was checked for outside edits
  (`2026-07-26 04:31:01 -0400`, 42,723 bytes). No unincorporated change was
  present. Both dwagon 3090s and fwuff's 3090 were idle, and fwuff had no stale
  SGLang/KTransformers process to terminate.
- A clean July SGLang DSpark source at `6cc9352df` now maps bundled draft
  prefixes `stages.N` to checkpoint keys `mtp.N`. The KTransformers expert-ID
  mask is functional rather than in-place, preventing a race with the
  asynchronous AMX CUDA-to-host transfer, and the zero-GPU-expert case bypasses
  invalid empty MoE weights. The focused wrapper test passes.
- The clean source incorrectly selected Hopper-only components on SM86. A
  self-contained Ampere dispatcher now skips FlashMLA metadata on capability
  8.6 and runs sparse V4 attention through a Triton kernel that decodes the
  packed FP8/UE8M0/BF16 KV layout by exact byte lookup. A constructed one-token
  cache oracle returned all 512 values bit-exactly (`max_abs=0`). The sparse
  prefill dequantizer uses the same raw-byte E4M3 lookup instead of Triton's
  unsupported `fp8e4nv` type and independently passed the same exact oracle.
- The DeepGEMM paged-indexer metadata and Hopper thread-block-cluster top-k-v2
  planner are bypassed on SM86. The graph-safe reference FP8 indexer retains a
  1-D sequence-length contract, while the fused top-k-v1 CUDA kernel remains
  active. Its 16,384-position oracle selected exactly positions 15,872–16,383
  with 512 unique outputs.
- TileLang MHC-pre's optional register-usage compiler pass was removed because
  this runtime serialized its TVM integer object into an invalid `ptxas`
  option. The exact fused prenorm kernel then compiled and produced finite
  output on SM86. The clean branch also bypasses its missing DeepGEMM MHC path.
- Target and draft loading now complete together under TP2. Rank 0 loads all
  43 target native-MXFP4 expert layers plus all three bundled `mtp.0/1/2`
  expert layers into two 52-thread AMX NUMA pools; rank 1 holds the sharded
  dense weights. DSpark initializes with block size 5, six verify tokens,
  mask token 128799, and the V4 Markov head. Both ranks allocate all 65,536
  configured token slots with roughly 15.2 GiB free after the pools.
- The older wrapper's live-activation transfer was replaced with a persistent
  32 MiB staging buffer, per-layer CPU CUDA streams, and completion events so
  AMX transfers do not alias the TP/NCCL stream. The first real request logged
  the native kernel's AVX-512 decode dispatch; populated prefill batches retain
  the configured AMX threshold.
- The current coherent eager service is
  `exo-dsv4-flash-dspark-v1l.service` on `127.0.0.1:30010`. Two identical
  greedy OpenAI arithmetic probes returned exactly `4`; the repeated warm
  probe took 1.884 seconds end to end. Two 32-token greedy sequence probes
  returned identical output IDs:
  `223,23,14,223,24,14,223,25,14,223,26,14,223,27,14,223,553,14,223,779,14,223,736,14,223,907,14,223,929,14,223,856`.
  DSpark accepted all 30/30 proposed draft tokens in both traces, with 100%
  acceptance, 5.333 accepted tokens per verify step, and six verify steps.
  The second trace completed in 2.229 seconds, or 14.36 output token/s
  end-to-end.
- This is a coherent DSpark baseline, not acceptance of the performance or
  capacity contract. Decode is still below 20 token/s, the service is limited
  to 65,536 tokens, CUDA graphs are disabled while the graph-safe CPU proxy is
  ported, and the compact ragged scheduler has no profiled SPS table so it
  degenerates to verify-all. The July sparse-prefill `sgl-kernel` call also
  has a newer `attn_sink` ABI than the installed Ampere runtime, so extend
  currently uses the validated unified Triton attention dispatcher.
- The clean DSpark KTransformers path now admits arbitrary profile-selected GPU
  experts. Logical expert IDs remain unchanged for the asynchronous AMX job,
  while a persistent logical-to-compact map masks CPU IDs and remaps only hot
  experts into dense GPU weight slots. The model loader applies the same map,
  so checkpoint expert `e` is placed in exactly the slot used by inference.
  Bundled `mtp.0/1/2` draft namespaces receive explicit all-CPU masks and
  therefore cannot accidentally borrow target layer 0–2 weights.
- The retained V16 decode profile has exact shape `[1,43,256]`. At a nominal
  32 experts per layer it selects the globally hottest 1,376 target slots;
  actual per-layer residency ranges from 0 to 45 because allocation follows
  observed use rather than a front-loaded quota. Four focused tests now cover
  immutable routing, non-contiguous compact remapping, global frequency
  selection, and draft all-CPU isolation.
- The first profiled launch exposed two clean-branch integration seams rather
  than a model result. Its July `ServerArgs` publishes the HF object through
  `get_model_config().hf_config`; a compatibility helper and a realistic test
  now cover that API. The next launch loaded target and draft completely, then
  rejected the first nonzero GPU expert layer because `moe_runner_backend=auto`
  chose the FP8 Triton runner for packed MXFP4 weights. Ampere hot experts now
  explicitly use the existing MXFP4 Marlin kernel, which losslessly repacks
  native E2M1/UE8M0 checkpoint values for SM86 execution.
- The July Marlin wrapper carried a contradictory SM90/SM120-only guard even
  though its quant-type registry admits FP4 E2M1 on every SM80+ device and its
  fused implementation contains an explicit non-atomic Ampere reduction path.
  The guard is removed while the group-size, tile-padding, and layer-shape
  validators remain. A focused capability test confirms native MXFP4,
  group-size-32 admission for SM86.
- The first SM86 Marlin launch reached post-load processing on both TP ranks,
  confirming that the backend passed capability and shape admission. It then
  found a profile edge case in target layers 0--2: each intentionally has zero
  selected GPU experts, but the wrapper called the backend repacker with an
  empty tensor list. Post-load GPU processing now runs only for nonempty GPU
  expert slices, matching the existing zero-expert inference bypass; CPU AMX
  loading remains unchanged.
- After that fix, all 43 target layers and all three DSpark draft stages
  loaded, and 65,536 cache slots were allocated with roughly 4.5 GiB free per
  3090. The first request compiled the SM86 Marlin template successfully but
  its first real hot-expert GEMM issued an illegal CUDA memory access; the
  following SiLU merely surfaced the asynchronous fault. The contradictory
  capability guard therefore did conceal a real Ampere safety issue, and
  Marlin is no longer used for native-MXFP4 experts on this machine.
- The clean July branch now has an explicit SM86 native-MXFP4 adapter. It
  preserves packed E2M1 weights and UE8M0 scales, wraps them in the validated
  strided layout, and executes both expert GEMMs through the portable
  `triton_kernels.matmul_ogs` gather/scatter path previously used for the
  coherent 20.89-token/s V17 target-only result. KTransformers remains
  responsible for the cold experts through AMX/AVX-512, and DeepSeek applies
  the routed scale once after the partial GPU and CPU results are merged.
- The first integrated portable-kernel service is coherent. Two greedy chat
  probes returned exactly `4`; two warmed 32-token traces returned identical
  IDs and DSpark accepted 30/30 proposals. The second trace took 2.440 seconds
  (13.11 output token/s end to end), while a 64-token trace took 3.879 seconds
  (16.50 token/s) with 100% acceptance. A one-second utilization trace showed
  bursty 11--95% GPU SM occupancy rather than sustained saturation, so the
  32-expert profile leaves a CPU/coordination tail. The portable layout also
  raised the computed cache capacity from 255,744 to 489,984 tokens and left
  6.5 GiB free per GPU at the current 65,536-token allocation. The next
  measured point raises global profile-guided residency from 1,376 to 1,892
  expert slots (44 nominal per layer), using that headroom to remove more AMX
  tail without changing the native format.
- The GPU44 point fit 65,536 slots and retained 3.3 GiB free per GPU, but it
  was slower: its repeated coherent 32-token trace took 2.621 seconds versus
  2.440 seconds at GPU32. The portable gather/scatter cost grows with the
  compact expert dimension faster than this profile removes CPU work, so
  GPU32 remains the measured optimum among these points. Residency is restored
  to 1,376 slots. For decode accounting, the 64-token GPU32 request took
  3.879 seconds while the same warm prompt-to-first-completion path takes
  approximately 1.06 seconds; subtracting that fixed prefill/first-token
  component gives about 22.35 decode token/s. A longer served trace is still
  required to report TPOT without relying on this subtraction.
- SGLang's own streamed `bench_serve` on the restored GPU32 configuration
  (random 8-token input, 512 forced output tokens, concurrency one) reports
  599.47 ms TTFT and 77.78 ms TPOT, or 12.86 decode token/s. Its mean DSpark
  acceptance length is only 4.11, so the short monotonically increasing
  sequence was not representative. This rejects the inferred 22-token/s
  result and leaves the strict decode target unmet. The service currently
  verifies all six draft positions because its SPS cost table is flat; the
  next launch enables the built-in per-step recorder with simulated
  one-token acceptance solely to profile the true GPU32 cost curve, after
  which real acceptance and a fitted compact-verification table will be
  restored.
- The first SPS-profiler attempt correctly refused to measure the eager-only
  service: its fitted verify costs would not transfer to graph replay. Review
  of KT-Kernel's serving bridge confirms that `submit_with_cuda_stream` and
  `sync_with_cuda_stream` become `cudaLaunchHostFunc` nodes. Consequently a
  full decode CUDA graph replays the native AMX/AVX-512 CPU expert task as
  well as the GPU work; it does not freeze the CPU result captured at startup.
  The Flash launcher now captures only the single-request full decode shape
  and leaves prefill eager. This is also the graph configuration documented by
  upstream KTransformers for V4 Flash, narrowed here to DSpark concurrency one.
- Full target and draft graphs captured in 4.1 and 1.5 seconds respectively,
  and the target capture executed the expected native AVX-512 six-token expert
  dispatch. The first replay nevertheless left an asynchronous illegal CUDA
  access that surfaced when DSpark loaded its acceptance kernel. Rather than
  trust a monolithic hybrid graph with an unknown bad node, the KTransformers
  expert block is now an explicit eager break under SGLang's segmented CUDA
  graph backend. This preserves capture around attention, routing, and dense
  work while replaying the complete overlapped CPU/GPU expert calculation with
  fresh tensors at each layer. All five focused wrapper tests and both
  worktree whitespace checks pass after the integration change.
- The first segmented capture then exposed two generic July-branch BCG seams,
  both before serving: DeepSeek V4 attention incorrectly fetched a
  TC-piecewise-only global context, and the BCG output bridge did not support
  the model's `LogitsProcessorOutput` dataclass. Attention now carries its
  static forward batch, backend, and layer explicitly across the eager break;
  the output bridge now allocates, copies, and slices every direct tensor field
  in `LogitsProcessorOutput`. Both files compile, the focused KTransformers
  suite remains 5/5, and a direct output-buffer oracle preserves logits and
  hidden states exactly.
- The next two replays found shape/type loss in BCG itself rather than model
  math. Its weak-reference walker converted SGLang named tuples into plain
  tuples, so nested `StandardDispatchOutput.hidden_states` disappeared; it now
  reconstructs named tuples recursively, with a CUDA data-pointer oracle.
  After that fix the draft replay showed that the generic shared-output path
  sliced five speculative hidden rows by request batch size one. This
  deployment captures exactly one target shape and one draft shape, so each
  BCG runner now retains its captured output at the native dimensions instead
  of applying cross-shape deduplication. Multi-shape behavior is unchanged.
- Native output retention restores the draft's complete five-token hidden
  state. The following replay reached DSV4 attention and exposed the last
  difference between monolithic and segmented metadata: replay resets the
  backend to lightweight raw verify metadata, while the Python attribute
  assignment made by a captured initializer is not itself replayed. The
  explicit attention break now runs that idempotent raw-to-full metadata
  upgrade eagerly once per replay before consuming `core_attn_metadata`.
- Synchronization at every expert entry/GPU completion/merge proved all 43
  target expert blocks and the first draft expert block clean. The fault
  occurred in the captured segment before draft layer 1; the attention-specific
  boundary showed why: the branch only broke target-verify/extend attention,
  leaving draft-decode DSV4 attention captured. Segmented mode now breaks DSV4
  attention in every forward mode, using the same explicit raw-to-full metadata
  path for both target verify and draft decode.
- The all-mode boundary localized the next asynchronous fault inside the
  second DSpark draft attention call, after its first MoE block completed
  cleanly. The underlying July backend already exposes a capture-stable DSV4
  metadata contract and uses it in breakable prefill, but decode still took the
  monolithic-graph raw-metadata path. Decode BCG capture/replay now stores the
  backend's full per-graph metadata object, refreshes it in place from the live
  padded batch before every replay, and suppresses the raw-restoring warmup
  hook. This preserves the addresses captured by graph-adjacent kernels while
  giving eager draft attention current sequence, indexer, compression, and
  cache-location data. The modified runner compiles and the worktree passes
  `git diff --check`; no Ruff executable is installed in the deployment
  runtime.
- `fwuff` had no model, GPU, or RDMA-serving process left from the prior
  session. Two stale `tail -F` log followers from that ended workload were
  terminated under the operator's explicit cleanup authorization; the host
  remained at 236 GiB available RAM with no GPU compute allocation.
- The first capture-stable decode launch eliminated the asynchronous CUDA
  failure: both ranks synchronized every target expert, all three DSpark draft
  experts, and graph capture successfully. Its first replay stopped at a host
  assertion before attention because `SGLANG_PREP_IN_CUDA_GRAPH` makes
  `_build_forward_metadata` return a raw wrapper for decode/verify. The BCG
  capture and replay-refresh hooks now eagerly run the existing raw-to-full
  materializer before storing or copying metadata; monolithic graph behavior
  is unchanged. Both edited modules compile, `git diff --check` passes, and
  the focused KTransformers wrapper suite remains 5/5.
- The next replay completed target verification and draft step 0, then
  synchronized an illegal access in draft attention step 1. This separated two
  CUDA-graph runners: target verify uses the generic decode runner fixed above,
  while DSpark proposals use `EAGLEDraftCudaGraphRunner`, which still restored
  raw metadata. The dedicated draft runner now uses the same capture-stable
  contract. Its multistep backend builds each step as logical `DECODE` metadata
  (including that step's cache-location slice), rather than incorrectly
  treating the outer proposal batch as block `TARGET_VERIFY`. The temporary
  per-attention raw-upgrade workaround is removed so it cannot overwrite the
  refreshed per-step cache rails. All four edited modules compile,
  `git diff --check` passes, and the focused suite is still 5/5.
- A fence before every eager attention invocation proves draft attention is
  not launching the bad access: step 0 attention and MoE synchronize, then the
  captured inter-step segment faults before step 1 reaches its entry fence.
  The segment spans DSpark's Python loop mutations of `ForwardBatch` fields,
  which segmented graph replay cannot reproduce. The draft proposal loop is
  therefore isolated as one eager BCG break behind
  `SGLANG_DSV4_DRAFT_BCG_EAGER=1`; target verification remains
  segmented/captured, so the SPS verify-cost profiler still measures the
  intended graph path. This is a correctness bridge, not the final draft
  optimization: the host mutations must ultimately move into capture-stable
  device buffers before draft segmentation can be re-enabled.
- DSpark does not instantiate the legacy `EAGLEDraftCudaGraphRunner`; its
  DeepSeek V4 block-draft model owns a normal `DecodeCudaGraphRunner`. The
  first eager-isolation switch therefore did not affect the failing graph.
  The generic runner now applies the same opt-in whole-forward break when
  `model_runner.is_draft_worker`, leaving the target worker's BCG untouched.
- That isolation first exposed unsupported nested graph breaks at capture:
  KTransformers' inner expert break tried to end the segment already ended by
  the outer draft-forward break. `eager_on_graph` now treats a missing current
  segment as an enclosing eager break and directly invokes the nested function.
  A direct nested-break oracle passes; 17 focused BCG/KTransformers tests pass,
  while the one unrelated GSM8K integration case cannot start because the
  deployment venv has no `sglang` console-script executable (module launch is
  used here).
- With nested breaks fixed, whole-forward draft replay passed all three draft
  MoEs, but its stage-1 pre-attention preparation still issued the illegal
  access before the attention entry fence. This rules out the inter-step graph
  segment and isolates the remaining difference to graph-prepared draft
  metadata/cache rails. DSpark now supports an opt-in draft-only graph disable;
  this deployment uses `SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1`, returning the
  three-stage block draft to its previously coherent eager metadata path while
  retaining the target verification graph required for SPS profiling.
- The draft-only eager launch confirms the chronology: eager target prefill
  completed all 43 layers, eager DSpark draft completed all three stages, and
  the target-verify BCG then completed layer 0 attention/MoE before faulting in
  the captured transition ahead of layer 1 attention. The old break was too
  deep inside `MqaAttentionBase`: query/KV projection and dynamic cache stores
  remained in the unsafe segment. Target BCG now breaks around the complete
  attention module, making query/KV preparation, cache writes, and attention
  eager as one unit while leaving MHC, routing, dense operations, and every
  native-MXFP4 expert break segmented.
- `v2n` showed that expanding the eager boundary to the complete attention
  module did not move the failure: target-verify layer 0 attention and its
  native-MXFP4 KTransformers expert both synchronized, then the replay faulted
  before layer 1 attention entered. This confines the bad captured work to the
  deferred layer-0 mHC post state being consumed by layer 1's fused mHC
  post/pre transition. That cross-layer fused transition now has its own
  breakable-graph eager boundary and entry/exit fences under the existing
  debug switch; the rest of the layer remains segmented. The modified model
  compiles, `git diff --check` passes, and the focused KTransformers wrapper
  suite remains 5/5.
- `v2p` captured all 43 target layers with the proposed fused-transition break
  but its first replay still faulted after layer 0 expert merge and before the
  new transition fence. Inspection corrected the premise: cross-layer fusion
  is opt-in and disabled in this deployment, so the new helper was never on
  the live path. In the unfused path, the sole operation between the
  synchronized KTransformers return and the next attention break is the
  trailing TileLang `hc_post`. That post-MoE kernel now has a narrow eager BCG
  boundary with entry/exit fences. The unused fused-transition boundary is
  retained for the later fused-MHC optimization.
- `v2q` proved the trailing `hc_post` kernel is also innocent: all 43 such
  kernels synchronized during capture, while live replay again faulted before
  the layer-0 post-MoE fence entered. Breakable replay ordering identifies the
  segment between KTransformers' eager expert function and that fence as the
  culprit. KTransformers `apply` returns a combine object, so routing combine,
  scaling, and communication still ran in a graph segment against an
  eager-created bridge. The eager boundary now encloses the complete MoE
  operation—dispatch, native-MXFP4 CPU/GPU experts, combine, and
  collectives—rather than only expert `apply`; its nested KTransformers break
  runs directly under the outer boundary.
- `v2r` eliminated the CUDA illegal access. Live target replay completed
  attention, complete MoE, and post-MoE mHC through layers 0 and 1, then
  stopped on a deterministic host assertion in layer 2's first C4 indexer:
  `indexer_metadata.page_table is core_metadata.page_table` was false. The BCG
  refresh replaced the core's page-table reference but copied the indexer's
  old captured object in place, breaking a deliberate shared-object contract.
  Replay refresh now re-links the indexer's page table and C4 sequence lengths
  to the refreshed core metadata. A new regression covers both identities and
  live values; it and the existing backend/KTransformers checks pass 7/7.
- `v2s` is the first coherent DSpark target-verify BCG replay on SM86. It
  crossed the repaired layer-2 C4 indexer, completed all 43 layers, and returned
  HTTP 200 in 2.818 seconds on the first four-output request. Because this SPS
  measurement launch deliberately sets `SGLANG_SIMULATE_ACC_LEN=1`, the
  multi-token text is not a semantic acceptance result; two separate
  one-output greedy arithmetic probes returned exactly `4` in 1.024 and 1.023
  seconds and the service remained active. Target verification is segmented,
  draft remains eager, and every expert stays in native checkpoint MXFP4 with
  profile-guided SM86 residency plus AMX/AVX-512 execution.
- The first four SPS fractions after `v2s` measured only 2.366–2.390 verify
  steps/s because the localization build still synchronized CUDA at every
  attention and expert substage. The profiler was terminated before it could
  write a misleading table. `SGLANG_KT_GRAPH_DEBUG_SYNC` is removed from the
  launcher for the real cost profile; the diagnostic code remains dormant for
  future fault isolation.
- Debug-free `v2t` remained coherent (`4` on a one-token arithmetic probe in
  1.493 seconds) and produced the first valid Flash DSpark SPS table:
  `/var/lib/exo/profiles/dsv4-native-mxfp4/flash-dspark-v1u-gpu32-sps.json`.
  Across forced verify fractions 1/6 through 6/6, median step time was
  335.47–344.36 ms (2.904–2.981 steps/s); the fitted additive-table self-check
  passed. The deployment launcher now loads this table and removes both SPS
  recording and simulated acceptance so the next run measures genuine DSpark
  decisions and output.
- Real-acceptance `v2u` loaded the SPS table and passed two repeated greedy
  arithmetic probes with exactly `4`. Two 32-token sequence probes returned
  identical IDs and text; DSpark accepted 28/30 drafts (93.33%), averaged
  5.333 accepted tokens per verify, and used six verifies. The warmed request
  took 2.884 seconds, or 11.10 output token/s, so graph coherence alone does
  not recover the eager baseline's performance and remains far below the
  20-token/s requirement.
- The next kernel experiment enables the branch's fused mHC post/pre path. It
  fuses the residual post map, pre-mixing GEMM, Sinkhorn preparation, and
  optional RMSNorm across layer boundaries instead of materializing separate
  TileLang post and pre kernels. BCG boundaries already cover the cross-layer
  fused call; DSpark auxiliary and final trailing `hc_post` calls are now also
  isolated so target verification preserves graph correctness while measuring
  the actual fusion.
- Fused-mHC `v2v` was coherent: repeated arithmetic returned exactly `4`, and
  both 32-token probes produced the same IDs, 28/30 draft acceptance, and
  5.333 accepted tokens/verify as `v2u`. It did not improve speed: the repeated
  trace took 2.947 seconds versus 2.884 seconds unfused. Fusion is therefore
  rejected on this SM86 configuration and the launcher returns to the
  numerically identical unfused MHC path.
- The next graph architecture keeps DSV4 attention, indexer, compressors, and
  unfused MHC inside captured segments, with only one complete MoE eager
  boundary per layer. This removes the outer attention, inner attention, and
  trailing-MHC Python callbacks while retaining the full-MoE boundary that
  eliminated the illegal access. Because captured attention reads metadata by
  address, BCG replay now copies page-table/SWA/C128 contents into the original
  captured tensors instead of replacing their objects; the indexer is re-linked
  to that same core storage.
- Reduced-break `v2w` is coherent and substantially faster. Two arithmetic
  probes returned exactly `4`; two 32-token sequence probes produced identical
  output IDs and the same 28/30 (93.33%) genuine draft acceptance as the
  earlier layouts. The repeated trace took 2.069 seconds, or 15.46 output
  token/s—39.3% faster than `v2u`'s 11.10 token/s. Target verification now
  uses 43 complete native-MXFP4 MoE callbacks rather than per-layer
  attention/MoE/MHC callbacks; the captured-attention storage regression plus
  existing focused checks pass 8/8.
- Official `bench_serve` for `v2w` completed one warmed random 8-input,
  512-output request in 26.425 seconds. Output throughput was 19.36 token/s,
  TTFT 507.48 ms, and TPOT 50.72 ms (19.72 token/s); mean DSpark acceptance
  length was 4.00. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v2w-gpu32-random8-out512.jsonl`.
  This is a 52.6% served-throughput gain over the prior 12.69-token/s graphless
  benchmark but remains just below the strict 20-token/s decode gate.
- The next test re-enables the three-layer DSpark draft CUDA graph. Draft
  attention now uses the same captured tensor identities that made target
  verification coherent, and its full-MoE calls use the repaired outer eager
  boundary. This removes the last draft-side Python/kernel-launch path while
  preserving native MXFP4 and real acceptance.
- Draft-graph `v2x` is rejected. Both target and draft graphs captured
  successfully (0.30 GiB and 0.07 GiB respectively), repeated arithmetic
  probes returned exactly `4`, and two 32-token probes produced the same target
  token IDs as coherent `v2w`. However, DSpark accepted 0/155 proposals
  (0% acceptance, 1.032 average accepted length, 31 verifies): the captured
  three-layer draft is replaying a stale/different dynamic state even though
  target verification remains numerically stable. The launcher restores
  `SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1`; target attention stays captured,
  while draft proposal remains eager until its complete replay-state contract
  is proven.
- Native AMX crossover `v2y` lowered
  `KT_MXFP4_AMX_MIN_EXPERT_TOKENS` from 8 to 4. Coherency held: repeated
  arithmetic returned exactly `4`, and two 32-token requests returned
  byte-identical token IDs. Performance regressed, however: the repeated
  32-token request took 2.344 seconds versus 2.069 seconds for `v2w`. This
  agrees with the earlier kernel crossover microbenchmark (AMX is only 0.912x
  AVX-512 at four tokens per expert). The deployment threshold is restored to
  8; the remaining decode work targets the AVX-512 MXFP4 microkernel itself.
- The native E2M1 decoder now expands packed low/high nibbles to 16-bit
  indices and uses AVX-512 `VPERMW` to map them directly to complete BF16
  values. This replaces four byte-table shuffles and four unpack stages per
  output row/K-group. The small-M output tile also grows from four to eight
  rows, amortizing activation handling when an expert receives several verify
  tokens. A core-pinned 256-output Flash microbenchmark improved one-token
  decode from 0.330 to 0.156 ms (6.36 to 13.41 GFLOP/s, 2.11x), two-token
  decode from 0.660 to 0.313 ms (2.11x), and four-token decode from 1.321 to
  0.627 ms (2.11x).
- Coherency gates for the new decoder pass: 4,096 randomized packed 16-byte
  groups are bit-exact against the prior byte-shuffle implementation; the
  complete AVX-512 mat-vec output is bit-exact against the independent
  mat-mat reference; AMX-versus-AVX relative L2 remains below 9e-9 with
  5.96e-8 maximum absolute error. The rebuilt isolated runtime extension is
  SHA-256
  `286505d9f8854eee26b0fec734c8253d2a5f26417240eff197cd4d1ef6cf75ba`,
  exports `AMXFP4_KGroup_MOE`, and contains 24 `VPERMW` instruction sites.
- `v2z` passed model-level coherency with the rebuilt extension: repeated
  arithmetic returned exactly `4`, and repeated 32-token sequences retained
  identical IDs. Two official random-8/output-512 probes were not comparable
  to `v2w` because the chosen prompts averaged only 2.55 and 2.68 accepted
  tokens per verify; they delivered 10.44 and 12.43 token/s respectively.
  Low-M AVX output was subsequently checked bit-exact at M=1/2/4/6 as well as
  the earlier larger shapes, ruling out a decode-layout regression.
- The production SPS curve was stale after the kernel change, so `v3a`
  re-profiled all six compact retained-token fractions with target graph
  replay, eager coherent draft, simulated one-token acceptance, and server-side
  step records. Median verify time is now 226.2–230.8 ms versus the old
  335.5–344.4 ms; the one-token cell improves from 2.981 to 4.422 verify
  steps/s (32.6% lower latency). All six cells matched the requested budgets,
  each collected 28 steady steps, and the additive-table reload self-check
  passed. Production table:
  `/var/lib/exo/profiles/dsv4-native-mxfp4/flash-dspark-v3a-vpermw-gpu32-sps.json`.
- Production `v3b` loaded the new SPS table and retained model coherency:
  repeated greedy arithmetic returned exactly `4`, and two 32-token probes
  returned byte-identical token IDs. An official four-prompt random-8 /
  output-128 run produced 512 output tokens in 40.24 seconds (12.72 token/s),
  with 568.58 ms mean TTFT, 74.71 ms mean TPOT, and 3.09 mean DSpark
  acceptance length. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3b-vpermw-gpu32-random4x128-seed42.jsonl`.
  The target verifier's one-token cell is now 32.6% faster, but three eager
  draft layers plus input-dependent low acceptance still dominate the full
  path. Draft CUDA-graph replay must therefore be made coherent rather than
  disabled.
- Draft-graph isolation `v3c` kept the three-layer model captured but removed
  its folded Markov sampler and shared proposal-buffer handoff. Acceptance
  remained effectively zero (`1.05` accepted length, `0.01` rate in the first
  logged sequence), proving the stale state is in draft-model replay rather
  than proposal sampling or the target epilogue.
- `v3d` added a same-batch graph/eager hidden-state oracle and used the eager
  result for continued diagnosis. The error exists on the first draft:
  relative L2 was `1.135310` at sequence length 1 and `0.916319` at sequence
  length 13, with maximum absolute errors of 366.5 and 464 respectively;
  later graph outputs became non-finite while eager outputs remained finite.
  Source inspection identified a structural difference from the repaired
  target graph: `DSparkV4Stage._run_ffn` bypassed the complete-MoE BCG wrapper,
  so only native expert execution crossed an eager bridge while routing
  combine and communication remained captured against its transient output.
  Draft stages now use the same complete dispatch/expert/combine boundary as
  target layers; the hidden-state oracle stays enabled for the next replay.
- The `v3e` graph/eager oracle is bit-exact after that boundary repair.
  Maximum absolute error, mean absolute error, and relative L2 were all exactly
  zero on both TP ranks for eight comparisons spanning sequence lengths 1
  through 25; all graph and eager tensors remained finite. A live 32-token
  request then accepted 22/55 proposals (2.91 tokens per verify) while the
  oracle still returned zero error. The doubled diagnostic execution is now
  removed, leaving the coherent draft model graph active with only the small
  proposal sampler eager for the next performance gate.
- Production-shape `v3f` kept only the proposal sampler eager and completed an
  official four-prompt random-8/output-128 run in 39.23 seconds:
  13.05 output token/s, 558.42 ms mean TTFT, 72.78 ms mean TPOT, and 2.91 mean
  accepted tokens per verify. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3f-draftgraph-eagersampler-random4x128-seed42.jsonl`.
  This is coherent but only 2.6% faster than `v3b`; the remaining eager
  base-logit, Markov sampling, confidence, acceptance, and commit path is now
  restored to the graph-folded implementation on top of the bit-exact draft
  model replay.
- Fully folded `v3g` passed coherence: repeated one-token probes matched,
  repeated 32-token greedy requests returned byte-identical text, and genuine
  proposal acceptance remained nonzero (2.67 and 2.91 accepted tokens per
  verify). Its comparable official random-8/output-128 run produced 512 tokens
  in 39.41 seconds: 12.99 token/s, 573.21 ms mean TTFT, 73.03 ms mean TPOT,
  and 2.83 mean accepted tokens. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3g-fully-folded-random4x128-seed42.jsonl`.
  Folding is therefore correct but not the bottleneck.
- The profiler's flat 226–231 ms curve was traced to the target graph grid,
  not an inherent low-token kernel floor. With decode graph bs=1, compact
  ragged verification captured only one six-token tier, so every budget from
  one through six rounded up to six and all padded candidates traversed 43
  MoE layers. The runner now supports opt-in fine-grained DSpark tiers:
  target graphs for 1, 2, 3, 4, 5, and 6 tokens, while larger request-count
  deployments retain their existing whole-block tiers. The launcher enables
  this mode; the next SPS profile will measure actual compact native-MXFP4
  work rather than padded six-token work.
- `v3h` captured and retained all six fine target tiers. Target graph memory
  rose only from 0.30 to 0.46 GB per rank and capture completed in 43.5
  seconds on the first compile. Two independent 32-token requests returned
  identical text and token IDs, and both accepted 19/60 drafts (2.67 output
  tokens per verify), establishing coherent replay across the new graph-key
  grid.
- The first fine-tier profile reached four cells before exposing a mutable
  staging-buffer lifetime bug. A previously unseen ragged size could allocate
  KTransformers' pinned ring under `torch.inference_mode()`, then a breakable
  graph host callback attempted to update that long-lived inference tensor
  outside the mode. `KExpertsCPUBuffer` now explicitly allocates ordinary
  mutable tensors under `torch.inference_mode(False)`. A direct lifetime
  oracle verified that every CPU/GPU staging tensor is non-inference and can
  be updated after leaving inference mode.
- `v3j` passed the former fifth-tier crash and completed all six forced
  fractions with perfect replay-tier matching. Median complete DSpark step
  times for M=1 through 6 were respectively 207.58, 211.63, 216.72, 220.22,
  222.79, and 226.52 ms. This proves padding removal is real, but also shows a
  207 ms floor from generating the five draft candidates plus per-layer CPU
  expert work.
- The upstream additive profiler binned M in groups of 64, collapsing all
  single-stream tiers into M=0 even after measuring them separately. The fit
  now preserves exact replay tiers by default, with a regression test; all 19
  profiler tests pass. The corrected table contains M probes 1 through 6 and
  is stored at
  `/var/lib/exo/profiles/dsv4-native-mxfp4/flash-dspark-v3j-finetiers-sps.json`.
- All three DSpark MTP expert layers had deliberately been forced to CPU
  because the 43-layer target activation profile cannot describe their
  independent namespaces. The runtime now admits an explicit draft-only GPU
  prefix while retaining every remaining expert in native MXFP4 on
  AMX/AVX-512. `v3k` begins with 192 of 256 experts per draft stage on the two
  SM86 GPUs; this targets the newly measured draft latency floor without
  changing weight format or weakening CPU execution.
- `v3k` is rejected. Although 192 native-MXFP4 experts in each of the three
  MTP stages fit with 2.4 GB free per GPU, two repeated greedy 32-token
  requests diverged and the official random-8/output-128 workload regressed
  to 11.46 token/s with 83.36 ms mean TPOT and 2.67 mean acceptance length.
  Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3k-draftgpu192-random4x128-seed42.jsonl`.
  Production is restored to `SGLANG_KT_DRAFT_GPU_EXPERTS=0`, keeping every
  draft expert on the coherent AMX/AVX-512 path.
- `v3l` recorded per-step GPU segments on the all-CPU DSpark path. The three
  draft stages require only about 9--12 ms; target verification requires
  about 153--175 ms and the whole step about 165--190 ms. For example, the
  four-token tier measured 10.95 ms draft, 154.10 ms target, and 165 ms total.
  The dominant cost is therefore the target breakable graph's 43 complete-MoE
  host boundaries, not draft generation. A 192-token probe took 14.775 s with
  2.46 mean acceptance length (78 verifies, 116/390 accepted drafts).
- `v3m`/`v3n` tested eager target verification while retaining the captured
  draft. The first implementation selected eager execution after
  `prepare_for_verify`, which paired graph-shaped attention metadata with an
  eager forward and failed coherency. Moving selection before metadata
  preparation fixed that invalid pairing, but repeated 32-token outputs still
  diverged and took 5.30--5.45 s. Live DSV4 attention metadata construction
  makes the path substantially slower than breakable replay. The experiment
  remains opt-in for diagnosis, while production restores
  `SGLANG_DSV4_TARGET_VERIFY_EAGER=0`.
- Production `v3o` restored coherent breakable target replay with six exact
  verification tiers. The official seeded four-request random-8/output-128
  benchmark produced 512 tokens in 41.31 seconds: 12.39 token/s, 545.23 ms
  mean TTFT, 76.96 ms mean TPOT, and 2.89 mean acceptance length. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3o-finetiers-random4x128-seed42.jsonl`.
  Fine tiers reduce padded work but cannot overcome input-dependent low
  acceptance; optimization returns to the 153--175 ms target-MoE segment.
- Profile-guided GPU40 `v3p` expanded the native-MXFP4 target hot set while
  retaining all misses on AMX/AVX-512. Three repeated one-token outputs were
  identical. The same official four-request workload improved to 14.49
  token/s, 531.92 ms TTFT, and 65.30 ms TPOT with 3.34 mean acceptance,
  leaving about 3.7 GB free per GPU. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3p-gpu40-random4x128-seed42.jsonl`.
- GPU48 retained only about 1.5 GiB per device after capture but passed
  repeated one-token coherency. Its seeded four-request trajectory had poor
  2.465-token mean acceptance and reached only 11.20 token/s, so it is not a
  representative long-stream gate. A forced-six 512-token run improved only
  to 14.01 token/s and was rejected. With the authentic dynamic SPS policy,
  the official one-request/512-output run produced 512 tokens in 26.11
  seconds: **19.61 output token/s end-to-end and exactly 50.00 ms TPOT
  (20.00 token/s decode)** with 3.62 mean acceptance. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v3t-gpu48-dynamic-random1x512-seed42.jsonl`.
- The first GPU48 32K-prefill attempts isolated the Ampere indexer fallback as
  both a memory and throughput defect. A 4,096-token chunk OOMed while making
  a 512 MiB contiguous cache copy; 1,024-token chunks then OOMed in the
  `[queries, history, 64 heads]` score tensor. Converting E4M3 operands and
  scores to FP16, accumulating only the head reduction in FP32, and making
  ReLU/weighting in-place passed a numerical oracle (2.643e-4 relative L2 and
  exact top-64 overlap), but the live run still missed a 256 MiB BMM
  allocation by 5.06 MiB. Its steady chunks measured only 407--443 token/s,
  so allocator tuning would not satisfy the prefill target.
- A fused SM86 paged-MQA kernel now loads the checkpoint's packed E4M3 query
  and cache directly, converts each page into BF16 shared memory, executes
  the 64x128 products with native Ampere BF16 MMA, and performs ReLU,
  per-head weighting, reduction, and UE8M0-derived cache scaling inside the
  CTA. It preserves the FP8 cache and emits only `[queries, history]` logits;
  the enormous per-head tensor is never allocated. Against the independent
  Torch path it measured 4.304e-4 relative L2, 0.00298 maximum absolute
  difference, and exact top-64 sets. At the synthetic 1,024-query/8,192-key
  deployment shape it used 74.8 MiB peak versus 5,283.5 MiB and ran in
  23.73 ms versus 25.41 ms. The production launcher can therefore restore
  4,096-token prefill chunks while keeping the decode-winning GPU48 placement.
- A live GPU48 4,096-token-chunk request crossed the former indexer OOM, but
  the next scheduler subdivision changed from 4,096 to 2,048 tokens and
  exposed a distinct KTransformers pool deadlock: both ranks and every CPU
  worker waited on futexes with no active CUDA work. The unit and client were
  drained cleanly. Production uses stable 2,048-token chunks pending a
  variable-size pool-lifetime repair; this is not an indexer correctness or
  allocation failure.
- With 2,048-token chunks, the fused indexer completed the first full GPU48
  32,768/64 benchmark without OOM. It measured 440.55 input token/s,
  71,910.93 ms TTFT, and 38.91 ms TPOT (25.70 decode token/s). This made CPU
  native-MXFP4 prefill, rather than attention or decode, the remaining
  throughput limit. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v4c-fused-indexer-2k-random32k-out64-seed42.jsonl`.
- Profiling the AMX packed-weight path found that each decoded 32x32 E2M1
  weight tile expanded BF16 values to FP32, multiplied by a scale, then
  converted back to BF16. Native UE8M0 scales are exact powers of two; all
  real Flash layer-3 scale bytes are 119--124. The AMX kernel now adds the
  UE8M0 exponent delta directly to each nonzero BF16 codepoint, eliminating
  two BF16-to-FP32 expansions, two vector multiplies, and the packed
  FP32-to-BF16 conversion per tile. All 16 signed E2M1 values at each observed
  exponent 119--124 are bit-exact against FP32 multiplication. Exotic
  exponent ranges retain the IEEE fallback. The rebuilt SM86 AMX/AVX512
  extension SHA-256 is
  `58fecf37d174e5ef1f96e310015a5a4129f448d348f04c9cb9107f6417a14ecf`.
- The accepted `v4d` served gate uses that exponent kernel, the fused SM86
  indexer, GPU48 profile placement, 2,048-token chunks, native MXFP4 target
  and draft weights, DSpark, AMX prefill, and AVX-512 decode. Two repeated
  arithmetic probes returned exactly `4`. Its official 32,768/64 result is
  **504.79 input token/s**, 61,910.84 ms TTFT (529.3 fresh token/s), and
  **47.41 ms TPOT (21.09 single-stream decode token/s)**. This is the first
  DSpark Flash deployment to clear both requested throughput thresholds on
  the same required-length served request. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v4d-amx-expscale-random32k-out64-seed42.jsonl`.
- Two subsequent full-length decode probes remained stable far beyond the
  64-token gate, but stopped at 2,294 output tokens on both attempts. The
  server log proves this was not EOS: the separate sliding-window KV pool
  reached its default 6,400-token limit and the scheduler aborted the request
  while the full KV pool was only 54% occupied. The first measured
  61,965.14 ms TTFT,
  45.39 ms TPOT (**22.03 token/s**), and 4.77-token mean DSpark acceptance;
  the repeat measured 61,928.46 ms TTFT, 45.84 ms TPOT
  (**21.82 token/s**), and 4.81 mean acceptance. This is a capacity
  configuration defect rather than instability or a throughput collapse;
  production is moving the hybrid SWA/full ratio from 0.10 to 0.15 so its SWA
  pool exceeds the 4,096-token window plus the requested 4,096-token decode.
  Artifacts:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v4e-amx-expscale-random32k-out4096-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v4f-amx-expscale-random32k-out4096-ignore-eos-seed42.jsonl`.
- `v4g` validates the capacity repair end to end. Each TP rank allocated
  exactly 65,536 full-cache tokens and 9,728 SWA tokens, retaining about
  2.05 GiB free after pool creation. The streamed benchmark then completed
  **32,768 input tokens and exactly 4,096 output tokens** with no retraction:
  61,623.44 ms TTFT (**531.75 fresh input token/s**), 48.03 ms mean TPOT
  (**20.82 single-stream decode token/s**), and 4.78-token mean DSpark
  acceptance. Both server-counted and retokenized output lengths are 4,096.
  This is the accepted Flash deployment gate for the user's complete
  32K-fresh/4K-decode workload, and it clears both requested speed targets
  without changing the checkpoint's native MXFP4 format. Artifact:
  `/var/lib/exo/benchmarks/dsv4-flash-dspark-v4g-amx-expscale-swa15-random32k-out4096-seed42.jsonl`.

## 2026-07-26 11:58 EDT — clean Pro PP3 capacity diagnosis and PP-DSpark bring-up

- A fwuff process audit after the previous session ended found no unrelated
  stale inference process to kill. The named PP3 service was retained while
  its collective was healthy and drained only after another pipeline rank
  failed, so no user workload outside this deployment was touched.
- The first clean-source Pro target load used the `21,23,17` target-layer
  partition. Rank 2 on fwuff completed all 17 layers with 13.07 GiB GPU
  memory available. Dwagon rank 1 was OOM-killed while rank 0 was still
  loading. A second launch removed the external hard `membind`, but failed at
  the same class of allocation despite 476 GiB aggregate RAM remaining.
  Kernel evidence reported `CONSTRAINT_MEMORY_POLICY,nodemask=0`; both local
  KTransformers workers also printed `Numa Worker Pool at NUMA 0`. The
  capacity failure was therefore an explicit KTransformers NUMA-placement
  defect, not insufficient aggregate host memory.
- The clean July SGLang tree now exposes `--kt-numa-nodes` and carries it
  through `KTConfig` into `KTMoEWrapper`. A direct native-extension oracle
  constructed a worker pool with `[numa:threads][1:2]` and confirmed
  `Numa Worker Pool at NUMA 1`. The Pro launcher now selects physical NUMA 0
  for local PP0, NUMA 1 for local PP1, and NUMA 0 for fwuff PP2 while keeping
  AMX worker CPU affinity local. The third target-only load is in progress
  under this corrected placement.
- The current AMX extension was copied to fwuff and verified earlier at
  SHA-256
  `58fecf37d174e5ef1f96e310015a5a4129f448d348f04c9cb9107f6417a14ecf`.
  The clean SGLang Python tree and the shared PP launcher are synchronized to
  fwuff.
- Pipeline-aware DSpark is now implemented in the clean source for its first
  fixed-width correctness gate. Only the last target pipeline rank owns the
  complete standalone draft. It prepares the next proposal one iteration
  ahead and sends the proposal, next bonus token, accepted lengths, and new
  sequence lengths around SGLang's existing tensor output ring. Earlier
  stages run target verification with the ordinary forward proxy and
  reconstruct identical scheduler/KV state from the returned tensors. The
  first post-prefill decode is a one-token target bootstrap that injects the
  target auxiliary hidden state and seeds the first proposal. This avoids a
  reverse collective and its associated pipeline deadlock/latency.
- Non-last PP stages no longer instantiate a redundant 39.1 GiB draft or a
  draft KV pool. The bundled draft is forced to `pp_size=1` on the last
  stage. PP result serialization and `DFlashDraftInputV2` batch
  filter/merge operations now preserve the proposal fields. PP DSpark is
  deliberately gated to `SGLANG_RAGGED_VERIFY_MODE=static` until the
  fixed-width ring passes live coherency; confidence-driven compact layouts
  will be re-enabled only after that baseline.
- All touched PP-DSpark modules pass `py_compile`, selected Ruff undefined
  name/import checks, `git diff --check`, and an import oracle using the
  deployment runtime. A two-rank, two-GPU Flash smoke launcher at
  `/root/exo/scripts/dsv4_flash_dspark_pp2_smoke_v1.sh` is ready for the live
  arithmetic/KV-coherency gate after the current Pro target load is measured.

## 2026-07-26 12:31 EDT — coherent Pro PP3 target-only baseline

- The corrected `21,23,17` target-layer load completed on all three ranks.
  Each rank allocated the exact production cache capacities: 1,048,576 full
  tokens, 104,704 SWA tokens, 262,144 tokens at concurrency four, and 8,192
  tokens at concurrency 128. Explicit KTransformers placement held: PP0 used
  271,255 MiB private memory on dwagon NUMA 0, while PP1 used 286,406 MiB on
  NUMA 1. Fwuff held the 17-layer PP2 stage in 218.5 GiB host memory and
  14.1 GiB GPU memory.
- Local PP ranks initially collided because both API processes inherited port
  30000. The launcher now assigns local rank 1 port 30001 while retaining the
  externally served rank-0 and remote rank-2 ports at 30000. No model or
  collective change was required.
- Fwuff's first breakable decode-graph compilation took 280.66 seconds.
  During that interval the local schedulers were correctly blocked in the
  pipeline commit collective and fwuff had active `ninja`/`nvcc` children;
  this was compilation, not a stale-process deadlock. Once compiled, three
  independent temperature-zero arithmetic requests all returned exactly
  `4`. The two warm repetitions completed in 1.800 and 1.958 seconds.
- A sustained target-only `/generate` baseline used a 512-token server-counted
  prompt and forced exactly 64 output tokens with no retraction or EOS stop.
  It completed in 26.273 seconds end to end. The scheduler reported native
  AMX and AVX-512 dispatches, but target-only decode was only approximately
  4--5 token/s after prefill. This is a coherent correctness/capacity
  baseline, not an accepted performance result; DSpark remains mandatory for
  the 20 token/s gate. Artifacts:
  `/tmp/dsv4-pro-decode64-request.json`,
  `/tmp/dsv4-pro-decode64.json`, and
  `/tmp/dsv4-pro-decode64.curl`.
- All three named transient PP services were drained after the baseline.
  Their scheduler processes released both local GPUs and fwuff's GPU; no
  unrelated process was killed.
- A variable native-expert shard contract is now implemented in the clean
  SGLang/KTransformers source. A weights-only plan supplies arbitrary global
  expert IDs per layer and EP rank, validates shape/range plus exact disjoint
  coverage, and maps routed global IDs to rank-local native MXFP4 pointer
  lists. Negative/nonlocal routes are masked safely. KTransformers slices
  weight and UE8M0 scale pointers without converting or requantizing the
  checkpoint. This also changes wrapper ownership to MoE-TP rank so every
  EP rank can own a CPU expert shard without replicating it across MoE-TP.
- The plan generator
  `/root/exo/scripts/build_dsv4_kt_expert_shard_plan.py` assigns the coldest
  profiled experts to fwuff and greedily balances the remaining activation
  load across dwagon's NUMA nodes subject to exact capacity counts. A Flash
  oracle generated a `92,92,72` partition across all 43 MoE layers;
  loading recovered those exact tensor shapes and a CUDA mapping test passed
  for local, remote, and negative expert IDs. New Python files pass
  `py_compile`; both source trees pass `git diff --check`. The existing Ruff
  findings in KTransformers are two pre-existing unused imports in
  `amx.py`, not additions from this change.

## 2026-07-26 12:43 EDT — live PP2 DSpark coherency and NUMA policy

- The first Flash PP2 startup exposed two generic upstream guards which still
  assumed all speculative PP was unsupported. `ServerArgs` now admits only
  the PP-aware `DSPARK` algorithm, still requiring the non-overlap scheduler.
  `ModelRunner` now permits a partitioned DSpark *target* whose checkpoint
  advertises bundled MTP layers; partitioned draft workers and every other
  unsupported MTP/PP configuration remain rejected.
- With those narrow guards corrected, PP0 loaded target layers `[0,22)` and
  explicitly skipped the draft. PP1 loaded target layers `[22,43)` plus the
  complete three-layer bundled DSpark draft. Both ranks allocated matching
  8,192-token target pools; only PP1 allocated the separate draft pool. The
  live fixed-width proposal/result ring then passed three deterministic
  arithmetic probes, each returning exactly `4`. Warm end-to-end latencies
  were 0.718 and 0.678 seconds.
- A forced 512-input/128-output request exercised 23 repeated proposal,
  verification, acceptance, scheduler-state, and KV-state transitions. It
  completed with zero retractions, 5.565 mean accepted tokens, 0.9391 draft
  acceptance, and exactly 128 output tokens. The warm repeat produced a
  byte-identical output (`SHA-256
  0dac84cb0d45dc2cff198de3c80bfc28c1ef09fb6b1b4f323bddfaf6b57d70c0`)
  with identical acceptance statistics. Without decode graphs the repeat
  took 7.699 seconds, so this is a correctness smoke rather than the final
  performance layout. Artifacts are
  `/tmp/dsv4-flash-pp2-dspark-decode128{,-repeat}.json`.
- NUMA residency after both target stages and the last-stage draft were fully
  resident was 74,998 MiB local versus 411 MiB remote for rank 0 (99.45%
  local), and 80,257 MiB local versus 1,889 MiB remote for rank 1 (97.70%
  local). KTransformers already applies strict `HWLOC_MEMBIND_BIND |
  HWLOC_MEMBIND_STRICT` on its worker threads after binding them to the
  selected socket. A process-wide hard `--membind` is therefore unnecessary
  for the performance-critical native expert pools and is harmful for Pro:
  CUDA-pinned/runtime allocations can exhaust a 387 GiB socket while the
  other socket is free. Local launchers now add spill-capable
  `numactl --localalloc` for non-KTransformers allocations, retain strict CPU
  binding, and retain KTransformers' strict expert-pool binding. This
  tightens first-touch locality without reintroducing the demonstrated Pro
  OOM.
- Topology inspection confirms GPU 0 is local to NUMA 0 and GPU 1 to NUMA 1,
  with bonded NVLink between them. Both InfiniBand HCAs attach to NUMA 0.
  For the final Pro PP3 layout, assigning the local pipeline stage that sends
  the large hidden-state activation to fwuff to GPU/NUMA 0 is preferable;
  the much smaller returned logits/proposal control path may cross the socket
  boundary instead.

## 2026-07-26 13:17 EDT — PP speculative graph and strict-core NUMA repairs

- The journal was re-read before this iteration and had not changed since the
  previous read (`SHA-256
  59bf06a304ee42852b75f6b9f821896084c6c01279757a1dbcb816586e662d32`,
  mtime 12:40:15 EDT).
- The `23,25,13` Pro DSpark load was active rather than stale. All 61 target
  layers and all three complete draft layers finished loading. It then failed
  during breakable target-verify graph capture: dwagon PP1 reported an mHC/SWA
  tensor contract of one hidden row versus six positions, while fwuff PP2
  reported `invalid prefill plan: num_q < num_w`. Both symptoms came from the
  same generic PP graph-buffer defect: `hidden_states` and `residual` were
  allocated by request count (`max_bs=1`) instead of speculative token-row
  count (`max_num_token=6`). The graph return path also sliced PP proxy output
  to one request row.
- Both graph buffer constructors now allocate PP proxy payloads by
  `max_num_token`; the graph return path returns the physical speculative token
  rows. A CPU construction oracle recovered shape `[6, 28672]`. A live Flash
  PP2 reload then captured the complete six-token target-verify graph on both
  ranks in 2.8 seconds, proving the original Pro failure is repaired.
- The last Flash stage subsequently exposed a separate draft-graph issue:
  its intentionally lazy PP embedding performs a CUDA-to-host token-ID copy
  inside capture. `LazySafetensorTokenEmbedding.forward` is now an explicit
  `eager_on_graph` segment so the small host row-page operation bridges its GPU
  output back into the captured draft graph. A second live Flash graph gate is
  in progress.
- The NUMA audit found that good memory residency had hidden a strict-core
  affinity bug. KTransformers allocated `numa_threads_count` by selected
  *subpool count* but indexed it by physical NUMA ID. A one-subpool selection
  of node 1 therefore read out of bounds, produced a bogus core offset of
  `32768`, printed `Core ... inside NUMA node 1 not found`, and left 49 of 51
  AMX workers free to migrate within the outer node cpuset. Both the compute
  pool and job distributor now size physical-ID accounting from the configured
  topology and reject invalid IDs.
- The rebuilt native extension has SHA-256
  `4e30d25cb1cd2525ffd096030fe41c250da14a0d1a2a2ae9a7df5372dc4d8704`.
  A direct node-1 oracle created a four-thread pool and proved singleton strict
  affinities for workers on CPUs 57, 58, and 59 plus the distributor on CPU
  56. Thus the final policy remains: strict core and strict KTransformers
  worker-memory placement, spill-capable process-wide `--localalloc`, and no
  process-wide hard `--membind` on dwagon. This recovers the affinity that was
  accidentally loose without reintroducing the demonstrated single-socket Pro
  OOM.
- A persistent raw-IB native-MXFP4 expert sidecar, profile-guided cold-expert
  plan generator, and KTEP local/remote partial-sum integration are implemented
  in the working tree. They preserve global routing IDs until the native local
  shard boundary, overlap local CPU work with the sidecar request, and never
  convert or requantize checkpoint weights. Protocol/coherency validation is
  still pending before this path is enabled in the Pro launcher.

## 2026-07-26 13:32 EDT — graph-replay admission and fwuff expert oracle

- The second Flash graph launch captured both six-token target graphs and the
  last-stage draft graph, but its first ordinary decode bootstrap was
  incorrectly admitted to the target-verify graph. The live batch had one
  output-cache location while compressed-attention verification correctly
  expanded six causal positions, causing a fail-fast
  `raw_out_loc.shape=[1]` versus `seq_lens_casual.shape=[6]` assertion before
  any response was emitted. Decode graph admission now requires
  `TARGET_VERIFY` mode and exactly `batch_size * num_tokens_per_bs` physical
  input rows whenever the captured graph is a speculative target graph. A
  focused construction oracle rejects both the ordinary bootstrap and a
  malformed one-row verify batch.
- The next launch passed that repair: three deterministic PP arithmetic probes
  returned exactly `4` in 1.271, 0.706, and 0.670 seconds. However, two forced
  512-input/128-output replays exposed the previously characterized draft-graph
  incoherency: acceptance collapsed to 0.0065/0.0117, verify counts rose to
  123/120, and the greedy outputs differed. This layout is rejected. Both Flash
  and Pro launchers now set `SGLANG_DSV4_DRAFT_DISABLE_CUDA_GRAPH=1`, retaining
  the coherent target verification graph while keeping proposal-side dynamic
  DSpark state eager. A target-graph/draft-eager live gate is loading.
- The rebuilt NUMA extension was installed on fwuff; both its staged package
  and execution venv now match SHA-256
  `4e30d25cb1cd2525ffd096030fe41c250da14a0d1a2a2ae9a7df5372dc4d8704`.
  During the latest fully loaded Flash gate, rank 0 had 74,395 MiB local and
  512 MiB remote (99.32% local), while rank 1 had 79,800 MiB local and 1,623
  MiB remote (98.01% local). KTransformers worker threads were individually
  pinned to CPUs 1–52 and 57–108 respectively, with no missing-core warnings.
  The remaining cross-node pages are chiefly general CUDA/PyTorch anonymous or
  shared-library mappings, not unpinned expert workers. Tightening the whole
  process back to hard `--membind` would trade this small residual for the
  already reproduced Pro OOM, so it is not justified.
- A one-layer/one-expert Flash sidecar oracle found and repaired an invalid-route
  remap bug: global sentinel `-1` must remain local sentinel `-1`, not be
  converted to expert 0. After repair, the fwuff sidecar returned
  byte-identical BF16 partial sums to a local native-MXFP4 KTransformers
  wrapper for both one-token AVX-512 and eight-token AMX cases, including with
  the production 60-thread fwuff pool. The one-token and eight-token raw IB
  round trips were 4.151 ms and 2.289 ms; an all-sentinel request returned an
  exact zero tensor. This validates checkpoint expert selection, global/local
  ID remapping, wire dtypes, AVX/AMX execution, and the InfiniBand endpoint in
  isolation. Full-model integration remains gated on the current Flash replay
  and a representative Pro routing profile.

## 2026-07-26 13:48 EDT — coherent PP graph subset and full sidecar integration

- Target-graph/draft-eager alone was insufficient: two 512/128 runs still
  collapsed to 0.0065/0.0117 acceptance with 123/120 verifies and differing
  greedy text. Pipeline-stage isolation then made the fault precise. PP0
  target-graph plus PP1 target-eager restored 0.9167 acceptance, 5.333 accepted
  tokens/verify, and 64 outputs in 12 verifies. Thus the first-stage target
  graph is coherent; replay on a non-first/final PP stage is not.
- The breakable capture path cloned PP proxy inputs while building its Python
  closure. Because only CUDA segments and explicit eager breaks execute during
  replay, that clone can retain capture-time contents; breakable PP now passes
  the stable replay buffer directly, while monolithic graph behavior is
  unchanged. This repair is structurally correct but did not by itself restore
  the final-stage graph: two graph/graph 512/64 runs still accepted only one
  draft each and produced differing outputs. The production-safe PP policy is
  therefore graph first stages and force the final target stage eager with
  `SGLANG_DSV4_TARGET_VERIFY_EAGER=1`; no incoherent final-stage graph is being
  promoted.
- A route-forcing full integration plan selected Flash layer 3 expert 180, the
  hottest expert for that recorded decode workload (120/1,152 routes), while
  retaining one native sidecar expert in layers 0–2 for loader coverage. The
  local KTransformers wrappers loaded the 255-expert complements, and fwuff
  loaded the four selected checkpoint experts without conversion. The
  integrated PP0-graph/PP1-eager service established its persistent IB client
  during target capture.
- Three remote-integrated arithmetic probes returned exactly `4`. Two
  independent forced 512-input/64-output runs completed with zero retractions,
  0.9167 acceptance, 5.333 accepted tokens/verify, and 12 verifies in 4.886 and
  4.883 seconds. Both were byte-identical to one another and to the coherent
  all-local reference:
  `SHA-256 5f3d1152ef635e4a1253afb3597bc55132ea1edbf9badd05c1e6a3999fd5b11a`.
  This is the full-model proof that local complement plus fwuff partial sum is
  numerically lossless and preserves DSpark/KV state, not merely a standalone
  transport oracle.
- The sidecar now validates layer, batch, hidden-size, and top-k headers before
  allocating payloads, and records low-rate per-layer request/token/compute
  timing counters (first and every 128th request). The Pro profiling run can
  therefore separate native remote compute time from client/wire latency.
  Recorder mode is an opt-in Pro launcher switch with a 2,048-forward buffer
  (~183 MiB per rank for `[61,384]` int32 rows), large enough for the real
  32K/4K profile without crowding a 24 GiB GPU.

## 2026-07-26 13:55 EDT — Pro-scale NUMA strictness decision

- The corrected KTransformers extension is active in the three-rank Pro
  profiling load. Native pools initialized as physical NUMA 1/52 threads on
  PP0 and physical NUMA 0/52 threads on PP1, with socket-exact process CPU
  affinities `56-111,168-223` and `0-55,112-167`.
- Once roughly 145--147 GiB of Pro expert state was resident per local rank,
  `numastat -p` measured PP0 at 145,545 MiB local versus 3,992 MiB remote
  (97.33% local), and PP1 at 147,061 MiB local versus 489 MiB remote (99.67%
  local). This reproduces the fully loaded Flash result at Pro scale rather
  than relying on the small direct pinning oracle.
- Process-wide hard `--membind` should not be restored. It previously made the
  Pro load OOM despite ample memory on the other socket, while the current
  `--cpunodebind` plus `--localalloc` policy and strict native worker/core
  binding keeps the performance-critical expert pages overwhelmingly local.
  The residual 0.33--2.67% cross-node residency is loader/runtime state, not
  enough to justify sacrificing capacity. This decision will be rechecked
  after model-ready graph capture, but tightening the native worker placement
  further is neither necessary nor useful at this point.

## 2026-07-26 14:13 EDT — first full Pro DSpark service and recorder repair

- The `23,25,13` all-local profiling deployment loaded all 61 target layers
  and the complete three-layer DSpark draft. Target load times were 777.70 s,
  807.73 s, and 339.57 s on PP0, PP1, and PP2; the bundled draft added
  383.41 s on fwuff because its loader scanned the large sharded checkpoint
  before reaching the three `mtp.*` expert namespaces. All ranks allocated the
  requested 1,048,576-token full cache. Post-capture GPU reserves were 5.21,
  5.60, and 9.49 GiB.
- Three independent OpenAI chat arithmetic probes returned exactly `4`. Two
  official concurrency-one 512-input/128-forced-output runs completed without
  retraction. They measured 230.18 and 219.95 ms TPOT with 2.825 and 3.15
  DSpark accept lengths. The common-prefix gibberish completions diverged later
  despite temperature zero, so they are not promoted as a byte-repeat
  coherence proof; the semantic gate remained exact. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-profile2-coherence-{a,b}-random512-out128-seed42.jsonl`.
- Starting the target expert recorder and then issuing a fixed-ID request
  exposed a DSpark-specific recorder defect. PP2 failed in
  `_SelectExpertsSinglePassGatherer.on_select_experts` because the V4 draft
  invoked the global target recorder with no current target layer index;
  indexing `[61,384]` with `None` made the accumulator rank incompatible with
  the flattened expert IDs. The peer-closed errors on PP0/PP1 were secondary.
- V4 DSpark now follows SGLang's other MTP/NextN implementations and wraps its
  three-stage draft forward in
  `get_global_expert_distribution_recorder().disable_this_region()`. The cold
  placement profile is target-only by design, so draft routes must not be
  mixed into it. The repaired source compiles on both hosts and matches
  SHA-256
  `8cfa553760d9673923b1dfa21a450ea2275624365210003513986a07302a6479`.
  No stale process or occupied deployment port remained after the crash.
  Profile run 3 is loading with the repair.

## 2026-07-26 14:40 EDT — full-load NUMA verdict and profile-4 repairs

- The journal was checked again before this iteration and had not changed
  outside this session since the 14:13 entry (SHA-256
  `b1ebeb27d324ad81fb953367d63f394fd6e6887eed8ad7434ac371a098098dbd`,
  110,654 bytes). The profile-3 target load then provided the requested
  post-load NUMA evidence. PP0 had 287,733 MiB on its selected NUMA 1 versus
  10,754 MiB remote (96.40% local); PP1 had 323,569 MiB on its selected NUMA 0
  versus 331 MiB remote (99.90% local). Their exact process affinities remained
  `56-111,168-223` and `0-55,112-167`.
- PP0's remote pages were distributed across many 55.79 MiB anonymous
  `bind:1` native-expert mappings only after the preferred socket approached
  capacity. Restoring process-wide hard `--membind` would reserve almost all
  remaining socket memory before graph/cache/runtime allocations and has
  already caused a reproducible Pro OOM. The final policy therefore remains
  spill-capable `--cpunodebind` plus `--localalloc` on dwagon, strict
  KTransformers worker/core and expert-memory binding, and hard binding on
  fwuff's single NUMA node. The 0.10--3.60% spill is a much smaller cost than
  losing the deployment to a single-socket capacity failure. Residency will
  be remeasured after remote and GPU tiers remove several GiB of CPU buffers.
- Native MXFP4 placement can now compose all three disjoint tiers in one
  layer: profile-selected hot experts retain their global GPU mask, remote
  experts win any overlap, and the local KTransformers wrapper receives the
  compact CPU complement plus a separate compact all-CPU ownership mask.
  A 384-expert direct oracle recovered 378 CPU, three remote, and three GPU
  experts with exact disjoint/exhaustive coverage. The focused wrapper and
  partition suite passes; the synchronized wrapper SHA-256 on both hosts is
  `cd3c570cc0b1a1b8cafa27819c7e299f228b5054ffc679d3f8feded1cbeaacc2`.
- The profile-3 draft repair passed its first semantic request exactly, but
  target verification under breakable CUDA-graph replay then failed in the
  recorder. The IDs were already flattened; the actual defect was that an
  eager graph break replays after the model loop's Python
  `with_current_layer(i)` scope has exited, making the current layer `None`.
  The eager MoE bridge now re-establishes `decoder_layer.layer_id` on every
  capture and replay. A focused regression test proves the recorder context
  encloses the native MoE call.
- The DSpark checkpoint loader now applies a model-supplied tensor-name
  predicate before calling `safe_open.get_tensor`. The bundled draft admits
  only `mtp.*`; when `SGLANG_KT_DRAFT_GPU_EXPERTS=0`, it also skips
  `mtp.*.ffn.experts.*` because the native KTransformers loader reads those
  exact MXFP4 tensors directly. This avoids streaming the complete 831 GiB
  target checkpoint merely to retain the final draft namespaces. Loader,
  replay-context, and three-tier tests pass together: 21 passed. Profile 4 is
  active on all three ranks with target-only recording enabled; its fwuff
  startup will quantify the old 376.43-second draft scan against the filtered
  path.

## 2026-07-26 14:57 EDT — 11.6x draft-load win and re-entrant recorder scope

- Profile 4 measured the checkpoint filter on the complete deployment. Target
  loads took 811.97 s, 824.07 s, and 236.82 s on PP0/PP1/PP2. Once all target
  ranks reached the barrier, the complete three-layer DSpark draft loaded in
  32.40 s on fwuff, versus 376.43 s before pre-materialization filtering: an
  11.62x restart improvement. All 384 native MXFP4 experts in each `mtp.*`
  stage were still loaded by KTransformers; only the redundant 831 GiB target
  scan and duplicate SGLang expert materialization disappeared.
- The first replay-context repair correctly handles graph replay, but profile
  4 exposed its capture-time inverse: during initial capture the outer model
  loop already owns the same recorder layer, so entering the same
  non-reentrant `Withable` asserted. The recorder now provides
  `with_current_layer_if_absent`: it preserves identical capture-time nesting,
  establishes the layer during replay, and raises on a genuinely mismatched
  nested layer. Sixteen focused recorder and loader tests pass.
- Profile 5 is active on a fresh rendezvous. PP0 retains the only coherent
  target verification graph. PP1 and PP2 now disable decode graph capture
  entirely because their production policy already forces target verify eager
  and draft eager; capturing graphs that will never execute only wastes startup
  time and memory.
- The tiering implementation now accepts multiple disjoint native-MXFP4
  sidecar plans/endpoints. This admits an explicit frequency-balanced
  opposite-socket tier on dwagon in parallel with the local native pool, while
  fwuff independently owns the lowest-frequency experts and the hottest
  profile slots remain on the GPUs. Legacy single-sidecar deployments remain
  compatible. Nine focused tests cover plural parsing, duplicate rejection,
  compact local complements, GPU remapping, and two- or three-dimensional
  merged profiles. No tier is enabled before the real Pro profile establishes
  its route coverage and memory-safe counts.

## 2026-07-26 15:06 EDT — compact GPU ownership and deployable sidecar tiers

- Profile-guided GPU ownership now compacts the native host wrapper even when
  no remote plan is configured. Previously a GPU mask prevented execution on
  the CPU but still allocated the full native expert-weight storage, so hot
  GPU placement consumed VRAM without returning host capacity. The native
  wrapper now owns exactly the non-GPU logical IDs and remaps routes through
  the same sparse global-to-local table used by remote tiers. The added
  GPU-only regression brings the focused tiering suite to ten passing tests.
- The route planner writes two disjoint native-MXFP4 plans from the merged
  target profile. It exactly reproduces KTransformers' global hot-GPU slot
  selection, gives fwuff the least-used non-GPU experts, and selects a
  frequency-balanced fixed-size opposite-NUMA set from the remainder. Synthetic
  61-by-384 oracles prove that GPU, opposite-socket, fwuff, and local ownership
  are disjoint and exhaustive. The initial memory-safe live candidate is 32
  opposite-socket experts and four fwuff experts per served layer; counts will
  be adjusted from measured route coverage rather than assumed skew.
- Sidecars can restrict loading to a pipeline layer interval. The planned
  layout is PP0 on NUMA 1 with a strict NUMA-0 sidecar for layers 0--22, PP1
  on NUMA 0 with a strict NUMA-1 sidecar for layers 23--47, and a strict
  single-node fwuff sidecar for cold experts in layers 0--47. Local native CPU
  work is submitted first, so the opposite socket and IB sidecar compute run
  while the owning socket's AMX/AVX-512 pool is active. Per-endpoint client
  timing separates CUDA-to-host preparation, wire/remote compute, and return
  copy latency.
- Full native state does not multiply the temporary arena by the number of
  layers: KTransformers shares one sequential MoE scratch buffer per process.
  At 33.46875 MiB per Pro expert, the two 32-expert opposite-socket sidecars
  need about 24.05 and 26.15 GiB of weights for their respective layer
  intervals plus one arena each. Four cold experts across the first 48 layers
  need about 6.28 GiB on fwuff plus one arena. Compact local and GPU ownership
  removes more duplicated host weight storage than these sidecars add.
- Profile 5 confirms the conservative graph policy materially improves fwuff
  headroom: PP2 peaked at 235 GiB while loading but settled at 166.8 GiB after
  its target stage, versus the earlier deployment that retained unused graph
  state. The rank is waiting at the pipeline barrier while dwagon completes
  its slower memory-mapped target load. No stale fwuff process is present.

## 2026-07-26 15:18 EDT — recorder gate passed; cancelled-warmup PP incident

- Profile 5 reached service-ready state with all 61 target and three draft
  stages. Target load times were 740.80, 787.74, and 274.08 seconds; the
  coherent PP0 target-verification graph captured in 3.08 seconds. All ranks
  allocated the full 1,048,576-token pool, and a deterministic OpenAI
  arithmetic gate returned exactly `4`.
- The first route-profile command exposed a benchmark harness hazard before a
  result could be accepted. `sglang.benchmark.serving` defaults to one warmup
  sequence even for a one-prompt run, so it began an unreported second full
  32K/4K request. That would exceed the 2,048-forward stat buffer in the worst
  case and contaminate the placement profile. The client was interrupted
  during the first prefill chunk, recording was stopped, and the partial
  profile was rejected.
- Interrupting the streaming client did not promptly cancel its native
  `/generate` request. A subsequent recorder dump and explicit abort exposed a
  pipeline-control ordering defect: PP0 and PP1 became idle while fwuff's main
  scheduler remained in the CUDA driver with the NCCL proxy spinning and all
  60 native workers asleep. Health and dump requests stopped making progress.
  This was confirmed with a native GDB stack rather than inferred from GPU
  utilization. The three exact profile-5 units were killed; no unrelated
  process was touched, and host memory and ports were verified released.
- Profile 6 is loading on fresh rendezvous `10.44.0.1:29533` with the same
  repaired source and placement. Its exact benchmark will set
  `--warmup-requests 0` and a stable explicit request ID. Recorder start,
  workload, stop, and collective dump will occur only at request boundaries;
  no PP control operation will be injected mid-prefill. The incident does not
  weaken the NUMA result: profile 5 again measured about 97.4% selected-node
  residency on PP0 and 99.8% on PP1 near full load.
- Source inspection found that the recorder dump would deadlock even at a
  clean request boundary. PP control messages are forwarded stage by stage,
  but `_StatAccumulator.dump` entered a world `all_reduce` on PP0 before its
  event loop could forward the dump to PP1. The endpoint communicator also
  fans out by DP size, not PP size. The stat recorder now skips that collective
  only when `pp_size > 1`, writes one uniquely world-ranked global-shape file
  per stage, and leaves the existing TP/EP reduction unchanged. The offline
  merge then sums the disjoint populated layer rows.
- Two focused unit tests prove that PP dumping never calls `all_reduce` and
  writes the world-rank suffix, while non-PP dumping still performs the exact
  original SUM and rank-0 write. Together with the replay-context tests:
  four passed. Both hosts run identical recorder source, SHA-256
  `89362ca716baf929cbb0892cddb4d89e5adddd0bb8622de407d2813a28039180`.
  Profile 6 was recycled before its local expert load began; profile 7 is
  active on `10.44.0.1:29534` with the retrievable PP recorder.
- Remote requests now compact by token row before leaving the GPU. Each tier
  builds a row mask from its global logical-expert mask, sends only hidden
  states/routes/weights for tokens that actually select that tier, and
  index-copies the returned partial sums into the full output shape. Disjoint
  tiers still add exactly. For a uniform six-of-384 route, the four-expert
  fwuff tier selects about 6.25% of prefill rows, reducing its expected
  hidden-state request/response payload by roughly 16x; the 32-expert
  opposite-NUMA tier selects about 41%, reducing payload about 2.4x. A
  two-tier oracle proves only the selected rows are sent and that scattered
  sums are exact.
- Multiple selected remote tiers now execute concurrently through a persistent
  bounded executor after their GPU row compaction. This overlaps the
  opposite-NUMA CPU pool, fwuff CPU pool, and already-submitted local native
  pool instead of serializing the two sidecar round trips. The concurrency
  oracle places both fake endpoints behind a two-party barrier, so it would
  time out under the former sequential implementation; all eleven focused
  tiering tests pass. The latest synchronized wrapper SHA-256 is
  `aadd466009605affb81a3dd20979898cb5abf045cd886000a8a2e540d18a29e4`.
  A scheduler-stream CUDA event gates both executor workers after compaction;
  this preserves cross-thread stream correctness without a device-wide
  synchronize, and the focused concurrency test passes with that guard.

## 2026-07-26 15:56 EDT — exact Pro route profile and measured final tier plan

- Profile 7 completed the exact production-shaped route trace with 32,768
  fresh input tokens, 4,096 forced output tokens, no benchmark warmup, and the
  repaired per-pipeline-rank recorder. It took 924.50 s, measured 4.43048
  output token/s, 163,224 ms TTFT with recording enabled, 185.901 ms TPOT,
  and 4.57727 average DSpark acceptance. The output length was exactly 4,096.
  Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-profile7-alllocal-routes-random32k-out4096-seed42.jsonl`.
- Each pipeline rank produced a global-shape `(2048,61,384)` recorder tensor
  with exactly 902 populated target forwards. The populated layer intervals
  are 0--22, 23--47, and 48--60. Every one of the 61 layers contains exactly
  228,762 selected expert routes. Durable raw copies are under
  `/var/lib/exo/profiles/dsv4-pro-profile7/rank{0,1,2}.pt`; their merged
  `(61,384)` tensor contains 13,954,482 routes, covers all layers, and has
  SHA-256
  `dc7dbeaa24434feee385bda6dc5a5c9535e0cd467dd8872e230f1d40dbb70267`.
- Real post-capture VRAM reserves support stage-specific hot-expert budgets:
  seven per PP0 layer, four per PP1 layer, and twelve per PP2 layer. These
  place 131, 133, and 158 unique native MXFP4 expert slots on the three GPUs,
  consuming about 4.282, 4.347, and 5.164 GiB. A direct mask oracle proves the
  staged planner reproduces each rank loader's global hot-expert mask exactly.
- The final first-48-layer layout assigns 64 frequency-balanced experts per
  layer to the opposite dwagon socket and four least-used experts per layer
  to fwuff. The measured route shares are 19.7482% GPU, 39.2953%
  opposite-socket, 0.0037% fwuff, and 40.9527% owning-socket native CPU.
  Thus the two dwagon AMX/AVX-512 pools receive almost equal remaining route
  work while fwuff keeps genuinely cold weights and continues to run PP2 plus
  the full DSpark draft. All four ownership sets are disjoint and exhaustive.
- The opposite-socket plan is
  `/var/lib/exo/plans/dsv4-pro-profile7/final/opposite-numa64.pt`, SHA-256
  `242224986957a307dc00f995a9bfe8df888ef77cd4ba818ada707aa242f169d6`;
  the fwuff plan is
  `/var/lib/exo/plans/dsv4-pro-profile7/final/fwuff-cold4.pt`, SHA-256
  `466a315c979091a97d5338c0739fe171682aa09ad71bbbbfa8cdfffbfe129e0a`.
  Both plans and the merged profile match by hash on dwagon and fwuff.
- The NUMA policy is retained on measured evidence. Near full load, PP0 placed
  180,045 MiB on selected NUMA 1 versus 4,856 MiB remote (97.37% local), and
  PP1 placed 190,626 MiB on selected NUMA 0 versus 444 MiB remote (99.77%
  local), while those selected nodes had only about 1.7 and 4.7 GiB free.
  A hard process-wide memory bind would convert this small capacity spill into
  an OOM. CPU execution, native worker pools, and sidecar memory remain
  strictly bound; only main-process allocation retains local-first spill.
  Final residency will be measured again after compact GPU/remote ownership.

## 2026-07-26 16:13 EDT — final tier launch and fwuff GPU-path portability fix

- All three native-MXFP4 sidecars loaded and became ready before the main
  ranks started. The NUMA-0 layer-0--22 sidecar occupies 50,323 MiB with
  50,083 MiB on node 0 (99.52% selected-node residency); the NUMA-1
  layer-23--47 sidecar occupies 54,640 MiB with 53,583 MiB on node 1
  (98.07%). The fwuff sidecar occupies 7,593 MiB on its sole NUMA node.
  Their listeners are `127.0.0.1:29562`, `127.0.0.1:29563`, and
  `10.44.0.2:29561`.
- The first compact main launch correctly constructed disjoint per-layer
  ownership on PP0/PP1: 68 remote experts (64 opposite-socket plus four
  fwuff), the profile-selected variable hot-GPU slice, and the compact local
  complement. PP0 and PP1 progressed normally. PP2 reached hot-GPU
  postprocessing, then exposed a previously dormant host-path defect:
  `mxfp4_triton_kernels_moe.py` defaulted to the dwagon-only absolute path
  `/root/exo/vendor/ktransformers/.../v4_triton_kernels_moe.py`. All earlier
  fwuff deployments used zero target GPU experts, so they never entered this
  portable SM86 GPU path.
- The adapter now resolves the proven portable kernel below the active Python
  source roots, while retaining `SGLANG_V4_TRITON_KERNEL_PATH` as an explicit
  override. Direct imports resolve to `/root/exo/vendor/ktransformers/...` on
  dwagon and `/var/lib/exo/sources/ktransformers-dsv4-native-mxfp4/...` on
  fwuff. Both synchronized files have SHA-256
  `f82fb70f8f1df74eb18517905e6f50f70017475f95cc55f5670d50e0ae56188a`.
- Synchronization also found fwuff's root filesystem at 100%. The model volume
  `/mnt/sanic` is separate and unaffected. Purging only regenerable pip cache
  removed 11,162.1 MiB; no model, profile, plan, source, or user artifact was
  deleted. The three main ranks were retired after PP2 failed because an NCCL
  process group cannot safely admit a replacement rank. The already-loaded
  sidecars remain healthy for the corrected restart.
- The apparently unexplained remaining disk pressure was traced to two stale
  OpenSM instances for absent GUIDs `0xe41d2d03004d32e1` and
  `0xe41d2d03004d32e2`. Each was continuously logging `umad_receiver` I/O
  errors into a 238--239 GiB file. The live `mlx5_0` port has GUID
  `0x248a070300a32154`; it remained Active/LinkUp at 100 Gb/s with SM LID 5.
  Only the two absent-port units were stopped and their generated error logs
  truncated. The valid subnet manager remained active, and fwuff root free
  space recovered to 324 GiB. This also satisfies the user's authorization to
  remove stale fwuff processes without disrupting the active IB fabric.

## 2026-07-26 16:44 EDT — final tier semantic gate rejects the compact GPU path

- The corrected three-rank launch completed target loading on PP0/PP1/PP2 in
  663.91/719.74/326.14 s, loaded the DSpark draft on fwuff in 26.52 s,
  allocated the full 1,048,576-token cache, captured PP0's breakable graph in
  34.74 s, and reached a live OpenAI endpoint. The portable SM86 GPU path was
  exercised successfully on fwuff rather than failing at import time.
- The first deterministic arithmetic semantic gate failed. For
  `What is 2+2? Respond with only the number.` at temperature zero and
  `max_tokens=32`, the model repeated fragments of the prompt instead of
  returning `4`. This run is rejected as incoherent and is not a benchmark
  candidate.
- The remote-only tier path had previously returned the exact answer, as had
  the all-local profile deployment. The new execution path in this rejected
  run is the profile-selected compact native-MXFP4 GPU expert set on all three
  SM86 GPUs (about 20--30% of measured routed work by stage). Ownership masks
  remain disjoint and exhaustive. The next gate is therefore a one-layer,
  one-expert numerical oracle comparing the real Pro GPU loader/kernel against
  the native KTransformers CPU MXFP4 result. No argument tuning or throughput
  benchmark will proceed until that oracle is correct.
- The NUMA decision is unchanged by this failure: it is a numerical failure in
  the GPU-expert path, not evidence about main-process cross-node allocation.
  The selected-node residency remains 97.37% and 99.77% near capacity, whereas
  hard process-wide memory binding previously caused a real OOM. Strict CPU
  worker and sidecar execution affinity remains in force.

## 2026-07-26 17:03 EDT — focused oracle identifies stale compact-shard loader

- A direct layer-10 oracle loaded seven real Pro experts and gave the portable
  SM86 GPU kernel and native AMX kernel identical packed MXFP4 weights, scales,
  activations, compact IDs, and top-6 routes. The result was numerically sound:
  cosine similarity `0.9999761`, relative mean difference `0.0027083`, and
  maximum BF16 difference `0.0625`. A one-expert low-amplitude case reached
  cosine `1.0000033` and relative mean difference `0.0000171`. The GPU kernel
  is therefore not the source of the rejected model output.
- The real opposite-NUMA sidecar failed the same oracle: cosine `0.11349` for
  six selected layer-10 experts and `0.02388` for a single requested expert.
  Comparing that single result against checkpoint experts proved that a request
  for global expert 5 was executing expert 0 with cosine `1.0000018`.
- The deployment environment explains the exact permutation. Both sidecar and
  main launchers put `/root/exo/vendor/ktransformers` on `PYTHONPATH`, but the
  importable package sources are one level lower under
  `kt-kernel/python`. Python consequently loaded the older installed
  `kt_kernel.utils.amx`, which has no `weight_expert_ids` compact-shard filter,
  while using the separately rebuilt current native extension. Every compact
  native CPU tier therefore stored checkpoint experts `0..N-1` but remapped
  routes as if it stored the profile-selected IDs. This affects local
  complements and both sidecar tiers and fully explains the semantic failure.
- The earlier attribution to the GPU path is superseded by this direct
  evidence. The repair must update the installed Python package without
  replacing the current rebuilt AMX/AVX-512 extension, restart all compact
  sidecars, and pass both per-tier numerical oracles before another full model
  launch.

## 2026-07-26 17:18 EDT — compact shards repaired and all sidecar tiers coherent

- Only `kt_kernel/utils/amx.py` differed between the current source and the
  installed runtime package. It was synchronized on both hosts while retaining
  the rebuilt host extension. Source, dwagon runtime, and fwuff runtime now
  share SHA-256
  `1ec33562f6a72fc036c7a0fb0315620651fdd51db1c17b008a90d95898cd33e7`;
  both runtime imports expose the `weight_expert_ids` filter. The native
  extension was deliberately not copied between hosts.
- Both launchers now inspect `NativeMoEWrapper.load_weights` and fail before
  model allocation if compact-shard support is absent. This prevents the stale
  installed-Python/current-extension split from silently recurring.
- A patched one-expert `KTMoEWrapper` loaded global expert 5 into compact slot
  zero and matched the direct checkpoint oracle bit-for-bit: cosine
  `1.0000019`, relative mean difference `0`, maximum difference `0`.
- All three sidecars were rebuilt from their selected global IDs. Independent
  real transport oracles now pass:
  - opposite NUMA-0, layer 10, six experts: cosine `1.0000012`, relative mean
    difference `8.79e-6`, maximum BF16 difference `0.03125`;
  - opposite NUMA-1, layer 30, six experts: cosine `0.99999994`, relative mean
    difference `3.52e-6`, maximum difference `0.015625`;
  - fwuff over IB, layer 10, all four cold experts: cosine `1.0000018`,
    relative mean difference `0`, maximum difference `0`.
- Strict sidecar placement remains effective after the rebuild. NUMA-0 holds
  50,192 MiB locally versus 141 MiB remotely (99.72% selected-node);
  NUMA-1 holds 53,497 MiB locally versus 1,153 MiB remotely (97.89%);
  fwuff's 7,611 MiB is on its only node. This supports retaining strict
  sidecar binding while leaving only the capacity-constrained main processes
  on local-first allocation.

## 2026-07-26 17:19 EDT — corrected full service coherent; NUMA and overlap gate

- The corrected `tiered3` full service completed target loading in 579.77 s
  on PP0, 620.25 s on PP1, and 463.73 s on PP2. The bundled DSpark draft then
  loaded on fwuff in 24.54 s. All three ranks allocated the
  1,048,576-token cache and the OpenAI endpoint became ready without an error
  or OOM.
- Both post-load semantic gates pass. Greedy arithmetic returned exactly `4`;
  a separate natural-language probe correctly returned Paris and the Seine.
  The repaired compact native-MXFP4 shard ownership is therefore coherent in
  the complete GPU/local-CPU/opposite-NUMA/fwuff pipeline, and the previously
  rejected prompt-repetition result is resolved.
- Final main-rank locality was 233,609.81/242,721.87 MiB (96.25%) on PP0's
  selected NUMA-1 node and 263,552.34/263,973.69 MiB (99.84%) on PP1's
  selected NUMA-0 node. The sockets retained only 9.1 and 11.2 GiB free.
  Sidecars remained strict: 50,180.54/50,295.11 MiB (99.77%) on NUMA 0 and
  53,474.05/54,610.68 MiB (97.92%) on NUMA 1; fwuff's PP2 remained wholly on
  its only node. PP0 and PP1 CPU affinity masks remained exactly their own
  sockets.
- This reaffirms the NUMA decision with full-residency evidence. Tightening
  the main ranks to hard process-wide `membind` would remove only PP0's
  3.75% capacity spill while eliminating the remaining headroom and
  reintroducing the already observed OOM failure mode. The spill is not the
  dominant performance cost: a coherent served 8-input/64-output baseline
  measured 217.20 ms TPOT (4.60 token/s), only slightly above the 4.43
  token/s all-local profile deployment. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered3-coherent-random8-out64-seed42.jsonl`.
- The baseline trace instead exposed a lost-overlap defect. The main wrapper
  submitted its local AMX/AVX-512 shard asynchronously, then synchronously
  waited for all remote sidecar tiers before launching the selected native
  MXFP4 GPU experts. Remote tiers now launch as persistent executor futures;
  their compact-transfer and remote AMX/AVX-512 work overlap both the local
  CPU shard and SM86 GPU kernel, and are joined only at the partial-sum merge.
  All 12 focused wrapper tests pass, including a nonblocking-join test and the
  two-tier concurrency test. The synchronized wrapper SHA-256 on dwagon and
  fwuff is
  `86a91db623ce050380903db2f0d241d1a42875e59895c1c9e8c3d400d39e65cc`.
  `tiered4` is loading on a fresh rendezvous for a direct served comparison.

## 2026-07-26 17:43 EDT — three-way overlap accepted; tiled AVX decode prepared

- `tiered4` loaded coherently and returned exactly `4`. The matched
  random-8/output-64 served probe reduced TTFT from 9,745.82 to 1,155.67 ms
  and TPOT from 217.20 to 179.24 ms, a 17.5% decode-latency reduction from
  joining remote work only after local CPU and GPU expert execution. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered4-overlap-random8-out64-seed42.jsonl`.
  A longer forced 256-token probe remained stable at 190.34 ms TPOT
  (5.25 token/s), 1,194.50 ms TTFT, and 4.97-token mean DSpark acceptance:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered4-overlap-random8-out256-seed42.jsonl`.
- A proposed AVX-512 decode optimization fused native UE8M0 power-of-two
  scaling into the transient BF16 exponent before `VDPBF16PS`. It was
  bit-exact but gave no core-pinned speedup (one-token K=4096 remained
  0.156 ms), so it was rejected and removed.
- The same benchmark exposed a useful kernel crossover that decode never
  selected. At the real Pro K=7168 shape, the existing four-token tiled
  AVX-512 kernel takes 0.244 ms versus 0.519 ms for independent mat-vec
  execution (2.13x faster); at six tokens it takes 0.516 versus 0.778 ms
  (1.51x). Native-MXFP4 decode now selects the tiled AVX-512 path when an
  expert receives at least four verification tokens, while retaining
  one/two-token mat-vec and the measured eight-token AMX crossover.
  The checked synthetic oracle remains bit-exact at M=1/2/4/6.
- Separate host-native Release extensions were rebuilt with CUDA stream
  integration, AMX, AVX-512 BF16/VNNI/VBMI, and SM86 support. The staged and
  installed dwagon binary has SHA-256
  `aa34ba86351ff52b847176c672c1394226ee3fbffad55fd7dd2224e6b8530868`;
  fwuff's separately compiled binary is
  `3e889040e3c7080717ccbfdfc77f5a69fa26e21c610cb3d3ac8d21a8eb5c31c7`.
- The rebuild audit found that fwuff's broad `kt_kernel_ext.*.so` loader glob
  was choosing an in-package recoverable `pre-expscale` backup by directory
  order instead of the canonical extension. The running fwuff PP2 and cold
  sidecar were therefore coherent but executing that older binary. Backups
  are now outside the import directory, and extension selection matches only
  Python's exact ABI suffix. Source and both installed `_cpu_detect.py` files
  have SHA-256
  `87af491933e6d2f3f13553e3e6374e60673ed3abbfef7e8de9494b0efdefe9b7`;
  clean debug imports on both hosts resolve the canonical native binaries.
  The new binaries will become active after the controlled main/sidecar
  restart and must pass the real layer oracle before another served result.

## 2026-07-26 17:56 EDT — rebuilt tiled tier passes residency and transport gates

- All three compact native-MXFP4 sidecars were restarted against the canonical
  host-native extensions. `/proc/*/maps` resolves only the canonical ABI file:
  SHA-256
  `aa34ba86351ff52b847176c672c1394226ee3fbffad55fd7dd2224e6b8530868`
  on dwagon and
  `3e889040e3c7080717ccbfdfc77f5a69fa26e21c610cb3d3ac8d21a8eb5c31c7`
  on fwuff; neither recoverable backup is mapped.
- Strict sidecar placement remains effective with the new binary. The NUMA-0
  sidecar has 50,194.37/50,321.93 MiB on node 0 (99.75%) and exact CPU
  affinity `0-55,112-167`. The NUMA-1 sidecar has 53,475.05/54,638.71 MiB
  on node 1 (97.87%) and exact affinity `56-111,168-223`. fwuff's 7,601.31
  MiB is on its sole node. The IB port is Active at 100 Gb/s with 0.414 ms
  two-packet ping average.
- The real-weight oracle now optionally exercises a running sidecar through
  the production wire protocol. Layer 10 on NUMA 0, layer 30 on NUMA 1, and
  layer 10 on fwuff all returned BF16 outputs byte-identical to direct native
  AMX/AVX-512 execution. The independent SM86 comparisons had cosine
  `>=0.9999975` and relative mean difference `<=0.0026175`. Six-token cases
  emitted the new tiled AVX-512 dispatch marker, so this checks the intended
  binary and path rather than only reload correctness.
- `tiered5` is loading on fresh rendezvous `10.44.0.1:29539` with the same
  coherent profile-guided ownership, three-way local/remote/GPU overlap,
  1,048,576-token cache, and DSpark configuration as `tiered4`. This is the
  controlled served before/after gate for the tiled decode kernel.

## 2026-07-26 18:14 EDT — tiled served result rejected; decode-call placement staged

- `tiered5` completed target loading in 600.26/590.79/487.05 seconds and the
  fwuff DSpark draft in 30.83 seconds. It allocated the full 1,048,576-token
  cache, mapped only the canonical rebuilt extensions, and returned exactly
  `4` at the arithmetic gate.
- Final main residency was 234,063.75/242,546.30 MiB on PP0's selected NUMA
  node (96.50%) and 263,449.67/263,810.40 MiB on PP1's selected node
  (99.86%). The nodes retained only 9.3 and 11.0 GiB free. This is another
  independent full-load confirmation that process-wide hard `membind` would
  risk OOM for a small spill reduction; native worker and sidecar placement
  remain strict.
- The matched random-8/output-64 served probe regressed to 327.80 ms TPOT; a
  no-warmup repeat measured 368.08 ms and exposed mean DSpark acceptance of
  only 2.42, versus 4.97 on the accepted `tiered4` run. These artifacts are
  diagnostic and rejected:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered5-tiled-avx-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered5-tiled-avx-random8-out64-seed42-repeat2.jsonl`.
- Focused follow-up does not find a tiled-kernel numerical fault. A real
  64-expert mixed-route six-token oracle returned the identical BF16 SHA-256
  `9f9ba97df691137ce7533fce3c6aa8d927303f6a272077873e0d1bbc85f8c327`
  at tiled thresholds four and 32. Old/new AMX M=8 outputs were also identical,
  and a 64-expert mixed twelve-token fwuff comparison between the previously
  mapped `pre-expscale` binary and the current canonical binary returned the
  same BF16 SHA-256
  `e0f60fc2322a3fd0bdfedd8b3159b2e32281b73ef816706eacd00b9c4bd7cfe9`.
  The service-level acceptance change is therefore not attributed to an
  arithmetic mismatch, but no tiled served speed claim is accepted.
- Analysis of all 893 recorded six-token target verifications found that the
  existing GPU layout optimizes route volume, not distinct expert calls.
  Repeated routes share one weight stream and are precisely where tiled/AMX
  kernels are efficient; singleton expert calls dominate decode bandwidth.
  A new decode-call profile and exact-slot planner preserve the measured
  131/133/158 GPU expert slots on PP0/PP1/PP2 while selecting by call
  frequency. The resulting disjoint/exhaustive plan projects bottleneck CPU
  call reductions of 14.5%, 39.7%, and 16.1% respectively, with the first 48
  layers split 13.24% GPU, 42.19% opposite NUMA, 0.0075% fwuff, and 44.57%
  owning CPU by call count.
- Explicit variable-width GPU mask plan loading is implemented and all 15
  focused KTEP tests pass. The synchronized wrapper SHA-256 is
  `7be960a871feb6c68b668ff3b682ca89bcfddb4a5a93852281373e39174f9a22`.
  The decode-call profile SHA-256 is
  `29a7686c2c3477eca83c415ed9e7cab051024a63f6b84d8589e853d49f0b5ea6`;
  mask/opposite/fwuff plan hashes are
  `f6310b31d39622988516c17845c8d6efaf5d496ec409213f3d8a25dbd7c33d3b`,
  `5f3c47e21d0a854062aaacc93b97bb3a8781c97aea32184a2e6b97a489b2794a`,
  and
  `8271dbfd19b3cdb02c347b3b543c21e0978800cf488aee2298d9d99f22b80fa2`.

## 2026-07-26 18:30 EDT — NUMA policy retained; bidirectional tier planned

- The journal was re-read after the user's change check. Its newest entry was
  still the 18:14 decode-call-placement section; no newer external edit was
  present. The file SHA-256 before this entry was
  `b3137764133d0f85e4fec1b208369124fcf74544d744322cf078bc67bf98c4b5`.
- The NUMA policy is retained rather than relaxed further or tightened
  process-wide. Two coherent full Pro loads put 96.25--96.50% of PP0 and
  99.84--99.86% of PP1 on their selected nodes, while strict main-process
  `membind` has already OOMed. CPU affinity, native AMX/AVX-512 worker memory,
  and sidecars remain strictly bound; only the capacity-constrained main
  process uses `--cpunodebind` plus local-first allocation. Cross-node spill is
  therefore a bounded 3.5--3.75% PP0 capacity valve, not an unmeasured default.
- A memory-neutral bidirectional decode plan now reuses otherwise idle dwagon
  CPU sockets while PP2 runs. It expands fwuff's cold tier from four to 36
  experts for layers 0--47, then assigns 64 PP2 experts per layer to each
  dwagon socket. The added 3,200 sidecar expert slots exactly replace 3,200
  main-rank slots. Measured PP2 decode calls divide as 25.15% GPU, 24.95%
  dwagon NUMA 0, 24.96% dwagon NUMA 1, and 24.95% fwuff local.
- The planner compiles, passes Ruff, and produced active-range-disjoint plans
  with exact shapes `(61,64)`, `(61,64)`, and `(48,36)`. Dwagon NUMA-0,
  NUMA-1, and fwuff plan SHA-256 values are respectively
  `c96dcd60b0175a24c4c23bdb1afac5c0b789a5d5358e03294fd76bd27a8af3dc`,
  `ba720d2125f635251af75d18ac023fca9f87355eb4233fc32baf1d887b353645`,
  and
  `f8fc7649ff730125571a7dc0372f1ea656c96dc18862e8ed68c3da559afaf209`.
  Identical files are staged on both hosts but remain inactive while the
  controlled `tiered6` decode-call-placement run loads.

## 2026-07-26 18:41 EDT — decode-call placement coherent but rejected

- `tiered6` completed target loading in 584.08/626.72/461.85 seconds and
  loaded the fwuff DSpark draft in 26.69 seconds. Explicit GPU-mask accounting
  produced exactly 131/133/158 expert slots on PP0/PP1/PP2. The service
  allocated the full 1,048,576-token cache, mapped only each host's canonical
  native extension, and returned exactly `4` plus a correct Paris/Seine
  sentence at independent semantic gates.
- Final main-rank locality remained strong under local-first allocation:
  PP0 held 233,246.18/242,661.23 MiB on selected NUMA 1 (96.12%), and PP1
  held 263,537.82/263,905.43 MiB on selected NUMA 0 (99.86%). CPU affinity
  was exactly `56-111,168-223` and `0-55,112-167`. Strict sidecars remained
  at 50,148.50/50,272.08 MiB on NUMA 0 (99.75%) and
  53,504.98/54,590.09 MiB on NUMA 1 (98.01%); fwuff's one-node main and
  sidecar held 203,748.20 and 7,614.78 MiB. This is further evidence against
  hard-binding the already capacity-saturated main ranks.
- The matched random-8/output-64 probe measured 1,731.67 ms TTFT and
  287.12 ms TPOT (3.48 token/s). The longer output-256 gate measured
  1,172.86 ms TTFT, 298.62 ms TPOT (3.35 token/s), and 3.225 mean DSpark
  acceptance. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered6-decodecall-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered6-decodecall-random8-out256-seed42.jsonl`.
- This layout is coherent but rejected. Its approximate target-verification
  interval is `298.62 * 3.225 = 963.05` ms, close to `tiered4`'s
  `190.34 * 4.97 = 946.00` ms. The GPU placement did not materially lower
  whole-model verify cost on this prompt, while lower DSpark acceptance
  explains most of the TPOT regression. The next controlled topology is the
  already-staged bidirectional CPU plan, which attacks PP-stage idleness
  independently of draft acceptance.

## 2026-07-26 18:59 EDT — cold bidirectional topology coherent but rejected

- `tiered7` completed target loading in 701.82/648.50/494.17 seconds, loaded
  the fwuff DSpark draft in 24.50 seconds, and allocated the complete
  1,048,576-token cache. Independent semantic gates returned exactly `4` and
  the correct Paris/Seine sentence. Every rank and sidecar maps the canonical
  native-MXFP4 extension; no experimental kernel is active in this result.
- NUMA locality remains controlled under the retained hybrid policy. PP0 held
  209,663.05/218,036.46 MiB on selected NUMA 1 (96.16%) and PP1 held
  236,767.20/237,144.36 MiB on selected NUMA 0 (99.84%), with exact
  socket-local CPU affinity. The four dwagon sidecars remained strict at
  50,087.21/50,209.82 MiB on NUMA 0 (99.76%),
  53,493.69/54,527.70 MiB on NUMA 1 (98.10%),
  28,502.27/28,624.83 MiB on NUMA 0 (99.57%), and
  28,045.34/28,624.16 MiB on NUMA 1 (97.98%). fwuff's main and sidecar remain
  wholly on its only NUMA node.
- Initial six-token graph-capture samples showed about 2--3 ms for fwuff/IB
  calls and 17--24 ms for calls to the opposite dwagon socket, but this is not
  an equal-work transport comparison. In the cold layout fwuff owned only
  0.35%/0.01% of stage-0/stage-1 decode calls while the opposite socket owned
  41.77%/42.66%; most fwuff requests therefore contained no selected expert
  work. Fresh-connection standalone gates with six selected experts measured
  32.31 ms on the stage-0 socket endpoint and 53.81 ms on fwuff, but include
  connection setup and are not served-path latency evidence either. The
  balanced run, with roughly equal call shares, is the controlled transport
  comparison. Hard-binding the already capacity-limited main process would
  not remove either sidecar RPC and would restore the demonstrated Pro OOM
  failure mode.
- The matched random-8/output-64 probe measured 1,669.16 ms TTFT and
  499.41 ms TPOT (2.00 token/s). The longer output-256 probe measured
  1,455.19 ms TTFT, 434.14 ms TPOT (2.30 token/s), and 2.875 mean DSpark
  acceptance. Its approximate target-verification interval is
  `434.14 * 2.875 = 1,248.15` ms, materially worse than `tiered4`'s 946.00
  ms. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered7-bidirectional-cold-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered7-bidirectional-cold-random8-out256-seed42.jsonl`.
- The cold layout is rejected. It deliberately left fwuff almost idle in
  layers 0--47 while overusing the opposite-socket sidecar. The already-staged
  memory-neutral balanced layout assigns about 28% of first-48-layer decode
  calls to fwuff and about 28% to the opposite socket, so that controlled
  topology is next.

## 2026-07-26 19:23 EDT — balanced bidirectional topology rejected

- The memory-neutral balanced planner assigned projected stage-0 calls
  11.81% GPU, 27.97% opposite NUMA, 27.75% fwuff, and 32.47% local; stage 1
  was 14.73%/28.29%/28.42%/28.56%. Plan SHA-256 values on both hosts were
  `7a49045ed98389452bcf23e5b1d037f2204df3f0589768f0220dfef671b3659a`
  for NUMA 0,
  `fcf802b1608fe773b07b0c33d5de934ae8bb01b4e328aa017fef4c335535ac89`
  for NUMA 1, and
  `75f1db2af7f6585662773c7877740af734393ae7f73775c1da3624a3a1c0d463`
  for fwuff.
- Before the main load, all five strict sidecars passed exact BF16 comparison
  against direct AMX/AVX-512 computation on real Pro experts, including the
  fwuff tier over InfiniBand. `tiered8` then loaded in
  519.36/508.35/449.64 seconds and loaded DSpark in 25.10 seconds. It
  allocated the full 1,048,576-token cache and passed exact arithmetic plus
  Paris/Seine semantic gates. All processes mapped only their canonical
  native-MXFP4 extension.
- NUMA evidence remained stable: PP0 held
  209,683.52/217,921.43 MiB (96.22%) on selected NUMA 1 and PP1 held
  236,742.02/237,061.63 MiB (99.87%) on selected NUMA 0. The four strict
  dwagon sidecars were 99.76%, 97.99%, 99.57%, and 97.85% local to their
  assigned socket; fwuff's rank and sidecar remained wholly on its sole
  node. The roughly 3.8% PP0 main-rank spill is again a bounded capacity
  valve. Sidecar expert weights are local to their worker socket and only
  compact activations/results traverse the RPC, so hard main-process
  `membind` would not remove this communication.
- The matched random-8/output-64 probe measured 1,588.22 ms TTFT and
  670.36 ms TPOT (1.49 token/s). The output-256 probe measured 1,441.40 ms
  TTFT, 570.68 ms TPOT (1.75 token/s), and 2.225 mean DSpark acceptance. Its
  approximate target-verification interval is
  `570.68 * 2.225 = 1,269.76` ms, effectively unchanged from the rejected
  cold layout's 1,248.15 ms and 34% worse than `tiered4`'s 946.00 ms.
  Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered8-bidirectional-balanced-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered8-bidirectional-balanced-random8-out256-seed42.jsonl`.
- Balanced placement is rejected. Splitting each stage across more RPC tiers
  added sparse-routing and join overhead without reducing the real
  whole-model verification interval. The next controlled experiment returns
  to the proven `tiered4` overlap topology and changes only the native
  multi-token AMX dependency graph.

## 2026-07-26 20:05 EDT — fine-grained AMX graph and measured NUMA cost

- The native MXFP4 kernel now uses a reusable fine-grained expert dependency
  graph for one-token decode and the six-token DSpark target verifier. Tasks
  are submitted stage-major: all gate/up producers precede their per-expert
  activation waiters, then pack and down tasks retain only their true
  dependencies. This overlaps expert stage tails without the old global
  gate/up, activation, and down barriers and avoids starving the worker pool.
  The source header SHA-256 is
  `b48cce75679bb81b2f1db948e0eaba49b3b9352ef8ac05efce8cfe790b927612`
  on both machines. The host-native extension hashes are
  `e376394a3d586cf7a55e730ac19fcbcb7b9b6ef7556dd752a051838daaf59c03`
  on dwagon and
  `a9654214395a312e0fddc9706c5639470c62c6fa2722d90017c627ff8c2bd30b`
  on fwuff. Recoverable pre-change binaries remain outside import paths under
  each runtime's `backups/` directory.
- Real layer-10 native-MXFP4 old/new comparison remained bit-exact. On dwagon,
  a mixed 64-expert six-token call improved from 10.688 to 10.340 ms median
  (3.3%), while one-token decode improved from 2.943 to 1.934 ms (34%). On
  fwuff the corresponding medians improved 4.990 to 4.781 ms (4.2%) and
  1.285 to 0.861 ms (33%). Exact hashes were unchanged between old and new:
  `9f9ba...` for the mixed-six oracle and `3ea1...` for one token. Three
  compact sidecar gates over real layer-10/layer-30 weights were also exact.
- `tiered9` changed only this kernel relative to the proven overlap topology.
  It allocated the full 1,048,576-token cache and passed exact `4` plus the
  Paris/Seine semantic gate. The 256-output runs were highly route/acceptance
  dependent: TPOT was 286.96, 193.36, 179.96, and 332.10 ms with mean
  acceptance 2.18, 2.57, 3.20, and 3.26. The best observed single-stream
  decode was 5.56 token/s, but another repeat fell to 3.01 token/s; this is
  not accepted as a stable service result. A captured OpenAI-chat run
  measured 298.69 ms TPOT and 3.10 mean server-side DSpark acceptance.
  Artifacts are the
  `/var/lib/exo/benchmarks/dsv4-pro-tiered9-finegraph-random8-out256-seed42*.jsonl`
  family.
- Remote-sidecar timing now records `active_routes` and `active_experts` at a
  low sampling rate. It confirms that work, not endpoint label, dominates
  the earlier apparent latency difference: six active expert routes are
  commonly 1.4--3.4 ms, while 30--36 routes over 16--35 distinct experts can
  take 10--38 ms. The previous 2--3 ms fwuff versus 17--24 ms opposite-socket
  observation compared unequal sparse workloads and must not be used as a
  transport conclusion.
- NUMA strictness was revisited with Intel PCM commit
  `c284f1412ec21cdb640f1a358418c1eb10b032d5`, built locally without changing
  the runtime. PP0's actual scheduler has every one of its 222 threads on
  NUMA 1 and 233,015.78/242,680.58 MiB resident there (96.02%); PP1 has every
  thread on NUMA 0 and 263,558.41/263,939.23 MiB there (99.86%). PP0's
  9,484.71 MiB anonymous spill is distributed across the 56 MiB expert
  buffers; it is not merely reclaimable file cache.
- Process-attributed Emerald Rapids counters over 26 active decode seconds
  measured 6,123,206,740 local and 259,833,998 remote DRAM accesses for PP0:
  4.0707% remote, approximately 15.07 GB/s local and 0.64 GB/s remote if each
  counted cache line is 64 bytes. The remote stream is tiny relative to the
  three 44.8 GB/s full-speed UPI links, and its share tracks page residency.
  It is a real but bounded single-digit-percent cost, not the cause of the
  fourfold remaining decode gap. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered9-pp0-pcm-numa-fixed64.csv`.
- Hard main-process `membind` remains unsafe for the current full-local
  topology because the selected nodes have only 8--9 GiB free while PP0
  would need to repatriate about 9.7 GiB plus allocation headroom; it has
  already OOMed on full load. The launch script now makes strict main binding
  opt-in with `DSV4_STRICT_MAIN_NUMA=1`. The next hybrid plan removes enough
  additional PP0/PP1 resident experts to test that policy honestly. CPU
  affinity, native worker allocation, and every sidecar remain strictly
  node-bound in either mode.

## 2026-07-26 20:27 EDT — strict tier10 exposes fwuff draft headroom limit

- The original profile-guided GPU mask was reconstructed exactly at
  `d1d03135f845a64110cdd467c97a4ca9dca30ae39210486e755c27ebd7b19b8f`.
  A one-PP2-RPC hybrid plan then assigned 64 experts/layer to the opposite
  dwagon socket, 36 first-48-layer experts to fwuff, and 64 PP2 experts to
  dwagon NUMA 0. Plan hashes were `9d41943...`, `a4268e4...`, and
  `8c5c8af...`; ownership was disjoint from the explicit GPU mask.
- All four compact sidecars passed direct native-MXFP4 equality before full
  load. Stage-0 layer 10 produced `cceb6a...`, stage-1 layer 30
  `2d5549...`, PP2 layer 52 `e9c907...`, and fwuff layer 10 `37f54f...`.
  The PP2 result was repeated from fwuff against the IB-facing
  `10.44.0.1:29564` endpoint and remained exactly `e9c907...`. Oracle logs
  are under
  `/var/lib/exo/benchmarks/dsv4-pro-tiered10-oracle-*.log`.
- `tiered10` enabled process-wide strict `membind` for both dwagon ranks.
  PP0 and PP1 completed native target loading in 509.78 and 487.66 seconds
  with no OOM events. The native allocator's message
  `alloc 0 from other numa` is not a fallback: source inspection shows it
  compares logical TP partition index 0 with the current physical CPU node.
  Its actual allocation is `posix_memalign`, which inherits the strict
  process policy. Final residency still requires a coherent rerun because
  pipeline teardown followed the remote failure.
- fwuff did not fit the 36-expert sidecar plus PP2 and the mandatory complete
  three-layer DSpark draft. The target stage reached draft initialization,
  then the kernel OOM-killed the scheduler while draft layer 2 was loading.
  The killed scheduler had 185,026,828 KiB anonymous RSS; the sidecar held
  about 51.1 GiB private plus normal host overhead on a 247 GiB machine.
  The other ranks then exited at their collective rather than serving a
  partial cluster. This run is rejected and no semantic or performance
  result is accepted from it.
- A reproducible plan-subsetting tool now preserves existing ownership tiers
  while reducing one remote tier by recorded activation frequency.
  `fwuff-balanced20-hot.pt` keeps the hottest 20 of the prior 36 experts per
  layer, has 960 slots, is overlap-free, and hashes to
  `8a5ef5ba85b195aff9ba2dbc9bfd0926cb7045468313f22679f7881a187da3a4`.
  It should free about 22.6 GiB on fwuff. Despite the width reduction it
  retains 23.43% of stage-0 and 30.49% of stage-1 decode calls; opposite
  dwagon remains at 28.80%/32.01%, and local shares become 38.45%/33.92%.
  The 64-expert single PP2 tier remains 44.60% of PP2 calls. This
  memory-safe plan is loading for `tiered11`.

## 2026-07-26 21:18 EDT — memory-safe tier11 plans and exact remote experts

- The first expanded PP2 plan used `-1` padding outside the recorded active
  layers. The runtime correctly rejected it because native plan rows must
  contain valid global expert IDs even when an active-range mask makes those
  rows unreachable. The corrected
  `dwagon-numa0-stage2-80.pt` uses valid placeholders outside layers 48--60;
  its active rows remain the original 64 balanced experts plus 16 cold
  profile experts. It hashes to
  `974f84329a93888657919da508120f0eb53f65f2308f30f7247829bfc6ad4dec`
  on both hosts.
- The 20-expert fwuff sidecar held about 32.97 GiB private and the 80-expert
  stage-2 dwagon sidecar about 34.4 GiB private. This leaves enough fwuff
  memory for the non-negotiable full DSpark draft, unlike tier10's
  36-expert plan.
- Real checkpoint oracles were exact after both adjustments. A fwuff
  layer-10 sidecar result hashes to `8c780cb...`; a layer-52 result routed
  over InfiniBand to the expanded stage-2 endpoint hashes to `915a4a...`.
  These are direct BF16 output equalities against a separately constructed
  native-MXFP4 expert, not language-level plausibility checks.

## 2026-07-26 22:24 EDT — tier12 coherent, strict, and too slow

- The complete memory-safe topology loaded PP0/PP1/PP2 targets in
  522.48/578.93/405.28 seconds, retained the full three-layer DSpark draft,
  and allocated the complete 1,048,576-token cache. Arithmetic returned
  exactly `4`; a separate factual gate correctly identified Paris and the
  Seine. The served model remained native MXFP4 throughout.
- Strict main-rank binding is now viable because remote ownership reduced
  socket demand. PP0 had 222,572.50 of 230,247.20 MiB on selected NUMA 1
  (96.67%); PP1 had 249,976 of 250,345 MiB on selected NUMA 0 (99.85%).
  CPU affinity and every native worker remained socket-local. Intel PCM over
  a fixed 64-token request attributed 5,225,815,086 local and 183,833,970
  remote accesses to PP0: 3.3983% remote, approximately 12.39 GB/s local and
  0.44 GB/s remote. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-strict-pp0-pcm-numa-fixed64.csv`.
- Compared with tier9's local-first 4.0707%, strict binding removes only
  0.6724 percentage points of measured remote accesses. This is a useful
  deterministic placement policy when the topology fits, but not a material
  performance breakthrough. The full-local topology must retain
  `--localalloc` because hard binding already OOMed; its roughly 4% remote
  traffic is not an important bottleneck on this machine.
- Official random-8/output-64 measured 1,346.10 ms TTFT and 420.74 ms TPOT
  (2.38 token/s), with mean DSpark acceptance 2.58. Output-256 measured
  1,369.45 ms TTFT, 516.376 ms TPOT (1.94 token/s), mean acceptance 2.29375,
  and an approximate target interval of 1,184 ms. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-strict-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-strict-random8-out256-seed42.jsonl`.
  The topology is coherent but rejected on performance.

## 2026-07-26 22:47 EDT — repeatability gate clarified

- Three identical temperature-zero, seed-42 64-token chats diverged beginning
  at the fifth generated token. This server has deterministic inference
  disabled and DSpark proposal scheduling enabled, so identical end-to-end
  text is not a valid coherence requirement. The captures remain at
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-determinism-fixed64-run{1,2,3}.json`.
- The native oracle now records every repeated CPU-output hash. The active
  fine-grained kernel was bitwise stable for 20 repeated one-token and
  six-token calls, and a six-route one-token case was exact for 100 repeats.
  Representative hashes are `f19850...`, `1a33c547...`, and `a46e4b3...`.
  Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-native-amx-finegraph-repeat20-q1.json`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-native-amx-finegraph-repeat20-q6.json`.
  Combined with exact sidecar and GPU oracles, there is no evidence of a
  native dependency-graph race.

## 2026-07-26 23:42 EDT — profile-driven worker-pool breakthrough

- Linux perf 6.18.35 was built locally from official kernel source because
  the installed tool did not match the running kernel. A request-only
  fixed-64 cycles profile showed that the three dwagon expert sidecars
  consumed 66.1% of aggregate sampled cycles and the two serving ranks
  33.9%. Sidecar shares were 27.149%, 22.698%, and 16.224%; rank shares were
  18.586% and 15.344%. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-dwagon-cycles-requestonly-fixed64.perf.data`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-dwagon-cycles-requestonly-symbols.report`.
- Per-process attribution exposed two avoidable costs. The native worker pool
  used a hard-coded 50 ms `clock_gettime` spin after every job; VDSO samples
  alone were 11.91%/14.62%/7.62% in the three sidecars and 9.44%/10.93% in
  the ranks. Tiny PyTorch route-remapping operations also invoked libgomp in
  each sidecar, contributing 7.68%/8.91%/6.45%.
- `worker_pool.cpp` now accepts `KT_WORKER_SPIN_US`, preserving 50 ms as the
  upstream-compatible default. Production launchers select 1 ms; zero was
  rejected by isolated A/B. Sidecars also set PyTorch intra-op and inter-op
  thread counts to one. The worker-pool source hashes to
  `cb7e1645ec0040949ab7c5fb60549965f02db689d96588957db947f859744971`.
  Rebuilt extension hashes are
  `866782bc90de02ab601b220dad3348260857f9386b84a3820a675fd1479a9c93`
  on dwagon and
  `5dc248c5ccf75776b289800f66572a584367b7f87c6ceccaaa84abbe57ef6772`
  on fwuff; recoverable pre-change binaries remain under each runtime's
  `backups/` directory.
- Isolated native A/B remained bitwise exact. On dwagon, six-token median
  improved from 6.260 to 5.892 ms with the 1 ms spin; the one-token median
  was 1.847 ms versus 1.773 ms at 50 ms, an accepted small active-call cost
  because pipeline stages are idle for most of a serial decode step. A zero
  spin regressed both medians. fwuff's six-token median with the new binary
  was 3.084 ms and retained hash `1a33c547...`.
- All four sidecars were restarted without reloading the main ranks. Live
  real-weight oracles were then bit-for-bit exact for PP0 layer 10, PP1
  layer 30, PP2 layer 52, and fwuff layer 10; even the InfiniBand result had
  zero BF16 difference. Artifacts are
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-spin1ms-sidecar-*-oracle.json`.
  Their hashes are respectively `7a41cc...`, `bb9deb...`, `a346eb...`, and
  `1d4a65...`.
- Existing rank-side TCP sockets could not survive sidecar replacement. The
  first renewal request produced the expected EOF, and current SGLang treats
  that remote-expert EOF as rank-fatal. The cluster is therefore performing
  a required full reload with the 1 ms worker-spin kernel active in sidecars
  and ranks. This next matched fixed-64 profile will determine whether the
  profile-guided change rescues tier12 or whether deployment returns to the
  faster proven tier4 topology.

## 2026-07-27 00:02 EDT — worker-pool gain accepted; tier12 topology rejected

- The full service reloaded with the 1 ms worker-spin kernel active in every
  rank and sidecar. Target load times were 575.45 seconds for PP1 and 410.50
  seconds for PP2; PP0 completed at the same final barrier. All stages
  allocated the full 1,048,576-token pools and retained the complete DSpark
  draft. Arithmetic returned exactly `42`, and a separate gate returned
  “Paris is the capital, and the Seine flows through it.”
- A second request-only cycles profile validates the kernel hypothesis.
  Across the same five dwagon processes and the same 40-second sampled
  window, attributed cycles fell from 1.974 trillion to 0.801 trillion
  (59.4%). The three sidecars fell from 66.1% to 29.8% of total cycles.
  Their aggregate VDSO cycles fell about 88%, rank VDSO cycles about 82%,
  and libgomp disappeared completely from the sidecars. New evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-spin1ms-dwagon-cycles-requestonly-fixed64.perf.data`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-spin1ms-dwagon-cycles-requestonly-tgid-dso.report`.
- The official random-8/output-64 result improved from 420.74 to 278.69 ms
  TPOT (33.8%), from 1,346.10 to 765.06 ms TTFT, and from 2.30 to 3.49 output
  token/s. Mean DSpark acceptance was 2.675. The longer output-256 result
  improved from 516.376 to 286.490 ms TPOT (44.5%) and from 1,369.45 to
  681.98 ms TTFT, with 3.47 output token/s and 2.2125 acceptance. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-spin1ms-random8-out64-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered12-spin1ms-random8-out256-seed42.jsonl`.
- This is an accepted authentic kernel/runtime improvement: it removes
  measured busy-wait and OpenMP overhead rather than changing benchmark
  arguments. Tier12 itself remains rejected. Its 286.49 ms long-run TPOT is
  still 50.5% slower than tier4's 190.34 ms because splitting each stage
  across three dwagon sidecars fragments sparse expert work and adds joins.
  The production topology is returning to the proven tier4 three-way overlap
  layout while retaining the new fine-grained AMX graph, configurable worker
  sleep, one-thread sidecar remapping, and corrected compact-shard semantics.

## 2026-07-27 00:28 EDT — tier13 restores the faster overlap topology

- The restored topology is live and coherent at `127.0.0.1:30000`. PP0 and
  PP1 run on dwagon, PP2 and the cold-expert endpoint run on fwuff over
  InfiniBand. PP0/PP1 each overlap their local native-MXFP4 partition with
  one strict opposite-socket 64-expert sidecar and the strict fwuff
  four-cold-expert sidecar; PP2 keeps its remaining experts local. GPU hot
  ownership remains 7/4/12 experts and the complete three-layer DSpark draft
  is present. Target loads completed in 613.95/547.81/451.19 seconds with
  the full 1,048,576-token pools. Arithmetic returned exactly `42`; an
  independent factual gate correctly returned Paris and the Seine.
- The exact plans have not drifted. The merged route profile hashes to
  `dc7dbeaa24434feee385bda6dc5a9535e0cd467dd8872e230f1d40dbb70267`;
  the opposite-socket and fwuff-cold plans hash to
  `242224986957a307dc00f995a9bfe8df888ef77cd4ba818ada707aa242f169d6`
  and
  `466a315c979091a97d5338c0739fe171682aa09ad71bbbbfa8cdfffbfe129e0a`.
  Direct real-checkpoint sidecar oracles were bit-exact for PP0 layer 10,
  PP1 layer 30, and the fwuff layer-10 cold subset. Their output hashes are
  `f26c3f93...`, `08d75f82...`, and `e47098e1...`; evidence is under
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-sidecar-*-oracle.json`.
- A fresh residency audit confirms that local-first remains the right NUMA
  policy for the large main ranks. PP0 has 59,559,377 of 62,118,051 mapped
  pages (95.88%) on selected NUMA 1; PP1 has 67,472,036 of 67,566,950
  pages (99.86%) on selected NUMA 0. The strict opposite-socket sidecars
  are 99.71% local on NUMA 0 and 97.91% local on NUMA 1. fwuff has one NUMA
  node and both its rank and sidecar are hard-bound there. Together with the
  fixed-request PCM result (4.07% remote under local-first versus 3.40%
  under strict), this closes the loosened-NUMA question: hard binding is
  retained wherever the compact topology makes it memory-safe, but the
  main ranks retain `--localalloc`. The residual cross-socket access is
  small and hard-binding the full-local topology already caused OOM.
- Random-8/output-64 produced 712.69 ms TTFT and 121.619 ms TPOT, or 7.62
  output token/s. It was not stable at that rate: two output-256 runs
  measured 249.867 and 180.597 ms TPOT (3.976 and 5.479 token/s) with
  mean DSpark acceptance 2.575 and 2.825. The short result therefore is not
  promoted as sustained performance. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-spin1ms-finegraph-random8-out64-seed42.jsonl`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-spin1ms-finegraph-random8-out256-seed42.jsonl`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-spin1ms-finegraph-random8-out256-seed42-repeat2.jsonl`.
- A 76.933-second cycles profile of the faster output-256 repeat attributes
  78.38% of aggregate dwagon cycles to the native KTransformers extension
  and 16.69% to the worker-pool VDSO path. PP0 and PP1 ranks account for
  38.031% and 39.986% of total cycles; the two sidecars account for 12.619%
  and 9.365%. fwuff PP2 averaged only 478% CPU and the cold sidecar 33.3%,
  demonstrating that the serial pipeline leaves substantial fwuff compute
  idle. The remaining stripped-extension hot addresses disassemble to the
  AVX-512 native-MXFP4 matvec loop: nibble expansion and `vpermw`, BF16 dot
  products, and UE8M0 scaling. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-dwagon-cycles-random8-out256-repeat2.perf.data`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-dwagon-cycles-random8-out256-repeat2-tgid-dso.report`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-dwagon-cycles-random8-out256-repeat2-symbols.report`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-fwuff-pidstat-random8-out256-repeat2.log`.
- Tier13 remains available as the coherent deployment baseline while the
  next optimization is built and proven against an isolated exact native
  oracle. No experimental binary will replace the live mapped extension
  without a repeatable core-pinned gain.

## 2026-07-27 00:51 EDT — decode task granularity breakthrough

- A candidate that sampled the worker-pool clock only every 64 idle polls was
  built in isolation and rejected without touching the live binary. All
  outputs were bit-exact, but paired six-token/500-repeat medians regressed
  from 5.572 to 5.926 ms and from 5.394 to 5.735 ms. The source was restored
  exactly; `worker_pool.cpp` again hashes to
  `cb7e1645ec0040949ab7c5fb60549965f02db689d96588957db947f859744971`.
- Native six-token perf evidence shows that the next bottleneck is within the
  MXFP4 matvec rather than routing branches: 510.659 billion instructions,
  352.197 billion cycles (IPC 1.45), 0.22% branch misses, and 1.67% cache
  misses. Topdown attributed 83.5% to backend bound and 58.4% to memory bound.
  Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-native-q6-repeat500-perf-stat.txt`,
  `/var/lib/exo/benchmarks/dsv4-pro-native-q6-repeat500-topdown-stat.txt`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-native-q6-repeat500-memorylevel-stat.txt`.
- A 50 ms CPU timeline around a fresh 64-token request confirms a serial
  target-verification wave. PP0 and PP1 were each active for approximately
  280 ms, followed by approximately 110 ms on PP2; the opposite-socket
  sidecars generally finished about 50 ms before their owning rank. This
  makes pipeline imbalance and absent candidate-token wavefronting real
  optimization targets, while fwuff's low mean utilization is not caused by
  InfiniBand saturation. The request itself was a slow route-variance sample
  at 317.07 ms TPOT and acceptance 2.83. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-timeline-random8-out64-seed42.jsonl`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-out64-cpu-timeline-dwagon.log`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered13-out64-cpu-timeline-fwuff.log`.
- SMT workers were decisively rejected: 104 workers regressed paired
  six-token medians by 9–17% versus 52 physical-core workers. In contrast,
  using all 56 physical cores in a dwagon socket improved paired medians
  from 5.442 to 5.189 ms and from 5.803 to 5.128 ms. The launchers now
  default dwagon ranks and sidecars to 56 physical workers; fwuff remains at
  its 60 physical workers.
- The native MXFP4 output-task width was reduced from 256 to 128. For the
  7,168-wide down projection this exposes 56 independent tasks instead of
  28, exactly matching one dwagon socket, while preserving the packed E2M1
  weights and UE8M0 scales unchanged. Paired dwagon six-token/500-repeat
  medians improved 3.51% and 5.36%; paired one-token/1,000-repeat medians
  improved 1.90% and 2.63%. Every repeat retained exact hashes
  `cace52c9...` and `08d410e3...`.
- The change was separately compiled for fwuff with native AMX, AVX512
  BF16/VBMI/VNNI, CUDA SM86, and Python 3.12. Paired six-token medians
  improved from 6.027 to 5.936 ms and from 6.044 to 5.986 ms. Paired
  one-token medians improved from 0.388 to 0.326 ms and from 0.446 to
  0.326 ms. The output hashes remained identical within every pair
  (`1a33c547...` and `f1985020...`). Candidate extension hashes are
  `606b57e06d1ff57fa3ff5cd1e23915576aabaf1ebac58b6ae7047a49ac72bc39`
  on dwagon and
  `c75da940d31b24ade56d5befdebc513e7a027baa7bf261a561609a641ac548df`
  on fwuff.
- This is accepted as an authentic kernel scheduling improvement. The
  current Tier13 processes still map the previous binaries and remain
  healthy; the accepted binaries and 56-core launch policy will enter
  service together in the next controlled full reload.

## 2026-07-27 01:24 EDT — tier14 validates N128 and all physical cores

- The accepted `N_BLOCK=128` binaries were installed with recoverable
  hash-named backups. The active hashes are
  `606b57e06d1ff57fa3ff5cd1e23915576aabaf1ebac58b6ae7047a49ac72bc39`
  on dwagon and
  `c75da940d31b24ade56d5befdebc513e7a027baa7bf261a561609a641ac548df`
  on fwuff. The pre-change `866782bc...` and `5dc248c5...` binaries remain
  in each runtime's `backups/` directory. The source header is restored to
  the accepted state and hashes to `77d431d6...`.
- Tier14 reloaded the proven three-way overlap topology with 56 physical
  native workers on each dwagon rank/sidecar and 60 on fwuff. Sidecars
  remained hard NUMA-bound; the two large dwagon ranks retained local-first
  allocation. Before rank startup, real-checkpoint oracles were bit-exact:
  PP0 layer 10 hashes matched at `7ccd122c...`, PP1 layer 30 at
  `69a8ec09...`, and fwuff layer 10 at `1ae8caf0...`, all with zero BF16
  difference.
- Target load times were 564.02 seconds on PP0, 633.66 seconds on PP1, and
  478.87 seconds on PP2. All ranks allocated the full 1,048,576-token pools;
  fwuff initialized the complete three-layer DSpark draft with gamma 5 and
  six target-verify tokens. The service is live at `127.0.0.1:30000`.
  Arithmetic returned exactly `42`; a separate semantic gate returned that
  Paris is France's capital and the Seine flows through it.
- Two official random-8/output-256, concurrency-one served runs measured
  626.44/610.63 ms TTFT and 120.163/100.100 ms TPOT, or 8.184/9.789 output
  token/s. Mean DSpark acceptance was 3.325 and 4.317, so both results are
  retained rather than promoting only the faster route sample. Compared
  with Tier13's best 180.597 ms TPOT, the accepted task-width and physical
  core changes improve sustained TPOT by 33.5–44.6%. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered14-nblock128-cores56-random8-out256-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered14-nblock128-cores56-random8-out256-seed42-repeat2.jsonl`.
- Tight `/health` polling is not free in this SGLang build: each readiness
  check is accompanied by a synthetic 256-token prefill/liveness pass before
  the HTTP 200 is returned. This explained the apparent idle-prefill stream;
  there was no stale benchmark client. Monitoring now avoids aggressive
  health polling.
- The native AMX crossover was measured rather than guessed. For one routed
  expert, counts 1–3 remain on the AVX512 matvec path. AMX is strongly worse
  at exactly four tokens, then is near break-even or modestly faster at
  five through seven. A mixed seven-expert/six-token route is more
  representative of correlated DSpark candidates: lowering the threshold
  from eight to five improved paired medians from 7.808 to 4.490 ms and
  from 7.642 to 4.657 ms while leaving four-token experts on AVX512.
  Threshold one is rejected; production launchers now select five.
- AMX arithmetic passed the independent portable SM86 native-MXFP4 oracle:
  relative mean difference was `2.62e-6`, maximum BF16 difference 0.03125,
  and cosine similarity approximately 1.000005. The checkpoint's packed
  E2M1 weights and UE8M0 scales remained native and unchanged.
- A separate `N_BLOCK=96` binary was built and rejected. It was neutral on
  one-token decode, regressed paired six-token AVX512 medians by about 10%,
  and changed AMX medians by less than 2%. The live binary was never touched,
  and source was restored to 128.
- Tier14 remains healthy while the accepted threshold-five AMX/AVX512 mix is
  staged for a controlled Tier15 reload. This next served A/B will determine
  whether correlated real routes capture the large isolated mixed-route
  gain.

## 2026-07-27 01:32 EDT — tier15 AMX-five coherence staging

- Tier14 was stopped cleanly for the controlled Tier15 reload. Tier15 keeps
  the accepted N128 binaries, 56/56/60 physical-core worker layout, strict
  binding for compact sidecars, and local-first allocation for the two large
  dwagon ranks. Its only intended arithmetic change is the measured
  `KT_MXFP4_AMX_MIN_EXPERT_TOKENS=5` crossover.
- Both dwagon sidecars loaded successfully. Six-token fixed-route real-weight
  oracles explicitly entered the AMX path and were bit-exact with the live
  endpoint: PP0 layer 10 hash
  `1cf7c8f1e20f682ee7e4d18d41fbf4e9dbea3ec58debca8695db2365a31b6615`
  and PP1 layer 30 hash
  `9e791b381d8f24f47bdc64426dddf7f0d9fdf91300b1d3dae8cce182660504c5`.
  Their independent portable-SM86 comparisons retained cosine similarity
  above 0.999998 and relative mean difference about 0.26%. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-sidecar-stage0-layer10-amx5-oracle.json`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-sidecar-stage1-layer30-amx5-oracle.json`.
- The first InfiniBand oracle caught deployment drift before any rank was
  launched: fwuff still mapped the prior launcher and reported threshold
  eight in `/proc/.../environ`. Its six-token result therefore retained the
  old AVX-era `1ae8caf0...` hash and differed slightly from the AMX reference.
  The accepted launcher was synced to fwuff (matching SHA-256
  `8e886b3179c3181c0f795761cf035881d1de447102c5c42b6ff881f0e88632bf`),
  and only the unconnected cold-four sidecar was restarted. The replacement
  process now reports threshold five; its post-reload oracle is pending.
- The reloaded fwuff sidecar explicitly entered AMX and was bit-exact with a
  same-host 60-core reference at hash
  `0216a2e393eeae11c81fd80fc1cbfb8514c654b91073387f3bda643700ece294`;
  InfiniBand round-trip was 8.24 ms. The pre-sync mismatch is retained as
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-sidecar-fwuff-layer10-pre-sync-drift.json`;
  the accepted exact oracle is
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-sidecar-fwuff-layer10-amx5-oracle.json`.
  fwuff's rank launcher was also synced before launch, correcting its stale
  threshold-eight/52-dwagon-worker defaults. Both host copies now hash to
  `71ea6fb669a821dff4b22d47b78351deb3a7e673d33a661b31de1202b6aadce2`.
- Tier15 target loads completed in 646.02/657.90/492.06 seconds. Every rank
  allocated its full 1,048,576-token pool and fwuff loaded the complete
  three-layer DSpark draft. Arithmetic returned exactly `42`; a separate
  semantic gate returned Paris as France's capital and the Seine as its
  river.
- Two official random-8/output-256 runs measured 592.45/604.81 ms TTFT and
  100.33/104.49 ms TPOT, or 9.77/9.39 output token/s, with mean DSpark
  acceptance 4.88/4.76. These are effectively tied with Tier14's best
  100.10 ms TPOT, not a served breakthrough. The lower threshold is
  numerically valid and useful in isolation, but it is not promoted on the
  strength of route variance. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-amx5-nblock128-cores56-random8-out256-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered15-amx5-nblock128-cores56-random8-out256-seed42-repeat2.jsonl`.

## 2026-07-27 01:54 EDT — 16-row AMX DSpark-tail kernel accepted

- Profiling the threshold-five path exposed a concrete AMX inefficiency:
  an expert receiving five or six DSpark verify rows was padded to 32 rows
  and issued all four tile dot products, although rows 16 through 31 could
  never contribute to output. The native kernel now keeps the identical
  K-group/FP32 accumulation order for live rows while omitting the second
  activation tile and its two dot products whenever `valid_m <= 16`.
  Checkpoint E2M1 nibbles and UE8M0 scales remain untouched.
- The change was compiled only into isolated candidates first. On dwagon,
  paired one-expert/six-token 500-repeat medians improved from 0.655 to
  0.539 ms and from 0.646 to 0.525 ms (17.8–18.8%). A representative
  mixed-seven-expert/six-token route improved from 3.013 to 2.801 ms and
  from 3.072 to 2.824 ms (7.0–8.1%). All repeats were bit-identical with
  hashes `3e010fac...` and `fa90ebfa...`.
- The independently compiled fwuff candidate improved paired six-token
  medians from 0.422 to 0.314 ms and from 0.420 to 0.316 ms
  (24.8–25.6%), again bit-identical. Candidate extension hashes are
  `42a289085f4972f47cbbf946e594463e2708ddc22aca400935b257ebff517bc2`
  on dwagon and
  `adf4d853670154b98be734090eddb105e8e6b456c22aefb843b49a4c066ca04d`
  on fwuff. The shared source header hashes to
  `66caf78ced36f31ce09f1f6796d9377646ab4eb73b2a154f0179c0f23520c832`.

## 2026-07-27 02:04 EDT — tier16 live-binary coherence and full load

- Tier15 was stopped cleanly and both accepted small-M native extensions were
  installed atomically with the prior binaries retained under each runtime's
  `backups/` directory. Tier16 holds the Tier15 placement, arguments, and
  AMX-five policy constant so its served A/B isolates only the 16-row kernel.
- All three real-checkpoint sidecars passed the post-install gate. PP0 layer
  10 and PP1 layer 30 remained bit-exact at hashes `1cf7c8f1...` and
  `9e791b38...`; their portable SM86 comparisons retained cosine similarity
  above 0.999998 and relative mean difference about 0.26%. fwuff remained
  bit-exact with its same-host 60-core reference at `0216a2e3...`, with a
  6.39 ms InfiniBand request round trip. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered16-smallm16-sidecar-stage0-layer10-oracle.json`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered16-smallm16-sidecar-stage1-layer30-oracle.json`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered16-smallm16-sidecar-fwuff-layer10-oracle.json`.
- Tier16 is now loading all three 1,048,576-token target ranks. Compact
  sidecars remain hard-bound to their owning sockets; the large dwagon ranks
  retain CPU binding plus local-first allocation because full strict memory
  binding is not capacity-safe in this topology.
- A separate, inactive Tier17 plan now preserves fwuff's four least-used
  experts per layer and adds 24 profile-balanced experts. The planner can
  reuse the exact accepted GPU mask, and a cold-only reproduction matched
  every prior GPU, opposite-NUMA, and fwuff expert ID before the new mode was
  used. The 28-expert plans are disjoint, retain every original cold expert,
  and project route shares of 24.46%/25.66% on fwuff for PP0/PP1, versus
  0.008%/~0% in the prior topology. Opposite-NUMA/local projected shares
  become 27.49%/31.15% and 26.00%/25.97%. This is staged only; Tier16 remains
  the controlled kernel A/B. Plan hashes are `060fd586...` for fwuff and
  `3ac479e0...` for opposite NUMA.
- Tier16 target loads completed in 582.97/593.99/459.69 seconds; the complete
  DSpark draft loaded in 23.81 seconds. All ranks allocated the
  1,048,576-token pool. Arithmetic returned exactly `42`, and the independent
  factual gate returned Paris and the Seine.
- The two official random-8/output-256 samples were dominated by DSpark
  acceptance variance: TPOT was 260.47/164.58 ms with acceptance 2.38/2.75,
  or 3.82/6.01 output token/s. Their approximate target-verification
  intervals were 620/453 ms. The second is about 9% below Tier15's roughly
  497 ms interval, consistent with the exact isolated small-M gain, but the
  pair is not a served-speed promotion. Both unfavorable samples are retained:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered16-smallm16-random8-out256-seed42.jsonl`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered16-smallm16-random8-out256-seed42-repeat2.jsonl`.

## 2026-07-27 02:25 EDT — tier17 balanced fwuff execution

- Tier16 was stopped cleanly. Tier17 retains its native extension, GPU mask,
  AMX/AVX512 policy, 56/56/60 physical workers, full context allocation, and
  PP partition; only expert ownership changes to the validated fwuff-28 plan.
  fwuff loaded 28 experts across all first 48 layers while retaining every
  prior least-used expert. Its projected final headroom is approximately
  28--31 GiB, so strict one-node binding remains capacity-safe.
- Real-checkpoint six-token oracles passed before rank startup. PP0 layer 10
  was bit-exact at `3ef6372e...`, PP1 layer 30 at `a9b774ec...`, and fwuff
  layer 10 at `0cac8b86...`; every sidecar difference was zero. The two
  independent SM86 comparisons retained cosine similarity at or above
  0.999998 and relative mean difference about 0.26%. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered17-sidecar-stage0-layer10-oracle.json`,
  `/var/lib/exo/benchmarks/dsv4-pro-tiered17-sidecar-stage1-layer30-oracle.json`,
  and
  `/var/lib/exo/benchmarks/dsv4-pro-tiered17-sidecar-fwuff28-layer10-oracle.json`.
- All three Tier17 target ranks are loading. A separate, uninstalled
  projection-specific task-width candidate also compiled successfully:
  64-row work exposes 48 tasks for the narrow 3,072-row gate/up matrices,
  while the 7,168-row down matrix retains the accepted 128-row/56-task
  schedule. Candidate hash is `9af62d10...`; Tier17 does not map it.
- Before the ranks became serviceable, readiness counts exposed an
  inclusive/exclusive launch-argument mistake in the newly created Tier17
  sidecars: `[0,22)`/`[23,47)` had omitted boundary layers 22 and 47. No
  Tier17 inference had run. The unconnected sidecars were stopped and are
  reloading with the correct `[0,23)`/`[23,48)`/`[0,48)` ranges. The launcher
  now names the parameter `LAYER_END_EXCLUSIVE`, and boundary-layer oracles
  are mandatory before service.
- Corrected readiness counts are exactly 23/25/48 shards. Boundary oracles
  are now bit-exact at PP0 layer 22 (`cb7c3046...`), PP1 layer 47
  (`dbf8872e...`), and fwuff layer 47 (`fb4d31d9...`), with zero sidecar
  difference. The two local boundary checks also retain SM86 cosine
  similarity above 0.999998. Artifacts use the
  `dsv4-pro-tiered17-sidecar-*-boundary-oracle.json` names.
- The 28-expert attempt then found the actual fwuff capacity edge while
  loading the complete DSpark draft: the PP2 unit peaked at 190.2 GiB and
  the strict sidecar used about 48 GiB, leaving insufficient kernel/system
  headroom on the 247 GiB host. The OOM killer terminated PP2 only; no
  inference ran, and the distributed group was discarded rather than reused.
- A capacity-safe replacement retains four cold plus 16 balanced experts per
  layer (20 total). It recovers about 13 GiB and projects fwuff route shares
  of 20.19%/23.87% on PP0/PP1; opposite/local shares are
  29.58%/33.33% and 26.89%/26.87%. The new fwuff/opposite plan hashes are
  `c009be49...` and `6044c69a...`. All cold experts are retained, all tiers
  are unique/disjoint, and the fwuff copy is hash-identical.
- Tier17c loaded exactly 23/25/48 sidecar shards. Live boundary oracles are
  bit-exact at PP0 layer 22 (`d10cd544...`), PP1 layer 47 (`dbf8872e...`),
  and fwuff layer 47 (`fb4d31d9...`), with zero endpoint difference and SM86
  cosine above 0.999998 where compared. Evidence:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered17c-fwuff20-*-boundary-oracle.json`.
- The Tier17c ranks are now loading. Because 16 additional remote experts per
  layer recover roughly 12--14 GiB per dwagon rank, hard main-process
  `membind` is restored for this capacity-safe topology. The service also
  sets `max_running_requests=4` against the same 1,048,576-token pool, making
  its allocation valid for either one 1M request or four 256K requests.

## 2026-07-27 02:58 EDT — Tier17c capacity result and Tier17d plan

- Tier17c completed all three native-MXFP4 target-stage loads under strict
  placement on dwagon. PP0 loaded layers 0--22 in 484.70 seconds and retained
  5.93 GiB GPU headroom; PP1 loaded layers 23--47 in 555.32 seconds and
  retained 6.17 GiB. PP2 loaded its target in 470.45 seconds with 8.97 GiB
  GPU headroom. This proves the tightened dwagon memory policy is viable for
  the fwuff-20 target topology.
- The topology was not viable after adding the mandatory complete DSpark
  draft on fwuff. During draft layer 2, the PP2 cgroup reached a 203.8 GiB
  peak while the independent fwuff-20 sidecar held roughly another 42 GiB;
  the 247 GiB host OOM-killed PP2. The other ranks then exited on the broken
  distributed group. No inference ran, and all Tier17c sidecars were stopped
  cleanly.
- Tier17d reduces fwuff ownership to the canonical four cold experts plus
  eight profile-balanced experts per layer. The new plan projects 13.4395%
  of total profiled routes on fwuff, 32.5172% on opposite NUMA, 34.2951%
  native-local, and preserves the exact 19.7482% GPU mask. Stage route shares
  are 11.7484%/14.9952% for fwuff. The 12-expert plan retains every canonical
  cold4 expert, has unique IDs, and is disjoint from both GPU and
  opposite-NUMA tiers on all 48 served layers.
- Tier17d plan hashes are
  `c00acb72de515e57cb3674b2a4f37495a50da68b603e7300744149b35f9a1a5d`
  for `fwuff-cold4-balanced8.pt` and
  `699ad93f0c70adf18840a5b88c5b993f6b6653e39ff1506cbe34e3ea0700e6a4`
  for `opposite-numa64.pt`. The fwuff copy is hash-identical. Correct
  `[0,23)`/`[23,48)`/`[0,48)` Tier17d sidecars are now loading; strict
  dwagon rank placement remains enabled for the next run.
- Tier17d sidecars reached exactly 23/25/48 ready shards. A first oracle pass
  deliberately exposed the difference between validator AVX-at-six
  (`threshold=8`) and the live AMX-at-six policy (`threshold=5`): the two
  valid paths differed by at most one or two BF16 ulps. Those mixed-policy
  artifacts are retained with `mixed-avx8-amx5` names. Under the exact live
  policy, all endpoints are bit-identical to the native reference at hashes
  `a8c21a77...`, `67a20762...`, and `89f54d4e...`; local SM86 comparisons
  retain cosine similarity above 0.999997. The canonical
  `dsv4-pro-tiered17d-fwuff12-*-boundary-oracle.json` artifacts hold the
  promotion results.
- Settled fwuff sidecar anonymous memory is 19.897 GiB. Its 214 GiB file
  accounting is reclaimable checkpoint page cache, corroborated by 215 GiB
  host-available memory and a 19.98 GiB process RSS. This saves about 22 GiB
  of resident sidecar memory versus fwuff20 and projects roughly 23 GiB
  headroom over the observed 203.8 GiB PP2-plus-draft peak.
- Tier17d ranks are loading on rendezvous `10.44.0.1:29569`. The run retains
  the full 1,048,576-token pool, four-running-request capacity, complete
  DSpark draft on PP2, accepted small-M binary, exact GPU mask, and strict
  main-process memory binding on both dwagon sockets.

## 2026-07-27 03:20 EDT — Tier17d coherent service result

- Tier17d completed target loads in 504.04/638.55/518.63 seconds. PP2 then
  loaded the complete three-layer DSpark draft in 27.38 seconds and allocated
  the full 1,048,576-token target pool with four-request partitioning. At its
  settled high-water point, fwuff held about 198 GiB rank RSS plus 20 GiB
  sidecar RSS and retained about 16 GiB host-available memory. The topology
  therefore resolves the fwuff-20 OOM while preserving meaningful remote
  expert work.
- Hard dwagon memory binding is active on every substantive mapping
  (`bind:1` on PP0 and `bind:0` on PP1). PP0 residency is 228,026.75 MiB on
  node 1 and 7,671.79 MiB on node 0, or 96.7451% selected-node residency.
  PP1 residency is 256,178.64 MiB on node 0 and 265.71 MiB on node 1, or
  99.8964%. PP0's small residual share is consistent with the earlier
  measured 3.40% remote-access share and is not a material bottleneck.
- End-to-end semantic gates passed exactly: arithmetic returned only `42`;
  the independent factual response identified Paris and the Seine. Artifacts
  are `dsv4-pro-tiered17d-fwuff12-semantic-42.json` and
  `dsv4-pro-tiered17d-fwuff12-semantic-paris-seine.json`.
- Official single-stream random-8/output-256 samples produced
  102.218/129.520 ms TPOT, 9.589/7.591 output token/s, 617.18/683.91 ms TTFT,
  and DSpark acceptance lengths 3.325/4.15. Approximate target-verification
  intervals were 340/537 ms. Tier17d is coherent and memory-safe, but this
  pair is not a speed promotion and remains below the 20 token/s objective.
  Artifacts are
  `dsv4-pro-tiered17d-fwuff12-random8-out256-seed42.jsonl` and its
  `repeat2` counterpart.
- Sampled sidecar timing during the pair averaged 5.35 ms locally and
  3.06 ms over InfiniBand; the latter confirms raw network service is not the
  primary regression. The next gates are a served 32K prefill baseline,
  followed by isolated A/B of the staged projection-width and AVX-tail
  kernel candidates.

## 2026-07-27 03:26 EDT — 32K workspace diagnosis and AVX-tail promotion gate

- The first exact 32,768-input/8-output Tier17d request exposed a GPU
  workspace limit rather than a host-capacity, InfiniBand, or NUMA failure.
  The full 1,048,576-token pool left PP0 with about 1.01 GiB free; a
  4,096-token prefill chunk then asked `_compute_q_b` for a 512 MiB
  `torch.empty_like(q)` allocation with only 22.94 MiB immediately
  available. Rank 0 correctly failed with CUDA OOM and the distributed group
  was discarded. The launcher now defaults both chunked prefill and maximum
  prefill tokens to 2,048, retaining the full 1M token pool and four-request
  capacity while halving that per-chunk workspace. The next live run will
  lower this to 1,024 only if the 2K gate is still insufficient.
- A separated native AVX512 breakthrough reuses decoded MXFP4 weight rows for
  exactly the two- and three-token expert tails while preserving the existing
  one-token mat-vec, fixed 128-column projection blocks, and AMX path at five
  or more routed tokens. Native MXFP4 bytes and UE8M0 scaling are unchanged.
  Every tested output hash matched the live kernel exactly.
- On dwagon, isolated one-expert gains were 41.54% at two tokens and 67.37%
  at three tokens; one/four/six-token changes were -0.60%/-0.73%/+2.30%.
  A realistic six-token, 32-route sample with 24 active experts improved
  from 8.454711 to 6.537172 ms, or 29.33%. The independently compiled fwuff
  kernel improved the same mixed workload from 4.609935 to 3.888558 ms
  (18.55%); its two/three-token gains were 38.99%/55.33%, with
  one/four/six-token changes of -1.64%/+0.06%/-0.69%. Evidence is retained
  under `/var/lib/exo/benchmarks/dsv4-kernel-candidate-ab/`, including the
  copied `fwuff/` results.
- The launchers now set
  `KT_MXFP4_AVX_TILED_MIN_EXPERT_TOKENS=2`. The promotion candidates are
  architecture-local builds with hashes
  `32cbe088f6263bfb02c08eb1d7279de1d800147f25bfab5fa293888b2cddc0df`
  on dwagon and
  `b1f6c80a53865c5d25fe6dae9b22e4d077291e00c8cd9ae93648a33a517ea364`
  on fwuff. No serving or sidecar process is active, and fwuff has 235 GiB
  available, so the imported runtime binaries can be promoted with rollback
  copies before Tier18.

## 2026-07-27 03:54 EDT — Tier18 coherent load and served AVX-tail rejection

- The architecture-local AVX-tail candidates were promoted only into the
  packages actually imported by the deployment, with rollback copies kept.
  Import-time hashes are
  `32cbe088f6263bfb02c08eb1d7279de1d800147f25bfab5fa293888b2cddc0df`
  on dwagon and
  `b1f6c80a53865c5d25fe6dae9b22e4d077291e00c8cd9ae93648a33a517ea364`
  on fwuff. Both synchronized launch scripts are hash-identical across hosts.
- Tier18 sidecars loaded exactly 23/25/48 compact layer shards. Their actual
  native residency is 50,133.46 MiB on selected NUMA 0 versus 192.01 MiB
  elsewhere, 53,538.32 MiB on selected NUMA 1 versus 1,103.66 MiB
  elsewhere, and 20,457.63 MiB on fwuff NUMA 0. Every substantive map carries
  the expected `bind:0` or `bind:1` policy.
- Live-policy boundary oracles after promotion are bit-exact between each
  endpoint and the promoted native reference at PP0 layer 22, PP1 layer 47,
  and fwuff layer 47. Independent SM86 comparisons retain row cosine above
  0.999994. Artifacts are
  `/var/lib/exo/benchmarks/dsv4-pro-tiered18-fwuff12-*-boundary-oracle.json`.
- Tier18 target loads completed in 517.76/631.27/523.67 seconds. The complete
  three-layer DSpark draft added 27.16 seconds on fwuff and initialized with
  gamma 5. The full 1,048,576-token pool and four-request partition allocated;
  PP0 retained 1.04 GiB after breakable target-graph capture, while fwuff
  retained 16 GiB host-available memory. End-to-end arithmetic returned
  exactly `42`, and the factual gate returned Paris and the Seine. Main-rank
  mappings retain hard binding: PP0 is 229,062.15 MiB on selected NUMA 1
  versus 7,336.52 MiB on NUMA 0 (96.90% selected), and PP1 is 256,646.82 MiB
  on selected NUMA 0 versus 347.32 MiB on NUMA 1 (99.86% selected).
- The promoted tail kernel is rejected as a served-speed promotion despite
  its isolated wins. Two official random-8/output-256 samples measured only
  5.259/5.694 output token/s and 188.250/173.717 ms TPOT, with
  658.38/653.29 ms TTFT. DSpark acceptance lengths were 2.825/2.869, lower
  than Tier17d's 3.325/4.15 and a major contributor to the end-to-end
  regression. Artifacts are
  `dsv4-pro-tiered18-fwuff12-avxtail23-random8-out256-seed42.jsonl` and its
  `repeat2` counterpart. The binary remains useful experimental evidence but
  will not become the deployment default without a same-service causal gate.
- Before retiring Tier18, the 2,048-token chunk fix will be tested on an exact
  fresh 32,768-token request. This separates the required long-prefill
  workspace correction from the rejected decode experiment.

## 2026-07-27 03:56 EDT — 2K prefill gate and Tier19 correction

- The exact fresh 32,768-input/8-output gate still exhausted PP0 VRAM at a
  2,048-token chunk. This time the failure moved past the earlier
  `_compute_q_b` allocation and into the portable SM86 native-MXFP4 hot-expert
  path: `triton_kernels.matmul_ogs.apply_allocation` requested 224.00 MiB
  with 128.94 MiB physically free. The process held 23.42/23.56 GiB; PyTorch
  reported 235.44 MiB reserved but unallocated, which was not available as a
  usable contiguous allocation. The failed distributed group was discarded.
- Chunk scaling is again linear: the corrected deployment default is now
  1,024 tokens, projecting this allocation to about 112 MiB while leaving the
  full 1,048,576-token pool and four-request partition unchanged. This is a
  necessary real-workload memory correction, not context-capacity reduction.
- Tier19 will restore the proven pre-tail runtime binaries on both hosts and
  restart all sidecars so every loaded process uses the same native kernel.
  Its only serving-argument change from Tier17d is the 1,024-token prefill
  chunk. This both rejects the unpromoted served AVX-tail experiment and
  provides the cleanest available long-prefill/decode comparison.

## 2026-07-27 04:18 EDT — Tier19 full-context prefill succeeds

- Tier19 restored the proven imported binaries at hashes `42a28908...` on
  dwagon and `adf4d853...` on fwuff. All three restarted sidecars again passed
  exact live-policy boundary oracles; artifacts are
  `/var/lib/exo/benchmarks/dsv4-pro-tiered19-fwuff12-*-boundary-oracle.json`.
- Target loads completed in 519.00/632.96/475.81 seconds, and the complete
  DSpark draft added 28.51 seconds. The full 1,048,576-token pool with four
  requests allocated, PP0 retained 1.05 GiB after graph capture, and fwuff
  retained 16 GiB host-available memory. Arithmetic coherence returned
  exactly `42`.
- Strict main-process placement remains valid: PP0 has 228,543.75 MiB on
  selected NUMA 1 versus 7,722.31 MiB on NUMA 0 (96.73% selected), and PP1
  has 256,598.36 MiB on selected NUMA 0 versus 175.13 MiB on NUMA 1
  (99.93% selected). The small PP0 spill remains consistent with the earlier
  3.40% measured remote-access share and is not a material bottleneck.
- The exact fresh 32,768-input/8-output benchmark completed successfully with
  every rank still healthy. It produced exactly 32,768 input and eight output
  tokens, 213,428.91 ms cold TTFT, 152.146 fresh input tok/s including about
  108 seconds of first-use compilation, and 215.37 seconds total duration.
  Steady 1,024-token chunks were predominantly 218--253 tok/s. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-tiered19-fwuff12-rollback-random32k-out8-seed42.jsonl`.
- This proves the 1K correction safely supports the required fresh 32K
  request without reducing the 1M cache or native MXFP4 weights. It does not
  satisfy the 500+ prefill objective. The next kernel work targets the
  512 MiB `_compute_q_b` copy and the proportional
  `matmul_ogs.apply_allocation` scratch so larger AMX prefill chunks can fit
  in the same VRAM envelope.
- Two official rollback random-8/output-256 samples measured 6.797/4.801
  output token/s and 145.020/206.359 ms TPOT, with DSpark acceptance lengths
  2.175/3.281. Their approximate target-verification intervals were
  315/677 ms. The first is faster than both AVX-tail samples despite lower
  acceptance, and the pair confirms that the experimental tail binary is not
  the served default. Artifacts are
  `dsv4-pro-tiered19-fwuff12-rollback-random8-out256-seed42.jsonl` and its
  `repeat2` counterpart. Large interval variance remains a separate
  scheduling/contention target.
- Source is staged to make the fused target and draft Q RMSNorm+RoPE kernels
  operate in place whenever the caller did not provide a destination. The
  CUDA implementation registers the complete disjoint `(token, head)` vector
  before its first store, so aliasing removes a full 128-head x 512-dimension
  BF16 copy: exactly 512 MiB at a 4,096-token chunk. A direct proof awaits
  releasing the live ranks because the intentionally full 1M service leaves
  insufficient VRAM even to create a second CUDA context.

## 2026-07-27 04:24 EDT — in-place Q kernel proof

- After retiring only the Tier19 ranks, the fused Q RMSNorm+RoPE kernel was
  tested directly on SM86 at the production 128-head x 512-dimension geometry.
  In-place and independently allocated outputs were byte-identical at
  1, 7, 64, and 1,024 tokens. The 1,024-token pair shared SHA-256
  `3508658898c7c88831c98bf30e7ddc2e0044b3133bcc057a9ca74c0eed05cc9d`.
- This is a true kernel/dataflow improvement: it removes the complete Q copy
  rather than reducing context, experts, precision, or cache. The patch is in
  both `deepseek_v4.py` and `deepseek_v4_dspark.py`. Tier20 will explicitly
  test 4,096-token chunks with the full 1M/four-request pool; the launcher
  retains its safe 1K default until that served gate succeeds.

## 2026-07-27 04:40 EDT — Tier20 isolates the second 4K attention copy

- Tier20 loaded the strict fwuff-12 topology with explicit 4,096-token chunks.
  Target load times were 665.24/706.40/363.74 seconds and the complete DSpark
  draft added 34.28 seconds. Every rank allocated the full 1,048,576-token
  target pool with four-request admission; PP0/PP1/PP2 reported 1.01/0.93/4.26
  GiB available after pool allocation. The deterministic arithmetic gate
  returned exactly `42`, so the in-place fused-Q change preserves served model
  semantics.
- The first exact 32,768-input/8-output request advanced beyond the removed
  `_compute_q_b` copy, then PP0 failed at a different full-size allocation:
  `debug_flash_mla_adapter._v4_triton_decode_dispatch` requested another
  512 MiB for sparse-attention output with only 62.94 MiB physically free.
  This is not the earlier Q allocation and not the prior hot-expert
  `matmul_ogs` scratch. The request was rejected and the complete distributed
  group was retired after PP0 failed.
- The portable SM86 sparse-attention kernels load the entire disjoint
  `(token, head)` Q vector into registers before their attention loop and make
  their only output store after Q's final use. Their BF16 production path now
  aliases output to Q instead of allocating a second
  `[tokens, 128, 512]` tensor. Direct SM86 out-of-place versus in-place tests
  were byte-identical at 1, 7, 64, and 1,024 tokens; the 1,024-token output
  hash was
  `4a15517be5cc9f218e4c3ac47e6d17c1cea85f58f02c23973cd9ca79e8ffbb16`.
  The synchronized adapter SHA-256 is
  `58e1d03b5e2db788cf13904c1d5c33a66efaa8fe5d2182eec7ef36559d4ea1ce`.
  Tier21 will repeat the full 4K served gate with both 512 MiB copies removed.

## 2026-07-27 04:57 EDT — Tier21 reaches the streamed output projection

- Tier21 retained the strict fwuff-12 topology, full 1,048,576-token/four-user
  pool, native MXFP4 weights, and explicit 4,096-token chunks. Target load
  times were 629.46/669.51/372.66 seconds; the complete DSpark draft added
  27.16 seconds. PP0 and PP2 retained 1.01 and 4.26 GiB respectively after
  pool allocation. The deterministic served arithmetic gate again returned
  exactly `42`.
- The exact 32,768-input/8-output request passed both in-place 512 MiB Q and
  sparse-attention buffers. PP0 then reached `DeepseekV4Attention.forward`'s
  next allocation, the BF16 `torch.einsum("tgd,grd->tgr", o, wo_a)` result.
  Its full `[4096, 128, 128]` output requested 128 MiB while only 62.94 MiB
  was physically free. This is a new, smaller allocation site; the benchmark
  stream was correctly rejected and no result artifact was accepted.
- The next target path preallocates the smaller final
  `[tokens, hidden_size]` output before Q/attention consume the remaining
  VRAM and streams unchanged BF16 `wo_a -> wo_b` math in 64-token tiles.
  A full flattened-size factorized A/B (`G*D=65536,G*R=16384,H=7168`,
  tested as `G=128,D=512,R=128`) had cosine 1.0, mean absolute difference
  `6.3947e-06`, and maximum BF16 difference `0.0078125`. The checkpoint's
  actual grouping is `G=16,D=4096,R=1024`; Tier22 is its served coherency
  gate. Decode and small prefill retain the original single-GEMM path.
  Synchronized target source SHA-256 on dwagon/fwuff is
  `de5bb4ccf9ae0b3d8ce9ba6c9cc813e3c987399f78bf9a5bca4eca156e5e368e`.
- A second independent prefill breakthrough compacts only valid masked GPU
  expert routes into top-1 route rows, gathers directly from the original
  token tensor, and reduces contributions in original slot order with an
  FP32 Triton kernel. This removes top-6-to-8 dense intermediates for CPU and
  remote routes. On real layer-10 Pro weights with three valid of six routes,
  the compact and baseline GPU outputs were byte-identical at SHA-256
  `338a5b6efa8775a11e3a0c89f3bf432b1a62fdb479498933e2e4add9792ea541`;
  both retained cosine `0.99996245` and relative mean difference `0.00267854`
  versus native AMX. Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-compact-gpu-routes-real-layer10-3of6-t64.json`
  and `dsv4-pro-baseline-gpu-routes-real-layer10-3of6-t64.json`. Synchronized
  portable-kernel SHA-256 is
  `e60c5e52db2877c756018e34cc490705ddceb770a25812607b25cacce9af8abc`.
- The fwuff kernel log now contains three additional hardware-corrected
  single-bit ECC events from the same `Card01, ChnF, DIMM0` at 03:43:15,
  03:43:19, and 03:43:34 CDT, at distinct physical addresses. There is still
  no uncorrected event, but recurrence under load makes this a DIMM-health
  concern rather than a one-off. Continue exact semantic/numerical gates and
  monitor APEI; hardware service should target that FRU independently of the
  CUDA-memory work. SMBIOS maps channel F to `P0_CHF_DIM0`, Samsung
  `M321R4GA3BB0-CWMQH`, serial `03894115`.

## 2026-07-27 05:13 EDT — Tier22 exposes output-buffer ordering headroom

- Tier22 loaded the same strict-NUMA fwuff-12 topology with the full
  1,048,576-token/four-user pool, native MXFP4 weights, DSpark, explicit
  4,096-token chunks, compact valid GPU routes, and the 64-token streamed
  output projection. Target rank loads completed in 640.30/666.09/331.59
  seconds and the complete draft added 34.49 seconds. The deterministic
  served arithmetic gate returned exactly `42`.
- The exact fresh 32,768-token request failed before reaching either the
  streamed projection or compact expert route. The first implementation
  allocated its 56 MiB `[4096, 7168]` final-output buffer before
  `_forward_prepare`; the subsequent FP8-Marlin `wq_b` operation required the
  unavoidable 512 MiB Q result with only 490.94 MiB physically free. PyTorch
  reported 22.31 GiB allocated and 198.20 MiB reserved but unallocated. This
  precisely identifies allocation lifetime/order, rather than projection
  arithmetic, as the regression from Tier21's successful passage through Q.
  No failed benchmark artifact is accepted.
- `_forward_prepare` no longer needs its normalized `[tokens, 7168]` input
  after it has produced Q/KV. The corrected implementation therefore decides
  the streamed path before Q, but reuses that now-dead, shape-identical input
  storage as the final `wo_b` output only after `_forward_prepare` returns.
  It allocates no early 56 MiB buffer, restores Tier21's Q headroom, and still
  avoids the full 128 MiB `wo_a` intermediate. The installed target source
  carrying this correction has SHA-256
  `9c8f65b5046b70921c54a8048db69161ba80b7931b3abc93bdb5c1c648626e0e`.
- A readiness probe accidentally used `/health_generate`, which is an active
  256-token generation endpoint in this server rather than a passive health
  check. All probe work was allowed to drain before the exact gate. Future
  readiness checks use only `/v1/models`.
- Tier22 ranks have been retired; the three already-oracled expert sidecars
  remain active. No additional fwuff corrected or uncorrected APEI event was
  logged during the Tier22 load and gate.
- The corrected reuse path was independently exercised at the checkpoint's
  actual `G=16,D=4096,R=1024,H=7168` geometry with 65 tokens and a 64-token
  tile, including the one-token tail. The caller-owned output pointer was
  preserved. Relative mean difference versus the one-shot BF16 projection was
  0.0016141 with cosine 0.99999636; incremental peak allocation fell from
  13,711,360 to 5,144,576 bytes. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-streamed-wo-actual-g16d4096r1024h7168-t65-c64.json`.

## 2026-07-27 05:34 EDT — Tier23 passes attention, reaches CPU-output staging

- Tier23 was the causal repeat with the corrected post-Q output-buffer reuse.
  Target loads completed in 654.01/678.90/370.49 seconds, the complete DSpark
  draft initialized with gamma 5, and every rank allocated the full
  1,048,576-token/four-request pool. PP0/PP1 retained 1.01/1.00 GiB after
  target graph capture and PP2 retained 4.32 GiB after both target and draft
  pools. Passive `/v1/models` readiness advertised `max_model_len=1048576`;
  the deterministic served arithmetic gate returned exactly `42`.
- The exact 32,768-input/8-output request passed the 512 MiB Q path, in-place
  sparse-attention output, and corrected streamed WO projection. It entered
  the first 4K MoE layer and then failed before GPU expert execution when
  `KExpertsCPUBuffer.get_buffer` eagerly allocated its first
  `[4096,7168]` GPU copy-out ring slot. The 56 MiB allocation had 50.94 MiB
  physically free; PyTorch held 436.41 MiB reserved but unallocated in
  fragmented blocks. No failed benchmark artifact is accepted.
- This buffer is redundant in the SGLang integration. A persistent 56 MiB
  shared staging tensor already protects the asynchronous GPU-to-host input
  copy; once the native CPU task synchronizes, that same tensor is dead and
  can receive the host-to-GPU result. The compact GPU route can likewise write
  its final deterministic reduction over the now-dead original hidden-state
  tensor, then merge CPU and remote contributions in place. This removes both
  eager GPU copy-out ring slots and the full-size CPU/remote sum temporaries
  without changing weights, routing, precision, chunk size, or context.
- fwuff logged another hardware-corrected single-bit ECC event during the
  native load at 04:19:50 CDT. It again identifies
  `fru_text: Card01, ChnF, DIMM0`, this time at physical address
  `0x3f6e68cd80`; the kernel soft-offline attempt reported the page as
  unhandleable. There is still no uncorrected event or numerical mismatch,
  but recurrence on the same FRU now strongly warrants DIMM replacement.
- The replacement CPU ring is lazy and accepts an explicit synchronized
  output tensor. At the production 4,096 x 7,168 BF16 shape,
  `KExpertsCPUBuffer.get_buffer` now allocates zero GPU bytes and leaves both
  device-output ring slots unmaterialized. The synchronized source/runtime
  SHA-256 on both hosts is
  `f4c0b39f32de8a1026d769ec0eacdfa8eccca0838a9ccd364397d68cacb052f6`.
- The compact reduction now writes over its dead input. A real layer-10
  three-valid-of-six oracle remains byte-identical to the prior compact and
  baseline output at
  `338a5b6efa8775a11e3a0c89f3bf432b1a62fdb479498933e2e4add9792ea541`,
  with cosine 0.99996245 versus native AMX. Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-compact-gpu-routes-inplace-reduce-real-layer10-3of6-t64.json`.
  The synchronized GPU kernel SHA-256 is
  `068837df0db6494bf6c95351f14127d4ddd4eea7d4768e0fa967426cd7982215`.
- The SGLang wrapper copies native output into the shared staging tensor only
  after the CPU-stream synchronization event, adds it into the GPU accumulator
  in place, and directly `index_add_`s each remote tier's compact unique-token
  rows. A two-tier overlapping-token unit oracle is byte-exact. The
  synchronized wrapper SHA-256 is
  `6108d2ce4a4ceb9541c4dd57d34700b1d5e4714b268a217b6c2aa8065230a4de`.

## 2026-07-27 05:52 EDT — Tier24 reaches remote result transfer

- Tier24 promoted the lazy native output ring, in-place compact reduction,
  and in-place CPU/remote accumulation. Target loads completed in
  662.42/689.62/353.60 seconds and the complete DSpark draft added 31.13
  seconds. All ranks again allocated the full 1,048,576-token/four-request
  pool, PP0 retained 1.01 GiB after graph capture, and served arithmetic
  returned exactly `42`.
- The exact fresh 32K request advanced through local native CPU submission,
  compact GPU execution, the reused native copy-out tensor, and the in-place
  local CPU merge. It then joined the first remote future. The old remote
  client attempted a one-shot 48 MiB `output_cpu.to(device=...)` for the
  selected compact token rows with 20.94 MiB physically free and failed.
  This is later than Tier23's eager 56 MiB native ring failure and proves the
  local full-token buffers were successfully removed. No failed benchmark
  artifact is accepted.
- The remote transport now receives the response directly from the persistent
  sidecar socket into pinned BF16 CPU storage. This removes the intermediate
  response bytearray copy and leaves the result on host. After the local CPU
  merge, the same shared GPU staging tensor is dead; remote rows stream through
  its first 64 rows and `index_add_` directly into the existing accumulator.
  No compact remote result tensor or full-token tier tensor is allocated on
  GPU.
- A live layer-0 sidecar oracle with 65 tokens and a 16-token transfer tile
  was byte-exact across the former GPU-return path, direct pinned receive, and
  streamed in-place merge, all at SHA-256
  `e4c3cc4295238b4f750ee6bf0ca156e6cc2119b1b93a70cbcafa5cece5c5668d`.
  Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-streamed-remote-merge-local-sidecar-layer0-t65-c16.json`.
  Synchronized transport/wrapper hashes are `6d4f6dda...` and `700c2a02...`.
- fwuff emitted a further burst of three hardware-corrected single-bit reads
  at 04:42:40--43 CDT, all on `Card01, ChnF, DIMM0`, at three distinct
  physical pages. One page was invalidated, one transparent-huge-page split
  failed, and one page was reported unhandleable. No uncorrected event or
  served semantic mismatch occurred, but this FRU must be replaced before the
  hardware can be considered production-stable.

## 2026-07-27 06:07 EDT — Tier25 completes first MoE, reaches MHC post

- Tier25 target loads completed in 663.25/674.20/343.13 seconds and the
  complete DSpark draft added 26.02 seconds. Full 1M/four-request pools and
  the target graph allocated with the same 1.01 GiB PP0 headroom; served
  arithmetic again returned exactly `42`.
- The exact fresh 32K request completed the entire first hybrid MoE:
  compact hot-expert GPU math, native local AMX work, reused native H2D
  copy-out, both streamed remote sidecar receives, and in-place CPU/remote
  accumulation. It then entered the post-FFN MHC map. `mhc_post` attempted
  to allocate a new `[4096,4,7168]` BF16 result of 224 MiB with 120.94 MiB
  physically free. This is later than Tier24's remote-transfer failure and
  proves the streamed transport/merge path succeeded. No failed benchmark
  artifact is accepted.
- The TileLang MHC kernel copies all four residual rows for each hidden tile
  into shared memory before any output store. The incoming residual is dead
  after `mhc_post`, so the DSV4-gated path now writes the result over that
  storage. A full production-geometry 4,096-token oracle was byte-exact
  between independent and aliased output across all 234,881,024 bytes at
  SHA-256
  `c6b1154146edb0efbab8b987b6975ba655e31da68881e8e8a4127a1ddcc8df5d`.
  Artifact:
  `/var/lib/exo/benchmarks/dsv4-pro-inplace-mhc-post-t4096-h7168-hc4.json`.
  Synchronized `mhc.py` SHA-256 is `eadbef698b64...`; activation is explicitly
  gated by `SGLANG_DSV4_INPLACE_MHC_POST=1`.
- fwuff logged another corrected single-bit read at 04:59:57 CDT on
  `Card01, ChnF, DIMM0`, physical address `0x9d72cc680`. The affected page was
  successfully invalidated. There is still no uncorrected event, but the same
  FRU continues to accumulate corrected errors under model residency.

## 2026-07-27 06:24 EDT — Tier26 clears MHC; GPU-hidden AMX sidecars

- Tier26 promoted the byte-exact in-place MHC post path. Target loads completed
  in 695.51/684.87/350.22 seconds and fwuff loaded the complete three-layer
  DSpark draft in another 30.11 seconds. All ranks provisioned the full
  1,048,576-token/four-request pools; PP0 retained 1.01 GiB after its target
  graph and `/v1/models` advertised `max_model_len=1048576`. The deterministic
  served arithmetic gate returned exactly `42`, proving the first real request
  passed the new in-place MHC path.
- Live NUMA residency during the target load was 97.20% on PP0's selected node
  and 99.74% on PP1's selected node. Aggregate memory still had roughly
  240 GiB available. This confirms the production compromise should remain
  strict CPU binding plus `localalloc`, with strict native pools and sidecars;
  forcing hard process-wide `membind` for the remaining ~2.8% PP0 mappings
  would reintroduce the already-observed load OOM for little locality benefit.
- The exact fresh 32K gate advanced past the previously failing first MoE and
  MHC post, then failed in the first layer's C4 indexer. Its FP8-Marlin
  `wq_b` projection attempted a 64 MiB 4,096 x 8,192 BF16 output with
  52.94 MiB physically free. This is an indexer allocation, not an MXFP4
  expert-path regression. The interrupted benchmark is invalid and no failed
  artifact is accepted.
- The three CPU-only native-expert sidecars were each holding an otherwise
  idle 256 MiB CUDA context. A new `KT_CPU_ONLY_SIDECAR=1` mode leaves only
  the sidecar's CPU expert mask unpinned and launches with CUDA hidden; all
  serving/client staging remains pinned. Both dwagon sidecars had selected
  physical GPU 0, so this recovers 512 MiB for PP1 and none for PP0; fwuff
  recovers 256 MiB for PP2. A live layer-0 oracle was byte-exact between
  GPU-visible and GPU-hidden sidecars, including direct pinned receive and
  streamed merge, at SHA-256
  `e4c3cc4295238b4f750ee6bf0ca156e6cc2119b1b93a70cbcafa5cece5c5668d`.
  Artifacts:
  `/var/lib/exo/benchmarks/dsv4-pro-sidecar-gpuvisible-reference-layer0-t65.json`
  and
  `/var/lib/exo/benchmarks/dsv4-pro-sidecar-gpuhidden-oracle-layer0-t65.json`.
- The updated native buffer source/runtime SHA-256 is
  `bc2299ea96bd8788d9e8590d592855a05d1c288707300cdeb40c72ce4f3b75be`.
  Tier27 sidecars are reloading on the same strict-NUMA endpoints with no
  NVIDIA process, recovering those contexts without changing weights,
  arithmetic, routing, context capacity, or chunk size.
- A Tier27 load coherency check confirmed PP0 still had Tier26's 5.93 GiB
  post-weight headroom, while PP2 rose from 8.97 to 9.23 GiB. Tier27 was
  retired before repeating a predictably unchanged PP0 32K failure. The
  launcher now exposes `DSV4_MEM_FRACTION_STATIC`; PP0 will use 0.985. This
  releases about 126 MiB of activation headroom while the DSV4 pool
  calculation still projects roughly 1.19M full tokens, above the required
  1,048,576. PP1/PP2 retain 0.99. This is an argument-level complement to the
  earlier byte-exact in-place kernel/dataflow changes, not a reduction of
  context capacity or expert precision.
- Tier28 proved the lower static fraction still calculated 1,187,072 full
  tokens and allocated the exact production pools
  (`c4_size=262144`, four requests, `max_total_num_tokens=1048576`), but the
  fixed pool shapes consumed the same physical bytes and PP0 still retained
  1.01 GiB after graph capture. It was retired without repeating the known
  32K allocation failure; `mem_fraction_static` is restored to 0.99.
- The actual peak fix is an env-gated prefill dataflow reorder. On the ordinary
  single-rank NVIDIA path identified by the Tier26 traceback, the C4 indexer
  and compressor now consume their short-lived projection outputs before the
  512 MiB main-attention Q is materialized. The same projections, cache
  stores, quantization kernels, and main Q execute in a different independent
  order; no weight or result dtype changes. This removes the 64 MiB indexer-Q
  overlap with main Q (and much more peak headroom) rather than merely reducing
  an argument. Synchronized, syntax-checked `deepseek_v4.py` SHA-256 is
  `113d0077b91c8a3e0e7f9bafa85b4cae92f218d3bf722c799a9826efbc70583a`;
  Tier29 enables it for prefills of at least 512 tokens.
- Tier29 provisioned the exact production pools, advertised 1M context, and
  returned the deterministic `42`. Its first 4K chunk passed the former
  64 MiB indexer-Q allocation, proving the reorder executed, but then failed
  when main-Q Marlin requested its 512 MiB output. At that point PyTorch had
  692.72 MiB reserved but unallocated in fragmented blocks and only
  52.94 MiB physically free. No failed benchmark artifact is accepted.
- Tier30 strengthens the same schedule without tiling or changing arithmetic.
  It reserves the contiguous 512 MiB main-Q buffer at the original successful
  allocation point, uses the first contiguous 64 MiB as caller-owned output
  for indexer FP8-Marlin, immediately quantizes/consumes it, and then has
  main-Q FP8-Marlin overwrite the complete caller-owned buffer. Marlin already
  accepts an output pointer at the kernel boundary; the new Python wrapper
  exposes it with pointer-preservation checks. This removes both failed
  allocations while executing the same packed FP8 weights and kernels.
  Synchronized, syntax-checked hashes are `681a4673...` (`deepseek_v4.py`),
  `90b34f63...` (`indexer.py`), and `7859ae42...`
  (`marlin_utils_fp8.py`).
- fwuff logged two more corrected single-bit ECC reads at 05:32:40 CDT on the
  same `Card01, ChnF, DIMM0`, physical addresses `0x24d37ca280` and
  `0x143f25380`; the latter page was unhandleable. No uncorrected event was
  reported. Exact gates remain mandatory, and the DIMM still requires
  replacement.

## 2026-07-27 07:32 EDT — Tier30 clears both Q projections; compressor output reuse

- Tier30 target loads completed in 661.97/651.65/31.54 seconds, all three
  ranks provisioned the exact 1,048,576-token/four-request production pools,
  and PP0 retained 1.01 GiB after target graph capture. Passive readiness
  advertised `max_model_len=1048576`; the deterministic served arithmetic
  gate returned exactly `42`.
- The exact fresh 32K request passed both caller-owned FP8-Marlin indexer Q
  and main Q projections. It then reached the indexer compressor's
  BF16-by-BF16-to-FP32 GEMM, which requested a new 20 MiB score output with
  only 20.94 MiB physically free. PyTorch had 195.72 MiB reserved but
  unallocated. The server terminated before completing the request, so no
  failed benchmark result is accepted.
- Compressor scores are immediately consumed into persistent compression
  state/cache before main Q is projected. Tier31 therefore exposes the
  existing `torch.mm(..., out=..., out_dtype=torch.float32)` contract and
  reuses typed FP32 views of the reserved 512 MiB main-Q block for both the
  indexer and core compressor, each with its own exact width. Stream ordering
  prevents either score lifetime from overlapping the next reuse. No kernel,
  accumulation dtype, weight, cache format, chunk size, or context argument
  changes.
- At the exact 4,096-token, 7,168-hidden production geometry, independent
  512-wide indexer-C4 and 2,048-wide core-C4 GEMM checks were bitwise equal
  to their ordinary allocations (`max_abs_diff=0`), preserved the caller
  pointer, and increased `torch.cuda.memory_allocated()` by zero bytes. The
  tested shared Q block was exactly 536,870,912 bytes.
- Syntax compilation and `git diff --check` pass. Tier31 source hashes are
  `c7e7832e...` (`gemm.py`), `f194ccf9...` (`compressor.py`),
  `1fa4a896...` (`compressor_v2.py`), `9a9ca314...` (`indexer.py`), and
  `0f31b03a...` (`deepseek_v4.py`).

## 2026-07-27 07:48 EDT — Tier31 clears compressors; cross-layer Q workspace

- Tier31 target loads completed in 671.23/696.79/448.97 seconds and the
  complete fwuff DSpark draft added 32.43 seconds. Every rank allocated the
  exact production 1M/four-request pools; PP0 retained 1.01 GiB after graph
  capture. Passive readiness reported `max_model_len=1048576`, and served
  arithmetic returned exactly `42`.
- Live target-load NUMA residency was 97.7% on PP0's selected node and 99.0%
  on PP1's selected node. This independently confirms that strict CPU binding
  plus `localalloc`, hard-bound native pools, and hard-bound sidecars remain
  the correct policy; hard whole-process memory binding is still unnecessary.
- The exact fresh 32K request passed indexer Q, both indexer/core compressor
  score GEMMs, and the main Q of the C4 layer. It then reached a later
  C128/no-indexer layer and failed when that layer independently requested
  another 512 MiB main-Q output. Only 52.94 MiB was physically free, while
  692.72 MiB was reserved but fragmented. This is the former Tier29 main-Q
  allocation at a later layer, proving Tier31's compressor reuse succeeded.
  No failed result is accepted.
- Tier32 makes the 4,096 x 128 x 512 BF16 main-Q block a single backend-owned
  512 MiB workspace allocated once and reused across all serial attention
  layers and later chunks. C4 indexer Q and C4/C128 compressor scores consume
  typed views first; caller-owned FP8-Marlin main Q then overwrites the block,
  attention consumes it, and the next layer reuses it. This prevents smaller
  intermediates from splitting a returned 512 MiB caching-allocator segment.
  Decode, graph capture, CP, NPU/HIP, packed FP8 weights, and all arithmetic
  dtypes are unchanged.

## 2026-07-27 08:07 EDT — Tier32 clears all attention projections

- Tier32 target loading completed and all ranks again provisioned the exact
  1,048,576-token/four-request production pools. The deterministic served
  arithmetic gate returned exactly `42`.
- The exact fresh 32K request passed the C4 indexer Q, both compressor score
  GEMMs, the C4 main Q, and a later C128/no-indexer main Q. This proves the
  backend-owned cross-layer Q workspace removed the later-layer 512 MiB
  allocation failure seen in Tier31.
- Execution advanced into the serial shared expert and failed when its
  post-SwiGLU activation requested 24 MiB at
  `DeepseekV2MLP.forward`; only 16.94 MiB was physically free while
  196.72 MiB was reserved but fragmented. The server terminated before
  completing the request, so no failed benchmark result is accepted.
- Tier33 reuses the dead contiguous prefix of the DSV4 main-Q workspace for
  this shared-expert activation. Attention has fully consumed main Q before
  the serial MLP begins, the clamp/SwiGLU kernel writes the prefix, and the
  down projection consumes it before the next layer can reuse main Q. The
  path is limited to shared experts with at least 512 tokens and only activates
  when the explicitly named DSV4 workspace has matching dtype, device,
  contiguity, and capacity. Decode, CUDA-graph capture, dense MLPs, other
  models/backends, weights, kernels, and arithmetic dtypes are unchanged.
- At the exact 4,096-token by 6,144 gate/up production geometry, the reused
  activation view was 25,165,824 bytes at the same pointer as the
  536,870,912-byte Q workspace, increased allocated CUDA memory by zero bytes,
  and was bitwise equal to an independently allocated output
  (`max_abs_diff=0`). Separate large dense-MLP and eight-token shared-expert
  checks retained the ordinary allocation path, confirming the large-prefill
  scope. Synchronized, syntax-checked `deepseek_v2.py` SHA-256 is
  `ee39318896212b79f8e5a60926d839aedda4abe26eab531bfc85cd1b798523c7`.
- Correction to the Tier31 wording: inspection of the retained transient units
  confirms `DSV4_STRICT_MAIN_NUMA=1` on both dwagon ranks, which maps to
  process-wide `--membind` in the current launcher. Thus the compact
  GPU/opposite-NUMA/fwuff topology has already tightened the main ranks as well
  as native pools and sidecars; the 97.7%/99.0% residency figures were measured
  under that hard binding. It fits because twelve experts per layer are
  offloaded to fwuff. The earlier full-local topology still cannot use hard
  binding without OOM, but that is no longer the deployed topology.

## 2026-07-27 08:22 EDT — Tier33 clears shared activation; caller-owned shared down output

- Tier33 completed target loads in 691.47/701.88/426.72 seconds. fwuff's
  complete DSpark draft added 27.24 seconds. All three ranks provisioned the
  exact 1,048,576-token/four-request pools with 4,096-token chunks; PP0
  retained 1.01 GiB after target graph capture, and passive readiness
  advertised `max_model_len=1048576`. The deterministic served arithmetic
  gate returned exactly `42`.
- The exact fresh 32K request passed the shared expert's post-SwiGLU
  activation, proving its 24 MiB Q-workspace view executed. It then failed at
  the immediately following FP8-Marlin shared down projection when Marlin
  requested its ordinary 56 MiB `[4096,7168]` output with 16.94 MiB
  physically free and 196.72 MiB reserved but fragmented. This is the
  predicted next allocation boundary. The server terminated, no benchmark
  result file was created, and no failed result is accepted.
- Tier34 exposes Marlin's existing `c=` caller-output contract through
  `Fp8LinearMethod` and `RowParallelLinear`. For a large DSV4 shared expert
  using Marlin at TP1, it places the 56 MiB down output immediately after the
  disjoint 24 MiB activation inside the same dead 512 MiB Q workspace. The
  down GEMM reads the first range and writes the second; the returned shared
  output is consumed before the next attention layer reuses the workspace.
  All default allocation paths, non-Marlin methods, dense MLPs, small
  batches/decode, TP>1, weights, and arithmetic dtypes remain unchanged.
- A full `4096 x 3072 @ 3072 x 7168` SM86 FP8-Marlin oracle used the exact
  checkpoint dimensions and a 536,870,912-byte Q workspace. The ordinary path
  allocated 58,720,256 bytes. The caller-owned path preserved its pointer at
  byte offset 25,165,824, was disjoint from its input, and was bitwise equal
  (`max_abs_diff=0`, `mean_abs_diff=0`, cosine `1.0`). Synchronized,
  syntax-checked hashes are `77b277c6...` (`fp8.py`), `d6588d3d...`
  (`linear.py`), and `9efd63fc...` (`deepseek_v2.py`); the already accepted
  caller-output Marlin implementation remains `7859ae42...`.
- Live loading again confirmed hard main-rank binding is capacity-safe for
  this compact topology. Mid-load selected-node residency was approximately
  97.5% for PP0 and 98.6% for PP1. No new ECC/EDAC event occurred; the
  recurring benign GPU1 driver allocation warning at distributed
  initialization was identical to prior successful launches.

## 2026-07-27 08:43 EDT — Tier34 reaches routed-input gather; Tier35 removes remaining route allocations

- Tier34 promoted the caller-owned shared-down output. All three target ranks
  again completed loading under hard process-wide NUMA memory binding, the
  exact 1,048,576-token/four-request pools allocated, and the deterministic
  arithmetic gate returned exactly `42`.
- The first exact fresh 32K request passed both shared-expert Q-workspace
  allocations. It then entered the first routed MoE and failed while compacting
  the remote tier input with `flat_x[selected_indices]`, which requested a
  fresh 48 MiB tensor with only 16.94 MiB physically free and 238.41 MiB
  reserved but fragmented. No benchmark artifact was produced or accepted.
- Tier35 changes the large-prefill route dataflow without changing arithmetic.
  Each remote tier now writes its compact activation directly into a disjoint
  slice of the dead main-Q workspace via `torch.index_select(..., out=...)`.
  The live cursor advances cumulatively across both tiers and is published for
  the later GPU route kernel. A production-geometry two-tier oracle used an
  80 MiB live prefix, 3,500 and 2,500 selected rows, and exact 50,176,000- and
  35,840,000-byte compact tensors. Both were bitwise equal to ordinary
  advanced indexing; their total 86,016,000-byte payload fit inside the
  existing 512 MiB workspace with only 625,664 bytes of incremental CUDA
  metadata.
- The compact native-MXFP4 GPU kernel now puts GEMM1 output, SwiGLU output,
  and GEMM2 output after that remote cursor in the same workspace. GEMM1 and
  GEMM2 reuse one range because GEMM1 is dead after SwiGLU; the activation is
  disjoint. `matmul_ogs` requires caller-owned outputs in canonical
  `(1, routed_rows, columns)` shape and returns the usual squeezed view. The
  corrected real layer-10/global-expert-351 oracle at 512 tokens and exact
  7,168/3,072 production dimensions was byte-exact with the independent
  allocation path. Both output hashes were
  `c454f6e3f6019b2162febb6f62a1b856e23fd58ba2c83d18a6656660368ff463`;
  maximum and mean differences were zero, the cursor was exact, and
  `torch.cuda.memory_allocated()` increased by zero bytes across the
  caller-owned call.
- Syntax compilation and `git diff --check` pass. dwagon and fwuff now share
  SHA-256 `3cbab1af...` (`deepseek_v2.py`), `a6e54d7e...`
  (`kt_ep_wrapper.py`), and `56026e8d...`
  (`v4_triton_kernels_moe.py`). `deepseek_v4.py` remains
  `ea90d6c5...`. Tier35 will retain the proven compact fwuff12 ownership,
  GPU-hidden sidecars, native MXFP4 weights, AMX prefill, AVX-512 decode, and
  strict main/worker/sidecar NUMA placement.

## 2026-07-27 09:00 EDT — Tier35 clears route temporaries; zero-GPU accumulator reuse

- Tier35 target loads completed in 643.32/654.61/251.78 seconds and fwuff's
  complete three-layer DSpark draft added 34.13 seconds. Every rank allocated
  the exact 1,048,576-token/four-request pools. PP0 captured its target verify
  graph with 1.01 GiB remaining, `/v1/models` advertised
  `max_model_len=1048576`, and deterministic served arithmetic returned
  exactly `42`.
- The exact fresh 32K request passed both remote input compactions and the
  compact GPU expert's caller-owned GEMM1/SwiGLU/GEMM2 workspace. This proves
  that all Tier35 ranges executed together without aliasing. It then reached
  layer 0, which intentionally has no profile-selected GPU expert, and failed
  when the wrapper requested a fresh 56 MiB `torch.zeros_like(x)` accumulator
  with only 16.94 MiB physically free. No benchmark artifact was created or
  accepted.
- For a zero-GPU-expert layer, the original activation has already been copied
  into the persistent native staging tensor and all remote compact tensors
  before the accumulator is initialized. Tier36 therefore reuses that dead
  input with `x.zero_()` for large DSV4 prefills when the MoE runner advertises
  in-place support. The path is explicitly gated by
  `SGLANG_DSV4_INPLACE_ZERO_GPU_MOE=1`, a minimum of 512 rows, and the DSV4
  main-Q workspace; decode and unrelated models retain the ordinary
  allocation.
- A full 4,096-token by 7,168-hidden stream-order oracle copied the input to
  the native staging tensor, zeroed the exact caller input, and merged the
  simulated native contribution. It preserved the caller pointer, allocated
  zero additional CUDA bytes across the call, and was bitwise exact across
  58,720,256 output bytes. Both hashes were
  `b0773585c59d4183200e137ce3a6b0da5252b41e6363b9fc2262fa53270d364e`.
  dwagon and fwuff share syntax-checked `kt_ep_wrapper.py` SHA-256
  `1d5fc4a237ec3b2402c9f74e637dba741edeb383657307c61d53ac95828c55a6`.
- Strict NUMA placement was not a blocker. At roughly half load PP0 held
  68.18 GiB private on NUMA 1 versus 1.53 GiB on NUMA 0 (97.8% selected-node)
  and PP1 held 73.05 GiB on NUMA 0 versus 0.53 GiB on NUMA 1 (99.3%).
  Later samples reached approximately 97.5% and 99.8%. The compact fwuff12
  topology should continue using hard main-rank binding; the residual remote
  pages are insignificant compared with the avoided cross-socket expert
  traffic.

## 2026-07-27 09:15 EDT — Tier36 clears first routed MoE; caller-owned FP8 indexer Q

- Tier36 target loads completed in 683.27/673.51/310.99 seconds; the complete
  fwuff DSpark draft loaded and all exact production pools allocated. The
  endpoint again advertised 1,048,576 tokens and returned deterministic
  arithmetic `42`.
- The exact fresh 32K request passed layer 0's zero-GPU in-place accumulator,
  including local AMX output and both streamed remote tiers, then entered the
  next attention layer. The fused C4 indexer attempted to allocate its ordinary
  32 MiB `[4096,64,128]` FP8 query with 16.94 MiB physically free and
  232.22 MiB reserved but fragmented. This is later than Tier35 and proves the
  zero-accumulator reuse executed successfully. No failed artifact was created
  or accepted.
- The C4 indexer's 64 MiB BF16 projection already occupies the first range of
  the persistent 512 MiB main-Q workspace. Tier37 places the fused kernel's
  32 MiB FP8 output in the immediately following disjoint byte range. The
  indexer logits consume it before the deferred main-Q projection overwrites
  the workspace. The fused RoPE/Hadamard/FP8 kernel, E4M3 dtype, scaling,
  weights, and cache path are unchanged; only its existing output pointer is
  caller-owned.
- A full 4,096-token, 64-head, 128-dimensional fused-kernel oracle used the
  exact 67,108,864-byte BF16 source and a caller FP8 view at byte offset
  67,108,864. It preserved the requested pointer and was byte-exact for both
  FP8 queries and FP32 weights. Reference and caller query hashes were
  `7a763b8720cda7e1b543fafcf2e839edef7a02cec737994eb86626ff3b4244a3`;
  both maximum differences were zero. Only the small 1 MiB FP32 weight output
  remains ordinarily allocated.
- Syntax compilation and `git diff --check` pass. Synchronized dwagon/fwuff
  hashes are `e71c30b3...` (`elementwise.py`), `696e17e0...`
  (`indexer.py`), and `8fbb61dc...` (`deepseek_v4.py`).

## 2026-07-27 09:31 EDT — Tier37 clears indexer Q; persistent initial mHC workspace

- Tier37 target loads completed in 678.78/717.97/435.77 seconds. Exact
  production pools and the target verify graph allocated, the full-context
  endpoint became ready, and deterministic arithmetic returned `42`.
- The fresh 32K request ran approximately 35 seconds before failing, versus
  roughly eight seconds at Tier36's FP8 indexer boundary. It passed the
  caller-owned FP8 indexer query and multiple full layer/chunk operations. A
  later first-rank model invocation then attempted the ordinary
  `hidden_states.unsqueeze(1).repeat(1,4,1)` initial mHC expansion, requesting
  224 MiB with 46.94 MiB physically free and 214.74 MiB reserved but
  fragmented. No failed benchmark artifact was created or accepted.
- Earlier chunks had necessarily allocated and freed the same 224 MiB tensor;
  the later failure is allocator fragmentation, not a larger live set.
  Tier38 therefore creates one persistent
  `[chunked_prefill_size,4,7168]` BF16 mHC-input workspace on the first large
  prefill, copies the broadcast embedding into it, and reuses it for later
  chunks. The workspace stays live through the first layer's mHC residual and
  is not reused until that model invocation completes. Small batches/decode
  retain the ordinary path; activation is gated by
  `SGLANG_DSV4_REUSE_MHC_INPUT_WORKSPACE=1`.
- The exact 4,096-token production oracle used a 234,881,024-byte workspace.
  Broadcast-copy output was byte-identical to `repeat`, preserved the
  persistent pointer, and allocated zero CUDA bytes on both the first measured
  fill and a second independent chunk. Both first-chunk hashes were
  `241abc66743fc9cb062abee43953a86c4a54abcd07d5e3fd52b7273a8fbdeb41`.
  Synchronized, syntax-checked `deepseek_v4.py` SHA-256 is
  `9aa9ec6a4f6cc091a5234273fd396af45962a118267ceaf49d9fd4d1195aae01`.
- fwuff logged one further hardware-corrected single-bit read during Tier37
  load at 08:19:35 CDT, physical address `0x3147be8780`, again on
  `Card01, ChnF, DIMM0`. No uncorrected event occurred. Exact gates remain
  mandatory, and the identified DIMM must be replaced before production
  stability can be claimed.

## 2026-07-27 09:52 EDT — Tier38 clears persistent mHC; shared-MLP gate/up reuse

- Tier38 target loads completed in 634.35/676.77/320.03 seconds. All three
  ranks allocated the exact 1,048,576-token/four-request pools, PP0 captured
  its target verify graph, `/v1/models` became ready, and the deterministic
  served arithmetic gate again returned exactly `42`.
- The exact fresh 32K request ran approximately 27 seconds and passed the new
  persistent 224 MiB initial-mHC workspace across later chunks. It then
  reached a shared expert's FP8-Marlin gate/up projection, whose ordinary
  `[4096,6144]` BF16 output requested 48 MiB with 46.94 MiB physically free
  and 214.43 MiB reserved but fragmented. The request failed and no benchmark
  artifact was created or accepted. PP0 and PP1 exited; the orphaned Tier38
  PP2 process on fwuff was explicitly stopped before validation and relaunch.
- Tier39 exposes the already coherent caller-output contract through
  `ColumnParallelLinear` and applies it only to large, TP1, shared-expert
  FP8-Marlin gate/up projections. The exact shared-MLP alias schedule uses no
  more than the previously proven 80 MiB Q-workspace live range: gate/up
  occupies bytes 0--50,331,648; SwiGLU occupies bytes
  58,720,256--83,886,080; and the down result reuses bytes 0--58,720,256 only
  after gate/up is dead. The activation and down output remain disjoint.
  Non-Marlin methods, TP>1, dense MLPs, batches below 512 rows, decode, and
  models without the named DSV4 workspace keep their original allocation
  behavior.
- `scripts/validate_dsv4_shared_mlp_workspace.py` loaded the real layer-0
  `w1`, `w3`, and `w2` FP8 checkpoint tensors and their UE8M0 block scales,
  repacked them through the production SM86 Marlin path, and exercised exact
  4,096-token, 7,168-hidden, 3,072-intermediate dimensions. The caller-owned
  gate/up and complete gate/up -> clamped SwiGLU -> down sequence were both
  bitwise equal to independent allocation. Both pointers were preserved,
  steady allocated memory changed by zero bytes, and the fixed workspace live
  range was 83,886,080 bytes. Gate/up SHA-256 was
  `9765a08f85de1e33b631d77960e3a96006b3101b1fe2762ac3cfa5e83faa6457`;
  final-output SHA-256 was
  `feddd0605888f11b9dcb6a56473ce4905cc2c5841a17275314ddba0f153b764a`.
  Marlin's warmed internal peak scratch was 5,373,952 bytes and returned to
  zero steady allocation.
- Syntax compilation and `git diff --check` pass. dwagon and fwuff share
  SHA-256 `904ed149...` (`linear.py`) and `484118b0...`
  (`deepseek_v2.py`). Tier39 will keep the compact fwuff12 ownership, native
  MXFP4 routed weights, GPU-hidden sidecars, DSpark, AMX prefill, AVX-512
  decode, and hard process-wide NUMA binding.

## 2026-07-27 10:14 EDT — Tier39 clears shared gate/up; bounded SM86 indexer logits

- Tier39 completed target loading in 683.33/694.17/472.22 seconds; fwuff's
  complete three-layer DSpark draft added 29.07 seconds. All three ranks
  provisioned the exact 1,048,576-token/four-request pools, PP0 captured its
  six-token target verify graph with 1.01 GiB remaining, and `/v1/models`
  advertised `max_model_len=1048576`. The deterministic served arithmetic
  gate returned exactly `42`.
- Hard process-wide NUMA policy remained active throughout this compact
  topology: every PP0 scheduler mapping reported `bind:1`, every PP1 mapping
  reported `bind:0`, and fwuff remained bound to node 0. The Linux
  `Mems_allowed_list=0-1` field only describes the cgroup's allowed nodes; it
  does not override the per-mapping bind policy. No NUMA relaxation was used.
- The exact fresh 32K request passed the new shared gate/up, clamped SwiGLU,
  and down alias schedule. It later reached an SM86 C4 indexer and failed when
  `tilelang_fp8_bf16_paged_mqa_logits` requested its ordinary 32 MiB FP32
  `[batch,max_c4_seq_len]` logits tensor with 22.94 MiB physically free and
  225.43 MiB reserved but fragmented. PP0 exited and the other ranks followed
  the broken process group. The official benchmark received a truncated
  stream, created no output artifact, and is not accepted. The orphaned PP2
  process on fwuff was explicitly stopped.
- Tier40 gives the Ampere BF16-tensor-core paged-MQA kernel a validated
  caller-output contract. The C4 indexer places logits in the disjoint suffix
  after its caller-owned FP8 query within the existing 512 MiB main-Q
  workspace, transforms each bounded row tile immediately into the persistent
  512-page selection, and reuses that suffix for the next tile. This avoids
  allocator traffic and bounds logits memory as context grows without
  changing FP8 checkpoint/cache data, BF16 MMA arithmetic, logits, or selected
  page sets. Top-k v2, non-SM86 backends, decode, small prefills, and paths
  without the named workspace retain their original behavior.
- `scripts/validate_dsv4_streamed_indexer_logits.py` exercised an exact
  32 MiB `2048 x 4096` FP32 logits geometry and forced three caller-workspace
  row tiles. Every logit was byte-exact; caller pointers were preserved; and
  every row selected exactly the same 512-page set as the ordinary full
  allocation. The logits SHA-256 was
  `6f86740972b80bc05afac5432e960f6ba67f25f17d8e57e7580725bbb6d01ffe`.
  The upstream top-k kernel does not define equal-score tie order: even two
  calls over identical full logits can permute page order, so set equality is
  the strict semantic invariant and was exact.
- Syntax compilation and `git diff --check` pass. Synchronized dwagon/fwuff
  hashes are `b5daf14e...` (`tilelang_kernel.py`), `00bff18f...`
  (`indexer.py`), and `132d2136...` (`deepseek_v4.py`). Tier40 will preserve
  the production 4,096-token chunk, full context/admission pools, strict NUMA
  placement, compact fwuff12 ownership, native MXFP4, AMX/AVX-512, and
  DSpark.

## 2026-07-27 10:33 EDT — Tier40 completes one 4K chunk; embedding gather reuse

- Tier40 provisioned the exact 1,048,576-token/four-request production pools,
  captured the PP0 target verification graph, became ready, and passed the
  deterministic served arithmetic gate with `42`. Strict process-wide NUMA
  binding remained active on PP0/PP1; the compact fwuff12 topology did not
  require the earlier full-local NUMA relaxation.
- The exact fresh 32K request completed its first full 4,096-token chunk at
  50.40 input token/s with exactly 28,672 tokens pending. This proves the
  caller-owned shared MLP and bounded SM86 indexer-logits paths across a
  complete production chunk. At the start of the second chunk, the ordinary
  BF16 embedding gather requested a fresh 56 MiB `[4096,7168]` result with
  36.94 MiB physically free and 277.08 MiB reserved but fragmented. The
  request failed before layer 0; no benchmark artifact was created or
  accepted. All Tier40 ranks are now stopped, while the three intended
  GPU-hidden expert sidecars remain active.
- Tier41 extends the unquantized embedding method and TP1
  `VocabParallelEmbedding` with a checked caller-output contract. For large
  prefills after the first chunk, `DeepseekV4Model` gathers embeddings into
  the contiguous prefix of the already persistent 512 MiB main-Q workspace.
  That workspace is idle until layer-0 attention. The result is then
  copied/broadcast into the existing persistent mHC input workspace before
  main-Q storage is reused, so no live tensors alias. The first chunk and
  models/backends without the named workspace retain the ordinary path.
- `scripts/validate_dsv4_embedding_workspace.py` loaded the actual
  `129280 x 7168` BF16 `embed.weight`, exercised an exact 4,096-token gather,
  and compared the caller-output `index_select` against `F.embedding`.
  Results were byte-exact, the caller pointer was preserved, and warmed
  allocated-memory delta was zero bytes. Output SHA-256 was
  `a6d8db003461449dcab950fc6297361caecef9a69a0a2e6ef354e44824c8f7ce`.
- Syntax compilation and `git diff --check` pass. Synchronized dwagon/fwuff
  SHA-256 values are `c6bff705...` (`unquant.py`), `b19b8e97...`
  (`vocab_parallel_embedding.py`), `c64da51f...` (`deepseek_v4.py`), and
  `04fb4ffb...` (the actual-checkpoint oracle). Tier41 keeps native MXFP4
  expert weights, AMX prefill, AVX-512 decode, DSpark, compact fwuff12
  ownership, and hard NUMA binding.
- During the Tier41 load, fwuff logged another hardware-corrected DRAM read at
  09:35:02 CDT on the already identified `Card01, ChnF, DIMM0`, physical
  address `0x1b4dc4f580`. This was a corrected scrub event; there was no
  uncorrected error or process kill. Exact post-load and served coherence
  gates remain mandatory, and this DIMM remains a production-maintenance
  blocker independent of software correctness.

## 2026-07-27 10:53 EDT — Tier41 clears embedding; protected mHC-pre suffix

- Tier41 target loads completed in 658.21/697.53/370.15 seconds. The complete
  three-layer DSpark draft added 32.35 seconds on fwuff and initialized with
  gamma 5/six target-verify tokens. All ranks allocated the exact
  1,048,576-token/four-request pools; PP0 captured its target graph and
  retained 1.01 GiB. `/v1/models` advertised `max_model_len=1048576`, and the
  deterministic served arithmetic gate returned exactly `42`.
- Live strict NUMA policy was rechecked rather than inferred from the launch
  arguments. Every one of PP0's 7,254 scheduler mappings was `bind:1`, every
  one of PP1's 6,870 mappings was `bind:0`, and their CPU masks matched those
  sockets. `Mems_allowed_list=0-1` remained only the broader cgroup allowance.
  The compact fwuff12 topology should keep hard main-process binding.
- The exact fresh 32K request completed one full 4,096-token chunk at 55.95
  input token/s with 28,672 pending, proving that chunk 2 passed the new
  caller-owned embedding gather. It then reached layer-0 TileLang `mhc_pre`,
  whose ordinary normalized `[4096,7168]` BF16 layer-input output requested
  another 56 MiB with 36.94 MiB physically free. PP0 failed, the other ranks
  were explicitly stopped, the client stream was truncated, and no benchmark
  artifact was created or accepted. InfiniBand port-counter deltas during
  the 68.53-second partial run were 1,083,106,593 transmit and 958,975,204
  receive data units, or approximately 4.33/3.84 GB because the counters use
  four-byte units.
- Tier42 gives `mhc_pre` a checked caller-output contract and places its large
  normalized output in the final 58,720,256 bytes of the persistent 512 MiB
  main-Q workspace. Q/KV projections consume that suffix as input while
  indexer and compressor intermediates are restricted to the disjoint prefix.
  Only after every input consumer completes may deferred main-Q overwrite the
  whole workspace. The protected suffix begins at byte 478,150,656; invalid
  overlap, shape, dtype, device, contiguity, or suffix placement is fatal.
- `scripts/validate_dsv4_mhc_pre_workspace.py` exercised the real production
  `4096 x 4 x 7168` geometry. Post mix, combination mix, and normalized layer
  input were all byte-exact against independent allocation; the caller pointer
  was preserved; zeroing the entire lower 478,150,656-byte prefix did not
  change the output; and live allocation fell by exactly 58,720,256 bytes.
  Output SHA-256 was
  `5c198ddbfdd05884906d5e8c5d7641c782ee3f0095f045dcc3df6c6534b09e1e`.
  Synchronized dwagon/fwuff hashes are `88873f2e...` (`mhc.py`),
  `cd5303db...` (`deepseek_v4.py`), and `63d491e9...` (oracle).
- The previously built, uninstalled projection-specific N64/N128 candidate
  was also tested while ranks were down. It remained bit-exact. A 64-row
  single-expert AMX case improved from 2.1233 to 1.8563 ms median (12.6%),
  while the contrived six-row/same-six-expert case regressed 2.3719 to
  2.5152 ms. A realistic six-token/64-expert sparse decode case was neutral:
  8.3228 versus 8.4084 ms median, with the candidate's minimum improving
  8.0755 to 7.8774 ms. Artifacts are the
  `dsv4-pro-nblock-split-{q6,q64,q6e64}-{baseline,n64}.json` family.
  The candidate is promising for prefill but is not promoted in Tier42, which
  stays a controlled memory-dataflow gate on the accepted `42a28908...` /
  `adf4d853...` runtime binaries.

## 2026-07-27 11:00 EDT — Tier42b protects mHC-pre input from every MoE writer

- A source audit before Tier42 finished loading found that the second
  per-layer `mhc_pre` call feeds the routed MoE directly from the protected
  main-Q-workspace suffix. Shared-expert MLP buffers fit in the lower
  83,886,080-byte prefix, but remote route compaction and the portable SM86
  GPU-MoE path still bounded caller-owned outputs against the complete
  workspace. The profiled routes fit below byte 478,150,656, but that was not
  a coherency invariant for arbitrary prompts.
- Tier42 was stopped only 143 seconds into loading. Both writers now derive a
  usable workspace limit from the live input tensor's storage offset whenever
  it aliases the main-Q workspace. Remote compact tensors and GPU GEMM
  intermediates may occupy only the prefix below that offset. Misaligned,
  noncontiguous, out-of-range, or already-overlapping layouts fail before a
  write instead of silently corrupting the normalized MoE input. Ordinary
  non-aliasing and decode paths retain the full-workspace or allocator path.
- Syntax checks pass and dwagon/fwuff are synchronized at SHA-256
  `33522775...` (`kt_ep_wrapper.py`) and `f02dcc9e...`
  (`v4_triton_kernels_moe.py`). Tier42b relaunched on rendezvous
  `10.44.0.1:29595` with the same native-MXFP4 binaries, compact fwuff12
  ownership, full production pools, DSpark, AMX/AVX-512 selection, and hard
  PP0/PP1 NUMA binding. No projection-N-block candidate is promoted in this
  memory/coherency gate.

## 2026-07-27 11:14 EDT — Tier42b completes exact fresh 32K

- Tier42b target loads completed in 530.65/574.26/256.94 seconds. All three
  ranks allocated the exact 1,048,576-token/four-request pools; PP0 captured
  the six-token DSpark target graph and retained 1.01 GiB. `/v1/models`
  advertises `max_model_len=1048576`, and the deterministic served arithmetic
  gate returned exactly `42`.
- Strict NUMA was verified from the running schedulers after final residency.
  PP0 had 58,403,124 pages on bound node 1 versus 2,067,445 on node 0
  (96.58% selected), with all 10,987 mappings marked `bind:1`. PP1 had
  65,501,698 pages on node 0 versus 75,407 on node 1 (99.89% selected), with
  all 11,506 mappings marked `bind:0`. CPU masks match the selected sockets.
  The broader `Mems_allowed_list=0-1` remains a cgroup allowance, not relaxed
  process policy; hard binding remains the production choice.
- The official exact fresh 32,768-input/8-output request completed end to end
  with 8/8 server-counted and retokenized output tokens and no rank failure.
  Whole-request input throughput was 202.9606 token/s, TTFT 159,118.81 ms,
  and short-gate TPOT 331.43 ms (3.02 token/s). The cold first chunk was
  56.60 token/s; subsequent PP0 chunks were 215--326 token/s, while PP1
  reached 485.43 token/s on its final measured chunk. This accepts the
  protected mHC-pre/MLP memory dataflow and full production admission, not
  the required performance targets.
- InfiniBand counters changed by 2,935,724,426 transmit and 2,448,147,480
  receive four-byte units on dwagon, mirrored by 2,448,146,976 transmit and
  2,935,723,074 receive units on fwuff: approximately 11.74/9.79 GB over the
  complete request. Artifact SHA-256 is `28a541dd...` at
  `/var/lib/exo/benchmarks/dsv4-pro-tiered42b-protected-mhc-random32k-out8-seed42.jsonl`;
  semantic artifact SHA-256 is `5512a062...`.

## 2026-07-27 11:19 EDT — A second 32K seed catches a short-tail invariant bug

- The warmed seed-43 repeat was not accepted. It completed four prefill
  launches before PP0 failed on a 4,032-token tail with
  `start=479068160, end=536870912, workspace=528482304`. The client stream
  was truncated and produced no benchmark artifact. PP1 and PP2 were stopped
  after PP0 exited; the expert sidecars remain active.
- This was a fail-closed validator error, not an observed overwrite. mHC
  correctly placed its 4,032-row output at the end of the full persistent
  536,870,912-byte workspace. The check incorrectly compared that end with
  the 528,482,304-byte live Q prefix view. The invariant now compares against
  the backing workspace extent while retaining the live output's exact
  storage, contiguity, start, and end checks. The protected start remains the
  live pointer offset, so indexer, compressor, remote, and compact-GPU writers
  remain bounded below it.
- The production mHC oracle now also checks the 4,032-token geometry:
  protected start 479,068,160, live Q extent 528,482,304, full backing extent
  536,870,912, and a real 49,414,144-byte overlap that may be overwritten only
  after mHC input consumers finish. The full 4,096-row run remains byte-exact,
  pointer-preserving, and saves exactly 58,720,256 live bytes with unchanged
  output SHA-256 `5c198dd...`.
- Python syntax checks and the extended GPU oracle pass. The synchronized
  dwagon/fwuff `deepseek_v4.py` SHA-256 is `0d458aac...`; the extended oracle
  SHA-256 is `8193d6d4...`. The next launch is Tier42c and must repeat both
  deterministic 32K seeds before this memory schedule is considered stable.

## 2026-07-27 11:41 EDT — Tier42c paused and both hosts cleanly drained

- Tier42c passed the corrected short-tail production oracle and all three
  pipeline ranks reached ready state, but it was paused at the user's request
  before either routed 32K coherency repeat ran. PP0 and PP1 stopped cleanly;
  PP2 then observed its peers close and exited. No Tier42c benchmark result is
  accepted.
- Startup exposed a separate production-admission issue: the configured
  `context_length=1048576` was advertised, but the common physical KV pool
  was scaled to 676,864 tokens. PP0 required 4,982.56 bytes per full token and
  initially admitted 736,512; PP1 required 5,411.45 bytes and admitted
  676,864; PP2 could admit 1,173,760. Reaching a real 1M context therefore
  still requires freeing roughly 2.2 GiB on PP1, most plausibly by moving
  additional GPU-resident experts off that stage without changing MXFP4
  weight precision.
- At shutdown, no DeepSeek V4 rank or local sidecar process remained on
  dwagon. The remaining fwuff expert sidecar
  `exo-dsv4-pro-tiered27-sidecar-fwuff.service` was stopped through systemd.
  A final two-host check found no matching DeepSeek/SGLang/sidecar process and
  no listener on ports 30000, 30001, 29561, 29562, 29563, or 29596. GPU
  processes belonging to an unrelated `nano_server` Docker workload on
  dwagon were intentionally left untouched.
