# Kimi K3 GPU kernel research

## Implemented SiTU gate/up MMID result

The first implementation is preserved as
`scripts/patches/kimi-k3/0004-kimi-k3-situ-mmid.patch`, against llama.cpp
`d29a524eeaf39155825d6f0ef373075fe585cb12` after patch `0003`. Its raw
SHA-256 is
`e777bd9351dad1f040fe61ee3cc3eadf80a64fb784b4a4e5daa897f95bcf73e2`
and its stable patch ID is
`b3d1eddceb226af14da08b6315d8b8be32bcea48`. The independently validated
clean implementation diff, before dropping a redundant test-run limiter
already supplied by `0003`, had SHA-256
`04c379d4e8021bdc7ce34b07264b970d55fd3cb2ace89cc3595ab7b6973fbb38`
and stable patch ID
`23d8f8c095c31f0fa8ddad791930dc0a6305edb8`.

It adds a first-class `GGML_GLU_OP_SITU`, an F32 CPU reference, a standalone
CUDA operation, and SiTU parameters in the existing MMVQ/MMVF fusion
epilogues. Pair admission is deliberately narrow: K3 dimensions
`[3584, 3072, 896]`, one decode token, 16 selected experts, no bias or scale,
and `IQ3_XXS` gate/up weights. The graph retains the original decomposition
for `IQ2_XS` and for placement backends that do not report SiTU support.

Controlled same-process RTX 3090 results were:

| Routed type/path | Time | Result |
| --- | ---: | --- |
| `IQ2_XS`, removed experimental paired fusion | 223.28 us | 8.86% slower than the 205.11 us legacy path |
| `IQ2_XS`, two MMIDs + standalone SiTU | 202.79 us | 1.93% lower latency than 206.77 us legacy |
| `IQ3_XXS`, admitted paired fusion | 261.67 us | 10.33% lower latency than 291.82 us legacy |
| `IQ3_XXS`, two MMIDs + standalone SiTU | 289.15 us | Separates activation fusion from paired-MMID gain |

The type split is explained by resources, not noise. The selected IQ2 small-K
kernel rises from 96 registers/thread and 1,536 bytes shared memory unfused to
116 registers and 3,072 bytes fused, reducing estimated register-limited warp
occupancy from 41.7% to 33.3%. The normal IQ3 kernel instead falls from 54 to
48 registers while shared memory rises only from 384 to 768 bytes, improving
estimated occupancy from 75% to 83.3%.

The removed IQ2 experiment is diagnostic evidence for non-admission; final
source contains no paired IQ2 dispatch. `GGML_CUDA_DEBUG` proves the positive
IQ3 case actually fuses exactly three
nodes, from gate `MUL_MAT_ID` through output `GLU`. The otherwise-identical
15-expert near miss and the exact IQ2 control emit no fusion trace. Distinct
gate/up correctness fixtures pass for fused IQ3, both legacy paths, the IQ2
control, and the shape near miss; three standalone SiTU fixtures cover extreme
values and both `beta`/`linear_beta` modes. Release `llama-server` and CPU/CUDA
tests build, and the final fat binary contains SM75, SM86, and SM120a cubins;
only SM86 was executed. This validation is microbenchmark, correctness-test,
and build evidence; the `0004` source has not run a full-size K3 model.

This is sound infrastructure but not a material model speedup. In
`UD-Q2_K_XL`, routed blocks 1--90 and 92 use `IQ2_XS` gate/up weights and only
block 91 uses `IQ3_XXS`. The measured saving is therefore about 30 us/token
when that Q2 block is CUDA-resident: roughly 0.00112% against the historical
0.372718-token/s Q2 mean and 0.000634% against the latest 0.210238-token/s
file-aware run. `UD-IQ2_XXS` instead has 46 `IQ1_M` and 46 `IQ2_XXS`
gate/up pairs, no `IQ3_XXS`, and its accepted placement keeps routed experts
on CPU, so this specialization gives it no benefit. Deploy the patch on both
RPC client and server because RPC operation support is not
version-negotiated.

Patch `0007` below adds scattered expert IDs; padded-row/stride coverage and
a regression benchmark of the generic SwiGLU/GeGLU fusion paths remain before
general upstreaming. A new 5090 should receive its own IQ2 tuning pass; the
Ampere result must not be extrapolated to Blackwell.

### IQ2_XS SiTU follow-up

Patch `0007-kimi-k3-iq2-situ-mmid-sequential.patch` fixes the important
negative result above without modifying frozen patch `0004`. The failed
generic IQ2 pairing kept gate and up accumulators live together. The new
exact K3 kernel instead uses four warps and four output rows per CTA in two
phases:

1. compute and reduce the gate MMID, retaining four gate scalars;
2. reuse the same accumulator and inter-warp scratch for the up MMID;
3. apply SiTU in the epilogue and write the final `[3072,16]` tensor.

On SM86 the emitted kernel uses 96 registers/thread, 1,552 bytes of shared
memory, and no stack or local memory. That restores the unfused IQ2 register
occupancy class while avoiding both projection intermediates and the
standalone activation launch; the removed generic pairing used 116 registers
and 3,072 bytes.

The completed same-binary RTX 3090 measurements were 162.25 and 162.45 us
fused, versus 170.14, 169.11, and 169.05 us with CUDA graph fusion disabled.
The approximate means are 162.35 versus 169.43 us, a 4.18% reduction. An
earlier independent bounded pair measured 161.50 versus 168.34 us, a 4.06%
reduction. The exact CPU-reference/CUDA comparison passed before test
hardening.

Final admission requires `IQ2_XS`, exact
`[3584,3072,896]` gate/up weights, a `[3584,1,1,1]` input, 16 selected
experts, a `[3072,16,1,1]` output, every unused higher dimension equal to
one, matching gate/up layout, and no bias or scale. Unsupported shapes fall
back; `GGML_CUDA_KIMI_K3_SITU_MMID=0` selects two MMIDs plus standalone
SiTU. The K3 model graph now emits first-class SiTU for supported IQ2
placements. A scattered far-ID fixture with expert-distinct quant block
scales and an IQ2 15-expert near miss were added and compile, but were not
rerun on a GPU after fwuff's 3090 was yielded to the full K3/DSpark run.

This specialization applies to the 91 `IQ2_XS` routed blocks in
`UD-Q2_K_XL`, rather than only its one IQ3 block, when those experts are
GPU-resident. Even the all-91 local ceiling is only about 0.64 ms/token, so
it is a real kernel win rather than a claim of a visible distributed-model
speedup. Patch `0007` does not add an operation or change RPC v7.0.1, but
matched client/server deployment is recommended so graph admission and
execution agree. Its SHA-256 is
`66fa3d71a75f886aed6ddcf6f4a6a37d7dbc3c93afa7c1eeb25cc1c37a34426a`;
stable patch ID:
`da7086b47540028b90b25a60c8c7a09e14a6b1dc`.

 The current work adds one K3-specific 896 experts / top-16 router specialization. NVCC emitted distinct SM75, SM86, and SM120a machine-code cubins from that source, but only SM86 was executed. Resource usage already suggests architecture tuning may eventually matter:

   Target                  Registers/thread    Local stack    Validation
  ━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SM75                                  80          224 B    Compile only
  ────────  ────────────────────────────────  ─────────────  ────────────────────────────────────
   SM86                                  72          224 B    Correctness and performance tested
  ────────  ────────────────────────────────  ─────────────  ────────────────────────────────────
   SM120a    80 without bias, 100 with bias          224 B    Compile only

  The router is not a tensor-core workload, so extensive architecture-specific tuning has limited value: even the 2.43× decode microbenchmark improvement saves at most about 1.73 ms/token across all 92 routers, and substantially less with only a few layers on local GPUs.

  ## Higher-value kernel 2: fused down-MMID and expert reduction

  Status: implemented and exported as
  `scripts/patches/kimi-k3/0006-kimi-k3-down-mmid-weighted-reduce.patch`.
  The final SM86 kernel uses four warps and two output rows per CTA, supports
  the actual `IQ1_M`, `IQ2_XS`, and `IQ3_XXS` down tensors, and performs
  router multiplication plus fixed-order top-16 reduction with explicit
  round-to-nearest operations. Final RTX 3090 latency was 66.26, 81.52, and
  96.26 microseconds respectively, versus 79.33, 91.95, and 128.70
  microseconds for the original graph. Exact admission rejects non-K3,
  non-batch-one, decorated, LoRA, warmup, and unsupported-placement cases;
  `LLAMA_KIMI_K3_FUSED_DOWN=0` forces fallback. CPU/CUDA checks passed for
  all three types and the full patch stack built SM86 release server, RPC
  server, and backend tests. Patch SHA-256:
  `3ca1b11b8073364fccea04b2e80c56c117a1ef03233bd0b712b361ef3c64be05`;
  stable patch ID: `1f829a7affbcdb35922186e5242e3e604a86aa21`.

  After SiTU, the current graph performs:

  16 down projections -> 16 vectors of length 3584
  multiply each vector by its router weight
  create 16 views
  reduce them through 15 separate add nodes

  For one decode token, the first intermediate alone is:

  3584 × 16 × 4 bytes = 229,376 bytes per layer

  Additional weighted outputs and reduction passes increase that traffic considerably.

  The fused kernel should instead calculate:

  partial[d] = sum over selected experts e:
      router_weight[e] * dot(down_weight[e, d, :], activated[e, :])

  and write one 3,584-element vector. For deterministic decode, a block can compute contributions from the 16 selected experts, reduce them in a fixed order, and perform one final store per output element. This avoids atomics, the 16-vector output, the separate weight multiplication, and 15 add launches.

  I would split this into two stages:

  1. Single-GPU graph fusion

     Recognize the existing down-MMID → router multiply → views → add chain and replace it with a CUDA fused operation. This is relatively self-contained and can be A/B tested without distributed-runtime changes.

  2. Explicit expert-partial primitive

     Introduce an operation accepting compact local expert weights, global-to-local IDs, router weights, and an ownership mask. It returns one locally reduced 3,584-element partial. That becomes the execution primitive for SmallEP-like distributed experts.

  For prompt batches, fixed-order reduction is less straightforward. The next version should group routes by nonempty expert and use a persistent grouped-MMQ kernel, followed by a segmented reduction. A naive design performing roughly 29 million FP32 atomic adds at 512 tokens is unlikely to be the best approach.

  ## How it enables SmallEP-like execution

  The intended per-layer flow is:

  dwagon:
    route once -> top-16 IDs and weights
         |
         +---- local owned experts concurrently
         |
         +---- send latent + remote IDs/weights to fwuff
                           |
  fwuff: fused gate/up SiTU -> fused down/weighted local reduction
                           |
                   return one 3584-float partial
         |
  dwagon: add local + remote partial -> continue next layer

  Per remote host and layer, approximate traffic is:

  - 3,584-float input latent: 14,336 bytes
  - IDs and weights: about 128 bytes
  - 3,584-float returned partial: 14,336 bytes

  That is about 28.1 KiB/layer or 2.53 MiB/token across 92 layers. EDR bandwidth is ample; the concern is 92 synchronization points and software/RPC latency.

  The supporting runtime plan is:

  1. Collect real per-layer routing counts and co-occurrence.
  2. Assign expert ownership to balance expected execution time—not contiguous expert IDs.
  3. Store each host’s experts compactly with a global→local map; -1 means unowned.
  4. Run both hosts concurrently through persistent RPC connections, pinned buffers, and CUDA streams.
  5. Return one partial per host, never individual expert vectors.
  6. Optionally duplicate the hottest experts on the modified 2080 Ti to avoid remote work for common routes.

  A good hardware sequence would be:

  1. Prove the expert-partial primitive between dwagon’s two 3090s.
  2. Test 3090 ↔ fwuff CPU correctness.
  3. Put the modified 2080 Ti on fwuff as a compact hot-expert or owned-expert device while reserving fwuff’s 3090 for DSpark.
  4. Compare that with installing the 2080 Ti locally, where it avoids the 92 EDR round trips.

  SiTU-MMID now covers both the one `IQ3_XXS` routed layer and the 91
  `IQ2_XS` routed layers through separate measured schedules. Down-MMID
  reduction is also complete. The next kernel work should be driven by
  measurements on the actual 2080 Ti and 5090, then by the explicit
  expert-partial primitive needed for SmallEP-like execution.
