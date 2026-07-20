# Exo Phase 1 Test Results

Last updated: 2026-07-20

This is the human-readable test ledger for the dwagon/fwuff Linux CUDA, NCCL,
and InfiniBand workstream. `FWUFFYDWAGON.md` remains the implementation plan;
the JSON receipts under `/var/lib/exo/benchmarks` are the machine-readable
source of truth. Update this file in the same commit that records each new test.

## Model throughput inventory

Rows are ordered by total model parameters, then by measured decode throughput.
Rows without a decode measurement follow the decoded rows for that model.

| Model | Benchmark run | Topology | Prefill tok/s | Decode tok/s | Evidence status |
| --- | --- | --- | ---: | ---: | --- |
| Ornith-1.0-397B | `bench_internal_agent_decode_4x512_256.jsonl#row-2` | dwagon; 2x RTX 3090; TP2/PP1; 100 CPUInfer threads across 2 NUMA nodes; AMXINT8; E4 | - | 17.422 | **IMPORTED HISTORICAL**; no Exo source or cleanup receipt |
| Ornith-1.0-397B | `bench_internal_agent_decode_4x512_256.jsonl#row-1` | same | - | 16.823 | **IMPORTED HISTORICAL**; no Exo source or cleanup receipt |
| Ornith-1.0-397B | `bench_internal_agent_prefill_4x20000_1.jsonl#rows-1-3` | same; four concurrent requests | 1,066.393 / 1,061.920 / 996.876 | - | **IMPORTED HISTORICAL** |
| Ornith-1.0-397B | `bench_internal_agent_prefill_4x20000_1.jsonl#rows-4-6` | same; four concurrent requests | 80.551 / 135.626 / 184.006 | - | **IMPORTED HISTORICAL, DEGRADED**; zero-TTFT anomaly |
| GLM-4.7 Flash 30B-A3B | `tp2-glm47flash-20260719-v1` | dwagon 1x RTX 3090 + fwuff 1x RTX 3090; MLX/NCCL TP2; dual QDR | 45.31 | 19.57 | **PASS (diagnostic)**; exact TP1 equality; not a controlled performance comparison |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260720-v9` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E44; CPU performance policy | 6.752822778* | 8.150791955 | **PASS (diagnostic)**; best reproducible native hybrid result, 4.12% above E40 V7 |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260719-v3` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E40 | 5.420292914* | 7.928729814 | **PASS (diagnostic)**; historical best, nearly reproduced by V7 under CPU performance policy |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260720-v7` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E40; CPU performance policy | 6.444521018* | 7.828144167 | **PASS (diagnostic)**; controlled V6 A/B gained 17.9% decode with no throttling |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260720-v8` | dwagon; PP2/TP1; 24/23 split; 2x RTX 3090 over NV4; full 56/56 cores; E40; CPU performance policy | 6.393873350* | 7.768722947 | **PASS (diagnostic)**; reverse-placement A/B was 0.76% slower than V7 |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v10` | PP3/TP1; v3 placement and 16/15/16 split; full 56/56/60 cores; E48; dual QDR | 6.976789422* | 7.228700241 | **PASS (diagnostic)**; best native hybrid PP3 result so far |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v8` | PP3/TP1; v3 placement and 16/16/15 split; full 56/56/60 cores; E48; dual QDR | 6.745452171* | 7.163711287 | **PASS (diagnostic)**; E48 partition control |
| GLM-4.7 Flash 30B-A3B | `glm47-kt-serving-local-dwagon-20260719-v10` | dwagon; 1x RTX 3090; TP1; 112 physical cores; 2x56 AMX pools; E4 | 6.066921877* | 7.112858950 | **ENGINEERING ONLY**; immutable measurement; final receipt failed closed after inference |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v6` | PP3/TP1; v3 placement and 16/16/15 split; full 56/56/60 cores; E32; dual QDR | 6.215342096* | 6.924257787 | **PASS (diagnostic)**; best PP3 result through v6 |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260720-v6` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E40; non-final LM head removed | 5.579114192* | 6.638956221 | **PASS (diagnostic)**; exact sanity, complete phase telemetry, no thermal throttling |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v5` | PP3/TP1; v3 placement and 16/16/15 split; full 56/56/60 cores; E16; dual QDR | 5.289909964* | 6.553438809 | **PASS (diagnostic)**; decode improved over E4; prefill-heavy output regressed |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260719-v4` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E40 | 5.793319759* | 6.515610808 | **PASS (diagnostic)**; clean replication exposed material run variance |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v3` | PP3/TP1; dwagon 2x RTX 3090 + 112 physical cores/2 AMX pools -> fwuff 1x RTX 3090 + 60 physical cores/AMX; 16/16/15 layers; E4; dual QDR | 5.766342756* | 6.290703294 | **PASS (diagnostic)**; exact semantic sanity, balanced dual-rail payload, clean unforced shutdown |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260719-v2` | dwagon; PP2/TP1; 24/23 split; 2x RTX 3090 over NV4; full 56/56 cores; E40 | 5.370222545* | 6.287111321 | **PASS (diagnostic)**; first complete local PP2 baseline |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v4` | same PP3/TP1 model and split; dwagon ranks swapped so cross-host rank 1 uses HCA-local NUMA0/GPU0; E4; dual QDR | 5.726548780* | 6.255591493 | **PASS (diagnostic)**; placement hypothesis did not improve throughput |
| GLM-4.7 Flash 30B-A3B | `glm47-pp3-diagnostic-dwagon-fwuff-20260720-v2` | same PP3/TP1 topology as v3 | 5.219545197* | 6.213798282 | **ENGINEERING ONLY**; inference valid; cleanup verifier false-negative fixed before v3; no process survived |
| GLM-4.7 Flash 30B-A3B | `glm47-pp2-local-dwagon-20260719-v5` | dwagon; PP2/TP1; 23/24 split; 2x RTX 3090 over NV4; full 56/56 cores; E42 | 6.559804369* | 5.970868191 | **PASS (diagnostic)**; prefill improved while decode regressed |
| GPT-OSS 20B | `tp2-gptoss20b-20260718-v1` | dwagon 1x RTX 3090 + fwuff 1x RTX 3090; MLX/NCCL TP2; dual QDR | 148.77 | 38.13 | **PASS (diagnostic)**; exact TP1 equality; not a controlled performance comparison |
| Llama 3.1 8B | `tp2-llama31-8b-20260718-v1` | dwagon 1x RTX 3090 + fwuff 1x RTX 3090; MLX/NCCL TP2; dual QDR | 134.15 | 26.02 | **PASS (diagnostic)**; exact TP1 equality; not a controlled performance comparison |
| Llama 3.2 3B | `tp2-llama32-3b-20260718-v1` | dwagon 1x RTX 3090 + fwuff 1x RTX 3090; MLX/NCCL TP2; dual QDR | 210.46 | 33.17 | **PASS (diagnostic)**; exact TP1 equality; not a controlled performance comparison |
| SmolLM2 135M | `tp3-smollm2-20260718-v6` | dwagon 2x RTX 3090 + fwuff 1x RTX 3090; MLX/NCCL TP3 over InfiniBand | 210.53 | 29.20 | **PASS (diagnostic)**; no per-rail PMA proof; not a controlled performance comparison |

This is an inventory, not a cross-row leaderboard: request shapes,
concurrency, quantization, and harness definitions differ. `*` The local v10
  and PP2 v2-v6 and PP3 v2-v6, v8, and v10 values are median end-to-end output rates for their
1,024-input/32-output prefill-heavy workload, not the input-token prefill
metric used by the older harness. Approximate live-log rates from failed v8/v9
transactions remain in the detailed ledger but are excluded here because no
exact measurement survived.

## Result meanings

- **PASS:** the asserted contract passed and cleanup was verified.
- **PASS (diagnostic):** correctness, transport, or lifecycle passed, but the
  receipt has `performance_comparable=false`; timing must not be used as a
  controlled hardware comparison.
- **PASS (artifact):** immutable source/build/install integrity passed without
  executing the model or making an inference-performance claim.
- **STAGE PASS:** exact model acquisition and cross-host manifest equality
  passed; staging makes no inference-performance claim.
- **EXPECTED FAIL:** a fail-closed check rejected an invalid or incomplete
  contract and exposed a defect that was subsequently fixed.
- **BLOCKED:** the requested check could not execute in the current toolchain.
- **IMPORTED HISTORICAL:** evidence recovered from another project's runtime;
  it was not run by Exo and does not inherit Exo's source, lifecycle, transport,
  integrity, or cleanup guarantees.

## Current summary

- Latest published proof-harness source: the commit containing this ledger
  update, built on `8292a300`.
- Latest completed live-validation source: the local PP2 harness at `8292a300`,
  using the exact SGLang `7fea582043df06ebdde549ee3de602a3d11b96c6`
  overlay and model contract listed below.
- Completed ladder rungs: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B,
  GPT-OSS 20B, and GLM-4.7 Flash.
- Latest GLM engineering result: local PP2 v9 retained the proven 23/24
  placement and CPU performance policy, raised GPU-resident experts from E40
  to E44, and used the head-patched runtime at the bounded 0.95 memory
  fraction. It reached 8.150791955 decode and 6.752822778 prefill-heavy output
  tok/s, gains of 4.122% and 4.784% over V7 and the best reproducible native
  hybrid result so far. Decode samples were tightly grouped at 8.114-8.153
  tok/s. Package power averaged 312.6/326.1 W during decode under 330 W PL1,
  temperatures peaked at 71/79 C, hardware throttle counts stayed at zero,
  all CPU policies were restored, and both GPUs were released. V8 established
  that reversing the partition costs about 0.76%, while V5 still shows that
  residency is not universally monotonic across policy/runtime states. All
  local PP2 results remain diagnostic while NCCL INFO and phase telemetry are
  enabled.
- Latest controlled proof result remains GLM-4.7 Flash MLX/NCCL TP=2: exact
  TP1 output equality, two-rank NCCL initialization, payload on both QDR rails,
  clean HCA health counters, instance deletion, process cleanup, and resource
  release.
- Latest kernel result: dwagon's fresh lease-harness v6 receipt passed through
  the exact overlay interpreter and claimed only `kt_bf16_amx_executed_v1`.
  Earlier native dwagon and fwuff receipts independently passed the same
  numerical gate; every run cleaned up without force.
- Latest artifact result: the official GLM-4.7 Flash BF16 snapshot now has a
  packaged contract covering every launch-relevant file, all 48 indexed
  shards, and their Hugging Face revision metadata. A leased full rehash
  verified that 62.4 GB contract on the shared read-only snapshot.
- Latest hybrid result: hybrid4 v1 is the reportable four-resident-GPU-expert
  GLM-4.7 Flash SGLang-KTransformers admission. It binds the immutable 62.4 GB
  BF16 checkpoint, AMX-BF16 kernel receipt, RTX 3090 experts 0-3 on routed
  layers 1-46, a two-GPU/two-AMX deterministic route, exact component and merge
  repeats, and real extend/decode logits. It completed and cleaned up without
  force. This is a correctness proof, not a performance result.
- Hybrid1 v2 confirms the hybrid1 v1 diagnosis: the corrected probe gives each
  in-place Triton fused-MoE invocation a fresh clone of the pristine dispatch
  input, and both combined, CPU, and GPU repeats were bitwise identical without
  relaxing admission tolerances.
- The instrumentation-free GLM-4.7 serving baseline now has a preserved v10
  full-CPU warm measurement: median decode output was 7.112858950 tok/s with
  cores 0-111, both NUMA nodes, two 56-thread AMX pools, one RTX 3090, and four
  resident GPU experts. The authoritative receipt failed closed only during a
  post-cleanup tuple/list comparison; commit `6040820a` fixes that defect. V10
  remains engineering evidence, and a fresh normal receipt is still required.
- Historical Ornith AMXINT8 conversion and serving receipts were recovered and
  hashed below. They inform the GLM hybrid-runtime work but are not Exo tests.
- Commit `65a04353` routes the disposable backend child's OS-level stdout to the
  validator's stderr while retaining the sealed memfd evidence channel. The
  parent now reserves stdout for one JSON control response, and the harness keeps
  rejecting prefixed, suffixed, or multiple JSON records instead of parsing the
  last line. The broad focused slice passes 749 tests; repository-wide
  Basedpyright and Ruff pass, and changed Python files are formatted.
- Next work: iterate PP3 placement and expert residency, then compare mixed
  TP/PP layouts. In parallel, use native SGLang EP on a smaller MoE as the
  first true expert-sharding proof; KTransformers `kt_ep` is local routing and
  is not distributed expert parallelism. Keep admission, stage-kernel, and
  serving-performance receipts separate.

## Automated validation

These suites overlap. Their passing counts must not be summed into a unique
test total.

| Scope | Result | Evidence |
| --- | --- | --- |
| Corrected SGLang launch, snapshot, and preflight slice | **PASS** | 94 passed |
| GLM-4.7 Flash target profile, preflight, snapshot verifier, and local process supervisor | **PASS** | 148 passed on 2026-07-19, including 25 supervisor lifecycle tests; inherited NCCL/SGLang isolation and shutdown/error-race receipt regressions included |
| Reproducible GLM-4.7 SGLang-KTransformers source integration | **PASS** | Exact clean-base replay produced SGLang `42504e59810130460fc24fdd17ef534cb8278a4b` and KTransformers `6e0a4480936effa7bf0ece429f78a00b29932bec`. The GLM Lite constructor now initializes inherited non-hash/shared-expert state and rejects hash-mode configs; old-source regression tests fail at the missing state while the new source passes. |
| Pinned dwagon GLM-4.7 native runtime build | **PASS (artifact)** | Build ID `ea9de367cfebe35dc6afe51c1bda5e7daf35d6f51114f404dfebd63d055eec20`; receipt SHA-256 `1f304ea5667445cdd66e9c6938e78682b7119a6c3c3b946cf42ce816a0639542`; all three exact CUDA/AMX runtime wheels were built from the admitted revisions |
| Immutable dwagon runtime overlay | **PASS (artifact)** | Install ID `b275ec08c01fdce2cd6adb64f10b35f0a5bda12af20899a1fac1de42aa29ecd3`; receipt SHA-256 `7a2b6fd01efb7f889f01ae2a47c66c2625c4162a93373ea116d3964fd405a5f5`; deterministic preflight reconstructed the ID and the old 9.9 GB base runtime remains untouched |
| Prior fwuff GLM-4.7 runtime and overlay | **SUPERSEDED (artifact)** | Build `e21ef087b1c50cf961339e1bd1a2e1a3f60047579f811385614297de6a2abfc9` and overlay `82d20634f743ed87ae9cc71f2b7f4936d9451363db1ca46a207218f22de51ef8` remain immutable evidence for SGLang `41d4d300...`; the current two-host run uses the newer runtime below. |
| Current PP3 native runtimes and overlays | **PASS (artifact)** | Dwagon build `c9c150d940bd2314eb2a9607bca37a0973ca70743690961f56e9c0d9a0d98d25` / overlay `32aa384b06c33fbc562462aeb4b9a0f2da1beb0a469ea225ea24671eca793e95`; fwuff build `386fe038bb32f834306d7992001ffb3b239c0cb81027c77fc4faee4cd982f63a` / overlay `563dba484c323139f05b7853384dc565d6f6f8282f38e365327fa1bb3d9782aa`. Both bind SGLang `3721d710102456b6bf849122e781129dc3f7d9c6` and KTransformers `f9ca69648421f5774215c4da9cf711dccf54f49e`; pure-source hashes match across hosts and each host has its own native KTransformers kernel. Install-receipt SHA-256 values are `85d9f67b557113f1b19ee85bf2c8427f6f5988700b87cebbd86110f69f36b7b6` and `9cae56658bd9d1f631b5e3bfd053cba086be4ca52600cba236f026f4b3b7ba78`. |
| Head-patched dwagon runtime and overlay | **PASS (artifact)** | Build `cdc759d2a86b8a01c09a3aa5fe2960c45bf891184260f3036c207206c5750153` and overlay `14b9e8f8577d812ea954cffa0d2833b9535e589a1fb8fc606c20e2edd3e00455` bind SGLang `7fea582043df06ebdde549ee3de602a3d11b96c6` and KTransformers `f9ca69648421f5774215c4da9cf711dccf54f49e`. The build-receipt SHA-256 is `053fa7158b83030a4d8436b24ccb764fd0a1af4756b50d55308a1da27f9afb09`; the install-receipt SHA-256 is `77ddc2f4c4f84b628a81d0d05868e0f973441f023da385483384abde2b0aa924`. |
| GLM PP LM-head source patch | **PASS (software)** | The deterministic mail patch creates SGLang `7fea5820...`, allocates `ParallelLMHead` only on the final PP rank, and saves exactly 634,388,480 bytes (605 MiB) on each non-final TP1 rank. The source/launch suite passed 101 tests; the intermediate-revision resume regression passed all 11 source-preparation tests. |
| Local PP2 phase-boundary telemetry | **PASS (software)** | Eight non-polling, non-fatal snapshots cover launch, readiness, sanity, both workload warmup/sample boundaries, and cleanup. The focused PP2/client suite passed 53 tests; targeted Ruff, formatting, and `git diff --check` passed. |
| PP3 stage CUDA/AMX kernel validation | **PASS** | `/var/lib/exo/benchmarks/glm47-pp3-runtime-validation-dwagon-fwuff-20260720-v1`; all three stage-local validators used their exact overlay interpreter and full 56/56/60-core affinity. Dwagon NUMA0/GPU0, dwagon NUMA1/GPU1, and fwuff NUMA0/GPU0 each derived only `kt_bf16_amx_executed_v1`; receipt SHA-256 values are `f4bc0230...`, `43ff516a...`, and `a1f8abbc...`. |
| GLM-4.7 PP3 engineering harness | **PASS (software)** | 260 harness/shared-supervisor tests pass for raw three-rank readiness/server info, semantic sanity, token workloads, NCCL logs, HCA deltas, ownership verification, and cleanup. The 14 harness-specific tests include real descendants with sanitized environments, fail-closed `/proc` read errors, and local/remote startup-handoff failures. |
| GLM-4.7 local PP2 engineering harness | **PASS (software)** | 88 focused PP2 and launch-contract tests pass. The harness binds the exact runtime/model receipts before launch, assigns NUMA0/GPU0 and NUMA1/GPU1 all 56 physical cores each, admits bounded 0.80-0.95 static-memory fractions, runs exact sanity plus canonical 1024/32 and 128/128 workloads, and retains a mode-0600 ownership journal until verified cleanup. |
| Leased two-host CUDA/AMX kernel validation | **PASS (historical)** | Dwagon v4 and fwuff v1 independently passed exact provenance, SM86 BF16 CUDA math, AMX-BF16 qlen 1/16, and the bidirectional non-default CUDA-stream bridge for the superseded source; each claimed only `kt_bf16_amx_executed_v1` and cleaned up without force |
| Official GLM-4.7 Flash BF16 model contract | **PASS (artifact)** | The packaged contract binds 54 launch-relevant files, 48 indexed shards totaling 62,444,175,504 bytes, exact Hugging Face metadata, tokenizer/template inputs, and absence of executable remote-code files; leased live verification returned 0 and cleaned up unforced |
| Focused GLM-4.7 source, build, overlay, model-contract, validator, launch, and preflight suite | **PASS** | 277 tests passed on 2026-07-19, including exact packaged-contract pinning, the repaired required-profile launch-plan fixture, isolated validator import, and canonical/raw PyTorch GPU UUID coverage |
| GLM Lite inherited-state fix and current source admission | **PASS** | Clean replay passed 20 focused SGLang constructor/forward, coverage, registry, and loader tests; Exo source/build/install/process-spec contracts passed 134 tests; affected model-contract/preflight tests passed 149. These suites overlap. |
| Dedicated leased GLM-4.7 live-validation harness | **PASS (software)** | 37 focused tests cover immutable preparation, exact runtime/GPU/HCA binding, kernel/model result semantics, descriptor-anchored scratch cleanup, delegated cgroup-v2 placement, pre-exec attachment, identity replacement, and fail-closed process/cgroup cleanup; strict targeted Basedpyright and Ruff pass |
| Lease plus GLM containment contract slice | **PASS (software)** | 103 tests validate the optional static containment contract, exact systemd invocation/UID/owner-token leaf binding, one-way runtime binding, immutable evidence, final result reconciliation, and backward compatibility for leases without containment |
| Broad GLM producer/consumer and lease regression slice | **PASS** | 672 tests passed on 2026-07-19 after cgroup containment and lease-evidence binding were added |
| GLM v3 receipt-failure regression slice | **PASS** | 275 tests passed on 2026-07-19 across the harness, lease, model validator, live bindings, kernel receipt, model receipt, and receipt integration; strict targeted Basedpyright reported 0 errors/warnings and repository-wide Ruff passed |
| Portable-Python sealed-evidence regression slice | **PASS** | 325 harness/lease/backend/live/receipt/reference/trace/model tests passed on 2026-07-19; repository-wide Ruff and touched formatting passed. A trivial disposable child also passed under the exact immutable CPython 3.12.13 overlay with libc memfd creation, all four required seals, and canonical evidence SHA-256 `af4daf371da4cad51875b9db9f1ed82c20c4a6f61dbba07518acb31451d2cf48`. |
| GLM v6 trace-evidence correction | **PASS** | Commit `53e18bdd` captures non-null `next_token_logits` for `MODEL_FORWARD`, captures non-null wrapper `hidden_states`, preserves exact nested backend errors, and admits the exact current v6 kernel receipt while auditing v4 as superseded. The broad focused slice passed 744 tests; repository-wide Basedpyright reported 0 errors/warnings, repository-wide Ruff passed, and all six changed Python files passed `ruff format --check`. |
| GLM runtime diagnostic/protocol isolation | **PASS** | Commit `65a04353` redirects the disposable backend child's inherited file descriptor 1 to the parent validator's stderr while sealed memfd remains the evidence transport. Regressions prove native `os.write(1, ...)` and fd2 diagnostics cannot pollute stdout, a clean single JSON response is accepted, and prefix/suffix/two-record contamination remains rejected. The focused validator/live/harness slice passed 128 tests and the broad focused slice passed 749 tests; repository-wide Basedpyright reported 0 errors/warnings/notes, repository-wide Ruff passed, the three changed Python files passed `ruff format --check`, and `git diff --check` passed. |
| GLM hybrid repeat-input isolation | **PASS (software)** | The layer-one validator now clones a pristine device input into a fresh dispatch for every invocation, synchronizes each result, and snapshots the combined output immediately. An input-mutating fake reproduces the pinned Triton runner's in-place contract and verifies both repeats start from `0.5`. All 267 GLM validation tests and the 1,200-test scripts/SGLang-KT regression slice passed; repository-wide Basedpyright and Ruff passed, and changed Python formatting is clean. Exact repeat admission remains unchanged. |
| GLM-4.7 BF16 CPU-control model admission | **PASS** | V8 completed normally with a bound model receipt, `model_checkpoint_verified=true`, `reportable=true`, and clean unforced cleanup. It derives all seven required wrapper, short-forward, NUMA/affinity, AMX, and CPU-routed-expert capabilities. `performance_comparable=false`: this validates the launch contract and real model execution, not serving throughput. |
| GLM-4.7 BF16 one-resident hybrid model admission | **PASS** | Hybrid1 v2 completed normally with a bound 30,474-byte model receipt, `model_checkpoint_verified=true`, `reportable=true`, and clean unforced cleanup. Its six common capabilities plus `kt_bf16_cpu_gpu_hybrid_executed_v1` bind exact layers 1-46, AMX-BF16 CPU experts, one RTX 3090 expert per layer, deterministic component/merged evidence, and real extend/decode. `performance_comparable=false`: this is a correctness/admission proof. |
| GLM-4.7 BF16 four-resident hybrid model admission | **PASS** | Hybrid4 v1 completed normally with a bound 30,752-byte model receipt, `model_checkpoint_verified=true`, `reportable=true`, and clean unforced cleanup. The independently loaded receipt derives the same narrow seven mixed-route capabilities while binding resident experts 0-3 on every routed layer, a GPU 0/1 plus AMX CPU 4/5 oracle, exact repeats/merge, and real extend/decode. `performance_comparable=false`: stage timings are not throughput data. |
| SGLang-KTransformers process-wide NUMA binding | **PASS (software)** | The local process supervisor now renders one exact `/usr/bin/numactl --physcpubind <cores> --membind <nodes> ...` command, so the configured memory-node policy applies to the whole SGLang server as well as the KTransformers worker-pool arguments. All 29 focused supervisor lifecycle tests pass. A fresh model-execution receipt is required before this changed launch source is admitted on hardware. |
| GLM-4.7 instrumentation-free serving baseline contract | **PASS (software)** | The dedicated profile preserves the pinned BF16/SM86/PP1 model and runtime contract while removing diagnostic timing/distribution hooks and inherited tuner variables. It disables radix caching and CUDA graphs for the first matched control, requires 1-63 GPU-resident experts, and keeps the prior hybrid4 canonical process-spec digest unchanged. The focused launch, generator, preflight, and model-validator slice passed 218 tests; targeted strict Basedpyright reported 0 errors, and Ruff/format checks passed. No serving or performance claim is made. |
| Contained GLM-4.7 serving-baseline admission phase | **PASS** | The existing lease, immutable deployment, delegated cgroup, NUMA/GPU/HCA preflight, kernel validation, model validation, and ownership-safe cleanup harness selected the exact `serving_baseline` launch mode in the clean v1 hardware run. It requires 1-4 resident GPU experts and remains admission-only with `performance_comparable=false`. The harness/generator slice passed 96 tests; strict source Basedpyright, Ruff, and formatting pass. |
| GLM-4.7 BF16 four-resident serving-profile admission | **PASS** | `/var/lib/exo/benchmarks/glm47-kt-serving-admission4-dwagon-20260719-v1` binds clean source `ca468528`, canonical process spec `dd08b778...`, the 62,444,175,504-byte model, four resident experts on layers 1-46, AMX CPU experts 4/5, GPU experts 0/1, exact component/merge repeats, and real extend/decode. All child stages returned 0, outer cleanup was unforced, and the lease, lock, GPU, ports, unit/cgroup, scratch, and owned PIDs were clear. `performance_comparable=false`: no server throughput was measured. |
| GLM-4.7 packaged-contract wheel inclusion | **PASS (artifact)** | `uv build --wheel` produced `exo-0.3.70-py3-none-any.whl`; its package contains the exact 16,218-byte `exo/worker/sglang_kt/manifests/glm47_flash_bf16_7dd20894.json` resource |
| Changed GLM-4.7 Flash Python files, strict targeted type checks and Ruff | **PASS** | Three targeted Basedpyright configurations reported 0 errors; repository-wide `ruff check` passed; all 16 changed Python files passed `ruff format --check` on 2026-07-19 |
| Repository-wide Basedpyright | **PASS** | `uv run --no-sync basedpyright` reported 0 errors, 0 warnings, and 0 notes after synchronizing the locked workspace environment |
| Repository-wide pytest collection | **BLOCKED** | A fresh `uv run pytest` on 2026-07-19 collected 1,980 tests, deselected 193, and selected 1,787, but stopped at the same 10 pre-existing collection errors: nine duplicate `tests.*` package imports under `src/exo/download/tests` and one image/MFlux import without the optional Torch runtime |
| Nix formatting | **BLOCKED** | `nix` is not installed on dwagon; Ruff formatting is clean for every changed Python file |
| Scheduler, supervisor, compute-resource lifecycle, and SGLang slice | **PASS** | 138 passed |
| TP2 harness and reciprocal-topology readiness | **PASS** | 260 passed at `e38ba19d` |
| Interprocess flush, channels, runner ordering, supervisor, and planner | **PASS** | 59 passed at `96230c4b` |
| HCA counter and dual-rail evidence suite | **PASS** | 224 passed |
| Model-revision API suite | **PASS** | 5 passed |
| Narrowed Exo-process matcher | **PASS** | 153 harness tests at `0c58070e` |
| Complete pinned-snapshot proof | **PASS** | 163 focused harness tests at `55a2199e` |
| Closed-channel behavior and shutdown race | **PASS** | 11 channel tests; the race regression passed 20 repeated runs at `35372e6c` |
| GPT-OSS TP1 oracle admission | **PASS** | 40 tests at `8bee9f93` |
| Chat-template date pin, including `strftime_now()` | **PASS** | 5 tests at `8bee9f93` |
| GPT-OSS exact revision card | **PASS** | 1 focused test at `8bee9f93` |
| GLM-4.7 Flash TP1/TP2 proof contract | **PASS** | 293 focused tests at `57a57e12`; full fake five-request GLM lifecycle passed, GPT request digest remained `1359d4d3e31a1b4630dbe3c7c5e3c4a2b0d9797b31df79a5f6b1edeaa9d8bcc7` |
| Targeted strict Basedpyright configurations for changed slices | **PASS** | 0 errors |
| Ruff checks and formatting for changed slices | **PASS** | Available local checks passed |
| Dashboard production build | **PASS** | `npm run build` passed |
| Dashboard static check | **KNOWN BASELINE** | Improved from 19 to 15 errors; 15 errors and 6 warnings remain, all pre-existing |
| Broad non-image Python baseline | **KNOWN BASELINE** | 459 passed, 5 skipped, plus the unchanged stale Rust-binding failure |
| Fresh full-repository pytest collection | **BLOCKED** | Installing all workspace packages removes the old `exo_tools` blocker, but collection still stops at the 10 unrelated package-name/optional-Torch errors above |
| Required type and format gates | **PARTIAL** | Required Basedpyright and Ruff gates pass. Nix is not installed on dwagon; all five changed Python files pass `ruff format --check`. |

### Host cgroup-v2 containment probes

These probes used only short-lived sleeping processes. They did not touch the
model, GPUs, AMX, storage, InfiniBand, or the benchmark lease and make no
performance claim.

| Probe | Result | Evidence and lesson |
| --- | --- | --- |
| Delegated service layout | **PASS (diagnostic)** | systemd 260 created `exo-containment-probe.service` at `/system.slice/exo-containment-probe.service/supervisor`, supplied invocation ID `f17ba218fa9a44eaac3ecb11e82e4064`, kept the unit root empty, and exposed all cgroup-v2 controllers under `Delegate=yes`. |
| Empty-leaf `cgroup.kill` | **PASS (diagnostic)** | Writing `1` to `cgroup.kill` on an empty `0700` sibling leaf returned successfully, `cgroup.events` remained `populated 0`, and the leaf was removed. |
| First harness containment invocation | **EXPECTED FAIL (setup)** | The transient service omitted `WorkingDirectory`, so the ad hoc `python -c` probe could not import `scripts` and exited before creating a validator leaf. The corrected probe set `/root/exo`; generated benchmark commands use absolute immutable script paths and do not depend on this ad hoc import setup. |
| Detached-descendant cleanup | **PASS (diagnostic)** | The real harness helpers placed outer PID 3461326 and detached `setsid` PID 3461327 in `/sys/fs/cgroup/system.slice/exo-glm47-b9ccc8603b8f2c76ec5a1f1b972eb733.service/validators-df6c87dd68918eca906ef969c5755d1d`; `cgroup.kill` removed both, `cleanup_owned_cgroup` returned true, and the exact leaf disappeared. |
| Sequential leaf reuse | **PASS (diagnostic)** | The same delegated validator leaf accepted and killed two successive pre-exec-attached processes (PIDs 3480737 and 3480738), then final cleanup removed the leaf. This proves one leaf can safely contain the generator, kernel validator, and model validator in sequence. |
| Unit collection | **PASS (diagnostic)** | `exo-containment-probe.service`, `exo-cgroup-empty-kill-probe.service`, and `exo-glm47-b9ccc8603b8f2c76ec5a1f1b972eb733.service` all report `LoadState=not-found`; all three unit cgroup paths are absent. |

## Live interconnect and lifecycle tests

| Test | Result | Key result or receipt |
| --- | --- | --- |
| MLX/NCCL port 1, port 2, and merged-port smoke | **PASS (diagnostic)** | `all_sum` and `all_gather` passed over forced `NET/IB`; 64 MiB all-reduce measured 18.91, 19.18, and 19.13 Gb/s |
| Exo end-to-end namespace `fwuffydwagon-nccl-poc-v1` | **PASS (historical)** | Two CUDA nodes, TP=2 load/warmup, forced merged IB, 44-token chat in 0.37 s |
| Exo end-to-end namespace `fwuffydwagon-nccl-poc-v2` | **PASS (historical)** | Bootstrap-derived ConnectX-3 GIN disable, merged IB, 44-token chat in 0.47 s |
| Live shutdown preflight | **EXPECTED FAIL** | `/var/lib/exo/benchmarks/exo-nccl-shutdown-preflight-20260718T0450`; rejected unowned fwuff ComfyUI PID 5334 and launched nothing |
| QDR x8/x8 baseline v1 | **EXPECTED FAIL** | Reserved ports overlapped the Linux ephemeral range; no valid traffic result |
| QDR x8/x8 baseline v2 | **PASS (diagnostic)** | 31.74 Gb/s port-1 row only; old strict post-case probe encountered expected TCP `TIME_WAIT` |
| QDR x8/x8 baseline v3 | **PASS** | `/var/lib/exo/benchmarks/ib-qdr-x8x8-20260718-v3`; 31.74 Gb/s standalone rails and 32.56 Gb/s native dual-port aggregate |
| Independent per-rail QDR v4 | **PASS** | `/var/lib/exo/benchmarks/ib-qdr-x8x8-independent-20260718-v4`; 32.56 Gb/s concurrent aggregate, zero selected HCA health deltas; result SHA `7598faedf6e5343727e65352f00c17241ea5247891063760ba3511b076c538fd` |

## Live model tests

| Model and test | Result | Key evidence |
| --- | --- | --- |
| Llama 3.2 1B TP=2 | **PASS (historical)** | Both ranks ready, warmups complete, deterministic chat valid; predates strict receipt contract |
| SmolLM2 135M exact stage | **STAGE PASS** | `/var/lib/exo/benchmarks/smollm2-stage-20260718-v1` |
| SmolLM2 135M TP1 oracle | **PASS** | `/var/lib/exo/benchmarks/tp1-smollm2-20260718-v1`; 42 input, 32 output tokens; completion SHA `3f466ee4633b7a26654a26badb971ab41a08908ed4adf4ad9a5277a498c596ca` |
| SmolLM2 135M TP=3 v1-v5 | **EXPECTED FAIL / diagnostic** | See the failure ledger below; each attempt isolated one harness or lifecycle defect |
| SmolLM2 135M TP=3 v6 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/tp3-smollm2-20260718-v6`; all three ranks initialized, all five generations matched TP1; mean latency 1.546 s, prefill 210.53 tok/s, decode 29.20 tok/s; no per-rail PMA proof in this historical receipt |
| Llama 3.2 3B exact stage | **STAGE PASS** | `/var/lib/exo/benchmarks/llama32-3b-stage-20260718-v1`; 18 identical files; model-manifest SHA `28a0a458ee0fb3cbf3516347d90fc9137fb5451ca9e4ab426c1100f061348079` |
| Llama 3.2 3B TP1 oracle | **PASS** | `/var/lib/exo/benchmarks/tp1-llama32-3b-20260718-v1`; completion SHA `80eecfdc6ec2fef25a1157fce542b16d004fecb3ccb6653641dcbab71b8a954f` |
| Llama 3.2 3B TP=2 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/tp2-llama32-3b-20260718-v1`; exact TP1 equality, 1.406 s mean, 210.46/33.17 prefill/decode tok/s, 202,940,352 and 202,929,084 PMA bytes on the two rails |
| Llama 3.1 8B exact stage | **STAGE PASS** | `/var/lib/exo/benchmarks/llama31-8b-stage-20260718-v1`; model-manifest SHA `39df1f70f021e6f2852549a1d43c8120eb305993a796d077d0d3ed396a63cc7d` |
| Llama 3.1 8B TP1 oracle | **PASS** | `/var/lib/exo/benchmarks/tp1-llama31-8b-20260718-v1`; completion SHA `09b309bfc50e29dd9ceb2f192dc1f798150b0867530fc8fe6a0b919fcb87fe21` |
| Llama 3.1 8B TP=2 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/tp2-llama31-8b-20260718-v1`; exact TP1 equality, 1.919 s mean, 134.15/26.02 prefill/decode tok/s, 306,863,856 and 306,854,072 PMA bytes on the two rails |
| GPT-OSS 20B exact stage | **STAGE PASS** | `/var/lib/exo/benchmarks/gptoss20b-stage-20260718-v1`; 27 identical files, 3 weight shards, 12,076,119,168 indexed bytes; model-manifest SHA `017c642a3c72c47b74bb8720e7aa4b0c5cb8802c366345da8356929ab411e03a` |
| GPT-OSS 20B TP1 oracle | **PASS** | `/var/lib/exo/benchmarks/tp1-gptoss20b-20260718-v1`; three identical 76-input/32-output-token generations; completion SHA `48ef860d9ff0fc9aea399357da450eb3bcb6495825f25043254f4ba4f22dd9d1` |
| GPT-OSS 20B TP=2 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/tp2-gptoss20b-20260718-v1`; exact TP1 equality, 1.652 s mean, 148.77/38.13 prefill/decode tok/s, 208,864,908 and 204,248,888 matched PMA bytes on the two rails |
| GLM-4.7 Flash exact stage | **STAGE PASS** | `/var/lib/exo/benchmarks/glm47flash-stage-20260719-v1`; 26 identical files, 4 weight shards, 16,852,202,496 indexed bytes; model-manifest SHA `c9de2620a4cd99025abfc4758555637f3a8dbedb8cb69daf198be5edb3e6d64e` |
| GLM-4.7 Flash TP1 oracle | **PASS** | `/var/lib/exo/benchmarks/tp1-glm47flash-20260719-v1`; three identical non-thinking 16-input/32-output-token generations; completion SHA `de1349c105ffe29ab10b68492986aa6c081672d045b02d474570fbf5bda3a40d` |
| GLM-4.7 Flash TP=2 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/tp2-glm47flash-20260719-v1`; exact TP1 equality, 2.374 s mean, 45.31/19.57 prefill/decode tok/s, 199,084,700 and 199,080,292 matched PMA bytes on the two rails; reported peak memory 9,380,021,417 bytes |
| GLM-4.7 Flash local PP2 v1 | **EXPECTED FAIL (setup)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260719-v1`; both stages loaded E40 and initialized NUMA-local 56-thread AMX pools plus NCCL P2P/IPC, then SGLang rejected `mem_fraction_static=0.8` because the 19.13 GB weight allocation left less than its reserve. Cleanup was ownership-verified and complete. Result SHA `5ea458018b6beae7ab75f043f15ff07582c23ab237431d686200afe8bfe38e38`. |
| GLM-4.7 Flash local PP2 v2 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260719-v2`; the bounded 0.90 memory-fraction fix admitted E40 with a 24/23 split, exact `EXO_SANITY_OK`, 5.370222545 prefill-heavy output tok/s, 6.287111321 decode tok/s, local P2P/IPC, and clean verified shutdown. Result SHA `a48011a9a73ea28180938b1aa1f81737b65f0b06c4900b8c7352c05fad6c27ae`. |
| GLM-4.7 Flash local PP2 v3 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260719-v3`; reversed 23/24 split, exact sanity, 5.420292914 prefill-heavy output tok/s, and 7.928729814 decode tok/s with all three decode samples between 7.775 and 7.974. Cleanup was complete. Result SHA `671704e0de3702cea8a2e8223a84192c299cc68f1a68b9b31ae26176dafc8cd7`. |
| GLM-4.7 Flash local PP2 v4 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260719-v4`; clean 23/24 replication reached 5.793319759 prefill-heavy output tok/s and 6.515610808 decode tok/s, proving the v3 peak is not yet reproducible. Exact sanity, local P2P/IPC, and verified cleanup passed. Result SHA `595428c55139d171c4766742f6958e95819c87d1505abbe6c037c065b4167951`. |
| GLM-4.7 Flash local PP2 v5 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260719-v5`; E42 at memory fraction 0.92 used 19.13/19.99 GB for weights, passed exact sanity, reached 6.559804369 prefill-heavy output tok/s and 5.970868191 decode tok/s, and cleaned up completely. Result SHA `ed6d74b8c565aef947250087561c93d7c6f487bc8ac965cde41c880312a6160f`. |
| GLM-4.7 Flash local PP2 v6 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260720-v6`; the exact SGLang `7fea5820` runtime omitted the 605 MiB LM head from non-final rank 0, retained E40 and the 23/24 split, passed exact `EXO_SANITY_OK`, reached 5.579114192 prefill-heavy output tok/s and 6.638956221 decode tok/s, and cleaned up completely. All eight non-polling telemetry boundaries were complete; the three decode samples were 6.634080338/6.645821799/6.638956221 tok/s, CPU package temperatures peaked at 78/82 C, GPU clocks held 1695 MHz, and post-run hardware throttle counters were zero. Result SHA `39079b44d2bbcf70efd29b8464eb39e827d40950b0d7ecfdbc894789bfb5c2cf`. |
| GLM-4.7 Flash local PP2 v7 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260720-v7`; an identical V6 A/B temporarily selected the `performance` governor/EPP across all online CPU policies and restored the verified original state on shell exit. Exact sanity passed; 6.444521018 prefill-heavy output tok/s and 7.828144167 decode tok/s improved 15.512% and 17.912%, respectively. Decode samples were 7.806325808/7.866586537/7.828144167 tok/s, decode package power averaged 318.8/323.1 W under 330 W PL1, temperatures peaked at 72/79 C, throttle counters remained zero, and cleanup was complete. Result SHA `876ceae85e57feac3a2aa3c6cd9da693671ff69324b6b08031be8b60bd9b02ae`. |
| GLM-4.7 Flash local PP2 v8 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260720-v8`; V7's controlled placement A/B reversed only the partition to 24/23. Exact sanity passed; 6.393873350 prefill-heavy output tok/s and 7.768722947 decode tok/s were 0.786% and 0.759% below V7. Decode samples were 7.810123418/7.768722947/7.674028049 tok/s, decode package power averaged 321.0/318.1 W under 330 W PL1, temperatures peaked at 86/79 C, throttle counters remained zero, all CPU policies were restored, both GPUs released to 1 MiB, and cleanup was complete. Receipt content SHA `a26104bbaa02961abbf8fe2ec1db3499cfda35051891d00f23d39217304b3126`; file SHA `4d0b06a2ec01318672dc7025782bacd4cec15a89374b5d9b318ad39af26754b8`. |
| GLM-4.7 Flash local PP2 v9 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp2-local-dwagon-20260720-v9`; the head-patched 23/24 runtime raised residency to E44 at the bounded 0.95 memory fraction under the CPU performance policy. Exact sanity passed; 6.752822778 prefill-heavy output tok/s and 8.150791955 decode tok/s improved 4.784% and 4.122% over E40 V7 and 2.801% over the historical V3 decode peak. Decode samples were 8.114369977/8.150791955/8.153355273 tok/s; decode package power averaged 312.6/326.1 W under 330 W PL1, temperatures peaked at 71/79 C, hardware throttle counts remained zero, all CPU policies restored, both GPUs released to 1 MiB, and cleanup was complete. Receipt content SHA `07f54c889e14095646883c60e29c34ec66b1898bcabff18ccb21a2d4fb47a524`; file SHA `4396e91da879ab4fe952b7618cd04d633de863cb561b301703d6708e77ecfd38`. |
| GLM-4.7 Flash PP3 v1 | **EXPECTED FAIL (setup)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v1`; Gloo resolved an IPv6 dwagon endpoint against IPv4 fwuff and failed before model load. Per-host `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` fixed the family mismatch in place; cleanup passed. Result SHA `5003b64ee47c6a7bd0f9522ac9a0ff0944cd4903c9352862b7ae90aeba68f2`. |
| GLM-4.7 Flash PP3 v2 | **ENGINEERING ONLY** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v2`; exact semantic sanity and all workloads completed at 5.219545197 prefill-heavy output tok/s and 6.213798282 decode tok/s. A child-environment cleanup assumption caused a false-negative receipt after all processes were gone. Result SHA `ce2e6ce4dc5e0917ad4b27e208acbc1f1a247e46584e7d056a0d58f92fbb550f`. |
| GLM-4.7 Flash PP3 v3 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v3`; exact `EXO_SANITY_OK`, 5.766342756 prefill-heavy output tok/s, 6.290703294 decode tok/s, balanced 27,223,772/27,217,344-byte dual-rail payload, zero HCA health deltas, and three ownership-verified unforced rank cleanups. Result SHA `4b8399bd2b94307e7fffd41c6461ef99c83530b41b870dfce33d767c9d52d0e5`. |
| GLM-4.7 Flash PP3 v4 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v4`; HCA-local cross-host placement, exact semantic sanity, 5.726548780 prefill-heavy output tok/s, 6.255591493 decode tok/s, balanced dual-rail payload, zero HCA health deltas, and clean unforced shutdown. Result SHA `50b046b3c5e7ba61e194ab59be00d64941ba7454c450dbdff8304a45464f9632`. |
| GLM-4.7 Flash PP3 v5 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v5`; E16 used only 5.87-6.63 GB GPU weight memory per stage, exact semantic sanity passed, prefill-heavy output was 5.289909964 tok/s, decode was 6.553438809 tok/s, both rails were balanced with zero health deltas, and cleanup was unforced. Result SHA `adcb81b03f8781dc1f2ce55c5ee1287d258481d6673bad895f3ef9daf3c284b8`. |
| GLM-4.7 Flash PP3 v6 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v6`; E32 used 10.09-10.85 GB GPU weights per stage, exact sanity passed, prefill-heavy output was 6.215342096 tok/s, decode was 6.924257787 tok/s, HCA health remained clean, and shutdown was unforced. Result SHA `ab00ff5d7617ac7e7d126932fe1eac5c32761061d7c5fbd801c0592c639f6e74`. |
| GLM-4.7 Flash PP3 v7 | **EXPECTED FAIL (setup)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v7`; the E48 command mistyped the immutable fwuff overlay ID, failed before remote model launch, and cleaned both started local ranks unforced. The corrected path is used by v8. Result SHA `ee2467f467037f1a8de1d4526e4a4ae06617112cf15d07686ab2dff12c853923`. |
| GLM-4.7 Flash PP3 v8 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v8`; corrected E48 run used 14.31-15.22 GB GPU weights per stage, passed exact sanity, reached 6.745452171 prefill-heavy output and 7.163711287 decode tok/s, retained clean HCA health, and cleaned all ranks unforced. Result SHA `4db51926d9ca089a8268dd18ee254c010dc680efeab33eae50d9a1e779fe8753`. |
| GLM-4.7 Flash PP3 v9 | **EXPECTED FAIL (setup)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v9`; the command incorrectly placed fwuff's immutable runtime overlay under `/mnt/sanic` instead of `/var/lib/exo`. It failed before the remote model launched and left no rank running. Result SHA `7de8537a50101b536cc2a861da7047e784958f5f9d85d0dfc3c713ecb90cef2b`. |
| GLM-4.7 Flash PP3 v10 | **PASS (diagnostic)** | `/var/lib/exo/benchmarks/glm47-pp3-diagnostic-dwagon-fwuff-20260720-v10`; the E48 16/15/16 split passed exact sanity, reached 6.976789422 prefill-heavy output and 7.228700241 decode tok/s, carried balanced 27,222,332/27,215,904-byte rail payload, added no HCA health errors, and cleaned all ranks unforced. Result SHA `c0cad5464a6b2bf3f5b373ed20403fc05467e30831df73ebcec6e307d99a34ae`. |

All strict TP runs above that are marked clean completed ownership-confirmed process
termination, instance deletion, lease removal, lock release, and reserved-port
release. The four HCA-enabled TP=2 receipts also have zero selected HCA
health/error deltas and `dual_rail_payload_verified=true`.

## Imported historical Ornith receipts

These artifacts came from
`/home/kassie/projects/ornith-ktransformers-oscar-superrepo` and `/mnt/sanic`.
They are listed for hardware/runtime transfer work and are deliberately excluded
from the Exo live-test table above.

| Imported artifact or run | Result | Exact recovered evidence |
| --- | --- | --- |
| Ornith 397B BF16 to AMXINT8 conversion | **IMPORTED HISTORICAL: PASS** | `/mnt/sanic/ornith-amxint8-conversion.log`; created `2026-06-29 00:06:54 -0400`, final mtime `01:29:40 -0400`; two 52-thread NUMA pools converted 60 fused-MoE layers with 512 experts/layer and wrote 369,891 tensors across 61 shards; SHA `86cb8ba6b3eb116683aa8072ad1ae5fbc011fc713444e9c97713c8934219d87a` |
| Four-request decode, row 1 | **IMPORTED HISTORICAL: PASS** | 10,518 input/1,024 output tokens, no errors, 60.870 s, 16.823 output tok/s, 16.564 s mean TTFT, 172.837 ms mean TPOT, 28 tok/s peak |
| Four-request decode, row 2 | **IMPORTED HISTORICAL: PASS** | 10,518 input/1,024 output tokens, no errors, 58.778 s, 17.422 output tok/s, 16.207 s mean TTFT, 166.142 ms mean TPOT, 28 tok/s peak |
| Four-request prefill, rows 1-3 | **IMPORTED HISTORICAL: PASS** | Each row completed 79,377 input/four output tokens without request errors; 1,066.393/1,061.920/996.876 input tok/s in 74.435/74.749/79.626 s |
| Four-request prefill, rows 4-6 | **IMPORTED HISTORICAL: DEGRADED** | Each row completed the same token counts without request errors, but only 80.551/135.626/184.006 input tok/s in 985.429/585.265/431.384 s and each recorded zero TTFT despite long end-to-end latency; not a stable baseline |

Decode receipt:
`runs/bench_internal_agent_decode_4x512_256.jsonl`, created
`2026-06-30 12:46:41 -0400`, final mtime `13:11:35 -0400`, SHA-256
`13ba0acb465305bbe36649e217dae0182b8d98eaf283df00e88b1fa81b78b23d`.
Prefill receipt:
`runs/bench_internal_agent_prefill_4x20000_1.jsonl`, created
`2026-06-30 12:49:03 -0400`, final mtime `14:40:19 -0400`, SHA-256
`4fd9c2edb5412628b4e69562d6a9ccbbc9121d375cbadb0a42b614b0363df857`.

Every inference row reports Ornith-1.0-397B FP8 GPU weights, BF16
activations, OSCAR INT2 KV/group 64, CUDA TP=2/PP=1, AMXINT8 CPU weights,
100 CPUInfer threads, two NUMA pools on nodes 0 and 1, four GPU experts per
layer, frequency placement, dynamic expert updates, CUDA graphs at batch sizes
1/2/4, four requests, 163,840 context, and a 655,360-token pool. This is strong
configuration and successful-serving evidence for a hybrid dual-RTX-3090 run,
but it is not counter-verified proof that AMX tile and GPU expert kernels ran
concurrently. No launch/server log, AMX instruction counter, backend-specific
expert count, overlap trace, GPU identity/utilization record, model/source hash,
deterministic output oracle, exact per-row timestamp, or cleanup receipt
survives. The converted AMXINT8 directory is also gone and cannot be rehashed.

The superrepo README's 13.47 output tok/s frequency result predates the AMX
launcher (`99d6430` came before `b28784b`) and is not an AMX result. The actual
surviving AMX-configured decode rows are the 16.823 and 17.422 tok/s entries
above.

### Imported source revisions and applicability

- Historical heads: KTransformers `56dc52a`, embedded SGLang `f2d46685c`, and
  orchestration superrepo `55934e6`.
- The initial GLM-4.7 Flash BF16 smoke starts from the smaller stable pair
  KTransformers `8e46e5896c3d993a1285052f2618f5a9f01882d4` and embedded SGLang
  `5d6bef9f61637aaeaf047bf8209def2af3eaa83f`. No recovered Ornith commit is
  required. Exo's reproducible mail patches produce admitted revisions
  KTransformers `6e0a4480936effa7bf0ece429f78a00b29932bec` and SGLang
  `42504e59810130460fc24fdd17ef534cb8278a4b`, with fatal registration,
  structured 46-layer wrapper/mask coverage, and fail-closed exact loading of
  every resident expert's gate, up, and down projections from the checkpoint.
  The current patch also initializes the inherited non-hash and shared-expert
  state that the first real GLM extend exposed as missing.
- Highest-value generic candidates to mine forward are KTransformers `e5f1771`
  for fused BF16 expert conversion, SGLang `4c267d946` for per-layer frequency
  placement with dynamic updates, `464ffce91` for AMXINT8 full-prefill fallback,
  `af0fea990` for FP8/Marlin layerwise prefill, and `477d69557` for benchmark
  token accounting. KTransformers `1e86695` is specific to channel-FP8 source
  conversion; `56dc52a` and SGLang `f2d46685c` target packed MXFP4 experts.
- SGLang `ab2d8be5f` contains the large Ornith/OSCAR INT2 port. Its generic KV
  kernels may be research input, but Ornith uses Qwen3.5 hybrid GDN/full
  attention while GLM-4.7 Flash uses MLA compressed cache. No recovered receipt
  validates OSCAR on GLM, and the existing unified MHA/GQA INT2 pool is not a
  drop-in MLA cache.
- GLM-4.7 Flash is therefore a proxy for lifecycle, SM86 packaging, NUMA and
  affinity, AMX conversion/execution proof, expert placement, CPU/GPU staging,
  overlap, and telemetry. It is not a proxy for GLM-5.2 NSA/DSA, IndexShare,
  MTP, FP8 KV/FlashMLA, PP=3 behavior, or 753B-scale memory pressure. Use a
  runtime-supported BF16 source for AMXINT8 conversion, force 0/1/2/4 resident
  expert sweeps because the 16.85 GB Flash checkpoint fits a 3090, and retain a
  separate deterministic GLM-5.2 architecture fixture. The 0-expert control
  requires its own fail-closed CPU-only target and execution receipt; the mixed
  profile intentionally admits only 1-63 resident GPU experts.

## SmolLM2 TP=3 failure ledger

| Attempt | Result | Finding and disposition |
| --- | --- | --- |
| v1 | **EXPECTED FAIL** | Overbroad process matching treated unrelated arguments containing `/root/exo` as Exo; fixed by `0c58070e` |
| v2 | **EXPECTED FAIL** | Probe hashed only payload files instead of the complete pinned snapshot; fixed by `55a2199e` |
| v3 | **EXPECTED FAIL** | Disabled remote API prevented a reciprocal topology graph; readiness contract corrected |
| v4 | **EXPECTED FAIL** | Wrapper rejected a 300 s cleanup grace below the computed 829.25 s minimum; no child launched; historical manifest conservatively records cleanup failure |
| v5 | **EXPECTED FAIL** | `multiprocessing.Queue` feeder/GIL stall prevented all ranks receiving `ConnectToGroup`; bounded flush added by `96230c4b` |
| v6 | **PASS (diagnostic)** | Three-rank proof completed and cleaned up normally |

## GPT-OSS artifact integrity

- Model revision: `773a7da77e569019bb0fd17a554b263738d669a3`.
- Verified model locations:
  `dwagon:/var/lib/exo/models/mlx-community--gpt-oss-20b-MXFP4-Q8--773a7da77e569019bb0fd17a554b263738d669a3`
  and
  `fwuff:/mnt/sanic/exo/models/mlx-community--gpt-oss-20b-MXFP4-Q8--773a7da77e569019bb0fd17a554b263738d669a3`.
- Stage result/manifest/runtime/config SHA-256:
  `515200e9f2dfd2693b183774100fffdf5677134c975ddc88bc46b2e6e5ff68f7`,
  `9ad4cc793608d09ed53df2a9ee5543cebc079c9dfe8122b2800ec751672cc627`,
  `477b4c9efa82260b98721f7d76deff195b4bc6b32fa5cea77eacb3d5d11cff9f`,
  `44f4cc16649ad2881ae24bc8cd526959c164121ecd4133449e8133d47a54b3be`.
- TP1 result/manifest/runtime/log/raw-config SHA-256:
  `62123e4dc8c57dc2b896676dc7961a881053566195d5484a5f8cb37689faf217`,
  `85ddb30d445bda83c925d912ab9dd865fb85953650ff389b7b9c1d44d19a5e1e`,
  `76ff69c5de2cf88c25f94a761973c064cb21b590caee1a7967e24b4f8028f66f`,
  `9fe6a701286ffaba95e3fc4ee8db8c6fb9561751d202baf845795ec92e75ed79`,
  `2d5304b9ba99d0e8dfe7a1e6f785eecf7bf6b766ba18ea066615c5e23e5a8ba1`.
- TP2 result/manifest/runtime/dwagon-log/fwuff-log/config SHA-256:
  `2b72d7967cfb41378639c795eeb6b3a32e3ab92431cde1b169d5ed4c395d85df`,
  `e3ae468223cf26d55cd5bae37caf62f5e72b9146e870956e6a836bbb3f847cf6`,
  `225887935a6a2f73fac2c96efd61545585bd75666bbb4b6d46404166c8dc5c98`,
  `db46ccc4dbc54ab7c1a0d49a4c7de88870802b825b227dd9f6d8a5ebe4396851`,
  `7238881f92798045a8485b8f8cc1f12430fa078228488ed252452569d25b8afd`,
  `c087ec61417858c9c5d03395ad3ba5469ac7f1c8ac84cee838184ca5ae301613`.

## GLM-4.7 Flash artifact integrity

- Model revision: `1454cffb1a21737e162f508e5bc70be9def89276`.
- Verified model locations:
  `dwagon:/var/lib/exo/models/mlx-community--GLM-4.7-Flash-4bit--1454cffb1a21737e162f508e5bc70be9def89276`
  and
  `fwuff:/mnt/sanic/exo/models/mlx-community--GLM-4.7-Flash-4bit--1454cffb1a21737e162f508e5bc70be9def89276`.
- Stage result/manifest/runtime/config SHA-256:
  `d3eb9fd0e854b13494a115c872dc127fa569e525a14e115054283284f271fdec`,
  `f6360fd6eac23adf709fa109eee0dd45122ff5d09b706c7ae3e94f87e29d1a8b`,
  `b7cd82896c65af0bd4cee5a6417cfe34c845c669af096e17d274d5bdb1766ca8`,
  `538831b0a93226c70d4d808d3331555a99bcb1a998feb8e91777efdcf3c05c58`.
- TP1 result/manifest/runtime/log/raw-config SHA-256:
  `0064768cb29f20072b22a6ac7fff5b9b947c0da70dcd0e446e0e25c820eaa890`,
  `2fbe883160900818e4d9cf46ab7c5c9a12d2a2d755996fee114d9e8def65ca53`,
  `c689c0358e802a00b6dc584c54750431f229ff283c614e5ac3975b274f0dbde3`,
  `6fffc3286c57a6b9da6408ffe916c52dacc9fac3fccdacddf56cf3dee048ba38`,
  `719a89b29bb51fa8a8936d4f8fc03c9d4c251103ccc2cf77088abb4fc23a23ca`.
- TP2 result/manifest/runtime/dwagon-log/fwuff-log/config SHA-256:
  `503ac516604293c9e41e7fb7b7c2fa2805fbdaa8827c2053d4caae2c1c7bfc91`,
  `7f1ca0a897d1ec85c858a6cbf8504c4de522fd3ef62d96b75b8f1ad56082f568`,
  `f0bdf0541d7ae67c808dbd4fde03d3819c85bcf1ad040ed7b80b7a98edb91240`,
  `feaffc4148da1578b60dd48e2293800bec4e4290b7fab6c180f02f5147df33a8`,
  `f8c3ebda6bba21fd27cd4eb10e1125916a00bd6aaaa388a6f0b8f379ff9c2232`,
  `7b3ff5a572c6319fc2f65632b1dd710f7f4e365929fee1ab6bb2710dec201a4f`.

### Official BF16 hybrid source

- **PASS (artifact integrity):** `zai-org/GLM-4.7-Flash` revision
  `7dd20894a642a0aa287e9827cb1a1f7f91386b67` is verified at
  `/mnt/sanic/exo/models/zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67`.
- All 58 Hugging Face local-download metadata records name that revision, no
  `.incomplete` file remains, and a final `hf download --dry-run` reported zero
  of 58 files and zero bytes pending.
- The 48 Safetensors shards total 62,444,175,504 bytes. `config.json` SHA-256 is
  `dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb`;
  `model.safetensors.index.json` SHA-256 is
  `91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791`.
  The new exact-profile verifier accepted the live snapshot as BF16 with 47
  main layers. The index's embedded `metadata.total_size` is 31,221,488,576,
  not the real shard byte total; the packaged model contract therefore binds
  the actual size and SHA-256 of every shard instead.
- The canonical contract is
  `src/exo/worker/sglang_kt/manifests/glm47_flash_bf16_7dd20894.json`, pinned by
  the launch profile at SHA-256
  `4e7333f341ddc5855aa4253d454e3210d84427fae0159eb956104ff00c437479`.
  It covers 54 launch-relevant files: config, index, all indexed shards,
  tokenizer inputs, generation config, and chat template. It also verifies
  each file's exact Hugging Face revision/blob metadata, rejects unindexed or
  extra shards, and rejects executable remote-code files.
- The leased full-snapshot validation passed at
  `/var/lib/exo/benchmarks/glm47-model-contract-20260719-v1` in 586.367 seconds
  with `profiler=none`, return code 0, and unforced cleanup. Result and manifest
  SHA-256 are
  `9b7b5a5a5f054e79f0606111fbdc9a4e7aa80506d762cffc537a75c437c9d284`
  and
  `2f7666e286ad89ec3ec556a56e5250ea42fba9b9ce175f89334b42dda0e7f419`.
  This is artifact identity evidence only; it does not prove a model load,
  forward pass, wrapper coverage, expert routing, or launch admission.
- The current clean prepared source is
  `/var/lib/exo/sources/ktransformers-glm47-6e0a448`, at KTransformers
  `6e0a4480936effa7bf0ece429f78a00b29932bec`, embedded SGLang
  `42504e59810130460fc24fdd17ef534cb8278a4b`, llama.cpp
  `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`, and pybind11
  `bb05e0810b87e74709d9f4c4545f1f57a1b386f5`.
- The current native dwagon build is
  `/var/lib/exo/runtimes/glm47-sglang-kt/dwagon/ea9de367cfebe35dc6afe51c1bda5e7daf35d6f51114f404dfebd63d055eec20`.
  Its build ID is the final path component and its receipt SHA-256 is
  `1f304ea5667445cdd66e9c6938e78682b7119a6c3c3b946cf42ce816a0639542`.
  The completed receipt binds the complete canonical build inputs and wheels.
- The immutable dwagon overlay is
  `/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/b275ec08c01fdce2cd6adb64f10b35f0a5bda12af20899a1fac1de42aa29ecd3`.
  Its install ID is the final path component and its receipt SHA-256 is
  `7a2b6fd01efb7f889f01ae2a47c66c2625c4162a93373ea116d3964fd405a5f5`.
  The overlay pins its base runtime and three newly built wheels without
  modifying the old 9.9 GB environment.
- The independent fwuff native build below pins the superseded SGLang revision
  and remains historical evidence only. Rebuild fwuff from the current source
  before a two-host GLM validation. Its path is
  `/var/lib/exo/runtimes/glm47-sglang-kt/fwuff/e21ef087b1c50cf961339e1bd1a2e1a3f60047579f811385614297de6a2abfc9`,
  with build-receipt SHA-256
  `051b78a5238adac99721eb268c95d8ab5e8721d8c1c61114bbda967468433dfb`.
  Its host-native KT wheel SHA-256 is
  `f57c574cc190f8817a51cf0b08c5f2165e2f761c3a1b2999560abcbdcc792d45`;
  the pure-Python KTransformers and SGLang wheels matched the superseded dwagon
  build byte-for-byte.
- Fwuff's immutable overlay is
  `/var/lib/exo/runtimes/glm47-sglang-kt-overlay/fwuff/82d20634f743ed87ae9cc71f2b7f4936d9451363db1ca46a207218f22de51ef8`,
  with install-receipt SHA-256
  `fe57f9fe10160ebf2f0a0ba3731e69c8ddb4608841f07880bac2ff6bd0640eb4`.
- The kernel validator can prove only
  `kt_bf16_amx_executed_v1`: exact runtime provenance, SM86 BF16 CUDA math,
  AMX BF16 at query lengths 1 and 16, and a two-way
  CUDA-to-pinned-host-to-AMX-to-pinned-host-to-CUDA dependency chain. It cannot
  satisfy a GLM model-execution receipt or authorize a model launch.
- Model admission remains fail-closed until Exo parses receipts from verified
  files and binds the kernel/build/install receipts, the now-pinned model
  contract, runtime artifact hashes, wrapper coverage for layers 1-46,
  a real short forward, and the CPU-only or mixed expert-execution capability.
  Caller-constructed Pydantic receipt objects are not sufficient evidence.
- The live kernel receipts below establish AMX execution but still do not
  establish a SGLang-KTransformers model load, CPU/GPU hybrid forward, output
  parity, or performance result.

### Kernel validation attempt ledger

| Attempt | Result | Evidence and lesson |
| --- | --- | --- |
| `glm47-kt-kernel-dwagon-20260719-v1` | **EXPECTED FAIL (diagnostic)** | The clean deployment was absent from the overlay interpreter's import path, so `exo` failed to import before Torch, CUDA, or AMX execution. Cleanup was unforced and the lease was released. Result/manifest SHA-256: `3229885ee0fb809493ae5ab1bef59326ffcc9faf7551539fb15214c9b9dacda4` / `38f7d2c81f350b5f541c8af39f11f78886754d2aaab94c3ac195226b3c597d72`. |
| `glm47-kt-kernel-dwagon-20260719-v2` | **EXPECTED FAIL (diagnostic)** | Supplying the clean source exposed an accidental import of Exo's `aiofiles`-dependent download stack through `preflight_collector.py`; failure again preceded Torch, CUDA, and AMX. The artifact identity helper is now isolated in a standard-library-only module with a regression test. Cleanup was unforced and the lease was released. Result/manifest SHA-256: `ff242d7164e9805f0e18434e115300ee85ac054516e3c5d798d917c184e463f6` / `321b449f7283a73f5c8fb7f8f97b7b9bbe424431a0123b583a291469941011a5`. |
| `glm47-kt-kernel-dwagon-20260719-v3` | **EXPECTED FAIL (diagnostic)** | CUDA BF16, AMX BF16 at query lengths 1 and 16, and the bidirectional CUDA-stream bridge all executed within numerical tolerance. Admission failed only because Torch exposed the selected GPU UUID without `nvidia-smi`'s `GPU-` prefix. The validator now retains the raw value and compares a syntax-validated canonical value. Receipt/result/manifest SHA-256: `824ed5604e93380b59a485fdbf76cce9632591faf8edecd269e4e4214fdad995` / `d219e67258d146636253d2533a65c7906c1abe1bbf5da674f55b90b29e08b2ef` / `5a2e56bf95609041237049e1ff878a4254155a36d86179513dc759d36a4f8eae`. Cleanup was unforced and the lease was released. |
| `glm47-kt-kernel-dwagon-20260719-v4` | **PASS** | Exact dwagon build/runtime provenance, SM86 CUDA BF16, direct AMX-BF16 qlen 1/16, and the non-default CUDA-to-host-to-AMX-to-host-to-CUDA dependency chain passed. Relative L1 errors were `0.0014008`, `0.0033588`, `0.0036068`, and `0.0040199`, all below `0.02`. Receipt/result/manifest SHA-256: `b4f6fd1718bb3145a17c97cf8113bbfcd186416cfde3cd0fcc9eada301b78eef` / `3b2ad0463a575cf659f7793a2ef65c684b50207de39836d5e508b8da1e6ffb61` / `b5173d4efedcd83d8283b0800dc58263f113a97200cb54e89972911c709febc5`. Capability is only `kt_bf16_amx_executed_v1`; cleanup was unforced. |
| `glm47-kt-kernel-dwagon-20260719-v5` | **EXPECTED FAIL (preflight)** | The new lease harness rejected an earlier owner-started `/proc` diagnostic shell whose still-running command line contained the literal SEP/PAX search terms. No generator, CUDA, or AMX validator started; no kernel receipt was created. The diagnostic completed normally, harness cleanup was unforced and proven, the scratch path was removed, and the lease/lock/GPU were clean. Result/manifest/runtime-metadata SHA-256: `4ed3d2b0e7a4cc7cffd57e86d777e010b72b93b0d2992f9d859cdac893661aea` / `aa11f2340f75ca1e3b22d93d02a0eb3110176c16932e84173a68010a516fb62f` / `b25c83da3d98317ea25a65df641ca4bd0455f8608beefa3c0027ef0b6b043925`. |
| `glm47-kt-kernel-dwagon-20260719-v6` | **PASS (diagnostic)** | The dedicated lease harness passed exact overlay-interpreter/build provenance, SM86 CUDA BF16, direct AMX-BF16 qlen 1/16, and the CUDA-host-AMX-host-CUDA dependency chain. Relative L1 errors were `0.0014008`, `0.0033588`, `0.0036068`, and `0.0040199`, all below `0.02`; capability is only `kt_bf16_amx_executed_v1`. Both QDR ports were physically `LinkUp` with exact GIDs but subnet `INIT`, recorded under `metadata_only` with no traffic expected. Model verification/reportability and performance comparability are false. Receipt/result/manifest/runtime-metadata SHA-256: `53132ee63e92ef62ec752ebb44d98f61316fc21ea57c2022e9c18d5f8f9fd5e4` / `4cd9c6ad3ae14b8810a8c2b0dec8fa9bcfd780ad51ea60673327f267fb986e3c` / `1419c96652c9186320beb8de1bdab93f5f9f10c9576fa375c6d4db226b2c9e28` / `147c521b6ba28b7563cb5db2fe976985dfa793d03ee5618393278dc9903f42bc`. Cleanup was unforced; lease, lock, GPU processes, and scratch were clean. |
| `glm47-kt-kernel-fwuff-20260719-v1` | **PASS** | Fwuff independently reproduced the same four deterministic numerical errors using its own native build and GPU UUID. Local and remote receipt/log hashes matched, no remote validator survived, and cleanup was unforced. Receipt/result/manifest SHA-256: `efe770fc84f7e28614e0d2c9ff3ca3b9e9337d511fbd78d19c5a3bade65bb782` / `093ec960fc39aaace7fc99b3abe2d6daaf02f73aba7277f14c9c06da3a7c9d3d` / `be8a450a83a424a4d39604bf2f658d04bedf707da92d58a8418fa5843fc657d4`. Capability is only `kt_bf16_amx_executed_v1`. |

### Model validation attempt ledger

| Attempt | Result | Evidence and lesson |
| --- | --- | --- |
| `glm47-kt-cpu-control-dwagon-20260719-v1` | **EXPECTED FAIL (preflight)** | The generator and fresh overlay kernel validation passed, then the model validator rejected the exact snapshot before hashing or loading because its 117 files and 6 directories were writable (`0644`/`0755`). No model receipt was created. The revision-pinned NFS snapshot on fwuff was subsequently made immutable (`0444` files, `0555` directories) without changing content. Result/manifest/runtime-metadata/kernel-receipt SHA-256: `d061131a2bbdca73fd9f7659cb977bb72aea9a0d4a553e49d5d8e684fbc868ed` / `4417fb1c5b484873891e2c879329e56b3f480a2659d018c40718251a972bc611` / `954feef4b3a2be0148614b58db273498dd92492f28d3fc6bb0ed21eea27cf7ed` / `742f5838e9457e86caacab6ba06d0ff2ee8b6aa598677651d2ba4733654dc45f`. Cleanup was unforced; lease, lock, GPU processes, and scratch were clean. |
| `glm47-kt-cpu-control-dwagon-20260719-v3` | **EXPECTED FAIL (diagnostic)** | Clean source `b4c9710f` ran in the delegated systemd/cgroup-v2 containment. The generator passed in 0.401 s and the fresh kernel validator passed CUDA BF16, direct AMX BF16 qlen 1/16, and the CUDA-stream bridge in 8.121 s. The model validator then spent 588.969 s verifying the immutable 62.4 GB snapshot before strict receipt admission rejected only `host.process.cwd="/"`; the transient service had inherited systemd's root working directory, while execution-path evidence intentionally excludes `/`. No model construction, GPU allocation, forward, or model receipt occurred. Commit `5ddf2aa7` sets the immutable deployment as the service working directory and admits the kernel receipt before model traversal. Result/manifest/runtime-metadata/kernel-receipt/process-spec SHA-256: `47043acfd362e7e7825568b095cd201b59a162d74a695ef83f2ec523223035a8` / `0fd6e8c7a6f0199d46a945f5ac93bf1b0fd292b17b61554777c7287b1ece073d` / `508a93c38589171c2feffd194096d9b77f07acc9f19e213d7a0a23b028433a9c` / `a034ee6ca9d4062fc2a8ba2787ad6ac9c9c3bf3e80217f22d9cea7eb4fe09aa2` / `0f86db96dcc2dc6ff7e0565359cea5a2af57389fe2edb68c1de14a1882e00142`. Result was nonreportable and non-comparable; cgroup kill/empty/removal, unit collection, lease/lock release, scratch removal, and idle GPUs were all verified. |
| `glm47-kt-cpu-control-dwagon-20260719-v4` | **EXPECTED FAIL (diagnostic)** | Clean source `14340320` proved the transient-service cwd fix and early strict kernel-receipt admission. Generator and fresh CUDA/AMX validation passed in 0.370 s and 7.605 s; the model stage then failed in 45.349 s before child creation because pinned CPython 3.12.13 exposed no callable `os.memfd_create`. No model construction, model-load GPU allocation, forward, or model receipt occurred. Result/manifest/runtime-metadata/kernel-receipt/process-spec SHA-256: `05732f405f7c5cc622d5a539a10d362fa464718abafac4869971c86861ca2125` / `657aa82595d4bbe4e2d908eb75e7813b9fc83419deaf6f07d13dded7b208de8d` / `4e3def2938ecf98a885add0d5c3caad7066bf2678291436968e0cd1c27214807` / `8bdb2453c7d9e47381df2ce24642e867b3283c5f8638d852659bdbdd5caaf31c` / `99b41e60975a189b65d6627c004f7543dfbd53410b032f788c27ad7d29d7a01a`. Result was nonreportable/non-comparable; cleanup was unforced and the lease, lock, cgroups, unit, processes, GPU, and scratch were clean. `profiler=none`; the run did not use the unsafe profiler drivers, although preflight observed the already-loaded `pax` and `sep5` modules. |
| `glm47-kt-cpu-control-dwagon-20260719-v5` | **EXPECTED FAIL (diagnostic)** | Clean source `ca732fe6` passed generation in 0.368 s and fresh CUDA/AMX validation in 8.393 s. The 144.213 s model stage used sealed memfd evidence successfully, independently verified the immutable 62,444,175,504-byte checkpoint in parent and child, loaded all 48 BF16 shards, built KT/AMX wrappers for routed layers 1-46, and passed the layer-one AMX probe twice against its FP32 reference. The first real eight-token extend then failed before expert dispatch at `deepseek_v2.py:770`: `Glm4MoeLiteSparseMoeBlock` lacks `is_hash`. Zero experts were GPU-resident; weights used 4.52 GB, the 4,096-token BF16 KV cache used 0.21 GB, and 18.46 GB remained available, directly disproving the claimed unavoidable 2 GB ceiling for this split. No extend result, decode, model receipt, performance result, or InfiniBand traffic was produced. Result/manifest/runtime-metadata/kernel-receipt/process-spec SHA-256: `0ae112e4acd713270914a2a51ac7a344d4d9224933881ed7168c2f2a1fa14010` / `dfc1ad00e53bdab4f3c3e9eddb4dda8be993802bb1f748dafce51571b94787f9` / `28f50e5d2ce61a494de9ee793504c71eca467b248da9e25f63bb7087008225f7` / `0fe16ea3d3a67e63c216ea69adfa23b79c7402b09e77c258746a204f8a8b6933` / `54a8d049f1ec26d73cfd8e9f85f526bb1c626a9bae0598bbf3adf19420215d4f`; canonical process-spec SHA-256 `8718ec41c3f4cad77baa50759a9ea0e002bfb500b11537f295540f4d2c9ac6f8`. Cleanup was unforced and complete; the lease, lock, cgroup/unit, owned processes, ports, GPU, and scratch were clean. `profiler=none`; loaded `pax`/`sep5` modules were observed but never used. |
| `glm47-kt-cpu-control-dwagon-20260719-v6` | **EXPECTED FAIL (diagnostic)** | Immutable source `0af52113` used the corrected SGLang `42504e598...` build and overlay. Its immutable deployment is `/var/lib/exo/deployments/glm47-kt-cpu-control-dwagon-20260719-v6`. Generation passed in 0.370105447 s and fresh CUDA/AMX validation passed in 8.024723519 s. The 131.365563637 s model stage loaded all 48 BF16 shards, emitted exact wrapper coverage for layers 1-46, and did not reproduce the former `is_hash` failure. Weight loading took 18.08 s; model weights occupied 4.52 GB, the 4,096-token BF16 KV cache occupied 0.21 GB, and 18.46 GB remained available. The public wrapper retained only `live GLM-4.7 backend failed`. Code-path inspection therefore provides the current, explicitly unverified diagnosis: SGLang's model-forward result contains non-null `next_token_logits` and `hidden_states=None`, while the trace collector tests field presence before nullness and tries to snapshot `hidden_states`, leading internally to `builtins.NoneType is missing dtype`. No verified extend/decode trace, model receipt, performance result, or InfiniBand traffic was produced; status was `validation_failed`, return code 1, and the result is nonreportable/non-comparable. Result/manifest/runtime-metadata/kernel-receipt/process-spec-file SHA-256: `50008be5e03d3b92c30db19a610eb44b9b4d7d5b33f2e8701b4cdd6f9635d465` / `41dd13ccd7e170455b2b66b83012935254ad559ce3a88e96ee007670add98486` / `bebca51f18b18a9f01403f724f50828fe43eb4203f5dc659c140b82e246a63c0` / `5cfffa1e450f0dbcded077f7496e5b9d3094bed1f73aeab370f2ebb2775867c6` / `c282b9e51c8540141648ea62787640d84d2f3a201bfef17889b8a6ba337b3838`; canonical process-spec SHA-256 `bdeed9d41a5a573b8406980bbfe10a53e22671ac8a007684c87d40d64c8fa94a`. Config SHA-256 was `2dddf4ea4efa364bb83a5028d6302c0708646ae366847f3fb5846b01ee5def5e`; orchestrator/validator SHA-256 values were `53d23cc5458fe6a6f92e9f3751cebbf6bcf3f8a78d7f2e439e435d5db0eba2db` / `c8424ab5b9c39258b468e34e200d92f7524cbe9f829dfff2f07167dd57b1bd94`. Cleanup was unforced and complete; the lease, lock, unit, owned processes, ports, GPU, and scratch were clean. `profiler=none`; the unsafe profiler was not invoked. |
| `glm47-kt-cpu-control-dwagon-20260719-v7` | **EXPECTED FAIL (diagnostic)** | Clean source `d9ff2920` used the corrected native build/overlay and immutable deployment `/var/lib/exo/deployments/glm47-kt-cpu-control-dwagon-20260719-v7`. Generation and fresh CUDA/AMX validation passed in 0.380145077 s and 8.474109744 s. The 188.229354069 s model child returned 0 and published a valid 29,006-byte receipt: all 48 BF16 shards loaded; layers 1-46 used `NativeMoEWrapper` / `AMXBF16_MOE` / `kt_ep` with zero resident GPU experts and stable mask `50680b69...`; every wrapper ran once for extend and decode. The layer-one experts 0-3 ran entirely on CPU twice with deterministic output and relative L1 error `0.0035165481 < 0.02`. The eight-token extend produced finite FP32 `[1,154880]` logits, argmax 3764, KV 0-to-8, and logits SHA-256 `a4e959ff...`; decode produced argmax 10, KV 8-to-9, and logits SHA-256 `008e5ae3...`. Weight loading took 19.38 s; model weights occupied 4.52 GB, the 4,096-token BF16 KV cache occupied 0.21 GB, and 18.46 GB remained available. However, native Gloo/BF16/CPUInfer/AMX diagnostics occupied the first 193 lines of the 194-line stdout stream before the valid final JSON response. The strict outer harness rejected the whole stream before its six-field receipt-binding step, so outer status is `validation_failed`, `model_checkpoint_verified=false`, and the run is nonreportable/non-comparable. An offline current-source receipt load and pure admission-binding audit passed, but the full launch-time 62.4 GB rehash was not repeated during that audit and does not convert the outer result. Model/kernel/canonical-process-spec/raw-process-spec/model-contract SHA-256: `2800220a897b77d720ad4da01fd83629418265d671285cd726f6a82c36c08b90` / `a2031382a7754b5515b23413b07d9890854957fc84c412b7d09b69ecb25b83a5` / `c4889b5f1e0d29c50e76e057451a50fc15e46bb4ca11076c6803852fe60d03fb` / `7d5e85d7906206ee9f47a9e94d48cdb2594f21c8c50a37594122664ce1212a39` / `4e7333f341ddc5855aa4253d454e3210d84427fae0159eb956104ff00c437479`. Result/manifest/runtime-metadata/config SHA-256: `4cbcb1eaace84749336a295fc03d7bf3a03ba6c28e5eefc39212d45467c51121` / `360f74c8f902307dc218bc1c1099fca51f60d5741b0fc27c21829bd59fe41b4b` / `892f3792033e3fbc4c39cc933bd42494cf7687d0fc9544011536b02c4c361a0b` / `7f211abd648412a8184b598eefcf5b6a4ae7eadf845f3f888fd2c27d002d86f2`; orchestrator/validator SHA-256 `87165caefa568653eb1c9d855d66c174daa70391911a1325479d7b207c89150d` / `6aabd10392e84f20ec1ff2305241cfcd04a934c1ab56b4592f73159be6dac1dc`. Model stdout/stderr and kernel stdout/stderr SHA-256: `fd737984a14055bc0a46d07b56012fb2a9edd1f3f1590a8292d45f3d0eacb1d7` / `11913d305fd9a5b4acf197249bd0a587140d4274bca90d9880ce642c24d92f41` / `c68d0886ef2912165e765f017533c982f6aaad6bcece4d404af28d3cfee6636e` / `e0ac13c829ddff8b29a7df65ad0d8302f3b2db4cf682fd9b0874e992df05617b`. No InfiniBand traffic was expected under `metadata_only`. Cleanup was unforced and complete; `profiler=none` and the unsafe profiler was not invoked. |

| `glm47-kt-cpu-control-dwagon-20260719-v8` | **PASS** | Clean source `187d6e67` and immutable deployment `/var/lib/exo/deployments/glm47-kt-cpu-control-dwagon-20260719-v8` completed the outer admission transaction. Generation, fresh CUDA/AMX validation, and model validation returned 0 in 0.371195338 s, 7.988936580 s, and 175.630093307 s. The outer result is `completed`, `model_checkpoint_verified=true`, and `reportable=true`; the independently reloaded 29,006-byte receipt derives `glm47_flash_kt_wrapper_active_v1`, exact layers 1-46, SM86 short-forward, physical NUMA, CPU affinity, executed AMX-BF16, and CPU-routed-expert execution capabilities. Every routed layer used `NativeMoEWrapper` / `AMXBF16_MOE` / `kt_ep` with zero resident GPU experts and stable mask `50680b69...`. Layer-one experts 0-3 ran entirely on CPU twice with deterministic output and relative L1 error `0.0035165481 < 0.02`. The eight-token extend produced finite FP32 `[1,154880]` logits, argmax 3764, KV 0-to-8, and logits SHA-256 `a4e959ff...`; decode produced argmax 10, KV 8-to-9, and logits SHA-256 `008e5ae3...`. Weight loading took 18.24 s; weights used 4.52 GB, the 4,096-token BF16 KV cache used 0.21 GB, and 18.46 GB remained. The protocol fix is proven in the real path: stdout is one 340-byte JSON line and all 408 native diagnostic lines are in stderr. Model/kernel/canonical-process-spec/raw-process-spec/model-contract SHA-256: `e927dcae4adf1dc8183293bb58f043fc25f56279b798c4420c3616ec296d6bc2` / `3f12008626f80ad0460d044d1ae9ac4cee5f6600bc74e82da9dc029b6546db24` / `550b4a05778264ad7e9fad78ceab39e701c6cb9d8d8a53f3b00a0cef436c53f0` / `f776542edf76edf7b65a6f596a8ee6413b234671b66fe00976d70cd7cc332ef1` / `4e7333f341ddc5855aa4253d454e3210d84427fae0159eb956104ff00c437479`. Result/manifest/runtime-metadata/config SHA-256: `3bd68b25b23f740cfb05a5389a80eac7c74b28d7510e370d70516cee116d4ab2` / `342f81506902bf3b1ac20b48fe7f11b9bee04d836db4fb7e1d03073d1cd5425f` / `a148204c3990fcb73135ff5797b1c0e61c81ae6bb51051881c7e06aa037474ec` / `799946a2757d943b35ac5548a17d69ba739066a67682fff525e3e0f02a78bc08`; orchestrator/validator SHA-256 `967a00bdfd400a607b517066bb35f22a19c96909c74fa54434a8bf0e140fb130` / `d121f7af964cf22af2c3f986b12689b25fddb7d994979df8dfc3faa97d69c24c`. Model stdout/stderr SHA-256: `f7590902af7f599a73f2d261254d3a6074ceff1de8127126edf89ef01f4ad707` / `e2f743d6dd0ade781e02323b64c5e0ac5b85ef880f0afeb4adfc929b0ac19c3e`. No InfiniBand traffic was expected under `metadata_only`; `performance_comparable=false`, so stage timings are diagnostic only. Cleanup was unforced and complete: lease and lock free, unit inactive/collected, owned PIDs gone, ports clear, GPUs idle, and scratch absent. `profiler=none`; unsafe drivers were not invoked. |
| `glm47-kt-hybrid1-dwagon-20260719-v1` | **EXPECTED FAIL (diagnostic)** | Clean source `f4563351` and immutable deployment `/var/lib/exo/deployments/glm47-kt-hybrid1-dwagon-20260719-v1` ran from 18:52:21 through 18:54:40 UTC. Generation and the standalone CUDA/AMX kernel gate passed in 0.363319265 s and 8.003729318 s. The 126.868393462 s model stage verified and loaded all 48 BF16 shards, configured one resident GPU expert per routed layer, and emitted exact `NativeMoEWrapper` / `AMXBF16_MOE` / `kt_ep` coverage for layers 1-46 with mask `47751fa6...`. Weight loading took 18.58 s; weights used 5.38 GB, a 4,096-token BF16 KV cache used 0.21 GB, and 17.62 GB remained. The second combined layer-one CPU/GPU result did not reproduce the first hash, so the validator failed closed before component evidence, extend/decode, or model receipt publication. Pinned source proves that Triton's fused MoE has `inplace=True` and mutates `hidden_states`; the v1 validator reused one dispatch/input for both invocations. Hybrid1 v2 subsequently confirmed this source-backed diagnosis with fresh inputs and bitwise-identical repeats; the exact-repeat gate was not relaxed. The standalone process spec and kernel receipt remain valid narrow evidence, but wrapper-load diagnostics are not model admission. Result/manifest/runtime-metadata/kernel-receipt/raw-process-spec SHA-256: `efac8c357cb99778cc411e768fd4d356a156836cab3e3b49813a6bb321415438` / `d97bf384b01a443c0332db9f3151687c6b30e70f438e8723a0c1c84efee07865` / `81392715ac9a3503dc6178c6937a87f8e2d921de693b89f01fa8853ac1e6e115` / `006ed662524e3206c468e1d79627d69590094f03994ba2662dfeffa9c74a1514` / `c51ce1fa02f3133398711935f140d343d85152582b15052b040d0722f6f385bd`; canonical process-spec/config SHA-256 `7b7e6007cb9ef0f50f30377c144e625452a7e72604a2297dc7fcfb341f9914c0` / `a3bea8394423dbe259cd840e9511717d7cf42ef2b72bbdb9afcd1d9abf923df9`; orchestrator/validator/model-stderr SHA-256 `ee2ebe26cdd47fea9d092bdb6b50e93f7310d5c8c309e8ae0265406827bc7f39` / `73d79918b8ada747ae73a66afced2219504b00548e290e3cb00eca5d7743cc1f` / `009b443f71f27f24dba5b772c491749e20bd99e4819e0360a845f5d874b9b500`. Model stdout was empty and no model receipt exists. No InfiniBand traffic was expected under `metadata_only`; the result is nonreportable/non-comparable. Cleanup was unforced and complete: lease, unit/cgroup, PIDs, ports, GPUs, and scratch were clear. `profiler=none`; already-loaded unsafe modules were observed but not invoked. |
| `glm47-kt-hybrid1-dwagon-20260719-v2` | **PASS** | Clean source `460339f3` and immutable deployment `/var/lib/exo/deployments/glm47-kt-hybrid1-dwagon-20260719-v2` ran from 19:19:37 through 19:22:56 UTC. Generation, fresh CUDA/AMX validation, and model validation returned 0 in 0.363347796 s, 8.208259849 s, and 187.547120813 s. The outer transaction completed with `model_checkpoint_verified=true`, `reportable=true`, and `performance_comparable=false`; an independent current-source load accepted all receipt parents and derived the six common GLM/NUMA/affinity/AMX capabilities plus `kt_bf16_cpu_gpu_hybrid_executed_v1`. All 48 BF16 shards loaded; routed layers 1-46 used exact `NativeMoEWrapper` / `AMXBF16_MOE` / `kt_ep` coverage with one resident GPU expert and stable mask `47751fa6...`. Layer-one selected expert 0 on GPU and experts 1-3 on AMX CPU. Combined, CPU, and GPU repeats were bitwise exact; relative L1 errors were `0.0039218321`, `0.0035969012`, and `0.0037137014`, and the reconstructed BF16 merge error was `0.0013251120`, all below `0.02`. This fresh corrected run confirms that v1's reused, in-place-mutated dispatch input caused its repeat failure. The real eight-token extend produced finite FP32 `[1,154880]` logits, argmax 353, KV 0-to-8, and SHA-256 `9ea5e4cb...`; decode produced argmax 10, KV 8-to-9, and SHA-256 `d75a5c62...`. Weight loading took 18.44 s; weights used 5.38 GB, the 4,096-token BF16 KV cache used 0.21 GB, and 17.62 GB remained. Result/manifest/runtime-metadata/kernel/model-receipt SHA-256: `c360827f30692c94659a303ceddfbc4d7efaaa5a436597bb263566bfe28b3936` / `514fcc5888b26dc05d218158c226ea3709aa0ccd95ba6096f442254a4f19bfe4` / `c0be04f227993e30198fb9b812d80921f5ce8fe724642950dbff0b0f9b001a96` / `6f121e9683180f6b79048c2772488bb423ff54e409ae519f700ac8159aedf488` / `b2472fe3b36b6d0df094dec5f48bd25e1b72cc2f258410121bf45e5aca36afde`; canonical/raw process-spec and config SHA-256: `53dbf8901e6f28616866e6eb8491c46fa4daaa425231e31c9d319853d2cba444` / `7d99c5b2c2a0d5118fc9ec1554ac9bf67d010c652e7e3eaf32161b01e761e65a` / `51311c8cd51d1f3c7e37b3e8739126a0a0b98b2f60ba4467839e047b2f74c396`; orchestrator/validator/model-stdout/model-stderr SHA-256: `defadcf200e2bf8283867c0b3972903f3bbdfcdeac89ad087055fc2505448494` / `7ed9cdfc5bc9195307b0fdb48d2ccaba11c71002ecec82ee3be4c27eef470d80` / `932035dd8645bdc53ae7af096ccfd7326954a0843ec71b6c3d17bffe4a79241f` / `4d184104d5cf65177a6abacc0d66413faee714d4f1332947b0ec7bcc43c1e0f5`. No InfiniBand traffic was expected under `metadata_only`; stdout was one valid JSON line and `profiler=none`. Cleanup was unforced and complete: lease/lock, transient unit/cgroup, owned PIDs, reserved ports, GPUs, and scratch were clean. |
| `glm47-kt-hybrid4-dwagon-20260719-v1` | **PASS** | Clean source `ab275213` and immutable deployment `/var/lib/exo/deployments/glm47-kt-hybrid4-dwagon-20260719-v1` ran from 19:42:16 through 19:45:32 UTC. Generation, fresh CUDA/AMX validation, and model validation returned 0 in 0.375188291 s, 7.890541470 s, and 184.926367361 s. The completed outer transaction is checkpoint-verified and reportable but correctly sets `performance_comparable=false`; an independent strict load accepted all receipt parents and derived the six common capabilities plus only `kt_bf16_cpu_gpu_hybrid_executed_v1`. Every routed layer 1-46 used exact `NativeMoEWrapper` / `AMXBF16_MOE` / `kt_ep` coverage with resident experts 0-3 and stable mask `672d9eb5...`. The layer-one oracle selected `(0,1,4,5)`, ran experts 0/1 on GPU and 4/5 through AMX CPU, and produced bitwise-exact repeats. Combined/CPU/GPU relative L1 errors were `0.0042289809`, `0.0037121559`, and `0.0039452631`; reconstructed BF16 merge error was `0.0013043246`, all below `0.02`, and combined output equaled the merged output exactly. Extend/decode logits were finite FP32 `[1,154880]`, retained argmax 353 then 10 and KV 0-to-8-to-9, and had hashes `88272df4...` / `0b263284...`; every wrapper ran once in each mode and the mask remained stable. Weight loading took 18.61 s; weights used 7.77 GB, the 4,096-token BF16 KV cache used 0.21 GB, and 15.23 GB remained. The runtime used valid fallback Triton tiles because the RTX 3090 E=4 gate/up and down config files are absent; correctness passed, but the performance impact is not measured. Result/manifest/runtime-metadata/kernel/model-receipt SHA-256: `87824b1332e50885cd002f7ef4e9271ee43a2b7be89e76f6456be10a5ee94a51` / `b8e5c579d670487db95c9d8781ce81343aca305016e5cd693d9973c52b3e85d0` / `025a8bb00d0312efac2290a8215df0bb969305d8f74d8f3c48ec161b276e2b33` / `f6b863e90f32119a01035b7629aa2181405405c207ba5cb9c6ba0bb129bcdb98` / `0cb1cf8e8e929c7e8ee1b20c679fcd1d869238227537bc306ecd9ab10fc672a8`; canonical/raw process-spec, config, metadata, and deployment-receipt SHA-256: `ddf02c48485e17a4075bdbdf9b6d1e89788c692394b0758c4c5be53225d861bc` / `c0c64e2d6f14998204a0ab25bd096b086d75654794465e8d639f3a0becb77284` / `34048f4b9d2b778cca1fd4566e2531c16c2b391044c989674222f12d3db83331` / `19fb37dec2d764d881e530eec61dc006ad7371091761715b8ec2be24e85b5369` / `6d1b9b8967a1dd60d3a6d54e463975099a13ed9edd9ad24fd5c31b3e581eb8ed`; orchestrator/validator/model-stdout/model-stderr SHA-256: `f43909dea65d2f6564e10a30e033de6a30fe9b57740ce2855ba993fd7d6222ec` / `badfab0fc095a1b967c5a8b39ddeb4c79a77fd24d17b4c253ab4985fb46e2394` / `88d03069525ff1c6c6b4be53b4a0843969be849e190f89a5d9cb75b482d83934` / `958318a13a3710399eed34688edbb11086bfef94f28af5019ffc32852dfe3263`. Stdout was exactly one 336-byte JSON line; no InfiniBand traffic or profiler was used. Cleanup was unforced and complete: lease/lock, unit/cgroup, wrapper/child/owned PIDs, ports, GPUs, and scratch were clear. |
| `glm47-kt-serving-admission4-dwagon-20260719-v1` | **PASS** | Clean source `ca468528` and immutable deployment `/var/lib/exo/deployments/glm47-kt-serving-admission4-dwagon-20260719-v1` ran from 20:47:10 through 20:50:27 UTC. Generation, fresh CUDA/AMX validation, and model validation returned 0 in 0.403311116 s, 8.616654461 s, and 184.746718479 s. The exact target was `glm47_flash_bf16_sm86_serving_baseline_v1`, with diagnostic timing/distribution variables absent, radix cache and CUDA graphs disabled, and four resident experts on every routed layer 1-46. The layer-one `(0,1,4,5)` oracle ran experts 0/1 on GPU and 4/5 through AMX CPU; component and merge repeats were bitwise exact, all relative L1 errors were below `0.02`, and real extend/decode retained finite FP32 `[1,154880]` logits with hashes `88272df4...` / `0b263284...` and KV 0-to-8-to-9. Weights used 7.77 GB, the 4,096-token BF16 KV cache used 0.21 GB, and 15.23 GB remained. Result/manifest/runtime/kernel/model-receipt/raw-process-spec SHA-256: `3efcce3cd14eddce4512aff1068c6fc1d56203ddf49e1731ffb6c3eb35a5f2bf` / `f2e6dfc700912ffbe7889e5d9329f0d667972e6d79189e63a28f1aed94277ecf` / `5db094ca2cc22050f6355fee1d16edbb935d5d97c4c62d79dd0837dac6863269` / `bd555606145ad2878a5a6be68bc82d2f8eee30e992e69c2d8ac45d0cd8d1d6a4` / `2a65aef916d7a13e17912186e342a8cb8e46eebd3592d144ea6e2f94c9dd8df0` / `cde4191cbd50bfafc2bc53352b8757d32b1d96a91bd3fc3e93aea12bdd8eb4b3`; canonical process-spec/config/metadata/deployment-receipt SHA-256: `dd08b778aba30343442e15924b6bcc2b79adcd83b11bb072360a562707798e48` / `4cd6d6a5fb39e24807f28109484c431df495ff82a3e81762c36a5badae46217c` / `0fba6b6dc35b1b1d6cd980d8de596d2d99f258efbf2b8596f4568b9940674f76` / `0a6589c230c92b794e21ac473d7a854cee4eecfe89092d659c934bb6c4e43261`. The outer transaction is reportable admission evidence but deliberately sets `performance_comparable=false`; no server or InfiniBand traffic was measured. Cleanup was unforced and complete: lease/lock, unit/cgroup, owned PIDs, ports, GPU, and scratch were clear. `profiler=none`; loaded SEP/PAX modules were recorded but no device had a user and neither driver was invoked. |

The prepared but never executed `glm47-kt-cpu-control-dwagon-20260719-v2`
deployment predates cgroup containment and its 15-minute metadata window has
expired. It is retained as an unused artifact only. V3 through v7 are completed
diagnostic artifacts and must not be reused. V8 is the admitted CPU-control
baseline and must also remain immutable. Hybrid1 v1 is a completed diagnostic
and must not be reused; hybrid1 v2 is the admitted one-resident baseline and
must remain immutable. Hybrid4 v1 is the admitted four-resident baseline and
must also remain immutable. Establish serving baselines against these controls
before tuning. V4's
canonical process-spec/config SHA-256 values are
`f5c954368c0e66bdb5ba494cb4dcf3176da1bc054369f0516051d71924530de5` and
`212791580cee9e8a028d5b0b8237ddfc474385bbc3455537b45d13b59ad9428d`.

### File-backed GLM-4.7 admission checkpoint

- Exo now loads kernel-validation receipts only from explicit per-GPU paths
  paired with expected raw-file SHA-256 values. The strict v1 consumer rejects
  unknown or partial evidence and derives only `kt_bf16_amx_executed_v1`.
- The real dwagon v4 receipt loaded successfully from
  `/var/lib/exo/benchmarks/glm47-kt-kernel-dwagon-20260719-v4/runtime-validation-receipt.json`
  with pinned SHA-256
  `b4f6fd1718bb3145a17c97cf8113bbfcd186416cfde3cd0fcc9eada301b78eef`.
  This was a read-only parser validation, not a new hardware benchmark.
- Exact model-contract verification is likewise available only through an
  explicit model/revision/method/path/hash binding; no newest-receipt scan or
  caller-constructed kernel receipt is accepted by the local collector.
- Launch specs serialize and canonically hash the pinned GLM-4.7 model-contract
  identity. Preflight requires the exact raw contract receipt, canonical
  contract, index, shard count, map count, and physical-byte evidence, plus a
  separate exact kernel receipt and the still-required model-level execution
  receipt. Kernel evidence alone remains fail-closed.
- The focused launch-spec, preflight, collector, and receipt-loader suites pass:
  `190 passed`. No profiler or hardware workload ran for this checkpoint.

### Pre-launch admission revalidation checkpoint

- A successful preflight now retains one immutable admission binding per rank:
  the canonical process-spec digest, exact model snapshot/contract observation,
  retained model-runtime summary, and exact kernel receipt observation. Missing,
  duplicate, cross-rank, or spec-mismatched bindings are rejected.
- Before starting any rank, the local supervisor now recomputes every launch-spec
  digest, fully verifies each distinct model snapshot against its exact contract,
  and reloads each distinct kernel receipt from its path with the pinned raw-file
  SHA-256. Shared snapshot and GPU evidence is revalidated only once per host.
- A preflight spanning multiple nodes now fails closed unless the supervisor is
  given a cluster admission barrier. That barrier runs after local verification
  and before any local process starter. Exo event/command integration for the
  verify/commit barrier remains to be implemented; until then, a distributed
  SGLang-KT launch is deliberately refused rather than only locally gated.
- Artifact verification runs in a cancellable worker process with a separate
  900-second admission timeout, before the readiness timeout begins. Cancellation
  terminates that verifier process instead of leaving an unowned scan. A normal
  supervisor stop cancels the active verification/barrier scope promptly.
  Synthetic tests prove that an admission error or timeout results in zero
  process-start attempts and a clean empty stop receipt.
- The focused admission, launch-spec, preflight, collector, supervisor, and
  kernel-receipt suites pass: `227 passed`. All model/kernel loaders were mocked
  in the new admission tests, so this checkpoint did not repeat the 62 GB live
  model scan or run a GPU, AMX, InfiniBand, benchmark, or profiler workload.
- The model-level runtime summary is retained but is not yet file-backed or
  reloadable. It therefore remains the next fail-closed blocker before a real
  GLM-4.7 launch. Kernel receipt schema v1 is treated as durable build/hardware
  capability evidence, not same-boot evidence: it lacks a boot ID, so a future
  per-boot claim requires a v2 receipt or a fresh validator run.

### File-backed model-execution receipt checkpoint

- Exo now has a strict schema-v1 GLM-4.7 model-execution receipt consumer. It
  accepts only the pinned official BF16 model, Torch `2.9.1+cu128`, CUDA `12.8`,
  SM86, exact source/build identities, wrapper coverage for layers 1-46, a
  canonical 47-by-64 expert mask, deterministic layer-one CPU/GPU routing and
  numerical evidence, and one eight-token extend plus one decode invocation.
  Capabilities are derived from validated evidence rather than accepted from the
  receipt.
- Collection retains an operator-supplied binding for the exact process-spec
  digest, validator digest, receipt path, and raw-file digest. Preflight relates
  that bound file to the independently verified model contract and kernel
  receipt. Admission reloads the same file using those independent expectations,
  so a caller-constructed Pydantic summary cannot authorize launch.
- The GLM-4.7 collector skips the overlapping legacy runtime-receipt path. A
  software integration test writes canonical JSON and exercises the real loader
  through local collection, preflight, and admission reload with only the model
  snapshot and kernel-receipt effects substituted.
- The focused launch-spec, model-receipt, integration, preflight, collector,
  admission, supervisor, and kernel-receipt suites pass: `279 passed`. Ruff and
  strict changed-module basedpyright are clean. No model load, GPU, AMX,
  InfiniBand, benchmark, or profiler workload ran for this checkpoint.
- No live model-execution receipt exists yet. The admission-grade producer and
  its real layer-one probe/full short forward remain the launch blocker.
- The historical kernel validator resolved the overlay interpreter symlink and
  recorded the base CPython path. That is valid kernel/build evidence, but it
  cannot equal a launch spec that must invoke `overlay/venv/bin/python` to load
  the pinned packages. Future receipts now preserve the invoked absolute
  executable path; the focused kernel-validator suite passes `30 passed`.
  Fresh per-host kernel receipts are required before the model proof.

### Admission-grade GLM-4.7 producer checkpoint

- The model-receipt producer now runs the pinned PP1/TP1 SGLang-KTransformers
  loader only in a disposable child bound by exact CPU cores, memory nodes, GPU
  UUID, overlay interpreter, and sanitized launch environment. Host and GPU
  headroom are checked before the heavy runtime executes, and evidence is
  released only after distributed cleanup and a clean child exit.
- The live backend requires the KTEP wrapper, `NativeMoEWrapper`, and
  `AMXBF16_MOE` on every routed layer 1-46. It runs a deterministic layer-1
  probe twice, compares CPU-only or CPU/GPU merged BF16 output against a
  memory-bounded checkpoint oracle, and traces one eight-token extend plus one
  decode through the pinned SGLang batch/forward path. Captured trace tensors
  are detached and cloned at successful return so reused runtime buffers cannot
  falsify repeat evidence.
- Parent and child independently bind the canonical process spec, exact model
  contract, every byte of all 48 model shards, raw kernel receipt, current
  executable/affinity/NUMA/GPU identity, and an exact read-only 22-file
  validator import closure. The selected NUMA nodes must use `MPOL_BIND` and
  retain the complete 62,444,175,504-byte checkpoint plus 64 GiB. Parent and
  child both reverify the immutable snapshot; the parent repeats those checks
  after child exit before create-new canonical publication.
- Direct Torch work runs under `inference_mode`. The otherwise random SGLang
  auxiliary/NCCL port is pinned to the process spec's reserved service port and
  checked again after `PortArgs` construction. The child evidence channel uses
  a sealed memfd, the retained child session leader anchors owned process-group
  cleanup, profiler/loader controls are stripped, and bytecode writes are
  disabled so root execution cannot mutate the admitted source closure.
- A separate inert CLI creates the exact CPU-control (zero resident experts) or
  hybrid (one through four residents) process spec. It strictly reparses and
  re-canonicalizes the payload, reports both raw-file and canonical-spec
  digests, and refuses output replacement, symlink traversal, or a replaced
  output parent.
- The combined producer, backend, trace, oracle, receipt, generator, admission,
  preflight, collector, supervisor, kernel-receipt, and schema suite passes
  `569 passed`. Strict source, receipt, generator, and validator-test
  Basedpyright configurations report zero errors, and Ruff is clean. This is a
  software checkpoint against a complete fake pinned-runtime surface: no model
  load, GPU, AMX, InfiniBand, benchmark, or profiler workload ran.
- Live execution against the installed pinned runtime remains the decisive
  proof. The historical dwagon kernel receipt names the base CPython rather
  than the overlay interpreter, so a fresh kernel receipt is required before
  the CPU-control model run. Multi-host admission still requires the separate
  event-sourced verify/commit/abort barrier.

### Dedicated leased live-validation harness checkpoint

- `scripts/run_sglang_kt_glm47_validation.py` prepares separate immutable
  orchestrator and exact 22-file validator trees, an immutable configuration,
  and lease metadata bound to the pinned model contract, model snapshot,
  runtime interpreter symlink chain and executable hash, build receipt, GPU
  UUID/PCI identity, CPU/NUMA allocation, ports, and HCA GIDs.
- Local validation explicitly uses `hca_requirement=metadata_only`: both QDR
  ports must retain physical `LinkUp`, expected GIDs, and rate metadata, while
  subnet `INIT` is admissible because the run sends no network traffic. Future
  distributed/comparison runs must use `active` and a benchmark-owned subnet
  manager. Every harness phase remains `performance_comparable=false`.
- Kernel-only completion cannot claim model verification or reportability.
  CPU-control and hybrid completion become reportable only after the pinned
  model-level checkpoint is produced, reparsed, admission-bound, and all owned
  process and scratch cleanup is proven.
- Cleanup records process group/start identities and an owner token, discovers
  detached descendants through `/proc`, re-signals descendants created during
  termination, waits through a quiet interval, reaps adopted children, and
  fails closed on unreadable process state or receipt-publication errors.
  Result, preflight, partial pipeline, command, and cleanup evidence survive
  validation and cleanup failures. The harness rejects active SEP/PAX or other
  profiler use; dormant loaded modules are evidence only and are never invoked.
- The harness suite passes `30 passed`; the unchanged producer/consumer suite
  separately passes `569 passed`. Strict targeted Basedpyright reports zero
  errors and Ruff lint/format checks pass. This checkpoint ran no model, GPU,
  AMX, InfiniBand, benchmark, or profiler workload.

### Leased GLM-4.7 synthetic MoE tuner checkpoint

- Commit `29e80bf9` adds a schema-v3, lease-contained RTX 3090 tuner for the
  exact pinned SGLang fused-MoE gate/up and down kernel ABI used by the four
  resident-expert KTransformers path. It enforces the complete canonical quick
  or balanced search space, deterministic route strata, fallback-bracketed CUDA
  timing, numerical rejection, immutable runtime authorization, and
  identity-safe descendant cleanup.
- The output is deliberately a candidate-only synthetic bundle. It records that
  the GLM checkpoint was not consumed, that synthetic kernel weights are
  generated, and that production AMX/GPU concurrency is not reproduced. No
  bundle can be adopted until a future tuned serving profile binds its exact
  configuration hashes and wins a matched end-to-end warm-serving comparison.
- The focused tuner suite passes `91 passed`; Ruff lint and format checks pass,
  the independent blocking review found no remaining correctness or containment
  issue, and ordinary push updated `ldyeax/exo` without force. No CUDA kernel,
  model, AMX, InfiniBand traffic, benchmark, or profiler ran at this checkpoint.

### Profiler safety incident

- A previous out-of-tree VTune SEP/PAX kernel profiler (`sep5`/`pax`) crashed
  dwagon. Treat the server as stable now, but never load or use those drivers
  again. Profiling for this work must remain driverless, using `perf` and
  ordinary application, CUDA, and runtime counters only.

### Post-crash InfiniBand control-plane restoration

- On 2026-07-19, both ConnectX-3 cards initially reported both physical links
  `LinkUp` at 40 Gb/s but subnet state `INIT`. No benchmark lease existed and no
  OpenSM process was active. Fwuff's enabled `opensm.service` had been skipped
  at boot because `/sys/class/infiniband_mad/abi_version` did not yet exist
  after the card-slot change.
- The in-tree `ib_umad` module was loaded on fwuff and its already-enabled
  `opensm.service` was started with the existing `PORTS=ALL` configuration. It
  launched exactly two `/usr/sbin/opensm` instances, one for each fwuff port
  GUID. Both ports on both hosts then reported `ACTIVE`, physical `LinkUp`, and
  40 Gb/s; dwagon LIDs are 1/1 with SM LIDs 2/4, and fwuff LIDs are 2/4 with SM
  LIDs 2/4. This was infrastructure restoration only: no payload traffic, GPU,
  model, performance measurement, or profiler ran.

### Post-reboot Nvidia device-node restoration

- On 2026-07-19, dwagon's 610.43.03 open Nvidia kernel modules were loaded and
  both RTX 3090 PCI functions were bound to `nvidia`, but `/dev/nvidia*` was
  absent and `nvidia-smi` could not communicate with the driver. Recreating the
  standard nodes with `nvidia-modprobe` restored both expected GPU UUIDs;
  `GPU-a442b72e-6727-6322-ba5d-5a9512b79886` and
  `GPU-63a7760a-6164-0758-9228-03dbf35d721c` each reported 24,576 MiB total,
  1 MiB used, and 0% utilization. Fwuff's GPU was independently idle, both QDR
  rails remained `ACTIVE` at 40 Gb/s, and fwuff's persistent OpenSM service
  remained active. This was a control-plane repair and idle health check only;
  no CUDA workload, model, network payload, benchmark, or profiler ran.

### ConnectX-3 firmware and topology checkpoint

- The user-completed hardware work is documented in `infiniband_cards.md`.
  Both HCAs now negotiate PCIe 3.0 x8 with 512-byte maximum payloads. Fwuff's
  card was backed up and cross-flashed from OEM QDR PSID `ISL1090110018`
  firmware 2.40.5030 to generic FDR PSID `MT_1090120019` firmware 2.42.5000.
  The hardware-level post-flash tests reported 52.90 Gb/s aggregate
  dwagon-to-fwuff and 52.74 Gb/s fwuff-to-dwagon over the two QDR rails, versus
  roughly 32 Gb/s before, with no error/discard increments. Preserve those as
  externally produced hardware evidence; a new leased Exo receipt still needs
  to reproduce the baseline.
- Read-only post-change verification found dwagon's HCA at `38:00.0`, NUMA 0,
  Gen3 x8 and both ports `ACTIVE` at 4X QDR. The NUMA-0 RTX 3090 is now UUID
  `GPU-63a7760a-6164-0758-9228-03dbf35d721c` at `27:00.0` with PCIe x8; UUID
  `GPU-a442b72e-6727-6322-ba5d-5a9512b79886` is at `d8:00.0`, NUMA 1, PCIe
  x16. NV4 remains active. Existing admission receipts with the prior PCI/NUMA
  identities must not be reused.
- Fwuff's enabled Ollama service had automatically occupied 23,262 MiB on its
  RTX 3090 through `llama-server` PID 9196 without a benchmark lease. The unit
  was disabled and stopped; it then reported `disabled`/`inactive`, no compute
  process, and 1 MiB idle GPU use before any Exo work. No benchmark or model
  transfer was started during these checks.

### Guarded serving and model-staging release checkpoint

- Commit `83e9e2e5` adds the fail-closed GLM-4.7 serving benchmark harness and
  receipt extensions. Its release slice passed `177` focused tests, Ruff lint
  and format checks, repository-wide Basedpyright with zero issues, and an
  independent final review. It binds the exact lease invocation, systemd
  containment, local block-device model provenance, local HCA counters, and
  bracketed fwuff host/fabric state before a result can be comparable.
- Commit `1b7e19a0` hardens the two-host model-staging transaction for an
  existing verified remote source. It transports the canonical model contract,
  retains descriptor-pinned tree identities, freezes the published snapshot,
  and records installation truth at the successful `renameat2` boundary before
  directory `fsync` or signal delivery can fail. Its release slice passed `145`
  focused tests, Ruff lint and format checks, repository-wide Basedpyright with
  zero issues, and an independent final review.
- Both commits were pushed normally to
  `ldyeax/exo:agent/linux-cuda-nccl`. Identical clean deployments for
  `1b7e19a05809a4ea252237e28f1448c25502c364` are frozen on both hosts at
  `/var/lib/exo/deployments/linux-cuda-nccl-1b7e19a05809a4ea252237e28f1448c25502c364`.
  Their staging-script SHA-256 is
  `04e6d1182d4a77cf1a92df54646e0c6549d0fb50883203859d85d079b687d3bf`.

### GLM-4.7 BF16 local-NVMe staging receipt

- The leased transaction
  `/var/lib/exo/benchmarks/glm47-bf16-local-stage-dwagon-20260719-v1`
  completed normally from 2026-07-20 02:48:05 through 03:04:58 UTC. It copied
  the exact fwuff source through dwagon's read-only NFS view into
  `/var/lib/exo/models/zai-org--GLM-4.7-Flash--7dd20894a642a0aa287e9827cb1a1f7f91386b67`,
  then atomically published a root-owned, non-writable tree containing 117
  files and 62,465,293,519 bytes.
- Source, local, and final remote verifications are exactly equal: revision
  `7dd20894a642a0aa287e9827cb1a1f7f91386b67`, 48 shards,
  31,221,488,576 indexed bytes, and model-contract SHA-256
  `4e7333f341ddc5855aa4253d454e3210d84427fae0159eb956104ff00c437479`.
  Fwuff's snapshot was preexisting and unchanged; no remote install occurred.
- Cleanup was confirmed and unforced. Both owned fwuff helpers exited, no
  `.stage` path, lease, or held lock remained, and the command returned zero.
  Staging deliberately records `reportable=false`,
  `performance_comparable=false`, and no performance claim because it is an
  artifact-correctness transaction, not a benchmark. Result/manifest/runtime
  SHA-256 values are
  `4402e93021291b8138660008b1bd7d27cf77c8dc9c31ca29ac1b51d083ae49d8`,
  `a60105f975aa667e9372205f53345c4571f4c11c42bb2829aaf3258c97dce0b6`,
  and `d82b90508b00ef0a1399e960a43de51f4a979a695426fe58b14dd98d2d561344`.
- Coarse owned-process counters were used only to monitor progress. They are not
  benchmark evidence. The 10 GbE NFS path visibly left the repaired dual-rail
  fabric unused, motivating a separate artifact-transport implementation:
  content-addressed revision caching, concurrent per-shard IPoIB/RDMA transfer
  over both rails, optional 10 GbE overflow, and placement-aware transfer of
  only shared tensors plus assigned layers/experts. Safetensors that mix
  placements within one file require range-aware loading or a one-time
  placement-aligned repack. Start with registered host buffers and pinned-host
  GPU staging; do not assume GPUDirect RDMA support on RTX 3090/ConnectX-3.

### GLM-4.7 BF16 local warm-serving iterations

- `glm47-kt-serving-local-dwagon-20260719-v8` completed the semantic sanity
  request, two interleaved warmup pairs, and three measured request pairs with
  the 1,024-input/32-output prefill and 128-input/128-output workloads. It was
  an intentionally superseded 16-core, single-NUMA diagnostic. Live SGLang
  logging showed roughly 2.1-2.4 decode tok/s, but the harness lost its owner
  token during post-run cleanup because `setproctitle` rewrote the environment.
  No immutable measurement was published, and its failed-closed receipt is not
  performance evidence.
- `glm47-kt-serving-local-dwagon-20260719-v9` repeated the same workload with
  all 112 physical cores, both NUMA nodes, two 56-thread AMX pools, one RTX
  3090, and four resident GPU experts per routed layer. All requests completed,
  with live logs around 6-7.5 decode tok/s. The pinned SGLang SIGTERM handler
  drained all requests and then intentionally SIGKILLed its own process tree;
  the harness rejected that post-drain `-9` return before serializing the
  measurement. Cleanup succeeded, but no exact performance result survives.
- `glm47-kt-serving-local-dwagon-20260719-v10` is the first preserved
  full-CPU engineering measurement. Semantic sanity produced exactly
  `EXO_SANITY_OK`. All measured requests had zero cached tokens and stable
  output hashes. The 1,024/32 samples had median total latency 6.418898395 s,
  median TTFT 1.251576989 s, and median output rate 6.066921877 tok/s. The
  128/128 samples had median total latency 19.086266795 s, median TTFT
  1.230329836 s, and median output rate 7.112858950 tok/s; individual decode
  rates were 6.728878438, 7.539495531, and 7.112858950 tok/s. TTFT includes
  HTTP and queue time through the first streamed output event.
- V10 bound cores 0-111, NUMA nodes 0/1, two 56-thread AMX pools, BF16 weights,
  four resident GPU experts on every MoE layer, and only UUID
  `GPU-63a7760a-6164-0758-9228-03dbf35d721c`. It therefore exposed the full
  physical CPU but not both local GPUs: the current GLM serving contract is
  TP1/single-GPU. The measurement records affinity and allocation rather than
  sampled CPU/GPU saturation. Server logs reported 7.77 GB GPU weight use,
  0.21 GB BF16 KV cache allocation, and 14.78 GB remaining GPU memory. Fwuff
  stayed idle and the InfiniBand counters confirm that no payload traversed the
  fabric.
- The immutable v10 measurement is
  `/var/lib/exo/benchmarks/glm47-kt-serving-local-dwagon-20260719-v10/warm-serving-measurement.json`
  with SHA-256
  `2900337411fc3088428b905bedc2d11b866fbacb0464bc9c083de9ea40d9bea8`.
  Its normal authoritative performance receipt remains failed closed: final
  validation compared a JSON list with the producer's equivalent tuple after
  all inference and cleanup had passed. Commit `6040820a` fixes that comparison
  using canonical JSON and adds positive plus deployment-substitution tests;
  65 harness tests, Ruff, and repository-wide Basedpyright pass. The failed
  receipt is preserved rather than overwritten, so v10 is engineering evidence
  and the next run must produce the first authoritative performance receipt.

### GLM-4.7 BF16 three-stage pipeline iterations

- PP3 v1 exposed a concrete transport setup defect rather than a model defect:
  Gloo selected IPv6 locally and IPv4 remotely. Binding both Gloo and NCCL to
  the intended per-host Ethernet bootstrap interface corrected it without
  restaging weights or rebuilding either runtime.
- V2 and v3 use stages `[0,16)`, `[16,32)`, and `[32,47)` with full physical
  core allocations 56/56/60, NUMA nodes 0/1/0, one RTX 3090 per stage, four
  resident GPU experts per routed layer, BF16 CPU experts, and dual-QDR NCCL.
  Every timed request follows exact semantic sanity plus two warmups and three
  samples for both 1,024/32 and 128/128 workloads.
- V3 is the first fully passing transaction. Its median 128/128 client decode
  rate was 6.290703294 tok/s, median TTFT was 0.943137972 s, and end-to-end
  output rate was 6.036021923 tok/s. Its 1,024/32 median client decode rate was
  7.264835224 tok/s, TTFT was 1.281804928 s, and end-to-end output rate was
  5.766342756 tok/s. The two rails carried nearly identical payload and all
  health/error counter deltas were zero.
- V3 is 11.56% slower in decode than the earlier one-GPU dwagon v10 engineering
  baseline. This is not yet a controlled topology comparison because the
  harnesses and instrumentation differ, but it establishes that the initial
  balanced-layer PP3 partition pays more pipeline/transport overhead than it
  recovers from three concurrent CPU/GPU stages at batch one. The next runs
  therefore test NUMA/HCA boundary placement and expert residency before adding
  more machinery.
- V4 moved dwagon rank 1 from NUMA1/GPU1 to the HCA-local NUMA0/GPU0 placement
  while moving rank 0 in the opposite direction. Decode fell 0.56% to
  6.255591493 tok/s and prefill-heavy end-to-end output fell 0.69% to
  5.726548780 tok/s. Rail payload was effectively identical to v3, confirming
  that roughly 54.4 MB of cross-host traffic per complete run is not enough for
  the removed UPI hop to matter. Retain v3's mapping and change expert residency
  next.
- V5 restored the v3 mapping and raised resident experts from E4 to E16. Decode
  improved 4.18% to 6.553438809 tok/s and TTFT fell to 0.877620794 s, while the
  1,024/32 end-to-end output rate regressed 8.26% to 5.289909964 tok/s. GPU
  weights occupied only 5.87-6.63 GB per stage, leaving room to sweep E32. The
  result also motivates a later full-GPU prefill threshold control rather than
  assuming one expert placement is optimal for both phases.
- V6 raised residency to E32. It improved decode another 5.66% over E16 to
  6.924257787 tok/s and reversed the prefill regression, reaching 6.215342096
  prefill-heavy output tok/s. Relative to E4 v3, those are 10.07% and 7.79%
  improvements. Each stage still had roughly 12.3-13.1 GB free after weights,
  so E48 is a valid final upper sweep point before changing the partition.
- V7 was a command-only setup failure caused by a mistyped fwuff overlay ID;
  both started local ranks cleaned up and no remote model process launched. V8
  corrected that one path and completed E48. Decode improved another 3.46% over
  E32 to 7.163711287 tok/s, while prefill-heavy output improved 8.53% to
  6.745452171 tok/s. Relative to E4 v3, the gains are 13.88% and 16.98%.
  E48 leaves 7.9-8.8 GB free per stage and becomes the partition-control
  baseline.
- V9 made no performance attempt because its command placed fwuff's existing
  runtime overlay under the model-volume mount. V10 corrected only that path
  and moved one routed layer from rank 1 to fwuff, producing stages `[0,16)`,
  `[16,31)`, and `[31,47)`. Against v8, decode improved 0.91% to 7.228700241
  tok/s and prefill-heavy output improved 3.43% to 6.976789422 tok/s. Both rails
  remained balanced and clean. Retain 16/15/16 as the best observed split, but
  stop the partition sweep: the batch-one decode gain is below 1%, so a broader
  split search is lower value than local PP2/TP2 controls.

### Distributed expert-parallel research result

- KTransformers `kt_ep` routes experts between CPU and GPU inside one process;
  it is not true cross-rank expert sharding. Its current global/local expert-ID
  assumptions also prevent safely combining it with native distributed EP.
- The pinned SGLang runtime has native `--ep-size`. Its standard `none`
  dispatcher masks non-local experts and NCCL-reduces partial expert results,
  making it the shortest correct proof path on SM86. DeepEP/DeepGEMM are not an
  appropriate RTX 3090 cross-host baseline.
- Start with `allenai/OLMoE-1B-7B-0924` revision
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, but do not launch the four-arm
  TP2/EP1 versus TP2/EP2 comparison yet. Pinned SGLang incorrectly applies a
  full-width 2,048-element Q/K RMSNorm to 1,024-wide TP2 shards. First shard the
  norm weights and all-reduce the paired FP32 Q/K sum-of-squares once per layer,
  then rebuild both runtimes and prove TP2 parity against full-vector RMSNorm.
  After that fix, keep `--moe-a2a-backend none` and `--moe-runner-backend
  triton` for the first local and cross-host proof.
- GLM-4.7 has 64 routed experts. BF16 EP2 still leaves about 25.875 GiB of
  routed weights per rank before shared weights, activations, and KV cache, so
  it cannot fit a 24 GiB RTX 3090. EP3 is not a valid even expert/head topology;
  EP4 is the first plausible native-BF16 distributed layout. OSCAR remains a
  later KV-cache compression and placement candidate, not an expert dispatcher.

## Pending tests

1. Run controlled dwagon-only GLM PP2/TP1 and TP2/PP1 optimization ladders with
   both NUMA CPU pools and both RTX 3090s, using the same semantic sanity and
   1,024/32 plus 128/128 workloads as PP3.
2. Add feasible mixed TP/PP trials after the local controls, and implement the
   operational collective stage-local receipt producer before claiming
   production Exo PP3 admission. The current engineering harness directly
   launches the audited three-stage runtime and does not pretend that gap is
   closed.
3. Fix and prove TP-sharded OLMoE Q/K RMSNorm, then run native SGLang TP2/EP1
   versus TP2/EP2 locally and across InfiniBand. Use it to measure whether true
   weight-sharded EP can amortize all-reduce at the relevant batch and sequence
   sizes.
4. Convert the verified GLM BF16 source to AMXINT8 only after BF16 placement and
   hybrid execution comparisons; keep packed-GPU mode disabled initially.
5. Resume exact staging, deterministic TP1, and strict TP=2 for Qwen3-Coder 30B
   A3B, followed by Qwen3.5 35B A3B.
6. After the first larger-model correctness proof, complete at least five
   distinct dwagon-only optimization runs and five distinct dwagon-plus-fwuff
   InfiniBand optimization runs. Each run needs repeated samples and a recorded
   hypothesis/lesson. Keep a matched-artifact comparison workload; when a
   different exact-revision HF quantization or format wins one track, add a
   quality-gated matched-format control so topology and format effects remain
   separable.
7. Re-run the preserved QDR receipts after the ConnectX-5 EDR hardware swap.
