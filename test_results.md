# Exo Phase 1 Test Results

Last updated: 2026-07-19

This is the human-readable test ledger for the dwagon/fwuff Linux CUDA, NCCL,
and InfiniBand workstream. `FWUFFYDWAGON.md` remains the implementation plan;
the JSON receipts under `/var/lib/exo/benchmarks` are the machine-readable
source of truth. Update this file in the same commit that records each new test.

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

- Latest published proof-harness source: clean commit
  `57a57e1221a5f1199cc90e9dc9391ec925395f2c`.
- Latest completed live-benchmark source: clean commit
  `6da455c8e3eb5e21d09812f10e44a31712545816`.
- Completed ladder rungs: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B,
  GPT-OSS 20B, and GLM-4.7 Flash.
- Latest live result: GLM-4.7 Flash TP=2 passed exact TP1 output equality,
  two-rank NCCL initialization, payload on both QDR rails, clean HCA health
  counters, instance deletion, process cleanup, and resource release.
- Latest kernel result: independent dwagon and fwuff native runtimes each
  passed the leased SM86 CUDA plus AMX-BF16 validator and claimed only
  `kt_bf16_amx_executed_v1`. Both runs cleaned up without force.
- Latest artifact result: the official GLM-4.7 Flash BF16 snapshot now has a
  packaged contract covering every launch-relevant file, all 48 indexed
  shards, and their Hugging Face revision metadata. A leased full rehash
  verified that 62.4 GB contract on the shared read-only snapshot.
- Historical Ornith AMXINT8 conversion and serving receipts were recovered and
  hashed below. They inform the GLM hybrid-runtime work but are not Exo tests.
- Next work: bind file-backed build/install/kernel/model receipts into model
  admission, then run the real loader/short-forward gate before CPU-only and
  mixed AMX-BF16/RTX-3090 PP1 correctness controls. The Qwen3-Coder ladder rung
  is deferred until that hybrid path has trustworthy model execution receipts.

## Automated validation

These suites overlap. Their passing counts must not be summed into a unique
test total.

| Scope | Result | Evidence |
| --- | --- | --- |
| Corrected SGLang launch, snapshot, and preflight slice | **PASS** | 94 passed |
| GLM-4.7 Flash target profile, preflight, snapshot verifier, and local process supervisor | **PASS** | 148 passed on 2026-07-19, including 25 supervisor lifecycle tests; inherited NCCL/SGLang isolation and shutdown/error-race receipt regressions included |
| Reproducible GLM-4.7 SGLang-KTransformers source integration | **PASS** | Exact clean-base and partially initialized result-state replays produced SGLang `41d4d300a21fd2f486681d56f1017789dfb355fe` and KTransformers `7e70d7518edd26af6a0638593037d68c9b6bd6bf` without mutating rejected dirty parent or dependency sources |
| Pinned dwagon GLM-4.7 native runtime build | **PASS (artifact)** | Build ID `44df90375778d5af6b735a696a730efaa5b3a8de1f5081517892636fe1616a69`; receipt SHA-256 `a29b56a9a703d19b99c0f92adb591452b7e0899eb1ffabbf1e1522a0e8555e64`; validator reconstructed the complete source, bootstrap, toolchain, command, layout, and wheel provenance |
| Immutable dwagon runtime overlay | **PASS (artifact)** | Install ID `91418a4ab5c6bc0e3896cbf7021ba6eb1c81010ac702dd6924391e5a3a048b42`; receipt SHA-256 `51a6fa03a675f10e1791e3a15dec51de77b34b9ea730b11b6fc0d84872f21eb5`; the old 9.9 GB base runtime remains untouched |
| Pinned fwuff GLM-4.7 native runtime build | **PASS (artifact)** | Build ID `e21ef087b1c50cf961339e1bd1a2e1a3f60047579f811385614297de6a2abfc9`; receipt SHA-256 `051b78a5238adac99721eb268c95d8ab5e8721d8c1c61114bbda967468433dfb`; native KT wheel/extension SHA-256 `f57c574cc190f8817a51cf0b08c5f2165e2f761c3a1b2999560abcbdcc792d45` / `b2f60ec18aba53223e27cfd925f2c23083a281c109cf53ca06f1ac98bff6e99b` |
| Immutable fwuff runtime overlay | **PASS (artifact)** | Install ID `82d20634f743ed87ae9cc71f2b7f4936d9451363db1ca46a207218f22de51ef8`; receipt SHA-256 `fe57f9fe10160ebf2f0a0ba3731e69c8ddb4608841f07880bac2ff6bd0640eb4`; its old base runtime remains untouched |
| Leased two-host CUDA/AMX kernel validation | **PASS** | Dwagon v4 and fwuff v1 independently passed exact provenance, SM86 BF16 CUDA math, AMX-BF16 qlen 1/16, and the bidirectional non-default CUDA-stream bridge; each claimed only `kt_bf16_amx_executed_v1` and cleaned up without force |
| Official GLM-4.7 Flash BF16 model contract | **PASS (artifact)** | The packaged contract binds 54 launch-relevant files, 48 indexed shards totaling 62,444,175,504 bytes, exact Hugging Face metadata, tokenizer/template inputs, and absence of executable remote-code files; leased live verification returned 0 and cleaned up unforced |
| Focused GLM-4.7 source, build, overlay, model-contract, validator, launch, and preflight suite | **PASS** | 277 tests passed on 2026-07-19, including exact packaged-contract pinning, the repaired required-profile launch-plan fixture, isolated validator import, and canonical/raw PyTorch GPU UUID coverage |
| GLM-4.7 packaged-contract wheel inclusion | **PASS (artifact)** | `uv build --wheel` produced `exo-0.3.70-py3-none-any.whl`; its package contains the exact 16,218-byte `exo/worker/sglang_kt/manifests/glm47_flash_bf16_7dd20894.json` resource |
| Changed GLM-4.7 Flash Python files, strict targeted type checks and Ruff | **PASS** | Three targeted Basedpyright configurations reported 0 errors; repository-wide `ruff check` passed; all 16 changed Python files passed `ruff format --check` on 2026-07-19 |
| Repository-wide Basedpyright in the existing `.venv` | **BLOCKED** | The environment cannot resolve installed project dependencies (including `httpx`, AnyIO, and pytest), producing dependency-driven diagnostics across the untouched tree; `uv run` could not complete the pinned MLX wheel acquisition |
| Repository-wide pytest collection | **BLOCKED** | `tests/conftest.py` imports unavailable `exo_tools`; collection stopped before tests ran |
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
| Fresh full-repository pytest collection | **BLOCKED** | Root collection lacks external `exo_tools`; adding `tools/src` exposes 23 pre-existing collection/import failures |
| Required `uv run basedpyright` and `nix fmt` | **BLOCKED** | `uv` and Nix are not installed on dwagon; direct full Basedpyright has environment-wide dependency/stub failures |

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
  KTransformers `7e70d7518edd26af6a0638593037d68c9b6bd6bf` and SGLang
  `41d4d300a21fd2f486681d56f1017789dfb355fe`, with fatal registration,
  structured 46-layer wrapper/mask coverage, and fail-closed exact loading of
  every resident expert's gate, up, and down projections from the checkpoint.
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
- The final clean prepared source is
  `/var/lib/exo/sources/ktransformers-glm47-7e70d75`, at KTransformers
  `7e70d7518edd26af6a0638593037d68c9b6bd6bf`, embedded SGLang
  `41d4d300a21fd2f486681d56f1017789dfb355fe`, llama.cpp
  `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`, and pybind11
  `bb05e0810b87e74709d9f4c4545f1f57a1b386f5`.
- The final native dwagon build is
  `/var/lib/exo/runtimes/glm47-sglang-kt/dwagon/44df90375778d5af6b735a696a730efaa5b3a8de1f5081517892636fe1616a69`.
  Its build ID is the final path component and its receipt SHA-256 is
  `a29b56a9a703d19b99c0f92adb591452b7e0899eb1ffabbf1e1522a0e8555e64`.
  Static validation reconstructed the complete canonical build inputs and ID.
- The immutable dwagon overlay is
  `/var/lib/exo/runtimes/glm47-sglang-kt-overlay/dwagon/91418a4ab5c6bc0e3896cbf7021ba6eb1c81010ac702dd6924391e5a3a048b42`.
  Its install ID is the final path component and its receipt SHA-256 is
  `51a6fa03a675f10e1791e3a15dec51de77b34b9ea730b11b6fc0d84872f21eb5`.
  The overlay pins its base runtime and three newly built wheels without
  modifying the old 9.9 GB environment.
- The independent fwuff native build is
  `/var/lib/exo/runtimes/glm47-sglang-kt/fwuff/e21ef087b1c50cf961339e1bd1a2e1a3f60047579f811385614297de6a2abfc9`,
  with build-receipt SHA-256
  `051b78a5238adac99721eb268c95d8ab5e8721d8c1c61114bbda967468433dfb`.
  Its host-native KT wheel SHA-256 is
  `f57c574cc190f8817a51cf0b08c5f2165e2f761c3a1b2999560abcbdcc792d45`;
  the pure-Python KTransformers and SGLang wheels match dwagon byte-for-byte.
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
| `glm47-kt-kernel-fwuff-20260719-v1` | **PASS** | Fwuff independently reproduced the same four deterministic numerical errors using its own native build and GPU UUID. Local and remote receipt/log hashes matched, no remote validator survived, and cleanup was unforced. Receipt/result/manifest SHA-256: `efe770fc84f7e28614e0d2c9ff3ca3b9e9337d511fbd78d19c5a3bade65bb782` / `093ec960fc39aaace7fc99b3abe2d6daaf02f73aba7277f14c9c06da3a7c9d3d` / `be8a450a83a424a4d39604bf2f658d04bedf707da92d58a8418fa5843fc657d4`. Capability is only `kt_bf16_amx_executed_v1`. |

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

### Profiler safety incident

- A previous out-of-tree VTune SEP/PAX kernel profiler (`sep5`/`pax`) crashed
  dwagon. Treat the server as stable now, but never load or use those drivers
  again. Profiling for this work must remain driverless, using `perf` and
  ordinary application, CUDA, and runtime counters only.

## Pending tests

1. Finish the file-backed model-level validation receipt and launch-time
   revalidation chain around the completed exact snapshot and kernel receipt
   bindings, then run a real one-layer loader and short-forward check. Keep
   kernel-level capability evidence separate from model-level launch admission.
2. Run the fail-closed 0-expert BF16 CPU-routed control and the mixed 1/4
   resident-GPU-expert controls with complete 46-layer wrapper and routing
   evidence.
3. Convert the verified BF16 source to AMXINT8 only after BF16 parity and hybrid
   execution evidence pass; keep packed-GPU mode disabled initially.
4. Resume exact staging, deterministic TP1, and strict TP=2 for Qwen3-Coder 30B
   A3B, followed by Qwen3.5 35B A3B.
5. After the first larger-model correctness proof, complete at least five
   distinct dwagon-only optimization runs and five distinct dwagon-plus-fwuff
   InfiniBand optimization runs. Each run needs repeated samples and a recorded
   hypothesis/lesson. Keep a matched-artifact comparison workload; when a
   different exact-revision HF quantization or format wins one track, add a
   quality-gated matched-format control so topology and format effects remain
   separable.
6. Re-run the preserved QDR receipts after the ConnectX-5 EDR hardware swap.
