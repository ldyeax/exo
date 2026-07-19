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
- **STAGE PASS:** exact model acquisition and cross-host manifest equality
  passed; staging makes no inference-performance claim.
- **EXPECTED FAIL:** a fail-closed check rejected an invalid or incomplete
  contract and exposed a defect that was subsequently fixed.
- **BLOCKED:** the requested check could not execute in the current toolchain.

## Current summary

- Latest published proof-harness source: clean commit
  `57a57e1221a5f1199cc90e9dc9391ec925395f2c`.
- Latest completed live-benchmark source: clean commit
  `6da455c8e3eb5e21d09812f10e44a31712545816`.
- Completed ladder rungs: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B,
  GPT-OSS 20B, and GLM-4.7 Flash.
- Latest result: GLM-4.7 Flash TP=2 passed exact TP1 output equality, two-rank
  NCCL initialization, payload on both QDR rails, clean HCA health counters,
  instance deletion, process cleanup, and resource release.
- Next model: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` at revision
  `6e302ea604ad9ab206367e2c501d1571023e7b6d`.

## Automated validation

These suites overlap. Their passing counts must not be summed into a unique
test total.

| Scope | Result | Evidence |
| --- | --- | --- |
| Corrected SGLang launch, snapshot, and preflight slice | **PASS** | 94 passed |
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

## Pending tests

1. Run exact staging, deterministic TP1, and strict TP=2 for Qwen3-Coder 30B
   A3B, followed by Qwen3.5 35B A3B.
2. Re-run the preserved QDR receipts after the ConnectX-5 EDR hardware swap.
