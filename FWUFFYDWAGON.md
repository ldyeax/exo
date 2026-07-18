# FWUFFYDWAGON: GLM-5.2 on dwagon and fwuff

## Objective

Bring Exo to a working, efficient Linux/NVIDIA deployment that can serve GLM-5.2 across `dwagon` and `fwuff`, with CPU AMX expert execution and InfiniBand transport where they improve performance. Target one interactive coding-agent session at up to 262,144 total tokens before attempting GLM-5.2's one-million-token limit.

## Hardware Baseline

### dwagon

- Dual-socket Intel Xeon 8570-class system: 112 cores/224 threads total, AMX BF16/INT8, AVX-512 BF16/FP16/VNNI.
- 768 GB installed DDR5-6000 RDIMM; the current CPU/firmware may train it below the DIMM rating.
- Two RTX 3090 GPUs at `16:00.0` and `27:00.0`, both currently attached to NUMA 0, with an active NV4/NVLink connection. Preserve the bridge when testing slot changes.
- Dual-port Intel X710 10 GbE on NUMA 1.
- NVIDIA driver 610.43.03 and MLX CUDA 13 are healthy. The ConnectX-3 HCA is PCIe 3.0 x8 on NUMA 1.

### fwuff

- Single-socket Intel Xeon system: 60 cores/120 threads, AMX BF16/INT8 and the same relevant AVX-512 capabilities.
- 256 GB installed DDR5-6000 RDIMM; the current CPU/firmware may train it below the DIMM rating.
- One RTX 3090 at PCIe 4.0 x16.
- Dual-port Intel X550 10 GbE.
- The ConnectX-3 HCA is currently constrained to PCIe 3.0 x4; move it to the available x16-wired slot before expecting dual-rail scaling.
- `/mnt/sanic` is a 3x2 TB RAID0 model volume.

### Interconnect

- One MCX354A ConnectX-3 VPI QDR card is installed in each server. Both ports are active at 4X QDR, 40 Gb/s raw per port.
- Planned upgrade: a matched pair of Mellanox `MCX555A-ECAT` ConnectX-5 VPI single-port EDR/100GbE cards and an EDR-rated QSFP28 100G direct cable. Preserve all current QDR results as the pre-upgrade baseline.
- Use the kernel `mlx4_core`/`mlx4_ib` drivers, `rdma-core`, `perftest`, Mellanox Firmware Tools, and one OpenSM instance per disconnected direct-connect rail. Current MLNX_OFED releases no longer support ConnectX-3.
- Keep 10 GbE as the management/control plane. QDR ports do not aggregate automatically; select and benchmark both rails explicitly.
- Default to host-staged NCCL/InfiniBand. RTX 3090 GPUDirect RDMA is not an officially supported configuration; test `nvidia-peermem` only as an optional A/B path.
- NCCL 2.28 GIN requires ConnectX-4 or newer. On these ConnectX-3 cards, NCCL crashed in `ncclNetInit` while initializing its GIN plugin; `NCCL_GIN_ENABLE=0` alone was insufficient. Set both `NCCL_GIN_ENABLE=0` and `NCCL_GIN_TYPE=0`. GIN is a device-side API, not the ordinary host-launched NCCL/verbs transport, so disabling it loses no standard IB collective capability. `NCCL_NET=IB` selects the generic IB transport and fails rather than silently falling back to Socket; `NCCL_NET_GDR_LEVEL=LOC` deliberately keeps this baseline host-staged. [NCCL network selection](https://github.com/NVIDIA/nccl/blob/ae7aed194dc63c65d1bf5c0385ba3d68d3b64c8c/src/plugin/net.cc#L351-L380) [NCCL GDR level](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/docs/userguide/source/env.rst#L1113-L1143)
- Verified `ib_write_bw`: 28.46 and 28.56 Gb/s single-rail, 29.80 Gb/s concurrent aggregate. The aggregate ceiling is consistent with `fwuff`'s PCIe x4 HCA placement.

## Model and Memory Findings

- GLM-5.2 has about 750B total and 40B active parameters, 78 layers, 256 routed experts with top-8 routing plus a shared expert, MLA, DSA, IndexShare, and one MTP layer.
- BF16 weights are roughly 1.51 TB and cannot fit in the combined one-TiB host memory. The official FP8 checkpoint is roughly 704 GiB and does fit, but leaves limited headroom.
- The structured config audit used Hugging Face commit `ba978f7d347eaf65d22f1a86833408afdb953541`. Treat that as the audited configuration reference, not an automatic download choice: pin the complete checkpoint to one exact commit at acquisition time and bind its raw `config.json` hash into every runtime validation receipt. [Audited GLM-5.2-FP8 config](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json)
- Current Exo/MLX does not implement GLM-5.2's cross-layer IndexShare behavior correctly. Do not advertise GLM-5.2 on the MLX engine until parity with the changes in MLX-LM PR 1410 is demonstrated.
- The production path should let Exo orchestrate an external SGLang + KTransformers engine rather than forcing GLM-5.2 through an unsuitable MLX-CUDA implementation. AMX is still a target, but it must be proven by the executed kernel backend rather than inferred from CPU flags or a startup label.
- KTransformers v0.6.3 is the first release with explicit GLM-5.2 support. Treat KTransformers `ce7c3ddbe93f7ac1f992375eed54058bbc512646` and its SGLang fork `8b636f9008dbad58c0a8e481b03e794739e6c146` as audited base revisions, not launch-ready production pins. The SGLang fork requires the `transformers-kt==5.6.0.post1` distribution, whose imported `transformers` module version must also be verified. The existing `/mnt/sanic/kk2` checkouts predate this support and must not be used without updating and revalidation. [Pinned SGLang dependency](https://github.com/kvcache-ai/sglang/blob/8b636f9008dbad58c0a8e481b03e794739e6c146/python/pyproject.toml#L62-L74)
- The exact base runtime is blocked on RTX 3090/SM86: stage-local TP groups still broadcast from global rank 0, the GLM NSA indexer invokes a DeepGEMM path whose configured architectures exclude SM86, and FP8 KV selects FlashMLA builds that target SM90a or newer. Require a committed runtime fork plus a PP=3 initialization test and GLM-5.2 NSA prefill/decode short-forward receipt on each GPU before full-model work. [Global-root KT broadcast](https://github.com/kvcache-ai/sglang/blob/8b636f9008dbad58c0a8e481b03e794739e6c146/python/sglang/srt/layers/moe/kt_ep_wrapper.py#L1965-L2011) [NSA indexer](https://github.com/kvcache-ai/sglang/blob/8b636f9008dbad58c0a8e481b03e794739e6c146/python/sglang/srt/layers/attention/nsa/nsa_indexer.py#L359-L458) [DeepGEMM architectures](https://github.com/kvcache-ai/sglang/blob/8b636f9008dbad58c0a8e481b03e794739e6c146/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py#L11-L34) [FlashMLA build targets](https://github.com/kvcache-ai/sglang/blob/8b636f9008dbad58c0a8e481b03e794739e6c146/sgl-kernel/cmake/flashmla.cmake#L20-L30)
- KTransformers also needs CPU fixes before the preferred topology: physical NUMA IDs are incorrectly used as dense vector indices, worker threads can replace inherited affinity with overlapping full-node core selections, and the audited FP8 expert path dispatches AVX-512 rather than AMX tiles. Require patched physical-NUMA mapping, explicit process CPU sets with binding failures made fatal, and an executed-AMX backend receipt. [Worker-pool NUMA code](https://github.com/kvcache-ai/ktransformers/blob/ce7c3ddbe93f7ac1f992375eed54058bbc512646/kt-kernel/cpu_backend/worker_pool.cpp#L268-L441) [FP8 dispatch](https://github.com/kvcache-ai/ktransformers/blob/ce7c3ddbe93f7ac1f992375eed54058bbc512646/kt-kernel/operators/amx/la/amx_raw_kernels.hpp#L528-L616)

## KV Cache Conclusion

The claim that every GPU or memory node always needs a complete KV cache is false. Each pipeline stage stores cache only for its local layers. Exo's current MLX tensor sharder also splits Llama, Qwen softmax-attention, and GPT-OSS KV heads; Qwen linear-attention state is split along tensor dimensions. Some latent/index state, prefix-cache entries, vision state, and model-specific tensors can still be replicated within a stage, so the initial GLM topology uses TP=1 and every model still needs measured VRAM accounting.

For GLM-5.2, estimated per-token, per-layer cache is about 1,284 bytes in BF16 or 788 bytes in FP8. At 256K tokens, the preferred three-stage split is approximately 9.4, 8.8, and 6.3 GiB in BF16 or 5.8, 5.4, and 3.9 GiB in FP8 before allocator padding and workspace. The audited FP8 KV implementation is not an SM86 path, so begin patched-runtime validation at short context with a proven Ampere-compatible backend; enable FP8 KV only after its own short-forward receipt. Use concurrency 1, bounded prefix caching, and conservative resident-GPU expert counts. Consider OSCAR INT2 or host spill only after GLM-specific correctness validation. Earlier Ornith runs left 4.8-10 GB free per RTX 3090, so 2 GB free is not an unavoidable limit.

## Preferred Runtime Topology

Run three logical pipeline resources with TP=1:

| Rank | Host/resource | Layers | CPU expert weight estimate |
| --- | --- | --- | --- |
| 0 | dwagon, NUMA 0, RTX 3090 at `16:00.0` | 0-29 (30) | about 243 GiB |
| 1 | dwagon, RTX 3090 at `27:00.0`; target NUMA 1 after hardware reshuffle | 30-57 (28) | about 252 GiB |
| 2 | fwuff, NUMA 0, RTX 3090 at `6a:00.0` | 58-77 (20) | about 180 GiB |

Set `SGLANG_PP_LAYER_PARTITION=30,28,20`. Starts 0, 30, and 58 are valid full-indexer boundaries. This ordering keeps rank 0 to rank 1 traffic local (and eligible for NVLink) and crosses InfiniBand only between rank 1 and rank 2. Start with zero resident GPU experts, then profile 1, 2, and 4 experts per stage. Disable MTP and deferred expert work until correctness passes.

The current hardware enumeration places both dwagon GPUs on NUMA 0. Do not claim the preferred production topology is NUMA-local until a slot reshuffle or measured cross-socket fallback validates it. The unpatched KTransformers runtime must not receive a single-node list such as `--kt-numa-nodes 1`: its physical-ID indexing can go out of bounds and its distributor can bind memory to NUMA 0 instead. The patched runtime receipt must cover the exact CPU and memory-node assignment for every stage.

If three logical resources cannot be launched reliably, use PP=2/TP=1 with `SGLANG_PP_LAYER_PARTITION=38,40`. That fallback requires swapping one eight-DIMM set so dwagon has 640 GB and fwuff has 384 GB; one dwagon GPU remains idle. Do not use equal `39,39`, which begins the second stage at an unsafe shared-indexer boundary.

## Exo Implementation Plan

### Phase 1 implementation status

- Added an additive `MlxNcclInstance` restricted to multi-rank, full-layer Tensor shards on `MlxCuda` nodes.
- Added one shared, real IPv4 rank-0 coordinator address and the MLX 0.32 NCCL environment contract (`MLX_RANK`, `MLX_WORLD_SIZE`, `NCCL_HOST_IP`, `NCCL_PORT`).
- Preserved root-owned `CUDA_VISIBLE_DEVICES` and NCCL transport policy in runner children. Runner bootstrap detects eligible `mlx4_core` devices and defaults both GIN controls to `0`, while preserving explicit operator overrides. Its `NCCL_IB_HCA` parser now honors NCCL include/exclude, exact-name, and port syntax, so an explicit future mlx5-only selection is not disabled merely because a ConnectX-3 remains installed.
- Made MLX control collectives NCCL-safe: GPU/default stream instead of a forced CPU stream, and `int32` rather than unsupported boolean cancellation reductions.
- Exposed `MlxNccl` through placement previews and the benchmark harness while preserving the existing `both` default as Ring+JACCL. Valid Tensor placements are returned normally; Pipeline+NCCL produces an explicit incompatibility because MLX NCCL does not implement `send`/`recv`.
- Added `scripts/mlx_nccl_smoke.py`, also deployed as `fwuff:/root/mlx-nccl-smoke/nccl_smoke.py`. Port 1, port 2, and merged dual-port runs all passed `all_sum`/`all_gather` correctness with `NCCL_NET=IB` and no socket fallback.
- Completed the first end-to-end Exo proof in namespace `fwuffydwagon-nccl-poc-v1`: exactly two nodes advertised `MlxCuda`; placement chose `192.168.40.248` as the coordinator; both Tensor ranks loaded and warmed up; NCCL used merged device `NET/IB/2` across both HCA ports with no socket fallback; and a 44-token chat request returned successfully in 0.37 seconds.
- Repeated the end-to-end proof in namespace `fwuffydwagon-nccl-poc-v2` after removing `NCCL_GIN_ENABLE` and `NCCL_GIN_TYPE` from both parent environments. Bootstrap detected each `mlx4_core` HCA, both runner logs reported `NCCL_GIN_TYPE=0`, NCCL again selected merged `NET/IB/2`, and the same 44-token chat completed successfully in 0.47 seconds. This verifies that ConnectX-3 startup no longer depends on launch-script GIN flags.
- Root-caused the v2 deletion hang: the worker canceled `RunnerSupervisor` as soon as it received `TaskAcknowledged`, before `TaskStatus.Complete` and `RunnerShutdown` could be forwarded. The supervisor now buffers child `RunnerShutdown`, terminates and awaits the physical process, and only then forwards the terminal state. The worker retains supervisor/GPU ownership throughout graceful, three-second timeout, instance-deletion, and already-closed task-channel paths; an unconfirmed stop fails closed without releasing the resource.
- Serialized local runner creation with instance deletion. A stale precomputed `CreateRunner` cannot recreate a deleted instance, a failed `TaskCreated` publication cannot leave an untracked supervisor, and deletion emits positive shutdown acknowledgements only for assigned local runners that genuinely have no supervisor. Missing runner status still leases its assigned GPU until a confirmed terminal acknowledgement.
- Added immutable `SglangKtLaunchPlan` and canonical `SglangKtProcessLaunchSpec` contracts for exact model/runtime revisions and the target `30,28,20` layer, GPU, CPU/NUMA, KTransformers-method, HCA, distributed coordinator, and service bindings. The builder expresses dwagon as logical ranks 0 and 1 plus fwuff as rank 2 using PP=3, TP=1, and `nnodes=3`. The common `--dist-init-addr` is the actual torch/NCCL rendezvous; the ignored per-stage `--nccl-port` and its false port gate were removed.
- Added local SGLang preflight fact collection with injected runtime, filesystem, GPU/CPU/NUMA/HCA inventory, and endpoint effects. A structured default verifier parses the exact snapshot `config.json` and requires the GLM-5.2 architecture/topology, IndexShare frequency/pattern/skip inputs `4/null/3`, all 78 expanded `indexer_types`, FP8 e4m3 128x128 quantization, an exact revision receipt, and a SHA-256-bound complete snapshot. Pipeline starts must be full indexers in the observed config, not only a hard-coded model-name assumption; `39,39` is rejected while starts 30, 38, and 58 are legal for the audited snapshot.
- The pure group preflight now fails closed unless every rank also has a target-specific validation receipt tied to the exact model/config hash, SGLang/KTransformers and `transformers-kt` versions, torch/CUDA/kernel build IDs, GPU UUID and SM86 capability, KV dtype, CPU set, NUMA nodes, and executed CPU backend. Required capabilities cover stage-local TP broadcasts, GLM-5.2 NSA prefill/decode on SM86, physical-NUMA mapping, process affinity, and actual AMX execution. The default version/SHA probe never fabricates these receipts, so the known-broken audited base cannot be launched accidentally. Launch specs explicitly bound `--max-total-tokens` and `--mem-fraction-static` and require removal of inherited `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE`. These types remain outside the active `Instance` union until a patched runtime, validation runner, external-process lifecycle, and API proxy exist.
- Added validated NVIDIA compute-resource discovery and resource IDs, including GPU UUID, PCI address, model name, total VRAM, Linux sysfs NUMA node, and CPU affinity. NVML PCI domains are normalized to sysfs BDFs; missing or malformed locality degrades to unknown without losing GPU enumeration. `MlxNccl` can create one runner per selected GPU, binds each child by GPU UUID before MLX import, reserves resources already used by current or legacy instances, validates resource ownership, and contains per-rank task-start failures.
- Default NCCL placement remains one GPU per physical node. `use_all_compute_resources=true` explicitly selects every available GPU only when tensor dimensions are divisible and each rank satisfies `ceil(checkpoint size / world size) + max(1 GiB, 10% VRAM)`. This is a conservative weight/headroom gate, not a KV-cache or workspace model. Placement previews expose the policy and account host memory in proportion to ranks per node.
- Corrected GLM-4.7-Flash and Qwen3-Coder-family EOS handling. Unknown MLX families now fall back to checkpoint `generation_config.json` and then `config.json`, preserving multiple EOS IDs instead of silently assuming a single token.
- Added `EXO_MLX_VISION_LOADING=eager|lazy|disabled`. Use `disabled` for the text-only Qwen3.5 ladder run so every tensor rank avoids loading a replicated vision tower; `lazy` retains image capability but defers vision weights until the first image.
- Added exact Hugging Face revision support throughout model cards, config/index/file-list lookup, downloads, progress, MLX loading, and sibling vision/processor repositories. Exact pins use revision-suffixed directories and fail closed on absent/mixed receipts. Coordinator status, active cancellation scopes, throttles, worker backoff/readiness, placement scoring, `/models`, Ollama tags, and missing-instance notifications now use `(model_id, revision)` identity, so a stale `main` snapshot cannot suppress or authorize a pinned load. Model-wide Cancel/Delete remains backward compatible and affects every tracked revision. `migrate_hugging_face_local_dir_to_pinned_revision()` can atomically rename an existing `hf download --local-dir` snapshot only after every file's metadata proves the requested SHA.
- Pinned and metadata-audited every checkpoint in the nine-model NCCL ladder. All nine now have built-in cards with exact 40-hex Hugging Face revisions; the Qwen3.5 card also pins its same-repository vision weights. Added a tenth, 143 MB `SmolLM2-135M-Instruct-8bit` card whose hidden, attention, KV, and MLP dimensions are all divisible by three for the two-dwagon-GPU plus one-fwuff-GPU proof.
- Completed dashboard support for `MlxNccl`: persisted runtime selection, tensor-only launch validation, exact preview filtering, CUDA/NCCL automatic placement preference, advanced controls, instance labels, topology metadata, and prefill/decode wrappers. An NCCL-only `Use all available GPUs` control now persists and propagates through preview, onboarding, automatic, and fallback launch paths, including stale-response rejection and the selected minimum-node count. The production dashboard build passes. `npm run check` improved from 19 errors to 15; the remaining 15 errors and 6 warnings all predate this work.
- Aggregated generation task status per runner/rank with monotonic terminal handling. Duplicate acknowledgements, late completion, one-rank failure, and runner crashes can no longer make a distributed task appear complete or erase a terminal result incorrectly.
- Added `scripts/benchmark_lease.py`, which implements the exclusive lease and manifest heartbeat required by `/ai/coordinate.md`. It does not replace benchmark-specific preflight, telemetry, remote ownership checks, or cleanup.
- Focused tests for the integrated EOS, vision policy, compute-resource, shutdown, placement, bootstrap, and launch-plan slices pass. The corrected SGLang launch/snapshot/preflight slice has 94 passing tests; the final combined scheduler, supervisor, compute-resource lifecycle, and SGLang slice has 138 passing tests. Ruff lint and format checks pass, and strict targeted Basedpyright configurations report zero errors. The broad non-image baseline remains 459 passed and 5 skipped with the unchanged stale Rust binding test failure. A fresh full-repository attempt did not supersede that baseline: root collection stops because the external `exo_tools` module is absent, and `src/exo` collection stops in MLX/image tests because this source-only sandbox exposes no CUDA-capable device. `nix fmt` could not run because Nix is not installed on dwagon.
- A leased live shutdown regression was aborted at preflight because fwuff's only RTX 3090 had an unowned ComfyUI compute process (PID 5334, 256 MiB). Nothing was launched or killed; the abort manifest is `/var/lib/exo/benchmarks/exo-nccl-shutdown-preflight-20260718T0450/manifest.json`. Repeat the live deletion test only when that GPU is idle.
- Source implementation at this stopping point is committed through `cefea632` on the isolated `agent/linux-cuda-nccl` worktree; the plan update follows as a documentation-only commit. The local remotes are normalized as `upstream=exo-explore/exo` and reserved `origin=ldyeax/exo`; publication is pending creation of the GitHub fork because `ldyeax/exo` does not yet exist and `gh` is not installed on dwagon.
- Both Codex tasks must follow `/ai/coordinate.md`: one exclusive two-host benchmark lease, separate worktrees/deployments, unique namespaces/ports, preflight, ownership-safe cleanup, and per-run manifests. Since coordination was reaffirmed, this work used only source inspection and unit/static tests: it did not contact `fwuff`, load a model, touch `/mnt/sanic`, initialize CUDA/NCCL/InfiniBand, reserve service ports, or run a performance measurement.
- Remaining limitations before general NVIDIA support: GPU placement has only one-per-node or all-available policies rather than arbitrary subsets; KV/workspace/prefix-cache VRAM is not modeled; GPU enumeration still requires NVML even though locality now comes from sysfs; SGLang external-process lifecycle, API proxying, patched-runtime capability collection, and live SM86 GLM/NSA validation are not implemented; and live multi-GPU dashboard/hardware validation remains pending.

1. **Resource model and discovery**
   - Add a `ComputeResourceId` and represent each GPU as a schedulable resource rather than allowing only one runner per node.
   - Discover GPU UUID/index, VRAM, PCI address, NUMA node, CPU affinity, AMX features, local memory, NIC speed, and InfiniBand devices.
   - Replace `node_to_runner` state with resource-to-runner assignments and retain versioned backward reading of existing state.

2. **Engine and transport separation**
   - Separate `EngineKind` (`Mlx`, `SglangKt`) from `CollectiveTransport` (`Ring`, `Jaccl`, `Nccl`).
   - Add `MlxNcclInstance` using MLX's native NCCL backend and its rank/world/host environment contract.
   - Add a typed `SglangKtInstance` containing model revision, layer ranges, GPU/NUMA binding, CPU and memory affinity, KTransformers method, per-stage resident-expert budget, context/concurrency, selected HCA rails, health state, and rank-0 endpoint.
   - Proxy Exo's OpenAI-compatible stream, cancellation, and errors to the external engine's rank-0 endpoint.

3. **Placement and downloads**
   - Replace host-RAM-only proportional placement with GPU VRAM, host RAM, KV cache, workspace, expert residency, and configurable headroom accounting.
   - Add model-declared legal pipeline boundaries and reject invalid partitions.
   - Parse the Safetensors index and download only each stage's required layer shards plus shared configuration/tokenizer tensors.
   - Initially expose GLM-5.2 only through `SglangKt`, with exact minimum runtime versions and cache rules.

4. **KTransformers/SGLang pipeline fixes**
   - Broadcast from each pipeline stage's local TP root, not hard-coded global rank 0.
   - Flush deferred work at the stage-local final layer; keep deferral disabled until this is tested.
   - Map physical NUMA IDs to dense local slots, pass the real node ID to memory binding, consume explicit process CPU OS-ID sets, and make every binding failure fatal.
   - Add an Ampere-compatible GLM NSA indexer/attention path and prove both prefill and decode on SM86; do not force the current FP8 KV/FlashMLA build on RTX 3090.
   - Make the FP8 expert method execute and report a real AMX-tile kernel, or choose another validated method. The current `AMX` label is insufficient because its FP8 dispatch selects AVX-512.
   - Existing KTransformers POSIX shared-memory names already include a random UUID and TP rank; no PP-rank naming patch is required unless a deterministic collision is reproduced.
   - Make resident-expert budgets stage-specific.
   - Extend the executor to represent two logical nodes/resources on dwagon and one on fwuff, with explicit GPU, NUMA, service endpoints, and one shared rendezvous setting.

5. **Upstream work to evaluate or mine**
   - Exo PR 2103 for native Linux MLX-CUDA fixes.
   - PRs 2216, 2213, 2212, and 2010 for VRAM placement, accelerator representation, headroom, and profiling concepts.
   - PR 2214 for manual pipeline partitioning and PR 2195 for link bonding concepts.
   - Do not adopt PR 2129's hard-coded raw TCP/numpy relay as the production data plane.

## Hardware Preparation

1. Keep both healthy NVIDIA drivers and the verified MLX CUDA 13/NCCL 2.28.9 user-space stack; exact driver versions may differ if both satisfy the CUDA ABI.
2. A/B test moving dwagon's NUMA-1 RTX 3090 from x8 to a local x16 slot while preserving NVLink.
3. Place dwagon's HCA near the NUMA-1 inter-host pipeline stage; use an x8-or-better slot on fwuff.
4. Both QDR rails and two OpenSM instances are operational. Make the two OpenSM processes persistent across reboot, then repeat `ib_write_bw`, `ib_read_bw`, and the MLX NCCL diagnostic.
5. The current dual-rail result cannot meet the 1.7x target because `fwuff` is PCIe x4. Re-test the target after moving that HCA.
6. Install the selected `MCX555A-ECAT` pair in PCIe 3.0 x16-or-better slots with NUMA/root-complex placement chosen for the inter-host GPU boundary. Re-run the identical perftest, NCCL, and model benchmark manifests over the EDR QSFP28 link.
7. Free at least 650 GB on dwagon and retain at least 300 GB free on fwuff before full GLM-5.2 staging. Use NFS only for initial loading, not the inference hot path.
8. Do not add slower nodes until profiling proves their memory contribution exceeds their pipeline/network penalty.

### RDMA HCA upgrade research (2026-07-18)

1. **Do the free fix first:** move fwuff's current ConnectX-3 from PCIe 3.0 x4 to an x8-or-better slot. Its measured 29.80 Gb/s dual-rail aggregate matches the roughly 31.5 Gb/s x4 payload ceiling; x8 should permit roughly 55-60 Gb/s after tuning.
2. **Best-value matched pair:** ConnectX-5 VPI `MCX555A-ECAT`, one EDR InfiniBand/100GbE QSFP28 port on PCIe 3.0 x16. Prefer the dual-port `MCX556A-ECAT` only at similar cost because PCIe 3.0 x16 cannot sustain two 100 Gb/s ports simultaneously. A sensible used target is at or below roughly $150 per clean, full-height card. [Official ConnectX-5 VPI specifications](https://networking-docs.nvidia.com/connectx5vpihw/specifications)
3. **Lowest-cost useful modernization:** ConnectX-4 VPI `MCX455A-ECAT` or dual-port `MCX456A-ECAT`, EDR/100GbE on PCIe 3.0 x16. It is the first `mlx5` generation and the minimum generation supported by current NCCL GIN, but buy it only when materially cheaper than ConnectX-5. [Official ConnectX-4 VPI overview](https://networking-docs.nvidia.com/connectx4vpihw/introduction)
4. **Best performance/future option that fits these hosts:** ConnectX-6 VPI `MCX653105A-HDAT` (or ConnectX-6 DE `MCX683105AN-HDAT` when cheaper), one HDR InfiniBand/200GbE QSFP56 port on PCIe 4.0 x16. Target surplus pricing around $250-$350 per card; one 200 Gb/s port already consumes most of PCIe 4.0 x16, so a dual-port model adds little for this direct two-host link. [Official ConnectX-6 compatible-products table](https://networking-docs.nvidia.com/connectx6fwrn/20395124lts/firmware-compatible-products)
5. **Skip ConnectX-7 for the RTX 3090 cluster:** NDR 400 Gb/s requires PCIe 5.0 x16 plus costlier OSFP-era cabling and cooling, while a 3090 and these host-staged paths cannot use the premium. [Official ConnectX-7 specifications](https://networking-docs.nvidia.com/connectx7hw/specifications)
6. Keep native VPI InfiniBand for the simplest direct connection. Use a passive EDR-rated QSFP28 DAC for ConnectX-4/5 or an HDR-rated QSFP56 DAC for ConnectX-6; do not assume the current QDR QSFP+ optical path will train at EDR/HDR. RoCE-only ConnectX-6 Dx is a fallback only when unusually cheap, since it gives up native InfiniBand and adds Ethernet flow-control configuration.
7. Place each HCA under the same upstream PCIe root complex/NUMA side as the GPU handling the inter-host boundary, then verify negotiated speed and width with `lspci -vv`. [CUDA GPUDirect RDMA topology guidance](https://docs.nvidia.com/cuda/gpudirect-rdma/)
8. Do not justify an HCA purchase on GPUDirect or GIN alone. NVIDIA documents GPUDirect RDMA for Tesla/Quadro-class GPUs rather than GeForce; query CUDA device attribute 116 and treat `nvidia-peermem`/DMA-BUF on the RTX 3090 as an unsupported A/B experiment. Host-staged NCCL still benefits from EDR/HDR. [CUDA GPUDirect RDMA documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/) [NCCL GPU Direct troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
9. For more than two or three hosts, use an InfiniBand switch rather than a daisy chain: used SB7xxx/Switch-IB 2 for EDR or QM87xx for HDR. Continue running one OpenSM instance per direct-connect subnet until then.

## Proof-of-Concept Sequence

1. Completed MLX CUDA preflight and forced-IB NCCL collectives on port 1, port 2, and both ports. Large 64 MiB all-reduce measured 18.91, 19.18, and 19.13 Gb/s respectively.
2. Completed `mlx-community/Llama-3.2-1B-Instruct-4bit` with Tensor=2 and `MlxNccl`. The 730 MB model supports tensor sharding. Both ranks reached `RunnerReady`, both warmups completed, and a deterministic chat request returned a valid completion. Verified model locations:
   - `dwagon:/var/lib/exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit`
   - `fwuff:/mnt/sanic/exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit`
   - Exact revision on both hosts: `08231374eeacb049a0eade7922910865b8fce912`, verified from every Hugging Face local-download metadata record.
   Before the next pinned run, migrate each verified legacy directory to its revision-suffixed name while holding the `/ai/coordinate.md` lease and after confirming no process uses it:

   ```bash
   # dwagon
   .venv/bin/python -c 'from pathlib import Path; from exo.download.download_utils import migrate_hugging_face_local_dir_to_pinned_revision as migrate; from exo.shared.types.common import ModelId; print(migrate(Path("/var/lib/exo/models"), ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"), "08231374eeacb049a0eade7922910865b8fce912"))'

   # fwuff, from its isolated Exo deployment
   .venv/bin/python -c 'from pathlib import Path; from exo.download.download_utils import migrate_hugging_face_local_dir_to_pinned_revision as migrate; from exo.shared.types.common import ModelId; print(migrate(Path("/mnt/sanic/exo/models"), ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"), "08231374eeacb049a0eade7922910865b8fce912"))'
   ```

3. Run `mlx-community/SmolLM2-135M-Instruct-8bit@0f0d9b8218915bc34d401e1a340b8c049d300d5e` with Tensor=3 and `MlxNccl` across dwagon's two GPUs and fwuff's GPU before larger models. Its Llama hidden size 576, 9 attention heads, 3 KV heads, and MLP width 1536 are all divisible by three; the indexed weights are 142,955,136 bytes. Use this as a scheduler, multi-runner lifecycle, and collective-correctness proof rather than a performance result.
   The dashboard can now request this topology by selecting NCCL Tensor, two minimum nodes, and `Use all available GPUs`. Hold the `/ai/coordinate.md` lease for the model migration/download and the complete live run.
4. Keep `mlx-community/Qwen3-0.6B-8bit` for single-GPU and Ring/Pipeline regression only. Pipeline over NCCL requires adding `ncclSend`/`ncclRecv` support to MLX first.
5. Re-run the known Ornith-1.0-35B FP8/MXFP4 workload as an AMX/OSCAR regression baseline.
6. Validate the hybrid AMX offload path with DeepSeek-V4-Flash (284B total/13B active, mixed FP4/FP8, 1M context support).
7. Commit a KTransformers/SGLang runtime fork that fixes stage-local broadcasts, SM86 NSA, physical NUMA mapping, process CPU affinity, and the executed AMX backend. Update Exo's accepted runtime SHAs only after the PP=3 initialization and per-stage capability receipts pass.
8. Build a tiny deterministic GLM IndexShare fixture and require single-rank versus pipeline output parity. Keep deferred work at zero until stage-local final-layer flushing has its own parity test.
9. Start full GLM-5.2 FP8 weights at 4K with a proven Ampere-compatible KV backend, then test 32K, 128K, and finally 245,760 input tokens plus a 16,384-token output reserve. Do not select FP8 KV merely to meet the cache estimate.
10. Tune resident experts at 0/1/2/4, then enable MTP and deferred work independently with correctness and performance A/B tests.

### Pre-ConnectX-5 MLX/NCCL benchmark ladder

Run Tensor=2, concurrency 1, prefix cache off, and require at least 1 GiB free VRAM per rank. Use 256/4K/8K contexts for the last two models; test 8K and then 32K on smaller models where headroom permits. Each checkpoint must be present on both nodes.

| Order | Model and pinned revision | Approximate checkpoint per node | Purpose |
| --- | --- | ---: | --- |
| 1 | `mlx-community/Llama-3.2-1B-Instruct-4bit@08231374eeacb049a0eade7922910865b8fce912` | 0.7 GB | transport and lifecycle control; live TP=2 pass |
| 2 | `mlx-community/Llama-3.2-3B-Instruct-4bit@7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` | 1.8 GB | small dense scaling |
| 3 | `mlx-community/Llama-3.1-8B-Instruct-4bit@90215b22ec18e72f623dde2ea7af4097025160e2` | 4.5 GB | medium dense scaling |
| 4 | `mlx-community/gpt-oss-20b-MXFP4-Q8@773a7da77e569019bb0fd17a554b263738d669a3` | 12.1 GB | MXFP4 CUDA and GPT-OSS sharder/parser |
| 5 | `mlx-community/GLM-4.7-Flash-4bit@1454cffb1a21737e162f508e5bc70be9def89276` | 16.9 GB | closest current MLX GLM/MoE probe |
| 6 | `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit@6e302ea604ad9ab206367e2c501d1571023e7b6d` | 17.2 GB | coding/MoE baseline |
| 7 | `mlx-community/Qwen3.5-35B-A3B-4bit@1e20fd8d42056f870933bf98ca6211024744f7ec` | 20.4 GB | hybrid-attention/MoE sharder probe |
| 8 | `mlx-community/Llama-3.3-70B-Instruct-4bit@de2dfaf56839b7d0e834157d2401dee02726874d` | 39.7 GB | safer dense capacity ceiling; start at 8K or less |
| 9 | `mlx-community/Qwen3-Coder-Next-4bit@7b9321eabb85ce79625cac3f61ea691e4ea984b5` | 44.8 GB | practical two-rank capacity and coding-intelligence ceiling |

The complete ladder is about 158.2 GB per node. All nine exact revisions have built-in Exo cards and passed metadata/config audits, but only rung 1 has a live two-host result; rungs 2-9 currently have architecture/import and divisibility review only. The old TP bit-exact test is skipped, Darwin/ring-specific, and calls a stale sharder signature; replace it with enabled TP=1-versus-TP=2 quantized-logit/token tests before treating a successful load as correctness.

Default placement uses one GPU per node and leaves dwagon's second GPU idle; the explicit all-resource policy is now implemented but does not make a three-rank tensor run valid automatically. Common hidden sizes and KV-head counts in this ladder are not divisible by three. Rung 9 cannot use TP=4 with the current sharder because it has two KV heads; support for replicating KV heads when world size exceeds KV-head count is separate work. Prefix cache is disabled in the benchmark requests because it is created per runner, can duplicate entries, and currently evicts according to host RAM rather than device VRAM.

## AMX Verification Model

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Pinned revision: `60d8d70770c6776ff598c94bb586a859a38244f1`
- Verified download location: `fwuff:/mnt/sanic/llm_models/DeepSeek-V4-Flash`
- Verified checkpoint: 46 Safetensors shards, `model.safetensors.index.json`, configuration/tokenizer files, and 159,609,485,896 indexed bytes (149 GiB on disk).
- Verification result: zero files referenced by the index are missing, and Hugging Face download metadata records the pinned revision above.

## Acceptance Criteria

- One-hour concurrency-1 soak without deadlock or OOM, with at least 1 GiB free VRAM per rank and 10% host-memory headroom.
- Correct reasoning and tool-call parsing plus long-context retrieval checks.
- At least 3 output tokens/s at an 8K warm context for the initial GLM-5.2 target.
- The runtime reports the loaded kernel build and measured executed backend; AMX conversion/offload changes execute AMX tiles and remain within one quality point on the selected coding evaluation.
- InfiniBand meets the dual-rail bandwidth and error criteria above; GPUDirect is enabled only if it wins a stable A/B test.
- Exo changes pass `uv run basedpyright`, `uv run ruff check`, `nix fmt`, and `uv run pytest`, plus Linux CUDA and two-host integration tests.

## Alternative Coding Models

1. Kimi K2.7 Code: first intelligence-oriented fallback; native INT4, 256K context, and coding focus, but larger than DeepSeek-V4-Flash.
2. DeepSeek-V4-Flash: best initial AMX/hardware-fit candidate and validation model.
3. Ornith/Qwen3.5 397B: known local baseline with measured performance around 14.4 output tokens/s.
4. DeepSeek-V4-Pro: stretch target whose roughly 850 GB checkpoint leaves too little operational headroom initially.
5. Qwen3-Coder-Next: smaller, faster lower-tier coding candidate.

## Defaults and Assumptions

- Optimize first for one interactive coding-agent request, not throughput serving.
- Use only currently owned hardware for the first implementation cycle.
- Use PP=3/TP=1 and FP8 KV as the preferred GLM-5.2 topology.
- Keep 10 GbE available for control and fallback; use NCCL over InfiniBand for the production collective path.
- Preserve exact model revisions and benchmark inputs so results can be reproduced.
- Keep deployments, caches, service configuration, and test environments root-owned under `/root`, `/etc/exo`, `/var/lib/exo`, `/var/cache/exo`, or `/mnt/sanic`; do not depend on user home directories.
- For ConnectX-3 runs, require `NCCL_NET=IB`, `NCCL_GIN_ENABLE=0`, `NCCL_GIN_TYPE=0`, and `NCCL_NET_GDR_LEVEL=LOC` until separate GPUDirect validation succeeds.
