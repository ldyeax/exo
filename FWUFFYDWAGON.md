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
- Use the kernel `mlx4_core`/`mlx4_ib` drivers, `rdma-core`, `perftest`, Mellanox Firmware Tools, and one OpenSM instance per disconnected direct-connect rail. Current MLNX_OFED releases no longer support ConnectX-3.
- Keep 10 GbE as the management/control plane. QDR ports do not aggregate automatically; select and benchmark both rails explicitly.
- Default to host-staged NCCL/InfiniBand. RTX 3090 GPUDirect RDMA is not an officially supported configuration; test `nvidia-peermem` only as an optional A/B path.
- NCCL 2.28 GIN requires ConnectX-4 or newer. On these ConnectX-3 cards, NCCL crashed in `ncclNetInit` while initializing its GIN plugin; `NCCL_GIN_ENABLE=0` alone was insufficient. Set both `NCCL_GIN_ENABLE=0` and `NCCL_GIN_TYPE=0`. Ordinary MLX NCCL collectives do not require GIN, and GIN is distinct from GPUDirect RDMA.
- Verified `ib_write_bw`: 28.46 and 28.56 Gb/s single-rail, 29.80 Gb/s concurrent aggregate. The aggregate ceiling is consistent with `fwuff`'s PCIe x4 HCA placement.

## Model and Memory Findings

- GLM-5.2 has about 750B total and 40B active parameters, 78 layers, 256 routed experts with top-8 routing plus a shared expert, MLA, DSA, IndexShare, and one MTP layer.
- BF16 weights are roughly 1.51 TB and cannot fit in the combined one-TiB host memory. The official FP8 checkpoint is roughly 704 GiB and does fit, but leaves limited headroom.
- Current Exo/MLX does not implement GLM-5.2's cross-layer IndexShare behavior correctly. Do not advertise GLM-5.2 on the MLX engine until parity with the changes in MLX-LM PR 1410 is demonstrated.
- The production path should let Exo orchestrate an external SGLang + KTransformers engine. This preserves AMX CPU expert execution and avoids forcing GLM-5.2 through an unsuitable MLX-CUDA model implementation.
- Gate full-model work on an RTX 3090/SM86 import and short-forward test for the GLM-5.2 sparse-attention and IndexShare kernels.

## KV Cache Conclusion

The claim that every GPU or memory node always needs a complete KV cache is false for pipeline parallelism. Each pipeline stage stores cache only for its local layers. Tensor-parallel ranks may replicate some latent/index state within a stage, so the initial GLM topology uses TP=1.

For GLM-5.2, estimated per-token, per-layer cache is about 1,284 bytes in BF16 or 788 bytes in FP8. At 256K tokens, the preferred three-stage split requires approximately 5.8, 5.4, and 3.9 GiB of cache before allocator padding and workspace. Use FP8 KV, concurrency 1, bounded prefix caching, and conservative resident-GPU expert counts. Consider OSCAR INT2 or host spill only after GLM-specific correctness validation. Earlier Ornith runs left 4.8-10 GB free per RTX 3090, so 2 GB free is not an unavoidable limit.

## Preferred Runtime Topology

Run three logical pipeline resources with TP=1:

| Rank | Host/resource | Layers | CPU expert weight estimate |
| --- | --- | --- | --- |
| 0 | dwagon, NUMA 0, RTX 3090 at `16:00.0` | 0-29 (30) | about 243 GiB |
| 1 | dwagon, RTX 3090 at `27:00.0`; target NUMA 1 after hardware reshuffle | 30-57 (28) | about 252 GiB |
| 2 | fwuff, NUMA 0, RTX 3090 at `6a:00.0` | 58-77 (20) | about 180 GiB |

Set `SGLANG_PP_LAYER_PARTITION=30,28,20`. Starts 0, 30, and 58 are valid full-indexer boundaries. This ordering keeps rank 0 to rank 1 traffic local (and eligible for NVLink) and crosses InfiniBand only between rank 1 and rank 2. Start with zero resident GPU experts, then profile 1, 2, and 4 experts per stage. Disable MTP and deferred expert work until correctness passes.

The current hardware enumeration places both dwagon GPUs on NUMA 0. Do not claim the preferred production topology is NUMA-local until a slot reshuffle or measured cross-socket fallback validates it.

If three logical resources cannot be launched reliably, use PP=2/TP=1 with `SGLANG_PP_LAYER_PARTITION=38,40`. That fallback requires swapping one eight-DIMM set so dwagon has 640 GB and fwuff has 384 GB; one dwagon GPU remains idle. Do not use equal `39,39`, which begins the second stage at an unsafe shared-indexer boundary.

## Exo Implementation Plan

### Phase 1 implementation status

- Added an additive `MlxNcclInstance` restricted to multi-rank, full-layer Tensor shards on `MlxCuda` nodes.
- Added one shared, real IPv4 rank-0 coordinator address and the MLX 0.32 NCCL environment contract (`MLX_RANK`, `MLX_WORLD_SIZE`, `NCCL_HOST_IP`, `NCCL_PORT`).
- Preserved root-owned `CUDA_VISIBLE_DEVICES` and NCCL transport policy in runner children. Runner bootstrap now detects `mlx4_core` devices and defaults both GIN controls to `0`, while preserving explicit operator overrides.
- Made MLX control collectives NCCL-safe: GPU/default stream instead of a forced CPU stream, and `int32` rather than unsupported boolean cancellation reductions.
- Kept NCCL out of placement previews/dashboard for now; the proof uses the direct API. Current MLX NCCL does not implement `send`/`recv`, so Pipeline+NCCL is rejected by placement and instance validation.
- Added `scripts/mlx_nccl_smoke.py`, also deployed as `fwuff:/root/mlx-nccl-smoke/nccl_smoke.py`. Port 1, port 2, and merged dual-port runs all passed `all_sum`/`all_gather` correctness with `NCCL_NET=IB` and no socket fallback.
- Completed the first end-to-end Exo proof in namespace `fwuffydwagon-nccl-poc-v1`: exactly two nodes advertised `MlxCuda`; placement chose `192.168.40.248` as the coordinator; both Tensor ranks loaded and warmed up; NCCL used merged device `NET/IB/2` across both HCA ports with no socket fallback; and a 44-token chat request returned successfully in 0.37 seconds.
- Repeated the end-to-end proof in namespace `fwuffydwagon-nccl-poc-v2` after removing `NCCL_GIN_ENABLE` and `NCCL_GIN_TYPE` from both parent environments. Bootstrap detected each `mlx4_core` HCA, both runner logs reported `NCCL_GIN_TYPE=0`, NCCL again selected merged `NET/IB/2`, and the same 44-token chat completed successfully in 0.47 seconds. This verifies that ConnectX-3 startup no longer depends on launch-script GIN flags.
- Deleting the v2 instance removed it from state, but both distributed runners remained in `RunnerShuttingDown` with their shutdown tasks running until the two Exo nodes were stopped. No Exo process was left running. Treat bounded NCCL runner shutdown and peer-loss recovery as required lifecycle work before a soak test.
- Focused placement, serialization, bootstrap, and NCCL initialization tests pass (67 tests). The broad non-image suite passes 459 tests with 5 skipped; its only failure is the unchanged `rust/exo_rs/tests/test_python.py`, whose stale three-argument `NetworkingHandle.new` call no longer matches the four-argument binding. Ruff lint and format checks pass. Basedpyright reaches only the unchanged unnecessary-ignore error at `src/exo/utils/info_gatherer/info_gatherer.py:359`. `nix fmt` could not run because Nix is not installed on dwagon.
- Remaining limitations before general NVIDIA support: exactly one runner per node, host-RAM rather than VRAM-aware placement, NVML-only CUDA capability advertising, and no dashboard control.

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
   - Include logical resource/PP rank in shared-memory names when two ranks run on dwagon.
   - Make resident-expert budgets stage-specific.
   - Extend the launcher to represent two logical nodes/resources on dwagon and one on fwuff, with explicit GPU, NUMA, ports, and rendezvous settings.

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
6. Free at least 650 GB on dwagon and retain at least 300 GB free on fwuff before full GLM-5.2 staging. Use NFS only for initial loading, not the inference hot path.
7. Do not add slower nodes until profiling proves their memory contribution exceeds their pipeline/network penalty.

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
3. Keep `mlx-community/Qwen3-0.6B-8bit` for single-GPU and Ring/Pipeline regression only. Pipeline over NCCL requires adding `ncclSend`/`ncclRecv` support to MLX first.
4. Re-run the known Ornith-1.0-35B FP8/MXFP4 workload as an AMX/OSCAR regression baseline.
5. Validate the hybrid AMX offload path with DeepSeek-V4-Flash (284B total/13B active, mixed FP4/FP8, 1M context support).
6. Build a tiny deterministic GLM IndexShare fixture and require single-rank versus pipeline output parity.
7. Start full GLM-5.2 FP8 at 4K, then test 32K, 128K, and finally 245,760 input tokens plus a 16,384-token output reserve.
8. Tune resident experts at 0/1/2/4, then enable MTP and deferred work independently with correctness and performance A/B tests.

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
- AMX conversion/offload changes remain within one quality point on the selected coding evaluation.
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
