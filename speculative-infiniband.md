# GLM-5.2 Speculation over InfiniBand

Status: implementation decision note for the local GLM-5.2 TP2 + KTransformers
MTP lane, 2026-07-25.

Evidence labels used below:

- **Local**: derived from the pinned local checkpoint, source, or benchmark
  receipts named in this document.
- **Source**: stated by an upstream project, hardware vendor, or paper.
- **Hypothesis**: a design expectation that still needs measurement here.

## Recommendation

Use `fwuff` as the first remote TP1 native-MTP draft service:

- Keep `dwagon`'s two RTX 3090s and NVLink as the TP2 target/verifier.
- On `fwuff`, run layer 78 only: attention, dense matrices, embedding, and head
  on its dedicated RTX 3090; routed experts through its 60-core/120-thread
  Intel AMX host; and the one-layer draft KV locally resident.
- Send accepted token IDs and BF16 target hidden rows to `fwuff`; return
  proposed token blocks over the installed ConnectX-5 EDR/100G link using
  permanently registered pinned-host buffers.

This path reuses the already-proven SM86 Marlin + AMXINT4 execution split. It
does not require a new 5090 kernel port or expert-format conversion before the
first large test.

Build it only after two cheap gates:

1. Finish the matched local MTP quality gate and benchmark local top-k-1 chains
   at 1, 3, 5, and 7 draft steps.
2. Run the same layer-78 compute path in isolation on `fwuff`. Stop if its
   measured compute/resource-isolation benefit is not larger than the measured
   EDR staging, synchronization, and round-trip cost.

The first networked design should be synchronous and deliberately narrow:

- A designated TP2 target rank sends BF16 target hidden rows to one TP1 draft
  service.
- The draft service retains its own one-layer KV state and returns candidate
  token IDs. For a top-k-1 chain, the target can reconstruct the static tree.
- Candidate metadata is broadcast over the existing target TP group. Both
  target ranks verify locally; all sampling remains on the target.
- Use permanently registered pinned-host ring buffers. GPUDirect is unproven
  under the observed RTX 3090 runtime capabilities and is not a dependency of
  the first design.
- Use Exo/Zenoh for discovery, admission, and health only. Do not put JSON,
  RPC allocation, or model-state transfer on the per-cycle critical path.

Use only the EDR path for latency-critical draft traffic. Reserve the legacy
QDR path for bulk SmallEP, prefill, or artifact movement; never stripe a tiny
request across EDR and QDR.

If the synchronous path wins, add adaptive two-to-four-token lookahead with
versioned, double-buffered tentative tails and rollback. PipeInfer is a useful
design reference for that later stage, not the first implementation.

The user-supplied Ryzen 9 9950X3D + RTX 5090 WSL2 machine is the second
comparison: first as an isolated pure-GPU draft-compute experiment and only
later as a network endpoint if its host transport is independently admitted.

### Non-goals

Do not:

- move the target's 78-layer KV cache, logits, or probability vectors to the
  draft host;
- move target verification or final sampling off the TP2 target;
- load the complete GLM-5.2 target on `fwuff`;
- split the latency-critical MTP layer across the 2080 Ti or 1080 Ti nodes;
- add a serial remote call for the current one-step/K=1 configuration;
- run n-gram speculation remotely;
- stripe latency-critical descriptors across EDR and QDR;
- make GPUDirect a prerequisite or first-rung optimization on the 3090 path;
- change SGLang's speculative-engine version while proving the transport;
- assume SGLang prefill/decode disaggregation is remote MTP support; or
- build multiple remote drafters before `fwuff` beats the best local
  control.

### Current topology and chronology

**Local.** This note resolves an older-document ambiguity:

- [`FWUFFYDWAGON.md`](FWUFFYDWAGON.md) correctly describes `fwuff` as a
  single-socket 60C/120T Intel AMX host with 256 GB DDR5 and one RTX 3090.
  Its line that calls ConnectX-5 a “planned upgrade” predates installation and
  is stale.
- [`OSDI26.md`](OSDI26.md) records the newer topology: `dwagon` retains two
  NVLink-connected RTX 3090s, and both ConnectX-3 and ConnectX-5 are present;
  `fwuff`'s GPU remains on its sole NUMA node.
- [`parallelism.md`](parallelism.md) records the target-local NV4 link, the
  admitted host-staged network path, and the observed zero CUDA capability
  attributes on `fwuff`. Those attributes make GPUDirect unproven in the
  installed runtime; the dedicated audit below avoids treating them as an
  immutable hardware-support verdict.
- The current link-state receipt records `mlx5_0` active at 100 Gb/s and
  `mlx4_0` active at 40 Gb/s. No local receipt yet measures application
  throughput or small-message latency on the new EDR link.

## Exact local payload contract

### Model dimensions

**Local.** These values come from
`/mnt/sanic/glm52-AMXINT4-W8A16-hybrid/config.json` (SHA-256
`c0663ef196206e5de0b4a6e216a54d8c10d8853ef455e3578e81ab6ffcd9182a`).
The hybrid manifest binds the source config as
`817f5fb39ca5d4c4b5648de89ca00deaea7537d8c2f130172a459252a05c1073`.

| Quantity                        |                                              Value |
| ------------------------------- | -------------------------------------------------: |
| Target causal layers            |                                                 78 |
| NextN/MTP layers                |                                                  1 |
| Hidden width                    |                                              6,144 |
| Vocabulary                      |                                            154,880 |
| Activation and KV dtype         |                                               BF16 |
| MLA latent width                | `kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` |
| Routed experts / active experts |                                            256 / 8 |
| MTP shares index work           |               `index_share_for_mtp_iteration=true` |

The official GLM-5.2 description confirms that MTP uses KVShare and IndexShare
and is designed for multi-step prediction. That is a **source** architecture
claim; it is not a performance result for this runtime.

### Draft artifact

**Local.** A remote process must materialize modules that the current local
draft borrows from its target process. Exact serialized tensor bytes are:

| Component                       |                         Bytes | Provenance                                                  |
| ------------------------------- | ----------------------------: | ----------------------------------------------------------- |
| BF16 embedding                  |                 1,903,165,440 | `model.embed_tokens.weight` in the hybrid manifest          |
| W8 `lm_head` qweight + scale    |                   951,892,480 | `lm_head.*` in the hybrid manifest                          |
| Layer-78 non-expert body        |                   366,722,944 | sum of `model.layers.78.*` in the hybrid manifest           |
| Layer-78 AMXINT4 routed experts |                 4,848,615,424 | safetensor metadata for all 3,072 `blk.78.*.numa.*` tensors |
| **Total**                       | **8,070,396,288 (7.516 GiB)** | sum above                                                   |

The expert bytes are two required partitions of 2,424,307,712 bytes, not two
replicas. They are in
`/mnt/sanic/glm52-AMXINT4/model-00076-of-00077.safetensors`; the index is
`/mnt/sanic/glm52-AMXINT4/model.safetensors.index.json`. The hybrid manifest is
`/mnt/sanic/glm52-AMXINT4-W8A16-hybrid/hybrid-checkpoint-manifest.json`
(SHA-256
`692cc7a5ae4b2d4c5136ef8ddbd602966904eccb7e540ff2f4b641459069d14a`,
content ID
`2d176d81a2808d097519890b51ed8a4a2df66dbc0563b6771deaf3aa59700025`).

For the primary `fwuff` design, this splits cleanly into:

- **3,221,780,864 B (3.001 GiB) on its RTX 3090** for embedding, W8 head, and
  layer-78 non-expert work; and
- **4,848,615,424 B (4.516 GiB) in its Intel host path** for the existing
  AMXINT4 experts.

With top-8 routing and equal serialized expert sizes, one MTP step selects
about `4,848,615,424 * 8 / 256 = 151,519,232 B` of AMX expert payload. The
head alone is another 951,892,480 B of GPU weights per full scan. These are
traffic lower bounds; only isolated timers decide whether `fwuff` closes the
synchronous break-even inequality.

The unquantized BF16 expert body is
`256 * 3 * 6,144 * 2,048 * 2 = 19,327,352,832 B = 18 GiB`; the local OSDI
note independently records it as roughly 18 GiB. `fwuff` must map/touch only
layer 78, not the complete model's expert set. Its 256 GB DRAM is therefore
sufficient for this service even though it cannot host a full-model replica.

The AMXINT4 layout is not a CUDA artifact. Only the optional pure-GPU 5090
comparison needs a lossless repack or explicitly bound
dequantize/requantize step into a supported GPU W4 layout. That conversion
needs a new content ID, and all W8/W4 extensions must execute on SM120 before
the 5090 result is admitted.

### Per-token and prefix bytes

Let `H=6144`, `R=576`, `L` be prompt tokens, `A` accepted draft tokens, and
`E=A+1` emitted target tokens in a verification cycle.

**Local derivation:**

- One BF16 target feature row: `2H = 12,288 B`.
- One layer of latent MLA KV: `2R = 1,152 B/token`.
- All 78 target layers: `78 * 1,152 = 89,856 B/token/rank`.
- Current TP2 aggregate target latent KV:
  `2 * 89,856 = 179,712 B/token`.
- One BF16 vocabulary vector: `2 * 154,880 = 309,760 B`.
- One FP32 vocabulary vector: `4 * 154,880 = 619,520 B`.
- Initial MTP prefill feature transfer: `12,288L B`.
- Steady-state target-to-draft feature transfer: `12,288E B/cycle`.
- A static top-k-1 proposal chain needs only `4K B` of int32 candidate IDs,
  plus a fixed descriptor. A generic dynamic tree also needs small parent and
  position arrays, but not draft vocabulary probabilities.

The installed SGLang verifier selects target hidden states at every accepted
index, including the target bonus/fallback token. Therefore the steady-state
feature cost is one 12 KiB row per emitted token, not one row per verification
cycle. The remote service keeps only its 1,152 B/token one-layer KV; it must
never receive the target's 78-layer KV.

| Prompt length | Feature prefill | TP2 target latent KV, for comparison |
| ------------: | --------------: | -----------------------------------: |
|           512 |           6 MiB |                            87.75 MiB |
|         1,024 |          12 MiB |                            175.5 MiB |
|         8,192 |          96 MiB |                            1,404 MiB |

The target-KV values exclude allocator padding, page metadata, and index state.
They are a payload comparison, not a memory-allocation receipt.

### Ideal wire serialization

**Local derivation.** These are one-way payload-only times (`8 * bytes / bit
rate`), before PCIe copies, protocol overhead, doorbells, CUDA synchronization,
or an application round trip.

| Payload                               | Measured 31.74 Gb/s QDR rail | 100 Gb/s | 200 Gb/s |
| ------------------------------------- | ---------------------------: | -------: | -------: |
| One 12 KiB feature row                |                      3.10 us | 0.983 us | 0.492 us |
| One TP2 target-KV token, 179,712 B    |                     45.30 us | 14.38 us |  7.19 us |
| 512-token feature prefill, 6 MiB      |                     1.586 ms | 0.503 ms | 0.252 ms |
| 512-token TP2 target KV, 92,012,544 B |                     23.19 ms |  7.36 ms |  3.68 ms |
| One FP32 vocabulary vector            |                     156.1 us | 49.56 us | 24.78 us |

The 31.74 Gb/s value is the reportable single-rail result in
`/var/lib/exo/benchmarks/ib-qdr-x8x8-independent-20260718-v4`. **Local:** the
newer link-state receipt
`/var/lib/exo/peer-artifact-deployment/ipoib-dwagon-20260725-v1.json` records
`mlx5_0` active at 100 Gb/s and the legacy `mlx4_0` ports active at 40 Gb/s.
That proves negotiated link rate, not 100 Gb/s application payload throughput;
the EDR ring still needs its own latency and throughput receipt.

The older “planned ConnectX-5” sentence in `FWUFFYDWAGON.md` is stale. The
current OSDI topology and link-state receipt are authoritative: ConnectX-5 is
installed and active between `dwagon` and `fwuff`; the ConnectX-3 QDR path
remains available separately.

The table shows why the hidden-row protocol is bandwidth-light and why full KV
or probability transport is the wrong design. **Hypothesis:** small-message
software latency and GPU/host synchronization will dominate serialization.
Keep the tiny synchronous ring on EDR alone. Use QDR separately for future
bulk work so striping and slow-rail tail latency cannot delay a proposal.

## Losslessness invariants

The target-only verifier is the correctness boundary:

1. Keep `speculative_accept_threshold_single=1.0` and
   `speculative_accept_threshold_acc=1.0`. Lower values deliberately relax
   acceptance and are outside this design.
2. Keep temperature, penalties, logit bias, grammar masks, top-k/top-p
   renormalization, target probabilities, RNG, rejection, and final sampling
   on the target.
3. The remote draft is deterministic for a committed request state and returns
   candidate IDs/tree metadata only. It never returns an authoritative token.
4. Greedy verification accepts only target-argmax matches. Sampling uses the
   pinned target-only kernel: rejected candidate mass is removed from target
   mass and the fallback is sampled from the nonnegative residual.
5. Never substitute classical `p/q` rejection sampling without transporting
   or otherwise reproducing the complete draft distribution `q`. That would
   turn each proposal into as much as 619,520 B of FP32 probabilities.
6. A timeout, restart, malformed descriptor, or epoch mismatch falls back to
   local/vanilla target decoding before target state is committed.

**Local.** These behaviors are visible in the pinned
`TreeSpeculativeSamplingTargetOnly` kernel and
`EagleVerifyInput.verify`: target probabilities are constructed after target
logit processing, `draft_probs` begins at zero, rejected candidate target mass
is recorded, and final sampling uses the residual. Accepted hidden rows are
then selected for the next draft extend. The pinned SGLang source bundle ends
at commit `1218b2f8965b5a27c9d9004ff9324373414be322`.

**Source.** Exact speculative decoding preserves the target distribution; the
original paper supplies the general rejection-sampling result. EAGLE and
EAGLE-2/EAGLE-3 supply feature-level and tree-drafting designs.

Distribution preservation does not promise the same sampled token sequence as
vanilla decoding because RNG consumption and execution order may differ.
Require fixed greedy/logit parity plus the matched quality gate; do not demand
sampled hash identity as the only correctness criterion.

## Synchronous `fwuff` draft design

```text
dwagon TP2 rank 0 (CX-5-local placement)         fwuff TP1 hybrid draft service
  accepted IDs + target hidden rows    ------->  RTX 3090 dense/head + AMX experts
  candidate chain/tree (IDs only)      <-------  proposed token block
           |
           +-- NVLink target-TP broadcast --> dwagon rank 1
                   both ranks verify; target samples and commits

                                                  draft KV remains on fwuff
```

### Target side

- Select one coordinator rank. Only it owns the remote queue pair and request
  epochs.
- Prove that the post-target hidden row is full-width and identical on both
  target ranks. If it is not, reconstruct it with the existing target TP
  collective before transport.
- Broadcast the proposal descriptor and candidate chain through the existing
  target TP group before verification.
- Run the existing target verification and sampling path unchanged.
- Send the `A+1` verified IDs and selected hidden rows in one `ADVANCE`
  message. Do not advance remote committed state speculatively.

### Draft side

- Run only layer 78 at TP1 on `fwuff`: embedding, attention/dense work, and
  head on the RTX 3090; routed experts in the existing Intel AMXINT4 path.
- Keep the one-layer 1,152 B/token KV pool on `fwuff`; at 8K it is only 9 MiB
  before allocator padding.
- On `OPEN`, consume prompt IDs and all target prompt hidden rows, build the
  one-layer draft KV, and return the first proposal.
- On `ADVANCE`, atomically commit the accepted chain, discard the rejected
  tentative branch, extend from the supplied target features, and return the
  next proposal.
- Keep one committed state and at most one tentative generation per request
  in the synchronous version.
- Bind every request to the target model/config, tokenizer, draft content ID,
  dtype, K, top-k, and speculative-engine contract.

For `fwuff`, “draft content ID” means the existing hybrid manifest plus exact
layer-78 AMX artifact and runtime identities. For a later 5090 conversion, it
means the new converted artifact ID.

### SGLang seam

**Local.** The pinned scheduler constructs an `EAGLEWorker` beside every target
worker with the same `gpu_id` and `tp_rank`. The useful boundaries are:

- `Scheduler` draft-worker construction;
- `EAGLEWorker.draft`;
- target-local `EAGLEWorker.verify`; and
- `EAGLEWorker.forward_draft_extend_after_decode`.

Current local admission is intentionally hardcoded to TP2, one step, top-k 1,
and two draft tokens. Implement a feature-gated
`RemoteEAGLEWorker`/coordinator at the scheduler seam and give its draft service
an independent `draft_tp_size=1`. Keep `verify` local. Replace draft forward
and post-verify draft extend with `PROPOSE` and `ADVANCE`; do not proxy the
whole worker API over a generic RPC. Widen depth only under an exact launch
receipt. The local ownership and MTP patches are recorded under
`scripts/patches/osdi26/`.

Upstream SGLang's NIXL and Mooncake prefill/decode code can donate connection
bootstrap, buffer registration, and lifecycle patterns. Its semantics transfer
KV between prefill and decode workers; they do not implement remote feature
drafting. Keep those concepts separate. **Local:** none of NIXL, Mooncake,
`ucp`, or `ucxx` is importable in the pinned runtime, so the first prototype
needs an explicitly pinned transport dependency or a narrow verbs
implementation; there is no drop-in backend today.

## RDMA ring and epoch protocol

Use one EDR reliable-connected queue pair and two fixed slots per admitted
request. Allocate and register all descriptor, ID, hidden-row, and result
buffers once at service start. Do not stripe a descriptor or cycle across the
EDR and QDR links.

Each descriptor carries:

```text
protocol_version
target_boot_uuid
draft_boot_uuid
request_id
request_incarnation
cycle_sequence
committed_position
model_content_id
sampling_contract_hash
message_kind
payload_offset
payload_bytes
row_count
candidate_count
```

Message kinds are `OPEN`, `PROPOSAL`, `ADVANCE`, `CANCEL`, `ACK`, and `ERROR`.
The ordering rule is payload write, memory fence, descriptor write, then
write-with-immediate/doorbell. A receiver validates identity, epoch, sequence,
position, bounds, and state before making the payload visible to a CUDA stream.

State rules:

- `(boot UUID, request ID, incarnation)` prevents slot-reuse ABA.
- `cycle_sequence` is monotonic. Duplicate `ADVANCE` is idempotent; an older
  sequence is discarded; a gap is an error.
- `committed_position` must match both peers before draft KV is mutated.
- The draft retains tentative state until the matching `ADVANCE`.
- Slot reuse waits for `ACK`; `CANCEL` creates a new incarnation.
- On timeout, the target abandons the remote request and continues
  local/vanilla. Late completions cannot mutate target or newly reused state.

Pinned-host staging is the admitted path: GPU-to-pinned-host copy, registered
EDR write, pinned-host-to-GPU copy. Registration in the decode loop is
forbidden. The existing launch posture uses `NCCL_NET_GDR_LEVEL=LOC`, and
`fwuff`'s RTX 3090 reports both GPUDirect-RDMA and DMA-BUF capability
attributes as zero. Those are observations from the installed CUDA
runtime—not a claim that replacing ConnectX-3 with ConnectX-5 can never work.
Do not spend the first benchmark cycle trying to overturn them.

### GPUDirect status and later proof

Keep these layers of evidence separate:

- **HCA:** ConnectX-5 is new enough for the legacy `nvidia-peermem` path and is
  a capable GPUDirect peer. Its active 100G link proves neither GPU-memory
  registration nor data movement.
- **GPU/support policy:** NVIDIA's current GPU Operator table lists “RTX GPU or
  higher” for DMA-BUF and `nvidia-peermem`, while the general CUDA GPUDirect
  guide describes Tesla and Quadro availability. Therefore there is no basis
  here for a blanket “all GeForce 3090 is unsupported” claim.
- **Installed runtime:** the observed CUDA GPUDirect-RDMA and DMA-BUF
  attributes are both zero. Treat the current `fwuff` configuration as
  **unproven and unsupported by its observed capability report** until a
  current-driver retest succeeds.
- **Kernel path:** DMA-BUF is NVIDIA's preferred path and requires the open GPU
  kernel module, CUDA 11.7+, and Linux 5.12+. Legacy `nvidia-peermem` requires
  compatible MLNX/DOCA-OFED and a loaded peer-memory module.
- **Topology:** one NUMA node is favorable but does not prove a common PCIe
  root. Check `lspci -t`, ACS, BAR1, link widths, and IOMMU pass-through/off.

The exact later proof is a bracketed host-versus-GPU test in both directions:

```text
# host-memory server, then client
ib_write_bw -d mlx5_0 -a
ib_write_bw -d mlx5_0 <peer> -a

# latency server, then client
ib_write_lat -d mlx5_0
ib_write_lat -d mlx5_0 <peer>

# GPU-memory sender; repeat with hosts reversed
ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> -a
ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> <peer> -a
ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> --use_cuda_dmabuf -a
ib_write_bw -d mlx5_0 --use_cuda=<gpu_id> --use_cuda_dmabuf <peer> -a
```

Require payload validation, no host fallback, stable p50/p95/p99, and clean
RDMA/HCA counters. For an NCCL confirmation, capture `NCCL_DEBUG=INFO`,
`NCCL_DEBUG_SUBSYS=NET`, the selected `mlx5_0` transport, explicit GDR
selection, and bracketed HCA counters. Even a passing bandwidth test does not
replace the custom 12,288 B ring-latency measurement. Adopt GPUDirect only if
that end-to-end ring wins the pinned-host A/B.

For the later asynchronous mode, keep a committed draft-KV prefix and two
alternating tentative tail buffers, each tagged with the base committed
position and cycle sequence. Look ahead only two to four tokens. A matching
`ADVANCE` splices the accepted tail into committed state; a rejection, timeout,
or version change discards the tentative tail without copying or rolling back
the committed prefix. Adapt depth from measured acceptance and remaining
output length.

## Break-even model

For a top-k-1 chain of `K` draft candidates:

- `A_K` is accepted draft tokens, `0 <= A_K <= K`.
- `E_K = 1 + E[A_K]` is expected emitted target tokens per cycle.
- `D_L(K)` and `D_R(K)` are local and remote draft compute.
- `V(K)` is target verification time for that candidate block.
- `N(K)` is remote transport/staging/synchronization.
- `O_L` and `O_R` are remaining orchestration overheads.
- `T_1` is vanilla target time for one emitted token.

```text
local_rate  = E_K / (D_L(K) + V(K) + O_L)
remote_rate = E_K / (D_R(K) + V(K) + N(K) + O_R)

remote beats the same local draft iff:
  D_L(K) + O_L > D_R(K) + N(K) + O_R

remote beats vanilla iff:
  (D_R(K) + V(K) + N(K) + O_R) / E_K < T_1
```

For different draft artifacts or precisions, acceptance may differ; compare
the complete rates, not draft latency alone.

For this hidden-row protocol:

```text
N(K) ~= RTT_doorbell
        + (12,288 * E_K + 4K) / effective_byte_rate
        + GPU/host staging
        + CUDA and TP synchronization
```

Thus network cost per emitted token is approximately
`RTT_doorbell / E_K + 12,288 / effective_byte_rate`, plus staging and sync.
At K=1, `E_K <= 2`, so at least half an RTT remains per output token. This is
why a serial remote K=1 design is rejected.

**Local.** The measured one-step run reported acceptance length 1.90 and
acceptance rate 0.95, but it was one prompt and is not a representative
estimate. **Source.** Z.ai reports acceptance length 5.47 versus 4.56 for a
seven-step coding experiment on a GLM-5.1 backbone. Use 5.47 only as a planning
scenario: it would amortize RTT to about `0.183 * RTT` per emitted token, not
as a prediction for this checkpoint.

## Hardware assessment

| Candidate                               | Facts                                                                                                                                                               | Operational assessment                                                                                                                                                                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dwagon`: 2x RTX 3090 + dual-socket AMX | **Local:** 112 physical cores, 768 GB DDR5, two 24 GB RTX 3090s with NV4, current TP2 MTP target                                                                    | Keep as target/verifier. Put the remote coordinator on the target rank closest to ConnectX-5; broadcast candidates to the peer over the existing TP/NVLink path.                                                                                                                                       |
| `fwuff`: RTX 3090 + single-socket AMX   | **Authoritative local context:** 60C/120T Intel AMX, 256 GB DDR5, dedicated 24 GB RTX 3090, ConnectX-5 EDR/100G and legacy ConnectX-3 QDR/40G on its sole NUMA node | **Primary remote drafter.** Reuse current SM86 W8/Marlin and AMXINT4. Keep only layer 78 and its draft state resident. Use pinned-host EDR; reserve QDR for bulk work.                                                                                                                                 |
| Ryzen 9 9950X3D + RTX 5090 under WSL2   | **User-supplied inventory; vendor facts:** 16C/32T, two DDR5 channels, 24 usable PCIe 5 lanes; 5090 has 32 GB, 1,792 GB/s, SM120, no NVLink                         | **Second comparison, not first deployment.** The full 7.516 GiB pure-GPU artifact fits, but AMD cannot run the Intel-AMX kernel. Convert experts, require CUDA 12.8+/SM120 kernel proof, and benchmark compute in isolation. WSL2/network transport needs separate admission before end-to-end claims. |
| DDR5 + RTX 2080 Ti node                 | **User-supplied inventory; vendor GPU facts:** 11 GB GDDR6, 616 GB/s, Turing/SM75                                                                                   | The artifact fits only nominally, leaving about 3.48 GiB before runtime/KV/graphs. No native BF16 path and current kernels may not target SM75. Defer to a small FP16 standalone draft or isolated comparison after `fwuff`.                                                                           |
| DDR5 + GTX 1080 Ti node                 | **User-supplied inventory; vendor GPU facts:** 11 GB GDDR5X, about 484 GB/s, Pascal/SM61, no Tensor Cores                                                           | Reject for native MTP on the latency-critical path. It lacks native BF16 and is outside the recent Turing/RTX GPUDirect prerequisite class. At most use it later for an independently useful tiny FP16 draft.                                                                                          |

**Local derivation.** Merely scanning the 951,892,480-byte head has ideal
bandwidth floors of 1.017 ms on the 3090, 0.531 ms on the 5090, 1.545 ms on
the 2080 Ti, and 1.967 ms on the 1080 Ti. Real draft time is higher and repeats
with each MTP step. This is a lower bound, not a benchmark.

**Source.** AMD rates two DIMMs on the 9950X3D at DDR5-5600 and four at
DDR5-3600, implying 89.6 or 57.6 GB/s theoretical two-channel bandwidth. That
is ample for 12 KiB staging but is not evidence that CPU MTP is fast. It also
does not alter `fwuff`'s priority: the installed Intel AMX artifact is directly
usable there.

The exact CPUs, DIMM population, HCA generation, and PCIe topology of the
2080 Ti/1080 Ti nodes are not bound by a local receipt here. Their table rows
are GPU-generation screening decisions, not node-level performance claims.

## Measured fwuff outcome

Stages 1 through 3 below have now produced live evidence with the Marlin-only
hybrid artifact. The service keeps GLM-5.2 layer 78, its draft KV, and the
AMXINT4 experts resident on fwuff; dwagon retains the complete TP2 target and
performs exact native EAGLE verification.

- Resident EDR round-trip p50 was 9.908 ms at K2, 14.288 ms at K3, and
  18.715 ms at K4. K4 was selected because target verification is much more
  expensive than those extra draft forwards.
- The first paired 240/16 K4 smoke exactly matched the existing local Marlin
  control's token hash. Generation-window throughput improved from 2.57460 to
  4.51690 tokens/s, and end-to-end throughput from 1.83805 to 2.17495 tokens/s.
- A longer attempt exposed a missing native page-64 cache-shadow transaction
  in the remote target worker. Restoring the inherited top-k-one
  `_draft_preprocess_decode()` allocator/mapping/rollback lifecycle fixed the
  sequence-256 CUDA fault.
- The fixed 240/128 K4 request completed in 19.0310 seconds: 8.03440
  generation-window and 6.72587 end-to-end tokens/s, 94/136 draft tokens
  accepted, 3.76471 emitted tokens per verify, and 19/34 verifies accepting the
  full four-token draft. Fwuff finished cleanly at sequence 363 with zero
  service errors.
- Boundary-canonicalized batched prefill then improved the same 240/128 case
  to 2.13279 seconds TTFT, 8.64701 generation-window tokens/s, 7.60963
  end-to-end tokens/s, and 16.8208 seconds end to end. It accepted 96/124
  draft tokens, averaged 4.12903 emitted tokens per verify, and needed only 31
  verifies. Relative to serial fwuff prefill, TTFT fell 33.83%, generation
  throughput rose 7.62%, end-to-end throughput rose 13.14%, and end-to-end
  time fell 11.61%.

The durable paired receipt is:

`/var/lib/exo/benchmarks/glm52-fwuff-remote-eagle-k4-paired-smoke-20260726/attempt-06-native-page-bookkeeping/client-240x128.json`

Its file SHA-256 is
`8982a2e7aa4562b6b53a73d3df18092c78f2c8b6eb46c55f847389ca176568bd`.
The fwuff-side timing receipt SHA-256 is
`24e19614d208e613488ebf4b970bd0021b1b2a636b4b12f54221fa1cce9088ca`.

The winning batched-prefill receipt is:

`/var/lib/exo/benchmarks/glm52-fwuff-remote-eagle-k4-batched-prefill-20260726/attempt-02-boundary-decode-240x128/client-240x128.json`

Its file/content SHA-256 values are
`0bbb22d2bf3a285480021bc8e7a19262565ce4cdc93b400f1e0d5b7355dcd79e`
and
`d29d134df1c16aa5c98f1b89b744870150b32f7cfe713c8b5a0885668ca9ae98`.
The exact historical 16-token prefix passes. The 128-token continuation is not
bitwise equal to the earlier paired run: it shares 27 tokens, then follows a
coherent record-style continuation instead of the earlier repeated-text loop.
Target-only verification and all lifecycle gates remained active, but this
does not replace the still-pending representative quality gate.

The implementation lesson is important. Batching every multi-row commit cut
TTFT but also perturbed twelve five-row decode advances; acceptance fell to
0.440217 and end-to-end time rose to 22.5651 seconds. The final policy keeps
all `N<64` commits on `DECODE`; for prefill-sized `N`, ordinary `EXTEND`
processes `N-1` rows and the final row uses `DECODE`. That restores the
proposal boundary and retains serial decode semantics. The exact N=2/5/64/65
gate passed, and the warmed 128+112 committed work took 47.0510+30.7971 ms
instead of about 1.083 seconds.

This changes the next optimization order again. Fwuff prefill is no longer
material, and raw transport overlap still has little leverage. `K` denotes
the number of fwuff proposals verified in one target pass, not a model
version. The source-level service and bridge now admit fixed K5–K8 requests,
and a two-round isolated K5 wire/runtime smoke passed, but target-verified
evidence remains K4 and there is no paired K5 performance result yet. Run K5
next; retain it only for a quality-clean generation gain of at least 3% over
K4. K6 requires K5 conditional fifth-token acceptance of at least 0.80 and
inferred marginal target-row cost no greater than 25 ms.
Per-round adaptation remains later work because the open-request protocol,
TP tensor shapes, page transaction, and SGLang statistics are launch-fixed.
Any asynchronous lookahead must retain versioned draft KV and exact rollback;
merely double-buffering EDR cannot remove the dependency on newly accepted
target hidden states.

The paired telemetry also confirms that remote drafting does not repair the
target's internal TP imbalance: GPU0 averaged 9.03% utilization and GPU1
94.22%, while fwuff averaged 5.58% in aliased four-millisecond bursts. Fwuff
overlapped four of six GPU0-idle samples, filling some cluster-wide idle time,
but GPU0 itself remains a separate scheduler/spin optimization target.

## Speed-first benchmark and stop gates

Run the next decisive stage, then investigate encountered failures in parallel.
Use focused unit/lint checks for small implementation edits; do not interpose a
broad test campaign between every rung.

| Stage                             | Smallest decisive test                                                                                                                                                                                                    | Continue only if                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Local contract                 | Retain the existing exact 240/16 Marlin control and complete the still-pending representative fixed-logit/perplexity quality gate when a new dwagon-only campaign is authorized. Do not block paired tuning on another short smoke. | Quality passes before remote MTP becomes an unconditional default; historical exact output remains the current paired oracle.                       |
| 1. Isolated `fwuff` compute       | **Completed for K2/K3/K4.** Load only layer 78 with the existing SM86 Marlin + Intel AMXINT4 path; record GPU, AMX, head, and total `D_R(K)`, NaNs, candidates, and rollback.                                               | The service is coherent and its measured compute/resource-isolation value leaves plausible room for one EDR RTT. Otherwise stop before integration. |
| 2. Exact EDR transport            | Use the intended host-staged ring to ping-pong 12,288 B and 24,576 B, plus a 6 MiB prefix. Record p50/p95/p99 and payload validation. A 92 MiB target-KV transfer is an optional negative control, not a sweep.           | p99 `N(K)` fits the Stage-1 break-even budget. Do not add QDR striping or GPUDirect variants.                                                       |
| 3. Synchronous `fwuff` end to end | **Completed at K4 for 240/128.** Preserve target-only thresholds of 1.0, component timers, page-64 lifecycle, exact 16-token prefix, and fail-closed ownership.                                                            | Remote TPOT/throughput beats the historical local path and quality/lifecycle gates pass.                                                            |
| 4. Batched prefill, then adaptive | **Batched prefill completed; fixed K5 admitted.** Keep `N<64` serial; for larger commits use page-correct ordinary `EXTEND` for `N-1` rows and boundary `DECODE`. Exact N=2/5/64/65 and 128+112 gates passed. The depth-8-capable service returned two K5 chains; next run one paired K5 240/128 gate before changing the per-request fixed-depth protocol. | Retain K5 only for a quality-clean generation gain of at least 3% over K4. Try K6 only with at least 0.80 conditional fifth-token acceptance and no more than 25 ms inferred marginal target-row cost. Full-output bitwise identity remains outside this gate. |
| 5. Decode-dedicated scale         | Only after an async c1 win: concurrency 1/3/6, 8K prompts, then more than 200 long generations.                                                                                                                           | Quality, throughput, p95/p99 latency, and cleanup remain better than the local control.                                                             |
| 6. Prefill/SmallEP                | Add QDR-carried bulk SmallEP/prefill or artifact traffic while EDR remains reserved for draft control/data.                                                                                                               | Bulk work does not perturb EDR proposal tails or decode quality.                                                                                    |
| 7. Optional hardware comparisons  | Benchmark the 5090 pure-GPU conversion, then small 2080 Ti/1080 Ti standalone drafts only if the prior result motivates them.                                                                                             | A new host beats `fwuff` under the same complete-rate and quality contract.                                                                         |

If a full rung fails, split work immediately into independent lanes:
`fwuff` GPU/AMX compute, EDR ring/host staging, SGLang state integration, and
quality/rollback. Rejoin at the same failed gate rather than broadening the
test matrix.

## Risk register

| Risk                                                       | Earliest detector                                                     | Response                                                                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Current TP2/K=1 admission assumptions leak into remote TP1 | Focused construction/contract test and first frozen fixture           | Separate target TP size from draft TP size; fail closed on an unbound depth.                                          |
| `fwuff` AMX + GPU draft compute is too slow synchronously  | Stage-1 per-module timing                                             | Stop synchronous integration or proceed only if a bounded async overlap model closes.                                 |
| `lm_head` dominates draft time                             | Per-module isolated timers                                            | Consider a validated token map or a trained small standalone draft; do not hide it with networking.                   |
| RTT/synchronization exceeds compute saving                 | Exact 12/24 KiB ring p99 and break-even equation                      | Stop synchronous remote MTP or proceed only to an explicitly asynchronous prototype.                                  |
| Target TP ranks disagree on hidden state/tree              | Cross-rank hashes before first remote request                         | Fix coordinator broadcast/reconstruction before model benchmarking.                                                   |
| Late reply mutates reused request state                    | Epoch/duplicate/stale fault injection                                 | Enforce incarnation, sequence, committed-position checks and ACK-before-reuse.                                        |
| Acceptance thresholds or sampling drift                    | Launch receipt plus greedy/logit and quality gates                    | Fail closed; restore the target-only verifier and thresholds of 1.0.                                                  |
| GPUDirect remains unproven or is topologically poor        | Capability/topology audit, CUDA/DMA-BUF perftest, then exact-ring A/B | Use pinned-host staging; do not block the benchmark on GPUDirect.                                                     |
| QDR bulk work perturbs EDR proposal latency                | Bracketed EDR p99 while replaying bulk traffic                        | Preserve separate queues/HCAs and stop bulk work if proposal tails move.                                              |
| Optional 5090 conversion or SM120 kernels fail             | Frozen tensor/candidate fixtures and isolated first-kernel test       | Keep `fwuff` primary; fix and bind the converter before any 5090 network work.                                        |
| Old GPUs add latency without useful acceptance             | Isolated 2080 Ti/1080 Ti draft timing                                 | Remove them from the critical path; revisit only as independent small-draft sources.                                  |
| Long-prefix initialization erases short-request gains      | 512/1K/8K prefix timing                                               | Amortize over long generation, cache only request-owned draft state, or disable remote speculation for short outputs. |

## Primary and upstream references

- [Exact speculative decoding](https://arxiv.org/abs/2211.17192)
- [EAGLE feature-level drafting](https://arxiv.org/abs/2401.15077)
- [EAGLE-2 dynamic draft trees](https://arxiv.org/abs/2406.16858)
- [EAGLE-3 training-time feature fusion](https://arxiv.org/abs/2503.01840)
- [SpecInfer distributed tree verification](https://arxiv.org/abs/2305.09781)
- [Sequoia hardware-aware speculative trees](https://arxiv.org/abs/2402.12374)
- [PipeInfer asynchronous speculation](https://arxiv.org/abs/2407.11798)
- [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.2 technical blog, including KVShare/IndexShare MTP](https://huggingface.co/blog/zai-org/glm-52-blog)
- [SGLang speculative-decoding documentation](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/speculative_decoding.md)
- [SGLang EAGLE worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker.py)
- [SGLang EAGLE verification state](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_info.py)
- [SGLang NIXL disaggregation connection](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/nixl/conn.py)
- [SGLang speculative/disaggregation roadmap](https://github.com/sgl-project/sglang/issues/21703)
- [AMD Ryzen 9 9950X3D specifications](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x3d.html)
- [Intel AMX overview](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/what-is-intel-amx.html)
- [NVIDIA Ampere GA102 architecture and RTX 3090 memory bandwidth](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [NVIDIA RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
- [NVIDIA RTX 2080 Ti specifications](https://www.nvidia.com/content/nvidiaGDC/zz/en_ZZ/geforce/graphics-cards/rtx-2080-ti.html)
- [NVIDIA GTX 1080 Ti launch specifications](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1080-ti/)
- [NVIDIA CUDA GPU compute-capability lists](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA CUDA legacy GPU compute-capability lists](https://developer.nvidia.com/cuda/gpus/legacy)
- [CUDA 12.8 Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html)
- [CUDA GPUDirect RDMA guide](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [NVIDIA GPUDirect RDMA prerequisites](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html)
- [NCCL InfiniBand/GPU Direct troubleshooting and perftest procedure](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)
- [ConnectX-5 MCX555A EDR/100GbE product listing](https://docs.nvidia.com/networking/display/connectx5firmwarev16352000lts/firmware+compatible+products)
