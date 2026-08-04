 
• Target verification is the bottleneck because it means the entire target-model forward pass over the speculative block—not merely checking whether draft tokens match.
 
  ### Measured cycle breakdown
 
  From the final EP2 receipt:
     Component                    Time/cycle    Share
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━
   Target verification            47.64 ms    88.0% 
  ───────────────────────────  ────────────  ───────
   Draft generation                6.25 ms    11.6% 
  ───────────────────────────  ────────────  ───────
   Scheduling and acceptance       0.25 ms     0.5% 
  ───────────────────────────  ────────────  ───────
   Total                          54.14 ms     100%
 
  Each cycle commits an average of 2.40 tokens, yielding roughly 44.4 tok/s. See the /tmp/dsv4-oscar-int2/split-history-ep2-confirm-hotspot.json.
 
  At the same acceptance rate:
 
  - 80 tok/s allows a 30.0 ms total cycle.
  - Draft plus overhead already consumes 6.5 ms.
  - Target verification therefore must fall from 47.6 to about 23.5 ms—a 2.0× speedup.
  - Even perfect four-token acceptance at the present cycle time caps throughput around 73.9 tok/s.
 
  ### What happens during target verification
 
  For every speculative block, the target executes:
 
  - 43 sequential transformer layers, all containing MoE.
  - Attention projections and cache operations.
  - 41 compressed sparse-attention layers: 21 C4 and 20 C128.
  - C4/C128 scoring, top-k selection, Oscar INT2 decoding, and sparse attention.
  - MoE routing plus six routed experts per token and one shared expert.
  - Resident GPU MXFP4 experts and CPU-offloaded experts.
  - TP/EP synchronization and result merging.
  - Final normalization and vocabulary projection.
 
  CUDA graphs reduce launch gaps, but they do not eliminate any of this work or the dependency between consecutive layers.
 
  ### What we have ruled out
 
  - The acceptance bookkeeping is only about 0.25 ms. Optimizing it cannot materially help.
  - NVLink is active: all links showed traffic, and measured graph all-reduce latency was approximately 10 µs for 48 KiB. Even pessimistic collective accounting is below roughly
    0.9 ms/cycle.
 
  - The exposed CPU-expert wait averages about 0.045 ms per MoE invocation. CPU work is generally finishing under the resident GPU work, so the visible CPU tail is no longer the
    main bottleneck.

  - The draft model is only 6.25 ms/cycle.

  ### Likely composition of the remaining 47.6 ms

  We do not yet have a synchronized graph-internal profile, so assigning exact percentages would be speculation. The strongest candidates are:

  1. Small-batch resident MoE kernels

     Verification operates at approximately five rows. Tiny, sparsely routed W13 and W2 matrix multiplications cannot fill 82 SMs efficiently. The workload is dominated by
     launch, memory movement, routing, and low occupancy rather than the GPUs’ advertised tensor throughput.

  2. Remaining attention and indexer work

     Split-history Oscar accelerated the isolated T=5 C4/C128 decoder by roughly 2.2× and reduced full target verification by 6.87 ms. That proves attention history was
     important, but projections, C4 scoring/top-k, rotations, cache writes, output projection, and uncompressed attention remain.

  3. Repeated dense bandwidth-bound operations

     Norms, low-rank projections, residuals, shared experts, and the final head are each small but occur across 43 serial layers. At five rows, kernel fusion and memory traffic
     matter more than peak FLOPs.

  4. Rank-tail synchronization

     NVLink transfers themselves are fast, but each dependent stage proceeds at the speed of the slower rank. Small imbalances can accumulate across 43 layers.

  ### Why two 3090s resemble a lower-compute server

  The model is latency-bound, not aggregate-compute-bound. Two GPUs shard each layer, but they do not process the 43 dependent layers simultaneously. Tiny MoE groups and sparse-
  attention tiles launch too few useful thread blocks to saturate either GPU. CPU-offloaded weights also do not consume the second GPU’s tensor-core capacity.

  That is why doubling nominal compute does not translate into doubling decode speed.

  ### Highest-value next work

  - Add CUDA-graph-safe internal timing for attention/indexer, routed and shared MoE, projections/norms, collectives, and final head.
  - Build a fused T≈5 MoE path combining routing, W13, activation, W2, and scatter/merge.
  - Fuse C4 scoring, top-k, rotations, Oscar decode, and attention to avoid intermediate buffers.
  - Re-sweep speculative block length using committed tokens divided by whole-cycle time. Fixed-4 was chosen before the latest attention improvements.
  - Avoid targeting scheduler code or NVLink first; together they cannot recover the approximately 24 ms required for 80 tok/s.

  The central problem is therefore the 43-layer, extremely small-batch target forward. Reaching 80 tok/s requires roughly halving that forward’s latency, most plausibly through
  combined MoE and attention/indexer fusion—not one isolated parameter change.

