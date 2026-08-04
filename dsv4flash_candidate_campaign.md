# DeepSeek V4 Flash local candidate campaign

## Working diagnosis

The final measured decode plateau is the target-verification GPU stage, not
nominal aggregate tensor-core capacity, NVLink, or the CPU queue. Earlier in
the campaign the slow-rank CPU expert tail was material, and inline dispatch
plus MXFP4 scale folding improved it. After that work, 178 small-token samples
show only 0.045 ms mean CPU wait, while the promoted split-history candidate
still spends 47.635 ms in target verification. More arithmetic or residency is
useful only when it removes work from that measured critical path without
reducing speculative acceptance.

The campaign keeps the proven foundation fixed: local TP2/EP2/PP1 on GPUs 0
and 1, the route-weighted 56-thread winner per rank, socket-local NUMA nodes 0
and 1,
524,288-token capacity, decode and speculative CUDA graphs, the qualified
small-row routing kernel, the SM86 small-batch GEMM path, custom all-reduce v2,
and the qualified AMX/AVX row thresholds. The public CLI still spells the byte
carrier `fp8_e4m3`, but admitted physical history storage is calibrated Oscar
INT2: 272 bytes per shared row, 40 bytes per C4 scorer row, and protected BF16
SWA. Generic FP8/INT4/selective-BF16 candidates cannot be represented.
Candidate runs cannot inherit unrecorded `DSV4_*`, `SGLANG_*`, `KT_*`, or
`NCCL_*` settings.

The historical three-repetition Oscar g14 baseline is 34.420 mean / 34.392
median decode token/s. The final coherent split-history reference is 44.690
mean / 44.735 median with 5.542 s median and 5.624 s maximum fresh TTFT. The
campaign uses physical post-graph memory and refuses a run below 1,536
MiB/GPU.

## CPU dispatch and attainable-rate budget

The 1,400-cycle Oscar baseline decomposes into 64.797 ms target verification,
6.605 ms draft work, and 0.250 ms residual step overhead. At its measured
2.4457 committed tokens per cycle, 80 and 90 token/s require complete cycles
of at most 30.571 and 27.175 ms respectively. This makes the target-side CPU
expert stream the only optimization large enough to close most of the gap.

A frozen, same-binary checkpoint layer-20 sweep with 24 experts, 24 route
cases, and 120 timed observations per row count selected **56 workers with a
1,000-us spin window and inline dispatch**. Using the 1,400 observed fixed-4
route counts as weights, it reduced the routed-expert median from 2,665.340 us
to 1,200.465 us, a **2.2203x** gain. The output SHA-256 was identical for every
M1--M6 arm. The former 72-worker hypothesis was rejected: 56-inline was
1.3777x faster than 72-inline at the same spin setting and 1.2757x faster than
72-inline with spin disabled. Seventy-two workers won only M4 and lost the
more important M1--M3 routes.

The complete command, NUMA/affinity contract, all five arms, output digests,
and recomputable weighting are frozen in
[`dsv4_flash_single_numa_inline_dispatch_layer20_2026-08-04.json`](scripts/data/dsv4_flash_single_numa_inline_dispatch_layer20_2026-08-04.json),
SHA-256
`cb63ca6db86aa0a46e24f001485306d44da3199f3d241dede344ed61c10c8667`.

That microbenchmark is promising but not itself a serving claim. Even if its
2.2203x gain accelerated the entire target verify, the Amdahl ceiling is about
67.86 token/s under the present host policy. Transferring the separately
observed 1.179x performance-governor gain raises that full-target optimistic
ceiling to about 77.38 token/s. If only 95% of target verification scales, the
corresponding ceilings are 64.67 and 72.77 token/s. A further stable kernel
gain is therefore still required to cross 80--90 token/s, and the accelerated
fraction must be very high.

The now-measured N128 LUT screen supplies such an offline kernel candidate. Its
two-pair geometric speedup is about 1.2802x, making the combined inline+LUT
target-side factor 2.8424x, or 3.3512x if the transactional host-policy gain
transfers. The resulting Amdahl projections are 82.48/93.38 token/s when the
whole target stage scales, and 77.03/85.92 token/s when 95% scales
(normal/performance host policy respectively). These are planning bounds, not
serving results; end-to-end promotion still depends on the live target-stage
fraction, acceptance, synchronization, and full campaign gates.

The implementation candidate removes the standalone NUMA distributor thread:
the already-pinned `TaskQueue` thread participates as logical worker zero and
the other 55 workers bind deterministically to distinct physical cores.
Admission requires all 56 unique bindings on the expected socket, zero inline
dispatch exceptions, and proof that the last dispatch ran as logical worker
zero on the `TaskQueue` thread. Plain task-queue pinning remains rejected
because it collides with the old worker-zero assignment.

The next kernel candidate folds each safe UE8M0 power-of-two scale directly
into the decoded E2M1 BF16 weight before `VDPBF16PS`. A raw full-checkpoint
scan covered all 33,024 routed-expert scale tensors and 8,657,043,456 scale
bytes: the observed domain is exactly 118--126, with no 0, 1, 253, 254, or 255
values. Serving still validates every loaded buffer and fails closed on OCP
NaN byte 255; campaign admission permits no fallback buffers. LUT and
exponent-add implementations are screened first at the production N-block 128.
The 64/96/256 variants are built only if a parity-clean scale-fold arm shows a
credible gain, avoiding a broad compile sweep with no path to serving value.

The N-block-128 LUT arm cleared that gate in both alternating checkpoint-backed
pairs: route-weighted latency improved **20.84%** and **22.91%**; every M1--M4
arm improved; M1--M6 outputs were bit-identical; and all 72 loaded test buffers
reported zero unsafe, NaN, rejected, fallback, or invalid-mode counts. A narrow
64/96/256 LUT sweep is now justified; only its route-weighted winner may enter
the serving artifact.

## Expert residency curve

One checkpoint expert weighs exactly 13,369,344 bytes (12.75 MiB). The
variable-width planner spends an equal byte budget on both EP ranks but can put
different experts in individual rank/layer pairs. It minimizes the five-prompt
slow-rank critical tail, not aggregate route frequency. Plans are hash-bound,
form an exact disjoint expert cover, and publish all 2 x 43 widths through
`/server_info`.

The measured Oscar curve rejected the offline residency hypothesis:

| Configuration | Mean decode | Committed tokens/cycle | Target verify | Minimum physical headroom |
|---|---:|---:|---:|---:|
| g14 baseline | **34.420** | **2.446** | 64.797 ms | 4,619 MiB |
| +86 expert slots/rank | 34.233 | 2.330 | **61.845 ms** | 3,497 MiB |
| +172 expert slots/rank | 33.462 | 2.300 | 62.592 ms | 2,377 MiB |

The extra residency shortened target verification, but the sampled natural
trajectories accepted fewer draft tokens and erased the gain. Both screens are
therefore rejected. Earlier plans that combined generic INT4 or selective BF16
with k183/k184/k190/k234 are retained only as pre-Oscar capacity studies; the
Oscar-only controller cannot launch them.

## Intentional executable candidates

The manifest is
[scripts/data/dsv4_flash_candidate_campaign_v1.json](scripts/data/dsv4_flash_candidate_campaign_v1.json).
Its executable cache contract is Oscar-only. The current stages are:

1. `g14-oscar-int2-baseline`: completed coherent three-repetition reference.
2. `residency-k86-oscar-int2`: completed screen; performance rejected.
3. `residency-k172-oscar-int2`: completed screen; performance rejected.
4. `residency-k172-oscar-int2-verify5` and
   `residency-k172-oscar-int2-verify6`: predeclared nonlinear DSpark tests, now
   blocked because their k172 prerequisite did not earn confirmation.

The next source-level candidate retains the g14 plan and changes only admitted
Oscar hot paths: masked writer work, once-per-query C4 rotation, and absorbed
inverse output rotation. A host performance-policy A/B is wrapped around one
otherwise identical stage and records exact restoration independently.

## Running the campaign

Freeze source changes before refreshing hashes. The helper recomputes the launcher, selected source-tree, source-artifact, executable-candidate, and offline plan/receipt hashes and writes the manifest atomically. A refresh intentionally invalidates receipts from an older source tree.

```bash
runtime_python=/var/lib/exo/runtimes/dsv4-native-mxfp4/dwagon/amx-prefill-v1/venv/bin/python

$runtime_python scripts/refresh_dsv4_candidate_campaign_manifest.py --write
$runtime_python scripts/run_dsv4_flash_candidate_campaign.py --dry-run

# Establish the three-repetition baseline first.
$runtime_python scripts/run_dsv4_flash_candidate_campaign.py --execute-next

# Screen one explicitly admitted Oscar candidate without walking lower priorities.
$runtime_python scripts/run_dsv4_flash_candidate_campaign.py \
  --execute-candidate residency-k86-oscar-int2 --stage screen

# Confirmation is a separate launch and is accepted only after a qualified screen.
$runtime_python scripts/run_dsv4_flash_candidate_campaign.py \
  --execute-candidate residency-k86-oscar-int2 --stage confirm

$runtime_python scripts/run_dsv4_flash_candidate_campaign.py --summarize

# Run one otherwise identical stage under a single reversible host policy.
$runtime_python scripts/run_dsv4_flash_cpu_policy_campaign.py \
  --work-dir /tmp/dsv4-oscar-cpu-policy-a-b \
  --receipt /tmp/dsv4-oscar-cpu-policy-a-b/cpu-policy.json \
  --execute-next
```

The CPU-policy wrapper publishes its receipt only after the original governor
and energy-performance-preference values are restored and verified. A failed
child stage still restores first and writes a failed receipt; no worker or rank
opens a nested host-global transaction.

`--execute-next` remains available for the complete priority-ordered matrix. The explicit selector does not weaken prerequisites, does not overwrite a receipt, requires a qualified baseline, and requires a qualified screen before confirmation.

Every launch uses five exact/near repeated-prompt phases with controlled radix flushes. It records strict semantic/tool coherency, acceptance, committed tokens/cycle, target-verify time, TTFT, post-graph physical memory, and traffic on all 16 local NVLink counters. `NCCL_P2P_LEVEL=NVL` and a topology label alone are not treated as proof; zero payload movement on any required counter rejects the run. The controller is local-only and has no fwuff execution path.

Natural-stop greedy responses from this model can take different valid reasoning trajectories even for the same prompt, changing completion length and expert routes. The standalone hotspot receipt therefore remains ineligible for a strict paired locality-attribution claim when its output hashes diverge. The campaign treats that drift only as an explicitly counted sampling nuisance: it accepts such a receipt into the cross-launch performance aggregate only after independently revalidating all five natural-stop phase receipts and their semantic checks, natural-stop metadata, canonical prompt fingerprints and phase hashes, trace/graph accounting, disabled recorder and verify-logit diagnostics, hash-bound expert plan, and positive reconciled deltas for every one of the 16 NVLink counters. Each exact and near group must also stay within a 25% completion-length spread inside its five-phase receipt. This bound admits the observed valid 214--259-token reasoning variation while still rejecting gross trajectory changes; the stronger candidate-versus-baseline exact, near, and overall mean-length ratios remain constrained to 0.90--1.10. Paired page-locality attribution remains ineligible whenever the exact trajectory drifts.

The summary records `trajectory_divergent_receipt_count`, `performance_claim_methodology=unpaired_natural_stop_ensemble`, `performance_claim_scope=whole_receipt_decode_ttft_only`, and per-receipt completion ranges. Any other reason for top-level ineligibility remains a hard failure. Candidate and baseline prompt fingerprints must match, and their exact, near, and overall mean completion lengths must each remain within a declared 0.90-1.10 fairness ratio so a shorter/easier completion cannot win on timing alone. Screens are followed by independent confirmation launches, and confidence intervals resample whole five-phase receipts rather than their correlated phases, so a favorable output trajectory cannot establish a winner by itself.

Debug tracing adds overhead. After a winner is confirmed, rerun that exact
hash-bound configuration without trace instrumentation for the final clean
throughput and TTFT claim, then perform the planned PP2 concurrency runs
separately. Full-window Oscar quality remains a separate acceptance gate even
if the short performance campaign succeeds.

## Intermediate combined artifact and split-history candidate

The inline-dispatch plus N128 LUT artifact has now passed both an independent
screen and confirmation. The confirmation measures 39.196 token/s mean,
38.821 token/s median, 54.503 ms target verification, 2.374 committed tokens
per cycle, 5.691 s maximum flushed TTFT, and 4,619 MiB free per GPU. Pooling
screen and confirmation gives a 1.1212x decode ratio over the same-campaign
baseline with bootstrap 95% interval [1.0850, 1.1580]. It is the new EP2
intermediate EP2 reference; the earlier 34.420 token/s table remains historical
evidence for the residency rejection, not the current performance floor.

The post-warmup trace proves positive payload movement on every NVLink
direction (2,618,773--2,618,966 KiB each) and finds only 0.045 ms mean CPU wait
over 178 sampled small-token hybrid observations. The candidate ordering is
therefore revised: first benchmark and qualify an Oscar-only split-history
attention kernel, then revisit the resident-GPU small-batch MXFP4 geometry.
Additional expert residency is not reconsidered without new route evidence.
At this point in the campaign the PP2 ledger remained unopened; the closure
below records its later, exactly-two-launch result.

## Campaign closure: split-history promotion and PP2 rejection

The Oscar-only split-history candidate passed its screen and independent
confirmation and is now the final EP2 campaign reference. Confirmation
measured **44.690 mean / 44.735 median token/s**, **5.542 s median / 5.624 s
maximum** fresh TTFT, **47.635 ms** mean target verification, and **2.402
committed tokens/cycle**. Its separate OpenCode semantic/tool receipt is
coherent, and every local NVLink counter advanced. The production EP launcher
now requires the fixed-address split-history workspace in addition to the
already promoted CPU inline-dispatch/N128-LUT artifact.

The PP2 ledger is no longer unopened: it contains exactly the two requested
launches and is exhausted. The direct 21/22 transfer measured 27.844 aggregate
native decode token/s; the one-knob 22/21 follow-up measured 28.606 token/s.
Both demonstrated true concurrent native and tool streams and positive traffic
on all 16 NVLink counters, but both failed the same lane-1 semantic marker and
missed the concurrent <=7 s TTFT target. Neither PP receipt is eligible and no
PP configuration is promoted. The final campaign handoff is coherent EP2,
not PP2.

The 80 and 90 token/s goals remain unmet. This is a measured plateau, not a
successful stretch claim: after the split-history gain, target verification is
still the dominant 47--48 ms stage.
