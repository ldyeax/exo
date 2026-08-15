# DeepSeek V4 Flash target-only baseline plan

Status: planned only. Do not start, stop, reconfigure, benchmark, or attach a
diagnostic to the live `dsv4f` server while it is serving real work.

## Question this run must answer

Does fixed-4 DSpark increase useful single-request decode throughput over the
same admitted Oscar TP2/EP2 target running ordinary one-token decode?

Acceptance rate alone cannot answer that question. The primary result is the
paired ratio

```text
DSpark useful decode token/s / target-only useful decode token/s
```

on identical prompts, cache states, target placement, target kernels, sampling,
and source revisions. TTFT, end-to-end latency, output trajectory, CPU/GPU time,
and memory are secondary results.

No qualifying target-only baseline has been found for the final configuration.
The prior target-only runs used TP1/PP2/EP1, predated Oscar, used a different KV
cache and prefill path, and were not accepted performance claims. They are useful
proof that SGLang can run DSV4 without speculation, but not a comparison with the
current TP2/EP2/PP1 server.

## Safety boundary

- Wait until the current OpenCode task has completed and the owner has declared
  `dsv4f` available for benchmarking. Do not terminate it to make room.
- Run this baseline on `dwagon`, because only that host has the exact target,
  Oscar artifact, CPU expert placement, and two-GPU topology being measured.
  A small `fwuff` run would not answer the comparison.
- Never run `dsv4f` and `dsv4f_no_dspark` concurrently. Both require both 3090s,
  the same CPU offload pools, and port 30010.
- Use a separate screen name, log directory, and receipt directory. Do not
  overwrite the current live log or prior `/tmp/dsv4-oscar-int2` receipts.
- Keep natural EOS and semantic output visible to the validator. Never use the
  old filler prompt, hidden text, or `ignore_eos` for a performance result.

## Required implementation before the run

The generic launcher already omits all speculative CLI arguments when
`DSV4_DISABLE_SPECULATIVE=1`, but the qualified OpenCode and Oscar EP2 wrappers
intentionally reject that setting. Do not bypass those guards with a direct
low-level invocation; that would silently lose parts of the admitted serving
contract.

Add one explicit diagnostic path:

1. Add a sibling launcher named
   `scripts/dsv4_flash_hybrid_ep2_dwagon_opencode_no_dspark.sh`, or equivalently
   a narrowly validated `target-only-baseline` mode shared with the production
   wrapper. The production launcher's default and fail-closed behavior must not
   change.
2. In target-only mode require `DSV4_DISABLE_SPECULATIVE=1`, unset
   `DSV4_DSPARK_FIXED_VERIFY_LEN`, and omit the draft profile/plan, SPS table,
   STS table, DSpark controls, draft model loading, and verify-graph capture.
3. Retain every target-side setting from the admitted launcher: exact model and
   Oscar admission hashes; TP2/EP2/PP1; frozen g14-p28 target plan; 56 CPU workers
   per rank; Oscar INT2 split history; 524,288 context and token pool; FP8 public
   carrier; full target-decode graph; repaired breakable-prefill graph; radix
   cache policy; chunk size; P2P check; NCCL prewarm; and all qualified target
   kernels. Do not spend the freed draft memory on more target experts or KV.
4. Make the mode fail if any speculative argument or draft weight is present.
   `/server_info` must report `speculative_algorithm` as null/empty and all
   speculative draft/block fields as null/empty. Startup logs must show a
   one-token target decode graph and no draft load or DSpark graph capture.
5. Extend `benchmark_dsv4_flash_hotspot.py` with an explicit decode mode such as
   `--decode-mode target-only`. Preserve the current DSpark default. In the new
   mode, validate the common Oscar/graph/topology contract, validate the absence
   of speculation, skip DSpark controls and trace requirements, and mark
   acceptance metrics as `not_applicable` rather than zero.
6. Aggregate direct decode as total completion tokens divided by total decode
   wall time across phases. Do not average phase rates. Record per-phase token
   hashes, finish reason, TTFT, decode time, useful decode token/s, cached-token
   count, and semantic result. Retain target plan provenance and NVLink proof.

Focused tests must cover a valid target-only `/server_info`, rejection of a
partially loaded drafter, absence of DSpark control calls, weighted aggregation,
and preservation of all existing DSpark behavior. Run the focused pytest file,
then `basedpyright`, `ruff`, formatting, and the normal test suite before using
the new path for a claim.

## Comparability contract

| Dimension | Fixed-4 DSpark arm | Target-only arm |
|---|---|---|
| Source and checkpoint | identical hashes | identical hashes |
| Target cache and kernels | admitted Oscar configuration | identical |
| Target expert placement | frozen g14-p28 | identical |
| Parallelism | TP2/EP2/PP1, one request | identical |
| Context/token pool | 524,288 / 524,288 | identical |
| Prefill/decode graph | breakable / full | identical target graph backends |
| Sampling | greedy, natural EOS | identical |
| Prompts and order | controlled exact/near pair | identical token IDs and order |
| Radix state | same five-phase protocol | identical flush/reuse protocol |
| Instrumentation | off for performance | off for performance |
| Difference under test | block-5 draft, fixed verify 4 | no draft or verification |

The target-only server will use one target row per decode step; that is the
intended difference. Extra memory freed by removing the drafter must remain free
so the experiment measures speculation rather than a simultaneous residency
change.

## Run sequence

### 1. Freeze provenance

After `dsv4f` becomes available, capture its `/server_info`, server log, current
commit, submodule revisions, relevant working-tree diff hashes, model/config
hashes, Oscar admission hash, frozen target-plan hash, GPU topology, driver, and
SGLang launch arguments. Preserve the current fixed-4 production receipt as
historical evidence, but do not use real OpenCode traffic as the paired workload.

### 2. Fixed-4 arm A

With no active user request, run three complete uninstrumented five-phase
hotspot receipts against fixed-4 `dsv4f`:

- the same 2,694-token exact/near prompt pair;
- 256 maximum output tokens;
- temperature 0 / greedy sampling;
- natural EOS;
- cold exact, warm exact without radix, hot exact, hot near, and warm near
  without radix in the existing order;
- cache flushes only where the phase contract calls for them;
- semantic, non-degeneration, stream-completion, both-GPU, target-plan, Oscar,
  and NVLink gates enabled.

Do not enable internal timing, verify-logit dumps, expert recording, or fixed
length diagnostics during the performance receipts. If the current server no
longer matches the frozen contract, relaunch the qualified fixed-4 configuration
after it is safe rather than accepting a mismatched A arm.

### 3. Target-only arm B

Only after `dsv4f` has been deliberately shut down and both GPUs have released
its allocations, launch the new diagnostic entrypoint under screen name
`dsv4f_no_dspark`, logging beneath `/tmp/dsv4f-no-dspark/`. Use the same cache
roots and admitted artifacts but distinct log and receipt paths.

The intended launch shape, after the sibling entrypoint exists, is:

```bash
mkdir -p /tmp/dsv4f-no-dspark
screen -dmS dsv4f_no_dspark -L \
  -Logfile /tmp/dsv4f-no-dspark/server.log \
  env -u SGLANG_DSPARK_DEBUG_DUMP \
  PYTHONUNBUFFERED=1 \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-codex \
  TRITON_CACHE_DIR=/tmp/triton-codex \
  SGLANG_DSV4_INTERNAL_TIMING=0 \
  SGLANG_V4_MXFP4_FUSED_T5_MOE=0 \
  SGLANG_DSV4_OSCAR_FUSED_C4_PIPELINE=0 \
  /root/exo/scripts/dsv4_flash_hybrid_ep2_dwagon_opencode_no_dspark.sh --launch
```

Before sending a benchmark request, require readiness, health, the full
`/server_info` contract, both GPUs active, and no speculative model fields. Then
run the identical three five-phase receipts using `--decode-mode target-only`
and `--http-only`. The exact command should be documented by the harness change;
do not improvise a lower-level request script that omits the semantic gates.

### 4. Fixed-4 arm A2

If practical, relaunch the unchanged fixed-4 server and repeat the three
receipts. A-B-A controls for thermal state, CPU page warmth, and source/runtime
drift. If A and A2 differ materially, the comparison is unstable and must not be
reduced to one speedup number.

### 5. Optional attribution, separate from performance

Only after the uninstrumented comparison, run one receipt per mode with internal
timing enabled. Use it to estimate direct target milliseconds per token and the
DSpark draft/verify/cycle decomposition. Mark these receipts diagnostic-only;
instrumented timings must not replace the uninstrumented throughput result.

## Analysis and decision gates

For every arm report pooled useful decode token/s, median per-receipt rate,
range or bootstrap interval, TTFT, end-to-end latency, completion-token count,
peak memory, and the five cache-state results. For DSpark also report mean
accepted length, generated-draft yield, cap-normalized scheduled acceptance,
and per-position survival. Acceptance is not defined for target-only decoding.

The comparison is valid only if:

- all natural outputs, streams, finish reasons, tool checks, and semantic gates
  pass;
- fixed-4 and target-only output-token hashes match for deterministic prompts,
  or any divergence is investigated and explicitly prevents a lossless speedup
  claim;
- model, source, Oscar, target-plan, graph, topology, prompt-token, and sampling
  hashes match;
- no fallback, eager target decode, missing GPU, or partial worker proof occurs;
- no unrelated user workload overlaps a measured phase.

Retain fixed-4 DSpark only if it produces a repeatable useful-throughput win. A
reasonable promotion gate is at least 5% pooled and median improvement, no
paired prompt regression, and no TTFT or correctness regression. Treat a result
within 5% as overhead-neutral and prefer the simpler target-only path unless a
representative mixed/concurrent workload demonstrates a separate benefit. If
target-only wins, measure and report the loss directly; do not try to rescue
DSpark by quoting acceptance rate alone.

The measured direct target time also gives a clean break-even test:

```text
required mean accepted length = DSpark whole-cycle time / direct target time per token
```

Compare that required value with observed mean accepted length by task domain.
This determines whether better prediction could plausibly cross break-even and
how much improvement is required before investing in retraining or longer draft
geometry.

## Deliverables

- one immutable server contract and log for each launch;
- three fixed-4 A receipts, three target-only B receipts, and preferably three
  fixed-4 A2 receipts;
- an output-hash and semantic parity table by phase;
- pooled throughput and TTFT comparisons with uncertainty;
- one optional diagnostic timing receipt per mode;
- a final keep/disable-DSpark decision and a quantified break-even accepted
  length, with all rejected or ineligible runs retained but clearly labeled.
