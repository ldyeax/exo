# Kimi K3 local inference campaign

## Objective and completion gate

This journal records the deployment, tuning, research, failures, fixes, and
measurements for running Kimi K3 on dwagon and fwuff. The original baseline
campaign was complete only after both `UD-Q2_K_XL` and `UD-IQ2_XXS` had five
coherent, comparable served-inference runs with measured prompt-processing and
decode rates plus the supporting latency, memory, power, and system statistics
that were available.

The operating bias is deliberately aggressive: start from the best-supported
high-performance configuration, measure real behavior, and spend time fixing
observed bottlenecks rather than constructing a large synthetic preflight.

Original baseline status: **complete**. Both requested quantizations have
independent 5/5 semantic, 5/5 exact-length performance, and 5/5 paired
completion gates. Later `tensor_memset` and file-aware-loader follow-ups were
explicitly reduced to one representative Q2 pair each; those A/B probes do not
replace the five-run distributions. On this hardware, `UD-IQ2_XXS` is the
clear interactive/default quant:
all-local auto-fit reaches 7.302 prompt and 1.622 decode token/s mean, versus
5.075/0.373 for distributed `UD-Q2_K_XL`. Q2 remains the quality-first choice.

## Hardware and storage baseline

| Host | Relevant resources | Initial state |
| --- | --- | --- |
| dwagon | 2 x RTX 3090 (48 GiB aggregate VRAM), 768 GiB-class system RAM | About 715 GiB RAM available; local root volume has only about 73 GiB free |
| fwuff | 1 x RTX 3090 (24 GiB VRAM), 256 GiB-class system RAM | About 226 GiB RAM available; GPU idle |
| `/mnt/sanic` on fwuff | Shared model storage exported to dwagon over NFS | 5.6 TiB filesystem, about 1.9 TiB free at campaign start |
| Interconnect | 100 Gb/s-class InfiniBand hardware between hosts | Existing DeepSeek work found activation traffic cheap relative to expert-weight memory traffic; transport mode must still be measured |

Both hosts report pre-production Intel identifiers rather than retail SKUs:
`Genuine Intel(R) 0000`, CPUID family 6/model 207/stepping 1. Dwagon is two
sockets x 56 cores x two threads (112 physical/224 logical, two NUMA nodes);
fwuff is one socket x 60 cores x two threads (60 physical/120 logical, one
NUMA node). Dwagon's 3090s are at PCI addresses `16:00.0` and `d8:00.0` with
driver 610.43.03 and a CUDA 13.1 build toolchain. Fwuff's 3090 is at
`6a:00.0` with driver 595.84 and CUDA 12.4. Final Q2 uses fwuff's CPU-only RPC
device; its 3090 is deliberately unused.

The two requested text-model payloads fit together in the available space:

| Quant | Repository payload | Approximate GiB | Expected free space after download |
| --- | ---: | ---: | ---: |
| `UD-Q2_K_XL` | 861,277,858,912 bytes | 802.13 | About 1.0 TiB |
| `UD-IQ2_XXS` | 711,067,773,664 bytes | 662.23 | About 0.3 TiB |

The optional BF16 vision projector is under 1 GiB and does not materially
change that calculation. Actual free space will be rechecked between downloads
because other transfers are active. These are snapshot estimates, not final
receipts: unrelated concurrent activity subsequently freed substantial space,
so the independently measured post-IQ2 value of 2,451,280,392,192 free bytes
is authoritative.

## Reproducibility pins

- Model repository: `unsloth/Kimi-K3-GGUF`
- Model revision selected at download start:
  `3d4b61ab4b6789d401191c476cbb4567246db8f5`
- Revision timestamp reported by Hugging Face: 2026-07-29 18:03:55 UTC
- Repository-local pinned GGUF manifest:
  `scripts/data/kimi_k3_gguf_manifests.json`, SHA-256
  `6e58b10b6505c833e274e8a4c2ff6b0cc97c66854f883504b82ac262a567a000`.
  It records the revision's exact Hugging Face LFS SHA-256 OID and length for
  every requested shard. The sorted per-quant manifest hashes are
  `34273843bf7edc60eb9f74fcaa0265e80e2ac30be172b653387a090dde80a387`
  for Q2 and
  `7b166810d1ee0778067e20f854347bc52786bcdacb4eb2079dac6806e2d9a689`
  for IQ2.
- Q2 layout: 19 GGUF shards; the first metadata-heavy shard is only about
  6.9 MB, while the weight shards are mostly 47--50 GB.
- Text runtime: Unsloth's llama.cpp fork at the full-size fix commit
  `47c5bbdfd5ab5e847098f791a2cb9c0c90fb7dbd`, runtime version 10162, built
  independently on both hosts with CUDA SM86 and TCP RPC.
- The pinned Q2 metadata contains a 24,696-byte K3 Jinja template. Its exact
  raw SHA-256 is
  `05bb501f8ac31fa6b0bf04803b5ada49abf9cdd51c3c90a4719b739df0000722`.

## Starting hypotheses carried forward from earlier experiments

1. Kimi K3 is a 2.8-trillion-parameter MoE with roughly 104 billion active
   parameters per token. At these quant sizes, CPU memory bandwidth and expert
   locality are expected to dominate decode, while the three 3090s are most
   valuable for dense/shared tensors, attention, and selected layers that are
   repeatedly touched.
2. Pipeline-parallel placement that serializes unequal hosts is likely to lose
   to a placement that balances actual per-token work. The DeepSeek V4 work in
   this repository already demonstrated this on the same machines.
3. The Q2 quant is the quality-first target. IQ2 is the single-large-host and
   higher-throughput fallback: it fits comfortably in dwagon's available RAM,
   while Q2 requires either carefully managed mmap/offload or distribution.
4. A coherent served request is a more useful acceptance gate than a standalone
   kernel microbenchmark. Each reported run will therefore preserve the answer
   (or a deterministic semantic check), server timings, configuration, and
   contemporaneous resource telemetry.

## Work log

Wall-clock times in prose are host-local EDT (`UTC-04:00`) unless explicitly
marked UTC; JSON receipts use RFC 3339 UTC.

### 2026-07-30: campaign start

- Confirmed about 1.9 TiB free on `/mnt/sanic`, enough to begin Q2 without
  deleting existing models.
- Confirmed fwuff's RTX 3090 was idle. Unrelated active transfers and workers
  were left running; GPU-consuming services may be stopped if they create
  observed contention.
- Selected immutable model revision
  `3d4b61ab4b6789d401191c476cbb4567246db8f5` so later repository changes cannot
  silently alter benchmark inputs.
- Began from the current Kimi-aware llama.cpp development path rather than the
  older llama.cpp nested under the already-modified ktransformers checkout.
  Exact runtime commits and incorporated upstream work are recorded below.

### 2026-07-30: transfer and runtime bring-up

- The first aria2 invocation revealed that multiple URLs on one command line
  are interpreted as mirrors for one download. It therefore fetched only the
  6,934,144-byte first shard. The file was retained and renamed to its correct
  shard name.
- Added `scripts/download_kimi_k3_quant.sh`. It obtains the pinned repository
  tree, emits one explicit output name per URL, resumes partial files, and
  validates every completed byte count against Hugging Face metadata.
- Eight range connections per file then exposed a Hugging Face Xet behavior:
  a redirect is signed for one exact byte range, but aria2 tried to reuse it
  for different ranges and received HTTP 403. Reducing each file to one
  connection fixed the observed failure. Four files still download in
  parallel. The corrected service sustained roughly 180--250 MiB/s early in
  the transfer with no restart.
- A second Xet failure appeared late in shards 3 and 4: their signed range
  URLs returned HTTP 403 after most bytes had arrived. The files had their
  correct logical lengths but retained aria2 controls and one sparse hole
  each: 666,107,904 and 763,908,096 bytes, respectively. A single clean
  restart preserved every downloaded extent, obtained fresh signed URLs, and
  filled exactly those 1,430,016,000 missing bytes. Both controls disappeared
  and the shard sizes remained exact. The first restarted invocation hit a
  transient Hugging Face CDN DNS failure; the unit's existing restart policy
  retried once, after which shards 3, 4, and 6 completed and the later shards
  resumed with no current-invocation errors.
- Cloned the active K3 development lineage into isolated source trees on both
  hosts, leaving the dirty ktransformers checkout untouched:
  - upstream K3 text/chat base
    `cf67f0d24511864d2d3da0769108fd6fc16d00d1`;
  - Unsloth full-size text fix
    `47c5bbdfd5ab5e847098f791a2cb9c0c90fb7dbd`;
  - later MoonViT image commit
    `efc8bc38f0a9950cbb10ccef2cf48b951c39d3b2`.
- Both hosts successfully built CUDA SM86 plus RPC binaries from the branch.
  The benchmark runtime was then narrowed to the exact text-only full-size
  commit `47c5bbdf...`; the later vision converter has unresolved review
  findings and contributes nothing to the requested text benchmark.
- The text runtime is configured with native CPU code, static libraries,
  CUDA SM86, RPC, no UI or vision library, and explicit TCP RPC. Native
  InfiniBand/RDMA mode is not claimed: llama.cpp documents its negotiated RDMA
  transport for RoCEv2, while this fabric is native InfiniBand.
- Runtime enumeration found dwagon's two GPUs almost full. The owners were the
  Docker services `qwen3-tts-nano-gpu0` and `qwen3-tts-nano-gpu1`, using about
  23 GiB on each card while idle. Both were stopped under the task's explicit
  GPU-service authorization; each 3090 then reported 24,123 MiB free. Other
  voice services and transfers were left alone.

### 2026-07-30: InfiniBand recovery and RPC topology

- After the host reboot, all three physical InfiniBand links were `LinkUp` but
  stuck in `INIT`: fwuff's enabled OpenSM unit had been skipped because
  `ib_umad` was not loaded, and the nonpersistent IPoIB addresses were gone.
- Loaded `ib_umad`/`ib_ipoib` on fwuff, started its existing `PORTS=ALL`
  OpenSM unit, and restored all three fabrics to `ACTIVE`: one 100 Gb/s EDR
  and two 40 Gb/s QDR links.
- Used the repository's GUID-pinned, rollback-capable provisioner rather than
  assigning interfaces by observed name. It restored:
  - EDR: `dwagon 10.44.0.1/30` to `fwuff 10.44.0.2/30`;
  - QDR A: `10.44.1.1/30` to `10.44.1.2/30`;
  - QDR B: `10.44.2.1/30` to `10.44.2.2/30`.
- Saved exact apply/rollback receipts at
  `/var/lib/exo/peer-artifact-deployment/ipoib-{dwagon,fwuff}-kimik3-20260730.json`.
- A one-shot eight-stream iperf3 test over EDR delivered 92.6 Gb/s at fwuff's
  receiver for ten seconds. This is ample for layer-boundary activations and
  one-time remote tensor loading; it does not prove llama.cpp RPC itself will
  achieve the same rate.
- The pre-existing `/mnt/sanic` NFSv4.2 view was still using fwuff's
  `192.168.40.93` Ethernet address. Fwuff's NFS service was reachable over
  EDR, but its export ACL admitted only dwagon's Ethernet address. Added an
  exact, read-only export for `10.44.0.1`, retaining the original client
  clause and a dated copy of `/etc/exports`.
- A first NFSv4.2 EDR mount attempt was rejected as evidence despite naming
  `10.44.0.2`: Linux trunk discovery silently reused the existing NFS client
  and `findmnt` still showed `addr=192.168.40.93`. The dedicated model view at
  `/mnt/sanic-edr` therefore uses NFSv3, `nosharecache`, eight TCP connections,
  1 MiB reads, and a hard read-only mount. Its receipt shows
  `addr=10.44.0.2`, and routing selects `ibs5` with source `10.44.0.1`.
  A deliberately small 1 GiB direct read from a completed Q2 weight shard
  sustained 2.4 GB/s. This is already about 1.9 times the line-rate ceiling of
  10 GbE and makes EDR the selected cold-load path without perturbing the
  active model download.
- Verified llama.cpp can expose a CPU together with GPUs through RPC. A
  dwagon endpoint appeared at the client as one 773,610 MiB CPU device and two
  24,123 MiB CUDA devices. This also exposed a dangerous implementation
  detail: RPC reports total physical CPU RAM as free rather than current
  available RAM, so automatic fit is not safe.
- Selected the best-chance initial Q2 topology:
  - main server on dwagon;
  - about 635.1 GiB of weights in dwagon CPU RAM, interleaved over both NUMA
    nodes;
  - 17.017 and 17.229 GiB of weights on the local dwagon 3090s;
  - about 132 GiB on a fwuff CPU-only RPC endpoint over EDR.
  This leaves roughly 95.8 GiB of dwagon RAM margin before runtime allocations
  and avoids a current community-reported K3 `SOFT_MAX` failure on RPC CUDA.
  Fwuff's 3090 is reserved as a later measured A/B, not assumed beneficial.
- An exact loader audit corrected the initial tensor-split notation before the
  first model load. In this llama.cpp revision the values are normalized into
  layer ordinal boundaries; they are not byte or GiB targets.
  `--gpu-layers 20 --tensor-split 20,20,130` would round CUDA0 up to three K3
  blocks and likely exceed 24 GiB. The admitted starting split is `2,2,16`,
  which maps two blocks to each local 3090 and fifteen blocks plus output to
  fwuff's CPU, leaving blocks 0--73 on dwagon CPU. The launch gates now require
  at least 720,000 MiB host-available memory and 23,000 MiB free on each GPU;
  the idle system clears both while preserving graph/context headroom.
- The starting load mode is eager `none`, not mmap. GPU and RPC tensors are
  copied in either case, but mmap would leave roughly 630--640 GiB of local CPU
  tensors as evictable NFS-backed pages. Eager anonymous buffers cost a slower
  one-time load over the new 2.4-GB/s EDR NFS path and buy stable, disk-free
  steady inference. There is no swap, and both hosts retain explicit RAM
  margins; mlock is unnecessary.
- The benchmark's explicit slot erase exposed another exact-version
  requirement: all `/slots/{id}?action=...` requests are rejected unless
  llama-server starts with an already-existing `--slot-save-path`. The
  launcher now creates a private writable path, passes the option, and checks
  that this exact runtime advertises it. Prompt cache, idle-slot cache, and
  cache reuse remain disabled independently.
- Replaced fwuff's still-idle RPC unit before model load. Its original
  `--cpunodebind=0` admitted both physical CPUs and sibling hyperthreads; the
  unit used for that phase bound its 60 worker threads to physical CPUs
  `0-59` and memory node 0. It restarted once intentionally, was active on
  only the private EDR address, and had zero runtime restarts.
- Prepared, but did not promote, a surgical output-head A/B:
  `--override-tensor '^output[.]weight$=CUDA0'`. The anchored regex matches
  only the Q8_0 `output.weight` tensor, shape `(7168, 163840)`, exactly
  1,247,805,440 bytes. It raises CUDA0's raw weights from 17.017 to 18.179 GiB
  and leaves about 5.4 GiB on the currently idle card. The final residual
  mix/norm remain on RPC0, then only a roughly 28-KiB normalized hidden vector
  crosses to CUDA0 for the fully touched 1.162-GiB head matvec. The exact
  no-load CUDA/RPC parser check passes. Baseline peak VRAM, coherence, and a
  matched throughput run will decide whether the extra graph split and reduced
  headroom are worthwhile.
- Preserved the relevant upstream work in clean local branches rather than
  touching the modified ktransformers submodule. On both hosts,
  `baseline/kimi-k3-fullsize` points exactly to `47c5bbdf...`; Git verifies
  that the upstream K3 text head `cf67f0d...` is its ancestor. The separate
  vision worktree remains isolated and is not part of this text campaign.
- Re-audited the live upstream and fork heads before benchmarking. There is no
  newer K3 text-graph fix: upstream PR 26185 remains at `cf67f0d...`, and
  K3's model/context sources do not change after the pinned full-size fix.
  The fork's `23fac110...` head is a 133-file nightly carrier combining the
  unneeded vision commit, a broad upstream merge, and draft Inkling work, so
  it was deliberately not merged. One small upstream change is relevant only
  to the later output-head experiment:
  `f5b9bd39b56c7a7839a9795a100b6a00b84ac961` adds RPC protocol-v5
  `tensor_memset` and fixes the confirmed `--override-tensor` plus RPC crash.
  The accepted benchmark stays on the exact pin. A matching
  `benchmark/kimi-k3-rpc-memset` branch was preserved separately for a future
  output-head A/B rather than changing runtime code inside this campaign.
  On both hosts that branch is clean at
  `a30437bc3a2a661d1e9aad71b1160d9ad9bbfec1`, directly atop the accepted
  `47c5bbdf...`; its stable patch-id
  `4ff19987893206bb1155d542e111c4abd9caecb5` exactly matches upstream
  `f5b9bd39...`. The isolated worktree is
  `/var/lib/exo/sources/llama.cpp-kimik3-text-47c5bbdf-rpc-memset` on both
  machines. Dwagon's CUDA/RPC build is complete. Fwuff's worktree is also
  clean. At that stage its validation build was safely paused at 145/430
  objects: the unrelated 3.7-TiB root LV had zero bytes available, and
  resuming a build with a roughly 6-GiB observed peak would have been unsafe.
  No unrelated data was deleted. The broad nightly/vision carrier remained
  rejected. A later follow-up moved build data to Sanic and completed 430/430,
  as recorded below.
- Audited the benchmark harness against the pinned server endpoints and K3
  parser. Slot erase, exact input-token calibration, thinking-budget fields,
  final SSE usage/timings, and all expected timing names match this revision;
  the loop can finish only after five accepted semantic/performance pairs.
  The audit did reveal one diagnostic hole: llama-server may return an
  HTTP-200 SSE chunk containing `error` on a runtime failure. The harness now
  raises the bounded server error immediately instead of later misreporting
  missing tokens/timings; Ruff, Basedpyright, and bytecode compilation pass.
- Q2 completed in 1 h 4 min. The downloader and an independent check both
  found exactly 19 GGUFs totaling 861,277,858,912 bytes, zero `.aria2`
  controls, and no file whose allocated bytes were below its logical size.
  The post-Q2 space check found 3,162,348,978,176 bytes available on
  `/mnt/sanic`, enough for IQ2 with more than 2.4 TB left over. The hardened
  downloader therefore began the pinned 16-shard `UD-IQ2_XXS` transfer
  immediately.
- IQ2 completed successfully after 2h18m wall time, including its deliberate
  freeze during Q2 cold loads. Aria2 reported all 16 items `OK`, and the
  downloader verified exactly 711,067,773,664 repository bytes. An independent
  post-exit audit found 16 GGUFs, zero `.aria2` or temporary controls, the same
  exact logical total, 711,067,860,992 allocated bytes, and zero sparse files.
  `/mnt/sanic` retained 2,451,280,392,192 bytes free. The overlapping Q2
  diagnostic was stopped immediately after this stable-storage gate; it had
  produced three accepted pairs at prompt rates 4.065, 4.008, and
  3.875 token/s and decode rates 0.509, 0.379, and 0.346 token/s.
- A reviewed transactional holder now keeps every CPU policy at
  `performance/performance` during measurement: 224 policies on dwagon and
  120 on fwuff. It wraps the existing crash-recoverable helper, writes a
  hashed atomic receipt, rejects path collisions, tolerates signal storms,
  and restores the pre-campaign policy on termination. A real apply/restore
  smoke test passed before the long-lived services started. This is justified
  by the same-hardware DeepSeek evidence of roughly 18% better decode, while
  automatic NUMA balancing remains unchanged until there is measured evidence
  to alter it.
- A pre-existing CPU-only `voice-frontend` container is in a one-minute
  restart loop, and the remaining voice stack performs small periodic health
  checks. This cadence predates the final set, is stationary across its runs,
  and uses neither benchmark GPU; it was documented but not stopped because
  it is unrelated to the user's GPU-service authorization.
- The simultaneous IQ2 transfer was allowed to run until measured contention
  appeared rather than being stopped speculatively. At 34.449% complete its
  writers held one `/mnt/sanic` RAID member near 100% utilization, with roughly
  500--650 ms write latency, while the Q2 NFS reader fell to about
  87--110 MiB/s. `systemctl freeze` preserved the aria2 processes, partial
  extents, and resume controls without cancelling the download. Q2 then
  recovered to roughly 250--270 MiB/s. IQ2 was thawed after the first Q2 load
  became resident so the remaining writes overlapped compute rather than
  cold-load reads; it subsequently passed the exact 16-shard completion gate.
- The first complete Q2 residency pass exposed a real graph-memory limit:
  `ubatch=4096` requested an 8,908,255,104-byte (8,495.57-MiB) CUDA0 prompt
  compute buffer after that card already held 17.017 GiB of weights.
  `cudaMalloc` failed, and the server exited cleanly before serving a request;
  there was no host OOM or swap. The failed server log and both-host telemetry
  are preserved under
  `baseline/failed-ubatch4096/`. The immediate correction keeps the exact
  topology, logical batch, and 32K context but halves the physical microbatch
  to 2,048. This is expected to halve the dominant prompt graph allocation
  while retaining enough prompt parallelism for the 512-token benchmark.
- The corrected 2,048-microbatch load reached healthy service after
  35m53s. It is viable but tight: live steady allocations leave only about
  670 MiB free on CUDA0 versus 4,138 MiB on CUDA1. The originally prepared
  CUDA0 output-head override is therefore rejected on measured capacity.
  The later device-reordered N16 profile places the output head and last heavy
  blocks on CUDA1 naturally, retaining 2,170 MiB final headroom without an
  override. That achieves the intended placement while staying on the pinned
  runtime, so the RPC-`tensor_memset` A/B is closed as unnecessary here.
- The first accepted diagnostic pair measured 512-token prompt processing at
  4.0653 token/s and exact 128-token decode at 0.50865 token/s after a
  coherence-passing arithmetic gate. Telemetry confirms Q2 is fully anonymous
  resident (roughly 638 GiB on dwagon and 133.6 GiB in fwuff's RPC process),
  with zero swap and zero measured major faults during the performance
  request. IQ2 nevertheless writes roughly 184--225 MiB/s and saturates a
  RAID member. Direct Q2 disk or network contention is absent, but IQ2 will
  finish and change background conditions during the five-run set. The
  in-progress set was therefore diagnostic only; after IQ2 finished, all five
  Q2 pairs were restarted under one stable condition rather than mixing
  pre- and post-download runs.
- A source-correlated live audit found the Q2 baseline is CPU/RPC-compute
  bound. Fast and slow decode windows carry the same roughly
  1.56--1.59 MiB/token to fwuff and 0.641--0.648 MiB/token back, have zero
  major faults and stable NUMA residency, and use roughly 97--100 effective
  dwagon cores. EDR carries only about 7/3 Mbit/s. The return payload almost
  exactly identifies the 163,840-element FP32 logits vector; both GPUs execute
  briefly and then wait for CPU stages. Decode variation is internal
  route/cache/thread-efficiency variation, not storage or fabric saturation.
- The best-chance stable Q2 profile is therefore more aggressive than a
  head-only override:
  `--device CUDA0,RPC0,CUDA1 --gpu-layers 16 --tensor-split 2,11,3
  --ubatch-size 512 --no-op-offload`. Its exact intended placement is CPU
  blocks 0--77, CUDA0 blocks 78--79, RPC blocks 80--90, and CUDA1 blocks
  91--92 plus output. Projected raw residency is 17.0170 GiB on CUDA0,
  95.1957 GiB on RPC, and 20.5515 GiB on CUDA1. Relative to baseline it
  removes 37.568 GiB from the slower remote stage, adds 34.246 GiB to dwagon
  while retaining roughly 58 GiB steady host margin, and eliminates the full
  logits return. The 512 microbatch handles every fixed 512-token performance
  prompt in one piece while greatly reducing graph reserve.
- `--no-op-offload` is selected from measured behavior rather than convention.
  At batches of at least 32, the default streams CPU-weight operations through
  the first CUDA device; during Q2 prompts CUDA0 averages roughly 30% sampled
  utilization while CUDA1 is idle and the local process uses only 7--12 CPU
  cores. For this sparse model and exact 512-token workload, disabling that
  transfer is the highest-confidence prompt optimization and is inert for
  batch-one decode. The overlapping diagnostic set was terminated at three
  pairs when IQ2 completed; the required five Q2 pairs use this tuned profile
  with the background download stationary.
- The first local telemetry stream could not truly sample at 1 Hz:
  `/proc/<pid>/smaps_rollup` alone took 3.65 seconds on the 638-GiB process,
  and periodic `numa_maps` scans created much larger gaps. The sampler now
  accepts `--smaps-every 0 --numa-every 0`; a nine-sample test held a
  1.00003-second median and 1.00021-second maximum interval. The summarizer's
  status-only fallback correctly selects and interpolates `VmRSS`. The final
  Q2 set uses this mode; expensive live `numa_maps` scans are deliberately
  omitted from its timing windows.
- Fwuff's first final-set sampler stopped after 3,005 valid samples when its
  unrelated 3.7-TiB root filesystem reached 100% and the next 12-MiB JSONL
  write returned `ENOSPC`. The CPU RPC worker remained active with the same
  PID and zero restarts, and dwagon telemetry remained continuous. Sampling
  resumed on spacious `/mnt/sanic` at true 1 Hz with `Restart=on-failure`.
  The approximately 14-minute remote-only gap spans pair 3; that pair retains
  exact server timings and complete local telemetry but is explicitly labeled
  lacking continuous fwuff statistics. No bytes were deleted from the full
  unrelated root filesystem.
- RPC cold loading also isolated the single-flow transport ceiling. The EDR
  fabric delivers 92.6 Gb/s across eight iperf streams, but llama.cpp's one
  TCP stream moved remote tensors at only roughly 0.7--2.2 Gb/s. Both EDR
  IPoIB interfaces are currently in datagram mode with MTU 2,044 even though
  their maximum MTU is 65,520. Linux and NVIDIA document connected IPoIB as
  reducing packet count and improving large-message performance at the larger
  MTU, provided both peers are changed together. A receipt-backed connected
  mode/65,520-MTU A/B was considered, but the repository's own deployment
  record says this exact mlx5 IPoIB device previously rejected connected mode
  with `EINVAL`. Since live decode then used only about 7/3 Mbit/s, the
  capability probe was closed rather than spending a model reload on a
  non-bottleneck. The original GUID-pinned datagram topology remains active.
- The final Q2 candidate passed an exact no-load parser and capacity gate, then
  began a clean eager load as
  `--device CUDA0,RPC0,CUDA1 --gpu-layers 16 --tensor-split 2,11,3
  --ubatch-size 512 --no-op-offload`. Before graph construction the observed
  weight allocations are exactly 17,692 MiB on CUDA0 and 21,312 MiB on
  CUDA1. The local anonymous stage stabilized near 705.8 GB RSS while the
  reduced RPC stage began growing toward its projected 95.2 GiB. This
  confirms that reordering the devices moved the output head and last heavy
  blocks locally without the RPC `tensor_memset` patch. During and immediately
  after the sustained copy the kernel logged five corrected APEI
  general-processor machine-check events on socket 0 (two at 05:41, one at
  05:46, and two at 05:48). Hardware marked every event corrected with no
  required action; the last was 12 seconds before the benchmark harness
  started, the service stayed active, and there was no OOM, swap, or GPU
  error. No further machine check has appeared during the accepted set.
- The N16 candidate reached healthy service after 25m37s. Final graph/context
  allocation raised GPU usage to 18,286 MiB on CUDA0 and 21,954 MiB on
  CUDA1, leaving 5,838 and 2,170 MiB free. Its sacrificial 512-plus-8-token
  request passed and measured 4.934 prompt token/s and 0.417 decode token/s.
  The prompt result is about 23% above the approximately 4.0-token/s
  diagnostic baseline and supports the hypothesis that removing short-batch
  operation offload helps this sparse CPU-resident stage. Because placement,
  remote weight volume, device order, and microbatch also changed, this is not
  a controlled operation-offload A/B. It is a warmup result, not one of the
  five accepted measurements.
- Eleven additional corrected socket-0 general-processor events (386--396)
  arrived only after the Q2 campaign had completed, between 06:59:31 and
  07:05:04, while its receipts were being captured and the process was being
  torn down. They do not overlap any accepted Q2 request. The full sequence is
  still a real hardware-health signal to investigate after the campaign even
  though every event was corrected, no service restarted, and no answer gate
  failed.

### Local-document evidence trace

The campaign reviewed the repository's Markdown corpus before fixing the final
profiles. `coordinate.md` was deliberately excluded at the user's direction;
process-only documents were treated as instructions rather than performance
evidence.

- [`deepseekv4pro.md`](deepseekv4pro.md) provides the closest controlled
  evidence on these hosts: physical cores beat SMT, hard process-wide
  `membind` can OOM despite capacity on the other node, CPU performance policy
  improved decode by about 18%, fragmented RPC topologies lost to simpler
  ones, and expert-aware GPU residency helped much more than HCA placement.
  K3 therefore uses physical-core pools, interleaved/spill-capable host RAM,
  held performance policy, and the smallest RPC stage that fits.
- [`FWUFFYDWAGON.md`](FWUFFYDWAGON.md) establishes the dual-NUMA 112-core
  dwagon, 60-core fwuff, three 3090s, NVLink pair, and host-staged
  InfiniBand posture. Its fixed-artifact, warmup, semantic-gate, repeated-run,
  and telemetry discipline directly defines this campaign. Its warning that
  NFS is a load path rather than an inference hot path is why accepted models
  use eager anonymous residency.
- [`OSDI26.md`](OSDI26.md) argues that sparse batch-one decode is dominated by
  CPU memory traffic, that dense/attention tensors should consume VRAM while
  routed experts remain in DDR, and that the two sockets should behave as one
  bandwidth machine rather than serial pipeline stages. That is the IQ2
  dense-only auto-fit design. Its neutral/slower two-batch overlap and failed
  stream-loading results were not repeated here.
- [`parallelism.md`](parallelism.md) shows why batch-one pipeline latency adds
  its serial stages and why NVLink does not create a transparent 48-GiB pool.
  Q2's llama.cpp RPC is treated as a capacity topology, not a scaling claim;
  IQ2 stays all-local.
- [`speculative-infiniband.md`](speculative-infiniband.md) requires
  application evidence rather than link-state assumptions and rejects
  latency-critical EDR/QDR striping without a complete A/B. Current Q2 decode
  moves only about 7/3 Mbit/s over a measured 92.6-Gbit/s EDR path, so
  computation—not fabric bandwidth—is the target.
- [`infiniband_cards.md`](infiniband_cards.md) records the successful QDR
  cross-flash and its PCIe-limited 52.9-Gbit/s dual-port result. The earlier
  [`infiniband_card_limitation.md`](infiniband_card_limitation.md) is
  superseded chronology. EDR remains the K3 RPC/NFS path; further QDR tuning
  has no plausible payoff here.
- [`test_results.md`](test_results.md) independently supports performance
  policy, full physical-core pools, larger expert residency, and treating
  regressions as evidence. It also found batch-one PP slower and HCA-local
  stage movement nearly neutral, reinforcing placement work over network
  micro-tuning.
- [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md) supplies the cold-cache,
  discarded-warmup, exact-length, repeated-sample, and one-Hz telemetry
  conventions. The llama-server harness implements those with slot erasure,
  cache disabled, exact 512/128 performance pairs, and separate semantic
  gates.
- [`docs/peer-artifact-deployment.md`](docs/peer-artifact-deployment.md)
  records the authoritative EDR addresses and this mlx5 IPoIB device's prior
  connected-mode `EINVAL`. Its multi-link scheduler applies to immutable
  artifact copies, not single-flow inference RPC, so neither connected mode
  nor five-link striping was retried.
- The closest family guidance is
  [`Kimi-K2.md`](vendor/ktransformers/doc/en/Kimi-K2.md),
  [`Kimi-K2.5.md`](vendor/ktransformers/doc/en/Kimi-K2.5.md), and
  [`Kimi-K2-Thinking-Native.md`](vendor/ktransformers/doc/en/kt-kernel/Kimi-K2-Thinking-Native.md).
  They support dual-socket NUMA execution, CPU experts, and VRAM for dense/hot
  tensors, but their K2 runtimes and rates are not projected onto K3.
  [`experts-sched-Tutorial.md`](vendor/ktransformers/doc/en/kt-kernel/experts-sched-Tutorial.md)
  makes route-profiled hot-expert residency the strongest future code project;
  current K3 GGUFs fuse experts and cannot express it as a launch option.
- [`shared-host-weights.md`](vendor/ktransformers/doc/en/kt-kernel/shared-host-weights.md)
  informed the RSS/PSS and first-touch audit, but its shared mmap technique is
  inactive because accepted K3 weights are eager anonymous copies.
  [`DeepseekR1_V3_tutorial.md`](vendor/ktransformers/doc/en/DeepseekR1_V3_tutorial.md)
  and [`AMX.md`](vendor/ktransformers/doc/en/AMX.md) motivate a future
  K3 AMX/expert-kernel port; this llama.cpp path cannot accelerate
  Q2_K_XL/IQ2_XXS with its present AMX backend.

The remaining top-level, API, SFT, Rust, and patch-stack Markdown files provide
workflow or architecture context but no independent K3 tuning evidence.

### 2026-07-30: exact IQ2 placement design

- An exact 16-shard tensor-header audit accounts for 662.2269 GiB of tensor
  data: 604.2935 GiB of routed experts, 55.6091 GiB of block-local
  dense/attention/router tensors, and 2.3243 GiB of embedding/output tensors.
  The experts comprise 347.689 GiB IQ1_M, 217.916 GiB IQ2_XXS, and
  38.688 GiB IQ3_XXS. A MoE block carries 6.029--8.254 GiB of experts but only
  0.435--0.647 GiB of dense weights.
- This makes conventional whole-layer GPU offload the wrong IQ2 objective:
  keeping all routed experts on CPU and filling the 3090s with fully read dense
  tensors removes about ten times more per-token CPU traffic for each GiB of
  VRAM. In explicit mode, `--n-cpu-moe 93` is exact; `92` would accidentally
  put block 92's 6.029-GiB expert set on a GPU.
- The first-chance layout used this runtime's MoE-aware auto-fit on only
  dwagon's two GPUs: `--device CUDA0,CUDA1 --split-mode layer --fit on
  --fit-target 3072 --no-op-offload`. The fit code itself keeps every expert
  tensor on CPU, packs dense-only tensors back-to-front, and can use a partial
  layer. Unlike the Q2 RPC topology, there is no remote CPU device falsely
  advertising total RAM as free. The 3-GiB-per-GPU target is evaluated after
  model, context, and compute buffers rather than from raw weights alone.
- IQ2's physical microbatch is 512, not the launcher's original inherited
  4,096. The fixed benchmark prompt therefore still runs in one microbatch,
  while the smaller graph lets auto-fit retain more useful dense tensors in
  VRAM. It is expected to improve fit headroom and succeeded on the first
  attempt; no IQ2 U4096 performance A/B was claimed. This runtime does not
  propagate a failed fit status cleanly, so the load monitor will treat
  `failed to fit params` as an immediate failure instead of allowing a
  dangerous default-placement fall-through.
- `--no-op-offload` is deliberate for the short 512-token benchmark and for
  correctness conservatism. Default operation offload can stream many
  CPU-resident expert slices over PCIe once a route-count threshold is crossed;
  a 512-token prompt creates 8,192 expert routes per layer. Community Kimi-K2
  measurements on a comparable 578-GiB model found that this could collapse
  short-prompt processing while helping only at a full 4,096-token ubatch
  ([measurement thread](https://www.reddit.com/r/LocalLLaMA/comments/1mmmlqo)).
  Keeping host-weight operations on CPU also avoids making the newly added
  CUDA 13.1 path for the quant's IQ1/IQ2/IQ3 expert mix part of the first
  correctness gate.
- If auto-fit itself fails, the exact deterministic fallback is
  `--fit off --n-cpu-moe 93 --gpu-layers 63 --tensor-split 32,31
  --no-op-offload`. It leaves 624.546 GiB of raw tensors in host RAM, places
  19.012 GiB of dense weights on CUDA0 and 18.669 GiB on CUDA1, and already
  puts the output head on CUDA1. The high-confidence OOM retreat is 59 GPU
  layers with split `30,29` (17.718/17.586 GiB). If real telemetry leaves at
  least about 3.5 GiB free per card, the aggressive promotion is 67 layers
  with split `34,33` and ubatch 2,048 (20.095/19.963 GiB).
- A fwuff-GPU layout could place all dense blocks in 3 x 3090 VRAM and leave
  only 605.456 GiB on dwagon, but every remote dense block would still return
  to a local CPU expert operation. That creates repeated RPC latency and
  encounters the current K3 RPC-CUDA `SOFT_MAX` report. It is a conditional
  experiment, not the accepted-run starting point.

### 2026-07-30: IQ2 bring-up and accepted set

- The bold first-chance auto-fit profile succeeded without falling back:
  `CUDA0,CUDA1`, MoE-aware dense-only layer fit, 3,072 MiB target headroom per
  card, 112 physical CPU threads, NUMA interleave, 32K context, batch 4,096,
  microbatch 512, and operation offload disabled. It completed two temporary
  sizing passes and one durable eager residency pass, then became healthy in
  17m54s. The server stayed on one PID with zero restarts.
- Pre-request host residency was 656,456,068 KiB RSS (626.05 GiB):
  655,676,920 KiB `RssAnon`, 258,948 KiB `RssFile`, and 520,200 KiB
  `RssShmem`. The two GPUs used 20,762 and 20,504 MiB, leaving 3,362 and
  3,620 MiB free. This validates the 3-GiB auto-fit target and shows that the
  deterministic 63-layer fallback was unnecessary; the final telemetry
  summary supersedes this pre-request snapshot.
- The kernel logged two `NVRM: failed to allocate page table` messages during
  the temporary auto-fit sizing passes at 07:09 and 07:13. The planner
  released both trial arenas, the durable allocation completed, and the cards
  retained at least 3,274 MiB each in later live snapshots; there was no CUDA
  OOM or server restart. Three corrected socket-0 processor events (397--399)
  then arrived during final initialization, with the last one second before
  the benchmark began. No corrected or uncorrected machine check, GPU Xid,
  OOM kill, or swap
  use occurred inside either accepted IQ2 performance set.
- The model API identifies the 711,060,674,944-byte tensor payload as
  `IQ1_M - 1.75 bpw`. That aggregate label is not a contradiction: the exact
  tensor audit shows the routed-expert mixture includes 347.689 GiB IQ1_M,
  217.916 GiB IQ2_XXS, and 38.688 GiB IQ3_XXS weights.
- The first IQ2 campaign is a discarded diagnostic receipt. Its sacrificial
  warmup measured 6.888 prompt and 2.204 short-decode token/s, followed by
  three coherence-passing pairs at 7.489/1.608, 7.496/1.690, and
  7.534/1.592 prompt/decode token/s. Run 4 then exposed a real
  semantic-protocol issue rather than a runtime fault: it returned the correct
  function and both assertions, but put `FINAL=CODE` inside a Markdown fence
  and emitted the closing fence as the last line. The strict validator
  rejected it and the
  campaign stopped at three accepted pairs; that receipt is retained as
  `benchmark.jsonl` and does not contribute to the final five.
- The code-case prompt now explicitly requires plain text and forbids fences;
  its validator did not change. Python bytecode compilation, Ruff, and
  Basedpyright passed before a clean full replacement campaign began on the
  same warmed model PID as `benchmark-final.jsonl`. Explicit slot erasure and
  a new sacrificial warmup at 7.506/2.168 prompt/short-decode token/s reset the
  request protocol without reloading 662 GiB of weights. Q2, the discarded
  IQ2 diagnostic, and final IQ2 all use exact 512-token performance content
  SHA-256 `251cc4cc6db802c460d88db52530df5c69264f0cfca7506b3f48efe3363d424a`
  and rendered-prompt SHA-256
  `9b4f1362ee56436565dd9afe95da7d3218e9dc8b620836f3c26be3f866d77672`;
  only the unmeasured semantic case-4 instruction changed.

## Benchmark protocol

The final protocol was frozen after the first successful end-to-end request,
not after speculative tuning. The campaign receipts plus this journal record:

- quant and exact model/runtime revision;
- topology, device/tensor split, thread and batch settings;
- prompt and generated token counts;
- prompt-processing tokens/s and decode tokens/s;
- time to first token, end-to-end latency, and tokens per joule where telemetry
  permits;
- peak host RAM, per-GPU VRAM/utilization/power, CPU utilization, and network
  traffic;
- the generated response or a concise deterministic coherence verdict.

### `UD-Q2_K_XL` accepted runs

| Run | Configuration | Prompt tok/s | Decode tok/s | TTFT | Coherence | Important statistics |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | N16 tuned; arithmetic; 512 in/128 out | 4.900 | 0.457 | 104.487 s | PASS | Cache 0; 384.580 s elapsed |
| 2 | N16 tuned; geography; 512 in/128 out | 5.273 | 0.471 | 97.110 s | PASS | Cache 0; 369.003 s elapsed |
| 3 | N16 tuned; JSON transform; 512 in/128 out | 4.963 | 0.372 | 103.163 s | PASS | Cache 0; 447.690 s elapsed |
| 4 | N16 tuned; Python code; 512 in/128 out | 5.096 | 0.292 | 100.485 s | PASS | Cache 0; 538.337 s elapsed |
| 5 | N16 tuned; ordering; 512 in/128 out | 5.143 | 0.272 | 99.562 s | PASS | Cache 0; 570.210 s elapsed |

Q2 completed with one unchanged server PID and exact counters of five accepted
semantic runs, five accepted performance runs, and five accepted pairs.
Prompt throughput is 5.075 token/s mean, 5.096 median, and 5.071 weighted
(4.900--5.273, 2.60% CV). Decode is 0.373 token/s mean, 0.372 median, and
0.355 weighted (0.272--0.471, 21.91% CV). Mean/median TTFT are
100.961/100.485 seconds. The wide decode distribution is sequence/run
dependent, plausibly reflecting expert routing and thread efficiency; no slow
valid run was discarded and routing alone was not proved causal.

The one-Hz resource summary covers all five dwagon windows and four of five
fwuff windows (9/10 source-windows, 90%). Fwuff run 3 is explicitly
unavailable across its 833.049-second `ENOSPC` gap; no value is interpolated
or imputed and that host is excluded from run 3 cluster totals. Across
2,309.818 performance-window seconds, covered GPU-board energy was
528.427 kJ: 264.105/227.581 kJ on dwagon's two active cards and 36.741 kJ on
fwuff's intentionally idle card over its four covered windows. Whole-system
tokens/J cannot be claimed because neither host exposes RAPL here. Dwagon's
llama process averaged 42.29 CPU-core equivalents, peaked at 672.53 GiB RSS,
and retained at least 58.02 GiB `MemAvailable`; fwuff's RPC process averaged
35.21 cores, peaked at 96.05 GiB RSS, and retained at least 141.00 GiB.
Process major faults were zero on both hosts in every covered run. Covered EDR
traffic averaged only about 0.0195 Gbit/s, reinforcing that the final Q2
decode path was memory/compute-bound rather than fabric-bound. The immutable
partial-coverage receipt is
`telemetry-summary.json` (SHA-256
`f0c1b9fd372c224d070686fe6b62235b40ef190438bea83e41856975fae9d305`).

### `UD-IQ2_XXS` accepted runs

| Run | Configuration | Prompt tok/s | Decode tok/s | TTFT | Coherence | Important statistics |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | Auto-fit U512; arithmetic; 512 in/128 out | 7.398 | 1.734 | 69.211 s | PASS | Cache 0; 143.027 s elapsed |
| 2 | Auto-fit U512; geography; 512 in/128 out | 7.282 | 1.577 | 70.311 s | PASS | Cache 0; 151.456 s elapsed |
| 3 | Auto-fit U512; JSON transform; 512 in/128 out | 7.212 | 1.602 | 70.995 s | PASS | Cache 0; 150.911 s elapsed |
| 4 | Auto-fit U512; Python code; 512 in/128 out | 7.266 | 1.522 | 70.467 s | PASS | Cache 0; 154.545 s elapsed |
| 5 | Auto-fit U512; ordering; 512 in/128 out | 7.349 | 1.675 | 69.673 s | PASS | Cache 0; 146.104 s elapsed |

The clean IQ2 replacement receipt completed on the unchanged model PID with
exact counters of five accepted semantic runs, five accepted performance
runs, and five accepted pairs. Prompt throughput is 7.302 token/s mean, 7.282
median, and 7.301 weighted (7.212--7.398, 0.89% CV). Decode is 1.622 token/s
mean, 1.602 median, and 1.619 weighted (1.522--1.734, 4.58% CV). Mean/median
TTFT are 70.132/70.311 seconds, and mean/median end-to-end latency are
149.209/150.911 seconds. No accepted slow run was discarded.

Strict one-Hz telemetry fully brackets all five IQ2 performance windows
(746.043 seconds). The llama process averaged 106.05 CPU-core equivalents;
host CPU utilization averaged 47.89%. RSS peaked at 657,311,712 KiB
(626.861 GiB), minimum `MemAvailable` was 109,720,600 KiB (104.638 GiB), and
post-set `numastat` reported 276,195.81/365,710.14 MiB on nodes 0/1. Process
major faults and swap were zero; six interpolated host-wide major faults did
not occur in the process. `ibs5` moved zero bytes inside the accepted windows.

CUDA0/CUDA1 held 20,850/20,764 MiB. Their performance-window energy was
88.926/71.281 kJ, average board power 119.20/95.55 W, peak power
139.27/123.93 W, and peak temperature 62.29/45.00 C. One-Hz average GPU
utilization was only 3.05/1.74% despite 32/92% sampled peaks, so energy and
peak utilization are more informative than the alias-prone averages. Total
GPU-board energy was 160.207 kJ, or 0.003995 generated token per GPU joule
(250.32 GPU J/generated token) over complete prompt-plus-decode windows.
RAPL is unavailable, so this is not whole-system efficiency.

Relative to Q2, IQ2 is 1.439x faster on mean prompt processing, 4.352x faster
on mean decode (4.565x weighted), cuts mean TTFT by 30.54%, and cuts mean
end-to-end time by 3.096x. Its complete GPU-board token/J is at least 3.30x
Q2's optimistic covered-source upper bound; Q2's missing fwuff run-3 energy
can only lower that quant's true value. The throughput/capacity gain costs
quality: Unsloth's reported KLD/perplexity/top-1 agreement move from
0.1779/1.7359/90.390% on Q2 to 0.3784/2.1266/84.127% on IQ2.

### Final integrity and restoration

- Independent summary and raw-JSONL gates pass for both quantizations. Final
  IQ2 is exactly `complete`, 5 semantic/5 performance/5 paired, and every
  measured request is exactly 512 input/128 output tokens. Bash syntax and
  Python bytecode checks pass; Ruff and Basedpyright report zero findings; the
  focused telemetry regression test passes. `shellcheck` is unavailable.
- The final IQ2 campaign ran from 11:42:06.359 to 12:05:05.471 UTC. Corrected
  socket-0 processor event 406 arrived three seconds before it began; events
  407 and 408 arrived 39 and 61 seconds after it completed. Events 400--405
  likewise fell between the discarded diagnostic and replacement campaign.
  Thus no accepted IQ2 request overlaps a machine check. All were
  firmware-reported corrected general-processor events with no required
  action. There was no uncorrected event, GPU Xid, OOM kill, service restart,
  or swap. Every exposed Intel EDAC memory-controller/DIMM CE and UE counter
  remains zero, so the kernel data does not implicate a DIMM. The recurring
  APEI sequence remains a real socket-0 RAS issue for a later BMC/firmware and
  CPU stability investigation; this host has no `ras-mc-ctl`, `edac-util`, or
  `ipmitool` installed to identify a bank.
- IQ2 model and telemetry services stopped cleanly with exit status zero and
  zero restarts. Dwagon released the model to 1 MiB per GPU and about
  733 GiB `MemAvailable`. Its 224 CPU policies were transactionally restored
  to their original `powersave/balance_performance` state; the receipt has
  `status=restored`, verified application/restoration, and no failures.
  `/var/lib/exo/benchmarks/kimi-k3-q2-20260730-v1/cpu-policy-dwagon.json`
  has file SHA-256
  `5ab838415d1a66a519d2fdb68fe92b7ee9d347b5840f7e23bb33a89a117dbec6`
  and internal canonical receipt hash
  `7fb0ce692d0d88edbe851c1b51cebf978dcf01e2f55c0fd1e1fa52d787ce941f`.
- Fwuff's 120 policies were also restored exactly to
  `powersave/balance_performance`: comparison against the saved transaction
  journal found 120 policies and zero mismatches. Fwuff's full unrelated root
  LV prevented the holder from publishing its normal receipt, so the
  59,891-byte pre-stop recovery journal was preserved on `/mnt/sanic` before
  shutdown (SHA-256
  `81e82f70dea0df3cd0ba58e2835301cf525d58da0c912e7964cff0bf1ede526d`).
  Restoration occurred before receipt publication, but the journal confirms
  the later write failed with `ENOSPC` and the holder exited status 2. This is
  an evidence-publication failure, not a policy-restoration failure.
- The two intentionally stopped dwagon containers,
  `qwen3-tts-nano-gpu0` and `qwen3-tts-nano-gpu1`, were restarted and are
  healthy with zero restarts, using 22,049/21,841 MiB. Fwuff's benchmark RPC
  and GPU remain inactive/idle.
- At baseline-restoration time, direct free space on `/mnt/sanic` was
  2,451,272,343,552 bytes (about 2.23 TiB), while fwuff's root was at 100%
  with zero bytes available. No unrelated data was deleted, so the clean
  incorporated RPC patch branch was then only partially built on fwuff. The
  later relocation/completed build supersedes that historical state.
- The useful infrastructure intentionally remains: fwuff OpenSM is active,
  EDR IPoIB is `10.44.0.1/30` (`ibs5`) to `10.44.0.2/30` (`ibs2`), the
  dwagon view is a read-only NFSv3 `nconnect=8` mount at `/mnt/sanic-edr`, and
  fwuff retains the exact read-only EDR export. Network rollback receipts are
  `/var/lib/exo/peer-artifact-deployment/ipoib-{dwagon,fwuff}-kimik3-20260730.json`;
  fwuff's pre-change export is `/etc/exports.exo-20260730-kimik3-edr`.

### Immutable result receipts

Q2 artifacts are under
`/var/lib/exo/benchmarks/kimi-k3-q2-20260730-v1/tuned-n16`:

- `benchmark.jsonl`:
  `d349f9f77f2de4e27568ba55cda71201d56eef67717b882fd98d42948b00e61a`
- `benchmark.summary.json`:
  `e1857685a6b1f6bc3f1d256a83bc8424b909e974934dc6b0af196e5f702f537e`
- `telemetry-dwagon.jsonl`:
  `6c51040ab36b82fbd55170f94539f964dd6581919ceadbb933ff0d25857f79cb`
- `telemetry-fwuff-combined.jsonl`:
  `d7f3e6b0a44e78ae032c87e160de739168567b30e71868e9c448b20b84be4327`
- `telemetry-summary.json`:
  `f0c1b9fd372c224d070686fe6b62235b40ef190438bea83e41856975fae9d305`
- `server.log`:
  `07e2b95d6799df81542c2013e29ca5df324d8c8c7ed4411684293fd071a0efc8`

IQ2 artifacts are under
`/var/lib/exo/benchmarks/kimi-k3-iq2-20260730-v1/auto-fit-u512`:

- `benchmark-final.jsonl`:
  `de5f3bc4f92b700955de3c5f2881b62204683a1a23616b529432ea7453dff294`
- `benchmark-final.summary.json`:
  `4719ff5910abbd6f320c49bb2d1ee21ae213a92d87978a8af6ee4cb283c48644`
- `telemetry-dwagon.jsonl`:
  `c9944c7bf3ad2524f021ae095416ea11689a2931ad81b9e85ce5c05b28cba97c`
- `telemetry-summary-final.json`:
  `757a02c397835fb6eaec19a3bf06219cd5e31650caf70032eac671b0fe7338ab`
- `server.log`:
  `a9fbfbcb97ed24876d2db2643b2d679bc631573784c849a9a6ae44a9e19a59bf`

The final IQ2 benchmark harness hash is
`903de2f8348dca8534c1d647a45c032ab12f73705bad8d717087d2e51015d478`;
Q2 predates its semantic-only no-fence wording fix, so that hash is not
retroactively attributed to Q2. Q2/IQ2 launcher hashes are
`04ed354dc846af47a373ecdb92b6c572f6edc7fc9eb6edf0554a331e09ce34c4` and
`8db6c4ae804173aa3c6c0d318829b6f2f0bda29fc6fd9a6d2409e2ae5ac0ed17`.
The telemetry sampler/summarizer hashes are
`3fe51507a06d06ea78f1ff3f8d53dedbd7374f2350b7ffe79fe76c4eca6af43b` and
`45b71ebed12639a89c1e6e04336ea46baa88d71b1a2e11cd536b4ba1f4d82fd8`.
Model and runtime revisions remain the immutable pins recorded above.

## RPC `tensor_memset` follow-up

### Root-space audit, relocation, and build completion

- Fwuff's root LV was already a 3.7-TiB filesystem with essentially all ext4
  free and reserved blocks consumed. The task-created patched worktree was
  only 1.1 GiB, including a 900-MiB partial `build-kimik3-text` directory; it
  did not explain the multi-terabyte exhaustion. A read-only audit attributed
  about 2.0 TiB to `/home` and 1.2 TiB to `/var`. The largest `/var`
  components were about 576 GiB of Docker data, 474 GiB of logs, and 79 GiB
  of PostgreSQL data. Deleted-but-open files were negligible.
- Only the generated build directory attributable to this task was moved:
  `/var/lib/exo/sources/llama.cpp-kimik3-text-47c5bbdf-rpc-memset/build-kimik3-text`
  is now a symlink to
  `/mnt/sanic/builds/fwuff/llama.cpp-kimik3-text-47c5bbdf-rpc-memset-build`.
  The completed build occupies about 1.2 GiB on `/mnt/sanic`; source and
  unrelated root-resident data were not moved or deleted.
- The relocated build resumed from 145/430 objects and completed at 430/430.
  Its RPC server reports protocol v5.0.0, and the paired dwagon server reports
  runtime version 10163 at commit
  `a30437bc3a2a661d1e9aad71b1160d9ad9bbfec1`. Both builds have CUDA
  architecture 86 and RPC enabled. The stable patch ID remains the upstream
  `tensor_memset` patch recorded earlier.
- The full root filesystem had also prevented sshd's `xauth` helper from
  creating `/root/.Xauthority-n`. After space was moved, a synthetic trusted
  X11 handshake verified a valid remote display and cookie. The user
  subsequently confirmed X forwarding works. No sshd configuration workaround
  was installed.
- A real launcher compatibility issue was fixed: version 10163 prints a
  nine-character Git abbreviation, while the launcher had required a closing
  parenthesis immediately after eight characters. The gate now accepts any
  longer lowercase hexadecimal abbreviation only when it begins with the
  pinned full commit's first eight characters.

### Patched distributed load

- Both hosts were transactionally held at
  `performance/performance`. The exact accepted Q2 topology was reused:
  `CUDA0,RPC0,CUDA1`, 16 offloaded layers, tensor split `2,11,3`,
  512 microbatch, `--no-op-offload`, 32K context, and a CPU-only fwuff RPC
  endpoint on `10.44.0.2:50052`.
- The patched server reached healthy state in 25m48s with zero service
  restarts. Final GPU allocations exactly matched the baseline at
  18,286/21,954 MiB, and fwuff stabilized near 95.5 GiB RPC RSS. No RPC
  `tensor_memset`, protocol, assertion, CUDA application, OOM, or swap error
  occurred. Normal RPC verbosity does not log individual command dispatch, so
  this proves the patched client/server pair is stable but does not prove a
  nonzero `RPC_CMD_MEMSET_TENSOR` call count. A forced tensor override would
  exercise that exact command more directly, but the accepted N16 placement
  no longer needs an override and this follow-up did not introduce one.
- The kernel emitted one `NVRM: failed to allocate page table` warning during
  the load, but graph allocation later completed with the exact baseline VRAM
  footprint and the service remained healthy. Corrected socket-0 APEI
  processor events 439--443 occurred during load or before the measured
  campaign. Event 444 arrived 55 seconds after the accepted performance
  request. No corrected or uncorrected hardware event, Xid, OOM, restart, or
  process major fault overlapped the accepted request.
- The load trace exposed an important architectural inefficiency. Dwagon is
  the only GGUF reader, while fwuff's RPC service is a passive tensor-buffer
  endpoint. Fwuff-assigned weights therefore travel
  `fwuff /mnt/sanic -> NFS/EDR -> dwagon -> RPC/EDR -> fwuff RAM`.
  The current patch supplies a missing remote tensor operation but does not
  make RPC file-aware or parallelize model loading. Eliminating this roughly
  95-GiB round trip requires a new server-side GGUF-read operation or a
  distributed loader that gives fwuff shard/path plus tensor-offset metadata.
  Such a change should materially reduce cold-load time but should not be
  expected to change steady-state token throughput.

### One accepted representative A/B

The user requested one representative result rather than another five-run
campaign. The strict harness completed one sacrificial warmup, one coherent
arithmetic semantic gate, and its paired cache-cold performance request. A
watcher stopped the harness immediately after the accepted
`performance_result` was atomically written. Pair 2's semantic HTTP request
had just opened but produced no event and is not counted. The server returned
healthy and idle afterward.

| Metric | Patched representative | Baseline five-run mean | Baseline range | Difference from mean |
| --- | ---: | ---: | ---: | ---: |
| Prompt processing | 4.744242 tok/s | 5.074897 tok/s | 4.900342--5.272632 | -6.52% |
| Decode | 0.368137 tok/s | 0.372718 tok/s | 0.271966--0.470774 | -1.23% |
| TTFT | 107.924 s | 100.961 s | 97.110--104.487 | +6.90% |
| End-to-end | 455.621 s | 461.964 s | 369.003--570.210 | -1.37% |

The result is exactly 512 rendered input tokens and 128 output tokens, with
zero cached tokens and a length finish. The independent arithmetic gate
completed coherently with 124 input and 138 output tokens. Decode is within
the baseline distribution and only 0.91% below its median; prompt processing
is below every baseline sample. One observation is not enough to estimate a
new distribution, but there is no evidence that `tensor_memset` improves
steady-state speed. Its demonstrated value here is correctness and enabling
otherwise-invalid RPC tensor initialization/override paths.

The one-run telemetry join has strict, fully bracketed coverage from both
hosts:

- integrated GPU-board energy: 105,867.577 J, or 827.09 GPU-board
  J/generated output token across prompt plus decode; RAPL is unavailable, so
  this is not whole-system energy;
- average process CPU use: 42.63 core-equivalents on dwagon and 34.75 on
  fwuff; host CPU utilization was 19.43% and 31.00%;
- maximum process RSS: 705,161,868 KiB on dwagon and 100,627,292 KiB on
  fwuff; minimum `MemAvailable` was 65,131,852 and 139,342,792 KiB;
- zero process major faults and zero swap on both hosts;
- about 0.0198 Gb/s total over the EDR IPoIB interface during steady
  inference, confirming that compute rather than fabric bandwidth remains the
  decode bottleneck.

The then-current telemetry summarizer admitted this one historical window via
an explicit positive `--expected-runs` count while retaining five as its
default. The raw tensor-`memset` benchmark JSONL is deliberately truncated
after `performance_result`: it predates `configuration.run_count` and has no
terminal `campaign_completed`. Today's hardened summarizer intentionally does
not re-admit that truncated receipt. It requires the requested run count to
match both the campaign configuration and terminal accepted-pair count, which
prevents an interrupted five-run campaign from being reclassified as one
valid run. The later file-aware receipt is the canonical one-run schema with
`run_count=1` and terminal `accepted_pairs=1`.

### Follow-up receipts and restoration

Artifacts are under
`/var/lib/exo/benchmarks/kimi-k3-q2-rpc-memset-20260730-v1`:

- `benchmark.jsonl`:
  `ad07cb754caa084ea9b1a951516a36a2fb1755bbd1598a6c06a003391af223de`
- `telemetry-dwagon.jsonl`:
  `6769df5e9178b262a5d35ceb7e53af09efd4c9f664ad06630ec7165e6a195220`
- `telemetry-fwuff.jsonl`:
  `71ec974e7b2eec7a418ad4035be1dd9af0a8f2ba006b70250271c3909cf0e219`
- `telemetry-summary.json`:
  `85eddb55338f088baa1717780f35077cd4a3721bb06aa55c86ca4bbecdad669b`
- `server.log`:
  `99730f6ddd9a796696fb417063fb3947d90160e2a6a47132e3369d7af7b01abf`
- `cpu-policy-dwagon.json`:
  `1ef21e700d8d280c822ba6bcd4e831055ee166c03ffa578415f564158ee6eed5`
- fwuff `rpc-server.log`:
  `0f123033800b52eb2abc575ec7d0344861cf9883743b32060bc30655f8921a94`
- fwuff `cpu-policy-fwuff.json`:
  `da7ce6e5300016bffda8e570be41e1102ab8d8f40f9c247df81d2bc743fb16a7`

The contemporaneous Q2 launcher and telemetry summarizer hashes for this
tensor-`memset` receipt were
`11b9a11aad92bed3b29611700dcf67c6d5c207ca078010731794616d40124999`
and
`ce0b265128ddebdc655f58b30ce53df1487d1547f668ce3780541e3df0cac0a9`.
Both benchmark services and both samplers are inactive. CPU policy receipts
have `status=restored` and no failure; all 224 dwagon plus 120 fwuff policies
are back at `powersave/balance_performance`. The two TTS GPU containers are
healthy again.

## File-aware loading and expansion follow-up

### Complete Q2 stage and evidence-based source selection

- `UD-Q2_K_XL` is now atomically staged at
  `/mnt/llm-models/Kimi-K3-GGUF/UD-Q2_K_XL`: 19 shards and
  861,277,858,912 logical bytes. Follow-up service
  `exo-kimi-k3-verify-q2-content-20260730.service` (invocation
  `ad9459280af7463bac743abb8e4486b2`) checked every published shard against
  the pinned Hugging Face LFS SHA-256 in 35m22.844s wall time and upgraded the
  receipt in place at 22:36:04Z. The schema-2 receipt records exact names,
  exact sizes, non-sparse allocation, per-shard content hashes, revision
  `3d4b61ab4b6789d401191c476cbb4567246db8f5`, and quant-manifest hash
  `34273843bf7edc60eb9f74fcaa0265e80e2ac30be172b653387a090dde80a387`.
  Its SHA-256 is
  `f99872a96e71ef83597e62ef126e13d86387697b6d899ac90a63dfa64cb0328d`.
  The model filesystem has 3,294,314,041,344 bytes free with both quantizations
  published.
- The post-Q2 capacity gate for `UD-IQ2_XXS` passed: 16 shards require
  711,067,773,664 bytes, the conservative reserve is 88,286,646,550 bytes,
  and 4,005,382,529,024 bytes were available. The first four-worker service,
  `exo-kimi-k3-stage-iq2-20260730.service` (invocation
  `98f22be0a96a458682f8b1dd777cdd0f`), filled the complete partial from Sanic
  over EDR but did not publish it: after 32m05s, its long-running shell parsed
  a newly edited portion of the live script and failed with a syntax error.
  No size-only snapshot was represented as complete. The stager now re-execs
  from an already-open, unlinked copy of its own source before parsing
  arguments. Repository edits therefore cannot alter an active transaction,
  and no temporary script remains after the descriptor closes. A regression
  test truncates the original inode while the staged copy is blocked in source
  selection and proves that the immutable copy still emits the complete plan.
- Recovery service `exo-kimi-k3-verify-iq2-20260730.service` (invocation
  `ed8f4d6a95564d62a37ff16c1f7754db`) first proved all pinned names, lengths,
  and non-sparse allocations, skipped the redundant second network copy, and
  then checked all 16 destination shards against the pinned Hugging Face LFS
  SHA-256 values. It atomically published `UD-IQ2_XXS` at 17:59:13 after
  22m10.933s. The schema-2 receipt has SHA-256
  `02096e0704f427cd630d7ee399928a71b0a54bd709e93cfd07f223b157c1e874`;
  it records 16 files, 711,067,773,664 bytes, revision
  `3d4b61ab4b6789d401191c476cbb4567246db8f5`, and quant-manifest hash
  `7b166810d1ee0778067e20f854347bc52786bcdacb4eb2079dac6806e2d9a689`.
  The model RAID has 3,294,314,041,344 bytes free after publication.
- Fwuff's root-full incident was not caused by the new K3 source, build, or
  kernel threads. Their substantial fwuff artifacts are under
  `/mnt/sanic/exo`; the down-kernel SM86 payload was only 74 MB at
  `/mnt/sanic/exo/kimi-k3-kernels/down-mmid-baseline-sm86`, and the matching
  llama.cpp source/build was already under `/mnt/sanic/exo/sources`. The
  3.7-TB root LV was at 100% while the directly attached Sanic XFS array still
  had about 2.2 TB free. Read-only accounting attributed about 1.960 TB to
  unrelated `/home` data, 1.219 TB to `/var`, 199 GB to `/usr`, 143 GB to
  `/root`, and 83 GB to `/ai`; no user home or live Docker volume was moved.
- The immediately recoverable fault was two stale OpenSM instances for absent
  port GUIDs `0xe41d2d03004d32e1` and `0xe41d2d03004d32e2`. Their logs were
  281,330,700,288 and 284,849,410,048 bytes and consisted of a runaway
  `umad_receiver: ERR 5404` I/O-error loop. `ibstat` showed only the separate
  live `mlx5_0` GUID `0x248a070300a32154`, active at 100 Gb/s. I stopped only
  the two dead-port units, discarded the 527.3-GiB repetitive logs and
  interrupted archival copies after the user confirmed they had no credible
  retention value, and left the live subnet manager running. Root recovered
  to 374 GiB free (90% used); Sanic returned to about 2.2 TB free (61% used).
  The two known stale `/var/log/opensm.*` names are now symlinks to empty files
  under `/mnt/sanic/exo/system-logs/opensm`, so an accidental restart cannot
  refill root. All future substantial fwuff-side K3 source, build, model,
  benchmark, and temporary-transfer storage belongs under `/mnt/sanic/exo`;
  compatibility paths should be symlinks rather than root-backed copies.
- I initially preferred the local copy without measuring the two actual paths.
  That was a mistake. I then made a second incorrect inference from RSS growth
  that the HDD RAID was reading slowly and aborted the first file-aware load.
  `/proc/<pid>/io` disproved it: hundreds of GiB of RSS had been materialized
  while only about 9 MiB had been read. The aborted artifact directory is
  retained as a diagnostic rather than represented as a storage result.
- The hardware explains why locality is not enough. Dwagon's
  `/mnt/llm-models` is XFS on a ten-disk 7,200-RPM-class RAID0 with 1-MiB
  chunks. Fwuff's `/mnt/sanic` is XFS on a three-NVMe RAID0 and is exported to
  dwagon over the dedicated EDR IPoIB link as read-only NFSv3 with 1-MiB
  `rsize`, `nconnect=8`, and endpoint `10.44.0.2`.
- Matched read-only direct-I/O measurements used the same Q2 shard and offset,
  one synchronous reader, queue depth one, and no concurrent dwagon RAID
  transfer. They are source-path measurements, not inference benchmarks:

| Read shape | Dwagon HDD RAID | Sanic over EDR | EDR advantage |
| --- | ---: | ---: | ---: |
| 1 MiB requests, 8 GiB | 333 MB/s | 1,105 MB/s | 3.32x |
| 4 MiB requests, 16 GiB | 388.6 MB/s | 3,243.3 MB/s | 8.35x |
| 64 MiB requests, 16 GiB | 405.4 MB/s | 7,392.4 MB/s | 18.23x |

  The two larger EDR results benefited from fwuff's warm server cache, so the
  conservative figure is the live full-model phase: dwagon read
  759,028,195,328 bytes over NFS at roughly 1.3 GB/s while fwuff populated its
  own approximately 100-GB assignment locally. Even that cold-ish result
  decisively beats the local direct-read tests.
- The launcher therefore defaults dwagon to `/mnt/sanic-edr`, while fwuff maps
  assigned tensors directly from `/mnt/sanic`; the verified local snapshot is
  a fallback. The staging selector now gives this measured EDR source priority
  over generic local block storage on this cluster. The conclusion is specific
  to these mounts and should be remeasured after changing arrays or fabric.

### File-aware RPC loader

- Historical protocol-v6 context: the binaries used by the full-size
  representative result below were built from
  `d29a524eeaf39155825d6f0ef373075fe585cb12`, based on the already-tested
  `tensor_memset` ancestor `a30437bc3a2a661d1e9aad71b1160d9ad9bbfec1`.
  Protocol v6 patch `0000` added client `--rpc-tensor-source` and RPC-server
  `--tensor-source-root`: the client sent shard-relative path, file offset,
  and tensor size, and the server read the bytes into its backend buffer on a
  background worker. Dwagon's historical source/build is under
  `/mnt/llm-models/exo/sources`; fwuff's matching source/build is under
  `/mnt/sanic/exo/sources`, not fwuff's root filesystem.
- That historical v6 implementation was validated by a tiny mixed CPU/RPC
  model and the live Q2 load. The integration proved direct server-local reads,
  the barrier, and generated output; the full load raised fwuff's RPC RSS to
  about 95.5 GiB from local `/mnt/sanic` reads instead of sending those bytes
  `fwuff -> NFS -> dwagon -> RPC -> fwuff`. Commit `d29a524e` ordered
  RPC-bearing tensor contexts ahead of ordinary contexts but did not overlap
  RPC and local reads inside a mixed context.
- Historical follow-up `0001` added the tested two-pass prequeue: every
  admitted RPC tensor is enqueued before client-side reads, accepted tensors
  are skipped in the normal pass, and progress, validation,
  reject-after-accept protection, and the final barrier remain intact. Its
  mixed integration queued all 56 remote tensors before local reads, loaded
  12.56 MiB by RPC plus 4.94 MiB locally, and produced the expected completion.
  Its stable patch ID is
  `fd252af1865a0524b43c0efabc40feef2c696fa1`; its raw SHA-256 is
  `9419ac4c6d78203b08c4dd58053cec3d0c0017599fe13bd7eb7e43c0d8d426ad`.
- The implemented current hardening is
  `0005-rpc-attest-same-backing-tensor-source-files.patch`, which upgrades the
  wire protocol to major version 7. In the default
  `--rpc-tensor-source-mode same-backing-file`, the client attests each
  already-open GGUF shard by the exact `{inode, size, mtime_ns}` identity and
  the server requires its opened file to match. The server accepts only a
  direct-child `.gguf` name, opens the configured root directory and then the
  child with race-resistant `openat(..., O_NOFOLLOW)`, verifies a regular
  file with `fstat`, and keeps that descriptor open. Reads use `pread` from
  the held descriptor, and the server rechecks that its identity is unchanged
  before reuse and at synchronization.
- Held shard descriptors are bounded per client by
  `--tensor-source-max-files`, default 32; reaching the cap rejects another
  shard rather than silently reopening an unpinned path. Synchronization now
  returns exact completed tensor counts and bytes for the requested backend
  buffer. The loader tracks its expected count and byte total independently
  for every RPC buffer and rejects any missing, extra, cross-buffer, changed,
  or failed receipt.
- Strict same-backing-file mode is fail closed: a path, identity, queue, read,
  or receipt mismatch aborts loading. Compatibility fallback is available
  only through the explicit `--rpc-tensor-source-mode fallback`; it permits
  the historical client-read path while retaining reject-after-accept safety.
  The launcher defaults to strict mode and passes the 32-file server cap.
- A live read-only identity comparison of the canonical dwagon Sanic-over-EDR
  view with fwuff's local Sanic root matched `{inode, size, mtime_ns}` for all
  19 Q2 shards. The complete filename-ordered `0001`--`0005` follow-on stack
  on `0000` also completed its CPU/RPC build and RPC tensor-source integration,
  and all 24 Q2 topology-launcher tests passed. Patch `0005` has raw SHA-256
  `38f6e350a2b58523df5fdab0f6e89231229139bffaa1b52fbe27bb698566d6c2`
  and stable patch ID
  `7a36335bec0b22bc83dc2d7dc38a55a2c700f965`.
- Reproducible patches remain under `scripts/patches/kimi-k3`. The current
  `d29a524e` benchmark binaries on dwagon and fwuff contain historical
  protocol v6/`0000`, not protocol v7/`0005`; the representative result below
  is therefore not evidence for strict attestation. Protocol v7 changes the
  RPC wire ABI, so both endpoints must be rebuilt from the same complete patch
  stack before the next live load.
- The RPC device's advertised CPU MiB is physical capacity, not free memory.
  It is no longer used as a headroom gate. Every active RPC topology now
  queries fwuff over SSH and fails closed on the smaller of real
  `MemAvailable` and every finite cgroup-v2
  `memory.max - memory.current`
  ancestor. The live check reported 235,378 MiB effective/MemAvailable with
  no finite cgroup limit against the 200,000-MiB CPU-RPC requirement.
  Non-CPU RPC devices separately gate the parsed free-device-memory field.
  Exact device-name parsing also prevents a device description from
  masquerading as a missing `CUDA0`/`RPC0` enumeration.

### The real cold-start bottleneck and next default

- Source inspection located the long pre-read phase. CPU-assigned weights
  choose the first CUDA device's host buffer in `llama-model.cpp`, and every
  backend context is allocated serially before `load_all_data()` begins. The
  approximately 700-GiB CPU context consequently blocks in `cudaMallocHost()`.
  CPU `init_tensor` is null and this path does not clear the slab: the RSS
  sweep is page materialization/pinning, not RPC `tensor_memset` and not a
  storage read.
- `--no-host` is now the bold default for this topology. With
  `--no-op-offload`, no explicit steady-state use for whole-model CUDA pinning
  was identified, but steady-state neutrality remains unproven: the later
  representative regressed under a confounded `--no-host` plus NUMA-policy
  change. Its demonstrated benefit is avoiding the long serial
  `cudaMallocHost()` initialization.
  The flag affects model-weight selection only: CUDA-pinned upload staging and
  CPU/GPU boundary activation buffers remain available. Do not use the global
  `GGML_CUDA_NO_PINNED=1`, which also removes those useful small transfer
  buffers.
- Header inspection of all 19 shards found 698.465 GiB `IQ2_XS`,
  45.722 GiB `IQ3_XXS`, 55.503 GiB `Q8_0`, and 2.431 GiB `F32`. The dominant
  744.187-GiB routed-expert bulk has no x86 repack/AMX path and therefore
  simply moves from pinned to ordinary resident DRAM with the same compute
  kernels. Some eligible dense Q8 weights may select AMX and repack once;
  that is an acceptable one-time cost with potential prompt/decode benefit.
  `--no-host --no-repack` remains the strict pageable-versus-pinned control.
- A hybrid pinning threshold is not useful while operation offload is
  disabled. If large-prompt operation offload is re-enabled later, use a
  bounded 8--32-GiB profile-guided pool for frequently transferred dense
  weights and explicitly exclude the fused IQ2/IQ3 routed-expert tensors.

### Planned 3090/5090/2080 Ti topologies

`scripts/run_kimi_k3_q2.sh` now validates arbitrary aligned device/split
lists, per-GPU free-memory minima, exact required CUDA architectures, optional
RPC, and the following presets. Future-hardware presets fail closed when
architecture evidence is absent; `legacy` retains a compatibility warning:

| Preset | Devices | Offloaded layers | Split | Purpose |
| --- | --- | ---: | --- | --- |
| `legacy` | 3090, fwuff CPU RPC, 3090 | 16 | `2,11,3` | Current measured topology |
| `future-4gpu-rpc2` | fwuff RPC, 3 x 3090, 5090 | 12 | `2,2,2,2,4` | Safe first expanded Q2 profile |
| `future-4gpu-local` | 3 x 3090, 5090 | 10 | `2,2,2,4` | Bold all-local profile; requires nearly all 768 GiB free |
| `future-5gpu-local-2080` | 22-GB 2080 Ti, 3 x 3090, 5090 | 12 | `2,2,2,2,4` | Removes RPC if PCIe/host headroom permits |

- The 5090 is deliberately last so it receives output-side layers; the safe
  first expanded profile leaves approximately 703.6 GiB of Q2 tensor bytes on
  dwagon host RAM. The all-local four-GPU profile leaves approximately
  720.6 GiB and is too close to capacity unless nearly every GiB is recovered,
  hence its 750,000-MiB admission gate.
- Builds must contain `sm_86;sm_120` for the future four-GPU presets and
  `sm_75;sm_86;sm_120` when the modified 2080 Ti is local. A non-CPU fwuff RPC
  endpoint disables CUDA graphs by default because the modified 2080 Ti and
  the reported K3 RPC-CUDA path require the conservative mode.
- Keep the 22-GB 2080 Ti on fwuff initially. It can hold hot/owned experts or a
  quantized/FP16 draft fallback while fwuff's 3090 runs the BF16 DSpark draft.
  Moving it local is worthwhile only if the fifth device has clean PCIe
  topology and removing 92 RPC synchronization points beats the lost lanes
  and cooling headroom.

### K3-specific expert parallelism

- `scripts/build_kimi_k3_expert_ownership_plan.py` and its exact Q2 tensor-size
  data model all 92 routed layers x 896 experts. Routed expert weights total
  799,065,243,648 bytes. The deterministic planner accepts measured per-layer
  route counts, per-host capacity, and relative throughput, then uses
  load-aware assignment plus capacity repair rather than contiguous expert
  IDs.
- An unprofiled 650-GiB dwagon/180-GiB fwuff, 2:1-throughput smoke plan assigns
  605,799,161,856 bytes to dwagon and 193,266,081,792 bytes to fwuff with
  7,446,528 bytes spare. This proves capacity arithmetic, not optimal
  ownership; real route frequency and co-occurrence profiles are mandatory.
- The proposed SmallEP-like primitive routes once on dwagon, sends one
  3,584-float latent plus selected IDs/weights to each remote owner, executes
  local and remote experts concurrently, and returns one F32
  3,584-element partial per host. Traffic is about 28.1 KiB per remote host
  per MoE layer, or 2.53 MiB/token across 92 layers. EDR bandwidth is ample;
  the risks are 92 latency barriers and route imbalance.
- Expert weights must be compact on each owner with a global-to-local map.
  Profile-guided duplication of hot experts on the 2080 Ti can avoid common
  remote routes. First prove the local expert-partial primitive between
  dwagon's 3090s, then CPU RPC correctness, then compare the 2080 Ti on fwuff
  against installing it locally.

### K3 DSpark port

- The official [Kimi K3 DSpark draft](https://huggingface.co/Inferact/Kimi-K3-DSpark/blob/main/README.md)
  is a five-dense-layer, hidden-7,168/intermediate-14,336 model with
  rank-256 Markov recurrence and target checkpoints
  `[2,23,47,71,89]`. BF16 weights are about 7.1 GB. Its published mean
  acceptance is 3.85 tokens in greedy mode and 3.73 under sampling.
- An intentionally uncommitted llama.cpp vertical slice now converts the
  official architecture, exposes target inputs `[3,24,48,72,90]`, shares the
  target embedding/head, implements the dense RoPE/MLA draft, and always
  evaluates the physical seven-row block. Requested widths 3/5/7 cap returned
  candidates; they do not change the seven-row graph.
- Review found and the implementation fixed a subtle correctness blocker:
  single-slot `split_simple()` had marked the batch unequal and caused rows
  1--6 to index Markov weights by the MASK token rather than the preceding
  sample. The port now serializes one indivisible
  `[anchor + 6 x MASK]` block per sequence, uses the sequential path when
  there is one unique sequence, and rejects `n_ubatch < 7`.
- CPU/RPC and SM86 CUDA builds plus the focused CTest pass. A synthetic
  end-to-end target/draft run generated 12 draft tokens and accepted 3; a
  six-row microbatch rejected DSpark and safely fell back to target-only.
  Converter compilation and official-header mapping validation also pass.
  The original three-patch implementation has stable patch ID
  `65f93ce350585e42336d51031e2f7bca289ed91d`. Compatibility patch `0002d`
  has raw SHA-256
  `2439173c5bf8075ce648d4fbda662bda99526ed999b261e318732597d686dfeb`
  and stable patch ID `5c154e738b7ad434535f8c341e97f45810c088e2`; the complete four-file
  DSpark series has concatenated SHA-256
  `60c83d20d1e55a34cf8304c81d7dc1ce7746c751b8f89b80976bbbc1061054e8`.
- The official artifact is now pinned and converted rather than inferred from
  headers. Source repository commit
  `cf6b8244620e7ea4b0651d214f28e89eac75bed6` supplied a
  7,124,633,450-byte `model.safetensors` whose Git-LFS SHA-256 is
  `f9972a636d92a11994cdcfc88fd4c5b5d50d6eb2a89af016031593b8c65c2053`.
  The real conversion completed in 96 seconds with approximately 5.1 GB peak
  memory and produced a 4,782,905,984-byte
  `Kimi-K3-DSpark-BF16.gguf`, SHA-256
  `71989564e0bec353cc0b7f5ee3f333781de072da34a03871ad3a040d68b64de9`.
  `gguf_dump` reports architecture `dflash`, five blocks, hidden size 7,168,
  intermediate size 14,336, 64 attention heads, one KV head, target layers
  `[3,24,48,72,90]`, seven-token physical blocks, Markov rank 256, and mask
  token 163837. The apparent 2.4B converted parameter count is expected
  because DSpark shares the target embedding and output head.
- All of that storage is on fwuff's Sanic array at
  `/mnt/sanic/exo/models/Inferact-Kimi-K3-DSpark-cf6b8244`. A zero-copy
  20-file hardlink view at
  `/mnt/sanic/exo/model-views/Kimi-K3-Q2-DSpark-cf6b8244` combines the 19 Q2
  shards and the draft GGUF under the one direct-child source root required
  by strict RPC attestation. Dwagon sees the same inodes through
  `/mnt/sanic-edr/exo/model-views/Kimi-K3-Q2-DSpark-cf6b8244`; no 866-GB
  model copy was created.
- This is still not a claimed model-level speedup until the full Q2 target and
  real draft run together. The first live sweep uses request-level verifier
  caps 3/5/7 on fwuff's otherwise-idle 3090 without reloading either model.
  Shared embedding/head access, real acceptance, sampled decoding, and
  rollback remain the integration gates. Multi-slot drafting is correct but
  deliberately serialized.

### K3 CUDA kernels

- The first measured specialization fuses the exact 896-expert/top-16 router
  path. On an RTX 3090 it reduced batch-one router time from 31.96 to
  13.17 microseconds (2.43x) and batch-512 time from 177.56 to
  19.70 microseconds (9.01x). All 353 `TOPK_MOE` and 33 K3-focused checks
  passed. The fat binary contains SM75, SM86, and SM120a cubins; only SM86 was
  executed. Stable patch ID:
  `37a27b04d66aa431cd2dafd93c27d0061e72959a`; raw patch SHA-256:
  `82396307a9c71d56201e649b2b2400971ff19a957d65c4271d0a397e2baa44a7`.
- Router fusion is real but not the main bottleneck: even all 92 local routers
  save at most about 1.73 ms/token, approximately 0.28% of an IQ2 decode and
  much less in the distributed Q2 placement.
- The first SiTU gate/up implementation is complete, with the negative result
  used as a dispatch rule rather than hidden. A removed experimental paired
  `IQ2_XS` path measured 223.28 microseconds after a focused launch-policy
  tune versus 205.11 for the legacy decomposition, an 8.86% regression, so
  final source has no IQ2 pair dispatch. That experiment's small-K kernel grew
  from 96 to 116 registers/thread and from 1,536 to 3,072 bytes shared memory.
  `IQ3_XXS` behaves in the opposite direction:
  261.67 microseconds fused versus 289.15 for two MMIDs plus standalone SiTU
  and 291.82 legacy, 10.33% lower latency, while registers fall from 54 to 48.
- The patch therefore selects first-class SiTU only for an exact K3
  `[3584,3072,896]`, one-token, top-16, no-bias/no-scale `IQ3_XXS` pair whose
  actual placement backend reports support. `IQ2_XS` and unsupported
  Metal/Vulkan/SYCL paths preserve the original graph. A positive debug trace
  fused exactly three nodes from gate `MUL_MAT_ID` to output `GLU`; an IQ3
  15-expert shape near miss and exact IQ2 control did not fuse. Standalone
  SiTU passed 3/3 and distinct-weight K3 cases passed 5/5; the final source
  built release server/tests plus SM75/SM86/SM120a cubins. Patch `0004` has
  not yet run a full-size K3 model; this is microbenchmark, correctness-test,
  and build evidence.
- In `UD-Q2_K_XL`, only block 91 has IQ3 gate/up weights; the other 91 routed
  blocks are IQ2_XS. The saving is consequently about 30 microseconds/token
  when that block is CUDA-resident—approximately 0.00112% against the
  historical 0.372718-token/s Q2 mean and 0.000634% against the latest
  0.210238-token/s file-aware run. `UD-IQ2_XXS` has 46 IQ1_M and 46 IQ2_XXS
  gate/up pairs, no IQ3_XXS pair, and its accepted placement keeps routed
  experts on CPU, so this specialization does not accelerate it. This is
  useful fusion infrastructure, not a visible end-to-end speed claim. Series
  patch `0004`, rebased after
  `0003`'s generic whole-graph test hook, has SHA-256
  `e777bd9351dad1f040fe61ee3cc3eadf80a64fb784b4a4e5daa897f95bcf73e2`
  and stable patch ID
  `b3d1eddceb226af14da08b6315d8b8be32bcea48`. The validated pre-rebase clean
  diff has SHA-256
  `04c379d4e8021bdc7ce34b07264b970d55fd3cb2ace89cc3595ab7b6973fbb38`
  and patch ID
  `23d8f8c095c31f0fa8ddad791930dc0a6305edb8`; matched patched binaries are
  required on both RPC client and server.
- Follow-on patch `0007` makes the dominant `IQ2_XS` case profitable without
  changing frozen patch `0004`. Its exact K3 kernel uses four warps/four rows
  and two sequential phases: reduce the gate MMID to four shared scalars,
  reuse the same accumulators and 1,536-byte inter-warp scratch for the up
  MMID, then apply SiTU and write the final `[3072,16]` result. This removes
  the simultaneous accumulator lifetime that made the first generic IQ2
  pairing regress.
- The SM86 cubin uses 96 registers/thread, 1,552 bytes shared memory, and zero
  stack/local memory. The removed pairing used 116 registers and 3,072 bytes;
  the new kernel therefore returns to the unfused IQ2 register-occupancy
  class. In a controlled same-binary RTX 3090 comparison, fused runs were
  162.25 and 162.45 microseconds, while
  `GGML_CUDA_DISABLE_FUSION=1` runs were 170.14, 169.11, and 169.05
  microseconds. The approximate means are 162.35 versus 169.43
  microseconds, 4.18% lower latency. An earlier independent pair measured
  161.50 versus 168.34 microseconds, 4.06% lower.
- Admission is exact: K3 batch one, 896 experts/top-16,
  `[3584,3072,896,1]` `IQ2_XS` gate/up weights, matching layouts,
  `[3584,1,1,1]` input, `[3072,16,1,1]` output, every unused higher
  dimension equal to one, and no bias or scale. Unsupported cases retain the
  standalone-SiTU path. `GGML_CUDA_KIMI_K3_SITU_MMID=0` is the targeted
  kill switch. Kimi's routed and model-specific graph admission now accepts
  first-class IQ2 SiTU only when the placement backend reports support.
- The exact pre-hardening CPU-reference/CUDA fixture passed. The final source
  adds scattered far expert IDs, expert-distinct quant-block scales, an IQ2
  15-expert near miss, and strict fourth-dimension checks. Those additions
  compile in release `test-backend-ops`, `llama-server`, and
  `ggml-rpc-server`, but were not rerun on a GPU after fwuff's 3090 was
  yielded to the resident K3/DSpark benchmark. This patch has not run a
  full-size model.
- When all routed blocks are GPU-resident, `0007` applies to 91 of Q2's 92
  routed blocks rather than only the single IQ3 block. The microbenchmark
  ceiling is still only about 0.64 ms/token across all 91 layers, so this is
  a concrete kernel improvement, not a visible distributed-model speed
  claim. No operation was added and RPC remains v7.0.1. Deploying matched
  binaries on both ends is nevertheless recommended so model-graph admission
  and server execution select the same path.
- Exported patch
  `scripts/patches/kimi-k3/0007-kimi-k3-iq2-situ-mmid-sequential.patch`
  has SHA-256
  `66fa3d71a75f886aed6ddcf6f4a6a37d7dbc3c93afa7c1eeb25cc1c37a34426a`
  and stable patch ID
  `da7086b47540028b90b25a60c8c7a09e14a6b1dc`. The complete filename-ordered
  `0001`--`0007` stack apply-checks cleanly from `d29a524e`; the live
  K3/DSpark run intentionally remains on frozen `0006`.
- Patch `0006` now implements the higher-value decode primitive:
  `MUL_MAT_ID_WEIGHTED_REDUCE` accepts the exact K3 down weights
  `[3072,3584,896]`, 16 activated expert rows, selected IDs, and router
  weights, then writes one 3,584-element F32 partial. Four SM86 warps process
  four selected experts concurrently and each CTA produces two output rows.
  The router products and additions use `__fmul_rn`/`__fadd_rn` in slot order,
  so the reduction is deterministic and does not use atomics.
- Admission is deliberately narrow: K3 only, batch one, 896 experts/top-16,
  exact dimensions, `IQ1_M`/`IQ2_XS`/`IQ3_XXS` down weights, no LoRA, no
  output scale or bias, no weight-before-FFN path, and a GPU placement backend
  that explicitly reports operation support. Warmup shapes, CPU and other
  backends retain the old graph. `LLAMA_KIMI_K3_FUSED_DOWN=0` is the kill
  switch. Adding the operation advances the attested RPC stack from v7.0.0
  to v7.0.1, so client and server binaries must match.
- The same patch incorporates open llama.cpp PR
  [#25952](https://github.com/ggml-org/llama.cpp/pull/25952)'s structural
  post-MMID weighted reduction. The matcher is extended from top-15 to K3's
  exact 32-node top-16 short form; the 33-node scaled top-16 form still falls
  back. This is the fair optimized baseline for the first-class operation.
- Final same-binary RTX 3090/SM86 timings for the exact
  3072-by-3584-by-896, scattered-ID, distinct-activation/router-weight fixture
  were:

  | Down type | Original MMID/tail | PR #25952 post-MMID | Fused 4-warp x 2-row | vs original | vs post-MMID |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `IQ1_M` | 79.33 us | 80.71 us | 66.26 us | 16.48% faster | 17.90% faster |
  | `IQ2_XS` | 91.95 us | 90.07 us | 81.52 us | 11.34% faster | 9.49% faster |
  | `IQ3_XXS` | 128.70 us | 127.06 us | 96.26 us | 25.21% faster | 24.24% faster |

  One-row CTAs measured 77.52/87.06/101.35 us for
  `IQ1_M`/`IQ2_XS`/`IQ3_XXS`; four-row CTAs measured
  69.13/85.01/98.72 us. Two rows won all three formats. The final kernels use
  56/63/62 registers per thread respectively, 128 bytes of shared memory, no
  local stack, and no local spills.
- Direct CPU-versus-CUDA graph checks passed 3/3 for the three real down
  formats. The general post-MMID suite passed 5/5, including the exact
  top-16 boundary, scaled top-16 fallback, runtime-k, vector, scalar, and
  unaligned cases. The complete `0001`--`0006` stack built release
  `test-backend-ops`, `llama-server`, and `ggml-rpc-server` with RPC/RDMA and
  SM86 enabled, and the filename-ordered stack apply-checks cleanly from
  `d29a524e`. This remains microbenchmark/build evidence; `0006` has not yet
  run a full-size K3 model.
- Exported patch
  `scripts/patches/kimi-k3/0006-kimi-k3-down-mmid-weighted-reduce.patch`
  has SHA-256
  `3ca1b11b8073364fccea04b2e80c56c117a1ef03233bd0b712b361ef3c64be05`
  and stable patch ID
  `1f829a7affbcdb35922186e5242e3e604a86aa21`. The matched SM86 benchmark
  payload is
  `/mnt/sanic/exo/kimi-k3-kernels/down-mmid-dev-sm86`.
- The one-vector result is also the intended network unit for the
  SmallEP-like topology. A later prompt-batch implementation should compact
  active experts and use grouped work plus segmented reduction instead of
  millions of F32 atomics.

### File-aware Q2 representative result

The requested single representative run completed on the current
3090/RPC/3090 placement after one sacrificial warmup. It used the server-local
fwuff tensor reads, dwagon's `/mnt/sanic-edr` source, pageable model weights,
one cache-cold semantic gate, and one cache-cold performance request. The
semantic run produced exactly `FINAL=1080` from 124 input and 138 output
tokens. The performance run produced exactly 512 input and 128 output tokens,
zero cached tokens, and a length finish:

| Metric | File-aware representative | Prior patched representative | Prior five-run mean |
| --- | ---: | ---: | ---: |
| Prompt processing | 3.720789 tok/s | 4.744242 tok/s | 5.074897 tok/s |
| Decode | 0.210238 tok/s | 0.368137 tok/s | 0.372718 tok/s |
| TTFT | 137.609 s | 107.924 s | 100.961 s |
| End-to-end | 746.444 s | 455.621 s | 461.964 s |

This is a real regression signal, not a win to rationalize away. Against the
prior representative, prompt throughput fell 21.57%, decode fell 42.89%, TTFT
rose 27.51%, and end-to-end latency rose 63.83%. The generated content and
reasoning are byte-identical to the prior seeded run (SHA-256
`ac316f5c4c4900c944bf7dc5dd24cec8f61fdb22995a59792f6c4c2e5bff8e60`
and
`89c8d5847e0dd5cfe22bc0eccb86d0bc782e51461fc658263616ade697bd02d3`),
as is the rendered prompt. This rules out a different sampled token sequence
as the explanation.

The strict two-host telemetry join covered the entire accepted request:

- integrated GPU-board energy was 174,101.179 J, or 1,360.17 board
  J/generated token including prompt work, 64.45% above the prior
  representative;
- average process CPU use was 43.19 core-equivalents on dwagon and 33.42 on
  fwuff; host CPU use was 19.96% and 30.48%;
- maximum process RSS was 705,270,184 KiB and 100,634,980 KiB; minimum
  `MemAvailable` was 61,454,132 KiB and 139,952,124 KiB;
- neither process incurred a major fault, neither host swapped, and steady
  EDR traffic was only about 0.0121 Gb/s.

GPU power, clocks, CPU use, memory residency, and fabric traffic do not expose
an external contention event large enough to explain the loss. The SiTU build
overlapped only the semantic gate and was paused before the accepted
performance request; IQ2 staging began only after the model was unloaded.
The important confound is host-memory policy: this run introduced
`--no-host` and also manually disabled automatic NUMA balancing, whereas the
prior accepted run retained pinned host weights and the kernel default. It
therefore cannot assign causality to either change. Automatic NUMA balancing
has been restored to `1`; the launcher now records this value but never
mutates the host-global setting. `--no-host` remains the bold cold-start
default because it eliminates the approximately 700-GiB serial
`cudaMallocHost()` pass, but it is no longer described as steady-state
performance-neutral. A future controlled optimization run should compare
`KIMI_Q2_NO_HOST=1/0` with automatic NUMA balancing held at `1`.

The accepted artifacts are under
`/mnt/llm-models/exo/benchmarks/kimi-k3-q2-file-aware-20260730-run2-sanic-edr`:

- `benchmark.jsonl`:
  `e75b6fad1993413cae981a404fd4f5cf458fde9f6307affb0253dcef5b5ec377`
- `benchmark-summary.json`:
  `e88e894a0aaf6e897a2059e1992fbc147533a3444db2225114c5b8f26102b92b`
- `telemetry-dwagon.jsonl`:
  `4d9501b6a196f897fa251850aa88a97d17ac11666a8acd3c191f791134c03f56`
- `telemetry-fwuff.jsonl`:
  `b885ffa274e7ed03674ad2c38871811a819eb75fe3ddf2dff166cedd0e688a39`
- `telemetry-summary.json`:
  `93fea264f851633f9b8f6e8bbc4bd4d82d08911ccd50516605f70f01b60d59a7`
- `server.log`:
  `cb891d229a98364573e22550e8f032de873728682d8878b248fdba5c930cffd9`
- `rpc-server-fwuff.log`:
  `6bdc9aa38048847ac4f8968331ea41ef1d5ab40f66767821ad23b383c198e17a`
- restored CPU-policy receipts, dwagon then fwuff:
  `3b75971d53ed3674019e709ebc69d31fbb360c35db8fa177b3bc2b99e24dbb53`
  and
  `ac25f8147019b349eac37d1cabcf155cb2a1920c94542188c93ae4d1b8dfe6e2`

## Research ledger

All online material in this section was checked on 2026-07-30. Marketplace
prices are asking prices or individual historical observations, not appraisals
or verified completed-sale distributions.

### Model and quantization facts

- Kimi K3 has about 2.8 trillion total parameters and 104 billion active
  parameters per token. It has 93 transformer blocks: one dense block and 92
  MoE blocks, 896 routed experts with 16 selected per token, 69 KDA
  linear-attention blocks and 24 MLA blocks. The 1M-token maximum context is
  not the first deployment target.
- The official checkpoint's native MXFP4 applies to routed experts; attention,
  shared/dense parameters, output head, and vision components remain BF16.
  That is why the nominal native checkpoint is still about 1.56 TB rather than
  a uniform four-bit 1.4 TB-class artifact.
- The [Unsloth K3 quantization guide](https://unsloth.ai/docs/models/kimi-k3)
  reports the following quality/capacity tradeoff:

| Quant | Payload | Stated total memory | KLD | Perplexity | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UD-IQ1_S` | 553.20 GiB | Not selected | 0.5645 | 2.5789 | 78.875% |
| `UD-IQ1_M` | 604.31 GiB | Not selected | 0.4789 | 2.3639 | 81.219% |
| `UD-IQ2_XXS` | 662.23 GiB | 726 GB / 676.1 GiB | 0.3784 | 2.1266 | 84.127% |
| `UD-Q2_K_XL` | 802.13 GiB | 880 GB / 819.6 GiB | 0.1779 | 1.7359 | 90.390% |
| `UD-Q4_K_XL` | 1,405.06 GiB | Out of scope here | Not recorded | 1.4579 | Not recorded |
| `UD-Q8_K_XL` | 1,453.94 GiB | Out of scope here | KLD/reference baseline | 1.4581 | Not recorded |

Q2 is the right quality-first experiment on the combined hosts. IQ2 is
meaningfully less faithful, but its ability to fit dwagon alone eliminates an
RPC stage and could make it the more useful interactive configuration.

### Current local K3 runtime evidence

- The upstream implementation is still an open development PR:
  [ggml-org/llama.cpp #26185](https://github.com/ggml-org/llama.cpp/pull/26185).
  Its pinned head `cf67f0d...` adds hybrid KDA+MLA, cross-layer residual
  attention, latent MoE, situ activation, the MLA output gate, full-rank KDA
  gate, K3 chat parsing, and MXFP4 conversion support.
- [Unsloth PR #48](https://github.com/unslothai/llama.cpp/pull/48) adds the
  full-size fixes actually needed by these GGUFs. Commit `47c5bbdf...` reads
  `n_expert_used` per layer, sizes recurrent state from the 69 KDA layers,
  preserves expert tensor source order during conversion, and expands the K3
  graph budget for large ubatches. Its later vision commit is not needed here.
- A matching Q2_K_XL community system with 768 GB RAM plus a 92 GB GPU reported
  4.21 prompt and 3.13 decode tok/s on a short cold test, rising to 14.69
  prompt and 5.09 decode tok/s after a 5,038-token prompt. This is the most
  relevant target range in the
  [K3 implementation discussion](https://github.com/ggml-org/llama.cpp/pull/26185).
  Another 768 GB plus 2 x RTX 5090 report observed roughly 4 decode tok/s and
  50--70 prompt tok/s on longer, warmer prompts
  ([community report](https://www.reddit.com/r/LocalLLaMA/comments/1va0rce/first_kimi_k3_results_on_home_lab_4ts/)).
- Storage spill is a categorical failure mode, not a small regression: a Q2
  system with 512 GB RAM, 192 GB VRAM, and a 29 GB/s NVMe array managed only
  about 0.41 prompt and 0.23 decode tok/s
  ([community report](https://www.reddit.com/r/LocalLLaMA/comments/1v9cwfz/i_got_kimik3_running/)).
  Every admitted configuration here must therefore retain real RAM headroom
  and near-zero major faults after warmup.
- Even 8 x B200 only produced roughly 16.3--16.6 decode tok/s in a reported
  whole-layer split. Only one device worked on a given layer at a time. This
  reinforces the local DeepSeek lesson: more capacity devices do not imply
  linear batch-one decode scaling
  ([implementation discussion](https://github.com/ggml-org/llama.cpp/pull/26185)).
- The KDA recurrent state is modest relative to weights: approximately
  467 MB per sequence in a full-size report, with around 0.84 GiB of MLA KV at
  32K context. F16 K/V is therefore the conservative starting point
  ([implementation discussion](https://github.com/ggml-org/llama.cpp/pull/26185)).
- Prefix reuse has produced incorrect full-size KDA state. Keep
  `--cache-reuse 0`, disable request prefix caching in every measured gate, and
  erase the slot between accepted samples
  ([implementation discussion](https://github.com/ggml-org/llama.cpp/pull/26185)).
- Built-in llama.cpp warmup routes to only 16 of 896 experts. It is not a
  complete weight prefault. Skip the bug-prone built-in empty warmup, perform
  one sacrificial real prompt, and distinguish cold-page from warm steady
  state
  ([implementation discussion](https://github.com/ggml-org/llama.cpp/pull/26185)).
- llama.cpp RPC is explicitly proof-of-concept and unauthenticated. Binding it
  only to the private `/30` EDR interface is mandatory. Its documented RDMA
  transport is RoCEv2-oriented; TCP over measured 92.6-Gb/s IPoIB is the
  reproducible baseline
  ([RPC documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)).
- K3 RPC-CUDA has a reported `SOFT_MAX failed: invalid argument` warmup
  failure. Separately, open
  [issue #20315](https://github.com/ggml-org/llama.cpp/issues/20315) describes
  CUDA graph growth on RPC devices. Any later fwuff-GPU endpoint must disable
  CUDA graphs and pass a coherence gate before its speed is considered.
- Use `--split-mode layer`. The unsupported path is row/tensor-parallel
  splitting of individual tensors for this hybrid/linear architecture;
  `--tensor-split` is still valid in layer mode as a device layer-allocation
  proportion. A 4,096 microbatch was tried first, exposed the measured
  8.5-GiB Q2 graph allocation failure, and the accepted profiles use 512 with
  flash attention enabled.
- K3 is thinking-only. Preserve `reasoning_content`, use the K3 Jinja parser,
  and pass the template's `thinking_effort` explicitly. Official sampling is
  temperature 1.0/top-p 0.95 for normal work and top-p 1 for agentic use; the
  semantic benchmark uses deterministic sampling so configuration changes are
  easier to diagnose.
- Do not replace this GGUF's embedded template with the current
  [Moonshot discussion/PR #66](https://huggingface.co/moonshotai/Kimi-K3/discussions/66).
  The embedded template is byte-identical to the later Unsloth template at
  commit `6095bb6b51abc90e7e8f8804ad4f65be4a441262`. It rendered byte-identically
  to PR #66 for five representative text conversations and four representative
  tool/schema conversations, while also correctly honoring
  `reasoning_effort` and parsing JSON-string tool arguments into K3's typed
  tags. The live PR #66 ref still contains unresolved Minja constructs and
  ignored `reasoning_effort=low` in the audit. If a mixed or older GGUF later
  needs an override, pin the audited Unsloth template rather than the moving
  PR ref.

### Alternative runtimes and novel optimization directions

- Current official high-throughput recipes in
  [vLLM](https://recipes.vllm.ai/moonshotai/Kimi-K3) and
  [SGLang](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
  target large H100/H200/B200/B300 or MI35x fleets and native MXFP4/FP8 paths.
  They are not drop-in solutions for three Ampere consumer GPUs plus host RAM.
- `ik_llama.cpp` does not yet implement Kimi-Linear/K3, so its faster IQ
  kernels cannot currently accelerate the IQ2 experiment.
- The local llama.cpp AMX backend does not include MXFP4, Q2_K, or IQ2_XXS in
  its supported AMX quant types. Both hosts therefore fall back to generic x86
  kernels for much of K3's expert bulk despite having AMX. Porting the proven
  DeepSeek V4 MXFP4 AMX machinery in the local ktransformers fork is the
  highest-upside software project, but it requires a real K3 frontend and
  tensor-layout port rather than a flag.
- Profile-guided expert residency remains attractive. K3 selects only 16 of
  896 experts per token, while the current GGUF stores experts in fused
  tensors that cannot be overridden expert-by-expert. A runtime that records
  routing frequency, replicates hot experts into 3090 VRAM, and keeps cold
  experts in NUMA-local DDR5 could outperform whole-layer GPU offload at the
  same 72 GiB VRAM budget.
- Once stable, test `GGML_CUDA_P2P=1` across dwagon's NVLink pair. Whole-layer
  mode may not move enough data for a large win, and the flag must be rejected
  on any corruption or IOMMU instability.
- Kimi's DSpark draft reports a mean acceptance of 3.73 of seven speculative
  tokens, but llama.cpp's merged DSpark support currently accepts Qwen3
  backbones rather than K3. It is research, not a first-iteration speedup
  ([draft model](https://huggingface.co/Inferact/Kimi-K3-DSpark),
  [llama.cpp PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)).
- Cross-host expert partial sums are a later experiment. The DeepSeek work
  showed that concurrent remote expert tiers can help, but llama.cpp's
  whole-layer RPC is much simpler and avoids a network round trip inside every
  MoE block. Establish the layer baseline before considering a K3-specific
  expert sidecar.

### Fresh upstream and marketplace delta (2026-07-30)

The following is an implementation-input audit, not local K3 validation:

- Draft [llama.cpp PR #26322](https://github.com/ggml-org/llama.cpp/pull/26322)
  at head `8da6ce7` is the clearest full-MoE CUDA design reference found. It
  combines routing, activation reuse, gate/up, quantize-plus-SwiGLU, down
  projection, and weighted reduction, with 6--11% prefill gains reported for
  Qwen3.5 MoE on DGX Spark and RTX 5090. Its current graph admission requires
  the softmax-plus-normalized-SwiGLU pattern and rejects routing features K3
  needs, so K3 is excluded: borrow the dataflow and kernels, not the PR as a
  validated K3 patch.
- Open [PR #25952](https://github.com/ggml-org/llama.cpp/pull/25952) at head
  `fbc1bd7` is the nearer baseline: one CUDA kernel replaces the weighted
  expert-combine tail after `MUL_MAT_ID`. Its structural matcher supports
  top-k 2--15 because the long form reaches the 31-node fusion limit; K3's
  top-16 route therefore falls back. Extending that cap is required before
  its reported 3.6--7.1% non-K3 prefill gains can even be tested here.
- Merged [PR #24481](https://github.com/ggml-org/llama.cpp/pull/24481) supplies
  the upstream precedent for matching gate/up lanes from `MUL_MAT` or
  `MUL_MAT_ID`, optional scale/bias, and GLU as one fusion. Reuse that matcher
  framework for the K3 fused gate/up MMID work instead of adding a second
  model-name gate.
- Open [PR #26079](https://github.com/ggml-org/llama.cpp/pull/26079) found on
  an RTX 5090 that the K-quant MMVQ-to-MMQ crossover depends on model and
  quant, with best thresholds spanning 3--5 and MMQ gains of 20--55% at width
  eight in its dense microbenchmarks. This is not K3 or speculation
  validation, but it is strong reason to sweep K3 DSpark widths 3/5/7 on the
  5090 rather than assume seven.
- Closed, unmerged [PR #24524](https://github.com/ggml-org/llama.cpp/pull/24524)
  proposes a persistent hot-expert VRAM cache while CPU misses execute in
  parallel. The author reports concentrated expert reuse and 7--25% gains on
  two large spilling MoEs, but the large predominantly AI-generated change
  was not reviewed. Treat it only as a controlled, kill-switch A/B after K3
  route telemetry establishes a stable hot set; do not port it wholesale.
- Withdrawn [CPU TP/EP RFC #25209](https://github.com/ggml-org/llama.cpp/pull/25209)
  sketches quant-aligned load-time expert sharding and hierarchical
  shared-memory/UCX-InfiniBand reduction. It received no maintainer review, so
  its claimed results are not evidence for this cluster; it is only a useful
  topology reference for the K3 SmallEP experiment.
- Open [PR #26291](https://github.com/ggml-org/llama.cpp/pull/26291) parallelizes
  FNV hashing on warm RPC-cache loads. That can help only the transfer/cache
  fallback here; the strict protocol-v7 file-aware path reads an attested
  local shard directly and should not pay this hashing path.
- [GHSA-j8rj-fmpv-wcxw](https://github.com/ggml-org/llama.cpp/security/advisories/GHSA-j8rj-fmpv-wcxw)
  documents critical unauthenticated RPC remote code execution and lists no
  patched version. Binding RPC only to the private dwagon--fwuff InfiniBand
  `/30`, firewalling the port to the peer address, and never exposing it on a
  general LAN or the Internet are hard requirements, not optional hardening.

Marketplace observations below are asking-price snapshots, not sold
comparables or performance validation:

| Device | Technical value for this workload | Observed ask on 2026-07-30 | Decision delta |
|---|---|---:|---|
| [MI210](https://www.amd.com/en/products/accelerators/instinct/mi200/mi210.html) | 64 GB HBM2e, 1.6 TB/s; attractive cold-expert capacity, but `gfx90a` K3 and mixed-backend operation remain unproved | [used $5,144.99 plus shipping](https://www.ebay.com/itm/306934911271) | More expensive and much higher integration risk than the planned 3090+5090 path; rent-test first |
| [A100 80 GB PCIe](https://www.nvidia.com/en-eu/data-center/a100/) | 80 GB HBM2e, 1.935 TB/s; strongest single-card capacity/bandwidth option in this sample, but still cannot contain Q2 | [used $18,999](https://www.ebay.com/itm/178013232550) | Technically useful, economically noncompetitive |
| [RTX A6000](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20%281%29.pdf) | 48 GB CUDA/ECC in one active-cooled card, but only 768 GB/s | [used $3,599.99](https://www.ebay.com/itm/115925848642) | Capacity convenience does not offset less VRAM and far less output-device bandwidth than the planned 24-GB 3090 plus 32-GB 5090 |
| [A40](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf) | 48 GB CUDA/ECC, 696 GB/s, passive 300 W | [used $4,500](https://www.ebay.com/itm/277959875038) | Slower, dearer, and harder to cool than the A6000 example; no advantage over the planned pair |

None of these asks overturns the existing purchase order: add the 3090 and
5090, exploit their established CUDA paths, and revisit unusual 48--80 GB
cards only after a materially lower sold price or a K3 benchmark proves a
specific placement win.

### Second-hand OAM, SXM, and oddball hardware

#### The loose-module trap

The OCP OAM specification standardizes module mechanics and core electrical
interfaces, not arbitrary platform compatibility. A useful system still needs
the exact UBB/HGX board, host bridge/riser, BMC and VBIOS firmware, clock/reset
and power sequencing, cold plate or heatsink, fans/CDU, high-density cables,
and 48 V or proprietary power shelf
([OAM v1.5](https://www.opencompute.org/documents/ocp-accelerator-module-design-specification-v1p5-final-20220223-docx-1-pdf)).
Loose OAM/SXM modules, bare UBBs, and HGX trays should be valued as parts, not
as usable GPUs. Recent MI250 reverse-engineering reports describe HPE Cray
modules crashing AMDGPU or failing outside their EX235a platform
([ROCm report](https://www.reddit.com/r/ROCm/comments/1ux9w0u/is_the_amd_instinct_mi250_a_good_choice_for_aigc/),
[Level1Techs thread](https://forum.level1techs.com/t/someone-needs-to-figure-out-how-to-adapt-mi250-gpus-to-pcie/250596)).

#### MI250X: the only plausible used HBM bargain, with serious risk

- One MI250X OAM exposes two logical GCDs, 128 GB HBM2e total and 3.2 TB/s
  aggregate bandwidth
  ([AMD system documentation](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi250.html)).
  It consumes about 560 W.
- Q2 needs seven modules just to clear raw weight bytes; eight is the practical
  minimum at 1,024 GB decimal/~954 GiB. A normal four-OAM chassis cannot hold
  Q2.
- Loose HPE MI250/MI255 modules have appeared around $1,395--$2,095, including
  [this MI250 listing](https://www.ebay.com/itm/206238044118) and
  [this MI255 listing](https://www.ebay.com/itm/396999323615). Their intended
  platform is the proprietary liquid-cooled
  [HPE Cray EX235a](https://www.hpe.com/psnow/downloadDoc/HPE%20Cray%20Supercomputing%20EX%20QuickSpecs-a00094635enw.pdf?contentDisposition=attachment&deepLink=&form=false&hf=regular&id=a00094635enw.pdf&isFutureVersion=true&isLinearized=false&originalObjectName=&prelaunchSection=&preview=false&print=&r=&section=&softrollSection=&ver=16).
- A complete-ish four-MI250
  [Supermicro AS-4124GQ-TNMI listing](https://www.ebay.com/itm/397928458814)
  ended around $16,100 plus $1,100 freight. Two proven, populated four-module
  systems plausibly become a $40--55k project after CPUs, RAM, storage, NICs,
  and missing parts. GPU draw alone is 4.48 kW; expect roughly 6--8 kW at the
  wall.
- Software is the gating risk. Official K3 serving does not validate gfx90a.
  Rent or borrow MI250-class hardware and prove this exact GGUF, ROCm version,
  tensor placement, prompt/decode rate, and long-run stability before buying a
  platform.

#### MI300X and NVIDIA HGX/SXM: technically good, economically poor

- MI300X provides 192 GB HBM3 at 5.3 TB/s and 750 W
  ([AMD product page](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)).
  Five nominally clear Q2, while actual systems are eight-way. Used module
  asking prices around $22--25k and complete 1.5-TB systems in the
  $240--300k class eliminate the value case. A cheap bare
  [Quanta ZionEX tray](https://www.ebay.com/itm/366367739595) still lacks its
  ORV3 host, power, management, cables, and liquid cooling.
- Eight A100 80 GB provide only 640 GB, so Q2 needs at least 11 raw/12
  practical devices across systems. Bare HGX boards and $595 SXM-to-PCIe
  adapters do not solve power, cooling, NVSwitch, or host-bridge integration.
- Eight H100 80 GB also provide only 640 GB. Cheap H100 QS/ES or Code-10
  listings are not a bargain. H200 141 GB needs seven raw/eight conventional
  modules; one current used module was listed at
  [$24,999](https://www.ebay.com/itm/336513440641), while a complete used
  eight-way listing was
  [$513,540](https://www.ebay.com/itm/167882999886).
  DGX H100/H200-class systems draw up to about 10.2 kW and approach 99 dB
  ([NVIDIA DGX guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/dgxh100-user-guide.pdf)).
- Old V100/MI50 modules fail capacity arithmetic: Q2 needs roughly 28
  32-GB devices. Carrier count, PCIe roots, 8--10 kW draw, old software, and
  custom cooling erase the low per-module price.

#### Gaudi, GH200, and other unusual options

- Gaudi2 has 96 GB HBM2e, 2.45 TB/s, and 24 x 100GbE links
  ([Intel overview](https://www.intel.com/content/www/us/en/developer/articles/technical/habana-gaudi2-processor-for-deep-learning.html)).
  A credible eight-card HLS-2 with 1 TB host RAM was listed at $22,000 plus
  $1,500 freight
  ([listing](https://www.ebay.com/itm/257374642150)). It is excellent general
  research value, but 768 GB HBM is too small for Q2, IQ2 barely fits, and the
  validated Gaudi/vLLM matrix has neither GGUF UD-IQ2 nor K3
  ([compatibility matrix](https://docs.vllm.ai/projects/gaudi/en/latest/getting_started/compatibility_matrix.html)).
- Gaudi3 has 128 GB per 900-W OAM; eight fit Q2 but the UBB alone has been
  quoted near $125k, with no validated K3 path
  ([platform pricing discussion](https://www.servethehome.com/intel-gaudi-2-8x-oam-ubb-65k-gaudi-3-125k-and-includes-networking/)).
- GH200 is the most technically relevant oddball. One package combines
  96 GB HBM3 and 480 GB Grace LPDDR5X over a 900 GB/s coherent C2C link
  ([NVIDIA architecture](https://developer.nvidia.com/blog/?p=103652)).
  One cannot fit Q2, but a true dual-GH200/NVL2-class system can. Current
  single systems ask about $39--40k
  ([used Quanta](https://www.ebay.com/itm/327230452386));
  dual systems are roughly an $80--95k proposition.
- GH200's tiering is unusually well matched to giant MoE: keep dense/shared
  tensors in HBM and experts in coherent Grace memory. A custom llama.cpp
  Qwen3.5 122B report improved prompt from 8.67 to 83.59 tok/s and decode from
  7.11 to 61.91 tok/s
  ([llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/21112)).
  A Kimi K2 Q3 expert-placement report observed about 520 prompt/20 decode
  tok/s at short context and 102/16.1 at 131K
  ([community report](https://www.reddit.com/r/LocalLLaMA/comments/1pl1zpa/evening_fun_with_grace-and-hopper-unified_memory/)).
  This is the architecture to rent-test or watch for an exceptional complete
  dual-system deal, not a cheap module experiment.
- Qualcomm Cloud AI 100 Ultra is elegant on paper: 128 GB LPDDR4x, 548 GB/s,
  150 W; eight would provide 1 TB at 1.2 kW
  ([product page](https://www.qualcomm.com/data-center/products/cloud-ai-100-ultra)).
  Scarce supply, a proprietary compiler, and no K3/GGUF backend make it a dead
  end today.
- CXL memory expanders are a future *tier*, not a DDR5 replacement for K3.
  Samsung's published first-generation CMM-B result reaches about 35 GB/s for
  one CXL device under a mixed read/write test
  ([white paper](https://download.semiconductor.samsung.com/resources/white-paper/CMM-B_whitepaper-V2.pdf));
  even a 512-GB card is therefore far below dwagon's aggregate socket-local
  DDR5 bandwidth. It becomes interesting only after a runtime can place
  profile-cold experts independently. Today's fused llama.cpp expert tensors
  would drag the hot routes into the slow tier, so do not buy an early CXL
  module merely for this model.
- Intel Optane PMem is the genuinely cheap capacity oddball: used 512-GB
  modules currently ask roughly $329--350
  ([market search](https://www.ebay.com/shop/Intel-Optane-Persistent-Memory?_nkw=intel+optane+persistent+memory)).
  PMem 200 reaches 512 GB per DIMM but requires a compatible third-generation
  Xeon platform; in Memory Mode, DRAM is a hardware-managed cache in front of
  the slower PMem
  ([Intel mode description](https://www.intel.com/content/www/us/en/support/articles/000055901/memory-and-storage/intel-optane-persistent-memory.html)).
  K3's wide, input-dependent expert access is exactly the pattern most likely
  to miss that cache. A complete 1--2 TB Optane server could be a low-cost
  capacity experiment, but it is not a likely throughput upgrade over the
  present 768-GiB DDR5 host.
- Used Xeon Max 9480 CPUs are a more technically appealing experiment. Each
  provides 64 GB HBM2e at up to about 1 TB/s plus eight DDR5 channels and AMX,
  but the platform is only two-socket, so 128 GB of HBM cannot hold K3
  ([Intel architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/xeon-scalable-processor-max-series.html),
  [9480 specifications](https://www.intel.com/content/www/us/en/products/sku/232592/intel-xeon-cpu-max-9480-processor-112-5m-cache-1-90-ghz/specifications.html)).
  It is worthwhile only with explicit dense/hot-expert placement in HBM;
  generic cache mode and llama.cpp's current non-AMX K3 expert kernels leave
  most of the value unused.

#### PCIe oddballs and the current consumer-GPU market

- Use the already-owned modified 22-GB 2080 Ti before buying an experiment.
  llama.cpp's CUDA build covers SM75, but the current
  [KTransformers `kt-kernel`](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)
  requires compute capability 8.0 or newer and ships SM80/86/89/90. That makes
  the card a llama.cpp hot-expert/owned-layer device or an FP16/quantized draft
  fallback, not the primary KTransformers/DSpark GPU. Fwuff is the first
  location to try because its 3090 can remain the BF16 draft device.
- A genuine used RTX 3090 remains the safest incremental purchase. Two recent
  sold listings were approximately
  [$960](https://www.ebay.com/itm/366350059267) and
  [$1,200](https://www.ebay.com/itm/277167673465). Its known 24-GB CUDA path,
  936-GB/s-class bandwidth, 350-W board power, and existing local build/tests
  dominate the integration risk of unusual accelerators.
- Buy the RTX 5090 only near its $1,999 launch price or, at most, roughly
  $2,500. It supplies 32 GB and 1,792 GB/s but draws 575 W
  ([NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)).
  Current in-stock retail around
  [$4,329](https://www.newegg.com/p/pl?N=100007709+4131+8000+601469153)
  is poor value for 8 GB more capacity than a 3090. The main value here is
  output-side bandwidth and future Blackwell kernels, not solving Q2 capacity.
- The AMD Radeon Pro V620 is the most interesting cheap/returnable cold-expert
  experiment: 32 GB GDDR6, 512 GB/s, PCIe 4 x16, ECC, passive cooling, and
  300 W
  ([AMD specifications](https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html)).
  An active listing around
  [$450 with more than 600 sold](https://www.ebay.com/itm/157133307609)
  is extraordinary capacity per dollar. It still needs server airflow,
  BAR/MMIO and ROCm/Vulkan validation, and a separate HIP/Vulkan RPC process;
  it cannot run `kt-kernel`. Low 512-GB/s bandwidth also makes whole-layer
  pipeline placement risky. Test only with returns and use it for cold/owned
  experts.
- A tested PCIe MI100 around
  [$999](https://www.ebay.com/itm/285394293082) is the stronger AMD experiment:
  32 GB HBM2, approximately 1.2 TB/s, 300 W, and a current ROCm `gfx908` path
  ([AMD platform documentation](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi100.html)).
  It remains passive, adds a separate HIP runtime, and has no KTransformers
  GPU path. Prefer it over V620 only when bandwidth matters enough to justify
  roughly twice the price.
- Intel Data Center GPU Max 1100 is a watch/borrow candidate: 48 GB HBM2e,
  1,228.8 GB/s, and 300 W
  ([Intel specifications](https://www.intel.com/content/www/us/en/products/sku/232876/intel-data-center-gpu-max-1100/specifications.html)).
  llama.cpp's
  [official SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
  explicitly validates Max 1100/1550 and now includes fused MoE work. One unit
  sold for
  [$1,599.99](https://www.ebay.com/itm/157796596837), while a current
  no-return listing asks
  [$1,999.99](https://www.ebay.com/itm/327119960744). The capacity/bandwidth
  arithmetic is attractive, but K3/SYCL maturity and cross-backend RPC make a
  no-return purchase unjustified.
- A complete eight-Gaudi2 system at
  [$17,500](https://www.ebay.com/itm/168027117930) is remarkable general
  hardware per dollar: 768 GB HBM and about 4.8 kW of accelerator power. It is
  still a hard no for this task because llama.cpp/KTransformers have no Gaudi
  backend and the current Gaudi-vLLM matrix has no K3 path. Hardware value
  without a runnable software path is zero for this benchmark.
- Bare MI250X OAM modules around
  [$1,500](https://www.ebay.com/itm/287342587892) do not change the loose-module
  conclusion. A usable OAI platform needs the UBB, host interface, power,
  clocks/reset, control/security/management, and cooling defined by the
  [OCP Universal Baseboard specification](https://www.opencompute.org/documents/ocp-oai-universal-baseboard-design-specification-v1p0-2-pdf).
  The approximately
  [$16,100 plus freight four-MI250 platform](https://www.ebay.com/itm/397928458814)
  is the minimum kind of listing to evaluate, yet four modules provide only
  512 GB and cannot contain Q2. Require a boot-tested, returnable complete
  system and a rented `gfx90a` K3 proof; loose modules are parts.

#### Capacity-first DDR5 remains the rational used purchase

A used dual-EPYC Dell R7625 with 1.5 TB DDR5 sold around $10,888 in late 2025
([listing](https://www.ebay.com/itm/326340307148)); a current 24-DIMM CTO
chassis was about $4,700 before CPU/RAM
([listing](https://www.ebay.com/itm/206229382250)). A complete populated
1--1.5 TB, 16--24-channel server below roughly $12--15k is the only conventional
used purchase worth considering. It buys capacity, consolidation, and NUMA
bandwidth rather than HBM speed. Because dwagon already embodies this design,
software placement and an AMX/hot-expert path have higher near-term return.

Purchase ranking:

1. Keep the current DDR5 plus 3 x 3090 cluster and optimize software.
2. Consider only a complete, remotely demonstrated 8 x MI250X deployment after
   a rented gfx90a K3 proof.
3. Watch true dual-GH200/NVL2 systems for exceptional pricing; technically the
   best giant-MoE fit, not currently a bargain.
4. Treat an 8 x Gaudi2 HLS-2 as a general research bargain, not a K3 Q2 system.
5. Revisit CXL, Optane, or Xeon Max only after expert-granular tiering exists.
6. Reject loose OAM/SXM/HGX trays, HPE Cray modules without an EX235a, QS/ES
   H100, old 32-GB farms, and unsupported Qualcomm cards.

Before purchasing any module-based system, require exact platform/module part
numbers, serial photos, native firmware/VBIOS, BMC enumeration, `rocminfo` or
`nvidia-smi`, complete HBM ECC/RAS status, all fabric links at expected
width/speed, a 30-minute sustained burn, and written return terms. Inventory
retimers/switch heatsinks, management boards, bridge cards/cables, cold plates
or fan shrouds, CDU, rails, and the exact power shelf. Six continuous kilowatts
is about 20,500 BTU/h and roughly $7,900/year at $0.15/kWh; 10.2 kW is about
34,800 BTU/h and $13,400/year before cooling overhead.

## 2026-07-30 iteration closeout: result and next steps

This iteration stops without another model launch. The infrastructure,
research, conversion, and kernel work produced useful results, but the first
real Q2-plus-DSpark integration did not reach the API and therefore produced no
honest speculative-decoding benchmark. No K3 DSpark speedup is claimed.

### What is complete

- `UD-Q2_K_XL` and `UD-IQ2_XXS` have their earlier five-run benchmark series,
  and the file-aware Q2 follow-up has its requested representative run. The
  incomplete DSpark attempt does not replace those results.
- Fwuff reads model data directly from its locally attached `/mnt/sanic`;
  dwagon reads the same inodes through the EDR-mounted `/mnt/sanic-edr`.
  Model, source, build, draft, and run artifacts on fwuff are under
  `/mnt/sanic/exo`, not its root filesystem. The strict RPC loader avoids the
  wasteful `fwuff -> NFS -> dwagon -> RPC -> fwuff` tensor path.
- The file-aware RPC protocol is fail-closed at v7.0.1. Before the live
  attempt, all 20 target-plus-draft view files matched client and server
  `{inode,size,mtime_ns}` identity; the completion receipt SHA-256 was
  `084a02355154224106cb4445a98148e6c220bca0fbf6027420dab9a25397c5b5`.
- The official five-layer K3 DSpark draft is pinned, converted, and stored as
  the 4,782,905,984-byte
  `/mnt/sanic/exo/models/Inferact-Kimi-K3-DSpark-cf6b8244/Kimi-K3-DSpark-BF16.gguf`.
  Its SHA-256 is
  `71989564e0bec353cc0b7f5ee3f333781de072da34a03871ad3a040d68b64de9`.
  The zero-copy hardlink view combines it with the 19 Q2 shards.
- Matched CUDA/RPC v7.0.1 source and build trees exist on both hosts. The live
  binaries included patches through `0006`; `0007` and `0008` are preserved
  patches but were deliberately not slipped into a resident run.
- The new fused down-projection/weighted-reduction kernel (`0006`) improves
  exact K3 RTX 3090 microbenchmarks by 11.34--25.21%, depending on weight
  format. The register-preserving IQ2 SiTU gate/up kernel (`0007`) improves
  its exact operation by about 4.18%. These are real kernel wins but have not
  yet established a visible full-model tokens/second change.
- The speculative benchmark harness can sweep request-level verifier widths
  3, 5, and 7 while one target/draft pair remains resident, and records the
  selected width in its result metadata.
- The SmallEP-like design, route-profile ownership planner, loader topology,
  and hardware/OAM survey are captured above. The practical hardware ranking
  remains software first, then the already planned local 3090 and sensibly
  priced 5090. A complete, proven platform can merit rental or evaluation;
  loose OAM/SXM modules are not useful purchases by themselves.

### Full Q2 plus DSpark integration outcome

The one permitted attempt used the strict hardlink view and matched builds:

- target devices `CUDA0,RPC0,CUDA1`, tensor split `2,11,3`, and 16 GPU/RPC
  layers;
- fwuff endpoint `CPU,CUDA0` at `10.44.0.2:50052`, mapped as target `RPC0`
  and draft `RPC1`;
- draft device `RPC1`, with CUDA graphs disabled and no host-copy fallback.

Tensor materialization was genuinely parallel: fwuff's RPC allocation/read
stage stabilized while dwagon continued loading its local share. The complete
target loaded in about 12 minutes 59 seconds. Dwagon's main process reached
roughly 677 GiB RSS and fwuff's RPC process roughly 95 GiB. The real draft
artifact then loaded, but draft-context reservation aborted before API
startup:

```text
ggml-backend.cpp:898: pre-allocated tensor (output.weight) in a buffer
(CUDA1) that cannot run the operation (NONE)
```

This is not a file-loader, memory-capacity, or sequential-load failure.
DSpark deliberately borrows the target embedding and output head. The
embedding was CPU-resident and visible, but the 1,247,805,440-byte
`output.weight` was owned by target `CUDA1`; the draft scheduler had registered
only `RPC1` plus CPU. It consequently could not schedule the preallocated
shared head.

The fatal assertion invoked `systemd-coredump` on dwagon for an approximately
677-GiB process. The handler was stopped after consuming about 52 GiB of root
space, the failed server exited, and no `llama-server` core remains. Dwagon
recovered to about 69 GiB free on root with both GPUs released. The launcher
now wraps both main-server and RPC-server commands in
`/usr/bin/prlimit --core=0:0 --`, including the CUDA RPC environment form, so
this failure mode cannot refill a host root filesystem on the next attempt.

At the pause, fwuff still answered ICMP and TCP connection probes on ports 22
and 50052 but repeatedly timed out before the SSH banner and did not complete
an RPC application handshake or NFS stat. This is consistent with teardown,
reclaim, or a remote core handler rather than a dead fabric, but it is not
proven. Its exact idle RPC process and any coredump handler therefore remain a
cleanup item once SSH responsiveness returns. No reboot is justified from the
available evidence.

### Preserved fix and fallback topology

Patch
`scripts/patches/kimi-k3/0008-kimi-k3-dspark-borrow-ctx-other-backends.patch`
is the direct fix. For each shared tensor missing from the draft scheduler it
reuses the exact owning backend handle from `ctx_other`, preserving the live
RPC session and tensor-ID namespace. Draft-owned backends retain priority;
borrowed target backends are inserted before CPU fallback. The patch covers
shared embedding/output cases used by DFlash and Eagle3 and the shared
embedding case used by the Gemma4 assistant. It also replaces this opaque
assertion with an explicit owner-visibility error.

Patch `0008` has SHA-256
`a1d96ee3028789929ec91b4c2b28c0b07e9be0fed1afed9167ea9941068f62ca`
and stable patch ID
`c07821423d02ddbf4f672bbfc1969e48cfea2e2d`. The full patch series through
`0008` apply-checks cleanly, and the modified context plus focused synthetic
non-CPU shared-tensor test objects compile with CUDA/RPC. The timebox ended
during the 462-target full build at target 106, so final executable linking,
the synthetic runtime test, and full-model validation are still required.

If `0008` is temporarily excluded, the least disruptive command-line
workaround is:

```text
--device CUDA0,RPC0,CUDA1
--tensor-split 2,11,3
--spec-draft-device CUDA1,RPC1
--spec-draft-ngl all
```

Adding target `CUDA1` to the draft scheduler exposes the borrowed head while
keeping draft blocks 1--4 and output normalization on `RPC1`; only draft block
0 follows the inherited `2,11` split onto `CUDA1`. The current launcher
deliberately rejects a multi-device draft list, so its validation would need a
small, tested relaxation before using this fallback. Prefer validating
`0008`, because automatic owner discovery is more robust than encoding shared
tensor placement in a launch command.

### Ordered next steps

1. Recover cleanly before building or loading: when fwuff's SSH banner
   responds, inspect and stop only the exact K3 RPC process from
   `/mnt/sanic/exo/run/kimi-k3-dspark-v701/rpc-server.pid`, inspect any
   `systemd-coredump`, and verify root/Sanic free space, NFS, OpenSM, GPU, and
   EDR health. Keep only the useful live OpenSM service/log; do not recreate
   deleted failed-unit logs or root-side caches. Restore the two intentionally
   stopped dwagon TTS containers once K3 GPU work is no longer imminent.
2. Complete a matched, non-model validation of `0008`: finish the CUDA/RPC
   link, run its focused scheduler test, and exercise a tiny synthetic
   target/draft pair where the shared embedding and head live on a non-draft
   backend. Treat failure here as a code bug; do not spend another 13-minute
   Q2 load to diagnose it.
3. Apply `0007` and `0008` to the persistent v7.0.1 source trees on both
   machines and rebuild matching `llama-server` and `ggml-rpc-server`
   binaries. Recheck protocol/version parity and retain
   `GGML_CUDA_DISABLE_GRAPHS=1` for the fwuff CUDA RPC endpoint.
4. For the next single full load, retain strict same-backing-file attestation,
   the Sanic/EDR hardlink view, mixed `CPU,CUDA0` RPC, target
   `CUDA0,RPC0,CUDA1` split `2,11,3`, draft `RPC1`, and the new core-size
   guards. First require API health and a short semantic comparison with
   target-only output.
5. Only while that one load remains healthy and resident, sweep verifier
   widths 3, 5, and 7 without reloading. Record accepted candidates per target
   pass, rejection/rollback behavior, request latency, prompt and decode
   tokens/second, CPU/GPU/RAM/VRAM, RPC bytes, EDR rate, and energy. Select the
   width from measured throughput, not the published mean acceptance of 3.73
   at width seven. One coherent representative benchmark is sufficient for
   this integration run.
6. After correctness, compare `0007` enabled/disabled in the same resident
   process if its kill switch permits a fair request-level test. Pursue
   SmallEP-like expert ownership only as a separate runtime project: first
   local expert-partial correctness, then CPU RPC, then a profile-guided
   dwagon/fwuff split.
7. Put the modified 22-GB 2080 Ti on fwuff first if it is tried: use it for
   cold/owned experts or a quantized draft fallback while the 3090 remains the
   BF16 draft device. Move it to dwagon only if measured remote synchronization
   outweighs the extra local PCIe contention. Do not buy loose OAM modules;
   rent-test complete MI250X/GH200-class systems before considering them.

## 2026-08-01 cumulative publication

The previously local llama.cpp stack is now committed and published publicly
as [`ldyeax/exo_llama_cpp`](https://github.com/ldyeax/exo_llama_cpp), branch
`exo/kimi-k3-cumulative`, commit `651092c60`. It accumulates the strict
local-file RPC loader, DSpark integration, K3 top-k/SiTU and weighted-reduction
kernels, recurrent-state work, RPC tests, and the shared-backend fix described
above. No upstream pull request or maintainer-facing automation was created.

The root repository now tracks that branch directly at `vendor/llama.cpp`, so
the implementation is not hidden inside another project. KTransformers also
tracks the same commit from its active llama submodule; its active and archived
llama URL declarations now use the same fork and branch. The compatible
KTransformers graph is published as
[`ldyeax/exo_ktransformers`](https://github.com/ldyeax/exo_ktransformers),
branch `exo/glm52-osdi26-patched`, commit `f38772417`. Its nested SGLang
submodule continues to use
[`ldyeax/exo_sglang`](https://github.com/ldyeax/exo_sglang), branch
`bundle/glm52-fwuff-sglang`, commit `73e877ac5`.

Modern ggml removed the task-phase structure and split conversion traits used
by KTransformers' bundled llamafile 0.8.8 kernels. Compatibility is retained
inside KTransformers: it owns the legacy task ABI, adapts the split generic/CPU
traits, uses current CPU quantization declarations, calls `ggml_cpu_init()`,
and adds the moved ggml include roots. Both a CPU-only build and a CUDA 13.1
SM86 build of `kt_kernel_ext` completed successfully against llama commit
`651092c60`.

The cumulative llama tree also completed CUDA/NCCL builds of `llama-server`,
`ggml-rpc-server`, `test-backend-ops`, `test-dflash-plan`, and
`test-rpc-tensor-source`. The two focused unit tests passed, and the focused
`MUL_MAT_ID_WEIGHTED_REDUCE`, `SITU`, and `TOP_K` backend suite passed on both
local RTX 3090 devices. This was build/test validation only; no model was
launched.
