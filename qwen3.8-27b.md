# Qwen3.8 27B FP8 on two RTX 3090s

Date executed: 2026-08-15, America/New_York

Status: complete. The required direct llama.cpp TP2 and PP2 campaigns ran with both coherence gates before timing. A first cache-disabled SGLang TP2 configuration exceeded the 100 token/s target at 107.299 full-wall completion tok/s. The requested two-hour follow-up then raised the authoritative post-gate result to **177.909 mean scheduler decode tok/s**, **172.184 mean nonstream full-wall completion tok/s**, and **105.186 ms mean client TTFT on the 68-token short prompt**. The final server was stopped after the campaign.

Publication update, 2026-08-20: the exact llama.cpp campaign tree was committed and pushed to the authoritative `exo/kimi-k3-cumulative` branch as `24d04162132a8a792d686130da1717f70784d16e`; the exact SGLang campaign tree was committed and pushed to `exo/dsv4-cumulative-0801` as `e94ec2ab18e579539b401668895d77dd1db98fe0`. References below to uncommitted or dirty trees describe their state when the immutable benchmark receipts were captured. The promoted server is running again in detached screen session `qwen38-vs2026`; operational and Visual Studio 2026 custom-agent instructions are in [`qwen3.8-27b/README.md`](qwen3.8-27b/README.md).

## Final outcome

The best measured configuration uses the authoritative vendored SGLang fork, the raw Hugging Face FP8 checkpoint, SGLang's Ampere FP8-storage Marlin W8A16 path, the model's bundled one-layer MTP drafter, a top-k-two speculative tree with six steps and eight verification tokens, a canonical 32K FR-Spec vocabulary, and a private FP8-Marlin draft head. It allocates exact 262,144-token target and draft KV pools across both RTX 3090s. The older top-k-one/ReplaySSM configuration remains documented below as an immutable first-target receipt; the two-hour extension at the end supersedes it operationally.

| Result | Value |
|---|---:|
| Post-gate scheduler decode, mean | **177.909 tok/s** |
| Post-gate nonstream full-wall completion, mean | **172.184 tok/s** |
| Post-gate streaming post-first-token decode, mean | **177.487 tok/s** |
| Post-gate 68-token short-prompt client TTFT, mean | **105.186 ms** |
| Short-prompt scheduler prefill, mean | **843.885 tok/s** |
| 18,029-token scheduler prefill | **1,911.555 tok/s** |
| Short-speed samples, per mode | 1 warmup + 5 measured, 512 output tokens each |
| Exact-context allocation | 262,144 target tokens + 262,144 draft tokens |
| Initial llama.cpp TP2 decode median | 30.078 tok/s |
| Best llama.cpp speculative screening result | 88.705 tok/s |

On the same complete-HTTP-wall denominator, 172.184 tok/s is a **60.5% improvement** over the first 107.299 tok/s result that triggered this follow-up.

This is a measured win for the 68-token asynchronous-worker-pool code prompt with thinking explicitly disabled. Throughput remains workload dependent: on the frozen finalist, the one-shot systems and review history-gate prompts measured 117.507 and 109.092 scheduler decode tok/s. The speculative path passed both semantic coherence gates but is not claimed token-identical to the non-speculative target; details and the exact launch are in the two-hour extension below.

The direct llama.cpp work still matters: it satisfied the requested backend/topology comparison, added and validated the wide F8 path, established the 32–512 microbatch curve, and reached 88.705 tok/s with bundled MTP plus n-gram prediction. The outside-the-box backend investigation then supplied the final increment past 100 tok/s. The requested Unsloth fallback was therefore not started.

## Initial llama.cpp TP2 versus PP2 result

TP2 is the clear winner for the matched batch-one workload. Its median server-reported speed was **27.918 prompt tokens/s and 30.078 decode tokens/s**, versus **15.416 prompt tokens/s and 20.586 decode tokens/s** for PP2.

| Matched metric | TP2 | PP2 | TP2 advantage |
|---|---:|---:|---:|
| Server prompt rate, median | 27.918 tok/s | 15.416 tok/s | 81.1% higher |
| Server decode rate, median | 30.078 tok/s | 20.586 tok/s | 46.1% higher |
| Client time to first token, median | 18.342 s | 33.215 s | 44.8% lower |
| Client request latency, median | 22.598 s | 39.432 s | 42.7% lower |
| Client end-to-end total rate, median | 28.321 tok/s | 16.230 tok/s | 74.5% higher |
| Peak application-visible memory per GPU | 23,062 MiB | 22,856 MiB | TP2 used 206 MiB more |

The matched run used one warmup followed by five measured samples per topology, exactly 512 prompt tokens and 128 generated tokens per measured sample, concurrency one, and greedy sampling. Every measured sample produced the same output hash in both topologies.

The likely cause of the difference is pipeline utilization at batch one. TP2 kept both GPUs near 96% mean utilization while exchanging tensor payloads in both directions. PP2 alternated work between stages, resulting in 56.0% and 46.2% mean utilization and only the expected one-way stage traffic. This explanation is an inference from the utilization and NVLink counters; the rates themselves are measured.

## Initial optimization extension: 100 tok/s target exceeded

The optimization phase began at 02:29:55 EDT, with the requested six-hour fallback boundary at approximately 08:29:55. The final post-gate speed campaign completed at 04:12:36 EDT, so the 100 tok/s target was achieved more than four hours before the fallback boundary.

### Wide F8 kernel and microbatch sweep

The llama.cpp extension adds a dedicated wide F8 implementation in:

```text
vendor/llama.cpp/ggml/src/ggml-cuda/mmf-f8.cu
vendor/llama.cpp/ggml/src/ggml-cuda/mmf-f8.cuh
```

It consumes the GGUF's row-local BF16 scale plus 128 raw E4M3 bytes, tiles the wide matrix multiplication, and keeps weights FP8-resident. The standalone numerical harness exercised widths 16, 32, 64, 128, 256, and 512 on both GPUs. Maximum absolute error was no greater than `9.155e-5` in the wide tests. An additional K128 packed variant was also checked.

The new kernel was slower than cuBLAS on this model and SM86: approximately 210.8 prompt tok/s for the first implementation and 280.9 for the packed K128 revision, versus 748.5 for the cuBLAS control at microbatch 256. Consequently, the recommended llama.cpp launch sets:

```bash
export GGML_CUDA_F8_WIDE=0
```

This is an engineering result, not a hidden substitution: the custom F8 storage and one-token MMVQ path remain active, while wide prefill and multi-token verification intentionally use the faster cuBLAS route. The tiled source and numerical harness remain in the working tree for further optimization.

The required 32-through-512 sweep used batch 512, exact context 262,144, TP2, greedy sampling, and one screening sample per point:

| Microbatch | Server prompt rate | Server decode rate |
|---:|---:|---:|
| 32 | 244.040 tok/s | 31.453 tok/s |
| 64 | 406.763 tok/s | 31.184 tok/s |
| 128 | 568.043 tok/s | 31.201 tok/s |
| 256 | **748.528 tok/s** | 31.218 tok/s |
| 512 | 740.994 tok/s | **31.425 tok/s** |

Microbatch 256 is the prefill winner and was retained for the llama.cpp speculative sweep. These are single-sample screening measurements, not confidence intervals. Their receipts are `control-tp2-b512-ub{32,64,128,256,512}-bench.jsonl` in the optimization receipt root.

This resolves the earlier prompt-versus-decode anomaly. The original run forced batch and microbatch eight to remain on the narrow custom kernel and used verbose tracing, producing an artificially low 27.918 prompt tok/s. With an appropriate wide path, prompt processing was about 24 times faster than ordinary llama.cpp decode in the UB256 screen. The final real OpenCode session showed roughly 1.86–2.03 thousand prompt tok/s in SGLang prefill chunks, also comfortably faster than decode.

### llama.cpp bundled MTP and n-gram prediction

The source snapshot includes a one-layer Qwen MTP block in `mtp.safetensors`. It was reconverted into a single target-plus-MTP GGUF; the FP8 model matrices remain raw E4M3 storage, while the remaining two-dimensional BF16 endpoints were converted to Q8_0 to create enough exact-context headroom:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/Qwen3.8-27B-RawF8_E4M3-BF16Scale-Q8_0-Remaining2D-BundledMTP.gguf
bytes=27889309280
sha256=70d903b263a75fd81a1108c53799170fe6fa6f8556e3ce0e9b0fd161c741ff99
```

The validated artifact has 866 tensors, `qwen35.block_count=65`, and `qwen35.nextn_predict_layers=1`: 407 F8_E4M3 tensors, 99 Q8_0 tensors, and 360 F32 tensors. The full atomic conversion command, metadata assertions, per-tensor hashes, and source receipts are under `bundled-mtp-q8_0-conversion-receipts`.

The exact conversion is:

```bash
cd /root/exo/vendor/llama.cpp
env CUDA_VISIBLE_DEVICES= \
  PYTHONPATH=/root/exo/vendor/llama.cpp \
  /var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/bin/python \
  convert_hf_to_gguf.py \
  /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --outfile /mnt/sanic/qwen3.8-27b-fp8-sm86-opt/Qwen3.8-27B-RawF8_E4M3-BF16Scale-Q8_0-Remaining2D-BundledMTP.gguf \
  --outtype q8_0 \
  --fp8-storage-sm86
```

Do not add `--no-mtp`, which omits the drafter, or `--mtp`, which produces an MTP-only sidecar. In bundled mode, do not pass a separate draft model or device; llama.cpp creates the draft context over the same loaded model.

The screening curve below used cuBLAS for multi-token verification, Q8_0 target and draft KV, exact context 262,144, and a single measured sample at each point:

| Speculation | Server decode rate |
|---|---:|
| Non-speculative bundled GGUF | 31.695 tok/s |
| MTP K4 | 39.530 tok/s |
| MTP K7 | 52.510 tok/s |
| MTP K12 | 67.778 tok/s |
| MTP K16 | 72.558 tok/s |
| MTP K24 | 73.972 tok/s |
| ngram-mod 24 then MTP K24 | **88.705 tok/s** |
| ngram-mod 32 then MTP K32 | 86.190 tok/s |

The K32 regression locates the useful horizon for this prompt. The localized metadata arena multiplier in `ggml-backend-meta.cpp` was increased from 16 to 64 so large speculative graphs no longer overrun the rotating meta context; a 4,097-object host smoke passed. A separate Ampere-only M=1 packed MMVQ experiment is available behind `GGML_CUDA_F8_MMVQ_PACKED=1`. Static SM86 compilation reduced register counts, and exhaustive E4M3 plus 100,000 random CPU dot checks passed, but no final GPU A/B was run; it is therefore not part of the recommended configuration.

### Why the initial >100 tok/s backend changed

The parallel architecture investigation found that the authoritative vendored SGLang branch already contains a mature Ampere FP8-storage Marlin kernel for this checkpoint's exact 128-by-128 block shape. On SM86 it repacks raw E4M3 weights, folds scale bias, decodes into FP16/BF16 fragments, and uses tensor-core W8A16/HMMA tiles for both M=1 and wide GEMM. This is a closer fit than continuing to incrementally optimize the first llama.cpp tile.

SGLang can also use Qwen's native bundled MTP and ReplaySSM, avoiding repeated full recurrent-state snapshots during speculative verification. The initial >100 tok/s result therefore remained an Exo-checkout vendored-fork result, but changed from llama.cpp to SGLang. It still used both RTX 3090s in TP2 and kept inference weights, KV, recurrent state, and compute on GPU; CPU work was limited to normal process, HTTP, and tokenization orchestration. The later promoted SGLang tree configuration is documented separately at the end and does not use ReplaySSM.

The exact runtime is:

```text
/var/lib/exo/runtimes/dsv4-fwuff-cu130/venv/bin/python
torch 2.11.0+cu130
flashinfer-python 0.6.15.post1
sglang-kernel 0.4.5
triton 3.6.0
```

The SGLang changes used by the initial >100 tok/s configuration are part of the cumulative uncommitted patch on the authoritative `exo/dsv4-cumulative-0801` branch at base commit `0dfb8cbdaa314c2be75ee2d06ed955e1e8c5866e`. They:

- share the draft embedding and LM head before target KV profiling, then force Python and CUDA allocator reclamation after the final module rebind;
- make `--language-only` skip the unused vision tower;
- propagate `num_nextn_predict_layers=1` onto the nested Qwen text configuration, fixing a 64-times draft-layer memory-accounting error;
- provide an environment-gated normalization fallback for the installed FlashInfer combination;
- update both MTP draft graph runners to the cumulative branch's current capture API.

The cumulative base branch itself supplies the initial configuration's FP32 ReplaySSM checkpoint and small accepted-input replay ring; that machinery is not attributed to the campaign patch.

The same patch also adds static compatibility for the exact `RadixArk/Qwen3.8-27B-DSpark` checkpoint, but DSpark was not downloaded or run after native MTP crossed the target.

The installed FlashInfer package referenced a 32-CTA SM86 batch-prefill instantiation that its shipped template did not emit below head dimension 512. The durable overlay adds that missing instantiation. It is an exact copy of the overlay used during the successful run:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/flashinfer-sm86-overlay
size=94 MiB
paged-template-sha256=5ecda8a3f661abd7006eba8980395310aa8943c8b74e4bd1b51da3a5e59cc274
ragged-template-sha256=41795e3821670d1509cc575a68fba9fdb3bb4750a05fb4228ce581b809424d7a
```

The two modified templates, their stock-package hashes, and reconstruction instructions are also preserved inside the final extension receipt under `flashinfer-overlay/`, so the source delta does not depend on the external overlay directory remaining in place.

### Exact initial >100 tok/s SGLang launch

This command uses the durable overlay rather than the original equivalent `/tmp` copy. Verify first that CUDA devices 0 and 1 correspond to the UUID order in the hardware table above.

```bash
cd /root/exo/vendor/sglang
mkdir -p \
  /tmp/qwen38-sgl-xdg \
  /tmp/qwen38-sgl-torch-fixed \
  /tmp/qwen38-sgl-triton \
  /tmp/qwen38-sgl-flashinfer-fixed

model=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a

env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONPATH=/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/flashinfer-sm86-overlay:/root/exo/vendor/sglang/python \
  XDG_CACHE_HOME=/tmp/qwen38-sgl-xdg \
  TORCH_EXTENSIONS_DIR=/tmp/qwen38-sgl-torch-fixed \
  TRITON_CACHE_DIR=/tmp/qwen38-sgl-triton \
  FLASHINFER_WORKSPACE_BASE=/tmp/qwen38-sgl-flashinfer-fixed \
  SGLANG_FORCE_FP8_MARLIN=1 \
  SGLANG_DISABLE_FLASHINFER_NORM=1 \
  NCCL_P2P_LEVEL=NVL \
  PYTHONUNBUFFERED=1 \
  /var/lib/exo/runtimes/dsv4-fwuff-cu130/venv/bin/python \
  -m sglang.launch_server \
  --model-path "$model" \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 30022 \
  --tp-size 2 \
  --dtype bfloat16 \
  --quantization fp8 \
  --context-length 262144 \
  --max-total-tokens 262144 \
  --max-running-requests 1 \
  --max-mamba-cache-size 5 \
  --disable-radix-cache \
  --chunked-prefill-size 512 \
  --max-prefill-tokens 512 \
  --mem-fraction-static 0.96 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer \
  --linear-attn-backend triton \
  --language-only \
  --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 \
  --cuda-graph-bs-decode 1 \
  --page-size 1 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$model" \
  --speculative-num-steps 7 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 8 \
  --enable-gdn-replayssm-spec \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --log-level info
```

At top-k one, SGLang enforces `draft_tokens = steps + 1`; the paired 7/8 values are intentional. Do not change only one of them. `--cuda-graph-bs-decode 1` is the request batch size, not the speculative token width.

The startup gate is strict. Do not benchmark unless startup proves all of the following:

```text
FP8 Marlin selected for SM86
target KV pool #tokens = 262144
draft KV pool #tokens = 262144
target K cache = 2.00 GiB/rank and V cache = 2.00 GiB/rank
draft K cache = 0.13 GiB/rank and V cache = 0.13 GiB/rank
ReplaySSM record_len = 8
max_total_num_tokens = 262144
```

The successful launch retained about 2.98 GiB per rank after CUDA graph capture. A lower token count is not an exact-256K result.

The durable startup receipt is `sglang-mtp-d8-nocache-final-startup.log`, SHA-256 `fe0c97cbca43290ca4e1584872ed751374e43b152edec208ba775dc2e4688ba2`; the recovered exact launch record is `sglang-mtp-d8-nocache-final-launch-command.txt`, SHA-256 `531d3ded0383e9babe228494f68822e6e71b44ff9291a7b5217ec1a1f6c945ca`. They contain the resolved arguments and every allocation line above. The campaign proves that both 262,144-token pools were resident and usable for the tested requests; it did not submit a request whose occupied context itself reached 262,144 tokens.

`--disable-radix-cache` is part of the final correctness configuration. With the radix cache enabled, target-only greedy output changed after unrelated requests and then stabilized on immediate repeats. Cache-disabled target-only hashes were stable across a code → systems → review → code sequence. The cache-enabled D8 screen reached 133.852 tok/s, but that result is excluded from the recommendation because of the history-dependent hashes.

### Initial >100 tok/s coherence gates

This initial candidate repeated the required order: simple deterministic gate, real OpenCode tool-use gate as the `opencode` user, then its authoritative speed run.

Gate 1 command:

```bash
cd /root/exo
python scripts/benchmark_sglang_openai.py \
  --endpoint http://127.0.0.1:30022/v1/chat/completions \
  --model Qwen3.8-27B-FP8 \
  --prompt 'Return exactly four lines and no other text: PRODUCT=17*19 evaluated as an integer; NEXT=the next two Fibonacci numbers after 5,8; REVERSE=the word stressed reversed; CHECK=D8_FINAL_COHERENT.' \
  --max-tokens 96 \
  --warmups 0 \
  --samples 1 \
  --timeout-seconds 120 \
  --disable-thinking \
  --show-content \
  --output-jsonl /mnt/sanic/qwen3.8-27b-fp8-sm86-opt/sglang-d8-nocache-final-gate1.jsonl
```

It passed exactly:

```text
PRODUCT=323
NEXT=13,21
REVERSE=desserts
CHECK=D8_FINAL_COHERENT
```

Gate 1 receipt SHA-256 is `66ae0c23a549c5c948540436cf03f85d0c50e2222912585bb2b6da72c758d8cb`.

Prepare and run Gate 2:

```bash
install -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86-opt/qwen38-sglang-final-opencode.json \
  /tmp/qwen38-sglang-final-opencode.json
install -o opencode -g opencode -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86-opt/qwen38-sglang-final-opencode-fixture.txt \
  /home/opencode/workspace/qwen38-tp2-final-deployments.txt

cd /home/opencode/workspace
runuser -u opencode -- env \
  HOME=/home/opencode \
  NO_COLOR=1 \
  OPENCODE_CONFIG=/tmp/qwen38-sglang-final-opencode.json \
  /home/opencode/.opencode/bin/opencode run \
  --pure \
  --format json \
  --model local-qwen/Qwen3.8-27B-FP8 \
  --title qwen38-d8-nocache-final-coherence \
  'Use the read tool to inspect /home/opencode/workspace/qwen38-tp2-final-deployments.txt and do not answer until you consume the tool result. Review every row against the policy in the file. Identify every production violation with its exact field, ignore nonproduction rows when counting offenders, compute the sum of offender ports, and give a concise remediation for each offender. Do not modify files.'
```

Session `ses_ffb85fb5cffeIzsOpkCAM1tYG2` executed the structured `read` tool, consumed the exact ten-line fixture, correctly ignored the staging `audit` row, and reported:

- `billing`: `retries=1`, raise to at least 3;
- `search`: `timeout_ms=7000`, lower to at most 5000;
- `ingest`: `owner=UNASSIGNED`, assign a named owner;
- offender port sum `8222 + 8333 + 8444 = 24999`.

The archived OpenCode config SHA-256 is `f7e458bae3d3c0bd6327f14bd444f633ff2b28b3bc84b0164ad71d5b01c23912`; the fixture SHA-256 is `05b1f54ccaa053431fe34b4038a93d4722791460256bc902b4f285e81b2a0679`. The completed session was exported after the run to `qwen38-sglang-final-opencode-transcript.json`, SHA-256 `fe4b7b018e7e843438693f92cd56504031d1645e36fceadd2098c1156731e2ca`. That export contains the completed structured `read` result and final response.

### Initial >100 tok/s post-gate speed command and result

Run speed only after both gates pass:

```bash
cd /root/exo
python scripts/benchmark_sglang_openai.py \
  --endpoint http://127.0.0.1:30022/v1/chat/completions \
  --model Qwen3.8-27B-FP8 \
  --prompt 'Write a complete production-quality Python implementation of an asynchronous bounded worker pool. Include type hints, graceful cancellation, backpressure, structured error collection, docstrings, and a short executable example. Return only the code in one fenced block. Be thorough enough to use the full response budget.' \
  --max-tokens 512 \
  --warmups 1 \
  --samples 5 \
  --timeout-seconds 120 \
  --disable-thinking \
  --output-jsonl /mnt/sanic/qwen3.8-27b-fp8-sm86-opt/sglang-mtp-d8-nocache-post-gates-speed.jsonl
```

The client issues non-streaming greedy chat requests and computes `completion_tokens / complete HTTP wall time`, so the reported figure includes the short 68-token prefill and HTTP overhead. It is a conservative lower bound on steady decode rather than SGLang's internal pure-decode timer.

| Sample | Completion rate | Wall time | Output SHA-256 |
|---:|---:|---:|---|
| 1 | 107.492 tok/s | 4.7631 s | `0c50ee92...57ceeb` |
| 2 | 106.676 tok/s | 4.7996 s | `0c50ee92...57ceeb` |
| 3 | 108.256 tok/s | 4.7295 s | `0c50ee92...57ceeb` |
| 4 | 105.914 tok/s | 4.8341 s | `0c50ee92...57ceeb` |
| 5 | 108.155 tok/s | 4.7339 s | `0c50ee92...57ceeb` |
| **Mean** | **107.299 tok/s** | **4.7721 s** | identical |
| **Median** | **107.492 tok/s** | **4.7631 s** | identical |

All five responses were nonempty, stopped at the 512-token length limit, and had the same full output SHA-256 `0c50ee92a1251c1a9b581652225747798dbe375f931ff845220d51f2cf57ceeb`. The receipt SHA-256 is `74138ec4faadf7c847b0fc751da3437ef94c7052c17551dd304f8a43c9e08f45`.

The supporting benchmark client is `scripts/benchmark_sglang_openai.py`; its archived SHA-256 is `24e9a30b101618198dc9187ae297f3c5bb1d0bacb8f1b1181268b12a79f9f5aa`.

### Initial-campaign source validation

- llama.cpp `llama-server` and `test-quantize-fns` rebuilt successfully from the final dirty tree; `test-quantize-fns` passed through `f8_e4m3`.
- The focused SGLang regression set for early endpoint sharing, Qwen MTP layer accounting, language-only materialization, DSpark registration, Triton verify width, and ReplaySSM/TP synchronization passed: 25 tests plus 4 subtests.
- The SGLang precommit-focused Ruff selection `F401,F821,UP037` passed on every changed Python file.
- The benchmark client passed Python byte compilation, Ruff, and BasedPyright with zero errors or warnings.
- Root, llama.cpp, and SGLang `git diff --check` all passed.

### Initial >100 tok/s correctness boundary and fallback decision

Greedy speculative output was stable within and across cache-disabled runs but was not token-identical to the cache-disabled non-speculative target on the three checked prompts. The stable code hashes were:

```text
target-only: 2138d165...
D8 MTP:      0c50ee92...
```

This is consistent with the known risk class for quantized multi-token verification: a different batched numerical path can alter an argmax near a logit tie. Therefore the final candidate is **semantic-coherence approved, not lossless speculative decoding**. Deployments that require exact greedy token identity should keep speculation disabled and accept roughly 51 tok/s on this setup until the verifier discrepancy is fixed.

The server also logged that this checkpoint provides no FP8 KV scaling factors, so SGLang used scale 1.0 for the required FP8 KV cache. FP8 KV was necessary to fit the exact 262,144-token pools with MTP on 24 GiB cards. Both coherence gates passed, but this is an additional accuracy boundary to retain in any deployment decision.

The measured 100 tok/s goal was workload-specific but genuine: every post-gate code sample exceeded it, the two required coherence gates passed, exact context remained allocated, and the server had no cross-request radix history. The Unsloth/Q4 fallback was not started because the target was reached before the six-hour deadline. The exact Qwen3.8 DSpark checkpoint and TP2 compatibility path remain the most promising next experiment if broader workloads must also clear 100 tok/s.

### Optimization artifact index

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/
├── llama-optimization-final.patch
├── llama-untracked-source.tar.gz
├── sglang-optimization-final.patch
├── sglang-untracked-tests.tar.gz
├── root-untracked-harness.tar.gz
├── flashinfer-sm86-overlay/
├── benchmark_sglang_openai.py
├── Qwen3.8-27B-RawF8_E4M3-BF16Scale-Q8_0-Remaining2D-BundledMTP.gguf
├── bundled-mtp-q8_0-conversion-receipts/
├── control-tp2-b512-ub*-bench.jsonl
├── mtp-*-bench.jsonl
├── ngram*-bench.jsonl
├── sglang-d8-nocache-final-gate1.jsonl
├── sglang-mtp-d8-nocache-final-launch-command.txt
├── sglang-mtp-d8-nocache-final-startup.log
├── qwen38-sglang-final-opencode.json
├── qwen38-sglang-final-opencode-fixture.txt
├── qwen38-sglang-final-opencode-transcript.json
└── sglang-mtp-d8-nocache-post-gates-speed.jsonl
```

`git diff --binary` does not include untracked files. The two `*-optimization-final.patch` files are therefore explicitly the tracked working-tree diffs, not complete standalone source bundles. Reconstruct the exact tree by applying the tracked diff to its documented base commit and then extracting the corresponding supplemental archive at the submodule root. The root harness archive is validation source and is not needed by either server at runtime.

Tracked-diff and supplement identities:

```text
llama-optimization-final.patch
sha256=c3543896e608db15e5f41db49f0871b30a555e5ab5849f370073f88eede42c9e

llama-untracked-source.tar.gz
sha256=328f5e6b93c079f68c13ccb4ef7b8f0e709542fc141d57155077bcec77a65bfc
contains=ggml/src/ggml-cuda/mmf-f8.cu, ggml/src/ggml-cuda/mmf-f8.cuh

sglang-optimization-final.patch
sha256=dbce86e70df550d6129db9dda7725601725e59f1cf7816e9bd8a363e49f7949b

sglang-untracked-tests.tar.gz
sha256=000245f74025d63f905170fb9bc7376edfdd416d2516c3343ff6b6507217ffc2

root-untracked-harness.tar.gz
sha256=7db11985287c6c65f573a8af4cfaf490ea168382ab9525e619f3cee2da22ffbe
```

## Measurement scope

The initial TP2/PP2 measurements are direct `llama-server` measurements built from this checkout's authoritative vendored llama.cpp fork. The optimized winning measurements are direct `sglang.launch_server` measurements built from this checkout's authoritative vendored SGLang fork. Neither is a stock-upstream result.

The timed HTTP paths did **not** run through `uv run exo`, the Exo router, or the Exo API adapter. The server binaries were invoked directly to isolate and prove their modified inference backends. No active llama.cpp worker integration suitable for this model and topology was found in the root application. These should therefore be called **Exo-checkout vendored-backend results**, not Exo-router end-to-end results.

## Version and artifact identity

- Root checkout: `/root/exo`
- Root commit: `d7daa2482a197b1f5e8a479b3c48e94cd0374d77`
- Root branch during the run: `agent/linux-cuda-nccl`
- Authoritative llama.cpp branch required by `AGENTS.md`: `exo/kimi-k3-cumulative`
- llama.cpp base commit: `0c2743950c8dc10fc80791ecd0727e51a14c7aad`
- llama.cpp build number: `10170`; server fingerprint: `b10170-0c2743950`
- The SM86 FP8 work is an uncommitted working-tree patch on that base. It was not silently based on an older model-specific branch.
- Exact llama.cpp patch: `/mnt/sanic/qwen3.8-27b-fp8-sm86/llama-final.patch`
- Patch SHA-256: `30e41a2d84361bd81d0cce15caf38ee956cba3284f397d2c0a1ba3279eef794a`
- Server binary SHA-256: `8e3d864d76ad6baefb1f0be92efcd843cd757fe1aacfcf7ac67f9ebdf792b199`
- CUDA library SHA-256: `4e95660e3f210e493d7564dd7806dad8522a871d9646f1f8c7282c531250a6ae`
- Full binary/library hash list: `/mnt/sanic/qwen3.8-27b-fp8-sm86/final-runtime-sha256.txt`

The authoritative patch modifies 25 files across the converter, GGUF/GGML type definitions, CPU traits, model loader, endpoint placement, and CUDA conversion/getrows/MMVQ path. The working tree remains intentionally dirty; the patch plus base commit is the exact source receipt.

## Hardware

| Logical device in the commands | Physical GPU | UUID | PCI address | NUMA |
|---|---|---|---|---:|
| CUDA0 | NVIDIA GeForce RTX 3090 | `GPU-a442b72e-6727-6322-ba5d-5a9512b79886` | `00000000:16:00.0` | 0 |
| CUDA1 | NVIDIA GeForce RTX 3090 | `GPU-63a7760a-6164-0758-9228-03dbf35d721c` | `00000000:D8:00.0` | 1 |

- Driver: `610.43.03`
- Application-visible memory: 24,123 MiB per GPU
- Interconnect: four-link `NV4`
- The initial llama.cpp commands pin physical UUID order through `CUDA_VISIBLE_DEVICES`. The final SGLang command used numeric `0,1` only after verifying that it resolved to the same order shown above.
- GPU telemetry was sampled every 100 ms during the measured interval.

The full topology and `nvidia-smi -q` receipts are `final-topology.txt` and `final-nvidia-smi-q.txt` in the receipt root.

## Model identity and conversion

- Hugging Face model: `Qwen/Qwen3.8-27B-FP8`
- Exact revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Immutable source snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- Architecture: Qwen3.5, 64 layers, 48 recurrent/Gated Delta Net layers and 16 full-attention layers, hidden size 5120, FFN size 17408, 24 attention heads, 4 KV heads, head dimension 256.
- Declared maximum context: 262,144 tokens.
- Source quantization: E4M3 weights with BF16 inverse scales in 128-by-128 blocks.
- The Hugging Face cache was read but not modified.

Converted artifact:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86/Qwen3.8-27B-F8_E4M3-BF16Scale.gguf
bytes=29861427904
sha256=28dd79e50f2df2dc2b0e3eff09224e3af5d21492bde179269f8c6c9b0401c9a7
```

The GGUF contains 851 tensors: 400 `F8_E4M3`, 98 BF16, and 353 F32. Its llama file type is 42. The new storage block is K=128 with a bit-exact two-byte BF16 scale followed by 128 untouched E4M3FN bytes, for 130 bytes per block.

Exact conversion command:

```bash
cd /root/exo/vendor/llama.cpp
env PYTHONPATH=/root/exo/vendor/llama.cpp \
  /var/lib/exo/runtimes/glm47-kt/a4b0c45-sgl449c59f-py312-dwagon/venv/bin/python \
  convert_hf_to_gguf.py \
  /root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --outfile /mnt/sanic/qwen3.8-27b-fp8-sm86/Qwen3.8-27B-F8_E4M3-BF16Scale.gguf \
  --outtype bf16 \
  --no-mtp \
  --fp8-storage-sm86
```

`--fp8-storage-sm86` is mutually exclusive with the fork's pre-existing `--fp8-as-q8`. It validates FP8 plus 128-by-128 scale metadata, preserves raw FP8 bytes and exact BF16 scale bits, handles Qwen V-head permutation, and handles this snapshot's unusual shard names. `--outtype bf16` controls the non-FP8 residual tensors; it does not materialize the retained FP8 weights as BF16.

## FP8 kernel provenance and exact claim

The design source supplied for this task is [Adhitya Mohan's “FP8 as Storage, IMMA as Compute on Ampere” article](https://amohan.dev/blog/2026/fp8-as-storage-imma-ampere/). Ampere SM86 has no native FP8 MMA. The article treats FP8 as raw E4M3 bytes, decodes with a 256-entry LUT, applies scales, and explores INT8/IMMA compute.

The article's own RTX 3090 Ti 4096-cubed benchmark reported 2.914 ms for its fused extension, 2.267 ms for naive decode plus FP16 matmul, and 1.828 ms for cached FP16. Thus its literal INT8/IMMA route was slower in that experiment.

The local lineage was identified in the SGLang commit `2a58a0802ef4dddb90c5f91f1d4677cab28fc99f` (`dsv4 flash`), especially:

```text
python/sglang/kernels/ops/attention/dsv4/fp8_storage_indexer.py
python/sglang/srt/mem_cache/dsv4_kv_cache_dtype.py
```

That prior code is a DSV4 KV/indexer path, not a Qwen weight GEMM. The llama.cpp work therefore ports the proven storage/decode contract to model weights:

- a new raw `F8_E4M3` GGML/GGUF type;
- exact E4M3FN plus BF16-K128 decoding;
- GPU conversion and getrows support;
- a custom CUDA MMVQ path for one through eight columns, using Q8_1 activations while decoding resident raw E4M3FN weights on load;
- CPU reference traits for validation and non-serving utilities;
- a one-time runtime marker proving selection.

The marker in both final server logs is:

```text
SM86 FP8-storage MMVQ active: raw E4M3FN bytes + BF16 K128 scales, on-load floating decode
```

This is **FP8 storage plus on-load floating decode**, not native FP8 arithmetic and not a claim that the article's literal INT8/IMMA kernel ran. The weights remain FP8-resident in VRAM; no whole-model BF16/F16 or Q8_0 fallback was used.

## Build and validation

Exact initial configuration and build commands:

```bash
cd /root/exo/vendor/llama.cpp
cmake -S . -B build-qwen-f8-sm86 -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-qwen-f8-sm86 \
  --target llama-server llama-bench test-quantize-fns -j 8
```

After the final placement changes, the complete tree was rebuilt with:

```bash
cmake --build build-qwen-f8-sm86 --parallel 32
```

The captured CMake cache proves:

```text
CMAKE_BUILD_TYPE=Release
CMAKE_CUDA_ARCHITECTURES=86
GGML_CUDA_GRAPHS=ON
GGML_CUDA_NCCL=ON
NCCL_LIBRARY=/usr/local/lib/libnccl.so
LLAMA_BUILD_SERVER=ON
LLAMA_BUILD_TESTS=ON
LLAMA_CURL=OFF
```

`ldd` also resolves `libnccl.so.2`. This fork selects its NCCL path by default when compiled with NCCL, and neither final log contains the fallback warning. The logs do not print the communicator implementation name, so this is strong source/build plus negative-log evidence, not a direct NCCL communicator-name receipt.

Final validation passed:

- complete rebuild;
- `test-quantize-fns`, including `f8_e4m3`;
- Python syntax compilation for the converter sources;
- `git diff --check`;
- all 256 E4M3FN byte values checked against PyTorch during development;
- exact BF16 scale-bit checks;
- standalone CUDA MMVQ numerical tests at one and eight columns on each physical RTX 3090.

Each GPU independently produced:

```text
ncols=1 max_abs_error=0.000366211 max_reference=626.637
ncols=8 max_abs_error=0.00128174 max_reference=1534.48
PASS: F8_E4M3 CUDA MMVQ ncols 1 and 8
```

The fail-closed validation receipt is `/mnt/sanic/qwen3.8-27b-fp8-sm86/final-validation-pass.txt`. The standalone source and binary are archived as `f8_mmvq_test.cpp` and `f8_mmvq_test` in the same directory.

## Exact server commands

Run the campaigns sequentially. Do not keep TP2 and PP2 alive at the same time.

### TP2

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-a442b72e-6727-6322-ba5d-5a9512b79886,GPU-63a7760a-6164-0758-9228-03dbf35d721c \
  GGML_CUDA_P2P=1 \
  /root/exo/vendor/llama.cpp/build-qwen-f8-sm86/bin/llama-server \
  -m /mnt/sanic/qwen3.8-27b-fp8-sm86/Qwen3.8-27B-F8_E4M3-BF16Scale.gguf \
  --alias Qwen3.8-27B-FP8-SM86 \
  --host 127.0.0.1 \
  --port 30020 \
  -c 262144 \
  -np 1 \
  -b 8 \
  -ub 8 \
  --device CUDA0,CUDA1 \
  --split-mode tensor \
  --tensor-split 1,1 \
  -ngl all \
  --fit off \
  -fa on \
  -ctk f16 \
  -ctv f16 \
  --load-mode mmap \
  --threads 32 \
  --threads-batch 32 \
  --check-tensors \
  --perf \
  --metrics \
  --reasoning-format deepseek \
  --reasoning auto \
  --offline \
  --no-webui \
  --override-tensor '^token_embd\.weight$=CUDA0,^output\.weight$=CUDA1' \
  --verbosity 4 \
  --log-timestamps \
  --log-colors off \
  2>&1 | tee /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/server-final.log
```

### PP2

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-a442b72e-6727-6322-ba5d-5a9512b79886,GPU-63a7760a-6164-0758-9228-03dbf35d721c \
  GGML_CUDA_P2P=1 \
  /root/exo/vendor/llama.cpp/build-qwen-f8-sm86/bin/llama-server \
  -m /mnt/sanic/qwen3.8-27b-fp8-sm86/Qwen3.8-27B-F8_E4M3-BF16Scale.gguf \
  --alias Qwen3.8-27B-FP8-SM86 \
  --host 127.0.0.1 \
  --port 30021 \
  -c 262144 \
  -np 1 \
  -b 8 \
  -ub 8 \
  --device CUDA0,CUDA1 \
  --split-mode layer \
  --tensor-split 32,33 \
  -ngl all \
  --fit off \
  -fa on \
  -ctk f16 \
  -ctv f16 \
  --load-mode mmap \
  --threads 32 \
  --threads-batch 32 \
  --check-tensors \
  --perf \
  --metrics \
  --reasoning-format deepseek \
  --reasoning auto \
  --offline \
  --no-webui \
  --override-tensor '^token_embd\.weight$=CUDA0,^output\.weight$=CUDA1' \
  --verbosity 4 \
  --log-timestamps \
  --log-colors off \
  2>&1 | tee /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/server-final.log
```

`32,33` is intentional. This loader's split denominator includes the output pseudo-layer; the result is exactly text layers 0-31 on CUDA0 and 32-63 on CUDA1, with the embedding on CUDA0 and output on CUDA1.

### Flag rationale

- `-c 262144`: exact required context, with one slot receiving the full capacity.
- `-np 1`: request concurrency one and no context division between slots.
- `-b 8 -ub 8`: keeps prompt and decode matmuls within the custom MMVQ kernel's validated one-to-eight-column domain. This prioritizes proof of the requested path over larger-batch prefill throughput.
- `-fa on`: flash attention is explicitly enabled.
- `-ctk f16 -ctv f16`: stable F16 KV storage. Only 16 of the 64 hybrid layers use attention KV, but the exact 262,144-token cache is still 16 GiB logical.
- `-ngl all --fit off`: fail-closed all-layer GPU placement; no automatic memory-fit reduction.
- Explicit endpoint overrides: prevent the embedding/output endpoints from landing on host memory and make the PP boundary unambiguous.
- `--load-mode mmap`: maps the GGUF file without `mlock`; it does not place model operators on the CPU.
- `GGML_CUDA_P2P=1`: explicit peer transfers. The build has CUDA graphs and NCCL enabled.
- `--offline --no-webui`: no download, browser UI, or unrelated service work.
- Text-only model serving: no multimodal projector or vision weights were loaded.
- `--threads 32 --threads-batch 32`: CPU work is limited to orchestration/tokenization. No `numactl` or CPU affinity binding was applied; the physical GPU order was UUID-pinned.
- `--verbosity 4`: this was the lowest useful trace level for exact layer assignment and kernel evidence in this fork. It emits microbatch progress and may slightly depress absolute throughput. It was matched for both topologies.

## GPU-only placement and 262,144-token allocation proof

| Runtime fact | TP2 | PP2 |
|---|---|---|
| Repeating-layer assignment | layers 0-63 on distributed `Meta()` tensor device | layers 0-31 CUDA0; 32-63 CUDA1 |
| Endpoints | embedding CUDA0; output CUDA1 | embedding CUDA0; output CUDA1 |
| Loader result | 65/65 layers offloaded to GPU | 65/65 layers offloaded to GPU |
| Weight buffers | CUDA0 2,425.00 MiB; CUDA1 2,425.02 MiB; distributed `Meta()` 11,810.07 MiB | CUDA0 14,233.79 MiB; CUDA1 14,233.81 MiB |
| KV allocation | distributed `Meta()`; 16,384 MiB logical | CUDA0 8,192 MiB; CUDA1 8,192 MiB; 16,384 MiB logical |
| Recurrent state | 149.62 MiB logical | 74.81 MiB per GPU; 149.62 MiB logical |
| Pipeline log | graph splits 3 | `pipeline parallelism enabled`; graph splits 2 |
| Ready process memory | 23,016 MiB/GPU | 22,834 MiB/GPU |
| Peak measured memory | 23,062 MiB/GPU | 22,856 MiB/GPU |

In TP mode, `Meta()` is llama.cpp's distributed multi-backend device over the two CUDA children, not a CPU backend. The per-GPU physical memory agrees with each GPU holding its endpoint shard plus the distributed weight/KV/state shards. Neither final log contains `CPU model buffer`, an OOM, a fallback marker, NaN, or fatal error. Small `CUDA_Host` output/compute staging buffers of 0.95 MiB plus 4.16 MiB for TP2 or 16.16 MiB for PP2 are host orchestration buffers, not CPU model operators.

Both final logs prove `n_ctx=262144`, `n_ctx_seq=262144`, `n_ctx_slot=262144`, and the full logical 16 GiB F16 KV allocation. Peak headroom relative to the 24,123 MiB application-visible total was 1,061 MiB/GPU for TP2 and 1,267 MiB/GPU for PP2.

This proves full-context **capacity and allocation**, not a full-context request. The longest active OpenCode prompt was 7,909 tokens on TP2 and 7,913 on PP2. No request filled all 262,144 slots, so full-context numerical coherence and latency were not exercised.

## Required gate sequence and results

The order actually executed was:

1. TP2 simple coherency gate.
2. TP2 real OpenCode gate as the `opencode` user.
3. TP2 warmup and speed samples.
4. Stop TP2 and start PP2.
5. PP2 simple coherency gate.
6. PP2 real OpenCode gate as the `opencode` user.
7. PP2 warmup and speed samples.
8. Stop PP2.

### Gate 1: deterministic chat coherence

The exact request JSON files are:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/gate1-request.json
/mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/gate1-request.json
```

The request used temperature zero, seed 424242, `max_tokens=64`, thinking disabled, and required multiplication, sequence continuation, string reversal, and an exact topology marker. The actual run used byte-identical request files in `/tmp` and copied them into the receipt directories before sending. The following rerun calls consume those archived payloads directly:

```bash
curl --fail-with-body --silent --show-error \
  -D /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/gate1-headers.txt \
  -o /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/gate1-response.json \
  -w 'http_code=%{http_code}\ntime_starttransfer=%{time_starttransfer}\ntime_total=%{time_total}\n' \
  -H 'Content-Type: application/json' \
  --data-binary @/mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/gate1-request.json \
  http://127.0.0.1:30020/v1/chat/completions

curl --fail-with-body --silent --show-error \
  -D /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/gate1-headers.txt \
  -o /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/gate1-response.json \
  -w 'http_code=%{http_code}\ntime_starttransfer=%{time_starttransfer}\ntime_total=%{time_total}\n' \
  -H 'Content-Type: application/json' \
  --data-binary @/mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/gate1-request.json \
  http://127.0.0.1:30021/v1/chat/completions
```

TP2 returned HTTP 200 with `finish_reason=stop`:

```text
PRODUCT=323
NEXT=13,21
REVERSE=stressed
CHECK=TP2_COHERENT
```

TP2 Gate 1 server timing was 109 prompt tokens in 4,144.943 ms, or 26.297 tok/s, and 28 generated tokens in 886.834 ms, or 31.573 tok/s.

PP2 returned HTTP 200 with `finish_reason=stop`:

```text
PRODUCT=323
NEXT=13,21
REVERSE=stressed
CHECK=PP2_COHERENT
```

PP2 Gate 1 server timing was 109 prompt tokens in 6,882.702 ms, or 15.837 tok/s, and 28 generated tokens in 1,314.819 ms, or 21.296 tok/s.

Both responses and matching timing/header receipts are preserved in the topology directories.

### Gate 2: real OpenCode tool-use coherence

- OpenCode version: `1.18.15`
- Actual account: `uid=30033(opencode) gid=30033(opencode)`
- Working directory: `/home/opencode/workspace`
- Provider: `@ai-sdk/openai-compatible`
- Served model: `Qwen3.8-27B-FP8-SM86`
- The persistent OpenCode configuration was not edited. Each run used a topology-specific temporary file through `OPENCODE_CONFIG`; copies are in each final receipt directory.
- Sharing was disabled and `--pure` suppressed external plugins. The prompt prohibited edits, and the event stream contains only the required `read` tool call before the final answer.

To stage byte-identical copies from the receipts without changing the persistent OpenCode configuration:

```bash
install -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/opencode-config.json \
  /tmp/qwen38-tp2-final-opencode-config.json
install -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/opencode-config.json \
  /tmp/qwen38-pp2-final-opencode-config.json
install -o opencode -g opencode -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/opencode-fixture.txt \
  /home/opencode/workspace/qwen38-tp2-final-deployments.txt
install -o opencode -g opencode -m 0644 \
  /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/opencode-fixture.txt \
  /home/opencode/workspace/qwen38-pp2-final-deployments.txt
```

The exact TP2 model invocation inside the timestamp/status capture wrapper was:

```bash
runuser -u opencode -- env \
  HOME=/home/opencode \
  OPENCODE_CONFIG=/tmp/qwen38-tp2-final-opencode-config.json \
  PATH=/home/opencode/.opencode/bin:/usr/local/bin:/usr/bin:/bin \
  /home/opencode/.opencode/bin/opencode --pure run \
  --format json \
  --print-logs \
  --log-level INFO \
  --model local-qwen/qwen3.8-27b-fp8-sm86 \
  --variant low \
  --title qwen38-tp2-final-gate2 \
  'Use the read tool to inspect /home/opencode/workspace/qwen38-tp2-final-deployments.txt; do not answer until you have actually consumed the tool result. Review only the env=production rows against the policy stated in the file. Identify every offending service and its exact reason, compute the sum of the offender ports, and separately cite the staging audit timeout as evidence that you distinguished staging from production. Return exactly four lines in this format: OFFENDERS=<comma-separated names in file order>\nREASONS=<semicolon-separated exact violations in file order>\nPORT_SUM=<integer>\nSTAGING_EVIDENCE=<service and timeout_ms>.'
```

The exact PP2 model invocation inside the timestamp/status capture wrapper was:

```bash
runuser -u opencode -- env \
  HOME=/home/opencode \
  OPENCODE_CONFIG=/tmp/qwen38-pp2-final-opencode-config.json \
  PATH=/home/opencode/.opencode/bin:/usr/local/bin:/usr/bin:/bin \
  /home/opencode/.opencode/bin/opencode --pure run \
  --format json \
  --print-logs \
  --log-level INFO \
  --model local-qwen/qwen3.8-27b-fp8-sm86 \
  --variant low \
  --title qwen38-pp2-final-gate2 \
  'Use the read tool to inspect /home/opencode/workspace/qwen38-pp2-final-deployments.txt; do not answer until you have actually consumed the tool result. Review only the env=production rows against the policy stated in the file. Identify every offending service and its exact reason, compute the sum of the offender ports, and separately cite the staging audit timeout as evidence that you distinguished staging from production. Return exactly four lines in this format: OFFENDERS=<comma-separated names in file order>\nREASONS=<semicolon-separated exact violations in file order>\nPORT_SUM=<integer>\nSTAGING_EVIDENCE=<service and timeout_ms>.'
```

The archived TP2 config points to `http://127.0.0.1:30020/v1` with 300,000 ms header/chunk timeouts. The PP2 config points to `http://127.0.0.1:30021/v1` with 900,000 ms timeouts because its large agent prefill was slower.

TP2 session `ses_ffc21c9faffe2257jqna9jDxwd` completed the `read` tool call for `qwen38-tp2-final-deployments.txt`, consumed the full ten-line result, exited zero, and returned:

```text
OFFENDERS=billing,search,ingest
REASONS=retries=1 < 3;timeout_ms=7000 > 5000;owner=UNASSIGNED
PORT_SUM=24999
STAGING_EVIDENCE=audit timeout_ms=12000
```

Its wall interval was 01:21:28.297 through 01:26:27.674 EDT, about 299.38 seconds. The first model turn processed 7,583 prompt tokens at 28.15 tok/s and generated 53 tokens at 30.07 tok/s for the tool call. The second processed 274 prompt tokens at 27.15 tok/s and generated 464 tokens at 29.71 tok/s.

PP2 session `ses_ffc17fe04ffeZZPt0XA8JlUDLA` completed the corresponding `read` tool call, consumed the result, exited zero, and returned the same correct four semantic lines. Its wall interval was 01:32:10.278 through 01:41:10.936 EDT, about 540.66 seconds. The first model turn processed 7,583 prompt tokens at 15.50 tok/s and generated 57 tokens at 20.53 tok/s. The second processed 274 prompt tokens at 15.25 tok/s and generated 560 tokens at 20.22 tok/s.

The JSON event streams contain both the completed tool result and the final answer; this was not a root-owned `curl` substitute.

## Speed methodology

The identical benchmark driver is archived in both final topology directories as `stream-bench.py`; both copies have SHA-256 `1a264f5f4f49cb37c94d176bf9233bdfc236a37f873e37a3f64a598d3d5dc457`. The measured request was:

- native llama.cpp streaming `/completion` endpoint;
- 512 deterministic random token IDs generated by Python RNG seed `20260815`, each in the inclusive range 100 through 10000;
- prompt-token-array SHA-256 `d763269267aa8fe2f9e23b1d2fc384aeca45855a16b9c6b676388f1ea0a44ceb`;
- 128 generated tokens;
- `ignore_eos=true` and `cache_prompt=false`;
- seed `424242`;
- greedy sampling: temperature 0, top-k 1, top-p 1, min-p 0, repeat penalty 1;
- concurrency one;
- one unmeasured 64-prompt-token plus 16-generation-token warmup;
- five measured samples after warmup.

Exact invocations:

```bash
python3 /mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final/stream-bench.py \
  --port 30020 --samples 5 --topology TP2

python3 /mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final/stream-bench.py \
  --port 30021 --samples 5 --topology PP2
```

Every measured sample reported exactly 512 prompt tokens and 128 generated tokens. All ten outputs had SHA-256 `83dac9cf1799a4b516f30d3bd6dc3bedeb167877e875dd94631cba781f8dd6d9`.

Time to first token is client wall time from request start through the first non-empty streamed token. Client decode rate is based on first-to-last streamed token arrival. Server prompt and decode rates are the terminal llama.cpp timings. Client end-to-end total rate is `(512 + 128) / request latency`.

### Full speed results

| Metric | TP2 mean | TP2 median | TP2 range | PP2 mean | PP2 median | PP2 range |
|---|---:|---:|---:|---:|---:|---:|
| Client TTFT | 18.393 s | 18.342 s | 17.949-18.802 s | 33.183 s | 33.215 s | 33.047-33.224 s |
| Client latency | 22.637 s | 22.598 s | 22.153-23.059 s | 39.403 s | 39.432 s | 39.265-39.447 s |
| Client decode | 29.930 tok/s | 29.843 tok/s | 29.777-30.215 | 20.420 tok/s | 20.425 tok/s | 20.409-20.428 |
| Client end-to-end total | 28.279 tok/s | 28.321 tok/s | 27.755-28.890 | 16.243 tok/s | 16.230 tok/s | 16.224-16.300 |
| Server prompt | 27.849 tok/s | 27.918 tok/s | 27.235-28.530 | 15.431 tok/s | 15.416 tok/s | 15.412-15.494 |
| Server decode | 30.165 tok/s | 30.078 tok/s | 30.010-30.452 | 20.581 tok/s | 20.586 tok/s | 20.569-20.589 |

Measured intervals:

```text
TP2 2026-08-15 01:27:28.700-01:29:26.154 EDT
PP2 2026-08-15 01:41:39.756-01:45:02.459 EDT
```

### GPU and interconnect telemetry

| Topology | GPU | Mean/peak utilization | Mean/peak power | Peak memory |
|---|---:|---:|---:|---:|
| TP2 | 0 | 96.222% / 100% | 300.641 W / 323.40 W | 23,062 MiB |
| TP2 | 1 | 96.931% / 100% | 296.400 W / 347.25 W | 23,062 MiB |
| PP2 | 0 | 55.957% / 100% | 249.655 W / 289.30 W | 22,856 MiB |
| PP2 | 1 | 46.243% / 100% | 227.087 W / 285.80 W | 22,856 MiB |

TP2 accumulated 10,178,864 KiB, or 9.707 GiB, TX and the same RX on each GPU during the benchmark. The symmetric physical directions prove heavy tensor-parallel exchange.

PP2 accumulated 65,480 KiB, or 63.945 MiB, from GPU0 TX to GPU1 RX and zero in the reverse direction. That matches a one-way pipeline-stage handoff.

## Initial llama.cpp receipt index

All authoritative receipts are under:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86
```

Top-level receipts:

- `Qwen3.8-27B-F8_E4M3-BF16Scale.gguf`: converted model.
- `llama-final.patch`: exact nested-fork source diff on the recorded base commit.
- `final-exo-commit.txt`, `final-llama-commit.txt`, `final-exo-status.txt`, `final-llama-status.txt`: source state.
- `final-sha256.txt`, `final-runtime-sha256.txt`: model, patch, executable, and library hashes.
- `final-CMakeCache.txt`: compiler/backend configuration.
- `final-llama-server-ldd.txt`: resolved runtime libraries.
- `final-topology.txt`, `final-nvidia-smi-q.txt`: hardware receipts.
- `final-validation-pass.txt`: authoritative fail-closed final validation.
- `f8_mmvq_test.cpp`, `f8_mmvq_test`: standalone CUDA numerical test source and binary.

Per-topology authoritative directories:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86/tp2-final
/mnt/sanic/qwen3.8-27b-fp8-sm86/pp2-final
```

Each contains:

- `server-final.log` and `process-command.txt`;
- model/readiness and GPU-memory receipts;
- `gate1-request.json`, `gate1-response.json`, headers, timing, and timestamps;
- `opencode-config.json`, fixture, UID, timestamps, status, stderr, and complete JSON event stream;
- `stream-bench.py`, raw JSONL sample results, validated summary, timestamps, and status;
- 100 ms GPU telemetry, summarized GPU metrics, raw NVLink counter snapshots, and computed deltas.

The older `/mnt/sanic/qwen3.8-27b-fp8-sm86/tp2` directory contains exploratory/intermediate attempts and is not the final campaign. Use only `tp2-final` and `pp2-final` for the reported numbers.

## Initial llama.cpp campaign limitations

- Full 262,144-token memory allocation is proven; a 262,144-token inference request is not.
- These initial topology-comparison numbers are direct vendored llama.cpp backend results, not Exo router/API end-to-end measurements.
- The custom kernel is FP8-resident storage with on-load floating decode, not native FP8 compute and not the article's literal INT8/IMMA implementation.
- Batch/microbatch eight was deliberately selected to stay on the custom kernel, so these are not maximum possible large-batch prefill numbers for a fallback kernel.
- Trace verbosity four was matched across topologies but may impose a small absolute-throughput cost.
- NCCL linkage and default selection are proven, and no fallback warning occurred; the runtime log does not print an explicit communicator implementation name.
- No CPU model buffer or CPU model operator ran, but small host orchestration/staging buffers are present as expected.

## Two-hour optimization follow-up: promoted result

The additional optimization window ran on 2026-08-15 after the first 107.299 tok/s result. The promoted candidate is direct vendored SGLang TP2 on the same two RTX 3090s. It keeps radix caching disabled, allocates exact 262,144-token target and draft pools, and uses:

- the raw block-128 E4M3 checkpoint through SGLang's SM86 FP8-storage Marlin W8A16 path;
- native one-layer Qwen MTP/EAGLE with six steps, top-k two, and eight verification tokens (`S6/K2/D8`);
- a deterministic 32,768-token FR-Spec vocabulary reconstructed correctly across the TP-sharded LM head;
- a private per-rank FP8-Marlin draft head, leaving the shared target head untouched;
- pre-LM-head draft-extend pruning, so the D8 extension projects one selected row rather than all eight rows;
- 2,048-token prefill chunks; and
- batch-one CUDA graphs, max concurrency one, FP8 KV, and no CPU offload.

Top-k two is a tree, so ReplaySSM is intentionally absent: the current ReplaySSM verifier accepts only top-k one. Normal GDN intermediate state fits because `--max-mamba-cache-size 1` is sufficient when radix caching and request concurrency are both one.

### Post-gate speed and latency

The short workload is the exact 68-token asynchronous-worker-pool prompt in the prompt manifest, thinking disabled, greedy sampling, 512 output tokens, one warmup, and five measured requests. The nonstream and streaming campaigns were separate and both ran only after the simple gate and real OpenCode gate passed on this exact launch.

| Metric | Mean | Median | Range | Meaning |
|---|---:|---:|---:|---|
| Scheduler decode | **177.909 tok/s** | 177.889 | 177.575–178.279 | `(512 - 1) / (request_finished_ts - prefill_finished_time)` |
| Client post-first-token decode | **177.487 tok/s** | 177.470 | 177.419–177.574 | Harness-derived: 511 tokens over full wall time minus TTFT |
| Nonstream full-wall output rate | **172.184 tok/s** | 172.150 | 171.819–172.524 | 512 output tokens over complete HTTP wall time |
| Streaming full-wall output rate | **171.567 tok/s** | 171.535 | 171.530–171.644 | Complete streamed HTTP wall time |
| Scheduler short-prompt prefill | **843.885 tok/s** | 844.554 | 834.187–849.163 | 68 tokens over the server prefill phase |
| Client short-prompt TTFT | **105.186 ms** | 105.257 | 104.717–105.454 | Request start to first non-empty SSE text |
| Speculative accepted length | 4.92308 | 4.92308 | identical | 104 verification cycles per 512-token response |

All ten final short-prompt outputs were non-empty and had SHA-256 `f1c871902de699d15e189f73d75127515dac045b1d149f6d4df5eadb8c8b3cf6`.

The long probe used the frozen pre-follow-up report snapshot as the literal prompt: 52,030 UTF-8 bytes, SHA-256 `ebe2890f8514c1c1a5e1fcfd0e4d3fda40a0371878179af3ceaf81dba3f00e7d`, and 18,029 prompt tokens, followed by 16 output tokens. The exact prompt is archived as `long-prompt-exact.md`; the live report was extended only after the request. The post-gate nonstream request measured **1,911.555 prefill tok/s** over a 9.4316-second scheduler prefill phase. The separate streamed request measured **9.5668 seconds client TTFT** and 9.6545 seconds full wall time. One sample is reported because this probe exists to expose the prefill regime, not to estimate a decode distribution from only 15 post-first tokens.

This resolves the prompt-versus-decode question explicitly. The fixed-overhead 68-token request still prefills at about 4.7 times scheduler decode, while the 18,029-token request prefills at about 10.7 times scheduler decode. Prefill is therefore faster than decode as expected; the original 27.918 tok/s llama.cpp prompt result came from the deliberately narrow batch-eight kernel path, not an intrinsic property of this model.

Do not use SGLang's raw `decode_throughput` metadata as the headline. Its flush-interval accounting reported about 303 tok/s on the short workload and millions of tok/s on the 16-token probe. The scheduler timestamp calculation and the client harness calculation `(completion_tokens - 1) / (full wall time - TTFT)` above have clear denominators and agree closely. Because speculative responses can put multiple tokens in one SSE event, the latter is a post-first output rate rather than a literal per-token arrival interval.

### What improved during the extension

The important measured steps were:

| Candidate | Scheduler decode | Full-wall output | Disposition |
|---|---:|---:|---|
| Re-measured full-vocabulary top-k-one D8 | 111.556 tok/s | 109.057 tok/s | starting control |
| FR32, tree S5/K4/D8 | 171.632 | 166.284 | large tree/FR-Spec gain |
| FR32, tree S5/K2/D8 | 173.412 | 167.959 | narrower tree won |
| FR32, tree S6/K2/D8 before final head work | about 175.6 | about 169.9 | depth optimum |
| Add D8-to-one draft-extend head pruning | 176.393 | 170.640 | output/acceptance preserved |
| Add private FP8-Marlin FR head | about 177.7 | about 172.0 | accepted; small acceptance change |
| Promoted post-gate configuration | **177.909** | **172.184** | final |
| Disable overlap scheduler, gated control | 174.522 | 170.102 | regressed; not promoted |

The two source-level changes at the end are deliberately narrow. First, single-layer EAGLE now passes its already-computed `select_index` into `LogitsProcessor` before the draft LM head. Dense TP therefore reduces the extension head GEMM and TP logits from D8 rows to one row per request; gathered-buffer/DP behavior retains the old all-row shape. Second, `SGLANG_FR_SPEC_FP8_MARLIN_HEAD=1` verifies that the TP-reconstructed draft head is private, quantizes only it per output channel, and packs it through the existing Marlin machinery. Tests reject target-module or target-storage aliasing.

The topology sweep rejected S7/K2/D8, S6/K3/D8, wider D10/D12 verification, and top-k-four breadth. A 30K or 28K FR vocabulary was not promoted: the maximum compute saving was below 0.3% while a single extra verification cycle costs about 1%. Disabling global metrics, serializing CUDA connections, legacy custom all-reduce, speculative attention decode mode, and FlashInfer overlap-plan streaming were also rejected or unsafe. The latter can observe a stale tree mask at top-k greater than one. The final no-overlap control initially exposed a cumulative-branch signature mismatch; a narrow compatibility fix now accepts the scheduler's null PP proxy at TP2 and fails closed for a real PP proxy. After the simple and OpenCode gates both passed, its three measured samples regressed to 174.522 scheduler decode and 170.102 full-wall tok/s, so overlap remains enabled in the promoted launch.

Prefill chunks improved independently: 512 measured 1,819.554 tok/s, 1,024 measured 1,866.517 tok/s, and 2,048 measured 1,916.844 tok/s in screening on the same 18,029-token prompt. The promoted post-gate 2,048 result was 1,911.555 tok/s. A 4,096-token chunk was not attempted because its transient-activation estimate exceeded the 2.02 GiB per-rank post-graph headroom.

### Exact promoted launch

The exact executed shell script is archived as `launch-server.sh`; it points at the byte-identical temporary map used during the run. For replay after `/tmp` cleanup, use the durable FR32 map shown here:

```bash
cd /root/exo/vendor/sglang
model=/root/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
token_map=/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/extension-final-20260815/frspec/qwen38-frspec-english-python-opencode-32768.pt

env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0,1 \
  PYTHONPATH=/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/flashinfer-sm86-overlay:/root/exo/vendor/sglang/python \
  XDG_CACHE_HOME=/tmp/qwen38-sgl-xdg \
  TORCH_EXTENSIONS_DIR=/tmp/qwen38-sgl-torch-fixed \
  TRITON_CACHE_DIR=/tmp/qwen38-sgl-triton \
  FLASHINFER_WORKSPACE_BASE=/tmp/qwen38-sgl-flashinfer-fixed \
  SGLANG_FORCE_FP8_MARLIN=1 \
  SGLANG_FR_SPEC_FP8_MARLIN_HEAD=1 \
  SGLANG_DISABLE_FLASHINFER_NORM=1 \
  SGLANG_ENABLE_TORCH_INFERENCE_MODE=1 \
  NCCL_P2P_LEVEL=NVL \
  PYTHONUNBUFFERED=1 \
  /var/lib/exo/runtimes/dsv4-fwuff-cu130/venv/bin/python \
  -m sglang.launch_server \
  --model-path "$model" \
  --served-model-name Qwen3.8-27B-FP8 \
  --host 127.0.0.1 --port 30022 \
  --tp-size 2 --dtype bfloat16 --quantization fp8 \
  --context-length 262144 --max-total-tokens 262144 \
  --max-running-requests 1 --max-mamba-cache-size 1 \
  --disable-radix-cache \
  --chunked-prefill-size 2048 --max-prefill-tokens 2048 \
  --mem-fraction-static 0.96 --kv-cache-dtype fp8_e4m3 \
  --attention-backend flashinfer --linear-attn-backend triton \
  --language-only --disable-prefill-cuda-graph \
  --cuda-graph-max-bs-decode 1 --cuda-graph-bs-decode 1 \
  --page-size 1 --scheduler-recv-interval 16 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "$model" \
  --speculative-num-steps 6 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --speculative-token-map "$token_map" \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --decode-log-interval 100000 \
  --log-level info
```

Fail closed unless startup logs the following semantic assertions. The two KV lines are distinguished by order and size: the 2.00 GB allocation is the target and the following 0.13 GB allocation is the one-layer draft, once per rank.

```text
weight-only FP8 compression will be used leveraging the Marlin kernel
Built TP-aware FR-Spec LM head: global hot vocab=32768, local rows=16384, TP=2
Packed private TP FR-Spec draft LM head with FP8 Marlin
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 262144, K size: 2.00 GB, V size: 2.00 GB
KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 262144, K size: 0.13 GB, V size: 0.13 GB
Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=8
Capture draft decode CUDA graph begin. backend=full, num_tokens_per_req=2
max_total_num_tokens=262144, chunked_prefill_size=2048, max_prefill_tokens=2048
```

The promoted log shows 2.02 GiB per rank free after graph capture.

### Final coherence order

The frozen server ran a code → systems → review → identical code history gate first. The first and last code outputs had the same SHA-256. The systems and review requests measured 117.507 and 109.092 scheduler decode tok/s, demonstrating that the >100 result is broader than one code completion but still workload dependent.

The required order then ran:

1. The simple deterministic gate returned exactly `PRODUCT=323`, `NEXT=13,21`, `REVERSE=desserts`, and `CHECK=D8_FINAL_COHERENT`; output SHA-256 was `7562ad8a6eb8f4cf51327d55029557efc953de1995a2117463237ff9e3a00131`.
2. OpenCode 1.18.15 ran as UID 30033 (`opencode`), exited zero with empty stderr, and emitted a completed structured `read` event for the ten-line fixture. It correctly identified billing `retries=1`, search `timeout_ms=7000`, and ingest `owner=UNASSIGNED`, excluded staging audit, and computed offender-port sum 24,999.
3. Only after both gates passed, the 1+5 nonstream, 1+5 streaming, and long-prompt speed measurements ran.

### Validation and receipts

Focused SGLang validation passed: **34 tests plus 4 subtests**, all 24 selected changed source/test files compiled, the fork's targeted Ruff rules (`F401`, `F821`, `UP037`) passed, and `git diff --check` was clean. The two additional tests cover null-proxy binding and fail-closed behavior for the synchronous EAGLE compatibility path. The standalone benchmark client has 6 passing tests, scoped Ruff passes, and scoped BasedPyright reports zero errors. The map generator is preserved with two independently byte-identical canonical generations; it depends on the SGLang/Torch runtime and is not clean under the root environment's dependency-blind strict BasedPyright invocation.

Durable receipts are under:

```text
/mnt/sanic/qwen3.8-27b-fp8-sm86-opt/extension-final-20260815
```

They include the complete promoted server transcript, exact launch, literal prompt artifacts, the screening JSONLs supporting the optimization table, every promoted final JSONL, simple-gate console text, raw OpenCode events/status/stderr/timestamps/user/version, exported OpenCode transcript, validation commands/results, runtime/hardware/model hashes, canonical FR32 map and generation receipt, the exact FlashInfer overlay delta, tracked binary patches at their recorded base commits, and complete untracked-source tarballs. `SHA256SUMS` covers the receipt tree.

### Boundaries

- Exact 262,144-token capacity is proven by allocated target and draft pools on both ranks. The longest actual prompt was 18,029 tokens; no 262,144-token end-to-end prefill was attempted.
- The reported forward path is GPU-only in the inference sense: weights, KV, recurrent state, and model operators stay on GPU with zero CPU offload. HTTP, tokenization, sampling orchestration, and scheduling still use the host.
- The promoted result is direct vendored SGLang TP2, not an Exo-router end-to-end result and not the requested llama.cpp TP2/PP2 comparison. Those llama.cpp campaigns remain separately reported above.
- Ampere executes weight-only FP8 storage through Marlin W8A16/HMMA; this is not native FP8 tensor-core arithmetic.
- FP8 KV startup warns that absent checkpoint KV scales default to 1.0. The semantic gates passed, but this campaign is not a perplexity or broad quality evaluation.
- Speculative configurations changed greedy output hashes in some sweeps. Promotion is based on semantic coherence and history stability, not a claim of token identity to the non-speculative target.
- The speed distribution is concurrency one, one fixed deterministic prompt, and five measured samples. It is not a universal service-capacity claim.
- Shutdown tracebacks in the server transcript are intentional `KeyboardInterrupt` cleanup after all HTTP results completed, not inference crashes.
