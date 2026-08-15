# DeepSeek V4 Flash + OpenCode live setup

Last verified: 2026-08-09

This is the operational handoff for the local DeepSeek V4 Flash server and the
OpenCode client configured to use it. The stack was started and tested end to
end, including a real OpenCode `read` tool call and the model's follow-up
response after receiving the tool result.

## Current live state

- DeepSeek is running in detached Screen session `dsv4f`.
- The OpenAI-compatible endpoint is `http://127.0.0.1:30010/v1`.
- The served model ID is `deepseek-v4-flash`.
- Both RTX 3090s are active. At final verification, each SGLang scheduler used
  approximately 19.5 GiB of VRAM.
- The two Qwen TTS containers are stopped because they otherwise retain nearly
  all GPU memory. They must stay stopped while this DeepSeek instance is live.
- OpenCode `1.18.15` is installed at `/root/.opencode/bin/opencode`.
- The unprivileged `opencode` account has an independent OpenCode `1.18.15`
  install, user-owned provider config, and writable workspace.
- `/root/exo/opencode.json` is the canonical OpenCode configuration.
- `/root/.config/opencode/opencode.jsonc` is a symbolic link to that canonical
  file, so the same local model is available outside `/root/exo` too.

## Start using it

The server is already running. In a new shell:

```bash
source /root/.bashrc
cd /root/exo
opencode
```

If that shell has not picked up the installer-added `PATH` entry, use:

```bash
/root/.opencode/bin/opencode
```

For a non-interactive request:

```bash
opencode run --model local-dsv4/deepseek-v4-flash \
  "Inspect this repository and summarize its architecture."
```

OpenCode's configured model name is
`local-dsv4/deepseek-v4-flash`. Both its primary and small-model roles use the
local server, and automatic session sharing is disabled.

### Using the unprivileged `opencode` account

The account has its own install and config so `/root` can remain correctly
isolated with mode `0700`. From root, enter the account and start in its writable
workspace:

```bash
sudo -iu opencode
cd /home/opencode/workspace
opencode
```

The equivalent direct binary is:

```bash
/home/opencode/.opencode/bin/opencode
```

Account-specific paths:

| Purpose | Location |
| --- | --- |
| OpenCode binary | `/home/opencode/.opencode/bin/opencode` |
| OpenCode config | `/home/opencode/.config/opencode/opencode.jsonc` |
| Writable project workspace | `/home/opencode/workspace` |
| OpenCode state and sessions | `/home/opencode/.local/share/opencode` |

The PATH export is placed before the non-interactive guard in
`/home/opencode/.bashrc`, and `.bash_profile` sources that file. Consequently,
`opencode` resolves by name in interactive, login, and automation shells.

The account config is a user-owned snapshot of `/root/exo/opencode.json`, not a
link through `/root`; weakening `/root` traversal permissions is unnecessary.
After changing the canonical config, root can resynchronize the account with:

```bash
install -m 0644 -o opencode -g opencode \
  /root/exo/opencode.json \
  /home/opencode/.config/opencode/opencode.jsonc
```

#### Bubblewrap-isolated use

`/usr/local/bin/opencode-bwrap` points to the maintained checkout at
`/usr/src/opencode-bwrap/opencode-bwrap`. Run it from the project that should be
writable inside the sandbox:

```bash
cd /home/opencode/projecthorizons
opencode-bwrap
```

The wrapper replaces the rest of `$HOME` with an empty directory, binds only
the current workspace read-write, passes OpenCode config/cache/state, shares
the host network for `127.0.0.1:30010`, and mounts the resolved OpenCode binary
directory read-only. It resolves the executable before entering Bubblewrap and
executes its absolute path; this is required for the account-local install at
`/home/opencode/.opencode/bin/opencode`.

The installed wrapper was tested end to end from `~/projecthorizons`:

```bash
opencode-bwrap --pure run --format json \
  'Do not use tools. Reply with exactly BWRAP_DSV4_OK and no other text.'
```

It returned exactly `BWRAP_DSV4_OK` with exit status 0 in OpenCode session
`ses_016c78a60ffeCKDQ9BvnUmSW7v`.

## Files and configuration

| Purpose | Location |
| --- | --- |
| Canonical OpenCode config | `/root/exo/opencode.json` |
| Global OpenCode config link | `/root/.config/opencode/opencode.jsonc` |
| Pre-existing schema-only config backup | `/root/.config/opencode/opencode.jsonc.pre-dsv4-20260809` |
| DeepSeek launcher | `/root/exo/scripts/dsv4_flash_hybrid_ep2_dwagon_opencode.sh` |
| Live server log | `/tmp/dsv4-opencode-live/server-opencode-account.log` |
| Optimization and validation history | `/root/exo/dsv4flash_opencode.md` |

The provider uses OpenCode's `@ai-sdk/openai-compatible` adapter with:

```text
baseURL:       http://127.0.0.1:30010/v1
context limit: 524288
output limit:  32768
tool calls:    enabled
reasoning:     enabled
```

OpenCode stable `1.18.15` and its live schema use the singular `provider` key.
Some V2 documentation/examples use a plural `providers` shape; do not replace
the tested config with that shape without also migrating and revalidating the
installed OpenCode version.

The install was performed using the official installer documented by OpenCode:

```bash
curl -fsSL https://opencode.ai/install | bash
```

Official references:

- [OpenCode installation](https://opencode.ai/en/docs)
- [OpenCode CLI](https://dev.opencode.ai/docs/cli/)
- [Live stable configuration schema](https://opencode.ai/config.json)

## Server launch and lifecycle

### Fresh launch

First free both GPUs. Pausing these containers is not sufficient because paused
CUDA processes retain VRAM:

```bash
docker stop qwen3-tts-nano-gpu0 qwen3-tts-nano-gpu1
```

Then launch the tested configuration:

```bash
mkdir -p /tmp/dsv4-opencode-live
screen -dmS dsv4f -L -Logfile /tmp/dsv4-opencode-live/server-opencode-account.log \
  env -u SGLANG_DSPARK_DEBUG_DUMP \
  PYTHONUNBUFFERED=1 \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-codex \
  TRITON_CACHE_DIR=/tmp/triton-codex \
  SGLANG_DSV4_INTERNAL_TIMING=0 \
  SGLANG_V4_MXFP4_FUSED_T5_MOE=0 \
  SGLANG_DSV4_OSCAR_FUSED_C4_PIPELINE=0 \
  /root/exo/scripts/dsv4_flash_hybrid_ep2_dwagon_opencode.sh --launch
```

Cold startup takes several minutes because it loads target and draft weights,
captures CUDA graphs, and runs the 2,694-token warmup. Wait for this log line:

```text
The server is fired up and ready to roll!
```

### Status and logs

```bash
screen -ls
tail -F /tmp/dsv4-opencode-live/server-opencode-account.log
curl -fsS http://127.0.0.1:30010/health
curl -fsS http://127.0.0.1:30010/v1/models | jq
curl -fsS http://127.0.0.1:30010/server_info | jq
nvidia-smi
```

Attach to the server console with `screen -r dsv4f`; detach without stopping it
using `Ctrl-A`, then `D`.

### Stop DeepSeek and restore TTS

Only restore TTS after DeepSeek has released the GPUs:

```bash
screen -S dsv4f -X quit
docker start qwen3-tts-nano-gpu0 qwen3-tts-nano-gpu1
```

## Verified runtime contract

The live `/server_info` response was checked after startup:

| Setting | Verified value |
| --- | --- |
| Context length / total-token pool | `524288` / `524288` |
| Public KV cache dtype | `fp8_e4m3` |
| Parallelism | TP2, EP2, PP1 |
| CPU offload workers | 56 per rank |
| CUDA graphs | enabled |
| Decode / prefill graph backends | `full` / `breakable` |
| Speculation | DSPARK, fixed verify length 4 |
| Oscar split-history cache | enabled, fixed-address SM86 path |
| Oscar execution path | `sm86-oscar-int2-split-history-fp32-online-v1` |
| Max running requests | 1 |
| Experimental fused T5 MoE / fused C4 pipeline | disabled for this handoff |

The public API reports `fp8_e4m3`, but this admitted SM86 configuration uses the
Oscar INT2 split-history physical layout for the affected history cache. The
runtime still warns that public FP8 scaling defaults to 1.0. Basic and OpenCode
coherency passed, but quality near the full 524K limit remains something to
evaluate before relying on the extreme end of the window.

Only one request runs at a time in this memory-maximized configuration. Extra
OpenCode calls queue rather than execute concurrently.

## End-to-end OpenCode proof

The tests below used OpenCode itself, not a direct `curl` shortcut. `--pure`
disabled external plugins to make the transport/tool proof deterministic; it
does not bypass the OpenCode agent or provider stack.

### Unprivileged-account tool-call proof

This command was executed as UID 30033 from `/home/opencode`, using only the
account's PATH, config, state, and file permissions:

```bash
runuser -u opencode -- bash -lc \
  'cd /home/opencode/workspace && \
  opencode --pure run --format json \
  "Use the read tool to inspect /home/opencode/.config/opencode/opencode.jsonc. Then reply with exactly USER_OPENCODE_DSV4_OK and no other text."'
```

OpenCode emitted a completed `read` tool event, supplied the file result to
DeepSeek in a second model call, and returned exactly:

```text
USER_OPENCODE_DSV4_OK
```

OpenCode session: `ses_016ce7c66ffeI6P7NNCgeWBInR`. The command exited 0. This
proves the unprivileged account can resolve the local provider, reach the live
server, execute a tool under its own permissions, and complete the required
follow-up model turn.

### Global-config smoke test

This was run from `/tmp`, with no project-local model override, after installing
the global config link:

```bash
cd /tmp
/root/.opencode/bin/opencode --pure run --format json \
  'Do not use tools. Reply with exactly: GLOBAL_DSV4_READY'
```

Verified result:

```text
GLOBAL_DSV4_READY
```

OpenCode session: `ses_0171ce90affekMA0xHGGnx3xqb`. It completed with exit
status 0, proving that an arbitrary working directory resolves the global local
provider, selects DeepSeek by default, and reaches the live server.

### Exact-response transport test

```bash
/root/.opencode/bin/opencode --pure run --format json \
  --model local-dsv4/deepseek-v4-flash \
  'This is a transport smoke test. Do not use tools. Reply with exactly: OPENCODE_DSV4_OK'
```

Verified result:

```text
OPENCODE_DSV4_OK
```

OpenCode session: `ses_01722a1ccffen2tW8lY3aLXhEN`. The response completed with
exit status 0. OpenCode sent a roughly 9K-token agent context, so this is a much
heavier first-call test than a minimal API prompt.

### Real tool call plus follow-up

```bash
/root/.opencode/bin/opencode --pure run --format json \
  --model local-dsv4/deepseek-v4-flash \
  'Use the read tool to inspect /root/exo/opencode.json. Then reply with exactly the configured baseURL and no other text.'
```

OpenCode emitted a `tool_use` event for its `read` tool, read the file, returned
the tool result to DeepSeek in a follow-up model call, and produced exactly:

```text
http://127.0.0.1:30010/v1
```

OpenCode session: `ses_01721fe0fffeyM33FjrxJEH918`. This completed with exit
status 0 and proves the local model can perform the multi-turn tool protocol
needed for actual OpenCode work.

## Validation and troubleshooting

Confirm what OpenCode resolves globally from a directory outside the repo:

```bash
cd /tmp
/root/.opencode/bin/opencode --pure debug config | jq \
  '{model,small_model,enabled_providers,share,provider}'
/root/.opencode/bin/opencode --pure models local-dsv4
```

The second command should print:

```text
local-dsv4/deepseek-v4-flash
```

Common failure modes:

- `opencode: command not found`: source `/root/.bashrc` or use the absolute
  binary path.
- Connection refused: inspect the log and wait for the ready line. CUDA graph
  capture and warmup happen before the listener becomes usable.
- CUDA out of memory during startup: ensure both TTS containers are stopped,
  not merely paused, and use `nvidia-smi` to find any other GPU consumers.
- Requests appear serialized: this launch intentionally has
  `max_running_requests=1` to preserve the 524K cache and graph headroom.
- A first OpenCode request has higher TTFT than a small benchmark prompt:
  OpenCode supplies a large agent/tool system context. Prefix-cache-warmed
  follow-ups are substantially faster.
- Remote clients cannot connect: the tested listener is intentionally bound to
  `127.0.0.1` and has no authentication layer. Do not expose it publicly as-is.
- `EACCES` for `/root/exo` when using `runuser`: the new process inherited an
  inaccessible root-owned current directory. Use `sudo -iu opencode` or change
  to `/home/opencode/workspace` before invoking the account command.
