# Qwen3.8-27B-FP8 for Visual Studio 2026 Copilot

This folder starts the optimized 256K-context Qwen server on the two RTX 3090s and exposes it to Visual Studio 2026 as a tool-capable local model. It also contains a repository or user-level Copilot custom-agent definition.

The connection has two distinct pieces:

1. The **model provider** makes `Qwen3.8-27B-FP8` appear in Visual Studio's model picker.
2. The **custom agent** supplies the coding persona and lets Visual Studio offer its workspace tools to the selected model.

The agent file intentionally has no `model:` or `tools:` fields. Microsoft documents that omitting `model` uses the model currently selected in the picker, while omitting `tools` enables the tools available in that Visual Studio build. This avoids pinning an unrecognized Copilot-hosted model identifier or stale tool names.

## What runs here

```text
Visual Studio Copilot
  -> Ollama API on Windows 127.0.0.1:11434
  -> SSH local forward
  -> Ollama/OpenAI bridge on GPU host 127.0.0.1:11434
  -> SGLang OpenAI API on GPU host 127.0.0.1:30022
  -> Qwen3.8-27B-FP8, TP2 on GPU 0 and GPU 1
```

The bridge is necessary because the vendored SGLang Ollama adapter handles ordinary chat but currently drops Ollama tool schemas. `ollama_openai_bridge.py` translates discovery, thinking, chat history, structured tool calls, and streaming while using SGLang's already-qualified OpenAI tool parser.

The validated runtime remains GPU-only for weights, KV cache, and forward execution. Tokenization, HTTP handling, and scheduling still use the CPU.

| Item | Value |
| --- | --- |
| Model | `Qwen3.8-27B-FP8` |
| Context allocation | 262,144 tokens |
| SGLang API | `http://127.0.0.1:30022/v1` |
| Visual Studio/Ollama API | `http://127.0.0.1:11434` |
| Screen session | `qwen38-vs2026` |
| Parallelism | TP2 across both RTX 3090s |
| Measured short-prompt scheduler decode | 177.909 tokens/s mean |
| Measured 18,029-token prefill | 1,911.555 tokens/s |
| Measured 68-token client TTFT | 105.186 ms mean |

See [the full optimization and validation report](../qwen3.8-27b.md) for exact metric definitions, prompts, hashes, limitations, and receipts.

## 1. Start the server

From `/root/exo` on the GPU host:

```bash
./qwen3.8-27b/start-server-screen.sh
```

The command returns after creating a detached GNU screen session. Initial model load and CUDA graph capture take a few minutes. Check readiness with:

```bash
./qwen3.8-27b/status.sh
```

Inspect the live console with:

```bash
screen -r qwen38-vs2026
```

Detach without stopping it by pressing `Ctrl+A`, then `D`. The default combined log is `/tmp/qwen38-vs2026-screen.log`.

Stop the service deliberately with:

```bash
./qwen3.8-27b/stop-server-screen.sh
```

`launch-sglang.sh` contains the exact promoted S6/K2/D8, FP8-Marlin, FR32, 2048-token prefill configuration. `server.env.example` lists optional path, port, device, and session overrides. Export overrides before starting; the scripts do not execute the example file automatically.

## 2. Connect Windows without exposing the service

Both listeners bind to loopback. Keep them that way: neither endpoint has authentication.

On the Visual Studio machine, open PowerShell and keep this tunnel running:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& .\Connect-QwenTunnel.ps1 -HostName <gpu-host-or-ip> -UserName root
```

Run this from the `visual-studio` directory, or provide its full path. The equivalent plain command is:

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 11434:127.0.0.1:11434 root@<gpu-host-or-ip>
```

In another PowerShell window, prove discovery and structured tool calling before opening Visual Studio:

```powershell
& .\Test-QwenEndpoint.ps1
```

The expected final line starts with `PASS` and lists `completion, tools, thinking`.

## 3. Add Qwen to Visual Studio's model picker

Use Visual Studio 2026 version 18.4 or newer. In Copilot Chat:

1. Open the model picker and choose **Manage Models** or **Add Models**.
2. Choose the **Ollama** or local-model provider if it is present in your build.
3. Set the Ollama server URL to `http://127.0.0.1:11434`.
4. Refresh model discovery and select `Qwen3.8-27B-FP8`.
5. Leave the SSH tunnel open for the entire chat session.

The exact labels vary between stable and Insiders builds. There is an important product boundary: as of August 20, 2026, Microsoft's [Visual Studio BYOM documentation](https://learn.microsoft.com/en-us/visualstudio/ide/copilot-select-add-models?view=visualstudio) publicly documents provider keys for OpenAI, Anthropic, and Google, but does not document an arbitrary OpenAI-compatible base URL. Therefore:

- If your Visual Studio build shows an Ollama provider, use the steps above and this bridge.
- If it does not, update to the latest Visual Studio 2026 18.4+ servicing or Insiders build and check again.
- Do not paste `http://127.0.0.1:30022/v1` into an OpenAI API-key field; that field does not configure a base URL.
- If the provider is still absent, the built-in Visual Studio Copilot client cannot use this self-hosted endpoint in that build. VS Code's separate Custom Endpoint provider is not evidence that full Visual Studio has the same feature.

Microsoft also documents that Visual Studio BYOM affects Copilot Chat only, not inline completions or generated commit messages, and that BYOM is unavailable for Copilot Business and Enterprise accounts. Repository indexing or intent services can still contact Copilot services.

## 4. Install the custom agent

For only the repository containing the solution, run from the `visual-studio` directory:

```powershell
& .\Install-QwenAgent.ps1 -Scope Repository -RepositoryPath C:\src\your-repository
```

This copies the definition to:

```text
C:\src\your-repository\.github\agents\qwen38-local.agent.md
```

To make it available for every solution for the current Windows user:

```powershell
& .\Install-QwenAgent.ps1 -Scope User
```

That copies it to `%USERPROFILE%\.github\agents`, the user-level location documented by Microsoft. You can also copy `qwen38-local.agent.md` manually to either location.

After installation:

1. Reopen the solution or start a fresh Copilot Chat.
2. Select `Qwen3.8-27B-FP8` in the model picker.
3. Select **Qwen 3.8 Local Engineer** in the agent picker, or type `@` and choose it from the completion list.
4. Open the **Tools** panel and ensure file search, file read/edit, reference search, and terminal tools are enabled. Microsoft warns that tool names and enabled state can vary by Visual Studio version.
5. Start with a bounded gate such as: `Inspect this solution, read one relevant file with a tool, and report its path before proposing any edit.`

The authoritative custom-agent format and locations are documented in [Use built-in and custom agents with GitHub Copilot](https://learn.microsoft.com/en-us/visualstudio/ide/copilot-specialized-agents?view=visualstudio). Ollama's [tool-calling documentation](https://docs.ollama.com/capabilities/tool-calling) defines the request and response shape implemented by the bridge.

## Operations and troubleshooting

### Model does not appear

Confirm the tunnel and discovery endpoint from Windows:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
Invoke-RestMethod -Method Post -ContentType application/json `
  -Uri http://127.0.0.1:11434/api/show `
  -Body '{"model":"Qwen3.8-27B-FP8"}'
```

If these work but Visual Studio shows no Ollama provider, it is a Visual Studio channel/capability issue rather than a server issue.

### Agent appears but cannot edit or run commands

Open the Copilot Chat Tools panel and enable the required groups. The agent definition deliberately does not hard-code tool names because Microsoft says those names differ across Copilot platforms and Visual Studio versions.

### `502` from the bridge

The bridge is running but SGLang is not ready or exited. On the GPU host:

```bash
./qwen3.8-27b/status.sh
tail -n 100 /tmp/qwen38-vs2026-screen.log
nvidia-smi
```

### Port 11434 is already occupied on Windows

Use another local tunnel port, for example 11435:

```powershell
& .\Connect-QwenTunnel.ps1 -HostName <gpu-host> -LocalPort 11435
```

Then configure Visual Studio with `http://127.0.0.1:11435`.

### Context and concurrency expectations

The target and draft KV pools are allocated for 262,144 tokens, but the longest completed end-to-end prompt in this campaign was 18,029 tokens. This is a single-request speed configuration (`max-running-requests=1`); simultaneous Visual Studio conversations queue rather than run concurrently. The language-only launch does not accept images.

## Files

- `launch-sglang.sh`: exact optimized model launch.
- `run-server-stack.sh`: supervises SGLang and the Ollama bridge together.
- `start-server-screen.sh`, `stop-server-screen.sh`, `status.sh`: persistent operation.
- `ollama_openai_bridge.py`: dependency-free Ollama discovery/chat/tool adapter.
- `server.env.example`: optional system-specific overrides.
- `visual-studio/qwen38-local.agent.md`: custom-agent definition.
- `visual-studio/Install-QwenAgent.ps1`: repository or user-level installer.
- `visual-studio/Connect-QwenTunnel.ps1`: secure loopback tunnel.
- `visual-studio/Test-QwenEndpoint.ps1`: discovery and tool-call gate.
- `tests/test_ollama_openai_bridge.py`: protocol conversion regressions.
