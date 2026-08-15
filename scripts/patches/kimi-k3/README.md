# Kimi K3 llama.cpp patch set

These patches are historical, portable reconstruction artifacts for the Kimi
K3 runtime work. All of their live content is committed on the authoritative
llama.cpp branch `exo/kimi-k3-cumulative`, currently at
`0c2743950c8dc10fc80791ecd0727e51a14c7aad`; do not use this patch directory as
an alternative feature base. The patches target base commit
`a30437bc3a2a661d1e9aad71b1160d9ad9bbfec1`.

Apply in filename order:

1. `0000-rpc-load-model-tensors-from-server-local-GGUF-files.patch` is the
   committed protocol-v6 file-aware RPC implementation at
   `d29a524eeaf39155825d6f0ef373075fe585cb12`.
2. `0001-rpc-prequeue-server-local-tensors.patch` queues every admitted
   server-local RPC tensor before client-side reads begin. It preserves
   rejection fallback, progress accounting, validation, and the final barrier.
3. `0002-kimi-k3-dspark.patch`, `0002b-...`, `0002c-...`, and `0002d-...`
   are one logical K3 DSpark change. The split files preserve two previously
   untracked source files without modifying the temporary worktree index.
   `0002d` keeps the official converter working with Transformers 5, which
   moved `bytes_to_unicode` from the GPT-2 tokenizer module into
   `convert_slow_tokenizer`.
4. `0003-kimi-k3-top16-router.patch` is the exact 896-expert/top-16 CUDA
   router specialization.
5. `0004-kimi-k3-situ-mmid.patch` adds first-class SiTU plus the paired
   gate/up MMID epilogue. Its measured admission is deliberately limited to
   exact K3 `IQ3_XXS` decode; `IQ2_XS` remains on the faster legacy path.
   Patch `0007` later adds a separate register-preserving IQ2 specialization
   without changing this frozen result.
   It follows `0003` because its whole-graph benchmark uses `0003`'s generic
   single-run test hook. Apply it to both the RPC client and server—RPC
   operation support is not version-negotiated.
6. `0005-rpc-attest-same-backing-tensor-source-files.patch` upgrades the
   file-aware protocol to v7 and defaults to fail-closed same-backing-file
   attestation by inode, size, and nanosecond mtime. It pins one race-safe
   `openat(..., O_NOFOLLOW)` descriptor per direct-child shard, bounds the
   per-client descriptor cache, and verifies per-buffer completion receipts.
   `--rpc-tensor-source-mode fallback` is the explicit compatibility escape
   hatch.
7. `0006-kimi-k3-down-mmid-weighted-reduce.patch` adds a first-class
   batch-one K3 down-projection operation for `IQ1_M`, `IQ2_XS`, and
   `IQ3_XXS`. Its CUDA path performs the 16 selected expert dot products,
   router weighting, and deterministic fixed-order reduction without
   materializing the 16 expert outputs. It also incorporates the structural
   post-MMID weighted-reduction fusion from upstream PR #25952, extended to
   K3's exact 32-node top-16 short form. The RPC protocol becomes v7.0.1.
   Set `LLAMA_KIMI_K3_FUSED_DOWN=0` to force the legacy model graph.
8. `0007-kimi-k3-iq2-situ-mmid-sequential.patch` adds the dominant
   `UD-Q2_K_XL` gate/up specialization. Four SM86 warps compute four rows in
   two phases, retaining only the reduced gate scalars while reusing the
   accumulator and 1,536-byte reduction scratch for the up projection. This
   avoids the register-pressure regression of the removed paired IQ2
   experiment. Admission remains exact K3 batch-one/top-16 with no bias or
   scale; `GGML_CUDA_KIMI_K3_SITU_MMID=0` forces the standalone-SiTU
   fallback. It adds no operation and does not change RPC v7.0.1. SHA-256:
   `66fa3d71a75f886aed6ddcf6f4a6a37d7dbc3c93afa7c1eeb25cc1c37a34426a`;
   stable patch ID:
   `da7086b47540028b90b25a60c8c7a09e14a6b1dc`.
9. `0008-kimi-k3-dspark-borrow-ctx-other-backends.patch` fixes draft
   scheduler visibility for tensors borrowed from the target context. It
   registers the exact backend handle that owns each otherwise-missing shared
   embedding or output tensor, preserving RPC session and tensor-ID identity,
   while keeping draft-owned devices ahead of borrowed devices and CPU. This
   directly addresses the full-Q2 integration failure where DSpark's borrowed
   `output.weight` lived on target `CUDA1` but the draft scheduler knew only
   `RPC1` and CPU. It also adds a focused synthetic non-CPU shared-tensor
   scheduler test and a useful runtime error if no owner can be found.
   SHA-256:
   `a1d96ee3028789929ec91b4c2b28c0b07e9be0fed1afed9167ea9941068f62ca`;
   stable patch ID:
   `c07821423d02ddbf4f672bbfc1969e48cfea2e2d`. The complete stack
   apply-checks cleanly. The cumulative branch also passes the CPU/RPC/mtmd
   build, the CUDA 13.1 SM86 build, the focused scheduler/RPC/mtmd CTests, the
   12-test Kimi CUDA suite, and all 448 CPU backend operation-selection cases.
   Full-size Kimi model and live MoonViT image inference remain outstanding.

Patches `0001`, `0002*`, and `0003` apply to `d29a524e`; `0004` applies after
`0003`, `0005` applies after the earlier kernel patches, and `0006` applies
after `0005`; `0007` applies after `0006`, and `0008` applies after `0007`.
The complete filename-ordered sequence has been apply-checked. When
reconstructing an old tree, apply `0004` through `0008` on both RPC clients and
servers to select every specialization and scheduler fix consistently. Current
deployments should build the cumulative branch directly. These are experiments
rather than a claim that every combination has completed a full-size K3
integration run. See `kimik3.md` for exact validation, performance results,
limitations, and stable patch IDs.
