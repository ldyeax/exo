# DeepSeek V4 Flash release-audit patch

`0001-dsv4-flash-release-audit.patch` is a reproducible delta on top of the
accumulated local SGLang DSpark tree whose pre-patch working snapshot was based
on commit `6cc9352dfe6c5c013750e72b39c127870ef5b54f`.

The patch contains only changes selected by the 2026-08-01 release audit:

- physical-token KT host-buffer retention for decode and prefill CUDA graphs;
- a capture-time TP barrier at breakable-graph eager boundaries;
- DSpark draft-graph admission down to a measured-safe 0.25 GiB margin;
- target-only expert-distribution recording;
- streaming FP8 `wo_a` pairing and RunAI view lifetime protection;
- DeepSeek V4 strict-thinking parser and end-of-stream finalization support;
- Ampere guards for optional top-k-v2 metadata/kernels;
- 4D singleton-head RoPE tolerance and a valid raw-index output fallback;
- the referenced FP16 FE8M0 Marlin scale-dequant specialization;
- two small upstream scheduler/TBO completeness fixes.

The patch intentionally excludes the remote all-BF16 cache conversion and its
dependent kernels. The accepted dwagon path retains the 584-byte packed cache,
while the remote conversion uses 1024 bytes per token and contains incomplete
or dead paths when separated from its full layout change. It also excludes
ROCm, SM90+, RunAI-only transport, DSA/DeepEP, and unrelated large refactors.

Provenance:

- canonical remote deployment tree:
  `/mnt/sanic/projects/deploy-dsv4-general-release`
- remote tracked SGLang patch SHA256:
  `bb6b1e0dd0a699c5fbb540d7b7bc87cd53dfae1d276bfafc0770cd103816190b`
- selected local patch SHA256: obtain with
  `sha256sum scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch`

Verify an installed source tree without changing it:

```bash
git -C /var/lib/exo/sources/sglang-dspark-30261 apply --reverse --check \
  /root/exo/scripts/patches/dsv4-flash/0001-dsv4-flash-release-audit.patch
```

The launcher performs this check and refuses to continue if the audited patch
is absent. It does not copy from or modify Kassie's deployment folder.
