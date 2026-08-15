# OSDI26 GLM-5.2 hybrid-checkpoint patch stack

This directory is the historical portable patch form of the GLM-5.2 hybrid
W8A16 and MTP work. Its live, modernized content belongs on the authoritative
SGLang branch `exo/dsv4-cumulative-0801` and KTransformers branch
`exo/glm52-osdi26-patched`; do not use these recorded terminals as alternative
feature bases. Apply the SGLang series in filename order only to reconstruct
the original source result:

`1218b2f8965b5a27c9d9004ff9324373414be322`

The recorded terminal SGLang source is:

`720b40b2783b1a515134f4ab9fe820931cfbee36`

Apply the KTransformers series in filename order to:

`098740a24a29c25972d1249b157840bf156f2849`

The recorded terminal KTransformers source is:

`2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a`

The outer series advances only the `third_party/sglang` gitlink. It therefore
expects the corresponding SGLang commits or equivalent patched trees to be
available in that submodule.

## Historical local checkout

The fully integrated checkout used for the original campaign was recorded
inside the Exo repository at:

- `vendor/ktransformers`, root gitlink
  `45a3a797658140dcf8426cd3f2b2c6c969f8f5d8`;
- `vendor/ktransformers/third_party/sglang`, nested gitlink
  `720b40b2783b1a515134f4ab9fe820931cfbee36`.

The KTransformers integration commit is a direct child of the recorded
six-patch terminal `2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a`. It contains no
additional runtime code: it advances the nested SGLang gitlink from
`f3f6ccfbbcdd5ef5e650a74eecc1a233e07cec34` to the ninth-patch terminal
`720b40b2783b1a515134f4ab9fe820931cfbee36` and points that nested submodule
at its durable local repository.

At that time the root and nested URLs were deliberately machine-local:

- `/var/lib/exo/sources/ktransformers-glm52-osdi26-patched.git`;
- `/var/lib/exo/sources/sglang-glm52-osdi26-patched.git`.

Initialize only the two patched source repositories without fetching
KTransformers' unrelated optional submodules:

```bash
git -c protocol.file.allow=always submodule update --init vendor/ktransformers
git -c protocol.file.allow=always -C vendor/ktransformers \
  submodule update --init third_party/sglang
```

The live work now resides on the authoritative public branches named above.
The patch files and hashes below remain the portable historical reconstruction
path.

The SGLang stack:

1. Adds direct compact MLA `kv_b_proj` W8A16 kernels for Ampere.
2. Admits fused-rope and absorbed MLA paths without materializing BF16 KC/VC
   matrices.
3. Adds fail-closed, manifest-bound layer-78 MTP admission for the immutable
   hybrid checkpoint and persistent AMXINT4 experts.
4. Configures compact MLA metadata before FlashInfer planning and keeps the
   decision coherent across all attention layers.
5. Preserves ownership of the target embedding and Marlin-quantized LM head
   when those exact modules are shared with the MTP draft, including later
   reload/checkpoint post-processing.
6. Adds a separate fail-closed TP1 admission for the standalone `fwuff` MTP
   drafter. It binds logical AMX slots `[0, 1]` to physical NUMA nodes
   `[0, 0]` with `cpuinfer=60` and two thread pools, requires shared host
   weights to be off, and selects exactly the 38 standalone tensors from
   shards 00001 and 00005. The local TP2 admission and selector are unchanged.

The persisted `kv_b_proj` layout is backend-neutral, per-head GPTQ W8.
Grouped Marlin performs only a compact-to-compact repack at load and never
expands persistent weights back to BF16. The OSDI26 launcher now requires
`SGLANG_MLA_KV_B_W8_BACKEND=marlin`, rejects `auto` and `triton`, and rejects
any non-Marlin runtime marker. The Triton implementation in this historical
patch stack is retained only to reproduce the completed cross-check.

Patch SHA-256 values:

| Series | Patch | SHA-256 |
| --- | --- | --- |
| SGLang | `0001-feat-quant-run-compact-MLA-kv_b-W8-on-Ampere.patch` | `b1827ce2239c2a97d375162ada5d074a637d00dbbe41fbf0e7603ce0b3665342` |
| SGLang | `0002-fix-mla-admit-compact-W8-fused-rope-path.patch` | `733140dab528178e65e3cb315906f24738302bbce638110a0cebd61cb2739eb9` |
| SGLang | `0003-fix-mla-keep-compact-W8-on-absorbed-paths.patch` | `aa44cbfba981a6dbd5957e691153fd9213476c6ebf79e9109eb2f510d7273337` |
| SGLang | `0004-fix-mtp-admit-attested-hybrid-layer-78.patch` | `dbad7f3cabc3e889ff3175a626840ec0dc40473e16b294432e85486e045218b7` |
| SGLang | `0005-fix-mtp-bind-hybrid-admission-to-live-artifacts.patch` | `3d20705fbc19a2b70de15d2744d1a28a8603f010d3c9131a355f0d851d60356b` |
| SGLang | `0006-fix-mla-align-compact-W8-FlashInfer-metadata.patch` | `af6dfbcd0b698f8aaff5f0b730e3cbe9c0376b3d55df97023bd641b7b588337f` |
| SGLang | `0007-fix-mla-synchronize-compact-metadata-globally.patch` | `f56d3d42ac0729680ce732a0380571a1e293e9ad6f12336265ee37c3aa644c12` |
| SGLang | `0008-fix-mtp-preserve-shared-Marlin-module-ownership.patch` | `c2a7dbcaff220da446a6718ae9ddf7efd50b690d8f0b41e6c95ec15c933cc8e0` |
| SGLang | `0009-feat-mtp-admit-remote-fwuff-TP1-draft.patch` | `87c46c03d1e1fe3c688d373ca4a8d09e6189a1e40e62f697e8b593c6af9ee51f` |
| KTransformers | `0001-build-pin-compact-MLA-kv_b-W8-runtime.patch` | `073883d2c0f5aea9a8ebe57a7a86918fd93b4ad478277c3119503ba211bc9e38` |
| KTransformers | `0002-build-pin-hybrid-MLA-kv_b-MTP-runtime.patch` | `01c191d47d66fcd1db21abbbfc9170b656e0777354f56c2d3298099d18b92142` |
| KTransformers | `0003-build-pin-fail-closed-hybrid-MTP-admission.patch` | `1dde225dbefb97f03391d457c89874c305b3e7091509597580a21b6d7ea4427e` |
| KTransformers | `0004-build-pin-coherent-compact-MLA-metadata.patch` | `ac8bd03b1ebe4eeabf6e4b551b682c546e4a135ae4ec095835f43066a74396e0` |
| KTransformers | `0005-build-pin-globally-coherent-MLA-metadata.patch` | `3c3957ae7ce5a9fe5aba1057af95521bc724e42b3c5f7354ae11ecb738d34164` |
| KTransformers | `0006-build-pin-MTP-Marlin-ownership-fix.patch` | `29488e599b2c2d5dd8e4dd8b6a8f90df73611b7c9478216335adfb7f6d1dd940` |

Patch 0009 passed the complete focused GLM-5.2 MTP unit module: 61 passed
and one CUDA-only test skipped. Its exact 38-tensor selector matched the live
immutable hybrid index, and read-only admission against the live hybrid and
AMXINT4 manifests passed with logical slots `(0, 1)` mapped to physical NUMA
nodes `(0, 0)`. AST comparison against the patch base confirmed that the
existing local TP2 admission and local selector did not change.

Validation on two RTX 3090/SM86 devices passed all 78 focused Marlin,
historical Triton, compact-loading, dispatch, and MTP tests. The earlier
explicit-Triton and explicit-Marlin TP2+MTP receipts bind the backend in the
top-level configuration, process specification, and exact launch environment.
They also bind the immutable hybrid manifest and machine-check exactly 79
matching compact modules per TP rank before timing can pass. Both smokes passed
semantic coherency, accepted 8/8 draft tokens, reproduced
`6fe701f56abd403cf97996be6d050eb6c0868dfa521b870e05583a541f259cd1`,
and completed cleanup. The matched MTP-off control is
`/var/lib/exo/benchmarks/glm52-w8-final-target-gate-20260725/glm52-tp2-local-benchmark-result.json`;
the explicit-Triton and explicit-Marlin receipts are
`/var/lib/exo/benchmarks/glm52-w8-explicit-triton-attested-gate-20260725/glm52-tp2-local-benchmark-result.json`
and
`/var/lib/exo/benchmarks/glm52-w8-explicit-marlin-attested-gate-20260725/glm52-tp2-local-benchmark-result.json`.
Only the explicit-Marlin receipt is relevant to the forward backend policy;
the MTP-off control must be rerun under Marlin with a 78-module-per-rank
runtime census before the representative quality comparison.
The immutable checkpoint is produced by
`scripts/materialize_glm52_hybrid_checkpoint.py`.
