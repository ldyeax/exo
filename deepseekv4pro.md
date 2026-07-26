# DeepSeek V4 Pro notes

- Use DSpark.

## Local modified inference sources

The OSDI26 GLM-5.2 runtime sources are checked out inside this repository as
nested local submodules:

- `vendor/ktransformers` is pinned to integration commit
  `45a3a797658140dcf8426cd3f2b2c6c969f8f5d8`. The code-bearing
  KTransformers patch terminal is
  `2ba756c942a62de981a0f8d55ab7d1aea3ad5d9a`; the integration commit only
  changes the nested SGLang URL and advances its gitlink.
- `vendor/ktransformers/third_party/sglang` is pinned to
  `720b40b2783b1a515134f4ab9fe820931cfbee36`, which contains all nine
  recorded SGLang patches.

These are local-only submodules for now. Their URLs point to durable bare
repositories under `/var/lib/exo/sources`; replace those URLs when remote
forks are created.

The KTransformers history adds the AMXINT4 expert runtime used by the hybrid
path, fine-grained AMX decode dependencies, BF16 expert staging/export for SLP
and SmallEP, immutable shared host-weight mappings and leases for P/D, and
pins the integrated SGLang runtime.

The SGLang history adds GLM-5.2 TP/PP integration, two-batch attention/CPU-MoE
overlap, layer-78 MTP with persistent AMXINT4 experts, bounded SLP and
SmallEP/P-D plumbing, direct compact W8A16 MLA `kv_b_proj` execution on
Ampere, coherent FlashInfer metadata, shared Marlin module ownership, and the
standalone `fwuff` TP1 MTP drafter. New compact launches are Marlin-only;
the retained Triton source is historical cross-check code, not the supported
policy.

The checkpoint layout keeps routed and MTP experts in AMXINT4, large
GPU-resident matrices in compact weight-only INT8, and norms, sensitive
scalars, activations, and the initial KV cache in BF16. Do not expand compact
weights back to BF16 at load.
