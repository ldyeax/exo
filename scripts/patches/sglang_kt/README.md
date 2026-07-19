# Exo SGLang-KTransformers source patches

These mail patches make the GLM-4.7-Flash BF16 runtime source reproducible
without relying on unpublished dependency forks.

Apply them with `scripts/prepare_sglang_kt_source.py`. The script verifies every
base and result commit and applies patches in this order:

1. `0001-feat-fail-closed-on-GLM-Flash-KT-coverage.patch` converts SGLang
   `5d6bef9f61637aaeaf047bf8209def2af3eaa83f` to
   `42504e59810130460fc24fdd17ef534cb8278a4b`. The result also fixes the
   BF16 benchmark loader so resident gate/up/down projections cannot silently
   remain random or be copied into the wrong fused-weight slots, and initializes
   the non-hash GLM sparse-MoE state required by inherited forward methods.
2. `0002-build-pin-GLM-Flash-KT-registration.patch` converts KTransformers
   `8e46e5896c3d993a1285052f2618f5a9f01882d4` to
   `6e0a4480936effa7bf0ece429f78a00b29932bec` by recording that SGLang
   gitlink.

The KTransformers patch deliberately leaves `.gitmodules` unchanged. A plain
recursive clone cannot fetch the Exo-only SGLang result commit from the upstream
submodule remote; the preparation script initializes the upstream base before
applying and verifying both patches locally.
