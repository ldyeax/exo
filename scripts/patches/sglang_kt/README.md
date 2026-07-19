# Exo SGLang-KTransformers source patches

These mail patches make the GLM-4.7-Flash BF16 runtime source reproducible
without relying on unpublished dependency forks.

Apply them with `scripts/prepare_sglang_kt_source.py`. The script verifies every
base and result commit and applies patches in this order:

1. `0001-feat-fail-closed-on-GLM-Flash-KT-coverage.patch` converts SGLang
   `5d6bef9f61637aaeaf047bf8209def2af3eaa83f` to
   `41d4d300a21fd2f486681d56f1017789dfb355fe`. The result also fixes the
   BF16 benchmark loader so resident gate/up/down projections cannot silently
   remain random or be copied into the wrong fused-weight slots.
2. `0002-build-pin-GLM-Flash-KT-registration.patch` converts KTransformers
   `8e46e5896c3d993a1285052f2618f5a9f01882d4` to
   `7e70d7518edd26af6a0638593037d68c9b6bd6bf` by recording that SGLang
   gitlink.

The KTransformers patch deliberately leaves `.gitmodules` unchanged. A plain
recursive clone cannot fetch the Exo-only SGLang result commit from the upstream
submodule remote; the preparation script initializes the upstream base before
applying and verifying both patches locally.
