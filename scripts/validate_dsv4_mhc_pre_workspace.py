#!/usr/bin/env python3
"""Validate caller-owned DSV4 mHC-pre output at production dimensions."""

from __future__ import annotations

import hashlib

import torch

from sglang.srt.layers.mhc import mhc_pre


TOKENS = 4096
TAIL_TOKENS = 4032
HC_MULT = 4
HIDDEN_SIZE = 7168
MAIN_Q_ELEMENTS = TOKENS * 128 * 512


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def run(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_pre(
        residual=residual,
        fn=fn,
        hc_scale=scale,
        hc_base=base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
        layer_input_output=output,
    )


def main() -> None:
    torch.manual_seed(42)
    residual = torch.randn(
        (TOKENS, HC_MULT, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fn = (
        torch.randn(
            (HC_MULT * 2 + HC_MULT * HC_MULT, HC_MULT * HIDDEN_SIZE),
            dtype=torch.float32,
            device="cuda",
        )
        * 0.01
    )
    scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    base = torch.zeros(
        (HC_MULT * 2 + HC_MULT * HC_MULT,),
        dtype=torch.float32,
        device="cuda",
    )
    norm_weight = torch.randn(
        (HIDDEN_SIZE,), dtype=torch.bfloat16, device="cuda"
    )

    torch.cuda.synchronize()
    ordinary_baseline = torch.cuda.memory_allocated()
    reference_post, reference_comb, reference_output = run(
        residual, fn, scale, base, norm_weight, None
    )
    torch.cuda.synchronize()
    ordinary_live_delta = torch.cuda.memory_allocated() - ordinary_baseline

    workspace = torch.empty(
        (MAIN_Q_ELEMENTS,), dtype=torch.bfloat16, device="cuda"
    )
    output_elements = TOKENS * HIDDEN_SIZE
    caller_output = workspace[-output_elements:].view(TOKENS, HIDDEN_SIZE)
    caller_pointer = caller_output.data_ptr()
    caller_baseline = torch.cuda.memory_allocated()
    actual_post, actual_comb, actual_output = run(
        residual, fn, scale, base, norm_weight, caller_output
    )
    torch.cuda.synchronize()
    caller_live_delta = torch.cuda.memory_allocated() - caller_baseline

    if actual_output.data_ptr() != caller_pointer:
        raise AssertionError("mHC pre rebound the caller-owned output")
    for name, reference, actual in (
        ("post", reference_post, actual_post),
        ("comb", reference_comb, actual_comb),
        ("layer_input", reference_output, actual_output),
    ):
        if not torch.equal(reference.view(torch.uint8), actual.view(torch.uint8)):
            difference = (reference.float() - actual.float()).abs()
            raise AssertionError(
                f"mHC pre {name} mismatch: max_abs={difference.max().item()}"
            )

    protected_start_bytes = caller_output.data_ptr() - workspace.data_ptr()
    saved_output = actual_output.clone()
    workspace.view(torch.uint8)[:protected_start_bytes].zero_()
    torch.cuda.synchronize()
    if not torch.equal(saved_output.view(torch.uint8), actual_output.view(torch.uint8)):
        raise AssertionError("prefix workspace write corrupted protected mHC output")

    # The production scheduler can create a non-full final chunk (for
    # example, 4,032 rows after page/alignment accounting).  Its live main-Q
    # view is shorter than the persistent allocation, while mHC still places
    # its output at the end of the backing workspace.
    tail_output_elements = TAIL_TOKENS * HIDDEN_SIZE
    tail_q_elements = TAIL_TOKENS * 128 * 512
    tail_output = workspace[-tail_output_elements:].view(
        TAIL_TOKENS, HIDDEN_SIZE
    )
    tail_start_bytes = tail_output.data_ptr() - workspace.data_ptr()
    tail_end_bytes = (
        tail_start_bytes + tail_output.numel() * tail_output.element_size()
    )
    workspace_bytes = workspace.numel() * workspace.element_size()
    tail_q_bytes = tail_q_elements * workspace.element_size()
    if tail_end_bytes != workspace_bytes:
        raise AssertionError(
            "tail mHC output does not end at the backing workspace boundary"
        )
    if tail_end_bytes == tail_q_bytes:
        raise AssertionError(
            "tail oracle did not distinguish the live Q view from its backing "
            "workspace"
        )
    if tail_start_bytes >= tail_q_bytes:
        raise AssertionError(
            "tail oracle does not exercise the protected live-Q overlap"
        )

    saved_bytes = ordinary_live_delta - caller_live_delta
    expected_saved_bytes = output_elements * torch.bfloat16.itemsize
    if saved_bytes != expected_saved_bytes:
        raise AssertionError(
            "unexpected live-allocation saving: "
            f"expected={expected_saved_bytes}, actual={saved_bytes}"
        )

    print(f"output_shape={tuple(actual_output.shape)}")
    print(f"pointer_preserved=True byte_exact=True")
    print(f"protected_suffix_start_bytes={protected_start_bytes}")
    print(f"tail_protected_suffix_start_bytes={tail_start_bytes}")
    print(f"tail_q_bytes={tail_q_bytes}")
    print(f"workspace_bytes={workspace_bytes}")
    print(f"ordinary_live_delta_bytes={ordinary_live_delta}")
    print(f"caller_live_delta_bytes={caller_live_delta}")
    print(f"saved_live_bytes={saved_bytes}")
    print(f"output_sha256={tensor_sha256(actual_output)}")


if __name__ == "__main__":
    main()
