from __future__ import annotations

import json
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import pytest

from scripts.dsv4_flash_rtx5090_preflight import (
    TARGET_LAYER_COUNT,
    GpuInventory,
    Inventory,
    ToolchainInventory,
    build_rank_plan,
    evaluate_inventory,
    main,
    normalize_pci_bus_id,
    parse_version,
)

FUTURE_LAUNCHER_TEXT = """\
export DSV4_FLASHINFER_CUDA_ARCH_LIST="${DSV4_FLASHINFER_CUDA_ARCH_LIST:-8.6 12.0}"
export DSV4_TORCH_CUDA_ARCH_LIST="${DSV4_TORCH_CUDA_ARCH_LIST:-8.6;12.0}"
export DSV4_NCCL_P2P_LEVEL="${DSV4_NCCL_P2P_LEVEL:-}"
"""


def qualified_inventory() -> Inventory:
    return Inventory(
        gpus=(
            GpuInventory(
                0,
                "NVIDIA GeForce RTX 3090",
                "GPU-3090-a",
                "0000:16:00.0",
                24576,
                "8.6",
                0,
                16,
                16.0,
                16,
                16.0,
            ),
            GpuInventory(
                1,
                "NVIDIA GeForce RTX 3090",
                "GPU-3090-b",
                "0000:d8:00.0",
                24576,
                "8.6",
                1,
                16,
                16.0,
                16,
                16.0,
            ),
            GpuInventory(
                2,
                "NVIDIA GeForce RTX 5090",
                "GPU-5090",
                "0000:4a:00.0",
                32607,
                "12.0",
                0,
                16,
                32.0,
                16,
                32.0,
            ),
        ),
        toolchain=ToolchainInventory(
            torch_version="2.9.1+cu128",
            torch_cuda_version="12.8",
            flashinfer_version="0.6.9",
            triton_version="3.5.1",
            cuda_toolkit_version="13.1",
        ),
        topology="GPU0 GPU1 GPU2",
        pcie_p2p="GPU0 GPU1 GPU2",
    )


def gate_statuses(
    inventory: Inventory, launcher_text: str = FUTURE_LAUNCHER_TEXT
) -> dict[str, str]:
    return {
        gate.name: gate.status for gate in evaluate_inventory(inventory, launcher_text)
    }


def test_qualified_inventory_passes_automatic_gates() -> None:
    statuses = gate_statuses(qualified_inventory())

    assert all(status != "block" for status in statuses.values())
    assert statuses["pcie-p2p-bandwidth"] == "manual"
    assert statuses["power-and-cooling"] == "manual"
    assert statuses["oscar-int2-sm120-parity-quality"] == "manual"
    assert statuses["torch-sm120-compile-runtime"] == "manual"
    assert statuses["triton-sm120-compile-runtime"] == "manual"


def test_launcher_effective_defaults_must_be_mixed_arch_and_nccl_auto() -> None:
    statuses = gate_statuses(
        qualified_inventory(),
        """\
DSV4_FLASHINFER_CUDA_ARCH_LIST=8.6
DSV4_TORCH_CUDA_ARCH_LIST=8.6
DSV4_NCCL_P2P_LEVEL=NVL
""",
    )

    assert statuses["launcher-mixed-arch-defaults"] == "block"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_commented_launcher_defaults_do_not_qualify() -> None:
    statuses = gate_statuses(
        qualified_inventory(),
        "\n".join(f"# {line}" for line in FUTURE_LAUNCHER_TEXT.splitlines()),
    )

    assert statuses["launcher-mixed-arch-defaults"] == "block"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_launcher_variable_names_without_assignments_do_not_qualify() -> None:
    statuses = gate_statuses(
        qualified_inventory(),
        """\
echo DSV4_FLASHINFER_CUDA_ARCH_LIST
echo DSV4_TORCH_CUDA_ARCH_LIST
echo DSV4_NCCL_P2P_LEVEL
""",
    )

    assert statuses["launcher-mixed-arch-defaults"] == "block"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_obsolete_contract_marker_alone_does_not_qualify() -> None:
    statuses = gate_statuses(qualified_inventory(), "DSV4_RTX5090_PP3_CONTRACT_V1=1\n")

    assert statuses["launcher-mixed-arch-defaults"] == "block"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_unreachable_or_later_clobbered_defaults_do_not_qualify() -> None:
    unreachable = "if false; then\n" + FUTURE_LAUNCHER_TEXT + "fi\n"
    unreachable_statuses = gate_statuses(qualified_inventory(), unreachable)

    assert unreachable_statuses["launcher-mixed-arch-defaults"] == "block"
    assert unreachable_statuses["launcher-nccl-auto-default"] == "block"

    clobbered = FUTURE_LAUNCHER_TEXT + "\nunset DSV4_TORCH_CUDA_ARCH_LIST\n"
    clobbered_statuses = gate_statuses(qualified_inventory(), clobbered)

    assert clobbered_statuses["launcher-mixed-arch-defaults"] == "block"


def test_mixed_arch_defaults_do_not_excuse_an_nvl_default() -> None:
    launcher = FUTURE_LAUNCHER_TEXT.replace(
        "${DSV4_NCCL_P2P_LEVEL:-}", "${DSV4_NCCL_P2P_LEVEL:-NVL}"
    )
    statuses = gate_statuses(qualified_inventory(), launcher)

    assert statuses["launcher-mixed-arch-defaults"] == "pass"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_checked_in_base_launcher_defaults_fail_for_mixed_hardware() -> None:
    launcher = (Path(__file__).parents[1] / "dsv4_flash_0731_tp2_dwagon.sh").read_text(
        encoding="utf-8"
    )
    statuses = gate_statuses(qualified_inventory(), launcher)

    assert statuses["launcher-mixed-arch-defaults"] == "block"
    assert statuses["launcher-nccl-auto-default"] == "block"


def test_wrong_compute_capability_and_pcie_link_are_blockers() -> None:
    baseline = qualified_inventory()
    wrong_5090 = GpuInventory(
        2,
        "NVIDIA GeForce RTX 5090",
        "GPU-5090",
        "0000:4a:00.0",
        32607,
        "10.0",
        0,
        8,
        16.0,
        8,
        16.0,
    )
    inventory = Inventory(
        (*baseline.gpus[:2], wrong_5090),
        baseline.toolchain,
        baseline.topology,
        baseline.pcie_p2p,
    )
    statuses = gate_statuses(inventory)

    assert statuses["rtx5090-compute-capability"] == "block"
    assert statuses["rtx5090-pcie-max-link"] == "block"
    assert statuses["rtx5090-pcie-current-link"] == "block"


def test_downtrained_current_pcie_link_blocks_even_when_maximum_is_valid() -> None:
    baseline = qualified_inventory()
    downtrained_5090 = replace(
        baseline.gpus[2],
        current_link_width=8,
        current_link_speed_gt_s=16.0,
    )
    inventory = Inventory(
        (*baseline.gpus[:2], downtrained_5090),
        baseline.toolchain,
        baseline.topology,
        baseline.pcie_p2p,
    )
    statuses = gate_statuses(inventory)

    assert statuses["rtx5090-pcie-max-link"] == "pass"
    assert statuses["rtx5090-pcie-current-link"] == "block"


def test_missing_5090_is_a_blocker_without_crashing_specific_gates() -> None:
    baseline = qualified_inventory()
    inventory = Inventory(
        baseline.gpus[:2],
        baseline.toolchain,
        baseline.topology,
        baseline.pcie_p2p,
    )
    statuses = gate_statuses(inventory)

    assert statuses["gpu-count"] == "block"
    assert "rtx5090-compute-capability" not in statuses


def test_expected_slots_and_unique_gpu_identities_are_enforced() -> None:
    baseline = qualified_inventory()
    misplaced_5090 = GpuInventory(
        2,
        "NVIDIA GeForce RTX 5090",
        "GPU-3090-a",
        "0000:65:00.0",
        32607,
        "12.0",
        1,
        16,
        32.0,
        16,
        32.0,
    )
    inventory = Inventory(
        (*baseline.gpus[:2], misplaced_5090),
        baseline.toolchain,
        baseline.topology,
        baseline.pcie_p2p,
    )
    statuses = gate_statuses(inventory)

    assert statuses["gpu-identities"] == "block"
    assert statuses["rtx5090-topology"] == "block"
    assert statuses["pp3-rank-plan"] == "block"


def test_rank_plan_is_stable_when_cuda_indexes_change() -> None:
    baseline = qualified_inventory()
    reordered = Inventory(
        (
            replace(baseline.gpus[0], index=2),
            replace(baseline.gpus[1], index=0),
            replace(baseline.gpus[2], index=1),
        ),
        baseline.toolchain,
        baseline.topology,
        baseline.pcie_p2p,
    )

    plan = build_rank_plan(reordered)

    assert [assignment.detected_gpu_index for assignment in plan] == [0, 2, 1]


def test_rank_plan_uses_explicit_half_open_ranges_and_counts() -> None:
    plan = build_rank_plan(qualified_inventory())

    assert [
        (
            assignment.target_layer_start_inclusive,
            assignment.target_layer_end_exclusive,
            assignment.target_layer_count,
        )
        for assignment in plan
    ] == [(0, 18, 18), (18, 35, 17), (35, 43, 8)]


def test_rank_plan_is_contiguous_non_overlapping_and_covers_every_layer() -> None:
    plan = build_rank_plan(qualified_inventory())

    assert plan[0].target_layer_start_inclusive == 0
    assert plan[-1].target_layer_end_exclusive == TARGET_LAYER_COUNT
    assert all(
        previous.target_layer_end_exclusive == following.target_layer_start_inclusive
        for previous, following in pairwise(plan)
    )
    assert all(
        previous.target_layer_end_exclusive <= following.target_layer_start_inclusive
        for previous, following in pairwise(plan)
    )
    covered_layers = [
        layer
        for assignment in plan
        for layer in range(
            assignment.target_layer_start_inclusive,
            assignment.target_layer_end_exclusive,
        )
    ]
    assert covered_layers == list(range(TARGET_LAYER_COUNT))
    assert len(covered_layers) == len(set(covered_layers))


def test_manual_quality_gate_is_oscar_int2_sm120_not_generic_fp8() -> None:
    gates = {
        gate.name: gate
        for gate in evaluate_inventory(qualified_inventory(), FUTURE_LAUNCHER_TEXT)
    }

    assert "fp8-kv-quality" not in gates
    quality_gate = gates["oscar-int2-sm120-parity-quality"]
    assert quality_gate.status == "manual"
    assert "KV-only Oscar INT2" in quality_gate.detail
    assert "protected-BF16-SWA" in quality_gate.detail
    assert "decode/extend/prefill" in quality_gate.detail
    assert "CUDA graph replay" in quality_gate.detail
    assert all(
        length in quality_gate.detail for length in ("64K", "128K", "256K", "524K")
    )


def test_toolchain_version_floors_are_enforced() -> None:
    baseline = qualified_inventory()
    inventory = Inventory(
        baseline.gpus,
        ToolchainInventory(
            torch_version="2.7.0",
            torch_cuda_version="12.7",
            flashinfer_version="0.6.8",
            triton_version="3.3.0",
            cuda_toolkit_version="12.8",
        ),
        baseline.topology,
        baseline.pcie_p2p,
    )
    statuses = gate_statuses(inventory)

    assert statuses["cuda-toolkit"] == "block"
    assert statuses["torch-cuda"] == "block"
    assert statuses["flashinfer"] == "block"


def test_version_and_pci_parsers_accept_runtime_formats() -> None:
    assert parse_version("2.9.1+cu128", parts=3) == (2, 9, 1)
    assert parse_version("release 13.1, V13.1.80") == (13, 1)
    assert normalize_pci_bus_id("00000000:4A:00.0") == "0000:4a:00.0"


def test_json_cli_accepts_a_captured_inventory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inventory = qualified_inventory()
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "gpus": [asdict(gpu) for gpu in inventory.gpus],
                "toolchain": asdict(inventory.toolchain),
                "topology": inventory.topology,
                "pcie_p2p": inventory.pcie_p2p,
            }
        ),
        encoding="utf-8",
    )
    launcher_path = tmp_path / "future-launcher.sh"
    launcher_path.write_text(FUTURE_LAUNCHER_TEXT, encoding="utf-8")

    return_code = main(
        (
            "--inventory",
            str(inventory_path),
            "--launcher",
            str(launcher_path),
            "--json",
        )
    )

    assert return_code == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["schema"] == "dsv4-rtx5090-preflight-v4"
    assert report["automatic_blocked"] is False
    assert report["manual_pending"] is True
    assert report["admitted"] is False
    assert report["blocked"] is True
    assert report["commissioning_profile"]["parallelism"] == {
        "tensor": 1,
        "pipeline": 3,
        "expert": 1,
    }
    assert [rank["expected_pci_bus_id"] for rank in report["proposed_rank_order"]] == [
        "0000:d8:00.0",
        "0000:16:00.0",
        "0000:4a:00.0",
    ]
    assert [
        (
            rank["target_layer_start_inclusive"],
            rank["target_layer_end_exclusive"],
            rank["target_layer_count"],
        )
        for rank in report["proposed_rank_order"]
    ] == [(0, 18, 18), (18, 35, 17), (35, 43, 8)]
