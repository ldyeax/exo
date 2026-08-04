#!/usr/bin/env python3
"""Read-only admission checks for adding an RTX 5090 to the DSV4 host.

The checker deliberately does not launch CUDA work.  Run it with the DSV4
runtime Python after the card is installed, or provide a captured JSON
inventory while reviewing a proposed installation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, cast, final

type GateStatus = Literal["pass", "block", "manual"]

MINIMUM_5090_MEMORY_MIB = 31 * 1024
MINIMUM_3090_MEMORY_MIB = 23 * 1024
MINIMUM_CUDA_TOOLKIT = (12, 9)
MINIMUM_TORCH_CUDA = (12, 8)
MINIMUM_FLASHINFER = (0, 6, 9)
EXPECTED_3090_LOCATIONS = (("0000:d8:00.0", 1), ("0000:16:00.0", 0))
EXPECTED_5090_LOCATION = ("0000:4a:00.0", 0)
TARGET_LAYER_COUNT = 43


@final
@dataclass(frozen=True)
class GpuInventory:
    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_mib: int
    compute_capability: str
    numa_node: int | None
    max_link_width: int | None
    max_link_speed_gt_s: float | None
    current_link_width: int | None
    current_link_speed_gt_s: float | None


@final
@dataclass(frozen=True)
class ToolchainInventory:
    torch_version: str | None
    torch_cuda_version: str | None
    flashinfer_version: str | None
    triton_version: str | None
    cuda_toolkit_version: str | None


@final
@dataclass(frozen=True)
class Inventory:
    gpus: tuple[GpuInventory, ...]
    toolchain: ToolchainInventory
    topology: str
    pcie_p2p: str


@final
@dataclass(frozen=True)
class Gate:
    name: str
    status: GateStatus
    detail: str


@final
@dataclass(frozen=True)
class RankAssignment:
    pipeline_rank: int
    expected_gpu: str
    expected_pci_bus_id: str
    expected_numa_node: int
    detected_gpu_index: int | None
    detected_gpu_uuid: str | None
    target_layer_start_inclusive: int
    target_layer_end_exclusive: int
    target_layer_count: int


@final
class ParsedArguments(argparse.Namespace):
    inventory: Path | None
    launcher: Path
    json: bool

    def __init__(self, *, default_launcher: Path) -> None:
        super().__init__()
        self.inventory = None
        self.launcher = default_launcher
        self.json = False


def parse_version(value: str | None, *, parts: int = 2) -> tuple[int, ...] | None:
    if value is None:
        return None
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        return None
    parsed = tuple(int(part or 0) for part in match.groups(default="0"))
    return parsed[:parts]


def normalize_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"(?:(?P<domain>[0-9A-Fa-f]{4,8}):)?"
        r"(?P<bus>[0-9A-Fa-f]{2}):(?P<device>[0-9A-Fa-f]{2})\."
        r"(?P<function>[0-7])",
        value.strip(),
    )
    if match is None:
        raise ValueError(f"invalid PCI bus ID: {value!r}")
    domain = int(match.group("domain") or "0", 16)
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def read_optional_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def read_optional_link_speed(path: Path) -> float | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None
    match = re.match(r"([0-9]+(?:\.[0-9]+)?) GT/s PCIe", value)
    return float(match.group(1)) if match is not None else None


def run_command(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"{' '.join(arguments)} failed: {error}")
    return result.stdout.strip()


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_toolchain_inventory() -> ToolchainInventory:
    try:
        torch_output = run_command(
            (
                sys.executable,
                "-c",
                "import torch; print(torch.__version__); print(torch.version.cuda or '')",
            )
        ).splitlines()
    except (FileNotFoundError, RuntimeError):
        torch_version = None
        torch_cuda_version = None
    else:
        torch_version = torch_output[0] if torch_output else None
        torch_cuda_version = torch_output[1] if len(torch_output) > 1 else None

    try:
        nvcc_output = run_command(("nvcc", "--version"))
    except (FileNotFoundError, RuntimeError):
        cuda_toolkit_version = None
    else:
        match = re.search(r"release\s+(\d+\.\d+)", nvcc_output)
        cuda_toolkit_version = match.group(1) if match is not None else None

    return ToolchainInventory(
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        flashinfer_version=installed_version("flashinfer-python"),
        triton_version=installed_version("triton"),
        cuda_toolkit_version=cuda_toolkit_version,
    )


def collect_live_inventory() -> Inventory:
    query = run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    gpus: list[GpuInventory] = []
    for line in query.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {line!r}")
        index, name, uuid, raw_pci_bus_id, memory_total, capability = fields
        pci_bus_id = normalize_pci_bus_id(raw_pci_bus_id)
        sysfs_device = Path("/sys/bus/pci/devices") / pci_bus_id
        gpus.append(
            GpuInventory(
                index=int(index),
                name=name,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
                memory_total_mib=int(memory_total),
                compute_capability=capability,
                numa_node=read_optional_int(sysfs_device / "numa_node"),
                max_link_width=read_optional_int(sysfs_device / "max_link_width"),
                max_link_speed_gt_s=read_optional_link_speed(
                    sysfs_device / "max_link_speed"
                ),
                current_link_width=read_optional_int(
                    sysfs_device / "current_link_width"
                ),
                current_link_speed_gt_s=read_optional_link_speed(
                    sysfs_device / "current_link_speed"
                ),
            )
        )

    return Inventory(
        gpus=tuple(gpus),
        toolchain=collect_toolchain_inventory(),
        topology=run_command(("nvidia-smi", "topo", "-m")),
        pcie_p2p=run_command(("nvidia-smi", "topo", "-p2p", "p")),
    )


def require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError(f"{label} must use string keys")
    return cast(dict[str, object], mapping)


def require_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    return cast(list[object], value)


def optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def optional_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or null")
    return value


def optional_float(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number or null")
    return float(value)


def load_inventory(path: Path) -> Inventory:
    decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    root = require_mapping(decoded, label="root")
    raw_gpus = require_list(root.get("gpus"), label="gpus")
    gpus: list[GpuInventory] = []
    for position, raw_gpu in enumerate(raw_gpus):
        gpu = require_mapping(raw_gpu, label=f"gpus[{position}]")
        required_strings = ("name", "uuid", "pci_bus_id", "compute_capability")
        if any(not isinstance(gpu.get(name), str) for name in required_strings):
            raise TypeError(f"gpus[{position}] has a missing string field")
        index = gpu.get("index")
        memory_total_mib = gpu.get("memory_total_mib")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"gpus[{position}].index must be an integer")
        if not isinstance(memory_total_mib, int) or isinstance(memory_total_mib, bool):
            raise TypeError(f"gpus[{position}].memory_total_mib must be an integer")
        gpus.append(
            GpuInventory(
                index=index,
                name=str(gpu["name"]),
                uuid=str(gpu["uuid"]),
                pci_bus_id=normalize_pci_bus_id(str(gpu["pci_bus_id"])),
                memory_total_mib=memory_total_mib,
                compute_capability=str(gpu["compute_capability"]),
                numa_node=optional_int(
                    gpu.get("numa_node"), label=f"gpus[{position}].numa_node"
                ),
                max_link_width=optional_int(
                    gpu.get("max_link_width"),
                    label=f"gpus[{position}].max_link_width",
                ),
                max_link_speed_gt_s=optional_float(
                    gpu.get("max_link_speed_gt_s"),
                    label=f"gpus[{position}].max_link_speed_gt_s",
                ),
                current_link_width=optional_int(
                    gpu.get("current_link_width"),
                    label=f"gpus[{position}].current_link_width",
                ),
                current_link_speed_gt_s=optional_float(
                    gpu.get("current_link_speed_gt_s"),
                    label=f"gpus[{position}].current_link_speed_gt_s",
                ),
            )
        )

    raw_toolchain = require_mapping(root.get("toolchain"), label="toolchain")
    toolchain = ToolchainInventory(
        torch_version=optional_string(
            raw_toolchain.get("torch_version"), label="toolchain.torch_version"
        ),
        torch_cuda_version=optional_string(
            raw_toolchain.get("torch_cuda_version"),
            label="toolchain.torch_cuda_version",
        ),
        flashinfer_version=optional_string(
            raw_toolchain.get("flashinfer_version"),
            label="toolchain.flashinfer_version",
        ),
        triton_version=optional_string(
            raw_toolchain.get("triton_version"), label="toolchain.triton_version"
        ),
        cuda_toolkit_version=optional_string(
            raw_toolchain.get("cuda_toolkit_version"),
            label="toolchain.cuda_toolkit_version",
        ),
    )
    topology = root.get("topology", "")
    pcie_p2p = root.get("pcie_p2p", "")
    if not isinstance(topology, str) or not isinstance(pcie_p2p, str):
        raise TypeError("topology and pcie_p2p must be strings")
    return Inventory(tuple(gpus), toolchain, topology, pcie_p2p)


def version_gate(name: str, value: str | None, minimum: tuple[int, ...]) -> Gate:
    parsed = parse_version(value, parts=len(minimum))
    expected = ".".join(str(part) for part in minimum)
    if parsed is None:
        return Gate(name, "block", f"version is unavailable; require >= {expected}")
    if parsed < minimum:
        return Gate(name, "block", f"found {value}; require >= {expected}")
    return Gate(name, "pass", f"found {value}; require >= {expected}")


def find_gpu_by_pci_bus_id(
    inventory: Inventory, pci_bus_id: str
) -> GpuInventory | None:
    return next(
        (gpu for gpu in inventory.gpus if gpu.pci_bus_id == pci_bus_id),
        None,
    )


def build_rank_plan(inventory: Inventory) -> tuple[RankAssignment, ...]:
    specifications = (
        (0, "NVIDIA GeForce RTX 3090", "0000:d8:00.0", 1, 0, 18),
        (1, "NVIDIA GeForce RTX 3090", "0000:16:00.0", 0, 18, 35),
        (2, "NVIDIA GeForce RTX 5090", "0000:4a:00.0", 0, 35, 43),
    )
    assignments: list[RankAssignment] = []
    for (
        pipeline_rank,
        expected_gpu,
        pci_bus_id,
        numa_node,
        layer_start_inclusive,
        layer_end_exclusive,
    ) in specifications:
        detected = find_gpu_by_pci_bus_id(inventory, pci_bus_id)
        assignments.append(
            RankAssignment(
                pipeline_rank=pipeline_rank,
                expected_gpu=expected_gpu,
                expected_pci_bus_id=pci_bus_id,
                expected_numa_node=numa_node,
                detected_gpu_index=detected.index if detected is not None else None,
                detected_gpu_uuid=detected.uuid if detected is not None else None,
                target_layer_start_inclusive=layer_start_inclusive,
                target_layer_end_exclusive=layer_end_exclusive,
                target_layer_count=(layer_end_exclusive - layer_start_inclusive),
            )
        )
    return tuple(assignments)


def commissioning_profile() -> dict[str, object]:
    return {
        "parallelism": {"tensor": 1, "pipeline": 3, "expert": 1},
        "target_only": True,
        "speculative_decoding": False,
        "communication_overlap": False,
        "context_length": 524288,
        "max_total_tokens": 524288,
        "kv_cache_dtype": "fp8_e4m3",
        "swa_full_tokens_ratio": 0.0048828125,
        "cuda_graph": {"decode": "full", "prefill": "disabled"},
        "maximum_running_requests": 1,
        "flashinfer_cuda_arch_list": "8.6 12.0",
        "torch_cuda_arch_list": "8.6;12.0",
        "nccl_p2p_level": None,
        "cache_root": "/var/lib/exo/cache/dsv4-flash-opencode-pp3-sm86-sm120-v1",
    }


def unique_identity_gate(inventory: Inventory) -> Gate:
    indexes = tuple(gpu.index for gpu in inventory.gpus)
    uuids = tuple(gpu.uuid for gpu in inventory.gpus)
    pci_bus_ids = tuple(gpu.pci_bus_id for gpu in inventory.gpus)
    unique = (
        len(set(indexes)) == len(indexes)
        and len(set(uuids)) == len(uuids)
        and len(set(pci_bus_ids)) == len(pci_bus_ids)
        and all(uuid for uuid in uuids)
    )
    return Gate(
        "gpu-identities",
        "pass" if unique else "block",
        "GPU indexes, UUIDs, and PCI bus IDs must be present and unique",
    )


def gpu_location_gate(
    inventory: Inventory,
    *,
    name: str,
    expected_name: str,
    expected_locations: Sequence[tuple[str, int]],
) -> Gate:
    problems: list[str] = []
    for pci_bus_id, numa_node in expected_locations:
        gpu = find_gpu_by_pci_bus_id(inventory, pci_bus_id)
        if gpu is None:
            problems.append(f"missing {pci_bus_id}")
            continue
        if expected_name not in gpu.name.upper():
            problems.append(f"{pci_bus_id} is {gpu.name!r}")
        if gpu.numa_node != numa_node:
            problems.append(
                f"{pci_bus_id} reports NUMA {gpu.numa_node}, expected {numa_node}"
            )
    detail = (
        "; ".join(problems)
        if problems
        else "expected PCI bus IDs and NUMA nodes are present"
    )
    return Gate(name, "block" if problems else "pass", detail)


def rank_plan_gate(inventory: Inventory) -> Gate:
    assignments = build_rank_plan(inventory)
    identities_resolve = all(
        assignment.detected_gpu_index is not None
        and assignment.detected_gpu_uuid is not None
        for assignment in assignments
    )
    ordered_assignments = tuple(
        sorted(assignments, key=lambda assignment: assignment.pipeline_rank)
    )
    ranges_are_contiguous = (
        bool(ordered_assignments)
        and ordered_assignments[0].target_layer_start_inclusive == 0
        and ordered_assignments[-1].target_layer_end_exclusive == TARGET_LAYER_COUNT
        and all(
            previous.target_layer_end_exclusive
            == following.target_layer_start_inclusive
            for previous, following in pairwise(ordered_assignments)
        )
    )
    counts_are_exact = all(
        assignment.target_layer_count
        == assignment.target_layer_end_exclusive
        - assignment.target_layer_start_inclusive
        and assignment.target_layer_count > 0
        for assignment in ordered_assignments
    )
    valid = identities_resolve and ranges_are_contiguous and counts_are_exact
    return Gate(
        "pp3-rank-plan",
        "pass" if valid else "block",
        "rank order must resolve by PCI bus ID, never by mutable CUDA index; "
        "half-open target-layer ranges must cover [0, 43) contiguously and "
        "without overlap",
    )


def _unquote_shell_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] in {'"', "'"} and stripped[-1] == stripped[0]:
        return stripped[1:-1]
    return stripped


def launcher_default_value(launcher_text: str, variable: str) -> str | None:
    """Read one unambiguous top-level literal launcher default.

    This is intentionally a conservative source parser, not a shell evaluator.
    Assignments inside basic shell control-flow blocks, multiple assignments,
    unsets, and unknown/dynamic forms fail closed.
    """

    active_lines = tuple(
        line.split("#", maxsplit=1)[0].strip()
        for line in launcher_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    top_level_lines: list[str] = []
    control_depth = 0
    for line in active_lines:
        if re.match(r"^(?:fi|done|esac)\b|^\}", line):
            control_depth = max(control_depth - 1, 0)
        if control_depth == 0:
            top_level_lines.append(line)
        if re.match(r"^(?:if|case|for|while|until)\b", line) or re.match(
            r"^(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", line
        ):
            control_depth += 1
    if any(
        re.fullmatch(rf"unset\s+.*\b{re.escape(variable)}\b.*", line)
        for line in top_level_lines
    ):
        return None
    exact_assignment = re.compile(
        rf"^(?:export\s+)?{re.escape(variable)}\s*=\s*(?P<value>.*)$"
    )
    parameter_default = re.compile(rf"^\$\{{{re.escape(variable)}:-(?P<default>.*)\}}$")

    exact_values: list[str] = []
    exact_assignment_count = 0
    for line in top_level_lines:
        match = exact_assignment.fullmatch(line)
        if match is None:
            continue
        exact_assignment_count += 1
        raw_value = _unquote_shell_value(match.group("value"))
        default_match = parameter_default.fullmatch(raw_value)
        if default_match is not None:
            exact_values.append(default_match.group("default"))
        elif "$" not in raw_value and "`" not in raw_value:
            exact_values.append(raw_value)
    if exact_assignment_count:
        return (
            exact_values[0]
            if exact_assignment_count == len(exact_values) == 1
            else None
        )

    expansion = re.compile(rf"\$\{{{re.escape(variable)}:-(?P<default>[^}}]*)\}}")
    expansion_defaults = [
        match.group("default")
        for line in top_level_lines
        if (match := expansion.search(line)) is not None
    ]
    if expansion_defaults:
        return expansion_defaults[0] if len(expansion_defaults) == 1 else None

    if variable == "DSV4_NCCL_P2P_LEVEL":
        local_assignment = re.compile(r"^nccl_p2p_level\s*=\s*(?P<value>.*)$")
        literal_defaults: list[str] = []
        for line in top_level_lines:
            match = local_assignment.fullmatch(line)
            if match is None:
                continue
            raw_value = _unquote_shell_value(match.group("value"))
            if "$" not in raw_value and "`" not in raw_value:
                literal_defaults.append(raw_value)
        if literal_defaults:
            return literal_defaults[0] if len(literal_defaults) == 1 else None
    return None


def is_mixed_architecture_default(value: str | None) -> bool:
    if value is None:
        return False
    targets = tuple(target for target in re.split(r"[;\s]+", value) if target)
    return len(targets) == 2 and set(targets) == {"8.6", "12.0"}


def evaluate_inventory(inventory: Inventory, launcher_text: str) -> tuple[Gate, ...]:
    rtx5090s = tuple(gpu for gpu in inventory.gpus if "RTX 5090" in gpu.name.upper())
    rtx3090s = tuple(gpu for gpu in inventory.gpus if "RTX 3090" in gpu.name.upper())
    gates: list[Gate] = [unique_identity_gate(inventory)]

    gates.append(
        Gate(
            "gpu-count",
            "pass"
            if len(inventory.gpus) == 3 and len(rtx5090s) == 1 and len(rtx3090s) == 2
            else "block",
            f"found {len(inventory.gpus)} GPUs: {len(rtx5090s)} RTX 5090 and "
            f"{len(rtx3090s)} RTX 3090; require exactly 1 + 2",
        )
    )
    gates.extend(
        (
            gpu_location_gate(
                inventory,
                name="rtx3090-topology",
                expected_name="RTX 3090",
                expected_locations=EXPECTED_3090_LOCATIONS,
            ),
            gpu_location_gate(
                inventory,
                name="rtx5090-topology",
                expected_name="RTX 5090",
                expected_locations=(EXPECTED_5090_LOCATION,),
            ),
            rank_plan_gate(inventory),
        )
    )
    if len(rtx5090s) == 1:
        gpu = rtx5090s[0]
        gates.extend(
            (
                Gate(
                    "rtx5090-compute-capability",
                    "pass" if gpu.compute_capability == "12.0" else "block",
                    f"found {gpu.compute_capability}; require 12.0",
                ),
                Gate(
                    "rtx5090-memory",
                    "pass"
                    if gpu.memory_total_mib >= MINIMUM_5090_MEMORY_MIB
                    else "block",
                    f"found {gpu.memory_total_mib} MiB; require at least "
                    f"{MINIMUM_5090_MEMORY_MIB} MiB",
                ),
                Gate(
                    "rtx5090-pcie-max-link",
                    "pass"
                    if gpu.max_link_width == 16
                    and gpu.max_link_speed_gt_s is not None
                    and gpu.max_link_speed_gt_s >= 32.0
                    else "block",
                    f"max link is x{gpu.max_link_width or '?'} at "
                    f"{gpu.max_link_speed_gt_s or '?'} GT/s; require PCIe 5.0 x16",
                ),
                Gate(
                    "rtx5090-pcie-current-link",
                    "pass"
                    if gpu.current_link_width == 16
                    and gpu.current_link_speed_gt_s is not None
                    and gpu.current_link_speed_gt_s >= 32.0
                    else "block",
                    f"current link is x{gpu.current_link_width or '?'} at "
                    f"{gpu.current_link_speed_gt_s or '?'} GT/s; require the "
                    "installed card to negotiate PCIe 5.0 x16",
                ),
            )
        )

    undersized_3090s = tuple(
        gpu for gpu in rtx3090s if gpu.memory_total_mib < MINIMUM_3090_MEMORY_MIB
    )
    gates.append(
        Gate(
            "rtx3090-memory",
            "pass" if len(rtx3090s) >= 2 and not undersized_3090s else "block",
            "both retained RTX 3090s must expose at least "
            f"{MINIMUM_3090_MEMORY_MIB} MiB",
        )
    )
    gates.extend(
        (
            version_gate(
                "cuda-toolkit",
                inventory.toolchain.cuda_toolkit_version,
                MINIMUM_CUDA_TOOLKIT,
            ),
            version_gate(
                "torch-cuda",
                inventory.toolchain.torch_cuda_version,
                MINIMUM_TORCH_CUDA,
            ),
            version_gate(
                "flashinfer",
                inventory.toolchain.flashinfer_version,
                MINIMUM_FLASHINFER,
            ),
        )
    )

    flashinfer_arch_default = launcher_default_value(
        launcher_text, "DSV4_FLASHINFER_CUDA_ARCH_LIST"
    )
    torch_arch_default = launcher_default_value(
        launcher_text, "DSV4_TORCH_CUDA_ARCH_LIST"
    )
    nccl_p2p_default = launcher_default_value(launcher_text, "DSV4_NCCL_P2P_LEVEL")
    mixed_arch_defaults = is_mixed_architecture_default(
        flashinfer_arch_default
    ) and is_mixed_architecture_default(torch_arch_default)
    gates.append(
        Gate(
            "launcher-mixed-arch-defaults",
            "pass" if mixed_arch_defaults else "block",
            "statically parsed literal defaults are FlashInfer="
            f"{flashinfer_arch_default!r}, Torch={torch_arch_default!r}; require "
            "both SM86 and SM120 targets; this is advisory until a future "
            "launcher exposes a side-effect-free resolved contract",
        )
    )
    gates.append(
        Gate(
            "launcher-nccl-auto-default",
            "pass" if nccl_p2p_default == "" else "block",
            f"statically parsed DSV4 NCCL P2P literal is {nccl_p2p_default!r}; "
            "require an explicit empty value so NCCL chooses the mixed PCIe "
            "topology; runtime resolution remains a manual commissioning gate",
        )
    )
    gates.extend(
        (
            Gate(
                "torch-sm120-compile-runtime",
                "manual",
                "require an installed-GPU receipt proving an empty-cache Torch "
                "SM120 compile and execution against the detected RTX 5090 UUID; "
                "the CUDA build version is not runtime evidence",
            ),
            Gate(
                "triton-sm120-compile-runtime",
                "manual",
                "require an installed-GPU receipt proving an empty-cache Triton "
                "SM120 compile and execution against the detected RTX 5090 UUID; "
                "the package version is not runtime evidence",
            ),
            Gate(
                "pcie-p2p-bandwidth",
                "manual",
                "validate every proposed rank pair with `nvidia-smi topo -p2p p` "
                "and NVIDIA nvbandwidth; do not infer P2P from PCIe generation",
            ),
            Gate(
                "power-and-cooling",
                "manual",
                "verify PSU rail/cable capacity and sustained thermals for a 575 W "
                "card alongside both 3090s",
            ),
            Gate(
                "oscar-int2-sm120-parity-quality",
                "manual",
                "require installed-GPU receipts proving the KV-only Oscar INT2 "
                "history writer, C4 writer/scorer, protected-BF16-SWA mixed "
                "decode/extend/prefill paths, and CUDA graph replay on SM120; "
                "then qualify artifact-bound parity and task quality at 64K, "
                "128K, 256K, and 524K before advertising the full window",
            ),
        )
    )
    return tuple(gates)


def inventory_as_json(inventory: Inventory, gates: Sequence[Gate]) -> str:
    automatic_blocked = any(gate.status == "block" for gate in gates)
    manual_pending = any(gate.status == "manual" for gate in gates)
    admitted = not automatic_blocked and not manual_pending
    return json.dumps(
        {
            "schema": "dsv4-rtx5090-preflight-v4",
            "gpus": [asdict(gpu) for gpu in inventory.gpus],
            "toolchain": asdict(inventory.toolchain),
            "topology": inventory.topology,
            "pcie_p2p": inventory.pcie_p2p,
            "commissioning_profile": commissioning_profile(),
            "proposed_rank_order": [
                asdict(assignment) for assignment in build_rank_plan(inventory)
            ],
            "gates": [asdict(gate) for gate in gates],
            "automatic_blocked": automatic_blocked,
            "manual_pending": manual_pending,
            "admitted": admitted,
            "blocked": not admitted,
        },
        indent=2,
        sort_keys=True,
    )


def parse_arguments(arguments: Sequence[str]) -> ParsedArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    default_launcher = Path(__file__).with_name("dsv4_flash_0731_tp2_dwagon.sh")
    parser.add_argument(
        "--inventory",
        type=Path,
        help="read a captured JSON inventory instead of querying live hardware",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=default_launcher,
        help="launcher source to check for mixed-architecture and PP3 contracts",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    namespace = ParsedArguments(default_launcher=default_launcher)
    return parser.parse_args(arguments, namespace=namespace)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(sys.argv[1:] if arguments is None else arguments)
    try:
        inventory = (
            load_inventory(parsed.inventory)
            if parsed.inventory is not None
            else collect_live_inventory()
        )
        launcher_text = parsed.launcher.read_text(encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"RTX 5090 preflight input failed: {error}", file=sys.stderr)
        return 2
    gates = evaluate_inventory(inventory, launcher_text)
    if parsed.json:
        print(inventory_as_json(inventory, gates))
    else:
        for gate in gates:
            print(f"{gate.status.upper():6} {gate.name}: {gate.detail}")
        print("\nCaptured topology:\n" + (inventory.topology or "<unavailable>"))
        print("\nCaptured PCIe P2P matrix:\n" + (inventory.pcie_p2p or "<unavailable>"))
    return 1 if any(gate.status != "pass" for gate in gates) else 0


if __name__ == "__main__":
    raise SystemExit(main())
