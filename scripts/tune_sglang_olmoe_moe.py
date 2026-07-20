#!/usr/bin/env python3
"""Tune the pinned SGLang BF16 Triton MoE kernels for OLMoE on RTX 3090.

The serving configurations exercised by this tuner are deliberately fixed:
EP1 uses all 64 experts with a TP-sharded intermediate width of 512, while
EP2 maps one contiguous 32-expert half onto each rank and uses the full 1024
intermediate width.  Runtime imports are kept out of module import and
``--dry-run`` so the workload and artifact contract can be audited without a
CUDA installation.

Generated files are candidate configurations.  They are not production
admission evidence until a controlled serving A/B confirms the improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import stat
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

OLMOE_HIDDEN_SIZE: Final = 2_048
OLMOE_GLOBAL_EXPERTS: Final = 64
OLMOE_TOP_K: Final = 8
OLMOE_BATCH_ANCHORS: Final = (1, 8, 128, 1_024)
OLMOE_ROUTE_SEEDS: Final = (20_260_720, 20_260_721, 20_260_722)
DEFAULT_WARMUP_ITERATIONS: Final = 5
DEFAULT_MEASUREMENT_ITERATIONS: Final = 20
DEFAULT_INDEPENDENT_SAMPLES: Final = 5
DEFAULT_RELATIVE_L1_TOLERANCE: Final = 0.02
DEFAULT_MAX_ABSOLUTE_TOLERANCE: Final = 0.02
MINIMUM_STABLE_IMPROVEMENT: Final = 0.05
PINNED_TRITON_VERSION: Final = "3.5.1"
RTX3090_DEVICE_NAME: Final = "NVIDIA GeForce RTX 3090"
_CONFIG_VERSION_DIRECTORY: Final = "triton_3_5_1"
_RECEIPT_NAME: Final = "tuning-receipt.json"
_MANIFEST_NAME: Final = "manifest.json"
_FUSED_MOE_PARAMETER_NAMES: Final = (
    "A",
    "B",
    "bias",
    "C",
    "A_scale",
    "B_scale",
    "B_zp",
    "topk_weights",
    "topk_ids",
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
    "mul_routed_weight",
    "top_k",
    "config",
    "compute_type",
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_channel_quant",
    "block_shape",
    "no_combine",
    "a_use_tma",
    "b_use_tma",
    "c_sorted",
    "filter_expert",
)
_FUSED_MOE_OPTIONAL_DEFAULTS: Final = {
    "block_shape": None,
    "no_combine": False,
    "a_use_tma": False,
    "b_use_tma": False,
    "c_sorted": False,
    "filter_expert": True,
}

KernelStage = Literal["gate_up", "down"]


class KernelConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


_FALLBACK_CONFIG: Final[KernelConfig] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}
_RTX3090_CANDIDATE_TUPLES: Final = (
    (64, 64, 128, 8, 4, 2),
    (16, 32, 64, 1, 4, 2),
    (16, 64, 64, 1, 4, 2),
    (16, 64, 128, 1, 4, 2),
    (16, 32, 128, 1, 4, 2),
    (16, 64, 128, 1, 4, 3),
)


@dataclass(frozen=True)
class OlmoeTopology:
    ep_size: Literal[1, 2]
    local_experts: int
    intermediate_size: int
    rank_count: int

    def __post_init__(self) -> None:
        expected = {
            1: (64, 512, 1),
            2: (32, 1_024, 2),
        }[self.ep_size]
        if (self.local_experts, self.intermediate_size, self.rank_count) != expected:
            raise ValueError(
                f"EP{self.ep_size} must use exact OLMoE shape {expected!r}"
            )

    @property
    def gate_up_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.local_experts,
            2 * self.intermediate_size,
            OLMOE_HIDDEN_SIZE,
        )

    @property
    def down_weight_shape(self) -> tuple[int, int, int]:
        return (
            self.local_experts,
            OLMOE_HIDDEN_SIZE,
            self.intermediate_size,
        )


OLMOE_TOPOLOGIES: Final = (
    OlmoeTopology(ep_size=1, local_experts=64, intermediate_size=512, rank_count=1),
    OlmoeTopology(ep_size=2, local_experts=32, intermediate_size=1_024, rank_count=2),
)


@dataclass(frozen=True)
class TuningSpec:
    batch_anchors: tuple[int, ...] = OLMOE_BATCH_ANCHORS
    route_seeds: tuple[int, ...] = OLMOE_ROUTE_SEEDS
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    measurement_iterations: int = DEFAULT_MEASUREMENT_ITERATIONS
    independent_samples: int = DEFAULT_INDEPENDENT_SAMPLES
    relative_l1_tolerance: float = DEFAULT_RELATIVE_L1_TOLERANCE
    max_absolute_tolerance: float = DEFAULT_MAX_ABSOLUTE_TOLERANCE
    minimum_stable_improvement: float = MINIMUM_STABLE_IMPROVEMENT

    def __post_init__(self) -> None:
        if self.batch_anchors != OLMOE_BATCH_ANCHORS:
            raise ValueError(f"batch anchors must be exactly {OLMOE_BATCH_ANCHORS!r}")
        if len(self.route_seeds) != 3 or len(set(self.route_seeds)) != 3:
            raise ValueError("exactly three distinct route seeds are required")
        if any(type(seed) is not int or seed < 0 for seed in self.route_seeds):
            raise ValueError("route seeds must be nonnegative integers")
        if self.warmup_iterations < 1:
            raise ValueError("warmup_iterations must be positive")
        if self.measurement_iterations < 1:
            raise ValueError("measurement_iterations must be positive")
        if self.independent_samples < 3:
            raise ValueError("independent_samples must be at least three")
        for name, value in (
            ("relative_l1_tolerance", self.relative_l1_tolerance),
            ("max_absolute_tolerance", self.max_absolute_tolerance),
            ("minimum_stable_improvement", self.minimum_stable_improvement),
        ):
            if not math.isfinite(value) or not (0 < value < 1):
                raise ValueError(f"{name} must be finite and in (0, 1)")


@dataclass(frozen=True)
class RouteCase:
    name: str
    global_routes: tuple[tuple[int, ...], ...]
    local_routes: tuple[tuple[int, ...], ...]
    route_weights: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.global_routes:
            raise ValueError("route case must be named and nonempty")
        if not (
            len(self.global_routes) == len(self.local_routes) == len(self.route_weights)
        ):
            raise ValueError("route case matrices must have equal row counts")
        for global_row, local_row, weight_row in zip(
            self.global_routes,
            self.local_routes,
            self.route_weights,
            strict=True,
        ):
            if not (
                len(global_row) == len(local_row) == len(weight_row) == OLMOE_TOP_K
            ):
                raise ValueError("route case rows must have OLMoE top-k width")
            if len(set(global_row)) != OLMOE_TOP_K:
                raise ValueError("global top-k routes must be distinct per token")
            if any(
                expert < 0 or expert >= OLMOE_GLOBAL_EXPERTS for expert in global_row
            ):
                raise ValueError("global route is outside the OLMoE expert range")
            if any(not math.isfinite(weight) or weight <= 0 for weight in weight_row):
                raise ValueError("route weights must be finite and positive")
            if sum(weight_row) >= 1:
                raise ValueError("OLMoE top-k weights must remain unnormalized")


@dataclass(frozen=True)
class NumericalEvidence:
    scenario: str
    rank: int
    relative_l1: float
    max_absolute: float
    repeat_exact: bool
    global_route_sha256: str
    local_route_sha256: str

    def __post_init__(self) -> None:
        if not self.scenario or self.rank < 0:
            raise ValueError("numerical evidence must identify scenario and rank")
        if not math.isfinite(self.relative_l1) or self.relative_l1 < 0:
            raise ValueError("relative L1 evidence must be finite and nonnegative")
        if not math.isfinite(self.max_absolute) or self.max_absolute < 0:
            raise ValueError("max-absolute evidence must be finite and nonnegative")
        if not self.repeat_exact:
            raise ValueError("published numerical evidence must be repeat-exact")
        if not _is_sha256(self.global_route_sha256) or not _is_sha256(
            self.local_route_sha256
        ):
            raise ValueError("numerical route evidence requires SHA-256 bindings")


@dataclass(frozen=True)
class BracketTiming:
    route_seed: int
    candidate_microseconds: tuple[float, ...]
    fallback_before_microseconds: tuple[float, ...]
    fallback_after_microseconds: tuple[float, ...]

    def __post_init__(self) -> None:
        sample_counts = {
            len(self.candidate_microseconds),
            len(self.fallback_before_microseconds),
            len(self.fallback_after_microseconds),
        }
        if len(sample_counts) != 1 or not self.candidate_microseconds:
            raise ValueError("bracket timing samples must be nonempty and aligned")
        if any(
            not math.isfinite(sample) or sample <= 0
            for sample in (
                *self.candidate_microseconds,
                *self.fallback_before_microseconds,
                *self.fallback_after_microseconds,
            )
        ):
            raise ValueError("bracket timing samples must be finite and positive")

    @property
    def candidate_median_microseconds(self) -> float:
        return statistics.median(self.candidate_microseconds)

    @property
    def fallback_before_median_microseconds(self) -> float:
        return statistics.median(self.fallback_before_microseconds)

    @property
    def fallback_after_median_microseconds(self) -> float:
        return statistics.median(self.fallback_after_microseconds)

    def is_stable_win(self, minimum_improvement: float) -> bool:
        candidate = self.candidate_median_microseconds
        return all(
            (fallback - candidate) / fallback >= minimum_improvement
            for fallback in (
                self.fallback_before_median_microseconds,
                self.fallback_after_median_microseconds,
            )
        )


@dataclass(frozen=True)
class RankStageMeasurement:
    rank: int
    numerical_evidence: tuple[NumericalEvidence, ...]
    route_timings: tuple[BracketTiming, ...]

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("measurement rank must be nonnegative")
        route_seeds = tuple(timing.route_seed for timing in self.route_timings)
        if len(route_seeds) != 3 or len(set(route_seeds)) != 3:
            raise ValueError("rank timing must cover three distinct route seeds")
        if not self.numerical_evidence:
            raise ValueError("rank stage must include numerical evidence")
        if any(evidence.rank != self.rank for evidence in self.numerical_evidence):
            raise ValueError("numerical evidence rank differs from stage rank")
        scenarios = tuple(evidence.scenario for evidence in self.numerical_evidence)
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("numerical scenarios must be unique per rank and stage")

    @property
    def median_microseconds(self) -> float:
        return statistics.median(
            timing.candidate_median_microseconds for timing in self.route_timings
        )


@dataclass(frozen=True)
class StageMeasurement:
    stage: KernelStage
    config: KernelConfig
    ranks: tuple[RankStageMeasurement, ...]
    minimum_stable_improvement: float

    def __post_init__(self) -> None:
        if not self.ranks or tuple(rank.rank for rank in self.ranks) != tuple(
            range(len(self.ranks))
        ):
            raise ValueError("stage ranks must be contiguous and ordered")
        route_seed_sets = {
            tuple(timing.route_seed for timing in rank.route_timings)
            for rank in self.ranks
        }
        if len(route_seed_sets) != 1:
            raise ValueError("stage ranks must use identical route seeds")
        if not math.isfinite(self.minimum_stable_improvement) or not (
            0 < self.minimum_stable_improvement < 1
        ):
            raise ValueError("stage stable-improvement threshold is invalid")

    @property
    def worst_rank_median_microseconds(self) -> float:
        return max(rank.median_microseconds for rank in self.ranks)

    @property
    def stable_win(self) -> bool:
        return self.config != _FALLBACK_CONFIG and all(
            timing.is_stable_win(self.minimum_stable_improvement)
            for rank in self.ranks
            for timing in rank.route_timings
        )


@dataclass(frozen=True)
class CandidateMeasurement:
    config: KernelConfig
    gate_up: StageMeasurement | None
    down: StageMeasurement | None
    rejection: str | None = None

    def __post_init__(self) -> None:
        successful = self.gate_up is not None and self.down is not None
        if successful == (self.rejection is not None):
            raise ValueError("candidate must be either successful or rejected")


@dataclass(frozen=True)
class SelectedPair:
    gate_up: StageMeasurement
    down: StageMeasurement
    admitted_stable_win: bool

    def __post_init__(self) -> None:
        if self.gate_up.stage != "gate_up" or self.down.stage != "down":
            raise ValueError("selected pair has invalid stages")
        if self.gate_up.config["BLOCK_SIZE_M"] != self.down.config["BLOCK_SIZE_M"]:
            raise ValueError("selected gate-up/down configs must share BLOCK_SIZE_M")
        both_stable = self.gate_up.stable_win and self.down.stable_win
        if self.admitted_stable_win != both_stable:
            raise ValueError("selected-pair admission differs from stage evidence")
        if not self.admitted_stable_win and (
            self.gate_up.config != _FALLBACK_CONFIG
            or self.down.config != _FALLBACK_CONFIG
        ):
            raise ValueError("a non-admitted pair must retain both fallbacks")


@dataclass(frozen=True)
class AnchorResult:
    ep_size: Literal[1, 2]
    batch_size: int
    candidates: tuple[CandidateMeasurement, ...]
    selected: SelectedPair

    def __post_init__(self) -> None:
        topology = topology_for_ep_size(self.ep_size)
        if self.batch_size not in OLMOE_BATCH_ANCHORS:
            raise ValueError("anchor result uses a noncanonical batch size")
        observed_configs = tuple(
            _kernel_config_key(candidate.config) for candidate in self.candidates
        )
        expected_configs = tuple(
            _kernel_config_key(config) for config in build_rtx3090_search_space()
        )
        if observed_configs != expected_configs:
            raise ValueError("anchor result must cover the exact six-config sweep")
        successful_stages = tuple(
            stage
            for candidate in self.candidates
            for stage in (candidate.gate_up, candidate.down)
            if stage is not None
        )
        if any(len(stage.ranks) != topology.rank_count for stage in successful_stages):
            raise ValueError("candidate stage does not cover every topology rank")
        for stage in successful_stages:
            for rank in stage.ranks:
                route_seeds = tuple(timing.route_seed for timing in rank.route_timings)
                expected_scenarios = tuple(
                    f"softmax_topk_seed_{route_seed}" for route_seed in route_seeds
                ) + (("all_remote", "boundary_31_32") if topology.ep_size == 2 else ())
                if (
                    tuple(evidence.scenario for evidence in rank.numerical_evidence)
                    != expected_scenarios
                ):
                    raise ValueError(
                        "candidate numerical scenarios do not cover the topology contract"
                    )
        if not any(
            candidate.gate_up == self.selected.gate_up for candidate in self.candidates
        ) or not any(
            candidate.down == self.selected.down for candidate in self.candidates
        ):
            raise ValueError("selected stages are absent from candidate evidence")


@dataclass(frozen=True)
class RuntimeBindings:
    torch: Any
    triton: Any
    triton_language: Any
    invoke_fused_moe_kernel: Callable[..., None]
    moe_align_block_size: Callable[[Any, int, int], tuple[Any, Any, Any]]
    sglang_revision: str
    ktransformers_revision: str
    source_files: Mapping[str, Path]
    source_modules: Mapping[str, str]


@dataclass
class RuntimeInputs:
    hidden_states: Any
    gate_up_weights: Any
    down_weights: Any


@dataclass
class RuntimeWorkload:
    scenario: str
    route_seed: int | None
    inputs: RuntimeInputs
    route_weights: Any
    global_route_ids: Any
    local_route_ids: Any
    gate_up_reference: Any
    down_input: Any
    down_reference: Any
    global_route_sha256: str
    local_route_sha256: str


@dataclass(frozen=True)
class PreparedOperation:
    operation: Callable[[], None]
    output: Any
    expected: Any
    route_count: int


@dataclass
class _CandidateAccumulator:
    config: KernelConfig
    gate_up_ranks: list[RankStageMeasurement] = field(default_factory=list)
    down_ranks: list[RankStageMeasurement] = field(default_factory=list)
    rejection: str | None = None


class CandidateNumericalError(RuntimeError):
    """A candidate-specific numerical rejection with a healthy CUDA context."""


def _kernel_config_from_tuple(
    values: tuple[int, int, int, int, int, int],
) -> KernelConfig:
    block_m, block_n, block_k, group_m, warps, stages = values
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": warps,
        "num_stages": stages,
    }


def _copy_kernel_config(config: KernelConfig) -> KernelConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def _kernel_config_key(config: KernelConfig) -> tuple[int, int, int, int, int, int]:
    return (
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["num_warps"],
        config["num_stages"],
    )


def build_rtx3090_search_space() -> tuple[KernelConfig, ...]:
    """Return the exact bounded six-config search contract."""

    return tuple(
        _kernel_config_from_tuple(values) for values in _RTX3090_CANDIDATE_TUPLES
    )


def topology_for_ep_size(ep_size: int) -> OlmoeTopology:
    for topology in OLMOE_TOPOLOGIES:
        if topology.ep_size == ep_size:
            return topology
    raise ValueError(f"unsupported OLMoE EP size: {ep_size}")


def map_global_route_rows(
    global_routes: Sequence[Sequence[int]],
    *,
    topology: OlmoeTopology,
    rank: int,
) -> tuple[tuple[int, ...], ...]:
    """Map global OLMoE IDs to one rank's local experts, masking remote IDs."""

    if not 0 <= rank < topology.rank_count:
        raise ValueError(f"rank {rank} is outside EP{topology.ep_size}")
    lower = rank * topology.local_experts
    upper = lower + topology.local_experts
    rows: list[tuple[int, ...]] = []
    for row in global_routes:
        if len(row) != OLMOE_TOP_K:
            raise ValueError("global route rows must have width eight")
        if len(set(row)) != OLMOE_TOP_K:
            raise ValueError("global route rows must contain distinct experts")
        if any(expert < 0 or expert >= OLMOE_GLOBAL_EXPERTS for expert in row):
            raise ValueError("global route ID is outside [0, 64)")
        rows.append(
            tuple(expert - lower if lower <= expert < upper else -1 for expert in row)
        )
    if not rows:
        raise ValueError("global routes must be nonempty")
    return tuple(rows)


def build_ep2_correctness_cases(
    *, topology: OlmoeTopology, rank: int, batch_size: int
) -> tuple[RouteCase, ...]:
    """Build forced remote-only and 31/32-boundary EP2 validation routes."""

    if topology.ep_size != 2:
        return ()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    remote_start = 32 if rank == 0 else 0
    remote_base = tuple(range(remote_start, remote_start + OLMOE_TOP_K))
    boundary_base = (31, 32, 30, 33, 0, 63, 15, 48)
    weights = (0.14, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03)

    cases: list[RouteCase] = []
    for name, base in (("all_remote", remote_base), ("boundary_31_32", boundary_base)):
        rows = tuple(
            tuple(base[(column + row) % OLMOE_TOP_K] for column in range(OLMOE_TOP_K))
            for row in range(batch_size)
        )
        weight_rows = tuple(weights for _ in range(batch_size))
        cases.append(
            RouteCase(
                name=name,
                global_routes=rows,
                local_routes=map_global_route_rows(rows, topology=topology, rank=rank),
                route_weights=weight_rows,
            )
        )
    return tuple(cases)


def config_file_name(topology: OlmoeTopology, *, down: bool) -> str:
    suffix = "_down" if down else ""
    return (
        f"E={topology.local_experts},N={topology.intermediate_size},"
        f"device_name=NVIDIA_GeForce_RTX_3090{suffix}.json"
    )


def select_admitted_pair(
    candidates: Sequence[CandidateMeasurement],
) -> SelectedPair:
    successful = [
        candidate
        for candidate in candidates
        if candidate.gate_up is not None and candidate.down is not None
    ]
    fallback = next(
        (candidate for candidate in successful if candidate.config == _FALLBACK_CONFIG),
        None,
    )
    if fallback is None or fallback.gate_up is None or fallback.down is None:
        raise ValueError("a successful fallback measurement is required")

    gate_options = [
        candidate.gate_up
        for candidate in successful
        if candidate.gate_up is not None and candidate.gate_up.stable_win
    ]
    down_options = [
        candidate.down
        for candidate in successful
        if candidate.down is not None and candidate.down.stable_win
    ]
    pairs = [
        (gate, down)
        for gate in gate_options
        for down in down_options
        if gate.config["BLOCK_SIZE_M"] == down.config["BLOCK_SIZE_M"]
    ]
    if not pairs:
        return SelectedPair(
            gate_up=fallback.gate_up,
            down=fallback.down,
            admitted_stable_win=False,
        )
    gate, down = min(
        pairs,
        key=lambda pair: (
            pair[0].worst_rank_median_microseconds
            + pair[1].worst_rank_median_microseconds,
            _kernel_config_key(pair[0].config),
            _kernel_config_key(pair[1].config),
        ),
    )
    return SelectedPair(gate_up=gate, down=down, admitted_stable_win=True)


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _config_map_json_bytes(configs: Mapping[int, KernelConfig]) -> bytes:
    """Serialize nearest-anchor keys in numeric order, including tie order."""

    if tuple(configs) != tuple(sorted(configs)):
        raise ValueError("config-map anchors must be inserted in numeric order")
    ordered = {str(anchor): dict(configs[anchor]) for anchor in configs}
    return (json.dumps(ordered, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


def _open_absolute_nofollow(path: Path, flags: int) -> int:
    components = path.parts
    if not path.is_absolute() or not components or components[0] != "/":
        raise ValueError("bound path must be absolute")
    if any(component in ("", ".", "..") for component in components[1:]):
        raise ValueError("bound path contains an unsafe component")
    directory_descriptor = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        for component in components[1:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(components[-1], flags, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _validated_file_binding(
    *, path: Path, expected_sha256: str, description: str, maximum_bytes: int
) -> tuple[dict[str, object], bytes]:
    if not path.is_absolute() or "\0" in str(path):
        raise RuntimeError(f"{description} path must be absolute")
    if not _is_sha256(expected_sha256):
        raise RuntimeError(f"{description} expected SHA-256 is invalid")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = _open_absolute_nofollow(path, flags)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot open bound {description}: {path}") from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(f"{description} must be a regular file")
        if status.st_nlink != 1:
            raise RuntimeError(f"{description} must have exactly one hard link")
        if status.st_size <= 0 or status.st_size > maximum_bytes:
            raise RuntimeError(f"{description} size is outside the accepted bound")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        contents = b"".join(chunks)
        post_status = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(contents) != status.st_size or len(contents) > maximum_bytes:
        raise RuntimeError(f"{description} changed size while being authenticated")
    observed_sha256 = _sha256_bytes(contents)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{description} SHA-256 differs: {observed_sha256} != {expected_sha256}"
        )
    if (
        post_status.st_dev,
        post_status.st_ino,
        post_status.st_size,
        post_status.st_mtime_ns,
        post_status.st_ctime_ns,
    ) != (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    ):
        raise RuntimeError(f"{description} changed while it was being authenticated")
    try:
        path_status = path.lstat()
    except OSError as error:
        raise RuntimeError(
            f"{description} path changed after authentication"
        ) from error
    if (
        path_status.st_dev,
        path_status.st_ino,
        path_status.st_mode,
    ) != (status.st_dev, status.st_ino, status.st_mode):
        raise RuntimeError(f"{description} path changed during authentication")
    return (
        {
            "path": str(path),
            "sha256": observed_sha256,
            "size_bytes": len(contents),
            "device": status.st_dev,
            "inode": status.st_ino,
        },
        contents,
    )


def validate_live_input_bindings(
    *,
    runtime_install_receipt: Path,
    runtime_install_receipt_sha256: str,
    model_config: Path,
    model_config_sha256: str,
) -> Mapping[str, object]:
    """Authenticate the runtime install receipt and exact OLMoE config pre-CUDA."""

    runtime_binding, runtime_contents = _validated_file_binding(
        path=runtime_install_receipt,
        expected_sha256=runtime_install_receipt_sha256,
        description="runtime install receipt",
        maximum_bytes=16 * 1024 * 1024,
    )
    try:
        runtime_receipt = json.loads(runtime_contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("runtime install receipt is not valid JSON") from error
    if not isinstance(runtime_receipt, dict) or not runtime_receipt:
        raise RuntimeError("runtime install receipt must be a nonempty JSON object")
    if (
        runtime_receipt.get("schema_version") != 1
        or runtime_receipt.get("status") != "install_complete"
    ):
        raise RuntimeError("runtime receipt is not a completed schema-1 install")
    install_id = runtime_receipt.get("install_id")
    layout = runtime_receipt.get("layout")
    build = runtime_receipt.get("build")
    if (
        not _is_sha256(install_id)
        or not isinstance(layout, dict)
        or not isinstance(build, dict)
    ):
        raise RuntimeError("runtime receipt identity or layout is invalid")
    install_root_value = layout.get("install_root")
    runtime_python_value = layout.get("python")
    receipt_value = layout.get("receipt")
    site_packages_value = layout.get("site_packages")
    if not all(
        isinstance(value, str) and Path(value).is_absolute()
        for value in (
            install_root_value,
            runtime_python_value,
            receipt_value,
            site_packages_value,
        )
    ):
        raise RuntimeError("runtime receipt paths must be absolute")
    install_root = Path(cast(str, install_root_value))
    runtime_python = Path(cast(str, runtime_python_value))
    site_packages = Path(cast(str, site_packages_value))
    if (
        install_root.name != install_id
        or Path(cast(str, receipt_value)) != runtime_install_receipt
        or runtime_install_receipt.parent != install_root
        or runtime_python != Path(sys.executable).absolute()
        or not runtime_python.is_relative_to(install_root)
        or not site_packages.is_relative_to(install_root)
    ):
        raise RuntimeError("runtime install receipt does not bind this interpreter")
    sglang_revision = build.get("sglang_revision")
    ktransformers_revision = build.get("ktransformers_revision")
    if any(
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in (sglang_revision, ktransformers_revision)
    ):
        raise RuntimeError("runtime receipt build revisions are invalid")
    runtime_binding.update(
        {
            "schema_version": 1,
            "status": "install_complete",
            "install_id": install_id,
            "install_root": str(install_root),
            "runtime_python": str(runtime_python),
            "site_packages": str(site_packages),
            "sglang_revision": sglang_revision,
            "ktransformers_revision": ktransformers_revision,
        }
    )

    model_binding, model_contents = _validated_file_binding(
        path=model_config,
        expected_sha256=model_config_sha256,
        description="OLMoE model config",
        maximum_bytes=1024 * 1024,
    )
    try:
        model_value = json.loads(model_contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("OLMoE model config is not valid JSON") from error
    if not isinstance(model_value, dict):
        raise RuntimeError("OLMoE model config must be a JSON object")
    expected_fields: dict[str, object] = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_size": OLMOE_HIDDEN_SIZE,
        "intermediate_size": 1_024,
        "num_experts": OLMOE_GLOBAL_EXPERTS,
        "num_experts_per_tok": OLMOE_TOP_K,
        "norm_topk_prob": False,
        "torch_dtype": "bfloat16",
    }
    mismatches = {
        name: {"expected": expected, "observed": model_value.get(name)}
        for name, expected in expected_fields.items()
        if model_value.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"OLMoE model config fields differ: {mismatches!r}")
    model_binding["validated_fields"] = expected_fields
    return {
        "runtime_install_receipt": runtime_binding,
        "model_config": model_binding,
    }


def _rows_sha256(rows: Sequence[Sequence[int]]) -> str:
    return _sha256_bytes(_canonical_json_bytes([list(row) for row in rows]))


def _stage_json(measurement: StageMeasurement) -> dict[str, object]:
    return {
        "stage": measurement.stage,
        "config": dict(measurement.config),
        "stable_win": measurement.stable_win,
        "minimum_stable_improvement": measurement.minimum_stable_improvement,
        "worst_rank_median_microseconds": (measurement.worst_rank_median_microseconds),
        "ranks": [
            {
                "rank": rank.rank,
                "median_microseconds": rank.median_microseconds,
                "numerical_evidence": [
                    {
                        "scenario": evidence.scenario,
                        "rank": evidence.rank,
                        "relative_l1": evidence.relative_l1,
                        "max_absolute": evidence.max_absolute,
                        "repeat_exact": evidence.repeat_exact,
                        "global_route_sha256": evidence.global_route_sha256,
                        "local_route_sha256": evidence.local_route_sha256,
                    }
                    for evidence in rank.numerical_evidence
                ],
                "route_timings": [
                    {
                        "route_seed": timing.route_seed,
                        "candidate_microseconds": list(timing.candidate_microseconds),
                        "fallback_before_microseconds": list(
                            timing.fallback_before_microseconds
                        ),
                        "fallback_after_microseconds": list(
                            timing.fallback_after_microseconds
                        ),
                        "candidate_median_microseconds": (
                            timing.candidate_median_microseconds
                        ),
                        "fallback_before_median_microseconds": (
                            timing.fallback_before_median_microseconds
                        ),
                        "fallback_after_median_microseconds": (
                            timing.fallback_after_median_microseconds
                        ),
                        "stable_win": timing.is_stable_win(
                            measurement.minimum_stable_improvement
                        ),
                    }
                    for timing in rank.route_timings
                ],
            }
            for rank in measurement.ranks
        ],
    }


def _anchor_json(result: AnchorResult) -> dict[str, object]:
    return {
        "ep_size": result.ep_size,
        "batch_size": result.batch_size,
        "selected": {
            "admitted_stable_win": result.selected.admitted_stable_win,
            "shared_block_size_m": result.selected.gate_up.config["BLOCK_SIZE_M"],
            "gate_up": _stage_json(result.selected.gate_up),
            "down": _stage_json(result.selected.down),
        },
        "candidates": [
            {
                "config": dict(candidate.config),
                "rejection": candidate.rejection,
                "gate_up": (
                    _stage_json(candidate.gate_up)
                    if candidate.gate_up is not None
                    else None
                ),
                "down": (
                    _stage_json(candidate.down) if candidate.down is not None else None
                ),
            }
            for candidate in result.candidates
        ],
    }


def _write_exclusive_file(path: Path, contents: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o444,
    )
    try:
        offset = 0
        while offset < len(contents):
            offset += os.write(descriptor, contents[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_tuning_bundle(
    *,
    output_directory: Path,
    spec: TuningSpec,
    results: Sequence[AnchorResult],
    runtime_provenance: Mapping[str, object],
    input_bindings: Mapping[str, object],
    tuner_path: Path,
    expected_tuner_sha256: str,
) -> Mapping[str, object]:
    """Publish an exclusive, read-only candidate bundle with strong hashes."""

    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    expected_keys = {
        (topology.ep_size, batch_size)
        for topology in OLMOE_TOPOLOGIES
        for batch_size in spec.batch_anchors
    }
    result_by_key = {(result.ep_size, result.batch_size): result for result in results}
    if set(result_by_key) != expected_keys or len(result_by_key) != len(results):
        raise ValueError("results must exactly cover both topologies and all anchors")
    for result in results:
        for candidate in result.candidates:
            for stage in (candidate.gate_up, candidate.down):
                if stage is None:
                    continue
                if any(
                    tuple(timing.route_seed for timing in rank.route_timings)
                    != spec.route_seeds
                    for rank in stage.ranks
                ):
                    raise ValueError("result route seeds differ from the tuning spec")
                if stage.minimum_stable_improvement != spec.minimum_stable_improvement:
                    raise ValueError(
                        "result stable threshold differs from the tuning spec"
                    )
                for rank in stage.ranks:
                    for evidence in rank.numerical_evidence:
                        if (
                            evidence.relative_l1 > spec.relative_l1_tolerance
                            or evidence.max_absolute > spec.max_absolute_tolerance
                        ):
                            raise ValueError(
                                "result numerical evidence exceeds the tuning spec"
                            )
                    for timing in rank.route_timings:
                        if not (
                            len(timing.candidate_microseconds)
                            == len(timing.fallback_before_microseconds)
                            == len(timing.fallback_after_microseconds)
                            == spec.independent_samples
                        ):
                            raise ValueError(
                                "result timing sample count differs from the tuning spec"
                            )
    if set(input_bindings) != {"runtime_install_receipt", "model_config"}:
        raise ValueError("input bindings must cover runtime receipt and model config")
    for name, value in input_bindings.items():
        if not isinstance(value, Mapping) or not _is_sha256(value.get("sha256")):
            raise ValueError(f"input binding {name} lacks a valid SHA-256")

    tuner_path = tuner_path.resolve(strict=True)
    tuner_sha256 = _sha256_file(tuner_path)
    if not _is_sha256(expected_tuner_sha256) or tuner_sha256 != expected_tuner_sha256:
        raise RuntimeError("tuner source changed after live-run admission")
    config_contents: dict[str, bytes] = {}
    config_descriptions: dict[str, object] = {}
    for topology in OLMOE_TOPOLOGIES:
        topology_results = [
            result_by_key[(topology.ep_size, batch)] for batch in spec.batch_anchors
        ]
        for stage, down in (("gate_up", False), ("down", True)):
            name = config_file_name(topology, down=down)
            relative_path = f"configs/{_CONFIG_VERSION_DIRECTORY}/{name}"
            configs = {
                result.batch_size: _copy_kernel_config(
                    result.selected.down.config
                    if down
                    else result.selected.gate_up.config
                )
                for result in topology_results
            }
            contents = _config_map_json_bytes(configs)
            config_contents[relative_path] = contents
            config_descriptions[f"ep{topology.ep_size}_{stage}"] = {
                "relative_path": relative_path,
                "sha256": _sha256_bytes(contents),
                "shape": {
                    "E": topology.local_experts,
                    "N": topology.intermediate_size,
                },
                "batch_anchors": list(spec.batch_anchors),
            }

    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "olmoe_rtx3090_triton_moe_tuning_receipt_v1",
        "candidate": True,
        "deployment_admitted": False,
        "production_path_reproduced": False,
        "required_next_gate": "controlled OLMoE serving A/B",
        "model": {
            "architecture": "OlmoeForCausalLM",
            "hidden_size": OLMOE_HIDDEN_SIZE,
            "global_experts": OLMOE_GLOBAL_EXPERTS,
            "top_k": OLMOE_TOP_K,
            "normalize_top_k_weights": False,
            "dtype": "bfloat16",
        },
        "tuning_contract": {
            "batch_anchors": list(spec.batch_anchors),
            "route_seeds": list(spec.route_seeds),
            "candidate_configs": [
                dict(config) for config in build_rtx3090_search_space()
            ],
            "warmup_iterations": spec.warmup_iterations,
            "measurement_iterations": spec.measurement_iterations,
            "independent_samples": spec.independent_samples,
            "relative_l1_tolerance": spec.relative_l1_tolerance,
            "max_absolute_tolerance": spec.max_absolute_tolerance,
            "minimum_stable_improvement": spec.minimum_stable_improvement,
            "timing_order": "fallback-candidate-fallback",
            "selection_score": "sum of gate/down worst-rank medians",
            "alignment_in_timed_region": False,
        },
        "topologies": [
            {
                "ep_size": topology.ep_size,
                "rank_count": topology.rank_count,
                "local_experts": topology.local_experts,
                "intermediate_size": topology.intermediate_size,
                "gate_up_weight_shape": list(topology.gate_up_weight_shape),
                "down_weight_shape": list(topology.down_weight_shape),
                "route_mapping": (
                    "global IDs unchanged"
                    if topology.ep_size == 1
                    else "rank r owns [32*r,32*(r+1)); remote IDs map to -1"
                ),
            }
            for topology in OLMOE_TOPOLOGIES
        ],
        "runtime_provenance": dict(runtime_provenance),
        "input_bindings": dict(input_bindings),
        "tuner": {"path": str(tuner_path), "sha256": tuner_sha256},
        "config_files": config_descriptions,
        "limitations": [
            "moe_align_block_size and allocations are outside timed regions",
            "routing, activation, collectives, and final reduction are not timed",
            "EP2 ranks are measured serially on one visible GPU",
            "generated configs remain candidates until a serving A/B passes",
        ],
        "anchors": [_anchor_json(result_by_key[key]) for key in sorted(result_by_key)],
    }
    receipt_contents = _canonical_json_bytes(receipt, pretty=True)
    all_contents = {**config_contents, _RECEIPT_NAME: receipt_contents}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "olmoe_rtx3090_triton_moe_candidate_bundle_v1",
        "candidate": True,
        "deployment_admitted": False,
        "files": {
            relative_path: {
                "sha256": _sha256_bytes(contents),
                "size_bytes": len(contents),
            }
            for relative_path, contents in sorted(all_contents.items())
        },
        "tuner_sha256": tuner_sha256,
    }
    manifest_contents = _canonical_json_bytes(manifest, pretty=True)

    output_directory = output_directory.absolute()
    if output_directory.name in ("", ".", ".."):
        raise ValueError("output directory must have a final path component")
    output_directory.parent.resolve(strict=True)
    os.mkdir(output_directory, mode=0o700)
    configs_directory = output_directory / "configs"
    version_directory = configs_directory / _CONFIG_VERSION_DIRECTORY
    os.mkdir(configs_directory, mode=0o700)
    os.mkdir(version_directory, mode=0o700)
    for relative_path, contents in sorted(all_contents.items()):
        destination = output_directory / relative_path
        _write_exclusive_file(destination, contents)
    _write_exclusive_file(output_directory / _MANIFEST_NAME, manifest_contents)
    os.chmod(version_directory, 0o555)
    os.chmod(configs_directory, 0o555)
    os.chmod(output_directory, 0o555)
    return manifest


def _validate_fused_moe_kernel_signature(kernel: Callable[..., None]) -> None:
    signature = inspect.signature(kernel)
    if tuple(signature.parameters) != _FUSED_MOE_PARAMETER_NAMES:
        raise RuntimeError("pinned fused-MoE kernel interface changed")
    for name, parameter in signature.parameters.items():
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise RuntimeError(f"fused-MoE parameter {name} changed calling convention")
        expected = _FUSED_MOE_OPTIONAL_DEFAULTS.get(name, inspect.Parameter.empty)
        if parameter.default != expected:
            raise RuntimeError(f"fused-MoE parameter {name} changed its default")


def load_runtime_bindings() -> RuntimeBindings:
    """Load the pinned CUDA stack only after the live-run preconditions."""

    import torch
    import triton
    import triton.language as triton_language
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        invoke_fused_moe_kernel,
        moe_align_block_size,
    )

    _validate_fused_moe_kernel_signature(invoke_fused_moe_kernel)
    provenance = importlib.import_module("sglang._exo_build_provenance")
    config_loader = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config"
    )
    standard_dispatcher = importlib.import_module(
        "sglang.srt.layers.moe.token_dispatcher.standard"
    )
    olmoe_model = importlib.import_module("sglang.srt.models.olmoe")
    if getattr(provenance, "SCHEMA_VERSION", None) != 1:
        raise RuntimeError("pinned SGLang provenance schema must be version 1")
    sglang_revision = getattr(provenance, "SGLANG_REVISION", None)
    ktransformers_revision = getattr(provenance, "KTRANSFORMERS_REVISION", None)
    for name, revision in (
        ("SGLANG_REVISION", sglang_revision),
        ("KTRANSFORMERS_REVISION", ktransformers_revision),
    ):
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise RuntimeError(f"invalid pinned runtime {name}: {revision!r}")
    source_files: dict[str, Path] = {}
    source_modules: dict[str, str] = {}
    for name, value in (
        ("fused_kernel_implementation", invoke_fused_moe_kernel),
        ("align_helper", moe_align_block_size),
        ("config_loader", config_loader),
        ("standard_dispatcher", standard_dispatcher),
        ("olmoe_model", olmoe_model),
        ("embedded_build_provenance", provenance),
    ):
        source = inspect.getsourcefile(value)
        if source is None:
            raise RuntimeError(f"cannot locate runtime source for {name}")
        source_files[name] = Path(source).resolve(strict=True)
        source_modules[name] = getattr(value, "__module__", None) or getattr(
            value, "__name__", ""
        )
        if not source_modules[name]:
            raise RuntimeError(f"cannot identify runtime module for {name}")
    return RuntimeBindings(
        torch=torch,
        triton=triton,
        triton_language=triton_language,
        invoke_fused_moe_kernel=invoke_fused_moe_kernel,
        moe_align_block_size=moe_align_block_size,
        sglang_revision=sglang_revision,
        ktransformers_revision=ktransformers_revision,
        source_files=source_files,
        source_modules=source_modules,
    )


def _runtime_provenance(runtime: RuntimeBindings) -> Mapping[str, object]:
    torch = runtime.torch
    properties = torch.cuda.get_device_properties(0)
    executable_path = Path(sys.executable).absolute()
    executable_status = executable_path.lstat()
    executable_resolved = executable_path.resolve(strict=True)
    executable_symlink_target = (
        os.readlink(executable_path)
        if stat.S_ISLNK(executable_status.st_mode)
        else None
    )
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "triton_version": str(runtime.triton.__version__),
        "sglang_revision": runtime.sglang_revision,
        "ktransformers_revision": runtime.ktransformers_revision,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_uuid": str(getattr(properties, "uuid", "")),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": {
            "invoked_path": str(executable_path),
            "invoked_path_device": executable_status.st_dev,
            "invoked_path_inode": executable_status.st_ino,
            "symlink_target": executable_symlink_target,
            "resolved_path": str(executable_resolved),
            "resolved_sha256": _sha256_file(executable_resolved),
        },
        "runtime_source_sha256": {
            name: {
                "module": runtime.source_modules[name],
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for name, path in sorted(runtime.source_files.items())
        },
    }


def _validate_live_runtime(
    runtime: RuntimeBindings, input_bindings: Mapping[str, object]
) -> None:
    torch = runtime.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if str(runtime.triton.__version__) != PINNED_TRITON_VERSION:
        raise RuntimeError(
            f"Triton must be {PINNED_TRITON_VERSION}, got {runtime.triton.__version__}"
        )
    if str(torch.cuda.get_device_name(0)) != RTX3090_DEVICE_NAME:
        raise RuntimeError("tuner requires an NVIDIA GeForce RTX 3090")
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6):
        raise RuntimeError("tuner requires SM86 compute capability")
    runtime_receipt = input_bindings.get("runtime_install_receipt")
    if not isinstance(runtime_receipt, Mapping) or (
        runtime_receipt.get("sglang_revision") != runtime.sglang_revision
        or runtime_receipt.get("ktransformers_revision")
        != runtime.ktransformers_revision
    ):
        raise RuntimeError("loaded runtime revisions differ from the install receipt")
    site_packages_value = runtime_receipt.get("site_packages")
    if not isinstance(site_packages_value, str):
        raise RuntimeError("runtime receipt lacks its site-packages binding")
    site_packages = Path(site_packages_value).resolve(strict=True)
    if any(
        not source_path.is_relative_to(site_packages)
        for source_path in runtime.source_files.values()
    ):
        raise RuntimeError("loaded SGLang source is outside the bound runtime overlay")


def _create_runtime_inputs(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    batch_size: int,
    rank: int,
) -> RuntimeInputs:
    torch = runtime.torch
    generator = torch.Generator(device="cuda")
    generator.manual_seed(81_000_000 + topology.ep_size * 100_000 + batch_size + rank)
    hidden_states = torch.randn(
        (batch_size, OLMOE_HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate_up_weights = torch.randn(
        topology.gate_up_weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    down_weights = torch.randn(
        topology.down_weight_shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    return RuntimeInputs(
        hidden_states=hidden_states,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
    )


def _map_global_route_tensor(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    rank: int,
    global_route_ids: Any,
) -> Any:
    if topology.ep_size == 1:
        return global_route_ids.to(dtype=runtime.torch.int32)
    lower = rank * topology.local_experts
    upper = lower + topology.local_experts
    return runtime.torch.where(
        (global_route_ids >= lower) & (global_route_ids < upper),
        global_route_ids - lower,
        -1,
    ).to(dtype=runtime.torch.int32)


def _tensor_rows_sha256(tensor: Any) -> str:
    rows = cast(list[list[int]], tensor.detach().cpu().tolist())
    return _rows_sha256(rows)


def _attach_torch_reference(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    scenario: str,
    route_seed: int | None,
    inputs: RuntimeInputs,
    route_weights: Any,
    global_route_ids: Any,
    local_route_ids: Any,
) -> RuntimeWorkload:
    torch = runtime.torch
    batch_size = int(inputs.hidden_states.shape[0])
    route_count = batch_size * OLMOE_TOP_K
    token_ids = torch.arange(batch_size, device="cuda").repeat_interleave(OLMOE_TOP_K)
    flattened_routes = local_route_ids.reshape(-1)
    gate_up_reference = torch.zeros(
        (route_count, 2 * topology.intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert_index in range(topology.local_experts):
        positions = torch.nonzero(
            flattened_routes == expert_index, as_tuple=False
        ).reshape(-1)
        if positions.numel() == 0:
            continue
        expert_inputs = inputs.hidden_states[token_ids[positions]]
        gate_up_reference[positions] = torch.nn.functional.linear(
            expert_inputs, inputs.gate_up_weights[expert_index]
        )
    down_input = (
        torch.nn.functional.silu(gate_up_reference[:, : topology.intermediate_size])
        * gate_up_reference[:, topology.intermediate_size :]
    )
    down_reference = torch.zeros(
        (route_count, OLMOE_HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    flattened_weights = route_weights.reshape(-1)
    for expert_index in range(topology.local_experts):
        positions = torch.nonzero(
            flattened_routes == expert_index, as_tuple=False
        ).reshape(-1)
        if positions.numel() == 0:
            continue
        expert_outputs = torch.nn.functional.linear(
            down_input[positions], inputs.down_weights[expert_index]
        )
        down_reference[positions] = expert_outputs * flattened_weights[
            positions
        ].unsqueeze(-1)
    return RuntimeWorkload(
        scenario=scenario,
        route_seed=route_seed,
        inputs=inputs,
        route_weights=route_weights,
        global_route_ids=global_route_ids,
        local_route_ids=local_route_ids,
        gate_up_reference=gate_up_reference,
        down_input=down_input,
        down_reference=down_reference,
        global_route_sha256=_tensor_rows_sha256(global_route_ids),
        local_route_sha256=_tensor_rows_sha256(local_route_ids),
    )


def _create_seeded_workload(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    rank: int,
    inputs: RuntimeInputs,
    route_seed: int,
) -> RuntimeWorkload:
    torch = runtime.torch
    generator = torch.Generator(device="cuda")
    generator.manual_seed(route_seed)
    logits = torch.randn(
        (inputs.hidden_states.shape[0], OLMOE_GLOBAL_EXPERTS),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    probabilities = torch.softmax(logits, dim=-1)
    route_weights, global_route_ids = torch.topk(probabilities, OLMOE_TOP_K, dim=-1)
    route_weights = route_weights.to(dtype=torch.float32)
    global_route_ids = global_route_ids.to(dtype=torch.int32)
    local_route_ids = _map_global_route_tensor(
        runtime,
        topology=topology,
        rank=rank,
        global_route_ids=global_route_ids,
    )
    return _attach_torch_reference(
        runtime,
        topology=topology,
        scenario=f"softmax_topk_seed_{route_seed}",
        route_seed=route_seed,
        inputs=inputs,
        route_weights=route_weights,
        global_route_ids=global_route_ids,
        local_route_ids=local_route_ids,
    )


def _create_explicit_workload(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    inputs: RuntimeInputs,
    route_case: RouteCase,
) -> RuntimeWorkload:
    torch = runtime.torch
    global_route_ids = torch.tensor(
        route_case.global_routes, device="cuda", dtype=torch.int32
    )
    local_route_ids = torch.tensor(
        route_case.local_routes, device="cuda", dtype=torch.int32
    )
    route_weights = torch.tensor(
        route_case.route_weights, device="cuda", dtype=torch.float32
    )
    return _attach_torch_reference(
        runtime,
        topology=topology,
        scenario=route_case.name,
        route_seed=None,
        inputs=inputs,
        route_weights=route_weights,
        global_route_ids=global_route_ids,
        local_route_ids=local_route_ids,
    )


def _invoke_kernel(
    runtime: RuntimeBindings,
    *,
    workload: RuntimeWorkload,
    config: KernelConfig,
    aligned_routes: tuple[Any, Any, Any],
    activation: Any,
    weights: Any,
    output: Any,
    mul_routed_weight: bool,
    top_k: int,
) -> None:
    sorted_token_ids, expert_ids, num_tokens_post_padded = aligned_routes
    runtime.invoke_fused_moe_kernel(
        A=activation,
        B=weights,
        bias=None,
        C=output,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=workload.route_weights,
        topk_ids=workload.local_route_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
        top_k=top_k,
        config=config,
        compute_type=runtime.triton_language.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        no_combine=False,
        a_use_tma=False,
        b_use_tma=False,
        c_sorted=False,
        filter_expert=True,
    )


def _prepare_operation(
    runtime: RuntimeBindings,
    *,
    topology: OlmoeTopology,
    workload: RuntimeWorkload,
    config: KernelConfig,
    stage: KernelStage,
) -> PreparedOperation:
    torch = runtime.torch
    aligned_routes = runtime.moe_align_block_size(
        workload.local_route_ids, config["BLOCK_SIZE_M"], topology.local_experts
    )
    capacity = int(aligned_routes[0].shape[0])
    route_count = int(workload.local_route_ids.numel())
    if stage == "gate_up":
        output = torch.zeros(
            (capacity, 2 * topology.intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def operation() -> None:
            _invoke_kernel(
                runtime,
                workload=workload,
                config=config,
                aligned_routes=aligned_routes,
                activation=workload.inputs.hidden_states,
                weights=workload.inputs.gate_up_weights,
                output=output,
                mul_routed_weight=False,
                top_k=OLMOE_TOP_K,
            )

        expected = workload.gate_up_reference
    else:
        down_input = torch.zeros(
            (capacity, topology.intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        down_input[:route_count].copy_(workload.down_input)
        output = torch.zeros(
            (workload.inputs.hidden_states.shape[0], OLMOE_TOP_K, OLMOE_HIDDEN_SIZE),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def operation() -> None:
            _invoke_kernel(
                runtime,
                workload=workload,
                config=config,
                aligned_routes=aligned_routes,
                activation=down_input,
                weights=workload.inputs.down_weights,
                output=output,
                mul_routed_weight=True,
                top_k=1,
            )

        expected = workload.down_reference
    return PreparedOperation(
        operation=operation, output=output, expected=expected, route_count=route_count
    )


def _validate_prepared_operation(
    runtime: RuntimeBindings,
    *,
    spec: TuningSpec,
    prepared: PreparedOperation,
    workload: RuntimeWorkload,
    rank: int,
) -> NumericalEvidence:
    prepared.operation()
    runtime.torch.cuda.synchronize()
    first = prepared.output.reshape(-1, prepared.output.shape[-1])[
        : prepared.route_count
    ].clone()
    prepared.operation()
    runtime.torch.cuda.synchronize()
    repeated = prepared.output.reshape(-1, prepared.output.shape[-1])[
        : prepared.route_count
    ].clone()
    torch = runtime.torch
    if not bool(torch.isfinite(first).all().item()):
        raise CandidateNumericalError("candidate produced non-finite output")
    repeat_exact = bool(torch.equal(first, repeated))
    if not repeat_exact:
        raise CandidateNumericalError("candidate output was not repeatable")
    difference = (first.float() - prepared.expected.float()).abs()
    denominator = prepared.expected.float().abs().sum().clamp_min(1e-12)
    relative_l1 = float((difference.sum() / denominator).item())
    max_absolute = float(difference.max().item())
    if relative_l1 > spec.relative_l1_tolerance:
        raise CandidateNumericalError(
            f"relative L1 {relative_l1} exceeded {spec.relative_l1_tolerance}"
        )
    if max_absolute > spec.max_absolute_tolerance:
        raise CandidateNumericalError(
            f"max absolute {max_absolute} exceeded {spec.max_absolute_tolerance}"
        )
    return NumericalEvidence(
        scenario=workload.scenario,
        rank=rank,
        relative_l1=relative_l1,
        max_absolute=max_absolute,
        repeat_exact=repeat_exact,
        global_route_sha256=workload.global_route_sha256,
        local_route_sha256=workload.local_route_sha256,
    )


def _measure_cuda_sample(
    runtime: RuntimeBindings,
    operation: Callable[[], None],
    measurement_iterations: int,
) -> float:
    start = runtime.torch.cuda.Event(enable_timing=True)
    end = runtime.torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measurement_iterations):
        operation()
    end.record()
    runtime.torch.cuda.synchronize()
    return float(start.elapsed_time(end)) * 1_000 / measurement_iterations


def _measure_bracketed(
    runtime: RuntimeBindings,
    *,
    fallback_operation: Callable[[], None],
    candidate_operation: Callable[[], None],
    route_seed: int,
    spec: TuningSpec,
) -> BracketTiming:
    for _ in range(spec.warmup_iterations):
        fallback_operation()
        candidate_operation()
    runtime.torch.cuda.synchronize()
    before: list[float] = []
    candidate: list[float] = []
    after: list[float] = []
    for _ in range(spec.independent_samples):
        before.append(
            _measure_cuda_sample(
                runtime, fallback_operation, spec.measurement_iterations
            )
        )
        candidate.append(
            _measure_cuda_sample(
                runtime, candidate_operation, spec.measurement_iterations
            )
        )
        after.append(
            _measure_cuda_sample(
                runtime, fallback_operation, spec.measurement_iterations
            )
        )
    return BracketTiming(
        route_seed=route_seed,
        candidate_microseconds=tuple(candidate),
        fallback_before_microseconds=tuple(before),
        fallback_after_microseconds=tuple(after),
    )


def _candidate_rejection_reason(error: BaseException) -> str | None:
    if isinstance(error, CandidateNumericalError):
        return f"numerical: {error}"
    if error.__class__.__name__ in {"OutOfResources", "OutOfMemoryError"}:
        return f"resource: {type(error).__name__}: {error}"
    return None


def _measure_rank_stage(
    runtime: RuntimeBindings,
    *,
    spec: TuningSpec,
    topology: OlmoeTopology,
    rank: int,
    correctness_workloads: Sequence[RuntimeWorkload],
    timing_workloads: Sequence[RuntimeWorkload],
    config: KernelConfig,
    stage: KernelStage,
) -> RankStageMeasurement:
    numerical: list[NumericalEvidence] = []
    for workload in correctness_workloads:
        prepared = _prepare_operation(
            runtime,
            topology=topology,
            workload=workload,
            config=config,
            stage=stage,
        )
        numerical.append(
            _validate_prepared_operation(
                runtime,
                spec=spec,
                prepared=prepared,
                workload=workload,
                rank=rank,
            )
        )
    route_timings: list[BracketTiming] = []
    for workload in timing_workloads:
        if workload.route_seed is None:
            raise RuntimeError("timing workload lacks a route seed")
        fallback = _prepare_operation(
            runtime,
            topology=topology,
            workload=workload,
            config=_copy_kernel_config(_FALLBACK_CONFIG),
            stage=stage,
        )
        candidate = _prepare_operation(
            runtime,
            topology=topology,
            workload=workload,
            config=config,
            stage=stage,
        )
        route_timings.append(
            _measure_bracketed(
                runtime,
                fallback_operation=fallback.operation,
                candidate_operation=candidate.operation,
                route_seed=workload.route_seed,
                spec=spec,
            )
        )
    return RankStageMeasurement(
        rank=rank,
        numerical_evidence=tuple(numerical),
        route_timings=tuple(route_timings),
    )


def tune_anchor(
    runtime: RuntimeBindings,
    *,
    spec: TuningSpec,
    topology: OlmoeTopology,
    batch_size: int,
) -> AnchorResult:
    accumulators = [
        _CandidateAccumulator(config=config) for config in build_rtx3090_search_space()
    ]
    for rank in range(topology.rank_count):
        inputs = _create_runtime_inputs(
            runtime,
            topology=topology,
            batch_size=batch_size,
            rank=rank,
        )
        timing_workloads = tuple(
            _create_seeded_workload(
                runtime,
                topology=topology,
                rank=rank,
                inputs=inputs,
                route_seed=route_seed,
            )
            for route_seed in spec.route_seeds
        )
        explicit_workloads = tuple(
            _create_explicit_workload(
                runtime,
                topology=topology,
                inputs=inputs,
                route_case=route_case,
            )
            for route_case in build_ep2_correctness_cases(
                topology=topology, rank=rank, batch_size=batch_size
            )
        )
        correctness_workloads = (*timing_workloads, *explicit_workloads)
        for accumulator in accumulators:
            if accumulator.rejection is not None:
                continue
            try:
                for stage, destination in (
                    ("gate_up", accumulator.gate_up_ranks),
                    ("down", accumulator.down_ranks),
                ):
                    destination.append(
                        _measure_rank_stage(
                            runtime,
                            spec=spec,
                            topology=topology,
                            rank=rank,
                            correctness_workloads=correctness_workloads,
                            timing_workloads=timing_workloads,
                            config=accumulator.config,
                            stage=cast(KernelStage, stage),
                        )
                    )
            except BaseException as error:
                rejection = _candidate_rejection_reason(error)
                if rejection is None or accumulator.config == _FALLBACK_CONFIG:
                    raise
                accumulator.rejection = (
                    f"EP{topology.ep_size} rank={rank} stage={stage}: {rejection}"
                )
                accumulator.gate_up_ranks.clear()
                accumulator.down_ranks.clear()
                runtime.torch.cuda.empty_cache()
        del correctness_workloads, explicit_workloads, timing_workloads, inputs
        runtime.torch.cuda.empty_cache()

    candidates: list[CandidateMeasurement] = []
    for accumulator in accumulators:
        if accumulator.rejection is not None:
            candidates.append(
                CandidateMeasurement(
                    config=accumulator.config,
                    gate_up=None,
                    down=None,
                    rejection=accumulator.rejection,
                )
            )
            continue
        candidates.append(
            CandidateMeasurement(
                config=accumulator.config,
                gate_up=StageMeasurement(
                    stage="gate_up",
                    config=accumulator.config,
                    ranks=tuple(accumulator.gate_up_ranks),
                    minimum_stable_improvement=spec.minimum_stable_improvement,
                ),
                down=StageMeasurement(
                    stage="down",
                    config=accumulator.config,
                    ranks=tuple(accumulator.down_ranks),
                    minimum_stable_improvement=spec.minimum_stable_improvement,
                ),
            )
        )
    selected = select_admitted_pair(candidates)
    return AnchorResult(
        ep_size=topology.ep_size,
        batch_size=batch_size,
        candidates=tuple(candidates),
        selected=selected,
    )


def run_live_tuning(
    *,
    spec: TuningSpec,
    output_directory: Path,
    runtime_install_receipt: Path,
    runtime_install_receipt_sha256: str,
    model_config: Path,
    model_config_sha256: str,
) -> Mapping[str, object]:
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    tuner_path = Path(__file__).resolve(strict=True)
    tuner_sha256 = _sha256_file(tuner_path)
    input_bindings = validate_live_input_bindings(
        runtime_install_receipt=runtime_install_receipt,
        runtime_install_receipt_sha256=runtime_install_receipt_sha256,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
    )
    runtime = load_runtime_bindings()
    _validate_live_runtime(runtime, input_bindings)
    runtime_provenance = _runtime_provenance(runtime)
    runtime.torch.set_grad_enabled(False)
    results: list[AnchorResult] = []
    with runtime.torch.inference_mode():
        for topology in OLMOE_TOPOLOGIES:
            for batch_size in spec.batch_anchors:
                print(
                    f"tuning EP{topology.ep_size} M={batch_size} "
                    f"candidates={len(build_rtx3090_search_space())}",
                    flush=True,
                )
                result = tune_anchor(
                    runtime,
                    spec=spec,
                    topology=topology,
                    batch_size=batch_size,
                )
                results.append(result)
                print(
                    f"selected EP{topology.ep_size} M={batch_size} "
                    f"gate={_kernel_config_key(result.selected.gate_up.config)} "
                    f"down={_kernel_config_key(result.selected.down.config)} "
                    f"stable={result.selected.admitted_stable_win}",
                    flush=True,
                )
    final_input_bindings = validate_live_input_bindings(
        runtime_install_receipt=runtime_install_receipt,
        runtime_install_receipt_sha256=runtime_install_receipt_sha256,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
    )
    if _canonical_json_bytes(final_input_bindings) != _canonical_json_bytes(
        input_bindings
    ):
        raise RuntimeError("authenticated live inputs changed during tuning")
    final_runtime_provenance = _runtime_provenance(runtime)
    if _canonical_json_bytes(final_runtime_provenance) != _canonical_json_bytes(
        runtime_provenance
    ):
        raise RuntimeError("runtime source or identity changed during tuning")
    if _sha256_file(tuner_path) != tuner_sha256:
        raise RuntimeError("tuner source changed during tuning")
    return write_tuning_bundle(
        output_directory=output_directory,
        spec=spec,
        results=results,
        runtime_provenance=runtime_provenance,
        input_bindings=input_bindings,
        tuner_path=tuner_path,
        expected_tuner_sha256=tuner_sha256,
    )


def _parse_route_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "route seeds must be comma-separated integers"
        ) from error
    if len(seeds) != 3:
        raise argparse.ArgumentTypeError("exactly three route seeds are required")
    return seeds


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-install-receipt",
        type=Path,
        help="absolute authenticated runtime install receipt path (required live)",
    )
    parser.add_argument(
        "--runtime-install-receipt-sha256",
        help="expected lowercase SHA-256 for --runtime-install-receipt",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="absolute pinned OLMoE config.json path (required live)",
    )
    parser.add_argument(
        "--model-config-sha256",
        help="expected lowercase SHA-256 for --model-config",
    )
    parser.add_argument(
        "--route-seeds", type=_parse_route_seeds, default=OLMOE_ROUTE_SEEDS
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument(
        "--measurement-iters", type=int, default=DEFAULT_MEASUREMENT_ITERATIONS
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_INDEPENDENT_SAMPLES)
    parser.add_argument(
        "--relative-l1-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_L1_TOLERANCE,
    )
    parser.add_argument(
        "--max-absolute-tolerance",
        type=float,
        default=DEFAULT_MAX_ABSOLUTE_TOLERANCE,
    )
    parser.add_argument(
        "--minimum-stable-improvement",
        type=float,
        default=MINIMUM_STABLE_IMPROVEMENT,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed workload/artifact contract without loading CUDA",
    )
    return parser


def _spec_from_arguments(arguments: argparse.Namespace) -> TuningSpec:
    return TuningSpec(
        route_seeds=arguments.route_seeds,
        warmup_iterations=arguments.warmup_iters,
        measurement_iterations=arguments.measurement_iters,
        independent_samples=arguments.samples,
        relative_l1_tolerance=arguments.relative_l1_tolerance,
        max_absolute_tolerance=arguments.max_absolute_tolerance,
        minimum_stable_improvement=arguments.minimum_stable_improvement,
    )


def _dry_run_preview(spec: TuningSpec, output_directory: Path) -> Mapping[str, object]:
    return {
        "model": {
            "hidden_size": OLMOE_HIDDEN_SIZE,
            "global_experts": OLMOE_GLOBAL_EXPERTS,
            "top_k": OLMOE_TOP_K,
            "normalize_top_k_weights": False,
        },
        "batch_anchors": list(spec.batch_anchors),
        "route_seeds": list(spec.route_seeds),
        "candidate_configs": [dict(config) for config in build_rtx3090_search_space()],
        "minimum_stable_improvement": spec.minimum_stable_improvement,
        "topologies": [
            {
                "ep_size": topology.ep_size,
                "ranks": topology.rank_count,
                "gate_up_weight_shape": list(topology.gate_up_weight_shape),
                "down_weight_shape": list(topology.down_weight_shape),
                "config_files": [
                    config_file_name(topology, down=False),
                    config_file_name(topology, down=True),
                ],
            }
            for topology in OLMOE_TOPOLOGIES
        ],
        "output_directory": str(output_directory),
        "artifact_status": "candidate; serving A/B required",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    parsed = parser.parse_args(arguments)
    try:
        spec = _spec_from_arguments(parsed)
    except ValueError as error:
        parser.error(str(error))
    if parsed.dry_run:
        print(
            _canonical_json_bytes(
                _dry_run_preview(spec, parsed.output_dir), pretty=True
            ).decode("utf-8"),
            end="",
        )
        return 0
    required_live_arguments = {
        "--runtime-install-receipt": parsed.runtime_install_receipt,
        "--runtime-install-receipt-sha256": (parsed.runtime_install_receipt_sha256),
        "--model-config": parsed.model_config,
        "--model-config-sha256": parsed.model_config_sha256,
    }
    missing_live_arguments = [
        name for name, value in required_live_arguments.items() if value is None
    ]
    if missing_live_arguments:
        parser.error(
            "live tuning requires " + ", ".join(sorted(missing_live_arguments))
        )
    manifest = run_live_tuning(
        spec=spec,
        output_directory=parsed.output_dir,
        runtime_install_receipt=cast(Path, parsed.runtime_install_receipt),
        runtime_install_receipt_sha256=cast(str, parsed.runtime_install_receipt_sha256),
        model_config=cast(Path, parsed.model_config),
        model_config_sha256=cast(str, parsed.model_config_sha256),
    )
    manifest_bytes = _canonical_json_bytes(manifest, pretty=True)
    print(
        _canonical_json_bytes(
            {
                "output_directory": str(parsed.output_dir),
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "sglang_moe_config_dir": str(parsed.output_dir),
            },
            pretty=True,
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
