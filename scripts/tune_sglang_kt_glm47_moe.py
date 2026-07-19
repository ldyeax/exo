#!/usr/bin/env python3
"""Tune the pinned SGLang BF16 fused-MoE kernels for GLM-4.7-Flash.

The production KTransformers path routes against all 64 GLM experts, keeps a
small prefix of experts resident on the GPU, and replaces CPU expert IDs with
``-1`` before invoking the GPU kernel.  SGLang's stock tuner generates routes
only among local experts, so it cannot represent that workload when the GPU
expert count is smaller than top-k.

This script keeps Torch, Triton, and SGLang imports inside the disposable live
entry point.  Run it with the exact pinned runtime and under the repository's
benchmark lease.  ``--dry-run`` is safe in the Exo development environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import stat
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypedDict, cast

GLM47_HIDDEN_SIZE: Final = 2_048
GLM47_INTERMEDIATE_SIZE: Final = 1_536
GLM47_GLOBAL_EXPERTS: Final = 64
GLM47_TOP_K: Final = 4
GLM47_TUNING_SEED: Final = 20_260_719
GLM47_BATCH_ANCHORS: Final = (1, 8, 32, 128, 512, 1_024, 4_096)
DEFAULT_RELATIVE_L1_TOLERANCE: Final = 0.02
DEFAULT_MAX_ABSOLUTE_TOLERANCE: Final = 0.02
MINIMUM_SYNTHETIC_IMPROVEMENT: Final = 0.05
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
SearchProfile = Literal["quick", "balanced"]
RouteScenarioName = Literal["uniform", "zero_resident", "mixed", "resident_skew"]
CandidateRejectionCategory = Literal["out_of_resources", "numerical"]


class KernelConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


class CandidateNumericalError(RuntimeError):
    """A candidate-specific numerical rejection with a healthy CUDA context."""


@dataclass(frozen=True)
class Glm47MoeTuningSpec:
    """Exact synthetic workload contract used for one tuning bundle."""

    hidden_size: int
    intermediate_size: int
    resident_experts: int
    global_experts: int
    top_k: int
    batch_sizes: tuple[int, ...]
    seed: int
    warmup_iterations: int
    measurement_iterations: int
    independent_samples: int
    search_profile: SearchProfile
    relative_l1_tolerance: float
    max_absolute_tolerance: float

    def __post_init__(self) -> None:
        positive_dimensions = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "resident_experts": self.resident_experts,
            "global_experts": self.global_experts,
            "top_k": self.top_k,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resident_experts >= self.global_experts:
            raise ValueError(
                "resident_experts must be smaller than global_experts so the "
                "tuning workload exercises masked CPU routes"
            )
        if self.top_k > self.global_experts:
            raise ValueError("top_k cannot exceed global_experts")
        cpu_expert_count = self.global_experts - self.resident_experts
        if self.top_k > cpu_expert_count:
            raise ValueError(
                "top_k requires enough non-resident experts to construct the "
                "zero-resident validation scenario"
            )
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive anchors")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        if tuple(sorted(self.batch_sizes)) != self.batch_sizes:
            raise ValueError("batch_sizes must be strictly increasing")
        if self.warmup_iterations < 1:
            raise ValueError("warmup_iterations must be positive")
        if self.measurement_iterations < 1:
            raise ValueError("measurement_iterations must be positive")
        if self.independent_samples < 3:
            raise ValueError("independent_samples must be at least three")
        if not math.isfinite(self.relative_l1_tolerance) or not (
            0 < self.relative_l1_tolerance <= 1
        ):
            raise ValueError("relative_l1_tolerance must be in (0, 1]")
        if not math.isfinite(self.max_absolute_tolerance) or not (
            0 < self.max_absolute_tolerance <= 1
        ):
            raise ValueError("max_absolute_tolerance must be in (0, 1]")


@dataclass(frozen=True)
class NumericalEvidence:
    relative_l1: float
    max_absolute: float
    repeat_exact: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_l1) or self.relative_l1 < 0:
            raise ValueError("relative_l1 must be finite and nonnegative")
        if not math.isfinite(self.max_absolute) or self.max_absolute < 0:
            raise ValueError("max_absolute must be finite and nonnegative")


@dataclass(frozen=True)
class RouteScenario:
    name: RouteScenarioName
    global_routes: tuple[tuple[int, ...], ...]
    masked_routes: tuple[tuple[int, ...], ...]
    resident_route_count: int
    masked_cpu_route_count: int

    def __post_init__(self) -> None:
        if not self.global_routes or len(self.global_routes) != len(self.masked_routes):
            raise ValueError("route scenario matrices must be nonempty and aligned")
        route_count = sum(len(routes) for routes in self.masked_routes)
        if self.resident_route_count + self.masked_cpu_route_count != route_count:
            raise ValueError("route scenario counts do not cover every route")

    @property
    def resident_routes_per_token(self) -> float:
        return self.resident_route_count / len(self.global_routes)


@dataclass(frozen=True)
class TimingRouteStratum:
    name: str
    resident_route_count: int
    probability_weight: float
    global_routes: tuple[tuple[int, ...], ...]
    masked_routes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.global_routes:
            raise ValueError("timing route stratum must be named and nonempty")
        if len(self.global_routes) != len(self.masked_routes):
            raise ValueError("timing route matrices must be aligned")
        if self.resident_route_count < 0:
            raise ValueError("resident_route_count must be nonnegative")
        if not math.isfinite(self.probability_weight) or not (
            0 < self.probability_weight <= 1
        ):
            raise ValueError("probability_weight must be in (0, 1]")
        if (
            sum(expert_id >= 0 for routes in self.masked_routes for expert_id in routes)
            != self.resident_route_count
        ):
            raise ValueError("timing routes do not match their resident-count stratum")

    @property
    def resident_routes_per_token(self) -> float:
        return self.resident_route_count / len(self.global_routes)


@dataclass(frozen=True)
class ScenarioNumericalEvidence:
    scenario: RouteScenarioName
    evidence: NumericalEvidence


@dataclass(frozen=True)
class StratumTimingEvidence:
    name: str
    resident_route_count: int
    resident_routes_per_token: float
    probability_weight: float
    sample_microseconds: tuple[float, ...]
    fallback_before_microseconds: tuple[float, ...]
    fallback_after_microseconds: tuple[float, ...]
    numerical_evidence: NumericalEvidence

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("timing evidence must be named")
        if self.resident_route_count < 0:
            raise ValueError("resident_route_count must be nonnegative")
        if not math.isfinite(self.resident_routes_per_token) or (
            self.resident_routes_per_token < 0
        ):
            raise ValueError("resident_routes_per_token must be finite and nonnegative")
        if (self.resident_route_count == 0) != (self.resident_routes_per_token == 0):
            raise ValueError("resident route count and rate disagree")
        if not math.isfinite(self.probability_weight) or not (
            0 < self.probability_weight <= 1
        ):
            raise ValueError("probability_weight must be in (0, 1]")
        sample_count = len(self.sample_microseconds)
        if sample_count == 0 or not (
            sample_count
            == len(self.fallback_before_microseconds)
            == len(self.fallback_after_microseconds)
        ):
            raise ValueError("stratum timing samples must be nonempty and aligned")
        if any(
            not math.isfinite(sample) or sample <= 0
            for sample in (
                *self.sample_microseconds,
                *self.fallback_before_microseconds,
                *self.fallback_after_microseconds,
            )
        ):
            raise ValueError("stratum timing samples must be finite and positive")


@dataclass(frozen=True)
class StageMeasurement:
    stage: KernelStage
    config: KernelConfig
    sample_microseconds: tuple[float, ...]
    fallback_before_microseconds: tuple[float, ...]
    fallback_after_microseconds: tuple[float, ...]
    numerical_evidence: tuple[ScenarioNumericalEvidence, ...]
    timing_strata: tuple[StratumTimingEvidence, ...]

    def __post_init__(self) -> None:
        sample_counts = {
            len(self.sample_microseconds),
            len(self.fallback_before_microseconds),
            len(self.fallback_after_microseconds),
        }
        if sample_counts != {len(self.sample_microseconds)} or not (
            self.sample_microseconds
        ):
            raise ValueError(
                "candidate and fallback samples must be nonempty and aligned"
            )
        all_samples = (
            *self.sample_microseconds,
            *self.fallback_before_microseconds,
            *self.fallback_after_microseconds,
        )
        if any(not math.isfinite(sample) or sample <= 0 for sample in all_samples):
            raise ValueError("timing samples must be finite and positive")
        observed_scenarios = tuple(item.scenario for item in self.numerical_evidence)
        expected_scenarios = cast(
            tuple[RouteScenarioName, ...],
            ("uniform", "zero_resident", "mixed", "resident_skew"),
        )
        if observed_scenarios != expected_scenarios:
            raise ValueError("numerical evidence must cover fixed route scenarios")
        if not self.timing_strata:
            raise ValueError("timing must include resident-count strata")
        if not math.isclose(
            sum(stratum.probability_weight for stratum in self.timing_strata),
            1.0,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("timing stratum weights must sum to one")
        resident_counts = tuple(
            stratum.resident_route_count for stratum in self.timing_strata
        )
        if resident_counts != tuple(sorted(set(resident_counts))):
            raise ValueError("timing strata must have unique ordered resident counts")
        if not any(count > 0 for count in resident_counts):
            raise ValueError("timing strata must exercise positive resident routes")
        for stratum in self.timing_strata:
            if len(stratum.sample_microseconds) != len(self.sample_microseconds):
                raise ValueError("aggregate and stratum sample counts must match")
        aggregate_groups = (
            self.sample_microseconds,
            self.fallback_before_microseconds,
            self.fallback_after_microseconds,
        )
        stratum_groups = tuple(
            (
                stratum.sample_microseconds,
                stratum.fallback_before_microseconds,
                stratum.fallback_after_microseconds,
            )
            for stratum in self.timing_strata
        )
        for group_index, aggregate in enumerate(aggregate_groups):
            for sample_index, observed in enumerate(aggregate):
                expected = sum(
                    stratum.probability_weight
                    * stratum_groups[stratum_index][group_index][sample_index]
                    for stratum_index, stratum in enumerate(self.timing_strata)
                )
                if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError("aggregate timing does not match weighted strata")

    @property
    def median_microseconds(self) -> float:
        return statistics.median(self.sample_microseconds)

    @property
    def fallback_reference_microseconds(self) -> tuple[float, ...]:
        if not (
            len(self.sample_microseconds)
            == len(self.fallback_before_microseconds)
            == len(self.fallback_after_microseconds)
        ):
            raise ValueError("candidate and fallback sample counts must match")
        return tuple(
            (before + after) / 2
            for before, after in zip(
                self.fallback_before_microseconds,
                self.fallback_after_microseconds,
                strict=True,
            )
        )

    @property
    def relative_improvement_samples(self) -> tuple[float, ...]:
        return tuple(
            (fallback - candidate) / fallback
            for candidate, fallback in zip(
                self.sample_microseconds,
                self.fallback_reference_microseconds,
                strict=True,
            )
        )

    def is_stably_faster(self, minimum_improvement: float) -> bool:
        return all(
            improvement >= minimum_improvement
            for improvement in self.relative_improvement_samples
        )


@dataclass(frozen=True)
class CandidatePair:
    block_size_m: int
    gate_up: StageMeasurement
    down: StageMeasurement

    def __post_init__(self) -> None:
        if self.gate_up.stage != "gate_up" or self.down.stage != "down":
            raise ValueError("candidate pair stages are invalid")
        if (
            self.gate_up.config["BLOCK_SIZE_M"] != self.block_size_m
            or self.down.config["BLOCK_SIZE_M"] != self.block_size_m
        ):
            raise ValueError("candidate pair must share BLOCK_SIZE_M")


@dataclass(frozen=True)
class CandidateRejection:
    stage: KernelStage
    category: CandidateRejectionCategory
    reason: str


@dataclass(frozen=True)
class CandidateRecord:
    config: KernelConfig
    measurements: tuple[StageMeasurement, ...]
    rejections: tuple[CandidateRejection, ...]

    def __post_init__(self) -> None:
        measurement_stages = [measurement.stage for measurement in self.measurements]
        rejection_stages = [rejection.stage for rejection in self.rejections]
        if any(measurement.config != self.config for measurement in self.measurements):
            raise ValueError("candidate measurement config differs from record")
        if sorted((*measurement_stages, *rejection_stages)) != ["down", "gate_up"]:
            raise ValueError("candidate record must contain one outcome per stage")


@dataclass(frozen=True)
class AnchorTuningResult:
    batch_size: int
    route_scenarios: tuple[RouteScenario, ...]
    timing_route_strata: tuple[TimingRouteStratum, ...]
    selected_pair: CandidatePair
    candidate_records: tuple[CandidateRecord, ...]

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scenario_names = tuple(scenario.name for scenario in self.route_scenarios)
        if scenario_names != (
            "uniform",
            "zero_resident",
            "mixed",
            "resident_skew",
        ):
            raise ValueError("anchor must retain the fixed route scenarios")
        stratum_counts = tuple(
            stratum.resident_route_count for stratum in self.timing_route_strata
        )
        if not stratum_counts or not any(count > 0 for count in stratum_counts):
            raise ValueError("anchor timing must exercise positive GEMM routes")
        candidate_keys = tuple(
            _kernel_config_key(record.config) for record in self.candidate_records
        )
        if not candidate_keys or candidate_keys != tuple(sorted(set(candidate_keys))):
            raise ValueError("candidate records must be unique and canonically ordered")

    @property
    def gate_up_config(self) -> KernelConfig:
        return self.selected_pair.gate_up.config

    @property
    def down_config(self) -> KernelConfig:
        return self.selected_pair.down.config


@dataclass(frozen=True)
class RuntimeBindings:
    torch: Any
    triton: Any
    triton_language: Any
    invoke_fused_moe_kernel: Callable[..., None]
    moe_align_block_size: Callable[[Any, int, int], tuple[Any, Any, Any]]
    sglang_revision: str
    ktransformers_revision: str


@dataclass
class RuntimeWorkload:
    scenario: str
    hidden_states: Any
    gate_up_weights: Any
    down_weights: Any
    route_weights: Any
    global_route_ids: Any
    masked_route_ids: Any
    gate_up_reference: Any
    down_input: Any
    down_reference: Any


_FALLBACK_CONFIG: Final[KernelConfig] = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
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


def build_rtx3090_search_space(profile: SearchProfile) -> tuple[KernelConfig, ...]:
    """Return a bounded Ampere search space containing the deployed fallback."""

    if profile == "quick":
        tile_shapes = ((32, 64), (64, 64), (64, 128), (128, 64))
        schedules = ((4, 2), (8, 2))
    elif profile == "balanced":
        tile_shapes = (
            (32, 64),
            (32, 128),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
            (256, 64),
            (256, 128),
        )
        schedules = ((4, 2), (4, 3), (8, 2))
    else:
        raise ValueError(f"unsupported search profile: {profile}")

    configs_by_key: dict[tuple[int, int, int, int, int, int], KernelConfig] = {}
    for block_size_m in (16, 32, 64, 128):
        for block_size_n, block_size_k in tile_shapes:
            for num_warps, num_stages in schedules:
                for group_size_m in (1, 8):
                    config: KernelConfig = {
                        "BLOCK_SIZE_M": block_size_m,
                        "BLOCK_SIZE_N": block_size_n,
                        "BLOCK_SIZE_K": block_size_k,
                        "GROUP_SIZE_M": group_size_m,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                    configs_by_key[_kernel_config_key(config)] = config
    configs_by_key[_kernel_config_key(_FALLBACK_CONFIG)] = cast(
        KernelConfig, dict(_FALLBACK_CONFIG)
    )
    return tuple(configs_by_key[key] for key in sorted(configs_by_key))


def build_global_routes(
    spec: Glm47MoeTuningSpec,
    batch_size: int,
    scenario: RouteScenarioName = "uniform",
) -> tuple[tuple[int, ...], ...]:
    """Build one deterministic route scenario using unique global top-k IDs."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resident_ids = tuple(range(spec.resident_experts))
    cpu_ids = tuple(range(spec.resident_experts, spec.global_experts))
    generator = random.Random(spec.seed + batch_size * 65_537)
    routes: list[tuple[int, ...]] = []
    for token_index in range(batch_size):
        if scenario == "uniform":
            token_routes = generator.sample(range(spec.global_experts), spec.top_k)
        elif scenario == "zero_resident":
            cpu_offset = (spec.seed + token_index * spec.top_k) % len(cpu_ids)
            token_routes = [
                cpu_ids[(cpu_offset + slot) % len(cpu_ids)]
                for slot in range(spec.top_k)
            ]
        elif scenario == "mixed":
            resident = resident_ids[(spec.seed + token_index) % len(resident_ids)]
            cpu_offset = (spec.seed + token_index * (spec.top_k - 1)) % len(cpu_ids)
            token_routes = [resident] + [
                cpu_ids[(cpu_offset + slot) % len(cpu_ids)]
                for slot in range(spec.top_k - 1)
            ]
        elif scenario == "resident_skew":
            resident_count = min(spec.resident_experts, spec.top_k)
            resident_offset = (spec.seed + token_index) % len(resident_ids)
            cpu_offset = (spec.seed + token_index) % len(cpu_ids)
            token_routes = [
                resident_ids[(resident_offset + slot) % len(resident_ids)]
                for slot in range(resident_count)
            ] + [
                cpu_ids[(cpu_offset + slot) % len(cpu_ids)]
                for slot in range(spec.top_k - resident_count)
            ]
        else:
            raise ValueError(f"unsupported route scenario: {scenario}")
        rotation = (spec.seed + token_index) % len(token_routes)
        token_routes = token_routes[rotation:] + token_routes[:rotation]
        if len(set(token_routes)) != spec.top_k:
            raise ValueError("generated duplicate expert routes for one token")
        routes.append(tuple(token_routes))
    return tuple(routes)


def build_route_scenarios(
    spec: Glm47MoeTuningSpec,
    batch_size: int,
) -> tuple[RouteScenario, ...]:
    """Build fixed-order performance and correctness route coverage."""

    scenarios: list[RouteScenario] = []
    for scenario_name in cast(
        tuple[RouteScenarioName, ...],
        ("uniform", "zero_resident", "mixed", "resident_skew"),
    ):
        global_routes = build_global_routes(spec, batch_size, scenario_name)
        masked_routes = mask_global_routes(
            global_routes,
            resident_experts=spec.resident_experts,
            global_experts=spec.global_experts,
        )
        flattened = [expert_id for row in masked_routes for expert_id in row]
        scenarios.append(
            RouteScenario(
                name=scenario_name,
                global_routes=global_routes,
                masked_routes=masked_routes,
                resident_route_count=sum(expert_id >= 0 for expert_id in flattened),
                masked_cpu_route_count=sum(expert_id == -1 for expert_id in flattened),
            )
        )
    return tuple(scenarios)


def _build_total_resident_routes(
    spec: Glm47MoeTuningSpec,
    batch_size: int,
    resident_route_count: int,
) -> tuple[tuple[int, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    maximum_per_token = min(spec.resident_experts, spec.top_k)
    if not 0 <= resident_route_count <= batch_size * maximum_per_token:
        raise ValueError("resident_route_count exceeds the batch route capacity")
    resident_ids = tuple(range(spec.resident_experts))
    cpu_ids = tuple(range(spec.resident_experts, spec.global_experts))
    resident_counts = [0] * batch_size
    for route_index in range(resident_route_count):
        token_index = (spec.seed + route_index) % batch_size
        if resident_counts[token_index] >= maximum_per_token:
            raise RuntimeError("resident route distribution exceeded token capacity")
        resident_counts[token_index] += 1
    routes: list[tuple[int, ...]] = []
    for token_index, token_resident_count in enumerate(resident_counts):
        cpu_route_count = spec.top_k - token_resident_count
        resident_offset = (
            spec.seed + token_index * max(token_resident_count, 1)
        ) % len(resident_ids)
        cpu_offset = (spec.seed + token_index * max(cpu_route_count, 1)) % len(cpu_ids)
        token_routes = [
            resident_ids[(resident_offset + slot) % len(resident_ids)]
            for slot in range(token_resident_count)
        ] + [
            cpu_ids[(cpu_offset + slot) % len(cpu_ids)]
            for slot in range(cpu_route_count)
        ]
        rotation = (spec.seed + token_index) % spec.top_k
        token_routes = token_routes[rotation:] + token_routes[:rotation]
        if len(set(token_routes)) != spec.top_k:
            raise ValueError("generated duplicate expert routes for one timing token")
        routes.append(tuple(token_routes))
    return tuple(routes)


def build_timing_route_strata(
    spec: Glm47MoeTuningSpec,
    batch_size: int,
) -> tuple[TimingRouteStratum, ...]:
    """Build floor/ceil strata preserving expected uniform resident routes."""

    strata: list[TimingRouteStratum] = []
    numerator = batch_size * spec.top_k * spec.resident_experts
    lower_count, remainder = divmod(numerator, spec.global_experts)
    weighted_counts = (
        ((lower_count, 1.0),)
        if remainder == 0
        else (
            (lower_count, 1 - remainder / spec.global_experts),
            (lower_count + 1, remainder / spec.global_experts),
        )
    )
    for resident_route_count, probability in weighted_counts:
        global_routes = _build_total_resident_routes(
            spec, batch_size, resident_route_count
        )
        masked_routes = mask_global_routes(
            global_routes,
            resident_experts=spec.resident_experts,
            global_experts=spec.global_experts,
        )
        strata.append(
            TimingRouteStratum(
                name=f"resident_total_{resident_route_count}",
                resident_route_count=resident_route_count,
                probability_weight=probability,
                global_routes=global_routes,
                masked_routes=masked_routes,
            )
        )
    if not math.isclose(
        sum(stratum.probability_weight for stratum in strata),
        1.0,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("resident-count stratum probabilities do not sum to one")
    if not any(stratum.resident_route_count > 0 for stratum in strata):
        raise RuntimeError("timing strata do not exercise a resident expert GEMM")
    return tuple(strata)


def mask_global_routes(
    global_routes: Sequence[Sequence[int]],
    *,
    resident_experts: int,
    global_experts: int,
) -> tuple[tuple[int, ...], ...]:
    """Map resident global IDs to GPU slots and CPU expert IDs to ``-1``."""

    masked: list[tuple[int, ...]] = []
    expected_width: int | None = None
    for token_routes in global_routes:
        if expected_width is None:
            expected_width = len(token_routes)
        elif len(token_routes) != expected_width:
            raise ValueError("global routes must be rectangular")
        masked_token: list[int] = []
        for expert_id in token_routes:
            if not 0 <= expert_id < global_experts:
                raise ValueError(f"global expert ID out of range: {expert_id}")
            masked_token.append(expert_id if expert_id < resident_experts else -1)
        masked.append(tuple(masked_token))
    if not masked or expected_width == 0:
        raise ValueError("global routes must not be empty")
    return tuple(masked)


def _integer_matrix_bytes(values: Sequence[Sequence[int]]) -> bytes:
    return json.dumps(values, separators=(",", ":")).encode("ascii")


def route_sha256(values: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(_integer_matrix_bytes(values)).hexdigest()


def select_candidate_pairs(
    gate_up_measurements: Sequence[StageMeasurement],
    down_measurements: Sequence[StageMeasurement],
    *,
    minimum_improvement: float = MINIMUM_SYNTHETIC_IMPROVEMENT,
) -> tuple[CandidatePair, ...]:
    """Choose only stable improvements, grouped by the shared route layout."""

    best_gate_up: dict[int, StageMeasurement] = {}
    best_down: dict[int, StageMeasurement] = {}
    for measurement in gate_up_measurements:
        if measurement.stage != "gate_up":
            raise ValueError("gate_up_measurements contains a non-gate measurement")
        if measurement.config != _FALLBACK_CONFIG and not measurement.is_stably_faster(
            minimum_improvement
        ):
            continue
        block_size_m = measurement.config["BLOCK_SIZE_M"]
        current = best_gate_up.get(block_size_m)
        if (
            current is None
            or measurement.median_microseconds < current.median_microseconds
        ):
            best_gate_up[block_size_m] = measurement
    for measurement in down_measurements:
        if measurement.stage != "down":
            raise ValueError("down_measurements contains a non-down measurement")
        if measurement.config != _FALLBACK_CONFIG and not measurement.is_stably_faster(
            minimum_improvement
        ):
            continue
        block_size_m = measurement.config["BLOCK_SIZE_M"]
        current = best_down.get(block_size_m)
        if (
            current is None
            or measurement.median_microseconds < current.median_microseconds
        ):
            best_down[block_size_m] = measurement
    return tuple(
        CandidatePair(
            block_size_m=block_size_m,
            gate_up=best_gate_up[block_size_m],
            down=best_down[block_size_m],
        )
        for block_size_m in sorted(best_gate_up.keys() & best_down.keys())
    )


def select_admitted_pair(
    gate_up_measurements: Sequence[StageMeasurement],
    down_measurements: Sequence[StageMeasurement],
    *,
    minimum_improvement: float = MINIMUM_SYNTHETIC_IMPROVEMENT,
) -> CandidatePair:
    """Select the fastest stable pair, retaining fallback when noise dominates."""

    pairs = select_candidate_pairs(
        gate_up_measurements,
        down_measurements,
        minimum_improvement=minimum_improvement,
    )
    if not pairs:
        raise RuntimeError("no valid fallback or stable candidate pair")
    return min(
        pairs,
        key=lambda pair: (
            pair.gate_up.median_microseconds + pair.down.median_microseconds,
            pair.block_size_m,
            _kernel_config_key(pair.gate_up.config),
            _kernel_config_key(pair.down.config),
        ),
    )


def nearest_kernel_config(
    configs: Mapping[int, KernelConfig], batch_size: int
) -> KernelConfig:
    """Mirror SGLang's nearest-anchor lookup over ascending JSON keys."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not configs:
        raise ValueError("configs must not be empty")
    anchor = min(sorted(configs), key=lambda value: abs(value - batch_size))
    return configs[anchor]


def config_file_name(
    *,
    resident_experts: int,
    intermediate_size: int,
    device_name: str,
    down: bool,
) -> str:
    """Mirror the pinned SGLang BF16 config filename exactly."""

    normalized_device = device_name.replace(" ", "_")
    if not normalized_device or any(
        character in normalized_device for character in ("/", "\\", "\0")
    ):
        raise ValueError("device_name is not safe in a config filename")
    suffix = "_down" if down else ""
    return (
        f"E={resident_experts},N={intermediate_size},"
        f"device_name={normalized_device}{suffix}.json"
    )


def _canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        rendered = json.dumps(value, indent=2, sort_keys=True)
    else:
        rendered = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return f"{rendered}\n".encode("utf-8")


def _validate_relative_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\0" in component
    ):
        raise ValueError(f"unsafe descriptor-relative component: {component!r}")


def _open_directory_at(parent_descriptor: int, component: str) -> int:
    _validate_relative_component(component)
    return os.open(
        component,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent_descriptor,
    )


def _create_directory_at(parent_descriptor: int, component: str) -> int:
    _validate_relative_component(component)
    os.mkdir(component, mode=0o700, dir_fd=parent_descriptor)
    descriptor = _open_directory_at(parent_descriptor, component)
    os.fsync(parent_descriptor)
    return descriptor


def _write_new_file_at(
    directory_descriptor: int, file_name: str, contents: bytes
) -> None:
    _validate_relative_component(file_name)
    descriptor = os.open(
        file_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(contents)
            output.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_descriptor)


def _validated_output_descriptor(
    output_directory_descriptor: int,
    authorization_evidence: Mapping[str, object],
) -> int:
    authorized_descriptor = authorization_evidence.get("output_directory_descriptor")
    identity = authorization_evidence.get("output_directory_identity")
    if type(authorized_descriptor) is not int or (
        output_directory_descriptor != authorized_descriptor
    ):
        raise RuntimeError("output descriptor differs from authorization")
    if not isinstance(identity, Mapping):
        raise RuntimeError("authorization lacks output directory identity")
    expected_device = identity.get("device")
    expected_inode = identity.get("inode")
    if type(expected_device) is not int or type(expected_inode) is not int:
        raise RuntimeError("authorized output directory identity is invalid")
    status = os.fstat(output_directory_descriptor)
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError("authorized output descriptor is not a directory")
    if (status.st_dev, status.st_ino) != (expected_device, expected_inode):
        raise RuntimeError("authorized output descriptor identity changed")
    duplicate = os.dup(output_directory_descriptor)
    os.set_inheritable(duplicate, False)
    return duplicate


def _numerical_evidence_json(evidence: NumericalEvidence) -> dict[str, object]:
    return {
        "relative_l1": evidence.relative_l1,
        "max_absolute": evidence.max_absolute,
        "repeat_exact": evidence.repeat_exact,
    }


def _stage_measurement_json(measurement: StageMeasurement) -> dict[str, object]:
    return {
        "stage": measurement.stage,
        "config": measurement.config,
        "sample_microseconds": list(measurement.sample_microseconds),
        "fallback_before_microseconds": list(measurement.fallback_before_microseconds),
        "fallback_after_microseconds": list(measurement.fallback_after_microseconds),
        "fallback_reference_microseconds": list(
            measurement.fallback_reference_microseconds
        ),
        "relative_improvement_samples": list(measurement.relative_improvement_samples),
        "stable_minimum_five_percent_improvement": (
            measurement.config != _FALLBACK_CONFIG
            and measurement.is_stably_faster(MINIMUM_SYNTHETIC_IMPROVEMENT)
        ),
        "numerical_evidence": [
            {
                "scenario": scenario.scenario,
                **_numerical_evidence_json(scenario.evidence),
            }
            for scenario in measurement.numerical_evidence
        ],
        "timing_strata": [
            {
                "name": stratum.name,
                "resident_route_count": stratum.resident_route_count,
                "resident_routes_per_token": stratum.resident_routes_per_token,
                "probability_weight": stratum.probability_weight,
                "sample_microseconds": list(stratum.sample_microseconds),
                "fallback_before_microseconds": list(
                    stratum.fallback_before_microseconds
                ),
                "fallback_after_microseconds": list(
                    stratum.fallback_after_microseconds
                ),
                "numerical_evidence": _numerical_evidence_json(
                    stratum.numerical_evidence
                ),
            }
            for stratum in measurement.timing_strata
        ],
    }


def _route_scenario_json(scenario: RouteScenario) -> dict[str, object]:
    return {
        "name": scenario.name,
        "global_route_sha256": route_sha256(scenario.global_routes),
        "masked_route_sha256": route_sha256(scenario.masked_routes),
        "resident_route_count": scenario.resident_route_count,
        "masked_cpu_route_count": scenario.masked_cpu_route_count,
        "resident_routes_per_token": scenario.resident_routes_per_token,
    }


def _timing_route_stratum_json(
    stratum: TimingRouteStratum,
) -> dict[str, object]:
    return {
        "name": stratum.name,
        "resident_route_count": stratum.resident_route_count,
        "resident_routes_per_token": stratum.resident_routes_per_token,
        "probability_weight": stratum.probability_weight,
        "global_route_sha256": route_sha256(stratum.global_routes),
        "masked_route_sha256": route_sha256(stratum.masked_routes),
    }


def _candidate_record_json(record: CandidateRecord) -> dict[str, object]:
    return {
        "config": record.config,
        "measurements": [
            _stage_measurement_json(measurement) for measurement in record.measurements
        ],
        "rejections": [
            {
                "stage": rejection.stage,
                "category": rejection.category,
                "reason": rejection.reason,
            }
            for rejection in record.rejections
        ],
    }


def write_tuning_bundle(
    *,
    output_directory: Path,
    output_directory_descriptor: int,
    spec: Glm47MoeTuningSpec,
    triton_version: str,
    torch_version: str,
    cuda_version: str,
    sglang_revision: str,
    ktransformers_revision: str,
    device_name: str,
    gpu_uuid: str,
    results: Sequence[AnchorTuningResult],
    authorization_evidence: Mapping[str, object],
    tuner_path: Path,
    tuner_sha256: str,
) -> Mapping[str, object]:
    """Write a candidate bundle below an authorized directory descriptor."""

    if not results:
        raise ValueError("cannot write an empty tuning bundle")
    if len({result.batch_size for result in results}) != len(results):
        raise ValueError("tuning results contain duplicate batch sizes")
    result_by_batch = {result.batch_size: result for result in results}
    if tuple(sorted(result_by_batch)) != spec.batch_sizes:
        raise ValueError("tuning results do not exactly cover batch_sizes")
    runtime_contract = _validated_runtime_contract(authorization_evidence)
    authorization_sha256 = authorization_evidence.get("authorization_sha256")
    receipt_sha256 = authorization_evidence.get("receipt_sha256")
    if not _is_sha256(authorization_sha256):
        raise RuntimeError("authorization evidence lacks its canonical digest")
    if not isinstance(receipt_sha256, Mapping) or (
        receipt_sha256.get("tuner_script") != tuner_sha256
    ):
        raise RuntimeError("tuner digest differs from authenticated receipt")
    observed_runtime_values = {
        "triton_version": triton_version,
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "sglang_revision": sglang_revision,
        "ktransformers_revision": ktransformers_revision,
        "gpu_name": device_name,
        "gpu_uuid": gpu_uuid,
    }
    for field_name, observed_value in observed_runtime_values.items():
        if runtime_contract.get(field_name) != observed_value:
            raise RuntimeError(
                f"bundle {field_name} differs from authenticated runtime contract"
            )
    authorization_bytes = _canonical_json_bytes(authorization_evidence)

    version_component = triton_version.replace(".", "_")
    if not version_component or any(
        character in version_component for character in ("/", "\\", "\0")
    ):
        raise ValueError("triton_version is not safe in a config path")
    gate_up_name = config_file_name(
        resident_experts=spec.resident_experts,
        intermediate_size=spec.intermediate_size,
        device_name=device_name,
        down=False,
    )
    down_name = config_file_name(
        resident_experts=spec.resident_experts,
        intermediate_size=spec.intermediate_size,
        device_name=device_name,
        down=True,
    )
    gate_up_configs = {
        str(batch_size): result_by_batch[batch_size].gate_up_config
        for batch_size in spec.batch_sizes
    }
    down_configs = {
        str(batch_size): result_by_batch[batch_size].down_config
        for batch_size in spec.batch_sizes
    }
    gate_up_contents = _canonical_json_bytes(gate_up_configs, pretty=True)
    down_contents = _canonical_json_bytes(down_configs, pretty=True)
    config_directory_name = "configs"
    version_directory_name = f"triton_{version_component}"
    gate_up_relative_path = (
        f"{config_directory_name}/{version_directory_name}/{gate_up_name}"
    )
    down_relative_path = f"{config_directory_name}/{version_directory_name}/{down_name}"

    output_descriptor = _validated_output_descriptor(
        output_directory_descriptor, authorization_evidence
    )
    try:
        if os.listdir(output_descriptor):
            raise FileExistsError(
                "authorized tuning output descriptor must reference an empty directory"
            )
        configs_descriptor = _create_directory_at(
            output_descriptor, config_directory_name
        )
        try:
            version_descriptor = _create_directory_at(
                configs_descriptor, version_directory_name
            )
            try:
                _write_new_file_at(version_descriptor, gate_up_name, gate_up_contents)
                _write_new_file_at(version_descriptor, down_name, down_contents)
            finally:
                os.close(version_descriptor)
        finally:
            os.close(configs_descriptor)

        manifest: dict[str, object] = {
            "schema_version": 3,
            "artifact_type": "glm47_sglang_kt_fused_moe_candidate_v3",
            "candidate": True,
            "deployment_admitted": False,
            "performance_comparable": False,
            "benchmark_scope": "synthetic_separate_stage_kernels_only",
            "production_path_reproduced": False,
            "limitations": [
                "alignment is prepared outside timed regions",
                "the pinned filtered activation and final reduction are not timed",
                (
                    "concurrent CPU AMX expert execution and resource contention "
                    "are not reproduced"
                ),
                "resident-route strata preserve the uniform expected count but are synthetic",
                "stratum weights assume uniform global top-k routing, not a captured trace",
                "the untuned serving baseline profile does not consume this bundle",
                "the current serving_baseline profile strips SGLANG_* variables",
            ],
            "serving_admission_required": {
                "minimum_improvement": 0.03,
                "metric": "matched end-to-end serving performance",
                "baseline_target_profile": (
                    "GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE"
                ),
                "baseline_profile_is_untuned": True,
                "baseline_profile_consumes_this_bundle": False,
                "consumer_chain": (
                    "KTEP gpu_method.apply -> SGLang fused_moe config loader"
                ),
                "required_environment_binding": (
                    "SGLANG_MOE_CONFIG_DIR with exact config file SHA-256"
                ),
                "current_serving_baseline_strips_sglang_environment": True,
                "profile": "future tuned profile with exact config SHA-256 binding",
            },
            "tuner": {
                "path": str(tuner_path),
                "sha256": tuner_sha256,
            },
            "authorization": {
                "authorization_sha256": authorization_evidence.get(
                    "authorization_sha256"
                ),
                "evidence": dict(authorization_evidence),
                "evidence_sha256": hashlib.sha256(authorization_bytes).hexdigest(),
                "output_directory_identity": authorization_evidence.get(
                    "output_directory_identity"
                ),
            },
            "output_contract": {
                "descriptor_anchored": True,
                "harness_created_empty_directory": True,
                "semantic_path": str(output_directory),
                "identity": authorization_evidence.get("output_directory_identity"),
            },
            "shape": {
                "H": spec.hidden_size,
                "N": spec.intermediate_size,
                "E": spec.resident_experts,
                "global_experts": spec.global_experts,
                "top_k": spec.top_k,
            },
            "workload": {
                "resident_experts": spec.resident_experts,
                "batch_sizes": list(spec.batch_sizes),
                "global_experts": spec.global_experts,
                "top_k": spec.top_k,
            },
            "route_contract": {
                "cpu_experts_are_masked_to": -1,
                "resident_global_expert_ids": list(range(spec.resident_experts)),
                "seed": spec.seed,
                "timed_scenarios": (
                    "deterministic expected total resident-route count strata"
                ),
                "decode_actual_resident_gemm_timed": True,
                "correctness_scenarios": [
                    "uniform",
                    "zero_resident",
                    "mixed",
                    "resident_skew",
                ],
                "uniform_expected_resident_routes_per_token": (
                    spec.top_k * spec.resident_experts / spec.global_experts
                ),
            },
            "measurement_contract": {
                "batch_sizes": list(spec.batch_sizes),
                "warmup_iterations": spec.warmup_iterations,
                "measurement_iterations": spec.measurement_iterations,
                "independent_samples": spec.independent_samples,
                "search_profile": spec.search_profile,
                "relative_l1_tolerance": spec.relative_l1_tolerance,
                "max_absolute_tolerance": spec.max_absolute_tolerance,
                "jit_compilation_excluded": True,
                "timing_source": "CUDA events",
                "candidate_order": "canonical fixed order",
                "drift_mitigation": "fallback-before/candidate/fallback-after",
                "minimum_stable_synthetic_improvement": (MINIMUM_SYNTHETIC_IMPROVEMENT),
                "selection_fallback": (
                    "retain deployed fallback unless every paired sample meets threshold"
                ),
            },
            "runtime_contract": dict(runtime_contract),
            "runtime_parent_receipts": {
                receipt_name: {
                    "sha256": cast(Mapping[str, object], receipt_sha256).get(
                        receipt_name
                    ),
                    "binding": cast(
                        Mapping[str, object],
                        runtime_contract["receipt_bindings"],
                    ).get(receipt_name),
                }
                for receipt_name in (
                    "runtime_install",
                    "runtime_build",
                    "kernel_validation",
                )
            },
            "runtime_observed": {
                "torch_version": torch_version,
                "cuda_version": cuda_version,
                "triton_version": triton_version,
                "sglang_revision": sglang_revision,
                "ktransformers_revision": ktransformers_revision,
                "device_name": device_name,
                "gpu_uuid": gpu_uuid,
            },
            "config_files": {
                "gate_up": {
                    "relative_path": gate_up_relative_path,
                    "sha256": hashlib.sha256(gate_up_contents).hexdigest(),
                    "shape": {
                        "E": spec.resident_experts,
                        "N": spec.intermediate_size,
                    },
                    "batch_keys": list(spec.batch_sizes),
                },
                "down": {
                    "relative_path": down_relative_path,
                    "sha256": hashlib.sha256(down_contents).hexdigest(),
                    "shape": {
                        "E": spec.resident_experts,
                        "N": spec.intermediate_size,
                    },
                    "batch_keys": list(spec.batch_sizes),
                },
            },
            "anchors": [
                {
                    "batch_size": result.batch_size,
                    "route_scenarios": [
                        _route_scenario_json(scenario)
                        for scenario in result.route_scenarios
                    ],
                    "timing_route_strata": [
                        _timing_route_stratum_json(stratum)
                        for stratum in result.timing_route_strata
                    ],
                    "selected": {
                        "shared_block_size_m": result.selected_pair.block_size_m,
                        "gate_up": _stage_measurement_json(
                            result.selected_pair.gate_up
                        ),
                        "down": _stage_measurement_json(result.selected_pair.down),
                        "gate_up_retained_fallback": (
                            result.gate_up_config == _FALLBACK_CONFIG
                        ),
                        "down_retained_fallback": (
                            result.down_config == _FALLBACK_CONFIG
                        ),
                    },
                    "candidate_records": [
                        _candidate_record_json(record)
                        for record in result.candidate_records
                    ],
                }
                for result in results
            ],
        }
        manifest_contents = _canonical_json_bytes(manifest, pretty=True)
        _write_new_file_at(output_descriptor, "manifest.json", manifest_contents)
        return manifest
    finally:
        os.close(output_descriptor)


def _validate_fused_moe_kernel_signature(kernel: Callable[..., None]) -> None:
    signature = inspect.signature(kernel)
    if tuple(signature.parameters) != _FUSED_MOE_PARAMETER_NAMES:
        raise RuntimeError(
            "pinned fused-MoE kernel interface changed; expected parameters "
            f"{_FUSED_MOE_PARAMETER_NAMES!r}, got {tuple(signature.parameters)!r}"
        )
    for name, parameter in signature.parameters.items():
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise RuntimeError(
                f"pinned fused-MoE kernel parameter {name} changed calling convention"
            )
        expected_default = _FUSED_MOE_OPTIONAL_DEFAULTS.get(
            name, inspect.Parameter.empty
        )
        if parameter.default != expected_default:
            raise RuntimeError(
                f"pinned fused-MoE kernel parameter {name} changed its default"
            )


def load_runtime_bindings() -> RuntimeBindings:
    """Resolve the pinned CUDA runtime only inside the live tuner process."""

    import torch
    import triton
    import triton.language as triton_language
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        invoke_fused_moe_kernel,
        moe_align_block_size,
    )

    _validate_fused_moe_kernel_signature(invoke_fused_moe_kernel)
    provenance = importlib.import_module("sglang._exo_build_provenance")
    provenance_schema_version = getattr(provenance, "SCHEMA_VERSION", None)
    sglang_revision = getattr(provenance, "SGLANG_REVISION", None)
    ktransformers_revision = getattr(provenance, "KTRANSFORMERS_REVISION", None)
    if provenance_schema_version != 1:
        raise RuntimeError(
            "pinned SGLang provenance schema must be version 1, got "
            f"{provenance_schema_version!r}"
        )
    revisions = {
        "SGLANG_REVISION": sglang_revision,
        "KTRANSFORMERS_REVISION": ktransformers_revision,
    }
    for name, revision in revisions.items():
        if not isinstance(revision, str) or len(revision) != 40:
            raise RuntimeError(f"invalid pinned runtime {name}: {revision!r}")
    return RuntimeBindings(
        torch=torch,
        triton=triton,
        triton_language=triton_language,
        invoke_fused_moe_kernel=invoke_fused_moe_kernel,
        moe_align_block_size=moe_align_block_size,
        sglang_revision=sglang_revision,
        ktransformers_revision=ktransformers_revision,
    )


def _tensor_numerical_evidence(
    runtime: RuntimeBindings,
    *,
    actual: Any,
    expected: Any,
    repeated: Any,
    spec: Glm47MoeTuningSpec,
) -> NumericalEvidence:
    torch = runtime.torch
    if not bool(torch.isfinite(actual).all().item()):
        raise CandidateNumericalError("candidate produced non-finite output")
    repeat_exact = bool(torch.equal(actual, repeated))
    if not repeat_exact:
        raise CandidateNumericalError(
            "candidate output was not bitwise stable across repeats"
        )
    actual_float = actual.float()
    expected_float = expected.float()
    absolute_difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().sum().clamp_min(1e-12)
    relative_l1 = float((absolute_difference.sum() / denominator).item())
    max_absolute = float(absolute_difference.max().item())
    if relative_l1 > spec.relative_l1_tolerance:
        raise CandidateNumericalError(
            "candidate relative L1 error exceeded tolerance: "
            f"{relative_l1} > {spec.relative_l1_tolerance}"
        )
    if max_absolute > spec.max_absolute_tolerance:
        raise CandidateNumericalError(
            "candidate maximum absolute error exceeded tolerance: "
            f"{max_absolute} > {spec.max_absolute_tolerance}"
        )
    return NumericalEvidence(
        relative_l1=relative_l1,
        max_absolute=max_absolute,
        repeat_exact=repeat_exact,
    )


def _create_runtime_workload(
    runtime: RuntimeBindings,
    *,
    spec: Glm47MoeTuningSpec,
    batch_size: int,
    scenario: RouteScenario | TimingRouteStratum,
) -> RuntimeWorkload:
    torch = runtime.torch
    generator = torch.Generator(device="cuda")
    scenario_digest = hashlib.sha256(scenario.name.encode("ascii")).digest()
    scenario_seed_offset = int.from_bytes(scenario_digest[:4], byteorder="big")
    generator.manual_seed(spec.seed + batch_size + scenario_seed_offset)
    hidden_states = torch.randn(
        (batch_size, spec.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate_up_weights = torch.randn(
        (spec.resident_experts, 2 * spec.intermediate_size, spec.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    down_weights = torch.randn(
        (spec.resident_experts, spec.hidden_size, spec.intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    route_weights = torch.rand(
        (batch_size, spec.top_k),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    route_weights.div_(route_weights.sum(dim=-1, keepdim=True))
    global_route_ids = torch.tensor(
        scenario.global_routes, device="cuda", dtype=torch.int32
    )
    masked_route_ids = torch.tensor(
        scenario.masked_routes, device="cuda", dtype=torch.int32
    )

    route_count = batch_size * spec.top_k
    token_ids = torch.arange(batch_size, device="cuda").repeat_interleave(spec.top_k)
    flattened_routes = masked_route_ids.reshape(-1)
    gate_up_reference = torch.zeros(
        (route_count, 2 * spec.intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert_index in range(spec.resident_experts):
        route_positions = torch.nonzero(
            flattened_routes == expert_index, as_tuple=False
        ).reshape(-1)
        if route_positions.numel() == 0:
            continue
        expert_inputs = hidden_states[token_ids[route_positions]]
        gate_up_reference[route_positions] = torch.nn.functional.linear(
            expert_inputs, gate_up_weights[expert_index]
        )
    down_input = (
        torch.nn.functional.silu(gate_up_reference[:, : spec.intermediate_size])
        * gate_up_reference[:, spec.intermediate_size :]
    )
    down_reference = torch.zeros(
        (route_count, spec.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    flattened_weights = route_weights.reshape(-1)
    for expert_index in range(spec.resident_experts):
        route_positions = torch.nonzero(
            flattened_routes == expert_index, as_tuple=False
        ).reshape(-1)
        if route_positions.numel() == 0:
            continue
        expert_outputs = torch.nn.functional.linear(
            down_input[route_positions], down_weights[expert_index]
        )
        down_reference[route_positions] = expert_outputs * flattened_weights[
            route_positions
        ].unsqueeze(-1)

    return RuntimeWorkload(
        scenario=scenario.name,
        hidden_states=hidden_states,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        route_weights=route_weights,
        global_route_ids=global_route_ids,
        masked_route_ids=masked_route_ids,
        gate_up_reference=gate_up_reference,
        down_input=down_input,
        down_reference=down_reference,
    )


def _align_routes(
    runtime: RuntimeBindings,
    workload: RuntimeWorkload,
    config: KernelConfig,
    resident_experts: int,
) -> tuple[Any, Any, Any]:
    return runtime.moe_align_block_size(
        workload.masked_route_ids,
        config["BLOCK_SIZE_M"],
        resident_experts,
    )


def _invoke_gate_up(
    runtime: RuntimeBindings,
    *,
    workload: RuntimeWorkload,
    config: KernelConfig,
    aligned_routes: tuple[Any, Any, Any],
    output: Any,
    top_k: int,
) -> None:
    sorted_token_ids, expert_ids, num_tokens_post_padded = aligned_routes
    runtime.invoke_fused_moe_kernel(
        A=workload.hidden_states,
        B=workload.gate_up_weights,
        bias=None,
        C=output,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=workload.route_weights,
        topk_ids=workload.masked_route_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
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


def _invoke_down(
    runtime: RuntimeBindings,
    *,
    workload: RuntimeWorkload,
    config: KernelConfig,
    aligned_routes: tuple[Any, Any, Any],
    down_input: Any,
    output: Any,
) -> None:
    sorted_token_ids, expert_ids, num_tokens_post_padded = aligned_routes
    runtime.invoke_fused_moe_kernel(
        A=down_input,
        B=workload.down_weights,
        bias=None,
        C=output,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        topk_weights=workload.route_weights,
        topk_ids=workload.masked_route_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
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


@dataclass(frozen=True)
class PreparedStageOperation:
    operation: Callable[[], None]
    output: Any
    expected: Any
    route_count: int


def _prepare_stage_operation(
    runtime: RuntimeBindings,
    *,
    spec: Glm47MoeTuningSpec,
    workload: RuntimeWorkload,
    config: KernelConfig,
    stage: KernelStage,
) -> PreparedStageOperation:
    torch = runtime.torch
    aligned_routes = _align_routes(runtime, workload, config, spec.resident_experts)
    capacity = int(aligned_routes[0].shape[0])
    route_count = int(workload.masked_route_ids.numel())
    if stage == "gate_up":
        output = torch.zeros(
            (capacity, 2 * spec.intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def operation() -> None:
            _invoke_gate_up(
                runtime,
                workload=workload,
                config=config,
                aligned_routes=aligned_routes,
                output=output,
                top_k=spec.top_k,
            )

        expected = workload.gate_up_reference
    else:
        down_input = torch.zeros(
            (capacity, spec.intermediate_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        down_input[:route_count].copy_(workload.down_input)
        output = torch.zeros(
            (workload.hidden_states.shape[0], spec.top_k, spec.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )

        def operation() -> None:
            _invoke_down(
                runtime,
                workload=workload,
                config=config,
                aligned_routes=aligned_routes,
                down_input=down_input,
                output=output,
            )

        expected = workload.down_reference
    return PreparedStageOperation(
        operation=operation,
        output=output,
        expected=expected,
        route_count=route_count,
    )


def _validate_prepared_stage(
    runtime: RuntimeBindings,
    *,
    spec: Glm47MoeTuningSpec,
    prepared: PreparedStageOperation,
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
    return _tensor_numerical_evidence(
        runtime,
        actual=first,
        expected=prepared.expected,
        repeated=repeated,
        spec=spec,
    )


def _measure_cuda_sample(
    runtime: RuntimeBindings,
    operation: Callable[[], None],
    measurement_iterations: int,
) -> float:
    torch = runtime.torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measurement_iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) * 1_000 / measurement_iterations


def _measure_interleaved_cuda_operations(
    runtime: RuntimeBindings,
    *,
    fallback_operation: Callable[[], None],
    candidate_operation: Callable[[], None],
    warmup_iterations: int,
    measurement_iterations: int,
    independent_samples: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Measure fallback/candidate/fallback to bracket monotonic drift."""

    for _ in range(warmup_iterations):
        fallback_operation()
        candidate_operation()
    runtime.torch.cuda.synchronize()
    fallback_before: list[float] = []
    candidate: list[float] = []
    fallback_after: list[float] = []
    for _ in range(independent_samples):
        fallback_before.append(
            _measure_cuda_sample(runtime, fallback_operation, measurement_iterations)
        )
        candidate.append(
            _measure_cuda_sample(runtime, candidate_operation, measurement_iterations)
        )
        fallback_after.append(
            _measure_cuda_sample(runtime, fallback_operation, measurement_iterations)
        )
    return tuple(candidate), tuple(fallback_before), tuple(fallback_after)


def _validate_and_measure_stage(
    runtime: RuntimeBindings,
    *,
    spec: Glm47MoeTuningSpec,
    correctness_workloads: Sequence[RuntimeWorkload],
    timing_workloads: Sequence[tuple[TimingRouteStratum, RuntimeWorkload]],
    config: KernelConfig,
    stage: KernelStage,
) -> StageMeasurement:
    numerical_evidence: list[ScenarioNumericalEvidence] = []
    for workload in correctness_workloads:
        prepared = _prepare_stage_operation(
            runtime,
            spec=spec,
            workload=workload,
            config=config,
            stage=stage,
        )
        try:
            evidence = _validate_prepared_stage(runtime, spec=spec, prepared=prepared)
        except CandidateNumericalError as error:
            raise CandidateNumericalError(
                f"correctness scenario {workload.scenario}: {error}"
            ) from error
        numerical_evidence.append(
            ScenarioNumericalEvidence(
                scenario=cast(RouteScenarioName, workload.scenario),
                evidence=evidence,
            )
        )

    stratum_evidence: list[StratumTimingEvidence] = []
    for stratum, workload in timing_workloads:
        candidate = _prepare_stage_operation(
            runtime,
            spec=spec,
            workload=workload,
            config=config,
            stage=stage,
        )
        fallback = _prepare_stage_operation(
            runtime,
            spec=spec,
            workload=workload,
            config=_FALLBACK_CONFIG,
            stage=stage,
        )
        try:
            timing_numerical_evidence = _validate_prepared_stage(
                runtime, spec=spec, prepared=candidate
            )
        except CandidateNumericalError as error:
            raise CandidateNumericalError(
                f"timing stratum {stratum.name}: {error}"
            ) from error
        samples, fallback_before, fallback_after = _measure_interleaved_cuda_operations(
            runtime,
            fallback_operation=fallback.operation,
            candidate_operation=candidate.operation,
            warmup_iterations=spec.warmup_iterations,
            measurement_iterations=spec.measurement_iterations,
            independent_samples=spec.independent_samples,
        )
        stratum_evidence.append(
            StratumTimingEvidence(
                name=stratum.name,
                resident_route_count=stratum.resident_route_count,
                resident_routes_per_token=stratum.resident_routes_per_token,
                probability_weight=stratum.probability_weight,
                sample_microseconds=samples,
                fallback_before_microseconds=fallback_before,
                fallback_after_microseconds=fallback_after,
                numerical_evidence=timing_numerical_evidence,
            )
        )
    if not stratum_evidence:
        raise ValueError("timing workloads must not be empty")

    def weighted_samples(sample_group: int) -> tuple[float, ...]:
        samples_by_stratum = tuple(
            (
                stratum.sample_microseconds,
                stratum.fallback_before_microseconds,
                stratum.fallback_after_microseconds,
            )[sample_group]
            for stratum in stratum_evidence
        )
        return tuple(
            sum(
                stratum.probability_weight
                * samples_by_stratum[stratum_index][sample_index]
                for stratum_index, stratum in enumerate(stratum_evidence)
            )
            for sample_index in range(spec.independent_samples)
        )

    return StageMeasurement(
        stage=stage,
        config=config,
        sample_microseconds=weighted_samples(0),
        fallback_before_microseconds=weighted_samples(1),
        fallback_after_microseconds=weighted_samples(2),
        numerical_evidence=tuple(numerical_evidence),
        timing_strata=tuple(stratum_evidence),
    )


def tune_anchor(
    runtime: RuntimeBindings,
    *,
    spec: Glm47MoeTuningSpec,
    batch_size: int,
    search_space: Sequence[KernelConfig],
) -> AnchorTuningResult:
    search_keys = tuple(_kernel_config_key(config) for config in search_space)
    if search_keys != tuple(sorted(set(search_keys))):
        raise ValueError("search_space must be unique and in canonical fixed order")
    route_scenarios = build_route_scenarios(spec, batch_size)
    correctness_workloads = tuple(
        _create_runtime_workload(
            runtime,
            spec=spec,
            batch_size=batch_size,
            scenario=scenario,
        )
        for scenario in route_scenarios
    )
    timing_route_strata = build_timing_route_strata(spec, batch_size)
    timing_workloads = tuple(
        (
            stratum,
            _create_runtime_workload(
                runtime,
                spec=spec,
                batch_size=batch_size,
                scenario=stratum,
            ),
        )
        for stratum in timing_route_strata
    )
    gate_up_measurements: list[StageMeasurement] = []
    down_measurements: list[StageMeasurement] = []
    candidate_records: list[CandidateRecord] = []
    out_of_resources = runtime.triton.runtime.autotuner.OutOfResources
    for config in search_space:
        measurements: list[StageMeasurement] = []
        rejections: list[CandidateRejection] = []
        for stage in cast(tuple[KernelStage, KernelStage], ("gate_up", "down")):
            try:
                measurement = _validate_and_measure_stage(
                    runtime,
                    spec=spec,
                    correctness_workloads=correctness_workloads,
                    timing_workloads=timing_workloads,
                    config=config,
                    stage=stage,
                )
            except CandidateNumericalError as error:
                rejection = CandidateRejection(
                    stage=stage,
                    category="numerical",
                    reason=str(error),
                )
                rejections.append(rejection)
                print(
                    f"reject batch={batch_size} stage={stage} "
                    f"category={rejection.category} config={dict(config)} "
                    f"reason={rejection.reason}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            except out_of_resources as error:
                rejection = CandidateRejection(
                    stage=stage,
                    category="out_of_resources",
                    reason=str(error),
                )
                rejections.append(rejection)
                print(
                    f"reject batch={batch_size} stage={stage} "
                    f"category={rejection.category} config={dict(config)} "
                    f"reason={rejection.reason}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            measurements.append(measurement)
            if stage == "gate_up":
                gate_up_measurements.append(measurement)
            else:
                down_measurements.append(measurement)
        candidate_records.append(
            CandidateRecord(
                config=config,
                measurements=tuple(measurements),
                rejections=tuple(rejections),
            )
        )

    fallback_stages = {
        measurement.stage
        for measurement in (*gate_up_measurements, *down_measurements)
        if measurement.config == _FALLBACK_CONFIG
    }
    if fallback_stages != {"gate_up", "down"}:
        raise RuntimeError(
            "deployed fallback failed correctness or resource validation"
        )
    selected_pair = select_admitted_pair(
        gate_up_measurements,
        down_measurements,
        minimum_improvement=MINIMUM_SYNTHETIC_IMPROVEMENT,
    )
    return AnchorTuningResult(
        batch_size=batch_size,
        route_scenarios=route_scenarios,
        timing_route_strata=timing_route_strata,
        selected_pair=selected_pair,
        candidate_records=tuple(candidate_records),
    )


def _cuda_device_uuid(torch: Any, device_index: int) -> str:
    properties = torch.cuda.get_device_properties(device_index)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        raise RuntimeError("Torch did not expose the CUDA device UUID")
    return str(uuid)


def _require_contract_string(contract: Mapping[str, object], field_name: str) -> str:
    value = contract.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"runtime contract {field_name} must be a nonempty string")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: set[str],
    description: str,
) -> None:
    observed_keys = set(value)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(str(key) for key in observed_keys - expected_keys)
        raise RuntimeError(
            f"{description} keys differ from the authenticated schema: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _require_absolute_contract_path(
    value: object,
    description: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\0" in value
        or not Path(value).is_absolute()
    ):
        raise RuntimeError(f"{description} must be a nonempty absolute path")
    return value


def _validated_sha256_mapping(
    value: object,
    description: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{description} is invalid")
    mapping = cast(Mapping[object, object], value)
    if any(
        not isinstance(name, str) or not name or not _is_sha256(digest)
        for name, digest in mapping.items()
    ):
        raise RuntimeError(f"{description} is invalid")
    return cast(Mapping[str, object], value)


def _validated_runtime_contract(
    authorization_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    contract_value = authorization_evidence.get("runtime_contract")
    if not isinstance(contract_value, Mapping):
        raise RuntimeError("authorization lacks an authenticated runtime contract")
    contract = cast(Mapping[str, object], contract_value)
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "install_id",
            "install_root",
            "runtime_python",
            "base_runtime_python_sha256",
            "base_runtime_pip_freeze_sha256",
            "installed_distribution_record_sha256",
            "installed_distribution_file_counts",
            "build_id",
            "sglang_revision",
            "ktransformers_revision",
            "package_version",
            "runtime_wheel_sha256",
            "fused_moe_distribution_sha256",
            "kt_extension_sha256",
            "embedded_provenance_sha256",
            "torch_version",
            "triton_version",
            "torch_cuda_version",
            "gpu_uuid",
            "gpu_name",
            "compute_capability",
            "capabilities",
            "receipt_bindings",
        },
        "runtime contract",
    )
    if type(contract.get("schema_version")) is not int or (
        contract.get("schema_version") != 1
    ):
        raise RuntimeError("runtime contract schema must be version 1")
    for field_name in ("install_id", "build_id"):
        if not _is_sha256(contract.get(field_name)):
            raise RuntimeError(f"runtime contract {field_name} is not a SHA-256")
    _require_absolute_contract_path(
        contract.get("install_root"), "runtime contract install_root"
    )
    required_strings = (
        "sglang_revision",
        "ktransformers_revision",
        "package_version",
        "torch_version",
        "triton_version",
        "torch_cuda_version",
        "gpu_uuid",
        "gpu_name",
    )
    for field_name in required_strings:
        _require_contract_string(contract, field_name)
    for field_name in ("sglang_revision", "ktransformers_revision"):
        revision = cast(str, contract[field_name])
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise RuntimeError(
                f"runtime contract {field_name} is not a lowercase commit hash"
            )
    for field_name in (
        "kt_extension_sha256",
        "fused_moe_distribution_sha256",
        "base_runtime_python_sha256",
        "base_runtime_pip_freeze_sha256",
    ):
        if not _is_sha256(contract.get(field_name)):
            raise RuntimeError(f"runtime contract {field_name} is not a SHA-256")
    installed_record_hashes = _validated_sha256_mapping(
        contract.get("installed_distribution_record_sha256"),
        "runtime contract installed distribution RECORD hashes",
    )
    installed_file_counts = contract.get("installed_distribution_file_counts")
    if not isinstance(installed_file_counts, Mapping):
        raise RuntimeError("runtime contract installed distribution counts are invalid")
    distribution_file_counts = cast(Mapping[str, object], installed_file_counts)
    if set(distribution_file_counts) != set(installed_record_hashes) or any(
        type(file_count) is not int or file_count <= 0
        for file_count in distribution_file_counts.values()
    ):
        raise RuntimeError("runtime contract installed distribution counts are invalid")
    runtime_wheel_hashes = _validated_sha256_mapping(
        contract.get("runtime_wheel_sha256"),
        "runtime contract wheel hashes",
    )
    if runtime_wheel_hashes.get("sglang-kt") != contract.get(
        "fused_moe_distribution_sha256"
    ):
        raise RuntimeError("runtime contract fused-MoE distribution binding differs")
    runtime_python = contract.get("runtime_python")
    if not isinstance(runtime_python, Mapping):
        raise RuntimeError("runtime contract runtime_python binding is invalid")
    runtime_python_binding = cast(Mapping[str, object], runtime_python)
    _require_exact_keys(
        runtime_python_binding,
        {"path", "sha256", "symlink_chain"},
        "runtime Python binding",
    )
    _require_absolute_contract_path(
        runtime_python_binding.get("path"), "runtime Python path"
    )
    if not _is_sha256(runtime_python_binding.get("sha256")):
        raise RuntimeError("runtime contract runtime_python digest is invalid")
    symlink_chain = runtime_python_binding.get("symlink_chain")
    if not isinstance(symlink_chain, list) or any(
        not isinstance(target, str) or not target or "\0" in target
        for target in symlink_chain
    ):
        raise RuntimeError("runtime contract runtime_python symlink chain is invalid")
    embedded_provenance = _validated_sha256_mapping(
        contract.get("embedded_provenance_sha256"),
        "runtime contract embedded provenance",
    )
    if set(embedded_provenance) != {"kt-kernel", "sglang-kt"}:
        raise RuntimeError("runtime contract embedded provenance is invalid")
    capabilities = contract.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or not all(
            isinstance(capability_name, str) and capability_name
            for capability_name in capabilities
        )
    ):
        raise RuntimeError("runtime contract capabilities are missing")
    if len(set(cast(list[str], capabilities))) != len(capabilities) or (
        "kt_bf16_amx_executed_v1" not in capabilities
    ):
        raise RuntimeError("runtime contract capabilities are inconsistent")
    capability = contract.get("compute_capability")
    if capability != [8, 6]:
        raise RuntimeError("runtime contract is not for an RTX 3090 SM86 GPU")
    receipt_hashes = authorization_evidence.get("receipt_sha256")
    if not isinstance(receipt_hashes, Mapping):
        raise RuntimeError("authorization lacks receipt SHA-256 bindings")
    authorization_receipt_hashes = cast(Mapping[str, object], receipt_hashes)
    contract_receipts = contract.get("receipt_bindings")
    if not isinstance(contract_receipts, Mapping):
        raise RuntimeError("runtime contract receipt bindings are missing")
    runtime_receipt_bindings = cast(Mapping[str, object], contract_receipts)
    _require_exact_keys(
        runtime_receipt_bindings,
        {"runtime_install", "runtime_build", "kernel_validation"},
        "runtime contract receipt bindings",
    )
    if authorization_receipt_hashes.get("runtime_python") != runtime_python_binding.get(
        "sha256"
    ):
        raise RuntimeError("runtime Python receipt binding is inconsistent")
    for receipt_name in (
        "runtime_install",
        "runtime_build",
        "kernel_validation",
    ):
        if not _is_sha256(authorization_receipt_hashes.get(receipt_name)):
            raise RuntimeError(
                f"authorization lacks authenticated {receipt_name} receipt"
            )
        receipt_binding = runtime_receipt_bindings.get(receipt_name)
        if not isinstance(receipt_binding, Mapping):
            raise RuntimeError(
                f"runtime contract {receipt_name} receipt binding is invalid"
            )
        typed_receipt_binding = cast(Mapping[str, object], receipt_binding)
        _require_exact_keys(
            typed_receipt_binding,
            {"path", "sha256"},
            f"runtime contract {receipt_name} receipt binding",
        )
        _require_absolute_contract_path(
            typed_receipt_binding.get("path"),
            f"runtime contract {receipt_name} receipt path",
        )
        if typed_receipt_binding.get("sha256") != authorization_receipt_hashes.get(
            receipt_name
        ):
            raise RuntimeError(
                f"runtime contract {receipt_name} receipt binding is inconsistent"
            )
    return contract


def _validate_loaded_runtime(
    runtime: RuntimeBindings,
    runtime_contract: Mapping[str, object],
) -> Mapping[str, object]:
    torch = runtime.torch
    device_name = str(torch.cuda.get_device_name(0))
    gpu_uuid = _cuda_device_uuid(torch, 0)
    compute_capability = tuple(torch.cuda.get_device_capability(0))
    observed: dict[str, object] = {
        "sglang_revision": runtime.sglang_revision,
        "ktransformers_revision": runtime.ktransformers_revision,
        "torch_version": str(torch.__version__),
        "triton_version": str(runtime.triton.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "gpu_name": device_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": list(compute_capability),
    }
    for field_name, observed_value in observed.items():
        expected_value = runtime_contract.get(field_name)
        if field_name == "compute_capability" and isinstance(expected_value, tuple):
            expected_value = list(expected_value)
        if observed_value != expected_value:
            raise RuntimeError(
                f"loaded runtime {field_name} differs from authenticated contract: "
                f"{observed_value!r} != {expected_value!r}"
            )
    return observed


def run_live_tuning(
    *,
    spec: Glm47MoeTuningSpec,
    output_directory: Path,
) -> Mapping[str, object]:
    tuner_path = Path(__file__).resolve()
    tuner_sha256 = hashlib.sha256(tuner_path.read_bytes()).hexdigest()
    from run_sglang_kt_glm47_moe_tuning import validate_tuner_authorization

    authorization_evidence = validate_tuner_authorization(
        output_directory=output_directory
    )
    receipt_sha256 = authorization_evidence.get("receipt_sha256")
    if not isinstance(receipt_sha256, Mapping) or (
        receipt_sha256.get("tuner_script") != tuner_sha256
    ):
        raise RuntimeError(
            "authorized tuner digest differs from the pre-CUDA source digest"
        )
    runtime_contract = _validated_runtime_contract(authorization_evidence)
    output_directory_descriptor = authorization_evidence.get(
        "output_directory_descriptor"
    )
    if type(output_directory_descriptor) is not int:
        raise RuntimeError("authorization lacks the inherited output descriptor")
    output_precheck_descriptor = _validated_output_descriptor(
        output_directory_descriptor, authorization_evidence
    )
    try:
        if os.listdir(output_precheck_descriptor):
            raise FileExistsError(
                "authorized tuning output descriptor must be empty before CUDA"
            )
    finally:
        os.close(output_precheck_descriptor)
    runtime = load_runtime_bindings()
    torch = runtime.torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the pinned runtime")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one CUDA device must be visible; bind CUDA_VISIBLE_DEVICES first"
        )
    runtime_observed = _validate_loaded_runtime(runtime, runtime_contract)
    device_name = cast(str, runtime_observed["gpu_name"])
    torch.set_grad_enabled(False)
    search_space = build_rtx3090_search_space(spec.search_profile)
    results: list[AnchorTuningResult] = []
    with torch.inference_mode():
        for batch_size in spec.batch_sizes:
            print(
                f"tuning batch={batch_size} candidates={len(search_space)}",
                flush=True,
            )
            result = tune_anchor(
                runtime,
                spec=spec,
                batch_size=batch_size,
                search_space=search_space,
            )
            results.append(result)
            print(
                f"selected batch={batch_size} "
                f"gate_up_us={result.selected_pair.gate_up.median_microseconds:.3f} "
                f"down_us={result.selected_pair.down.median_microseconds:.3f} "
                f"gate_fallback={result.gate_up_config == _FALLBACK_CONFIG} "
                f"down_fallback={result.down_config == _FALLBACK_CONFIG}",
                flush=True,
            )
    return write_tuning_bundle(
        output_directory=output_directory,
        output_directory_descriptor=output_directory_descriptor,
        spec=spec,
        triton_version=str(runtime.triton.__version__),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        sglang_revision=runtime.sglang_revision,
        ktransformers_revision=runtime.ktransformers_revision,
        device_name=device_name,
        gpu_uuid=_cuda_device_uuid(torch, 0),
        results=results,
        authorization_evidence=authorization_evidence,
        tuner_path=tuner_path,
        tuner_sha256=tuner_sha256,
    )


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        batch_sizes = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch sizes must be comma-separated integers"
        ) from error
    if not batch_sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return batch_sizes


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=GLM47_HIDDEN_SIZE)
    parser.add_argument(
        "--intermediate-size", type=int, default=GLM47_INTERMEDIATE_SIZE
    )
    parser.add_argument("--resident-experts", type=int, required=True)
    parser.add_argument("--global-experts", type=int, default=GLM47_GLOBAL_EXPERTS)
    parser.add_argument("--top-k", type=int, default=GLM47_TOP_K)
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=GLM47_BATCH_ANCHORS,
    )
    parser.add_argument("--seed", type=int, default=GLM47_TUNING_SEED)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measurement-iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--search-profile", choices=("quick", "balanced"), default="quick"
    )
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
        "--dry-run",
        action="store_true",
        help="print the validated workload/search contract without loading CUDA",
    )
    return parser


def _spec_from_arguments(arguments: argparse.Namespace) -> Glm47MoeTuningSpec:
    return Glm47MoeTuningSpec(
        hidden_size=arguments.hidden_size,
        intermediate_size=arguments.intermediate_size,
        resident_experts=arguments.resident_experts,
        global_experts=arguments.global_experts,
        top_k=arguments.top_k,
        batch_sizes=arguments.batch_sizes,
        seed=arguments.seed,
        warmup_iterations=arguments.warmup_iters,
        measurement_iterations=arguments.measurement_iters,
        independent_samples=arguments.samples,
        search_profile=arguments.search_profile,
        relative_l1_tolerance=arguments.relative_l1_tolerance,
        max_absolute_tolerance=arguments.max_absolute_tolerance,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    parsed = parser.parse_args(arguments)
    try:
        spec = _spec_from_arguments(parsed)
    except ValueError as error:
        parser.error(str(error))
    if parsed.dry_run:
        preview = {
            "spec": {
                "H": spec.hidden_size,
                "N": spec.intermediate_size,
                "E": spec.resident_experts,
                "global_experts": spec.global_experts,
                "top_k": spec.top_k,
                "batch_sizes": list(spec.batch_sizes),
                "uniform_expected_resident_routes_per_token": (
                    spec.top_k * spec.resident_experts / spec.global_experts
                ),
                "route_scenarios": [
                    "uniform",
                    "zero_resident",
                    "mixed",
                    "resident_skew",
                ],
                "timing_route_strata": [
                    {
                        "resident_route_count": stratum.resident_route_count,
                        "resident_routes_per_token": (
                            stratum.resident_routes_per_token
                        ),
                        "probability_weight": stratum.probability_weight,
                    }
                    for stratum in build_timing_route_strata(spec, 1)
                ],
            },
            "candidate_count": len(build_rtx3090_search_space(spec.search_profile)),
            "output_directory": str(parsed.output_dir),
            "live_output_contract": (
                "harness-created empty inherited directory descriptor"
            ),
            "live_runtime_identity_source": (
                "authenticated harness runtime_contract receipts"
            ),
        }
        print(_canonical_json_bytes(preview, pretty=True).decode("utf-8"), end="")
        return 0
    manifest = run_live_tuning(
        spec=spec,
        output_directory=parsed.output_dir,
    )
    print(
        _canonical_json_bytes(
            {
                "output_directory": str(parsed.output_dir),
                "manifest_sha256": hashlib.sha256(
                    _canonical_json_bytes(manifest, pretty=True)
                ).hexdigest(),
                "sglang_moe_config_dir": str(parsed.output_dir),
            },
            pretty=True,
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
