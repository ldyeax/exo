#!/usr/bin/env python3
"""Build a deterministic, profile-guided two-host Kimi K3 expert plan.

The planner is deliberately limited to placement metadata.  It does not imply
that llama.cpp can yet load compact expert tables or execute cross-host expert
partials.

Inputs
------
``--expert-bytes`` is required and must be JSON containing either:

* ``{"expert_bytes": [[... 896 integers ...], ... 92 rows ...]}``, or
* ``{"bytes_per_expert_by_layer": [... 92 integers ...]}``.

The compact form is exact for GGUFs whose three fused expert tensors use one
fixed-size slab per expert within a layer, including the current Kimi K3 GGUFs.

``--route-counts`` is optional.  When supplied, it must contain a raw 92 by 896
JSON matrix or an object with that matrix under ``route_counts`` or
``logical_count``.  When omitted, the documented fallback assigns a count of
one to every expert.  That fallback is deterministic and throughput-aware, but
it is not profile-guided and the output records this fact.

Exactly two ``--host NAME:CAPACITY_BYTES:RELATIVE_THROUGHPUT`` arguments are
required.  Throughput is an arbitrary positive relative rate; only ratios
matter.  The predicted cost of a layer is:

    max(route_count_on_host_0 / throughput_0,
        route_count_on_host_1 / throughput_1)

Layers execute sequentially while the two hosts' expert work is assumed to run
concurrently.  Network latency, route co-occurrence, and runtime overhead are
not inferable from aggregate route counts and are intentionally not invented.

The assignment uses deterministic longest-processing-time balancing within
each layer.  If that unconstrained placement exceeds a host's byte budget, it
moves experts to the other host in order of minimum marginal makespan cost per
byte.  This is a deterministic heuristic, not a claim of global optimality.
The emitted ownership and global/local maps are validated before writing.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, cast

ROUTED_LAYER_COUNT: Final = 92
EXPERT_COUNT: Final = 896
MODEL_BLOCK_OFFSET: Final = 1
PLAN_KIND: Final = "kimi_k3_profile_guided_expert_ownership_v1"


class PlanError(ValueError):
    """Raised when inputs or the resulting ownership plan are invalid."""


@dataclass(frozen=True)
class HostSpec:
    name: str
    capacity_bytes: int
    relative_expert_throughput: float


@dataclass
class _PlacementState:
    owners: list[list[int]]
    route_loads: list[list[int]]
    used_bytes: list[int]


class _Arguments(argparse.Namespace):
    expert_bytes: Path
    route_counts: Path | None
    host: list[HostSpec]
    output: Path


def _as_object_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast(list[object], value)


def parse_host_spec(value: str) -> HostSpec:
    """Parse NAME:CAPACITY_BYTES:RELATIVE_THROUGHPUT."""
    parts = value.rsplit(":", maxsplit=2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "host must be NAME:CAPACITY_BYTES:RELATIVE_THROUGHPUT"
        )
    name, capacity_text, throughput_text = parts
    if not name:
        raise argparse.ArgumentTypeError("host name must not be empty")
    try:
        capacity_bytes = int(capacity_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "host capacity must be an integer number of bytes"
        ) from error
    try:
        throughput = float(throughput_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "host relative throughput must be a number"
        ) from error
    if capacity_bytes < 0:
        raise argparse.ArgumentTypeError("host capacity must not be negative")
    if not math.isfinite(throughput) or throughput <= 0:
        raise argparse.ArgumentTypeError(
            "host relative throughput must be finite and positive"
        )
    return HostSpec(
        name=name,
        capacity_bytes=capacity_bytes,
        relative_expert_throughput=throughput,
    )


def _read_json(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"could not read JSON from {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PlanError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _extract_matrix(payload: object, keys: Sequence[str], description: str) -> object:
    if isinstance(payload, dict):
        payload_object = cast(dict[object, object], payload)
        for key in keys:
            if key in payload_object:
                return payload_object[key]
        expected = " or ".join(repr(key) for key in keys)
        raise PlanError(f"{description} object must contain {expected}")
    return payload


def _validate_route_counts(raw: object) -> list[list[int]]:
    raw_rows = _as_object_list(raw)
    if raw_rows is None or len(raw_rows) != ROUTED_LAYER_COUNT:
        raise PlanError(
            f"route counts must have {ROUTED_LAYER_COUNT} rows of "
            f"{EXPERT_COUNT} experts"
        )
    result: list[list[int]] = []
    for layer_index, raw_row in enumerate(raw_rows):
        raw_values = _as_object_list(raw_row)
        if raw_values is None or len(raw_values) != EXPERT_COUNT:
            raise PlanError(
                f"route-count row {layer_index} must contain {EXPERT_COUNT} values"
            )
        row: list[int] = []
        for expert_id, raw_value in enumerate(raw_values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise PlanError(
                    f"route count [{layer_index}][{expert_id}] must be an integer"
                )
            if raw_value < 0:
                raise PlanError(
                    f"route count [{layer_index}][{expert_id}] must not be negative"
                )
            row.append(raw_value)
        result.append(row)
    return result


def load_route_counts(path: Path | None) -> tuple[list[list[int]], dict[str, object]]:
    """Load an exact profile or return the documented uniform fallback."""
    if path is None:
        return (
            [[1] * EXPERT_COUNT for _ in range(ROUTED_LAYER_COUNT)],
            {
                "kind": "uniform_fallback",
                "description": (
                    "No route profile was supplied; every layer/expert received "
                    "one synthetic route."
                ),
            },
        )
    payload = _read_json(path)
    raw_matrix = _extract_matrix(
        payload,
        ("route_counts", "logical_count"),
        "route-count",
    )
    return (
        _validate_route_counts(raw_matrix),
        {
            "kind": "profile",
            "sha256": _sha256(path),
        },
    )


def _validate_expert_bytes_matrix(raw: object) -> list[list[int]]:
    raw_rows = _as_object_list(raw)
    if raw_rows is None or len(raw_rows) != ROUTED_LAYER_COUNT:
        raise PlanError(
            f"expert bytes must have {ROUTED_LAYER_COUNT} rows of "
            f"{EXPERT_COUNT} experts"
        )
    result: list[list[int]] = []
    for layer_index, raw_row in enumerate(raw_rows):
        raw_values = _as_object_list(raw_row)
        if raw_values is None or len(raw_values) != EXPERT_COUNT:
            raise PlanError(
                f"expert-byte row {layer_index} must contain {EXPERT_COUNT} values"
            )
        row: list[int] = []
        for expert_id, raw_value in enumerate(raw_values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise PlanError(
                    f"expert bytes [{layer_index}][{expert_id}] must be an integer"
                )
            if raw_value <= 0:
                raise PlanError(
                    f"expert bytes [{layer_index}][{expert_id}] must be positive"
                )
            row.append(raw_value)
        result.append(row)
    return result


def load_expert_bytes(path: Path) -> tuple[list[list[int]], dict[str, object]]:
    """Load an exact expanded or per-layer compact expert-byte inventory."""
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise PlanError(
            "expert-byte JSON must be an object containing 'expert_bytes' or "
            "'bytes_per_expert_by_layer'"
        )
    payload_object = cast(dict[object, object], payload)
    if "expert_bytes" in payload_object:
        matrix = _validate_expert_bytes_matrix(payload_object["expert_bytes"])
        encoding = "per_expert_matrix"
    elif "bytes_per_expert_by_layer" in payload_object:
        raw_by_layer = payload_object["bytes_per_expert_by_layer"]
        raw_layer_values = _as_object_list(raw_by_layer)
        if raw_layer_values is None or len(raw_layer_values) != ROUTED_LAYER_COUNT:
            raise PlanError(
                "'bytes_per_expert_by_layer' must contain exactly "
                f"{ROUTED_LAYER_COUNT} integers"
            )
        by_layer: list[int] = []
        for layer_index, raw_value in enumerate(raw_layer_values):
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise PlanError(
                    f"bytes per expert for layer {layer_index} must be an integer"
                )
            if raw_value <= 0:
                raise PlanError(
                    f"bytes per expert for layer {layer_index} must be positive"
                )
            by_layer.append(raw_value)
        matrix = [[value] * EXPERT_COUNT for value in by_layer]
        encoding = "uniform_within_layer"
    else:
        raise PlanError(
            "expert-byte JSON must contain 'expert_bytes' or "
            "'bytes_per_expert_by_layer'"
        )
    return (
        matrix,
        {
            "encoding": encoding,
            "sha256": _sha256(path),
        },
    )


def _layer_makespan(
    route_loads: Sequence[int],
    hosts: Sequence[HostSpec],
) -> float:
    return max(
        route_loads[host_index] / host.relative_expert_throughput
        for host_index, host in enumerate(hosts)
    )


def _initial_placement(
    route_counts: Sequence[Sequence[int]],
    expert_bytes: Sequence[Sequence[int]],
    hosts: Sequence[HostSpec],
) -> _PlacementState:
    owners: list[list[int]] = []
    route_loads = [[0, 0] for _ in range(ROUTED_LAYER_COUNT)]
    used_bytes = [0, 0]
    for layer_index in range(ROUTED_LAYER_COUNT):
        layer_owners = [-1] * EXPERT_COUNT
        ordered_experts = sorted(
            range(EXPERT_COUNT),
            key=lambda expert_id: (
                -route_counts[layer_index][expert_id],
                -expert_bytes[layer_index][expert_id],
                expert_id,
            ),
        )
        for expert_id in ordered_experts:
            route_count = route_counts[layer_index][expert_id]
            candidate_keys: list[tuple[float, float, int]] = []
            for host_index in range(len(hosts)):
                candidate_loads = route_loads[layer_index].copy()
                candidate_loads[host_index] += route_count
                normalized = [
                    candidate_loads[index] / hosts[index].relative_expert_throughput
                    for index in range(2)
                ]
                candidate_keys.append(
                    (
                        max(normalized),
                        abs(normalized[0] - normalized[1]),
                        host_index,
                    )
                )
            chosen_host = min(range(2), key=lambda index: candidate_keys[index])
            layer_owners[expert_id] = chosen_host
            route_loads[layer_index][chosen_host] += route_count
            used_bytes[chosen_host] += expert_bytes[layer_index][expert_id]
        owners.append(layer_owners)
    return _PlacementState(
        owners=owners,
        route_loads=route_loads,
        used_bytes=used_bytes,
    )


def _move_delta(
    state: _PlacementState,
    route_counts: Sequence[Sequence[int]],
    hosts: Sequence[HostSpec],
    layer_index: int,
    expert_id: int,
    source_host: int,
) -> float:
    target_host = 1 - source_host
    current_loads = state.route_loads[layer_index]
    candidate_loads = current_loads.copy()
    route_count = route_counts[layer_index][expert_id]
    candidate_loads[source_host] -= route_count
    candidate_loads[target_host] += route_count
    return _layer_makespan(candidate_loads, hosts) - _layer_makespan(
        current_loads,
        hosts,
    )


def _best_layer_capacity_repair_move(
    state: _PlacementState,
    route_counts: Sequence[Sequence[int]],
    expert_bytes: Sequence[Sequence[int]],
    hosts: Sequence[HostSpec],
    source_host: int,
    layer_index: int,
) -> tuple[tuple[float, float, int, int, int], int] | None:
    target_host = 1 - source_host
    target_free_bytes = (
        hosts[target_host].capacity_bytes - state.used_bytes[target_host]
    )
    best_key: tuple[float, float, int, int, int] | None = None
    best_expert_id: int | None = None
    for expert_id, owner in enumerate(state.owners[layer_index]):
        if owner != source_host:
            continue
        byte_count = expert_bytes[layer_index][expert_id]
        if byte_count > target_free_bytes:
            continue
        delta = _move_delta(
            state,
            route_counts,
            hosts,
            layer_index,
            expert_id,
            source_host,
        )
        key = (
            delta / byte_count,
            delta,
            -byte_count,
            layer_index,
            expert_id,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_expert_id = expert_id
    if best_key is None or best_expert_id is None:
        return None
    return best_key, best_expert_id


def _apply_move(
    state: _PlacementState,
    route_counts: Sequence[Sequence[int]],
    expert_bytes: Sequence[Sequence[int]],
    source_host: int,
    layer_index: int,
    expert_id: int,
) -> None:
    target_host = 1 - source_host
    if state.owners[layer_index][expert_id] != source_host:
        raise RuntimeError("capacity repair selected an expert from the wrong host")
    state.owners[layer_index][expert_id] = target_host
    route_count = route_counts[layer_index][expert_id]
    byte_count = expert_bytes[layer_index][expert_id]
    state.route_loads[layer_index][source_host] -= route_count
    state.route_loads[layer_index][target_host] += route_count
    state.used_bytes[source_host] -= byte_count
    state.used_bytes[target_host] += byte_count


def _repair_capacity(
    state: _PlacementState,
    route_counts: Sequence[Sequence[int]],
    expert_bytes: Sequence[Sequence[int]],
    hosts: Sequence[HostSpec],
) -> None:
    total_expert_bytes = sum(state.used_bytes)
    total_capacity_bytes = sum(host.capacity_bytes for host in hosts)
    if total_capacity_bytes < total_expert_bytes:
        raise PlanError(
            "combined host capacity is insufficient: "
            f"required={total_expert_bytes}, available={total_capacity_bytes}"
        )
    largest_expert_bytes = max(max(row) for row in expert_bytes)
    for host_index, host in enumerate(hosts):
        if largest_expert_bytes > host.capacity_bytes:
            other_host = hosts[1 - host_index]
            if largest_expert_bytes > other_host.capacity_bytes:
                raise PlanError(
                    "an expert is larger than both host capacities: "
                    f"largest_expert_bytes={largest_expert_bytes}"
                )

    move_count = 0
    maximum_moves = ROUTED_LAYER_COUNT * EXPERT_COUNT
    repair_heap: list[tuple[tuple[float, float, int, int, int], int]] = []
    active_source_host: int | None = None
    while True:
        overfull_hosts = [
            host_index
            for host_index, host in enumerate(hosts)
            if state.used_bytes[host_index] > host.capacity_bytes
        ]
        if not overfull_hosts:
            return
        if len(overfull_hosts) != 1:
            raise PlanError("both hosts exceed capacity; combined capacity is invalid")
        source_host = overfull_hosts[0]
        if active_source_host != source_host:
            repair_heap.clear()
            active_source_host = source_host
            for layer_index in range(ROUTED_LAYER_COUNT):
                candidate = _best_layer_capacity_repair_move(
                    state,
                    route_counts,
                    expert_bytes,
                    hosts,
                    source_host,
                    layer_index,
                )
                if candidate is not None:
                    heapq.heappush(repair_heap, candidate)
        if not repair_heap:
            raise PlanError(
                "could not find a discrete expert assignment within both byte "
                "budgets; increase a host budget or change the exact byte inventory"
            )
        move_key, expert_id = heapq.heappop(repair_heap)
        layer_index = move_key[3]
        byte_count = expert_bytes[layer_index][expert_id]
        target_host = 1 - source_host
        target_free_bytes = (
            hosts[target_host].capacity_bytes - state.used_bytes[target_host]
        )
        if (
            state.owners[layer_index][expert_id] != source_host
            or byte_count > target_free_bytes
        ):
            candidate = _best_layer_capacity_repair_move(
                state,
                route_counts,
                expert_bytes,
                hosts,
                source_host,
                layer_index,
            )
            if candidate is not None:
                heapq.heappush(repair_heap, candidate)
            continue
        _apply_move(
            state,
            route_counts,
            expert_bytes,
            source_host,
            layer_index,
            expert_id,
        )
        next_candidate = _best_layer_capacity_repair_move(
            state,
            route_counts,
            expert_bytes,
            hosts,
            source_host,
            layer_index,
        )
        if next_candidate is not None:
            heapq.heappush(repair_heap, next_candidate)
        move_count += 1
        if move_count > maximum_moves:
            raise RuntimeError("capacity repair exceeded the number of experts")


def _host_layer_map(
    owners: Sequence[int],
    host_index: int,
) -> tuple[list[int], list[int]]:
    local_to_global = [
        expert_id for expert_id, owner in enumerate(owners) if owner == host_index
    ]
    global_to_local = [-1] * EXPERT_COUNT
    for local_id, global_id in enumerate(local_to_global):
        global_to_local[global_id] = local_id
    return local_to_global, global_to_local


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_new_output_path(
    output_path: Path,
    input_paths: Sequence[Path],
) -> None:
    """Refuse output aliases and any pre-existing destination."""

    if output_path.is_symlink():
        raise PlanError(f"refusing symlink output path: {output_path}")
    try:
        output_identity = output_path.resolve(strict=False)
    except OSError as error:
        raise PlanError(
            f"could not resolve output path {output_path}: {error}"
        ) from error
    for input_path in input_paths:
        try:
            input_identity = input_path.resolve(strict=True)
        except OSError as error:
            raise PlanError(
                f"could not resolve input path {input_path}: {error}"
            ) from error
        if output_identity == input_identity:
            raise PlanError(f"output path aliases input path: {input_path}")
    if output_path.exists():
        raise PlanError(f"refusing to overwrite existing output: {output_path}")


def atomic_write_new_text(destination: Path, content: str) -> None:
    """Durably publish a new file without replacing an existing path."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
    except OSError as error:
        raise PlanError(
            f"could not create temporary output beside {destination}: {error}"
        ) from error

    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as error:
            raise PlanError(
                f"refusing to overwrite output created concurrently: {destination}"
            ) from error
        temporary_path.unlink()
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except PlanError:
        raise
    except OSError as error:
        raise PlanError(f"could not publish output {destination}: {error}") from error
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


def build_plan(
    *,
    route_counts: Sequence[Sequence[int]],
    expert_bytes: Sequence[Sequence[int]],
    hosts: Sequence[HostSpec],
    route_profile_metadata: dict[str, object] | None = None,
    expert_byte_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build and validate deterministic K3 ownership and remapping metadata."""
    validated_counts = _validate_route_counts([list(row) for row in route_counts])
    validated_bytes = _validate_expert_bytes_matrix([list(row) for row in expert_bytes])
    if len(hosts) != 2:
        raise PlanError("exactly two hosts are required")
    host_tuple = tuple(hosts)
    if host_tuple[0].name == host_tuple[1].name:
        raise PlanError("host names must be unique")
    for host in host_tuple:
        if host.capacity_bytes < 0:
            raise PlanError("host capacity must not be negative")
        if (
            not math.isfinite(host.relative_expert_throughput)
            or host.relative_expert_throughput <= 0
        ):
            raise PlanError("host relative throughput must be finite and positive")

    state = _initial_placement(
        validated_counts,
        validated_bytes,
        host_tuple,
    )
    unconstrained_used_bytes = state.used_bytes.copy()
    unconstrained_makespan = sum(
        _layer_makespan(loads, host_tuple) for loads in state.route_loads
    )
    _repair_capacity(
        state,
        validated_counts,
        validated_bytes,
        host_tuple,
    )

    layer_payloads: list[dict[str, object]] = []
    for layer_index in range(ROUTED_LAYER_COUNT):
        host_payloads: list[dict[str, object]] = []
        for host_index, host in enumerate(host_tuple):
            local_to_global, global_to_local = _host_layer_map(
                state.owners[layer_index],
                host_index,
            )
            layer_owned_bytes = sum(
                validated_bytes[layer_index][expert_id] for expert_id in local_to_global
            )
            route_load = state.route_loads[layer_index][host_index]
            host_payloads.append(
                {
                    "host_index": host_index,
                    "host_name": host.name,
                    "owned_expert_count": len(local_to_global),
                    "owned_bytes": layer_owned_bytes,
                    "route_count": route_load,
                    "predicted_expert_time": (
                        route_load / host.relative_expert_throughput
                    ),
                    "local_to_global": local_to_global,
                    "global_to_local": global_to_local,
                }
            )
        layer_payloads.append(
            {
                "routed_layer_index": layer_index,
                "model_block_index": layer_index + MODEL_BLOCK_OFFSET,
                "expert_bytes_by_global_expert": validated_bytes[layer_index],
                "owner_by_global_expert": state.owners[layer_index],
                "hosts": host_payloads,
                "predicted_concurrent_makespan": _layer_makespan(
                    state.route_loads[layer_index],
                    host_tuple,
                ),
            }
        )

    final_makespan = sum(
        _layer_makespan(loads, host_tuple) for loads in state.route_loads
    )
    plan: dict[str, object] = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "model": {
            "architecture": "Kimi-K3",
            "routed_layer_count": ROUTED_LAYER_COUNT,
            "first_routed_model_block": MODEL_BLOCK_OFFSET,
            "expert_count_per_layer": EXPERT_COUNT,
        },
        "cost_model": {
            "description": (
                "Sum across layers of max(host route count / relative expert "
                "throughput); hosts execute concurrently within each layer."
            ),
            "includes_network_latency": False,
            "includes_route_cooccurrence": False,
            "algorithm": (
                "deterministic per-layer longest-processing-time placement plus "
                "minimum marginal-makespan-per-byte capacity repair"
            ),
            "global_optimum_claimed": False,
        },
        "inputs": {
            "route_profile": route_profile_metadata
            if route_profile_metadata is not None
            else {"kind": "caller_supplied"},
            "expert_bytes": expert_byte_metadata
            if expert_byte_metadata is not None
            else {"kind": "caller_supplied"},
        },
        "hosts": [
            {
                "host_index": host_index,
                "name": host.name,
                "capacity_bytes": host.capacity_bytes,
                "used_bytes": state.used_bytes[host_index],
                "remaining_bytes": (host.capacity_bytes - state.used_bytes[host_index]),
                "relative_expert_throughput": host.relative_expert_throughput,
                "unconstrained_used_bytes": unconstrained_used_bytes[host_index],
            }
            for host_index, host in enumerate(host_tuple)
        ],
        "predicted_total_concurrent_makespan": final_makespan,
        "unconstrained_predicted_total_concurrent_makespan": (unconstrained_makespan),
        "layers": layer_payloads,
    }
    validate_plan(plan)
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def validate_plan(plan: dict[str, object]) -> None:
    """Validate partition, inverse maps, byte accounting, and host budgets."""
    hosts = _as_object_list(plan.get("hosts"))
    layers = _as_object_list(plan.get("layers"))
    if hosts is None or len(hosts) != 2:
        raise PlanError("plan must contain exactly two hosts")
    if layers is None or len(layers) != ROUTED_LAYER_COUNT:
        raise PlanError(f"plan must contain {ROUTED_LAYER_COUNT} routed layers")
    recomputed_host_bytes = [0, 0]
    for layer_index, raw_layer in enumerate(layers):
        if not isinstance(raw_layer, dict):
            raise PlanError(f"layer {layer_index} must be an object")
        layer = cast(dict[object, object], raw_layer)
        owners_raw = _as_object_list(layer.get("owner_by_global_expert"))
        byte_counts = _as_object_list(layer.get("expert_bytes_by_global_expert"))
        layer_hosts = _as_object_list(layer.get("hosts"))
        if owners_raw is None or len(owners_raw) != EXPERT_COUNT:
            raise PlanError(
                f"layer {layer_index} must own exactly {EXPERT_COUNT} experts"
            )
        owners: list[int] = []
        for owner in owners_raw:
            if (
                isinstance(owner, bool)
                or not isinstance(owner, int)
                or owner not in (0, 1)
            ):
                raise PlanError(f"layer {layer_index} contains an invalid owner")
            owners.append(owner)
        if byte_counts is None or len(byte_counts) != EXPERT_COUNT:
            raise PlanError(f"layer {layer_index} has invalid expert-byte metadata")
        if layer_hosts is None or len(layer_hosts) != 2:
            raise PlanError(f"layer {layer_index} must contain two host maps")
        seen: set[int] = set()
        for host_index, raw_host_map in enumerate(layer_hosts):
            if not isinstance(raw_host_map, dict):
                raise PlanError(
                    f"layer {layer_index} host {host_index} map must be an object"
                )
            host_map = cast(dict[object, object], raw_host_map)
            local_to_global_raw = _as_object_list(host_map.get("local_to_global"))
            global_to_local = _as_object_list(host_map.get("global_to_local"))
            if (
                local_to_global_raw is None
                or global_to_local is None
                or len(global_to_local) != EXPERT_COUNT
            ):
                raise PlanError(
                    f"layer {layer_index} host {host_index} has invalid maps"
                )
            local_to_global: list[int] = []
            for global_id in local_to_global_raw:
                if isinstance(global_id, bool) or not isinstance(global_id, int):
                    raise PlanError(
                        f"layer {layer_index} host {host_index} "
                        "local map has invalid IDs"
                    )
                local_to_global.append(global_id)
            global_to_local_ids: list[int] = []
            for local_id in global_to_local:
                if isinstance(local_id, bool) or not isinstance(local_id, int):
                    raise PlanError(
                        f"layer {layer_index} host {host_index} "
                        "global map has invalid IDs"
                    )
                global_to_local_ids.append(local_id)
            if local_to_global != sorted(local_to_global):
                raise PlanError(
                    f"layer {layer_index} host {host_index} local map is not sorted"
                )
            expected_local = 0
            host_bytes = 0
            for expert_id in range(EXPERT_COUNT):
                mapped_local = global_to_local_ids[expert_id]
                if owners[expert_id] == host_index:
                    if mapped_local != expected_local:
                        raise PlanError(
                            f"layer {layer_index} host {host_index} maps are not inverse"
                        )
                    if local_to_global[expected_local] != expert_id:
                        raise PlanError(
                            f"layer {layer_index} host {host_index} maps are not inverse"
                        )
                    expected_local += 1
                    seen.add(expert_id)
                    raw_byte_count = byte_counts[expert_id]
                    if (
                        isinstance(raw_byte_count, bool)
                        or not isinstance(raw_byte_count, int)
                        or raw_byte_count <= 0
                    ):
                        raise PlanError(
                            f"layer {layer_index} expert {expert_id} has invalid bytes"
                        )
                    host_bytes += raw_byte_count
                elif mapped_local != -1:
                    raise PlanError(
                        f"layer {layer_index} host {host_index} maps an unowned expert"
                    )
            if expected_local != len(local_to_global):
                raise PlanError(
                    f"layer {layer_index} host {host_index} local map has extra IDs"
                )
            if host_map.get("owned_expert_count") != expected_local:
                raise PlanError(
                    f"layer {layer_index} host {host_index} expert count is wrong"
                )
            if host_map.get("owned_bytes") != host_bytes:
                raise PlanError(
                    f"layer {layer_index} host {host_index} byte count is wrong"
                )
            recomputed_host_bytes[host_index] += host_bytes
        if seen != set(range(EXPERT_COUNT)):
            raise PlanError(
                f"layer {layer_index} is not an exact disjoint expert partition"
            )

    for host_index, raw_host in enumerate(hosts):
        if not isinstance(raw_host, dict):
            raise PlanError(f"host {host_index} must be an object")
        host = cast(dict[object, object], raw_host)
        used_bytes = host.get("used_bytes")
        capacity_bytes = host.get("capacity_bytes")
        if (
            isinstance(used_bytes, bool)
            or not isinstance(used_bytes, int)
            or used_bytes != recomputed_host_bytes[host_index]
        ):
            raise PlanError(f"host {host_index} byte accounting is inconsistent")
        if (
            isinstance(capacity_bytes, bool)
            or not isinstance(capacity_bytes, int)
            or recomputed_host_bytes[host_index] > capacity_bytes
        ):
            raise PlanError(f"host {host_index} exceeds its exact byte budget")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic profile-guided ownership and remapping metadata "
            "for Kimi K3's 92 x 896 routed experts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "If --route-counts is omitted, every expert receives one synthetic "
            "route. This uniform fallback is throughput-aware but not workload-"
            "profiled, and is identified as such in the output."
        ),
    )
    parser.add_argument(
        "--expert-bytes",
        type=Path,
        required=True,
        help=(
            "JSON with exact expert_bytes[92][896] or bytes_per_expert_by_layer[92]."
        ),
    )
    parser.add_argument(
        "--route-counts",
        type=Path,
        help="Optional JSON route_counts/logical_count matrix with shape [92][896].",
    )
    parser.add_argument(
        "--host",
        type=parse_host_spec,
        action="append",
        required=True,
        help=("Repeat exactly twice: NAME:CAPACITY_BYTES:RELATIVE_EXPERT_THROUGHPUT."),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = cast(_Arguments, _argument_parser().parse_args())
    hosts = args.host
    if len(hosts) != 2:
        raise PlanError("exactly two --host arguments are required")
    route_counts, route_metadata = load_route_counts(args.route_counts)
    expert_bytes, byte_metadata = load_expert_bytes(args.expert_bytes)
    output_path = args.output
    input_paths = [args.expert_bytes]
    if args.route_counts is not None:
        input_paths.append(args.route_counts)
    validate_new_output_path(output_path, input_paths)
    plan = build_plan(
        route_counts=route_counts,
        expert_bytes=expert_bytes,
        hosts=hosts,
        route_profile_metadata=route_metadata,
        expert_byte_metadata=byte_metadata,
    )
    atomic_write_new_text(
        output_path,
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"wrote {output_path}: layers={ROUTED_LAYER_COUNT} "
        f"experts_per_layer={EXPERT_COUNT} "
        f"predicted_makespan={plan['predicted_total_concurrent_makespan']:.6f}"
    )


if __name__ == "__main__":
    main()
