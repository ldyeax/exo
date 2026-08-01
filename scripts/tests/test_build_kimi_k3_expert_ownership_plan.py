from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import build_kimi_k3_expert_ownership_plan as planner


def _uniform_matrix(value: int) -> list[list[int]]:
    return [[value] * planner.EXPERT_COUNT for _ in range(planner.ROUTED_LAYER_COUNT)]


def _host(
    name: str,
    capacity_bytes: int,
    throughput: float,
) -> planner.HostSpec:
    return planner.HostSpec(
        name=name,
        capacity_bytes=capacity_bytes,
        relative_expert_throughput=throughput,
    )


def test_uniform_fallback_is_deterministic_throughput_balanced_partition() -> None:
    route_counts = _uniform_matrix(1)
    expert_bytes = _uniform_matrix(10)
    total_bytes = planner.ROUTED_LAYER_COUNT * planner.EXPERT_COUNT * 10
    hosts = (
        _host("dwagon", total_bytes, 3.0),
        _host("fwuff", total_bytes, 1.0),
    )

    first = planner.build_plan(
        route_counts=route_counts,
        expert_bytes=expert_bytes,
        hosts=hosts,
        route_profile_metadata={"kind": "uniform_fallback"},
    )
    second = planner.build_plan(
        route_counts=route_counts,
        expert_bytes=expert_bytes,
        hosts=hosts,
        route_profile_metadata={"kind": "uniform_fallback"},
    )

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    layers = first["layers"]
    assert isinstance(layers, list)
    first_layer = layers[0]
    assert isinstance(first_layer, dict)
    owners = first_layer["owner_by_global_expert"]
    assert isinstance(owners, list)
    assert set(owners) == {0, 1}
    # Throughput balancing creates an interleaved, not contiguous, partition.
    transitions = sum(
        owners[index] != owners[index - 1] for index in range(1, len(owners))
    )
    assert transitions > 100
    layer_hosts = first_layer["hosts"]
    assert isinstance(layer_hosts, list)
    fast_count = layer_hosts[0]["owned_expert_count"]
    slow_count = layer_hosts[1]["owned_expert_count"]
    assert isinstance(fast_count, int)
    assert isinstance(slow_count, int)
    assert abs(fast_count - 3 * slow_count) <= 4
    planner.validate_plan(first)


def test_hot_route_profile_beats_contiguous_half_split() -> None:
    route_counts = [
        [100] * 16 + [1] * (planner.EXPERT_COUNT - 16)
        for _ in range(planner.ROUTED_LAYER_COUNT)
    ]
    expert_bytes = _uniform_matrix(1)
    total_bytes = planner.ROUTED_LAYER_COUNT * planner.EXPERT_COUNT
    hosts = (
        _host("dwagon", total_bytes // 2, 1.0),
        _host("fwuff", total_bytes // 2, 1.0),
    )

    plan = planner.build_plan(
        route_counts=route_counts,
        expert_bytes=expert_bytes,
        hosts=hosts,
    )

    layers = plan["layers"]
    assert isinstance(layers, list)
    for raw_layer in layers:
        assert isinstance(raw_layer, dict)
        owners = raw_layer["owner_by_global_expert"]
        assert isinstance(owners, list)
        assert sum(owners[expert_id] == 0 for expert_id in range(16)) == 8
        assert sum(owners[expert_id] == 1 for expert_id in range(16)) == 8

    contiguous_layer_makespan = sum(route_counts[0][:448])
    contiguous_total = planner.ROUTED_LAYER_COUNT * contiguous_layer_makespan
    predicted = plan["predicted_total_concurrent_makespan"]
    assert isinstance(predicted, float)
    assert predicted < contiguous_total * 0.7


def test_capacity_repair_enforces_exact_varying_layer_byte_budgets() -> None:
    route_counts = _uniform_matrix(1)
    bytes_by_layer = [7 + layer_index % 3 for layer_index in range(92)]
    expert_bytes = [
        [byte_count] * planner.EXPERT_COUNT for byte_count in bytes_by_layer
    ]
    total_bytes = sum(bytes_by_layer) * planner.EXPERT_COUNT
    hosts = (
        _host("dwagon", total_bytes * 60 // 100 + 128, 1.0),
        _host("fwuff", total_bytes * 45 // 100 + 128, 1.0),
    )

    plan = planner.build_plan(
        route_counts=route_counts,
        expert_bytes=expert_bytes,
        hosts=hosts,
    )

    raw_hosts = plan["hosts"]
    assert isinstance(raw_hosts, list)
    assert raw_hosts[1]["unconstrained_used_bytes"] > raw_hosts[1]["capacity_bytes"]
    assert raw_hosts[0]["used_bytes"] <= raw_hosts[0]["capacity_bytes"]
    assert raw_hosts[1]["used_bytes"] <= raw_hosts[1]["capacity_bytes"]
    assert sum(raw_host["used_bytes"] for raw_host in raw_hosts) == total_bytes
    planner.validate_plan(plan)


def test_loaders_accept_compact_bytes_and_document_uniform_fallback(
    tmp_path: Path,
) -> None:
    byte_path = tmp_path / "bytes.json"
    byte_path.write_text(
        json.dumps(
            {"bytes_per_expert_by_layer": [9_547_776] * planner.ROUTED_LAYER_COUNT}
        ),
        encoding="utf-8",
    )

    expert_bytes, byte_metadata = planner.load_expert_bytes(byte_path)
    route_counts, route_metadata = planner.load_route_counts(None)

    assert len(expert_bytes) == planner.ROUTED_LAYER_COUNT
    assert len(expert_bytes[0]) == planner.EXPERT_COUNT
    assert expert_bytes[0][895] == 9_547_776
    assert byte_metadata["encoding"] == "uniform_within_layer"
    assert route_counts[91][895] == 1
    assert route_metadata["kind"] == "uniform_fallback"
    assert "synthetic route" in str(route_metadata["description"])


def test_checked_in_q2_inventory_matches_the_pinned_gguf() -> None:
    inventory_path = (
        Path(__file__).parents[1] / "data" / "kimi_k3_ud_q2_k_xl_expert_bytes.json"
    )

    expert_bytes, metadata = planner.load_expert_bytes(inventory_path)

    assert metadata["encoding"] == "uniform_within_layer"
    assert sum(sum(layer) for layer in expert_bytes) == 799_065_243_648
    assert expert_bytes[0][0] == 9_547_776
    assert expert_bytes[11][0] == 10_579_968
    assert expert_bytes[90][0] == 12_644_352
    assert expert_bytes[91][0] == 9_547_776


def test_cli_emits_self_describing_loader_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bytes_per_expert = 9_547_776
    total_bytes = planner.ROUTED_LAYER_COUNT * planner.EXPERT_COUNT * bytes_per_expert
    byte_path = tmp_path / "bytes.json"
    output_path = tmp_path / "plan.json"
    byte_path.write_text(
        json.dumps(
            {
                "bytes_per_expert_by_layer": [bytes_per_expert]
                * planner.ROUTED_LAYER_COUNT
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_kimi_k3_expert_ownership_plan.py",
            "--expert-bytes",
            str(byte_path),
            "--host",
            f"dwagon:{total_bytes}:2",
            "--host",
            f"fwuff:{total_bytes}:1",
            "--output",
            str(output_path),
        ],
    )

    planner.main()

    emitted = json.loads(output_path.read_text(encoding="utf-8"))
    assert emitted["kind"] == planner.PLAN_KIND
    assert emitted["inputs"]["route_profile"]["kind"] == "uniform_fallback"
    assert emitted["inputs"]["expert_bytes"]["sha256"] == planner._sha256(byte_path)
    assert len(emitted["layers"][0]["hosts"][0]["global_to_local"]) == 896
    expected_hash = emitted.pop("plan_sha256")
    assert expected_hash == planner._canonical_sha256(emitted)


def test_output_path_refuses_input_alias_existing_file_and_symlink(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "bytes.json"
    input_path.write_text("input\n", encoding="utf-8")

    with pytest.raises(planner.PlanError, match="aliases input"):
        planner.validate_new_output_path(input_path, [input_path])

    existing_output = tmp_path / "existing.json"
    existing_output.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(planner.PlanError, match="refusing to overwrite"):
        planner.validate_new_output_path(existing_output, [input_path])
    assert existing_output.read_text(encoding="utf-8") == "preserve\n"

    symlink_output = tmp_path / "output-link.json"
    symlink_output.symlink_to(tmp_path / "not-created.json")
    with pytest.raises(planner.PlanError, match="symlink output"):
        planner.validate_new_output_path(symlink_output, [input_path])


def test_atomic_output_publication_never_replaces_existing_file(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "plan.json"

    planner.atomic_write_new_text(output_path, "first\n")

    assert output_path.read_text(encoding="utf-8") == "first\n"
    with pytest.raises(planner.PlanError, match="created concurrently"):
        planner.atomic_write_new_text(output_path, "second\n")
    assert output_path.read_text(encoding="utf-8") == "first\n"


def test_rejects_bad_shapes_counts_and_insufficient_capacity() -> None:
    expert_bytes = _uniform_matrix(10)
    route_counts = _uniform_matrix(1)
    total_bytes = planner.ROUTED_LAYER_COUNT * planner.EXPERT_COUNT * 10

    bad_counts = route_counts[:-1]
    with pytest.raises(planner.PlanError, match="92 rows"):
        planner.build_plan(
            route_counts=bad_counts,
            expert_bytes=expert_bytes,
            hosts=(
                _host("dwagon", total_bytes, 1.0),
                _host("fwuff", total_bytes, 1.0),
            ),
        )

    route_counts[0][0] = -1
    with pytest.raises(planner.PlanError, match="must not be negative"):
        planner.build_plan(
            route_counts=route_counts,
            expert_bytes=expert_bytes,
            hosts=(
                _host("dwagon", total_bytes, 1.0),
                _host("fwuff", total_bytes, 1.0),
            ),
        )

    with pytest.raises(planner.PlanError, match="combined host capacity"):
        planner.build_plan(
            route_counts=_uniform_matrix(1),
            expert_bytes=expert_bytes,
            hosts=(
                _host("dwagon", total_bytes // 3, 1.0),
                _host("fwuff", total_bytes // 3, 1.0),
            ),
        )
