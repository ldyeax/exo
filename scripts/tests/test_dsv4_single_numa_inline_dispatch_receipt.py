import hashlib
import json
import math
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_PATH = (
    REPOSITORY_ROOT
    / "scripts/data/dsv4_flash_single_numa_inline_dispatch_layer20_2026-08-04.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_inline_dispatch_receipt_is_internally_consistent() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    assert receipt["format"] == "dsv4_single_numa_inline_dispatch_benchmark_v1"
    assert receipt["schema_version"] == 1
    assert receipt["decision"]["winner"] == "inline56_spin1000"

    weights = receipt["production_weighting"]["weights_m1_to_m6"]
    assert weights == [395, 349, 293, 363, 0, 0]
    assert sum(weights) == receipt["production_weighting"]["cycle_count"] == 1400

    results = receipt["results"]
    assert set(results) == {
        "same_binary_default56_spin1000",
        "inline56_spin1000",
        "inline56_spin0",
        "inline72_spin1000",
        "inline72_spin0",
    }
    control = results["same_binary_default56_spin1000"]
    control_medians = control["median_us_m1_to_m6"]
    control_weighted = sum(
        latency * weight
        for latency, weight in zip(control_medians, weights, strict=True)
    ) / sum(weights)
    control_sum = sum(control_medians)

    for result in results.values():
        medians = result["median_us_m1_to_m6"]
        assert len(medians) == 6
        assert all(type(latency) in (int, float) and latency > 0 for latency in medians)
        weighted = sum(
            latency * weight for latency, weight in zip(medians, weights, strict=True)
        ) / sum(weights)
        assert math.isclose(
            result["production_weighted_mean_us"], weighted, rel_tol=1e-14
        )
        assert math.isclose(
            result["production_weighted_speedup_over_control"],
            control_weighted / weighted,
            rel_tol=1e-14,
        )
        assert math.isclose(
            result["equal_m1_to_m6_sum_speedup_over_control"],
            control_sum / sum(medians),
            rel_tol=1e-14,
        )

    winner = results[receipt["decision"]["winner"]]
    assert winner["production_weighted_mean_us"] == min(
        result["production_weighted_mean_us"] for result in results.values()
    )
    assert math.isclose(
        receipt["decision"][
            "production_weighted_speedup_over_same_binary_default_dispatch"
        ],
        control_weighted / winner["production_weighted_mean_us"],
        rel_tol=1e-14,
    )


def test_inline_dispatch_receipt_binds_command_source_and_quality() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    command = receipt["command_contract"]
    arms = command["arms"]

    assert command["common_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert command["common_environment"]["KT_TASK_QUEUE_PIN_FIRST_CORE"] is None
    assert (
        arms["same_binary_default56_spin1000"]["environment"][
            "KT_SINGLE_NUMA_INLINE_DISPATCH"
        ]
        is None
    )
    assert arms["inline56_spin1000"] == {
        "physical_cpu_binding": "0-55",
        "threads": 56,
        "environment": {
            "KT_SINGLE_NUMA_INLINE_DISPATCH": "1",
            "KT_WORKER_SPIN_US": "1000",
        },
    }
    assert arms["inline72_spin1000"]["physical_cpu_binding"] == "0-55,112-127"
    assert arms["inline72_spin1000"]["threads"] == 72

    benchmark_path = REPOSITORY_ROOT / command["benchmark_script"]
    assert _sha256(benchmark_path) == receipt["artifacts"]["benchmark_script_sha256"]
    for digest in (
        receipt["artifacts"]["extension"]["sha256"],
        receipt["artifacts"]["overlay_json_sha256"],
        receipt["artifacts"]["g14_baseline_receipt_sha256"],
        receipt["checkpoint"]["index_sha256"],
        receipt["checkpoint"]["selected_native_tensor_sha256"],
    ):
        assert len(digest) == 64
        int(digest, 16)

    quality = receipt["quality"]
    assert quality["all_arms_output_identical"] is True
    assert set(quality["by_m"]) == {"1", "2", "3", "4", "5", "6"}
    for measurement in quality["by_m"].values():
        assert len(measurement["output_sha256"]) == 64
        int(measurement["output_sha256"], 16)
        assert measurement["cosine_similarity"] > 0.99999
        assert measurement["relative_l1"] < 0.004
