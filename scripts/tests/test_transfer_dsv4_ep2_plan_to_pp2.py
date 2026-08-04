from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import transfer_dsv4_ep2_plan_to_pp2 as transfer


def make_source_plan(
    path: Path,
    *,
    source_gpu_experts_per_rank: int = 13,
) -> dict[str, object]:
    target_gpu_experts_per_layer = 2 * source_gpu_experts_per_rank
    source_cpu_experts_per_rank = (256 - target_gpu_experts_per_layer) // 2
    gpu_masks = torch.zeros((2, 43, 256), dtype=torch.bool)
    cpu_shards: tuple[list[list[int]], list[list[int]]] = ([], [])
    for layer in range(43):
        permutation = torch.roll(torch.arange(256, dtype=torch.int64), shifts=layer)
        gpu_masks[0, layer, permutation[:source_gpu_experts_per_rank]] = True
        gpu_masks[
            1,
            layer,
            permutation[source_gpu_experts_per_rank:target_gpu_experts_per_layer],
        ] = True
        cpu_split = target_gpu_experts_per_layer + source_cpu_experts_per_rank
        cpu_shards[0].append(
            permutation[target_gpu_experts_per_layer:cpu_split].tolist()
        )
        cpu_shards[1].append(permutation[cpu_split:].tolist())
    cpu_tensors = tuple(torch.tensor(shard, dtype=torch.int64) for shard in cpu_shards)
    semantics_sha256 = transfer._placement_semantics_sha256(gpu_masks, cpu_tensors)
    source: dict[str, object] = {
        "format": transfer.PLAN_FORMAT,
        "gpu_experts_mask_by_rank": gpu_masks,
        "cpu_expert_ids_by_rank": list(cpu_tensors),
        "gpu_rank_counts": torch.tensor(
            [source_gpu_experts_per_rank, source_gpu_experts_per_rank],
            dtype=torch.int64,
        ),
        "cpu_rank_counts": torch.tensor(
            [source_cpu_experts_per_rank, source_cpu_experts_per_rank],
            dtype=torch.int64,
        ),
        "global_num_experts": torch.tensor(256, dtype=torch.int64),
        "gpu_union_expert_count": torch.tensor(
            target_gpu_experts_per_layer, dtype=torch.int64
        ),
        "placement_semantics_sha256": semantics_sha256,
        "source_profiles": ["fivefold-exact", "fivefold-near"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, path)
    return source


def make_variable_source_plan(path: Path) -> dict[str, object]:
    gpu_masks = torch.zeros((2, 43, 256), dtype=torch.bool)
    cpu_ids: tuple[list[list[int]], list[list[int]]] = ([], [])
    gpu_counts = torch.zeros((2, 43), dtype=torch.int64)
    cpu_counts = torch.zeros((2, 43), dtype=torch.int64)
    for layer in range(43):
        permutation = torch.roll(torch.arange(256, dtype=torch.int64), shifts=layer)
        rank0_gpu_count = 14 + 2 * (layer % 2)
        rank1_gpu_count = 14 + 2 * ((layer // 2) % 2)
        union_count = rank0_gpu_count + rank1_gpu_count
        gpu_masks[0, layer, permutation[:rank0_gpu_count]] = True
        gpu_masks[
            1,
            layer,
            permutation[rank0_gpu_count:union_count],
        ] = True
        remaining = permutation[union_count:]
        split = (remaining.numel() + (layer % 2)) // 2
        cpu_ids[0].append(remaining[:split].tolist())
        cpu_ids[1].append(remaining[split:].tolist())
        gpu_counts[:, layer] = torch.tensor(
            [rank0_gpu_count, rank1_gpu_count],
            dtype=torch.int64,
        )
        cpu_counts[:, layer] = torch.tensor(
            [split, remaining.numel() - split],
            dtype=torch.int64,
        )

    maximum_cpu_count = int(cpu_counts.max())
    cpu_padded = torch.full((2, 43, maximum_cpu_count), -1, dtype=torch.int64)
    for rank in range(2):
        for layer in range(43):
            ids = cpu_ids[rank][layer]
            cpu_padded[rank, layer, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.int64,
            )
    semantics_sha256 = transfer._variable_placement_semantics_sha256(
        gpu_masks,
        cpu_padded,
        cpu_counts,
    )
    source: dict[str, object] = {
        "format": transfer.VARIABLE_PLAN_FORMAT,
        "gpu_experts_mask_by_rank": gpu_masks,
        "cpu_expert_ids_padded_by_rank": cpu_padded,
        "cpu_rank_counts_by_layer": cpu_counts,
        "gpu_rank_counts_by_layer": gpu_counts,
        "global_num_experts": torch.tensor(256, dtype=torch.int64),
        "min_gpu_experts_per_rank_per_layer": torch.tensor(
            int(gpu_counts.min()),
            dtype=torch.int64,
        ),
        "max_gpu_experts_per_rank_per_layer": torch.tensor(
            int(gpu_counts.max()),
            dtype=torch.int64,
        ),
        "rank_symmetric_widths": False,
        "placement_semantics_sha256": semantics_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(source, path)
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def make_ep_confirmation(
    path: Path,
    *,
    source_plan: Path,
    placement_semantics_sha256: str,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "format": transfer.EP_CONFIRMATION_FORMAT,
        "stage": "confirm",
        "qualified": True,
        "shutdown_method": "sigterm",
        "residual_compute_pids": [],
        "coherency": {
            "coherent": True,
            "deterministic_final_content": True,
            "forced_tool_call": {"accepted": True},
            "semantic_runs": [{"accepted": True}, {"accepted": True}],
        },
        "environment": {
            "DSV4_TENSOR_PARALLEL_SIZE": "2",
            "DSV4_PIPELINE_PARALLEL_SIZE": "1",
            "DSV4_EXPERT_PARALLEL_SIZE": "2",
            "DSV4_CONTEXT_LENGTH": "524288",
            "DSV4_MAX_TOTAL_TOKENS": "524288",
            "DSV4_KV_CACHE_DTYPE": "fp8_e4m3",
            "SGLANG_DSV4_OSCAR_INT2_KV_STORAGE": "1",
            "SGLANG_DSV4_INT4_KV_STORAGE": "0",
            "SGLANG_DSV4_INT4_C4_INDEXER_STORAGE": "0",
            "SGLANG_DSV4_SM86_C128_BF16_STORAGE": "0",
            "DSV4_CPUINFER_THREADS": "56",
            "KT_WORKER_SPIN_US": "1000",
            "KT_TASK_QUEUE_PIN_FIRST_CORE": "1",
            "KT_SINGLE_NUMA_INLINE_DISPATCH": "1",
            "KT_MXFP4_AVX_SCALE_FOLD_MODE": "lut-v1",
            "DSV4_STAGE_KT_CPU_OPTIMIZED_OVERLAY": "1",
            "DSV4_KT_CPU_OPTIMIZED_CANDIDATE": str(
                transfer.QUALIFIED_NATIVE_ARTIFACT
            ),
        },
        "kt_single_numa_inline_dispatch_server_proof": {
            "kt_single_numa_inline_dispatch_all_workers_active": True,
            "kt_single_numa_inline_dispatch_rank_coverage_valid": True,
        },
        "kt_mxfp4_avx_scale_fold_server_proof": {
            "kt_mxfp4_avx_scale_fold_all_workers_active": True,
            "kt_mxfp4_avx_scale_fold_rank_coverage_valid": True,
            "kt_mxfp4_avx_scale_fold_expected_n_block": 128,
            "kt_mxfp4_avx_scale_fold_requested_mode": "lut-v1",
            "workers": [
                {
                    "telemetry": {
                        "n_block": 128,
                        "lut_hash": transfer.QUALIFIED_LUT_HASH,
                    }
                },
                {
                    "telemetry": {
                        "n_block": 128,
                        "lut_hash": transfer.QUALIFIED_LUT_HASH,
                    }
                },
            ],
        },
        "oscar_contract": {
            "expected_server_info": {
                "dsv4_oscar_int2_kv_storage": True,
                "dsv4_kv_storage_mode": (
                    "oscar_int2_asymmetric+protected_swa_bfloat16"
                ),
                "dsv4_c4_kv_bytes_per_token": 272,
                "dsv4_c128_kv_bytes_per_token": 272,
                "dsv4_c4_indexer_bytes_per_token": 40,
                "dsv4_int4_kv_storage": False,
                "dsv4_int4_c4_indexer_storage": False,
                "dsv4_sm86_c128_bf16_storage": False,
            }
        },
        "plan": {
            "path": str(source_plan),
            "sha256": sha256_file(source_plan),
            "placement_semantics_sha256": placement_semantics_sha256,
        },
    }
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def make_direct_coherency_receipt(path: Path) -> dict[str, object]:
    expected_hash = "b" * 64
    receipt: dict[str, object] = {
        "schema_version": 2,
        "cache_policy": "flush_before_every_request",
        "coherent": True,
        "deterministic_final_content": True,
        "forced_tool_call": {
            "accepted": True,
            "http_status": 200,
            "finish_reason": "tool_calls",
            "saw_done": True,
            "issue_codes": [],
        },
        "semantic_runs": [
            {
                "accepted": True,
                "http_status": 200,
                "finish_reason": "stop",
                "saw_done": True,
                "issue_codes": [],
                "content_sha256": expected_hash,
                "expected_content_sha256": expected_hash,
            }
            for _ in range(3)
        ],
    }
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def make_direct_hotspot_confirmation(
    path: Path,
    *,
    source_plan: Path,
) -> dict[str, object]:
    phases: dict[str, object] = {}
    for phase_name in sorted(transfer.EP_HOTSPOT_PHASES):
        fresh = phase_name in transfer.EP_FRESH_TTFT_PHASES
        phases[phase_name] = {
            "radix_flush_before": fresh,
            "benchmark": {
                "accepted": True,
                "performance_claim_eligible": True,
                "http_status": 200,
                "finish_reason": "stop",
                "saw_done": True,
                "input_tokens": 2_694,
                "server_cached_tokens": 0 if fresh else 2_560,
                "time_to_first_token_seconds": 5.5 if fresh else 0.7,
                "semantic_validation": {
                    "enforced": True,
                    "passed": True,
                    "issue_codes": [],
                },
            },
        }
    counter_deltas: dict[str, int] = {}
    for link in range(4):
        counter_deltas[f"gpu0.link{link}.tx_kib"] = 100 + link
        counter_deltas[f"gpu1.link{link}.rx_kib"] = 100 + link
        counter_deltas[f"gpu1.link{link}.tx_kib"] = 200 + link
        counter_deltas[f"gpu0.link{link}.rx_kib"] = 200 + link
    receipt: dict[str, object] = {
        "receipt_version": transfer.EP_HOTSPOT_RECEIPT_VERSION,
        "accepted": True,
        "performance_claim_eligible": False,
        "measurement_mode": "trace",
        "phases": phases,
        "nvlink_traffic": {"counter_deltas": counter_deltas},
        "expert_plan_provenance": {
            "binding": "launcher-hash-and-kt-loader-validated",
            "expected_plan_path": str(source_plan),
            "expected_plan_sha256": sha256_file(source_plan),
        },
        "server_contract": {
            "tp_size": 2,
            "pp_size": 1,
            "ep_size": 2,
            "context_length": 524_288,
            "max_total_tokens": 524_288,
            "kv_cache_dtype": "fp8_e4m3",
            "kt_cpuinfer": 56,
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
            "disable_prefill_cuda_graph": False,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "breakable",
            "enable_p2p_check": True,
            "pre_warm_nccl": True,
            "dsv4_oscar_int2_split_history": True,
            "dsv4_oscar_int2_split_history_execution": (
                transfer.OSCAR_SPLIT_HISTORY_EXECUTION
            ),
            "dsv4_oscar_int2_split_history_split_map": (
                transfer.OSCAR_SPLIT_HISTORY_SPLIT_MAP
            ),
            "dsv4_oscar_int2_split_history_workspace_bytes": (
                transfer.OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
            ),
            "dsv4_oscar_int2_split_history_max_partial_rows": 32,
            "dsv4_oscar_int2_split_history_sink_owner": "stage2-exactly-once",
            "dsv4_oscar_int2_split_history_prefill_enabled": False,
            "dsv4_oscar_int2_split_history_fixed_address": True,
            "dsv4_oscar_int2_split_history_workers": {
                "worker_count": 2,
                "tp_pp_gpu_ranks": [[0, 0, 0], [1, 0, 1]],
                "worker_pids": [101, 202],
                "workspace_addresses": [303, 303],
                "workspace_bytes_per_worker": (
                    transfer.OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
                ),
                "fixed_address": True,
            },
        },
    }
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def test_source_plan_mutation_during_load_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "winner" / "ep2-g14.pt"
    make_source_plan(source_path, source_gpu_experts_per_rank=14)
    original_loader = transfer.load_source_placement

    def mutating_loader(*args: object, **kwargs: object):
        placement = original_loader(*args, **kwargs)
        source_path.write_bytes(source_path.read_bytes() + b"mutation")
        return placement

    monkeypatch.setattr(transfer, "load_source_placement", mutating_loader)

    with pytest.raises(transfer.PlanTransferError, match="changed while loading"):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
            expected_target_gpu_experts_per_layer=28,
        )


def test_stages_exact_gpu_union_and_cpu_complement(tmp_path: Path) -> None:
    source_path = tmp_path / "winner" / "ep2-g13.pt"
    source = make_source_plan(source_path)
    cache_root = tmp_path / "cache"

    result = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
    )

    assert result.parent == cache_root
    staged = torch.load(result, map_location="cpu", weights_only=True)
    assert isinstance(staged, dict)
    gpu_masks = staged["gpu_experts_mask_by_rank"]
    cpu_shards = staged["cpu_expert_ids_by_rank"]
    assert isinstance(gpu_masks, torch.Tensor)
    assert isinstance(cpu_shards, list) and len(cpu_shards) == 1
    assert isinstance(cpu_shards[0], torch.Tensor)
    assert tuple(gpu_masks.shape) == (1, 43, 256)
    assert tuple(cpu_shards[0].shape) == (43, 230)
    source_masks = source["gpu_experts_mask_by_rank"]
    assert isinstance(source_masks, torch.Tensor)
    assert torch.equal(gpu_masks[0], source_masks.any(dim=0))
    assert bool(torch.all(gpu_masks.sum(dim=2) == 26))
    for layer in range(43):
        assert torch.equal(
            cpu_shards[0][layer],
            torch.where(~gpu_masks[0, layer])[0],
        )

    transfer_receipt = staged["transfer_receipt"]
    assert isinstance(transfer_receipt, dict)
    assert transfer_receipt["format"] == transfer.TRANSFER_FORMAT
    assert transfer_receipt["source_ep2_plan"] == str(source_path)
    assert transfer_receipt["source_ep2_plan_sha256"] == sha256_file(source_path)
    assert (
        transfer_receipt["source_ep2_placement_semantics_sha256"]
        == source["placement_semantics_sha256"]
    )
    assert transfer_receipt["status"] == "complete"
    assert staged["gpu_rank_counts"].tolist() == [26]
    assert staged["cpu_rank_counts"].tolist() == [230]

    repeated = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
    )
    assert repeated == result
    assert not list(cache_root.glob("*.partial-*"))


def test_stages_g14_winner_as_g28_pp2_union(tmp_path: Path) -> None:
    source_path = tmp_path / "winner" / "ep2-g14.pt"
    source = make_source_plan(source_path, source_gpu_experts_per_rank=14)

    result = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=tmp_path / "cache",
        expected_target_gpu_experts_per_layer=28,
    )

    assert result.name.startswith("pp2-ep1-g28-")
    staged = torch.load(result, map_location="cpu", weights_only=True)
    assert isinstance(staged, dict)
    assert staged["gpu_rank_counts"].tolist() == [28]
    assert staged["cpu_rank_counts"].tolist() == [228]
    gpu_masks = staged["gpu_experts_mask_by_rank"]
    source_masks = source["gpu_experts_mask_by_rank"]
    assert isinstance(gpu_masks, torch.Tensor)
    assert isinstance(source_masks, torch.Tensor)
    assert torch.equal(gpu_masks[0], source_masks.any(dim=0))


def test_stages_variable_v2_union_as_target_only_ep1_v2(tmp_path: Path) -> None:
    source_path = tmp_path / "winner" / "ep2-variable.pt"
    source = make_variable_source_plan(source_path)
    cache_root = tmp_path / "cache"

    result = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
        expected_target_gpu_experts_per_layer=32,
    )

    assert result.name.startswith("pp2-ep1-g28to32-")
    staged = torch.load(result, map_location="cpu", weights_only=True)
    assert isinstance(staged, dict)
    assert staged["format"] == transfer.VARIABLE_PLAN_FORMAT
    assert "cpu_expert_ids_by_rank" not in staged
    gpu_masks = staged["gpu_experts_mask_by_rank"]
    gpu_counts = staged["gpu_rank_counts_by_layer"]
    cpu_padded = staged["cpu_expert_ids_padded_by_rank"]
    cpu_counts = staged["cpu_rank_counts_by_layer"]
    source_masks = source["gpu_experts_mask_by_rank"]
    source_gpu_counts = source["gpu_rank_counts_by_layer"]
    assert isinstance(gpu_masks, torch.Tensor)
    assert isinstance(gpu_counts, torch.Tensor)
    assert isinstance(cpu_padded, torch.Tensor)
    assert isinstance(cpu_counts, torch.Tensor)
    assert isinstance(source_masks, torch.Tensor)
    assert isinstance(source_gpu_counts, torch.Tensor)
    assert tuple(gpu_masks.shape) == (1, 43, 256)
    assert tuple(gpu_counts.shape) == (1, 43)
    assert tuple(cpu_counts.shape) == (1, 43)
    assert torch.equal(gpu_masks[0], source_masks.any(dim=0))
    assert torch.equal(gpu_counts[0], source_gpu_counts.sum(dim=0))
    assert int(gpu_counts.min()) == 28
    assert int(gpu_counts.max()) == 32
    assert torch.equal(cpu_counts, 256 - gpu_counts)
    assert staged["min_gpu_experts_per_rank_per_layer"].item() == 28
    assert staged["max_gpu_experts_per_rank_per_layer"].item() == 32
    for layer in range(43):
        count = int(cpu_counts[0, layer])
        assert torch.equal(
            cpu_padded[0, layer, :count],
            torch.where(~gpu_masks[0, layer])[0],
        )
        assert bool(torch.all(cpu_padded[0, layer, count:] == -1))

    receipt = staged["transfer_receipt"]
    assert isinstance(receipt, dict)
    assert receipt["source_plan_format"] == transfer.VARIABLE_PLAN_FORMAT
    assert receipt["target_model_scope"] == "target_only"
    assert receipt["target_tensor_parallel_size"] == 1
    assert receipt["target_pipeline_parallel_size"] == 2
    assert receipt["target_expert_parallel_size"] == 1
    assert receipt["target_gpu_experts_per_layer"] is None
    assert receipt["target_min_gpu_experts_per_layer"] == 28
    assert receipt["target_max_gpu_experts_per_layer"] == 32
    assert receipt["target_gpu_rank_counts_by_layer"] == gpu_counts.tolist()
    assert receipt["target_cpu_rank_counts_by_layer"] == cpu_counts.tolist()

    repeated = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
        expected_target_gpu_experts_per_layer=32,
    )
    assert repeated == result
    assert not list(cache_root.glob("*.partial-*"))


def test_rejects_variable_width_ceiling_that_disagrees_with_launcher(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-variable.pt"
    make_variable_source_plan(source_path)

    with pytest.raises(transfer.PlanTransferError, match="expected 31, got 32"):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
            expected_target_gpu_experts_per_layer=31,
        )


def test_rejects_winner_width_that_disagrees_with_launcher_budget(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-g14.pt"
    make_source_plan(source_path, source_gpu_experts_per_rank=14)

    with pytest.raises(transfer.PlanTransferError, match="expected 26, got 28"):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
            expected_target_gpu_experts_per_layer=26,
        )


def test_rejects_source_without_uniform_rank_width_before_cache_creation(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-not-g13.pt"
    source = make_source_plan(source_path)
    gpu_masks = source["gpu_experts_mask_by_rank"]
    assert isinstance(gpu_masks, torch.Tensor)
    gpu_masks[0, 0, torch.where(~gpu_masks[0, 0])[0][0]] = True
    source["placement_semantics_sha256"] = transfer._placement_semantics_sha256(
        gpu_masks,
        source["cpu_expert_ids_by_rank"],
    )
    torch.save(source, source_path)
    cache_root = tmp_path / "cache"

    with pytest.raises(transfer.PlanTransferError, match="uniform GPU expert count"):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=cache_root,
        )

    assert not cache_root.exists()


def test_rejects_declared_source_semantics_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    source = make_source_plan(source_path)
    source["placement_semantics_sha256"] = "0" * 64
    torch.save(source, source_path)

    with pytest.raises(
        transfer.PlanTransferError,
        match="placement_semantics_sha256 does not match",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
        )


def test_rejects_variable_gpu_count_metadata_mismatch(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-variable.pt"
    source = make_variable_source_plan(source_path)
    gpu_counts = source["gpu_rank_counts_by_layer"]
    assert isinstance(gpu_counts, torch.Tensor)
    gpu_counts[0, 0] += 1
    torch.save(source, source_path)

    with pytest.raises(
        transfer.PlanTransferError,
        match="GPU counts by layer do not match",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
        )


def test_rejects_variable_cpu_padding_that_is_not_sentinel(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-variable.pt"
    source = make_variable_source_plan(source_path)
    gpu_masks = source["gpu_experts_mask_by_rank"]
    cpu_padded = source["cpu_expert_ids_padded_by_rank"]
    cpu_counts = source["cpu_rank_counts_by_layer"]
    assert isinstance(gpu_masks, torch.Tensor)
    assert isinstance(cpu_padded, torch.Tensor)
    assert isinstance(cpu_counts, torch.Tensor)
    rank = 0
    layer = 1
    count = int(cpu_counts[rank, layer])
    assert count < cpu_padded.shape[2]
    cpu_padded[rank, layer, count] = 0
    source["placement_semantics_sha256"] = (
        transfer._variable_placement_semantics_sha256(
            gpu_masks,
            cpu_padded,
            cpu_counts,
        )
    )
    torch.save(source, source_path)

    with pytest.raises(
        transfer.PlanTransferError,
        match="padding must be -1",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
        )


def test_rejects_scalar_v1_rank_count_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    source = make_source_plan(source_path)
    source["gpu_rank_counts"] = torch.tensor(13, dtype=torch.int64)
    torch.save(source, source_path)

    with pytest.raises(
        transfer.PlanTransferError,
        match=r"gpu_rank_counts must have shape \(2,\)",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=tmp_path / "cache",
        )


def test_existing_transferred_plan_tamper_fails_closed(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    make_source_plan(source_path)
    cache_root = tmp_path / "cache"
    result = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
    )
    staged = torch.load(result, map_location="cpu", weights_only=True)
    assert isinstance(staged, dict)
    staged["gpu_selection_strategy"] = "tampered"
    torch.save(staged, result)

    with pytest.raises(
        transfer.PlanTransferError,
        match="staged PP2 plan verification failed",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=cache_root,
        )


def test_existing_transferred_plan_tensor_dtype_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    source = make_source_plan(source_path)
    cache_root = tmp_path / "cache"
    confirmation = tmp_path / "confirm.json"
    semantics = source["placement_semantics_sha256"]
    assert isinstance(semantics, str)
    make_ep_confirmation(
        confirmation,
        source_plan=source_path,
        placement_semantics_sha256=semantics,
    )
    result = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        cache_root=cache_root,
    )
    staged = torch.load(result, map_location="cpu", weights_only=True)
    assert isinstance(staged, dict)
    gpu_rank_counts = staged["gpu_rank_counts"]
    assert isinstance(gpu_rank_counts, torch.Tensor)
    staged["gpu_rank_counts"] = gpu_rank_counts.to(torch.int32)
    torch.save(staged, result)

    with pytest.raises(
        transfer.PlanTransferError,
        match="staged PP2 plan verification failed",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            cache_root=cache_root,
        )


def test_transfer_binds_final_ep_receipt_and_confirmed_native_tuple(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-g14.pt"
    make_source_plan(source_path, source_gpu_experts_per_rank=14)
    confirmation = tmp_path / "split-history-confirm-hotspot.json"
    coherency = tmp_path / "split-history-confirm-coherency.json"
    make_direct_hotspot_confirmation(confirmation, source_plan=source_path)
    make_direct_coherency_receipt(coherency)

    output = transfer.stage_transferred_plan(
        source_ep2_plan=source_path,
        source_ep_confirmation_receipt=confirmation,
        source_ep_coherency_receipt=coherency,
        cache_root=tmp_path / "cache",
        expected_target_gpu_experts_per_layer=28,
    )

    staged = torch.load(output, map_location="cpu", weights_only=True)
    receipt = staged["transfer_receipt"]
    assert receipt["source_ep_confirmation_receipt"] == str(confirmation)
    assert receipt["source_ep_confirmation_receipt_sha256"] == sha256_file(
        confirmation
    )
    assert receipt["source_ep_coherency_receipt"] == str(coherency)
    assert receipt["source_ep_coherency_receipt_sha256"] == sha256_file(coherency)
    assert receipt["qualified_native_artifact"] == str(
        transfer.QUALIFIED_NATIVE_ARTIFACT
    )
    assert (
        receipt["qualified_native_artifact_sha256"]
        == transfer.QUALIFIED_NATIVE_ARTIFACT_SHA256
    )
    assert receipt["qualified_scale_fold_n_block"] == 128
    assert receipt["qualified_scale_fold_lut_hash"] == "06d1a83dbf20f545"
    assert receipt["physical_kv_cache_storage"] == "oscar-int2-asymmetric"
    assert receipt["oscar_split_history"] is True
    assert (
        receipt["oscar_split_history_execution"]
        == transfer.OSCAR_SPLIT_HISTORY_EXECUTION
    )
    assert receipt["oscar_split_history_worker_identities_pp"] == [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1],
    ]
    assert receipt["context_length"] == 524_288
    assert receipt["cuda_graph_decode"] == "full"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("workspace", "both live split workspaces"),
        ("nvlink", "all 16 NVLink counters"),
        ("ttft", "fresh TTFT <= 7s"),
        ("coherency", "deterministic text/tool-call proof"),
    ),
)
def test_direct_final_ep_proofs_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    source_path = tmp_path / "ep2-g14.pt"
    make_source_plan(source_path, source_gpu_experts_per_rank=14)
    confirmation = tmp_path / "split-history-confirm-hotspot.json"
    coherency = tmp_path / "split-history-confirm-coherency.json"
    hotspot = make_direct_hotspot_confirmation(
        confirmation, source_plan=source_path
    )
    coherency_document = make_direct_coherency_receipt(coherency)
    if mutation == "workspace":
        contract = hotspot["server_contract"]
        assert isinstance(contract, dict)
        workers = contract["dsv4_oscar_int2_split_history_workers"]
        assert isinstance(workers, dict)
        workers["worker_pids"] = [101, 101]
        confirmation.write_text(
            json.dumps(hotspot, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "nvlink":
        traffic = hotspot["nvlink_traffic"]
        assert isinstance(traffic, dict)
        deltas = traffic["counter_deltas"]
        assert isinstance(deltas, dict)
        deltas["gpu0.link0.rx_kib"] = 0
        confirmation.write_text(
            json.dumps(hotspot, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif mutation == "ttft":
        phases = hotspot["phases"]
        assert isinstance(phases, dict)
        cold = phases["cold_first_exact"]
        assert isinstance(cold, dict)
        benchmark = cold["benchmark"]
        assert isinstance(benchmark, dict)
        benchmark["time_to_first_token_seconds"] = 7.01
        confirmation.write_text(
            json.dumps(hotspot, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        coherency_document["coherent"] = False
        coherency.write_text(
            json.dumps(coherency_document, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(transfer.PlanTransferError, match=message):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            source_ep_confirmation_receipt=confirmation,
            source_ep_coherency_receipt=coherency,
            cache_root=tmp_path / "cache",
            expected_target_gpu_experts_per_layer=28,
        )


def test_transfer_rejects_confirmation_without_physical_oscar(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "ep2-g14.pt"
    source = make_source_plan(source_path, source_gpu_experts_per_rank=14)
    semantics = source["placement_semantics_sha256"]
    assert isinstance(semantics, str)
    confirmation = tmp_path / "confirm.json"
    receipt = make_ep_confirmation(
        confirmation,
        source_plan=source_path,
        placement_semantics_sha256=semantics,
    )
    oscar_contract = receipt["oscar_contract"]
    assert isinstance(oscar_contract, dict)
    expected_server_info = oscar_contract["expected_server_info"]
    assert isinstance(expected_server_info, dict)
    expected_server_info["dsv4_oscar_int2_kv_storage"] = False
    confirmation.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        transfer.PlanTransferError,
        match="physical Oscar INT2 storage",
    ):
        transfer.stage_transferred_plan(
            source_ep2_plan=source_path,
            source_ep_confirmation_receipt=confirmation,
            cache_root=tmp_path / "cache",
        )


def test_cli_prints_only_content_addressed_plan_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    source = make_source_plan(source_path)
    cache_root = tmp_path / "cache"
    semantics = source["placement_semantics_sha256"]
    assert isinstance(semantics, str)
    confirmation = tmp_path / "confirm.json"
    make_ep_confirmation(
        confirmation,
        source_plan=source_path,
        placement_semantics_sha256=semantics,
    )

    status = transfer.main(
        [
            "--source-ep2-plan",
            str(source_path),
            "--source-ep-confirmation-receipt",
            str(confirmation),
            "--cache-root",
            str(cache_root),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    output_path = Path(captured.out.strip())
    assert output_path.is_file()
    assert output_path.parent == cache_root
    receipt = torch.load(output_path, map_location="cpu", weights_only=True)[
        "transfer_receipt"
    ]
    assert len(receipt["cache_identity_sha256"]) == 64


def test_identity_is_stable_and_json_serializable(tmp_path: Path) -> None:
    source_path = tmp_path / "ep2-g13.pt"
    make_source_plan(source_path)
    source = transfer.load_source_placement(source_path)
    identity = transfer._identity_document(
        source_plan=source_path,
        source_plan_sha256=sha256_file(source_path),
        source_placement=source,
        ep_confirmation=None,
    )

    encoded = json.dumps(identity, sort_keys=True)
    assert transfer.TRANSFER_FORMAT in encoded
