from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts import create_sglang_olmoe_stage_contract as stage
from scripts.sglang_olmoe_serving_client import token_ids_sha256


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == stage.OLMOE_SANITY_PROMPT
        assert add_special_tokens is True
        return [101, 102, 103]


def _snapshot() -> stage.SnapshotObservation:
    files = tuple(
        stage.SnapshotFileObservation(
            path=item.path,
            role=item.role,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
            huggingface_etag=item.huggingface_etag,
            metadata_sha256="f" * 64,
        )
        for item in stage.PINNED_SNAPSHOT_FILES
    )
    return stage.SnapshotObservation(
        model_id=stage.OLMOE_MODEL_ID,
        revision=stage.OLMOE_MODEL_REVISION,
        model_path=stage.OLMOE_MODEL_PATH,
        files=files,
        physical_weight_bytes=stage.OLMOE_PHYSICAL_WEIGHT_BYTES,
        indexed_weight_bytes=stage.OLMOE_INDEXED_WEIGHT_BYTES,
        weight_map_entries=stage.OLMOE_WEIGHT_MAP_ENTRIES,
        shard_count=3,
        full_file_content_rehash=True,
        huggingface_revision_metadata_verified=True,
        canonical_sha256="1" * 64,
    )


def _producer_sha256() -> str:
    return hashlib.sha256(Path(stage.__file__).read_bytes()).hexdigest()


def _managed_launch(ep_size: stage.ExpertParallelSize) -> stage.ManagedLaunchEvidence:
    namespace = f"exo-olmoe-ep-test-{'a' * 32}"
    command = (
        "/usr/bin/numactl",
        "--physcpubind",
        "0-111",
        "--interleave",
        "0,1",
        "/runtime/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        stage.OLMOE_MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        "62610",
        "--tp-size",
        "2",
        "--pp-size",
        "1",
        "--ep-size",
        str(ep_size),
        "--numa-node",
        "0",
        "1",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--dtype",
        "bfloat16",
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.9",
        "--max-running-requests",
        "1",
        "--random-seed",
        "20260720",
        "--disable-radix-cache",
    )
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": (
            "GPU-63a7760a-6164-0758-9228-03dbf35d721c,"
            "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
        ),
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_SOCKET_IFNAME": "lo",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_DEBUG": "INFO",
        "OMP_NUM_THREADS": "56",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
        "SGLANG_NUMA_BIND_V2": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "EXO_OLMOE_EP_NAMESPACE": namespace,
    }
    harness_path = Path(stage.__file__).with_name(
        "run_sglang_olmoe_ep_local_benchmark.py"
    )
    return stage.ManagedLaunchEvidence(
        schema_version=1,
        status="owned_runtime_listener_verified",
        harness_sha256=hashlib.sha256(harness_path.read_bytes()).hexdigest(),
        runtime_admission_sha256="5" * 64,
        pid=123,
        process_group_id=123,
        start_time_ticks=456,
        owner_token_sha256="6" * 64,
        ownership_namespace=namespace,
        command=command,
        launch_environment=tuple(sorted(environment.items())),
        listener_socket_inodes=(789,),
        listener_owner_pids=(123,),
        rank_local_numa_observation_sha256="7" * 64,
        verified_at_utc="2020-01-01T00:00:00+00:00",
    )


def _capture(
    ep_size: stage.ExpertParallelSize,
    *,
    output_ids: tuple[int, ...] = (201, 202),
) -> stage.SanityCapture:
    output_text = "42"
    input_ids = (101, 102, 103)
    return stage.SanityCapture(
        schema_version=1,
        status="captured",
        expert_parallel_size=ep_size,
        model_id=stage.OLMOE_MODEL_ID,
        model_revision=stage.OLMOE_MODEL_REVISION,
        model_path=stage.OLMOE_MODEL_PATH,
        sglang_revision=stage.OLMOE_SGLANG_REVISION,
        runtime_install_receipt_sha256="0" * 64,
        snapshot_canonical_sha256="1" * 64,
        prompt_text=stage.OLMOE_SANITY_PROMPT,
        tokenizer_class="_Tokenizer",
        input_ids=input_ids,
        input_ids_sha256=token_ids_sha256(input_ids),
        output_ids=output_ids,
        output_ids_sha256=token_ids_sha256(output_ids),
        output_text=output_text,
        output_text_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
        max_new_tokens=stage.OLMOE_SANITY_MAX_NEW_TOKENS,
        sampling_seed=stage.OLMOE_SANITY_SAMPLING_SEED,
        static_memory_fraction=0.9,
        server_host="127.0.0.1",
        server_port=62_610,
        server_info_sha256="2" * 64,
        managed_launch=_managed_launch(ep_size),
        producer_sha256=_producer_sha256(),
        captured_at_utc=(
            "2020-01-01T00:00:00+00:00" if ep_size == 1 else "2020-01-01T00:01:00+00:00"
        ),
    )


def test_public_snapshot_admission_rejects_tiny_fake_path(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n")

    with pytest.raises(stage.OlmoeStageContractError, match="pinned OLMoE location"):
        stage.verify_pinned_snapshot(tmp_path)


def test_contract_requires_exact_ep1_ep2_output_equivalence() -> None:
    first = stage.SanityCaptureBinding(receipt_sha256="3" * 64, capture=_capture(1))
    second = stage.SanityCaptureBinding(
        receipt_sha256="4" * 64, capture=_capture(2, output_ids=(201, 203))
    )

    with pytest.raises(ValidationError, match="not exactly equivalent"):
        stage.OlmoeStageContract(
            schema_version=1,
            status="published",
            canonicalization="exo-olmoe-stage-contract-v1",
            snapshot=_snapshot(),
            ep1=first,
            ep2=second,
            published_at_utc="2020-01-01T00:02:00+00:00",
            producer_sha256=_producer_sha256(),
        )


def test_contract_rejects_reversed_capture_order() -> None:
    first_capture = _capture(1).model_copy(
        update={"captured_at_utc": "2020-01-01T00:02:00+00:00"}
    )
    second_capture = _capture(2)

    with pytest.raises(ValidationError, match="ordered EP1 then EP2"):
        stage.OlmoeStageContract(
            schema_version=1,
            status="published",
            canonicalization="exo-olmoe-stage-contract-v1",
            snapshot=_snapshot(),
            ep1=stage.SanityCaptureBinding(
                receipt_sha256="3" * 64, capture=first_capture
            ),
            ep2=stage.SanityCaptureBinding(
                receipt_sha256="4" * 64, capture=second_capture
            ),
            published_at_utc="2020-01-01T00:03:00+00:00",
            producer_sha256=_producer_sha256(),
        )


def test_contract_rejects_capture_with_unmatched_managed_command() -> None:
    second = _capture(2)
    command = list(second.managed_launch.command)
    command[command.index("--tp-size") + 1] = "1"
    forged_launch = second.managed_launch.model_copy(update={"command": tuple(command)})
    forged_capture = second.model_copy(update={"managed_launch": forged_launch})

    with pytest.raises(ValidationError, match="managed launch"):
        stage.OlmoeStageContract(
            schema_version=1,
            status="published",
            canonicalization="exo-olmoe-stage-contract-v1",
            snapshot=_snapshot(),
            ep1=stage.SanityCaptureBinding(
                receipt_sha256="3" * 64, capture=_capture(1)
            ),
            ep2=stage.SanityCaptureBinding(
                receipt_sha256="4" * 64, capture=forged_capture
            ),
            published_at_utc="2020-01-01T00:02:00+00:00",
            producer_sha256=_producer_sha256(),
        )


def test_atomic_capture_publish_and_contract_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot()
    ep1_path = tmp_path / "ep1.json"
    ep2_path = tmp_path / "ep2.json"
    ep1_sha = stage._atomic_publish(ep1_path, _capture(1))
    ep2_sha = stage._atomic_publish(ep2_path, _capture(2))
    monkeypatch.setattr(stage, "verify_pinned_snapshot", lambda _path: snapshot)
    contract_path = tmp_path / "contract.json"

    receipt_sha = stage.publish_contract(
        snapshot_path=Path(stage.OLMOE_MODEL_PATH),
        ep1_capture=ep1_path,
        ep2_capture=ep2_path,
        output=contract_path,
    )
    loaded = stage.load_stage_contract(
        contract_path,
        tokenizer_loader=lambda _path: (_Tokenizer(), "_Tokenizer"),
    )

    assert loaded.receipt_sha256 == receipt_sha
    assert loaded.contract.ep1.receipt_sha256 == ep1_sha
    assert loaded.contract.ep2.receipt_sha256 == ep2_sha
    assert os.stat(contract_path).st_mode & 0o777 == 0o600


def test_capture_server_info_binds_memory_fraction() -> None:
    response = {
        "version": stage.PINNED_SERVER_VERSION,
        "model_path": stage.OLMOE_MODEL_PATH,
        "host": "127.0.0.1",
        "port": 62_610,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 2,
        "nnodes": 1,
        "node_rank": 0,
        "dtype": "bfloat16",
        "context_length": 4096,
        "max_running_requests": 1,
        "random_seed": stage.OLMOE_SANITY_SAMPLING_SEED,
        "mem_fraction_static": 0.9,
        "disable_radix_cache": True,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "numa_node": [0, 1],
    }

    stage._verify_capture_server_info(response, 2, 0.9, "127.0.0.1", 62_610)
    response["mem_fraction_static"] = 0.8
    with pytest.raises(stage.OlmoeStageContractError, match="server_info mismatch"):
        stage._verify_capture_server_info(response, 2, 0.9, "127.0.0.1", 62_610)
