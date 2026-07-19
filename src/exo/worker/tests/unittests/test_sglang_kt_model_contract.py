import hashlib
import json
import os
from pathlib import Path

import pytest

from exo.shared.types.common import ModelId
from exo.worker.sglang_kt.model_contract import (
    SglangKtModelContractError,
    calculate_sglang_kt_model_contract_sha256,
    create_sglang_kt_model_contract,
    load_sglang_kt_model_contract,
    verify_sglang_kt_model_snapshot,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)

MODEL_ID = ModelId("zai-org/GLM-4.7-Flash")
REVISION = "7" * 40


def git_blob_sha1(contents: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(contents)}\0".encode())
    digest.update(contents)
    return digest.hexdigest()


def write_hugging_face_metadata(snapshot: Path, filename: str) -> None:
    contents = (snapshot / filename).read_bytes()
    etag = (
        hashlib.sha256(contents).hexdigest()
        if filename.endswith(".safetensors") or filename == "tokenizer.json"
        else git_blob_sha1(contents)
    )
    metadata = snapshot / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f"{REVISION}\n{etag}\n1.0\n")


def write_snapshot(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Glm4MoeLiteForCausalLM"]}) + "\n"
    )
    (path / "model-00001-of-00002.safetensors").write_bytes(b"first-shard")
    (path / "model-00002-of-00002.safetensors").write_bytes(b"second-shard")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "model.layer.0.weight": "model-00001-of-00002.safetensors",
                    "model.layer.1.weight": "model-00002-of-00002.safetensors",
                },
            }
        )
        + "\n"
    )
    (path / "chat_template.jinja").write_text("{{ messages }}\n")
    (path / "generation_config.json").write_text("{}\n")
    (path / "tokenizer.json").write_text('{"version":"1.0"}\n')
    (path / "tokenizer_config.json").write_text("{}\n")
    for filename in (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        write_hugging_face_metadata(path, filename)


def write_contract(snapshot: Path, destination: Path) -> tuple[str, str]:
    contract = create_sglang_kt_model_contract(
        snapshot,
        model_id=MODEL_ID,
        revision=REVISION,
        ktransformers_method="BF16",
        full_indexer_layer_starts=(0,),
    )
    contents = canonical_sglang_kt_json(contract.model_dump(mode="json"))
    destination.write_bytes(contents)
    return (
        calculate_sglang_kt_model_contract_sha256(contract),
        hashlib.sha256(contents).hexdigest(),
    )


def test_exact_contract_round_trip_verifies_all_indexed_shards(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, receipt_sha256 = write_contract(snapshot, contract_path)

    assert contract_sha256 == (
        "7d2d25b18e0e887fc96b2ba7904728003a049448cb8f7723c56fe6784e2406e5"
    )

    loaded = load_sglang_kt_model_contract(
        contract_path,
        expected_contract_sha256=contract_sha256,
    )
    verified = verify_sglang_kt_model_snapshot(
        snapshot,
        contract_path,
        expected_contract_sha256=contract_sha256,
        expected_model_id=MODEL_ID,
        expected_revision=REVISION,
        expected_ktransformers_method="BF16",
    )

    assert loaded.receipt_sha256 == receipt_sha256
    assert verified.contract_sha256 == contract_sha256
    assert (
        verified.config_sha256
        == hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    )
    assert verified.shard_count == 2
    assert verified.weight_map_entries == 2
    assert verified.physical_weight_bytes == len(b"first-shardsecond-shard")


def test_contract_rejects_same_size_shard_mutation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"wrong-shard")

    with pytest.raises(SglangKtModelContractError, match="pinned SHA-256"):
        verify_sglang_kt_model_snapshot(
            snapshot,
            contract_path,
            expected_contract_sha256=contract_sha256,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
            expected_ktransformers_method="BF16",
        )


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
def test_contract_rejects_config_or_index_mutation(
    tmp_path: Path, filename: str
) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    artifact = snapshot / filename
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(SglangKtModelContractError):
        verify_sglang_kt_model_snapshot(
            snapshot,
            contract_path,
            expected_contract_sha256=contract_sha256,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
            expected_ktransformers_method="BF16",
        )


def test_contract_rejects_tokenizer_mutation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    (snapshot / "tokenizer.json").write_text('{"version":"2.0"}\n')

    with pytest.raises(SglangKtModelContractError, match="runtime file"):
        verify_sglang_kt_model_snapshot(
            snapshot,
            contract_path,
            expected_contract_sha256=contract_sha256,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
            expected_ktransformers_method="BF16",
        )


def test_contract_rejects_uncontracted_remote_code(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    (snapshot / "modeling_untrusted.py").write_text("raise RuntimeError\n")

    with pytest.raises(SglangKtModelContractError, match="executable code"):
        create_sglang_kt_model_contract(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            ktransformers_method="BF16",
            full_indexer_layer_starts=(0,),
        )


def test_contract_rejects_mixed_hugging_face_revision_metadata(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    metadata = (
        snapshot
        / ".cache/huggingface/download/model-00001-of-00002.safetensors.metadata"
    )
    lines = metadata.read_text().splitlines()
    metadata.write_text(f"{'8' * 40}\n{lines[1]}\n{lines[2]}\n")

    with pytest.raises(SglangKtModelContractError, match="does not bind"):
        create_sglang_kt_model_contract(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            ktransformers_method="BF16",
            full_indexer_layer_starts=(0,),
        )


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_contract_rejects_missing_or_unindexed_shard(tmp_path: Path, mode: str) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    if mode == "missing":
        (snapshot / "model-00002-of-00002.safetensors").unlink()
    else:
        (snapshot / "unindexed.safetensors").write_bytes(b"unexpected")

    with pytest.raises(SglangKtModelContractError, match="exactly match the index"):
        create_sglang_kt_model_contract(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            ktransformers_method="BF16",
            full_indexer_layer_starts=(0,),
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.safetensors",
        "/absolute.safetensors",
        "nested/shard.safetensors",
        "a\\b.safetensors",
    ],
)
def test_contract_rejects_unsafe_index_shard_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    index = snapshot / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {"tensor": unsafe_path},
            }
        )
    )

    with pytest.raises(SglangKtModelContractError, match="unsafe shard path"):
        create_sglang_kt_model_contract(
            snapshot,
            model_id=MODEL_ID,
            revision=REVISION,
            ktransformers_method="BF16",
            full_indexer_layer_starts=(0,),
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(SglangKtReceiptFileError, match="repeats key"):
        parse_sglang_kt_strict_json(b'{"status":"passed","status":"failed"}')
    with pytest.raises(SglangKtReceiptFileError, match="non-finite"):
        parse_sglang_kt_strict_json(b'{"value":NaN}')


def test_contract_loader_rejects_extra_fields_and_wrong_pin(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    document = json.loads(contract_path.read_text())
    document["untrusted"] = True
    contract_path.write_text(json.dumps(document))

    with pytest.raises(SglangKtModelContractError, match="invalid model contract"):
        load_sglang_kt_model_contract(
            contract_path,
            expected_contract_sha256=contract_sha256,
        )

    write_contract(snapshot, contract_path)
    with pytest.raises(SglangKtModelContractError, match="pinned canonical"):
        load_sglang_kt_model_contract(
            contract_path,
            expected_contract_sha256="f" * 64,
        )


def test_verifier_reloads_contract_and_rechecks_pin(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    load_sglang_kt_model_contract(
        contract_path,
        expected_contract_sha256=contract_sha256,
    )
    document = json.loads(contract_path.read_text())
    document["model_id"] = "untrusted/substitution"
    contract_path.write_text(json.dumps(document))

    with pytest.raises(SglangKtModelContractError, match="pinned canonical"):
        verify_sglang_kt_model_snapshot(
            snapshot,
            contract_path,
            expected_contract_sha256=contract_sha256,
            expected_model_id=MODEL_ID,
            expected_revision=REVISION,
            expected_ktransformers_method="BF16",
        )


def test_contract_rejects_empty_nondeterministic_model_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    contract_path = tmp_path / "contract.json"
    _contract_sha256, _receipt_sha256 = write_contract(snapshot, contract_path)
    document = json.loads(contract_path.read_text())
    document["model_id"] = ""
    contract_path.write_text(json.dumps(document))

    with pytest.raises(SglangKtModelContractError, match="invalid model contract"):
        load_sglang_kt_model_contract(
            contract_path,
            expected_contract_sha256="f" * 64,
        )


def test_bound_file_reader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(target)

    with pytest.raises(SglangKtReceiptFileError, match="without following symlinks"):
        read_sglang_kt_bound_file(symlink, maximum_bytes=100)
    with pytest.raises(SglangKtReceiptFileError, match="singly linked"):
        read_sglang_kt_bound_file(hardlink, maximum_bytes=100)


def test_bound_file_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "receipt.fifo"
    os.mkfifo(fifo)

    with pytest.raises(SglangKtReceiptFileError, match="singly linked regular"):
        read_sglang_kt_bound_file(fifo, maximum_bytes=100)
