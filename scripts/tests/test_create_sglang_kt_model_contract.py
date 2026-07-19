import hashlib
import json
import stat
from pathlib import Path

from scripts import create_sglang_kt_model_contract as creator

MODEL_ID = "zai-org/GLM-4.7-Flash"
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
    (path / "config.json").write_text('{"model_type":"glm4_moe_lite"}\n')
    (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {"model.weight": "model-00001-of-00001.safetensors"},
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
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        write_hugging_face_metadata(path, filename)


def arguments(snapshot: Path, output: Path) -> list[str]:
    return [
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--model-id",
        MODEL_ID,
        "--revision",
        REVISION,
        "--ktransformers-method",
        "BF16",
        "--full-indexer-layer-starts",
        "0",
    ]


def test_cli_atomically_publishes_idempotent_contract(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "contract.json"
    write_snapshot(snapshot)

    assert creator.main(arguments(snapshot, output)) == 0
    first_contents = output.read_bytes()
    assert creator.main(arguments(snapshot, output)) == 0

    assert output.read_bytes() == first_contents
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not tuple(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_refuses_to_replace_different_existing_contract(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "contract.json"
    write_snapshot(snapshot)
    output.write_text("{}")

    assert creator.main(arguments(snapshot, output)) == 1
    assert output.read_text() == "{}"


def test_cli_refuses_contract_inside_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    write_snapshot(snapshot)
    output = snapshot / "contract.json"

    assert creator.main(arguments(snapshot, output)) == 1
    assert not output.exists()


def test_cli_refuses_symlink_output(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    target = tmp_path / "target.json"
    output = tmp_path / "contract.json"
    write_snapshot(snapshot)
    target.write_text("{}")
    output.symlink_to(target)

    assert creator.main(arguments(snapshot, output)) == 1
    assert target.read_text() == "{}"
