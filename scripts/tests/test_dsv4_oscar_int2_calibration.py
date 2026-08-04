import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.dsv4_oscar_int2_calibration import (
    ADMISSION_FORMAT,
    ALGORITHM,
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    C4_ROTATION_COMPOSITION,
    C4_ROTATION_OBJECTIVE,
    CAPTURE_FORMAT,
    CHECKPOINT_FORMAT,
    EXPECTED_COMPRESSION_RATIOS,
    GROUP_SIZE,
    INDEX_GROUP_SIZE,
    INDEX_HEAD_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    OSCAR_CLIP_SOURCE_SHA256,
    OSCAR_ROTATION_SOURCE_SHA256,
    OSCAR_SOURCE_COMMIT,
    PROMPT_FORMAT,
    SHARED_LATENT_SOURCE,
    SHARED_ROTATION_OBJECTIVE,
    STATISTICS_VERSION,
    ArtifactExpectations,
    CalibrationGates,
    ModelGeometry,
    _fit_domain,
    _geometry_dict,
    _validate_statistics,
    admit_artifact_for_checkpoint,
    balanced_bit_reversal_order,
    build_block_hadamard,
    build_checkpoint_fingerprint,
    collect_statistics,
    compose_oscar_rotation,
    fake_quantize_oscar_int2,
    load_validated_artifact,
    make_balancing_permutation,
    orthogonality_max_abs,
    select_group_clip_ratios,
    sha256_file,
    shared_value_sst_hessian,
    tree_sha256,
    validate_artifact,
    validate_model_config,
)


def _config() -> dict[str, object]:
    return {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "compress_ratios": list(EXPECTED_COMPRESSION_RATIOS),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_json(checkpoint / "config.json", _config())
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    _write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {"model.embed.weight": "model-00001-of-00001.safetensors"}},
    )
    fingerprint_path = tmp_path / "fingerprint.json"
    _write_json(fingerprint_path, build_checkpoint_fingerprint(checkpoint))
    return checkpoint, fingerprint_path


def test_flash_compression_topology_covers_exact_43_layers() -> None:
    assert len(EXPECTED_COMPRESSION_RATIOS) == NUM_LAYERS
    assert EXPECTED_COMPRESSION_RATIOS.count(0) == 2
    assert EXPECTED_COMPRESSION_RATIOS.count(4) == 21
    assert EXPECTED_COMPRESSION_RATIOS.count(128) == 20
    assert [
        index for index, ratio in enumerate(EXPECTED_COMPRESSION_RATIOS) if ratio == 4
    ] == list(range(2, 43, 2))


def test_model_config_rejects_any_abi_or_topology_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, _config())
    geometry = validate_model_config(
        config_path,
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        checkpoint_sha256="a" * 64,
    )
    assert geometry.latent_dim == 448
    assert geometry.rope_dim == 64

    changed = _config()
    changed["compress_ratios"] = [0] * 43
    _write_json(config_path, changed)
    with pytest.raises(ValueError, match="compression topology"):
        validate_model_config(
            config_path,
            model_id="deepseek-ai/DeepSeek-V4-Flash",
            checkpoint_sha256="a" * 64,
        )


def test_checkpoint_fingerprint_hashes_config_index_and_every_shard(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _tiny_checkpoint(tmp_path)
    fingerprint = build_checkpoint_fingerprint(checkpoint)
    assert fingerprint["format"] == CHECKPOINT_FORMAT
    assert len(fingerprint["files"]) == 3
    original = fingerprint["checkpoint_sha256"]
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"changed")
    assert build_checkpoint_fingerprint(checkpoint)["checkpoint_sha256"] != original


def test_448_transform_is_exact_orthogonal_without_padding_or_crop() -> None:
    order = balanced_bit_reversal_order(LATENT_DIM)
    assert order.shape == (448,)
    assert sorted(order.tolist()) == list(range(448))
    assert order[:8].tolist() == [0, 224, 112, 336, 56, 280, 168, 392]

    block_hadamard = build_block_hadamard(LATENT_DIM)
    hessian = torch.diag(torch.linspace(0.1, 10.0, LATENT_DIM, dtype=torch.float64))
    rotation, eigenvalues = compose_oscar_rotation(hessian)
    _, eigenvectors = torch.linalg.eigh(hessian)
    permutation = make_balancing_permutation(eigenvalues.to(torch.float64))
    assert block_hadamard.shape == (448, 448)
    assert rotation.shape == (448, 448)
    assert eigenvalues.shape == (448,)
    assert orthogonality_max_abs(rotation) < 2.0e-5
    torch.testing.assert_close(
        rotation,
        (eigenvectors @ permutation @ block_hadamard).to(torch.float32),
    )


def test_c4_transform_retains_official_u_h_pbr_composition() -> None:
    hessian = torch.diag(torch.linspace(0.1, 10.0, INDEX_HEAD_DIM, dtype=torch.float64))
    rotation, eigenvalues = compose_oscar_rotation(
        hessian, hadamard_block_size=INDEX_GROUP_SIZE
    )
    _, eigenvectors = torch.linalg.eigh(hessian)
    hadamard = build_block_hadamard(INDEX_HEAD_DIM, INDEX_GROUP_SIZE)
    permutation = make_balancing_permutation(
        eigenvalues.to(torch.float64), INDEX_GROUP_SIZE
    )
    torch.testing.assert_close(
        rotation,
        (eigenvectors @ hadamard @ permutation).to(torch.float32),
    )


def test_shared_hessian_is_official_attention_weighted_value_sst() -> None:
    value = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
    hessian = shared_value_sst_hessian(value)
    torch.testing.assert_close(hessian, value / torch.trace(value))


def test_shared_domain_fit_spectral_map_uses_sst_not_qqt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def capture_hessian(
        hessian: torch.Tensor, *, hadamard_block_size: int = GROUP_SIZE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hadamard_block_size == GROUP_SIZE
        captured["hessian"] = hessian.clone()
        return (
            torch.eye(LATENT_DIM, dtype=torch.float32),
            torch.linalg.eigvalsh(hessian).to(torch.float32),
        )

    monkeypatch.setattr(
        "scripts.dsv4_oscar_int2_calibration.compose_oscar_rotation",
        capture_hessian,
    )
    generator = torch.Generator().manual_seed(19)
    train = torch.randn((32, LATENT_DIM), generator=generator)
    heldout = torch.randn((16, LATENT_DIM), generator=generator)
    query = torch.diag(torch.linspace(9.0, 1.0, LATENT_DIM, dtype=torch.float64))
    value = torch.diag(torch.linspace(1.0, 5.0, LATENT_DIM, dtype=torch.float64))
    payload = _fit_domain(
        domain="attention_shared_latent",
        train_rows=train,
        heldout_rows=heldout,
        train_query_covariance=query,
        heldout_query_covariance=query,
        train_value_covariance=value,
        heldout_value_covariance=value,
        statistics_sha256="b" * 64,
        layer_id=3,
        gates=CalibrationGates(
            maximum_heldout_relative_error=2.0,
            minimum_improvement_vs_unrotated=-2.0,
            minimum_train_rows=32,
            minimum_heldout_rows=16,
        ),
        clip_candidates=(1.0,),
        group_size=GROUP_SIZE,
    )
    torch.testing.assert_close(captured["hessian"], value / torch.trace(value))
    assert not torch.equal(captured["hessian"], query / torch.trace(query))
    assert payload["rotation_objective"] == SHARED_ROTATION_OBJECTIVE


def test_per_row_group_quantile_then_affine_uint2_matches_manual_reference() -> None:
    row = torch.linspace(-3.0, 2.0, GROUP_SIZE).view(1, GROUP_SIZE)
    reconstructed, thresholds = fake_quantize_oscar_int2(
        row, torch.tensor([1.0], dtype=torch.float32)
    )
    minimum = row.min()
    maximum = row.max()
    scale = (maximum - minimum) / 3.0
    zero = -minimum / scale
    expected = (((row / scale + zero + 0.5).floor().clamp(0, 3)) - zero) * scale
    torch.testing.assert_close(reconstructed, expected)
    torch.testing.assert_close(thresholds, torch.tensor([[3.0]]))


def test_clip_calibration_emits_per_group_rho_index_and_diagnostic_thresholds() -> None:
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn((48, INDEX_HEAD_DIM), generator=generator)
    rotation, _ = compose_oscar_rotation(
        torch.eye(INDEX_HEAD_DIM, dtype=torch.float64),
        hadamard_block_size=INDEX_GROUP_SIZE,
    )
    ratios, thresholds, sweep = select_group_clip_ratios(
        rows,
        rotation,
        torch.eye(INDEX_HEAD_DIM, dtype=torch.float64),
        candidates=(0.9, 0.96, 1.0),
        group_size=INDEX_GROUP_SIZE,
    )
    assert ratios.shape == (1,)
    assert thresholds.shape == (1,)
    assert torch.all((ratios >= 0.9) & (ratios <= 1.0))
    assert torch.all(thresholds > 0)
    assert sweep["candidate_ratios"] == [0.9, 0.96, 1.0]


def test_domain_fit_records_u_h_pbr_proof_and_heldout_gate() -> None:
    generator = torch.Generator().manual_seed(11)
    train = torch.randn((64, INDEX_HEAD_DIM), generator=generator)
    heldout = torch.randn((32, INDEX_HEAD_DIM), generator=generator)
    covariance = torch.eye(INDEX_HEAD_DIM, dtype=torch.float64)
    payload = _fit_domain(
        domain="c4_scorer",
        train_rows=train,
        heldout_rows=heldout,
        train_query_covariance=covariance,
        heldout_query_covariance=covariance,
        train_value_covariance=None,
        heldout_value_covariance=None,
        statistics_sha256="b" * 64,
        layer_id=2,
        gates=CalibrationGates(
            maximum_heldout_relative_error=2.0,
            minimum_improvement_vs_unrotated=-2.0,
            minimum_train_rows=32,
            minimum_heldout_rows=16,
        ),
        clip_candidates=(0.9, 1.0),
        group_size=INDEX_GROUP_SIZE,
    )
    assert payload["domain"] == "c4_scorer"
    assert payload["rotation_objective"] == C4_ROTATION_OBJECTIVE
    assert payload["rotation_composition"] == C4_ROTATION_COMPOSITION
    assert payload["rotation"].shape == (128, 128)
    assert payload["clip_mode"] == "per_row_quantile"
    assert payload["clip_ratios"].shape == (1,)
    assert len(payload["provenance_sha256"]) == 64
    assert payload["heldout_metrics"]["row_count"] == 32

    with pytest.raises(ValueError, match="heldout error gate"):
        _fit_domain(
            domain="c4_scorer",
            train_rows=train,
            heldout_rows=heldout,
            train_query_covariance=covariance,
            heldout_query_covariance=covariance,
            train_value_covariance=None,
            heldout_value_covariance=None,
            statistics_sha256="b" * 64,
            layer_id=2,
            gates=CalibrationGates(
                maximum_heldout_relative_error=0.0,
                minimum_improvement_vs_unrotated=-2.0,
                minimum_train_rows=32,
                minimum_heldout_rows=16,
            ),
            clip_candidates=(1.0,),
            group_size=INDEX_GROUP_SIZE,
        )


def test_statistics_validation_requires_exact_layer_coverage() -> None:
    geometry = ModelGeometry(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        model_type="deepseek_v4",
        config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        compression_ratios=EXPECTED_COMPRESSION_RATIOS,
    )
    statistics: dict[str, object] = {
        "format": "dsv4-oscar-int2-statistics",
        "format_version": STATISTICS_VERSION,
        "algorithm": ALGORITHM,
        "model": _geometry_dict(geometry),
        "provenance": {
            "prompt_manifest_sha256": "c" * 64,
            "capture_manifest_sha256": "d" * 64,
            "checkpoint_fingerprint_sha256": "e" * 64,
            "capture_input_set_sha256": "f" * 64,
            "calibration_prompt_tokens": 1,
            "train_latent_rows": 1,
            "heldout_latent_rows": 1,
            "shared_latent_source": SHARED_LATENT_SOURCE,
            "shared_rotation_objective": SHARED_ROTATION_OBJECTIVE,
            "c4_rotation_objective": C4_ROTATION_OBJECTIVE,
            "oscar_source_commit": OSCAR_SOURCE_COMMIT,
            "oscar_rotation_source_sha256": OSCAR_ROTATION_SOURCE_SHA256,
            "oscar_clip_source_sha256": OSCAR_CLIP_SOURCE_SHA256,
        },
        "layers": {},
    }
    statistics["statistics_sha256"] = tree_sha256(statistics)
    with pytest.raises(ValueError, match="exactly compressed layers"):
        _validate_statistics(statistics)


def test_v1_statistics_fail_closed_after_compressed_history_abi_change() -> None:
    with pytest.raises(ValueError, match="incompatible OSCAR statistics format"):
        _validate_statistics(
            {
                "format": "dsv4-oscar-int2-statistics",
                "format_version": 1,
            }
        )


def test_collector_fails_closed_before_accepting_incomplete_43_layer_capture(
    tmp_path: Path,
) -> None:
    checkpoint, fingerprint_path = _tiny_checkpoint(tmp_path)
    prompts_path = tmp_path / "prompts.json"
    _write_json(
        prompts_path,
        {
            "format": PROMPT_FORMAT,
            "format_version": 1,
            "prompts": [
                {"id": "train", "split": "train", "text": "train", "token_count": 1},
                {
                    "id": "heldout",
                    "split": "heldout",
                    "text": "heldout",
                    "token_count": 1,
                },
            ],
        },
    )
    manifest_path = tmp_path / "capture.json"
    _write_json(
        manifest_path,
        {
            "format": CAPTURE_FORMAT,
            "format_version": 1,
            "model": {
                "model_id": "deepseek-ai/DeepSeek-V4-Flash",
                "checkpoint_path": str(checkpoint),
                "checkpoint_fingerprint_path": str(fingerprint_path),
                "checkpoint_fingerprint_sha256": sha256_file(fingerprint_path),
            },
            "prompts": {"path": str(prompts_path), "sha256": sha256_file(prompts_path)},
            "chunks": [],
        },
    )
    with pytest.raises(ValueError, match="no train chunk for layer 2"):
        collect_statistics(manifest_path, tmp_path / "stats.pt")


def test_artifact_loader_rejects_absent_or_non_oscar_payload(tmp_path: Path) -> None:
    expectations = ArtifactExpectations(
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        prompt_manifest_sha256="c" * 64,
    )
    with pytest.raises(FileNotFoundError, match="absent"):
        load_validated_artifact(tmp_path / "missing.pt", expectations)
    malformed = {
        "format": ARTIFACT_FORMAT,
        "format_version": ARTIFACT_VERSION,
        "artifact_provenance_sha256": hashlib.sha256(b"malformed").hexdigest(),
    }
    with pytest.raises(ValueError, match="frozen runtime ABI"):
        validate_artifact(malformed)


def test_model_admission_rehashes_every_shard_and_emits_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, fingerprint_path = _tiny_checkpoint(tmp_path)
    fingerprint_sha256 = sha256_file(fingerprint_path)
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    geometry = validate_model_config(
        checkpoint / "config.json",
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        checkpoint_sha256=fingerprint["checkpoint_sha256"],
    )
    artifact_path = tmp_path / "artifact.pt"
    artifact_path.write_bytes(b"artifact-file-for-admission-test")
    artifact = {
        "model": _geometry_dict(geometry),
        "provenance": {
            "checkpoint_fingerprint_sha256": fingerprint_sha256,
        },
        "artifact_provenance_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        "scripts.dsv4_oscar_int2_calibration.load_validated_artifact",
        lambda _path: artifact,
    )
    receipt_path = tmp_path / "admission.json"
    receipt = admit_artifact_for_checkpoint(
        artifact_path=artifact_path,
        checkpoint_path=checkpoint,
        checkpoint_fingerprint_path=fingerprint_path,
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        receipt_path=receipt_path,
    )
    assert receipt["format"] == ADMISSION_FORMAT
    assert receipt["admitted"] is True
    assert receipt["checkpoint_sha256"] == fingerprint["checkpoint_sha256"]
    assert receipt["artifact_file_sha256"] == sha256_file(artifact_path)
    assert len(receipt["admission_sha256"]) == 64
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt

    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="fingerprint does not match"):
        admit_artifact_for_checkpoint(
            artifact_path=artifact_path,
            checkpoint_path=checkpoint,
            checkpoint_fingerprint_path=fingerprint_path,
            model_id="deepseek-ai/DeepSeek-V4-Flash",
            receipt_path=tmp_path / "rejected.json",
        )
