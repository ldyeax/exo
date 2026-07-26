from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.materialize_glm52_hybrid_checkpoint import (
    EXPERT_CONTENT_KIND,
    EXPERT_MANIFEST_KIND,
    EXPERT_MANIFEST_SCHEMA_VERSION,
    EXPERT_NUMA_NODES,
    MARLIN_W8_PACK_FACTOR,
    MLA_KV_LORA_RANK,
    MLA_NUM_HEADS,
    MLA_QK_NOPE_HEAD_DIM,
    MLA_V_HEAD_DIM,
    HybridCheckpointError,
    SourceTensor,
    _bf16_bits_to_float32,
    _canonical_json_bytes,
    _expert_artifact_identity,
    _float32_to_bf16_bits,
    _manifest_quantization_contract,
    _output_unit,
    _quantization_config,
    _quantize_matrix,
    _quantize_mla_kv_b_matrix,
    _quantize_packed_w8_matrix,
    _sha256_bytes,
    _validate_expert_artifact_identity,
    classify_source_tensor,
)


def _expected_quantized_bytes(
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(weights, axis=0)
    minimum = np.min(weights, axis=0)
    scales = np.maximum(
        np.abs(maximum / np.float32(127.0)),
        np.abs(minimum / np.float32(-128.0)),
    ).astype(np.float32)
    scales = np.where(
        np.isfinite(scales) & (scales > 0),
        scales,
        np.float32(1.0),
    ).astype(np.float32)
    scale_bits = _float32_to_bf16_bits(scales)
    stored_scales = _bf16_bits_to_float32(scale_bits)
    signed = np.clip(
        np.rint(weights / stored_scales[np.newaxis, :]),
        -128,
        127,
    ).astype(np.int16)
    return (signed + np.int16(128)).astype(np.uint8), scale_bits


def _unpack_gptq_k_lanes(
    packed: np.ndarray,
    *,
    input_features: int,
) -> np.ndarray:
    packed_unsigned = packed.view(np.uint32)
    unpacked = np.empty(
        (input_features, packed.shape[1]),
        dtype=np.uint8,
    )
    for byte_index in range(MARLIN_W8_PACK_FACTOR):
        unpacked[byte_index::MARLIN_W8_PACK_FACTOR] = (
            packed_unsigned >> np.uint32(8 * byte_index)
        ).astype(np.uint8)
    return unpacked


@pytest.mark.parametrize(
    ("input_features", "output_features"),
    (
        (MLA_QK_NOPE_HEAD_DIM, MLA_KV_LORA_RANK),
        (MLA_KV_LORA_RANK, MLA_V_HEAD_DIM),
    ),
)
def test_backend_neutral_gptq_pack_round_trip(
    input_features: int,
    output_features: int,
) -> None:
    generator = np.random.default_rng(73)
    source_float32 = generator.normal(
        0.0,
        0.25,
        size=(input_features, output_features),
    ).astype(np.float32)
    source_bf16 = _bf16_bits_to_float32(_float32_to_bf16_bits(source_float32))
    expected_unsigned, expected_scale_bits = _expected_quantized_bytes(source_bf16)

    qweight, scale_bits = _quantize_packed_w8_matrix(source_bf16)

    assert qweight.shape == (
        input_features // MARLIN_W8_PACK_FACTOR,
        output_features,
    )
    assert qweight.dtype == np.dtype(np.int32)
    assert scale_bits.shape == (1, output_features)
    assert scale_bits.dtype == np.dtype(np.uint16)
    np.testing.assert_array_equal(
        _unpack_gptq_k_lanes(qweight, input_features=input_features),
        expected_unsigned,
    )
    np.testing.assert_array_equal(scale_bits[0], expected_scale_bits)


def test_backend_neutral_gptq_pack_has_stable_little_endian_lane_order() -> None:
    source = np.repeat(
        np.asarray((-128.0, -1.0, 0.0, 127.0), dtype=np.float32)[:, None],
        64,
        axis=1,
    )

    qweight, scale_bits = _quantize_packed_w8_matrix(source)

    expected_word = np.asarray([0xFF807F00], dtype=np.uint32).view(np.int32)[0]
    np.testing.assert_array_equal(
        qweight,
        np.full((1, 64), expected_word, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        scale_bits,
        _float32_to_bf16_bits(np.ones((1, 64), dtype=np.float32)),
    )


def test_mla_kv_b_writer_splits_heads_and_transposes_v_with_nonzero_offsets(
    tmp_path: Path,
) -> None:
    source_shape = (
        MLA_NUM_HEADS,
        MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM,
        MLA_KV_LORA_RANK,
    )
    source_bits = np.zeros(source_shape, dtype=np.uint16)
    generator = np.random.default_rng(131)
    for head in (0, MLA_NUM_HEADS - 1):
        source_bits[head] = _float32_to_bf16_bits(
            generator.normal(
                head / 32.0,
                0.25,
                size=source_shape[1:],
            ).astype(np.float32)
        )

    source_prefix = 24
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(
        b"\xa5" * source_prefix + source_bits.tobytes(order="C") + b"\x5a" * 17
    )
    source = SourceTensor(
        name="model.layers.78.self_attn.kv_b_proj.weight",
        dtype="BF16",
        shape=(
            MLA_NUM_HEADS * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM),
            MLA_KV_LORA_RANK,
        ),
        shard_name=source_path.name,
        absolute_path=source_path,
        absolute_data_start=source_prefix,
        absolute_data_end=source_prefix + source_bits.nbytes,
    )
    unit = _output_unit(source, classify_source_tensor(source))
    assert unit is not None

    output_offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for tensor in unit.tensors:
        output_offsets[tensor.name] = (cursor, cursor + tensor.size_bytes)
        cursor += tensor.size_bytes
    data_start = 80
    destination_path = tmp_path / "destination.bin"
    destination_path.write_bytes(b"\xcc" * (data_start + cursor + 31))

    _quantize_mla_kv_b_matrix(
        source,
        destination_path,
        data_start,
        output_offsets,
    )

    stem = source.name.removesuffix(".weight")
    output_views = {
        tensor.name: np.memmap(
            destination_path,
            mode="r",
            dtype="<i4" if tensor.dtype == "I32" else "<u2",
            offset=data_start + output_offsets[tensor.name][0],
            shape=tensor.shape,
            order="C",
        )
        for tensor in unit.tensors
    }
    try:
        for head in (0, MLA_NUM_HEADS - 1):
            logical = _bf16_bits_to_float32(source_bits[head])
            expected_kc_qweight, expected_kc_scales = _quantize_packed_w8_matrix(
                logical[:MLA_QK_NOPE_HEAD_DIM]
            )
            expected_vc_qweight, expected_vc_scales = _quantize_packed_w8_matrix(
                logical[MLA_QK_NOPE_HEAD_DIM:].T
            )
            np.testing.assert_array_equal(
                output_views[f"{stem}.kc_qweight"][head],
                expected_kc_qweight,
            )
            np.testing.assert_array_equal(
                output_views[f"{stem}.kc_scales"][head],
                expected_kc_scales,
            )
            np.testing.assert_array_equal(
                output_views[f"{stem}.vc_qweight"][head],
                expected_vc_qweight,
            )
            np.testing.assert_array_equal(
                output_views[f"{stem}.vc_scales"][head],
                expected_vc_scales,
            )
    finally:
        for view in output_views.values():
            del view
    assert destination_path.read_bytes()[:data_start] == b"\xcc" * data_start


def test_ordinary_linear_writer_matches_gptq_pack_rows(
    tmp_path: Path,
) -> None:
    output_features = 64
    input_features = 128
    generator = np.random.default_rng(91)
    source_float32 = generator.normal(
        0.0,
        0.25,
        size=(output_features, input_features),
    ).astype(np.float32)
    source_bits = _float32_to_bf16_bits(source_float32)
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(source_bits.tobytes(order="C"))
    source = SourceTensor(
        name="model.layers.0.self_attn.q_a_proj.weight",
        dtype="BF16",
        shape=(output_features, input_features),
        shard_name=source_path.name,
        absolute_path=source_path,
        absolute_data_start=0,
        absolute_data_end=source_bits.nbytes,
    )
    qweight_name = "model.layers.0.self_attn.q_a_proj.qweight"
    scales_name = "model.layers.0.self_attn.q_a_proj.scales"
    qweight_bytes = input_features * output_features
    scales_bytes = 2 * output_features
    destination_path = tmp_path / "destination.bin"
    destination_path.write_bytes(b"\0" * (qweight_bytes + scales_bytes))

    _quantize_matrix(
        source,
        destination_path,
        0,
        {
            qweight_name: (0, qweight_bytes),
            scales_name: (
                qweight_bytes,
                qweight_bytes + scales_bytes,
            ),
        },
        chunk_bytes=4096,
    )

    qweight = np.memmap(
        destination_path,
        mode="r",
        dtype="<i4",
        offset=0,
        shape=(input_features // MARLIN_W8_PACK_FACTOR, output_features),
    )
    scale_bits = np.memmap(
        destination_path,
        mode="r",
        dtype="<u2",
        offset=qweight_bytes,
        shape=(1, output_features),
    )
    expected_unsigned, expected_scale_bits = _expected_quantized_bytes(
        _bf16_bits_to_float32(source_bits).T
    )
    np.testing.assert_array_equal(
        _unpack_gptq_k_lanes(qweight, input_features=input_features),
        expected_unsigned,
    )
    np.testing.assert_array_equal(scale_bits[0], expected_scale_bits)


def test_packed_w8_rejects_unsupported_or_nonfinite_shape() -> None:
    with pytest.raises(HybridCheckpointError, match="pack four K lanes"):
        _quantize_packed_w8_matrix(np.zeros((191, 64), dtype=np.float32))
    invalid = np.zeros((192, 64), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(HybridCheckpointError, match="NaN or infinity"):
        _quantize_packed_w8_matrix(invalid)


def test_mla_kv_b_policy_emits_only_backend_neutral_compact_tensors() -> None:
    source = SourceTensor(
        name="model.layers.78.self_attn.kv_b_proj.weight",
        dtype="BF16",
        shape=(
            MLA_NUM_HEADS * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM),
            MLA_KV_LORA_RANK,
        ),
        shard_name="model.safetensors",
        absolute_path=Path("/checkpoint/model.safetensors"),
        absolute_data_start=8,
        absolute_data_end=(
            8
            + 2
            * MLA_NUM_HEADS
            * (MLA_QK_NOPE_HEAD_DIM + MLA_V_HEAD_DIM)
            * MLA_KV_LORA_RANK
        ),
    )

    disposition = classify_source_tensor(source)
    unit = _output_unit(source, disposition)

    assert disposition == "quantize_mla_kv_b_w8"
    assert unit is not None
    assert tuple(tensor.name for tensor in unit.tensors) == (
        "model.layers.78.self_attn.kv_b_proj.kc_qweight",
        "model.layers.78.self_attn.kv_b_proj.kc_scales",
        "model.layers.78.self_attn.kv_b_proj.vc_qweight",
        "model.layers.78.self_attn.kv_b_proj.vc_scales",
    )
    assert tuple(tensor.shape for tensor in unit.tensors) == (
        (64, 48, 512),
        (64, 1, 512),
        (64, 128, 256),
        (64, 1, 256),
    )
    assert unit.size_bytes < source.size_bytes
    assert all(
        tensor.dtype != "BF16" or "scales" in tensor.name for tensor in unit.tensors
    )


def test_quantization_config_marks_backend_neutral_specialist_layout() -> None:
    config = _quantization_config()

    assert config["quant_method"] == "gptq"
    assert config["bits"] == 8
    assert config["group_size"] == -1
    assert config["desc_act"] is False
    assert config["sym"] is True
    assert config["exo_mla_kv_b_w8"] == {
        "bits": 8,
        "block_size_m": 8,
        "format": "gptq_packed_rows_per_head_v1",
        "implicit_bias": 128,
        "kv_lora_rank": 512,
        "num_attention_heads": 64,
        "pack_axis": "K",
        "pack_order": "little_endian_k_lanes_0_1_2_3",
        "qk_nope_head_dim": 192,
        "scale_compute_dtype": "float32",
        "scale_dtype": "bfloat16",
        "v_head_dim": 256,
    }
    dynamic = config["dynamic"]
    assert isinstance(dynamic, dict)
    assert not any("kv_b_proj" in pattern for pattern in dynamic)


def test_manifest_quantization_contract_requires_marlin_direct_consumption() -> None:
    contract = _manifest_quantization_contract()

    assert contract["kernel"] == {
        "ordinary_linear": "gptq_marlin_w8a16",
        "mla_kv_b": "grouped_marlin_w8a16",
    }
    assert contract["temporary_bf16_expansion_at_load"] is False
    assert contract["mla_kv_b_compact_to_compact_marlin_repack_at_load"] is True
    assert not any("triton" in key.lower() for key in contract)


def _write_expert_artifact_fixture(
    tmp_path: Path,
    *,
    actual_payload: bytes,
    manifested_payload: bytes | None = None,
) -> tuple[Path, Path, str]:
    expert_root = tmp_path / "experts"
    expert_root.mkdir()
    payload_path = expert_root / "weights.bin"
    payload_path.write_bytes(actual_payload)
    payload_path.chmod(0o444)
    expert_root.chmod(0o555)

    attested_payload = (
        actual_payload if manifested_payload is None else manifested_payload
    )
    files = [
        {
            "path": payload_path.name,
            "sha256": _sha256_bytes(attested_payload),
            "size_bytes": len(attested_payload),
        }
    ]
    content_id = _sha256_bytes(
        _canonical_json_bytes(
            {
                "files": files,
                "kind": EXPERT_CONTENT_KIND,
                "schema_version": EXPERT_MANIFEST_SCHEMA_VERSION,
            }
        )
    )
    manifest_path = tmp_path / "expert-manifest.json"
    manifest_path.write_bytes(
        _canonical_json_bytes(
            {
                "content_id": content_id,
                "files": files,
                "kind": EXPERT_MANIFEST_KIND,
                "numa_nodes": list(EXPERT_NUMA_NODES),
                "schema_version": EXPERT_MANIFEST_SCHEMA_VERSION,
            },
            pretty=True,
        )
    )
    manifest_path.chmod(0o444)
    return expert_root, manifest_path, content_id


def test_expert_artifact_identity_authenticates_and_rechecks_payload(
    tmp_path: Path,
) -> None:
    expert_root, manifest_path, content_id = _write_expert_artifact_fixture(
        tmp_path,
        actual_payload=b"immutable AMXINT4 payload",
    )

    identity = _expert_artifact_identity(
        expert_root,
        manifest_path,
        content_id,
    )

    assert identity.content_id == content_id
    assert tuple(path.name for path in identity.file_identities) == ("weights.bin",)
    _validate_expert_artifact_identity(identity)

    payload_path = expert_root / "weights.bin"
    payload_path.chmod(0o644)
    payload_path.write_bytes(b"mutated! AMXINT4 payload")
    payload_path.chmod(0o444)
    with pytest.raises(HybridCheckpointError, match="expert files changed"):
        _validate_expert_artifact_identity(identity)


def test_expert_artifact_identity_rejects_counterfeit_file_table(
    tmp_path: Path,
) -> None:
    expert_root, manifest_path, content_id = _write_expert_artifact_fixture(
        tmp_path,
        actual_payload=b"wrong payload",
        manifested_payload=b"right payload",
    )

    with pytest.raises(HybridCheckpointError, match="file hash differs"):
        _expert_artifact_identity(
            expert_root,
            manifest_path,
            content_id,
        )
