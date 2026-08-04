from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import audit_dsv4_mxfp4_scale_domain as audit

TensorValue = tuple[str, tuple[int, ...], bytes]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_RECEIPT = (
    REPOSITORY_ROOT
    / "scripts/data/dsv4_flash_mxfp4_scale_domain_2026-08-04.json"
)
CHECKPOINT_RECEIPT_SHA256 = (
    "1b027325b31524de03c6a7b580e5304125adb8154bff4a96c8ded4d6d349499c"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_safetensors(path: Path, tensors: Mapping[str, TensorValue]) -> None:
    header: dict[str, object] = {}
    contents = bytearray()
    for key in sorted(tensors):
        dtype, shape, value = tensors[key]
        start = len(contents)
        contents.extend(value)
        header[key] = {
            "data_offsets": [start, len(contents)],
            "dtype": dtype,
            "shape": list(shape),
        }
    raw_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + contents)


def _checkpoint(
    root: Path,
    *,
    layer_count: int = 2,
    expert_count: int = 2,
    scale_elements: int = 4,
    values: Callable[[int, int, str], bytes] | None = None,
    dtype: str = audit.EXPECTED_DTYPE,
) -> Path:
    root.mkdir()
    _write_json(
        root / audit.CONFIG_FILENAME,
        {
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "n_routed_experts": expert_count,
            "num_experts_per_tok": 6,
            "num_hidden_layers": layer_count,
        },
    )
    weight_map: dict[str, str] = {}
    total_size = 0
    for layer in range(layer_count):
        shard_name = f"model-{layer + 1:05d}-of-{layer_count:05d}.safetensors"
        tensors: dict[str, TensorValue] = {}
        for expert in range(expert_count):
            for projection_index, projection in enumerate(audit.PROJECTIONS):
                key = (
                    f"layers.{layer}.ffn.experts.{expert}."
                    f"{projection}.scale"
                )
                value = (
                    values(layer, expert, projection)
                    if values is not None
                    else bytes(
                        [
                            118
                            + (
                                layer + expert + projection_index + element
                            )
                            % 9
                            for element in range(scale_elements)
                        ]
                    )
                )
                assert len(value) == scale_elements
                tensors[key] = (dtype, (1, scale_elements), value)
                weight_map[key] = shard_name
                total_size += len(value)
        _write_safetensors(root / shard_name, tensors)
    _write_json(
        root / audit.INDEX_FILENAME,
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    return root


def _expectations(
    *, layer_count: int = 2, expert_count: int = 2, scale_elements: int = 4
) -> audit.AuditExpectations:
    return audit.AuditExpectations(
        layer_count=layer_count,
        expert_count=expert_count,
        scale_elements=scale_elements,
    )


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_audit_reads_complete_raw_e8m0_domain_deterministically(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")

    first = audit.audit_checkpoint(checkpoint, expectations=_expectations())
    second = audit.audit_checkpoint(checkpoint, expectations=_expectations())

    assert first == second
    coverage = _object(first["coverage"])
    assert coverage == {
        "expected_experts_per_layer": 2,
        "expected_layer_count": 2,
        "expected_projections": ["w1", "w2", "w3"],
        "expected_scale_elements_per_tensor": 4,
        "index_shard_count": 2,
        "selected_shard_count": 2,
        "tensor_count": 12,
        "total_scale_bytes": 48,
    }
    domain = _object(first["scale_domain"])
    assert domain["branchless_fold_admitted"] is True
    assert domain["minimum"] == 118
    assert domain["maximum"] == 125
    assert domain["safe_scale_bytes"] == 48
    assert domain["unsafe_scale_bytes"] == 0
    assert sum(cast(dict[str, int], domain["histogram"]).values()) == 48
    assert len(cast(list[object], first["selected_shards"])) == 2
    audit.validate_fold_safe(first)


def test_scale_content_digest_changes_when_one_scale_byte_changes(
    tmp_path: Path,
) -> None:
    first_checkpoint = _checkpoint(tmp_path / "first")

    def changed_values(layer: int, expert: int, projection: str) -> bytes:
        if (layer, expert, projection) == (1, 1, "w3"):
            return bytes((120, 120, 120, 121))
        return bytes((120, 120, 120, 120))

    second_checkpoint = _checkpoint(
        tmp_path / "second", values=changed_values
    )
    baseline_checkpoint = _checkpoint(
        tmp_path / "baseline",
        values=lambda _layer, _expert, _projection: bytes((120,) * 4),
    )

    first_digest = _object(
        audit.audit_checkpoint(
            first_checkpoint, expectations=_expectations()
        )["scale_domain"]
    )["scale_content_sha256"]
    baseline_digest = _object(
        audit.audit_checkpoint(
            baseline_checkpoint, expectations=_expectations()
        )["scale_domain"]
    )["scale_content_sha256"]
    changed_digest = _object(
        audit.audit_checkpoint(
            second_checkpoint, expectations=_expectations()
        )["scale_domain"]
    )["scale_content_sha256"]

    assert first_digest != baseline_digest
    assert changed_digest != baseline_digest


def test_audit_rejects_incomplete_cartesian_coverage(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    index_path = checkpoint / audit.INDEX_FILENAME
    index = cast(dict[str, Any], json.loads(index_path.read_text(encoding="utf-8")))
    weight_map = cast(dict[str, str], index["weight_map"])
    del weight_map["layers.1.ffn.experts.1.w3.scale"]
    _write_json(index_path, index)

    with pytest.raises(audit.ScaleDomainAuditError, match="missing_count=1"):
        audit.audit_checkpoint(checkpoint, expectations=_expectations())


def test_audit_rejects_non_e8m0_dtype(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint", dtype="U8")

    with pytest.raises(audit.ScaleDomainAuditError, match="must use F8_E8M0"):
        audit.audit_checkpoint(checkpoint, expectations=_expectations())


def test_audit_reports_exotic_domain_and_rejects_ocp_nan(tmp_path: Path) -> None:
    values = bytes((0, 1, 253, 254, 255))
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        layer_count=1,
        expert_count=1,
        scale_elements=len(values),
        values=lambda _layer, _expert, _projection: values,
    )
    receipt = audit.audit_checkpoint(
        checkpoint,
        expectations=_expectations(
            layer_count=1, expert_count=1, scale_elements=len(values)
        ),
    )

    domain = _object(receipt["scale_domain"])
    assert domain["branchless_fold_admitted"] is False
    assert domain["safe_scale_bytes"] == 0
    assert domain["unsafe_scale_bytes"] == 15
    assert domain["e8m0_nan_count"] == 3
    with pytest.raises(audit.ScaleDomainAuditError, match="OCP E8M0 NaN"):
        audit.validate_fold_safe(receipt)


def test_publish_receipt_is_canonical_idempotent_and_non_overwriting(
    tmp_path: Path,
) -> None:
    path = (tmp_path / "receipt.json").resolve()
    receipt = {"format": audit.RECEIPT_FORMAT, "version": 1}

    digest = audit.publish_receipt(path, receipt)

    contents = path.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == digest
    assert contents == b'{"format":"dsv4-mxfp4-ue8m0-scale-domain","version":1}\n'
    assert audit.publish_receipt(path, receipt) == digest
    with pytest.raises(audit.ScaleDomainAuditError, match="refusing to overwrite"):
        audit.publish_receipt(path, {**receipt, "version": 2})


def test_cli_fails_closed_unless_unsafe_forensics_are_explicit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values = bytes((0, 1, 253, 254, 255))
    checkpoint = _checkpoint(
        tmp_path / "checkpoint",
        layer_count=1,
        expert_count=1,
        scale_elements=len(values),
        values=lambda _layer, _expert, _projection: values,
    ).resolve()
    output = (tmp_path / "unsafe.json").resolve()
    common = [
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--expected-layers",
        "1",
        "--expected-experts",
        "1",
        "--expected-scale-elements",
        str(len(values)),
    ]

    assert audit.main(common) == 1
    assert not output.exists()
    assert "OCP E8M0 NaN" in capsys.readouterr().err
    assert audit.main([*common, "--allow-unsafe"]) == 0
    assert output.is_file()
    summary = json.loads(capsys.readouterr().out)
    assert summary["branchless_fold_admitted"] is False


def test_cli_requires_absolute_normalized_paths() -> None:
    with pytest.raises(SystemExit):
        audit.parse_arguments(
            ["--checkpoint", "relative", "--output", "/tmp/receipt.json"]
        )


def test_published_checkpoint_receipt_is_complete_and_content_addressed() -> None:
    contents = CHECKPOINT_RECEIPT.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == CHECKPOINT_RECEIPT_SHA256
    receipt = cast(dict[str, Any], json.loads(contents))
    coverage = cast(dict[str, Any], receipt["coverage"])
    domain = cast(dict[str, Any], receipt["scale_domain"])

    assert coverage["tensor_count"] == 43 * 256 * 3
    assert coverage["total_scale_bytes"] == 8_657_043_456
    assert domain["minimum"] == 118
    assert domain["maximum"] == 126
    assert domain["unsafe_scale_bytes"] == 0
    assert domain["e8m0_nan_count"] == 0
    assert domain["branchless_fold_admitted"] is True
    assert domain["scale_content_sha256"] == (
        "64a6d022552946e6c6bbe3e31f89370582c3df362aba78442da48f60d89e663f"
    )
