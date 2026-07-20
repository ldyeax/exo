#!/usr/bin/env python3
"""Run the pinned two-stage, dwagon-local GLM-4.7 Flash diagnostic.

Both stages use one NUMA-matched RTX 3090 and all 56 physical cores assigned
to that socket. The run requires semantic sanity before measuring the canonical
1024/32 and 128/128 workloads. NCCL transport selection remains automatic so
the local ranks may use NVLink/P2P rather than an InfiniBand network backend.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import signal
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path
from types import FrameType
from typing import Final, cast

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.shared.types.common import Host  # noqa: E402
from exo.shared.types.worker.sglang_kt import (  # noqa: E402
    SglangKtLaunchPlan,
    SglangKtStageSpec,
)
from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CONTEXT_LENGTH,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_LAYER_COUNT,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_PP2_LOCAL_DEFAULT_RESIDENT_GPU_EXPERTS,
    GLM_4_7_FLASH_PP2_LOCAL_DIAGNOSTIC_TARGET_PROFILE,
    GLM_4_7_FLASH_PP2_LOCAL_MAX_RESIDENT_GPU_EXPERTS,
    GLM_4_7_FLASH_PP2_LOCAL_PIPELINE_LAYER_PARTITION,
    GLM_4_7_FLASH_PP2_LOCAL_PIPELINE_LAYER_PARTITIONS,
    GLM_4_7_FLASH_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_pp2_local_diagnostic_process_launch_specs,
)
from exo.worker.sglang_kt.model_contract import (  # noqa: E402
    SglangKtModelContractError,
    load_sglang_kt_model_contract,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    SglangKtReceiptFileError,
    hash_sglang_kt_bound_file,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from scripts import install_sglang_kt_runtime as runtime_install  # noqa: E402
from scripts import run_sglang_kt_glm47_pp3_diagnostic as pipeline  # noqa: E402
from scripts.sglang_kt_glm47_serving_client import (  # noqa: E402
    Glm47NativeServingClient,
    prepare_glm47_serving_workload,
    run_glm47_serving_sanity,
    run_glm47_serving_workload,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

DEFAULT_DWAGON_IP: Final = "192.168.40.24"
DEFAULT_DISTRIBUTED_PORT: Final = 62500
DEFAULT_STAGE_PORTS: Final = (62510, 62511)
_LOG_MAXIMUM_BYTES: Final = 256 * 1024 * 1024
_RECEIPT_MAXIMUM_BYTES: Final = 4 * 1024 * 1024
_OWNERSHIP_JOURNAL_NAME: Final = "pp2-local-ownership-journal.json"
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)
_INSTALL_RECEIPT_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "install_id",
        "installer_sha256",
        "build",
        "base_runtime",
        "layout",
        "environment",
        "commands",
        "installed_distributions",
        "completed_at_utc",
    }
)
_EXPECTED_INSTALLED_DISTRIBUTIONS: Final = frozenset(
    {"kt-kernel", "ktransformers", "sglang-kt"}
)
_MANAGED_SIGNALS: Final = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_MODEL_STAGE_EVIDENCE_DIRECTORY: Final = Path(
    "/var/lib/exo/benchmarks/glm47-bf16-local-stage-dwagon-20260719-v1"
)
_MODEL_STAGE_RESULT_SHA256: Final = (
    "4402e93021291b8138660008b1bd7d27cf77c8dc9c31ca29ac1b51d083ae49d8"
)
_MODEL_STAGE_MANIFEST_SHA256: Final = (
    "a60105f975aa667e9372205f53345c4571f4c11c42bb2829aaf3258c97dce0b6"
)
_MODEL_STAGE_RUNTIME_SHA256: Final = (
    "d82b90508b00ef0a1399e960a43de51f4a979a695426fe58b14dd98d2d561344"
)


class Pp2LocalDiagnosticError(RuntimeError):
    """Raised when the local PP2 run cannot produce complete evidence."""


class Pp2LocalDiagnosticSignalError(Pp2LocalDiagnosticError):
    """Raised at a safe checkpoint after a managed termination signal."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(f"received managed signal {signal_number}")
        self.signal_number = signal_number


@dataclass(slots=True)
class _ManagedSignalState:
    signal_number: int | None = None
    cleanup_started: bool = False
    defer_depth: int = 0

    def handle(self, signal_number: int, _frame: FrameType | None) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number
        if not self.cleanup_started and self.defer_depth == 0:
            raise Pp2LocalDiagnosticSignalError(self.signal_number)

    def checkpoint(self) -> None:
        if self.signal_number is not None and not self.cleanup_started:
            raise Pp2LocalDiagnosticSignalError(self.signal_number)

    @contextmanager
    def defer(self) -> Iterator[None]:
        self.defer_depth += 1
        try:
            yield
        finally:
            self.defer_depth -= 1
        self.checkpoint()

    def begin_cleanup(self) -> None:
        self.cleanup_started = True


@dataclass(frozen=True, slots=True)
class Pp2LocalDiagnosticConfig:
    run_id: str
    result_directory: Path
    dwagon_runtime_python: str
    dwagon_runtime_install_receipt: Path
    dwagon_runtime_install_receipt_sha256: str
    dwagon_model_path: str
    local_source_directory: str
    dwagon_ip: str
    dwagon_socket_interface: str
    distributed_port: int
    stage_ports: tuple[int, int]
    pipeline_layer_partition: tuple[int, int]
    resident_gpu_experts: int
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    cleanup_timeout_seconds: float
    warmup_count: int
    sample_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Pp2LocalDiagnosticError(f"{description} must be a JSON object")
    object_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in object_mapping):
        raise Pp2LocalDiagnosticError(f"{description} must be a JSON object")
    return cast(dict[str, object], object_mapping)


def _required_string(values: dict[str, object], key: str, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value or len(value) > 16_384:
        raise Pp2LocalDiagnosticError(
            f"{description}.{key} must be a bounded nonempty string"
        )
    return value


def _required_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Pp2LocalDiagnosticError(f"{description} must be a SHA-256 digest")
    return value


def _json_compatible(value: object) -> object:
    return cast(
        object,
        json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
    )


def _normalized_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _verify_installed_distribution_record(
    *,
    install_root: Path,
    site_packages: Path,
    distribution: dict[str, object],
    expected_version: str,
) -> tuple[str, int]:
    name = _normalized_distribution(
        _required_string(distribution, "distribution", "installed distribution")
    )
    if set(distribution) != {
        "distribution",
        "version",
        "metadata_path",
        "record_sha256",
    }:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} does not match schema 1"
        )
    version = _required_string(
        distribution, "version", f"installed distribution {name}"
    )
    if version != expected_version:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} has version {version}, expected "
            f"{expected_version}"
        )
    metadata_path = Path(
        _required_string(
            distribution, "metadata_path", f"installed distribution {name}"
        )
    )
    expected_record_sha256 = _required_sha256(
        distribution.get("record_sha256"),
        f"installed distribution {name} RECORD digest",
    )
    try:
        resolved_install_root = install_root.resolve(strict=True)
        resolved_site_packages = site_packages.resolve(strict=True)
        resolved_metadata = metadata_path.resolve(strict=True)
    except OSError as error:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} path is unavailable: {error}"
        ) from error
    if (
        resolved_install_root != install_root
        or resolved_site_packages != site_packages
        or not resolved_metadata.is_relative_to(resolved_site_packages)
        or not resolved_metadata.is_relative_to(resolved_install_root)
        or resolved_metadata.name != "METADATA"
        or metadata_path != resolved_metadata
    ):
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} metadata escapes the immutable install"
        )
    metadata = read_sglang_kt_bound_file(
        metadata_path,
        maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
    )
    parsed_metadata = BytesParser().parsebytes(metadata.contents)
    if (
        _normalized_distribution(str(parsed_metadata["Name"] or "")) != name
        or str(parsed_metadata["Version"] or "") != version
    ):
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} metadata identity differs from receipt"
        )

    record_path = metadata_path.parent / "RECORD"
    record = read_sglang_kt_bound_file(
        record_path,
        maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
    )
    if record.sha256 != expected_record_sha256:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} RECORD hash changed"
        )
    try:
        record_text = record.contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} RECORD is not UTF-8"
        ) from error

    verified_paths: set[Path] = set()
    for row in csv.reader(record_text.splitlines()):
        if len(row) != 3 or not row[0]:
            raise Pp2LocalDiagnosticError(
                f"installed distribution {name} RECORD row is invalid"
            )
        recorded_path, recorded_hash, recorded_size = row
        relative = Path(recorded_path)
        parts = relative.parts
        if relative.is_absolute():
            raise Pp2LocalDiagnosticError(
                f"installed distribution {name} RECORD contains an absolute path"
            )
        if parts[:3] == ("..", "..", "bin"):
            lexical_candidate = site_packages / "bin" / Path(*parts[3:])
        else:
            if ".." in parts:
                raise Pp2LocalDiagnosticError(
                    f"installed distribution {name} RECORD escapes site-packages"
                )
            lexical_candidate = site_packages / relative
        try:
            lexical_status = lexical_candidate.lstat()
            candidate = lexical_candidate.resolve(strict=True)
        except OSError as error:
            raise Pp2LocalDiagnosticError(
                f"installed distribution {name} RECORD path is unavailable: {error}"
            ) from error
        if (
            stat.S_ISLNK(lexical_status.st_mode)
            or not stat.S_ISREG(lexical_status.st_mode)
            or not candidate.is_relative_to(resolved_install_root)
            or candidate in verified_paths
        ):
            raise Pp2LocalDiagnosticError(
                f"installed distribution {name} RECORD path is not an owned file"
            )
        verified_paths.add(candidate)

        expected_size: int | None = None
        if recorded_size:
            try:
                expected_size = int(recorded_size)
            except ValueError as error:
                raise Pp2LocalDiagnosticError(
                    f"installed distribution {name} RECORD size is invalid"
                ) from error
            if expected_size < 0:
                raise Pp2LocalDiagnosticError(
                    f"installed distribution {name} RECORD size is negative"
                )
        observed = hash_sglang_kt_bound_file(
            candidate,
            expected_size_bytes=expected_size,
        )
        if recorded_hash:
            algorithm, separator, encoded_digest = recorded_hash.partition("=")
            if algorithm != "sha256" or not separator or not encoded_digest:
                raise Pp2LocalDiagnosticError(
                    f"installed distribution {name} RECORD hash is unsupported"
                )
            padding = "=" * (-len(encoded_digest) % 4)
            try:
                expected_digest = base64.b64decode(
                    encoded_digest + padding,
                    altchars=b"-_",
                    validate=True,
                ).hex()
            except (ValueError, TypeError) as error:
                raise Pp2LocalDiagnosticError(
                    f"installed distribution {name} RECORD digest is invalid"
                ) from error
            if observed.sha256 != expected_digest:
                raise Pp2LocalDiagnosticError(
                    f"installed runtime file changed: {recorded_path}"
                )
        elif candidate != record_path:
            raise Pp2LocalDiagnosticError(
                f"installed distribution {name} RECORD omits a file digest"
            )
    if metadata_path not in verified_paths or record_path not in verified_paths:
        raise Pp2LocalDiagnosticError(
            f"installed distribution {name} RECORD is incomplete"
        )
    return name, len(verified_paths)


def _read_bound_json(
    path: Path,
    expected_sha256: str,
    description: str,
) -> tuple[dict[str, object], int]:
    try:
        bound = read_sglang_kt_bound_file(
            path,
            maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
        )
        document = _json_object(
            parse_sglang_kt_strict_json(bound.contents),
            description,
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(f"cannot bind {description}: {error}") from error
    if bound.sha256 != expected_sha256:
        raise Pp2LocalDiagnosticError(f"{description} SHA-256 changed")
    return document, len(bound.contents)


def _verify_pinned_model_contract(model_path_value: str) -> JsonObject:
    if model_path_value != pipeline.DEFAULT_DWAGON_MODEL_PATH:
        raise Pp2LocalDiagnosticError(
            "dwagon_model_path must be the pinned GLM-4.7-Flash contract path"
        )
    model_path = Path(model_path_value)
    try:
        resolved_model_path = model_path.resolve(strict=True)
    except OSError as error:
        raise Pp2LocalDiagnosticError(
            f"pinned GLM-4.7-Flash model path is unavailable: {error}"
        ) from error
    if (
        resolved_model_path != model_path
        or model_path.is_symlink()
        or not model_path.is_dir()
    ):
        raise Pp2LocalDiagnosticError(
            "pinned GLM-4.7-Flash model path must be an exact non-symlink directory"
        )

    result_path = _MODEL_STAGE_EVIDENCE_DIRECTORY / "benchmark-result.json"
    manifest_path = _MODEL_STAGE_EVIDENCE_DIRECTORY / "manifest.json"
    runtime_path = _MODEL_STAGE_EVIDENCE_DIRECTORY / "runtime-metadata.json"
    result, result_size = _read_bound_json(
        result_path,
        _MODEL_STAGE_RESULT_SHA256,
        "local-NVMe model-stage result",
    )
    manifest, manifest_size = _read_bound_json(
        manifest_path,
        _MODEL_STAGE_MANIFEST_SHA256,
        "local-NVMe model-stage manifest",
    )
    runtime_metadata, runtime_size = _read_bound_json(
        runtime_path,
        _MODEL_STAGE_RUNTIME_SHA256,
        "local-NVMe model-stage runtime metadata",
    )
    if (
        result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("completed_normally") is not True
        or result.get("cleanup_succeeded") is not True
        or result.get("local_installed") is not True
        or result.get("acquisition") != "local_copy_from_existing_remote_source"
        or manifest.get("status") != "completed"
        or manifest.get("command_return_code") != 0
        or manifest.get("return_code") != 0
        or manifest.get("cleanup_succeeded") is not True
        or manifest.get("cleanup_forced") is not False
        or manifest.get("benchmark_result") != result
        or manifest.get("runtime_metadata") != runtime_metadata
        or manifest.get("result_directory") != str(_MODEL_STAGE_EVIDENCE_DIRECTORY)
        or runtime_metadata.get("schema_version") != 1
        or runtime_metadata.get("run_id") != result.get("run_id")
    ):
        raise Pp2LocalDiagnosticError(
            "local-NVMe model-stage evidence is not one complete publication"
        )

    paths = _json_object(result.get("paths"), "model-stage paths")
    model = _json_object(result.get("model"), "model-stage model")
    local_verification = _json_object(
        result.get("local_verification"), "local model-stage verification"
    )
    if (
        paths.get("dwagon") != model_path_value
        or model.get("model_id") != str(GLM_4_7_FLASH_BF16_MODEL_ID)
        or model.get("revision") != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or result.get("source_verification") != local_verification
        or result.get("remote_verification") != local_verification
        or local_verification.get("model_id") != str(GLM_4_7_FLASH_BF16_MODEL_ID)
        or local_verification.get("revision") != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or local_verification.get("model_contract_sha256")
        != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
        or local_verification.get("shard_count") != 48
    ):
        raise Pp2LocalDiagnosticError(
            "local-NVMe model-stage evidence differs from the launch model"
        )
    staged_manifest = _json_object(
        local_verification.get("manifest"), "local model-stage file manifest"
    )
    if (
        len(staged_manifest) != 117
        or staged_manifest.get("config.json") != GLM_4_7_FLASH_BF16_CONFIG_SHA256
    ):
        raise Pp2LocalDiagnosticError(
            "local-NVMe model-stage manifest is not the pinned snapshot"
        )

    staging_config = _json_object(
        result.get("staging_config"), "model-stage configuration"
    )
    acquisition = _json_object(
        staging_config.get("acquisition"), "model-stage acquisition"
    )
    source_contract = _json_object(
        acquisition.get("source_model_contract"),
        "model-stage source model contract",
    )
    contract_path = Path(
        _required_string(
            source_contract,
            "path",
            "model-stage source model contract",
        )
    )
    if (
        staging_config.get("local_destination") != model_path_value
        or source_contract.get("sha256") != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    ):
        raise Pp2LocalDiagnosticError(
            "model-stage configuration does not bind the local model path"
        )
    try:
        loaded_contract = load_sglang_kt_model_contract(
            contract_path,
            expected_contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        )
    except SglangKtModelContractError as error:
        raise Pp2LocalDiagnosticError(
            f"cannot load pinned model contract: {error}"
        ) from error
    contract = loaded_contract.contract
    if (
        str(contract.model_id) != str(GLM_4_7_FLASH_BF16_MODEL_ID)
        or contract.revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or contract.ktransformers_method != "BF16"
        or local_verification.get("indexed_bytes") != contract.index_metadata_total_size
        or model.get("expected_indexed_bytes") != contract.index_metadata_total_size
    ):
        raise Pp2LocalDiagnosticError(
            "model contract identity differs from local publication evidence"
        )

    contract_files = {file.path: file for file in contract.files}
    config_contract = contract_files["config.json"]
    index_contract = contract_files["model.safetensors.index.json"]
    try:
        config_file = read_sglang_kt_bound_file(
            model_path / config_contract.path,
            maximum_bytes=1024 * 1024,
        )
        index_file = read_sglang_kt_bound_file(
            model_path / index_contract.path,
            maximum_bytes=64 * 1024 * 1024,
        )
        parse_sglang_kt_strict_json(config_file.contents)
        index_document = _json_object(
            parse_sglang_kt_strict_json(index_file.contents),
            "local safetensors index",
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(
            f"cannot bind local model metadata: {error}"
        ) from error
    if (
        len(config_file.contents) != config_contract.size_bytes
        or config_file.sha256 != config_contract.sha256
        or len(index_file.contents) != index_contract.size_bytes
        or index_file.sha256 != index_contract.sha256
        or staged_manifest.get("model.safetensors.index.json") != index_file.sha256
    ):
        raise Pp2LocalDiagnosticError(
            "local model config/index differs from the pinned contract"
        )
    index_metadata = _json_object(
        index_document.get("metadata"), "local safetensors index metadata"
    )
    weight_map = _json_object(
        index_document.get("weight_map"), "local safetensors weight map"
    )
    shard_contracts = tuple(
        file for file in contract.files if file.role == "weight_shard"
    )
    shard_names = tuple(sorted(file.path for file in shard_contracts))
    weight_map_values = tuple(weight_map.values())
    if not all(isinstance(value, str) for value in weight_map_values):
        raise Pp2LocalDiagnosticError(
            "local safetensors index has a non-string shard name"
        )
    mapped_shards = tuple(sorted(set(cast(tuple[str, ...], weight_map_values))))
    if (
        set(index_document) != {"metadata", "weight_map"}
        or index_metadata.get("total_size") != contract.index_metadata_total_size
        or len(weight_map) != contract.weight_map_entries
        or mapped_shards != shard_names
    ):
        raise Pp2LocalDiagnosticError(
            "local safetensors index structure differs from the pinned contract"
        )
    physical_weight_bytes = 0
    for shard in shard_contracts:
        shard_path = model_path / shard.path
        try:
            shard_status = shard_path.lstat()
        except OSError as error:
            raise Pp2LocalDiagnosticError(
                f"local model shard is unavailable: {shard.path}: {error}"
            ) from error
        if (
            stat.S_ISLNK(shard_status.st_mode)
            or not stat.S_ISREG(shard_status.st_mode)
            or shard_status.st_size != shard.size_bytes
        ):
            raise Pp2LocalDiagnosticError(
                f"local model shard identity changed: {shard.path}"
            )
        physical_weight_bytes += shard_status.st_size
    if physical_weight_bytes != contract.physical_weight_bytes:
        raise Pp2LocalDiagnosticError(
            "local model shard sizes differ from the pinned contract"
        )
    return {
        "model_id": GLM_4_7_FLASH_BF16_MODEL_ID,
        "model_revision": GLM_4_7_FLASH_BF16_MODEL_REVISION,
        "path": str(model_path),
        "model_contract": loaded_contract.model_dump(mode="json"),
        "config_sha256": config_file.sha256,
        "index_sha256": index_file.sha256,
        "weight_map_entries": contract.weight_map_entries,
        "shard_count": len(shard_contracts),
        "physical_weight_bytes": physical_weight_bytes,
        "verification_mode": "staging_receipt_plus_metadata_and_shard_size",
        "full_shard_content_rehash": False,
        "model_stage_evidence": {
            "directory": str(_MODEL_STAGE_EVIDENCE_DIRECTORY),
            "result": {
                "path": str(result_path),
                "size_bytes": result_size,
                "sha256": _MODEL_STAGE_RESULT_SHA256,
            },
            "manifest": {
                "path": str(manifest_path),
                "size_bytes": manifest_size,
                "sha256": _MODEL_STAGE_MANIFEST_SHA256,
            },
            "runtime_metadata": {
                "path": str(runtime_path),
                "size_bytes": runtime_size,
                "sha256": _MODEL_STAGE_RUNTIME_SHA256,
            },
        },
    }


def _verify_runtime_and_model_contract(
    config: Pp2LocalDiagnosticConfig,
) -> JsonObject:
    model_contract = _verify_pinned_model_contract(config.dwagon_model_path)

    try:
        receipt_file = read_sglang_kt_bound_file(
            config.dwagon_runtime_install_receipt,
            maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
        )
        document = _json_object(
            parse_sglang_kt_strict_json(receipt_file.contents),
            "runtime install receipt",
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(
            f"cannot bind runtime install receipt: {error}"
        ) from error
    if receipt_file.sha256 != config.dwagon_runtime_install_receipt_sha256:
        raise Pp2LocalDiagnosticError(
            "runtime install receipt SHA-256 differs from the required binding"
        )
    if frozenset(document) != _INSTALL_RECEIPT_KEYS:
        raise Pp2LocalDiagnosticError(
            "runtime install receipt does not match complete schema 1"
        )
    if document.get("schema_version") != 1 or document.get("status") != (
        "install_complete"
    ):
        raise Pp2LocalDiagnosticError(
            "runtime install receipt is not a completed schema-1 install"
        )
    completed_at = document.get("completed_at_utc")
    if not isinstance(completed_at, str):
        raise Pp2LocalDiagnosticError(
            "runtime install receipt lacks a completion timestamp"
        )
    try:
        completion_time = datetime.fromisoformat(completed_at)
    except ValueError as error:
        raise Pp2LocalDiagnosticError(
            "runtime install completion timestamp is invalid"
        ) from error
    if completion_time.utcoffset() is None:
        raise Pp2LocalDiagnosticError(
            "runtime install completion timestamp must include a timezone"
        )

    install_id = _required_sha256(document.get("install_id"), "runtime install ID")
    layout = _json_object(document.get("layout"), "runtime install layout")
    build = _json_object(document.get("build"), "runtime install build")
    base_runtime = _json_object(
        document.get("base_runtime"), "runtime install base runtime"
    )
    sglang_revision = _required_string(
        build, "sglang_revision", "runtime install build"
    )
    ktransformers_revision = _required_string(
        build, "ktransformers_revision", "runtime install build"
    )
    if (
        sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    ):
        raise Pp2LocalDiagnosticError(
            "runtime install revisions differ from the GLM-4.7 launch contract"
        )

    build_receipt_path = Path(
        _required_string(build, "receipt_path", "runtime install build")
    )
    build_receipt_sha256 = _required_sha256(
        build.get("receipt_sha256"), "runtime build receipt digest"
    )
    try:
        bound_build_receipt = read_sglang_kt_bound_file(
            build_receipt_path,
            maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
        )
        _json_object(
            parse_sglang_kt_strict_json(bound_build_receipt.contents),
            "runtime build receipt",
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(
            f"cannot bind runtime build receipt: {error}"
        ) from error
    if bound_build_receipt.sha256 != build_receipt_sha256:
        raise Pp2LocalDiagnosticError(
            "runtime build receipt content differs from install receipt"
        )

    output_root = Path(
        _required_string(layout, "output_root", "runtime install layout")
    )
    base_python = Path(
        _required_string(base_runtime, "python_path", "runtime install base runtime")
    )
    base_site_packages = Path(
        _required_string(
            base_runtime,
            "site_packages",
            "runtime install base runtime",
        )
    )
    try:
        plan = runtime_install.plan_runtime_install(
            build_receipt_path,
            base_python,
            base_site_packages,
            output_root,
        )
    except (OSError, runtime_install.RuntimeInstallError) as error:
        raise Pp2LocalDiagnosticError(
            f"runtime install inputs no longer satisfy their receipt: {error}"
        ) from error
    expected_plan = _json_object(
        _json_compatible(asdict(plan)),
        "recomputed runtime install plan",
    )
    for key in (
        "install_id",
        "installer_sha256",
        "build",
        "base_runtime",
        "layout",
        "environment",
        "commands",
    ):
        if document.get(key) != expected_plan.get(key):
            raise Pp2LocalDiagnosticError(
                f"runtime install receipt {key} differs from its content-addressed plan"
            )
    if plan.install_id != install_id:
        raise Pp2LocalDiagnosticError(
            "runtime install ID differs from its content-addressed plan"
        )

    recorded_receipt_path = Path(
        _required_string(layout, "receipt", "runtime install layout")
    )
    install_root = Path(
        _required_string(layout, "install_root", "runtime install layout")
    )
    recorded_python = Path(_required_string(layout, "python", "runtime install layout"))
    configured_python = Path(config.dwagon_runtime_python)
    try:
        resolved_python = recorded_python.resolve(strict=True)
        resolved_configured_python = configured_python.resolve(strict=True)
        resolved_receipt_path = recorded_receipt_path.resolve(strict=True)
    except OSError as error:
        raise Pp2LocalDiagnosticError(
            f"runtime install layout path is unavailable: {error}"
        ) from error
    if (
        recorded_receipt_path != receipt_file.path
        or resolved_receipt_path != receipt_file.path
        or install_root != receipt_file.path.parent
        or install_root.name != install_id
        or configured_python != recorded_python
        or resolved_configured_python != resolved_python
        or not os.access(recorded_python, os.X_OK)
    ):
        raise Pp2LocalDiagnosticError(
            "runtime install receipt does not bind the selected Python/install paths"
        )
    expected_resolved_python = Path(
        _required_string(
            base_runtime,
            "resolved_python_path",
            "runtime install base runtime",
        )
    )
    expected_python_sha256 = _required_sha256(
        base_runtime.get("python_sha256"), "runtime Python digest"
    )
    observed_python = hash_sglang_kt_bound_file(resolved_python)
    if (
        resolved_python != expected_resolved_python
        or observed_python.sha256 != expected_python_sha256
    ):
        raise Pp2LocalDiagnosticError(
            "runtime Python target differs from the install receipt"
        )

    base_runtime_pth = Path(
        _required_string(layout, "base_runtime_pth", "runtime install layout")
    )
    expected_pth_sha256 = _required_sha256(
        layout.get("base_runtime_pth_sha256"), "base-runtime .pth digest"
    )
    try:
        pth = read_sglang_kt_bound_file(
            base_runtime_pth,
            maximum_bytes=16_384,
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(
            f"cannot bind base-runtime .pth: {error}"
        ) from error
    expected_pth = f"{base_site_packages}\n".encode("ascii")
    if pth.contents != expected_pth or pth.sha256 != expected_pth_sha256:
        raise Pp2LocalDiagnosticError(
            "base-runtime .pth differs from the install receipt"
        )

    installed_values = document.get("installed_distributions")
    if not isinstance(installed_values, list):
        raise Pp2LocalDiagnosticError(
            "runtime install receipt installed_distributions must be an array"
        )
    installed_values = cast(list[object], installed_values)
    expected_versions = {
        wheel.distribution: wheel.version for wheel in plan.build.wheels
    }
    if frozenset(expected_versions) != _EXPECTED_INSTALLED_DISTRIBUTIONS:
        raise Pp2LocalDiagnosticError(
            "runtime install plan has an unexpected distribution set"
        )
    installed_file_counts: dict[str, int] = {}
    site_packages = Path(plan.layout.site_packages)
    for value in installed_values:
        distribution = _json_object(value, "installed distribution")
        raw_name = _required_string(
            distribution, "distribution", "installed distribution"
        )
        name = _normalized_distribution(raw_name)
        expected_version = expected_versions.get(name)
        if expected_version is None or name in installed_file_counts:
            raise Pp2LocalDiagnosticError(
                f"runtime install receipt has unexpected distribution {name}"
            )
        verified_name, file_count = _verify_installed_distribution_record(
            install_root=install_root,
            site_packages=site_packages,
            distribution=distribution,
            expected_version=expected_version,
        )
        installed_file_counts[verified_name] = file_count
    if frozenset(installed_file_counts) != _EXPECTED_INSTALLED_DISTRIBUTIONS:
        raise Pp2LocalDiagnosticError(
            "runtime install receipt does not bind all installed distributions"
        )

    try:
        final_receipt = read_sglang_kt_bound_file(
            receipt_file.path,
            maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
        )
        final_build_receipt = read_sglang_kt_bound_file(
            build_receipt_path,
            maximum_bytes=_RECEIPT_MAXIMUM_BYTES,
        )
    except SglangKtReceiptFileError as error:
        raise Pp2LocalDiagnosticError(
            f"runtime receipt changed during admission: {error}"
        ) from error
    if final_receipt != receipt_file or final_build_receipt != bound_build_receipt:
        raise Pp2LocalDiagnosticError("runtime receipt changed during admission")
    return cast(
        JsonObject,
        {
            "install_receipt": {
                "path": str(receipt_file.path),
                "size_bytes": len(receipt_file.contents),
                "sha256": receipt_file.sha256,
            },
            "install_id": install_id,
            "install_root": str(install_root),
            "runtime_python": {
                "path": str(recorded_python),
                "resolved_path": str(resolved_python),
                "sha256": observed_python.sha256,
            },
            "build_receipt": {
                "path": str(build_receipt_path),
                "size_bytes": len(bound_build_receipt.contents),
                "sha256": bound_build_receipt.sha256,
            },
            "build_id": plan.build.build_id,
            "sglang_revision": sglang_revision,
            "ktransformers_revision": ktransformers_revision,
            "installed_distribution_file_counts": installed_file_counts,
            "model": model_contract,
        },
    )


def _pipeline_ranges(
    partition: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if partition not in GLM_4_7_FLASH_PP2_LOCAL_PIPELINE_LAYER_PARTITIONS:
        raise Pp2LocalDiagnosticError("pipeline_layer_partition must be 24,23 or 23,24")
    first_end = partition[0]
    final_end = first_end + partition[1]
    if final_end != GLM_4_7_FLASH_LAYER_COUNT:
        raise Pp2LocalDiagnosticError(
            "pipeline_layer_partition must cover all GLM-4.7 Flash layers"
        )
    return ((0, first_end), (first_end, final_end))


def build_pp2_local_plan(config: Pp2LocalDiagnosticConfig) -> SglangKtLaunchPlan:
    """Build the exact NUMA 0/GPU 0 -> NUMA 1/GPU 1 local pipeline."""

    pipeline_ranges = _pipeline_ranges(config.pipeline_layer_partition)
    placements = (
        (
            pipeline.DWAGON_STAGE_ZERO_GPU,
            pipeline.DWAGON_STAGE_ZERO_CPUS,
            0,
        ),
        (
            pipeline.DWAGON_STAGE_ONE_GPU,
            pipeline.DWAGON_STAGE_ONE_CPUS,
            1,
        ),
    )
    stages = tuple(
        SglangKtStageSpec(
            pipeline_rank=rank,
            start_layer=pipeline_ranges[rank][0],
            end_layer=pipeline_ranges[rank][1],
            node_id=pipeline.DWAGON_NODE_ID,
            gpu_uuid=gpu_uuid,
            service_endpoint=Host(
                ip=config.dwagon_ip,
                port=config.stage_ports[rank],
            ),
            model_path=config.dwagon_model_path,
            ktransformers_weight_path=config.dwagon_model_path,
            cpu_cores=cpu_cores,
            memory_nodes=(memory_node,),
            cpu_infer_threads=len(cpu_cores),
            threadpool_count=1,
            ktransformers_method="BF16",
            resident_gpu_experts=config.resident_gpu_experts,
            max_deferred_experts_per_token=0,
            hca_devices=(),
        )
        for rank, (gpu_uuid, cpu_cores, memory_node) in enumerate(placements)
    )
    return SglangKtLaunchPlan(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        target_profile=GLM_4_7_FLASH_PP2_LOCAL_DIAGNOSTIC_TARGET_PROFILE,
        total_layers=GLM_4_7_FLASH_LAYER_COUNT,
        context_length=GLM_4_7_FLASH_CONTEXT_LENGTH,
        max_total_tokens=GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(
            ip=config.dwagon_ip,
            port=config.distributed_port,
        ),
        rank_zero_endpoint=stages[0].service_endpoint,
        stages=stages,
    )


def build_pp2_local_process_specs(
    config: Pp2LocalDiagnosticConfig,
) -> tuple[SglangKtProcessLaunchSpec, ...]:
    return build_glm_4_7_flash_bf16_pp2_local_diagnostic_process_launch_specs(
        build_pp2_local_plan(config),
        config.dwagon_runtime_python,
    )


def _configuration_receipt(config: Pp2LocalDiagnosticConfig) -> JsonObject:
    return {
        "host_name": "dwagon",
        "scope": "single_host",
        "pipeline_parallel_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_layer_partition": list(config.pipeline_layer_partition),
        "resident_gpu_experts_per_stage": config.resident_gpu_experts,
        "nccl_transport_policy": "automatic_local_p2p_nvlink_allowed",
    }


def _process_receipt(running: pipeline.RunningStage) -> JsonObject:
    owned = asdict(running.owned)
    owned["owner_token"] = hashlib.sha256(
        running.owned.owner_token.encode()
    ).hexdigest()
    return cast(JsonObject, owned)


def _log_receipts(config: Pp2LocalDiagnosticConfig) -> list[JsonValue]:
    receipts: list[JsonValue] = []
    for rank in range(2):
        path = config.result_directory / f"rank-{rank}.log"
        if not path.is_file():
            continue
        status = path.stat()
        if status.st_size > _LOG_MAXIMUM_BYTES:
            raise Pp2LocalDiagnosticError(
                f"rank {rank} log exceeds diagnostic size bound"
            )
        receipts.append(
            {
                "rank": rank,
                "path": str(path),
                "size_bytes": status.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return receipts


def _write_receipt(config: Pp2LocalDiagnosticConfig, payload: JsonObject) -> None:
    path = config.result_directory / "pp2-local-diagnostic-result.json"
    temporary = config.result_directory / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ownership_journal_path(config: Pp2LocalDiagnosticConfig) -> Path:
    return config.result_directory / _OWNERSHIP_JOURNAL_NAME


def _write_ownership_journal(
    config: Pp2LocalDiagnosticConfig,
    running: list[pipeline.RunningStage],
) -> None:
    if not running:
        raise Pp2LocalDiagnosticError(
            "ownership journal requires at least one started stage"
        )
    destination = _ownership_journal_path(config)
    temporary = config.result_directory / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    payload = {
        "schema_version": 1,
        "status": "active",
        "run_id": config.run_id,
        "updated_at_utc": _utc_now(),
        "started_rank_count": len(running),
        "processes": [asdict(stage.owned) for stage in running],
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("ownership journal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
        _fsync_directory(config.result_directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _clear_ownership_journal(config: Pp2LocalDiagnosticConfig) -> None:
    _ownership_journal_path(config).unlink()
    _fsync_directory(config.result_directory)


def run_diagnostic(config: Pp2LocalDiagnosticConfig) -> JsonObject:
    """Launch, measure, clean up, and publish one local PP2 receipt."""

    runtime_contract = _verify_runtime_and_model_contract(config)
    specs = build_pp2_local_process_specs(config)
    config.result_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    owner_token = uuid.uuid4().hex
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    running: list[pipeline.RunningStage] = []
    readiness: tuple[JsonObject, ...] = ()
    server_info: JsonObject | None = None
    sanity: JsonObject | None = None
    workloads: list[JsonValue] = []
    summaries: list[JsonValue] = []
    failure: BaseException | None = None
    cleanup: list[JsonValue] = []
    cleanup_complete = False
    journal_created = False
    journal_cleared = False
    signal_state = _ManagedSignalState()
    previous_handlers: dict[
        signal.Signals,
        signal.Handlers | int | Callable[[int, FrameType | None], object] | None,
    ] = {}
    try:
        for managed_signal in _MANAGED_SIGNALS:
            previous_handlers[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, signal_state.handle)
        try:
            for spec in specs:
                with signal_state.defer():
                    running.append(
                        pipeline.start_local_stage(spec, config, owner_token)
                    )
                    _write_ownership_journal(config, running)
                    journal_created = True
            readiness = pipeline.wait_for_all_stages(
                specs,
                running,
                config.readiness_timeout_seconds,
            )
            signal_state.checkpoint()
            rank_zero = specs[0]
            with Glm47NativeServingClient(
                f"http://{rank_zero.service_endpoint}",
                timeout_seconds=config.request_timeout_seconds,
            ) as client:
                server_info = cast(
                    JsonObject, client.server_info().model_dump(mode="json")
                )
                sanity_evidence = run_glm47_serving_sanity(
                    client,
                    config.dwagon_model_path,
                )
                sanity = cast(JsonObject, sanity_evidence.model_dump(mode="json"))
                if not pipeline.all_stages_alive(running):
                    raise Pp2LocalDiagnosticError(
                        "a stage exited during semantic sanity"
                    )
                signal_state.checkpoint()
                for kind in ("prefill", "decode"):
                    workload = run_glm47_serving_workload(
                        client,
                        prepare_glm47_serving_workload(kind),
                        warmup_count=config.warmup_count,
                        sample_count=config.sample_count,
                    )
                    if not pipeline.all_stages_alive(running):
                        raise Pp2LocalDiagnosticError(
                            f"a stage exited during the {kind} workload"
                        )
                    signal_state.checkpoint()
                    workloads.append(cast(JsonObject, workload.model_dump(mode="json")))
                    summaries.append(pipeline.summarize_workload(workload))
        except BaseException as error:
            failure = (
                Pp2LocalDiagnosticError(str(error))
                if isinstance(error, pipeline.Pp3DiagnosticError)
                else error
            )
        finally:
            signal_state.begin_cleanup()
            for stage in reversed(running):
                try:
                    receipt = pipeline.stop_local_stage(
                        stage,
                        config.cleanup_timeout_seconds,
                    )
                    cleanup.append(
                        {
                            "rank": stage.owned.rank,
                            **receipt.model_dump(mode="json"),
                        }
                    )
                except BaseException as error:
                    cleanup.append(
                        {
                            "rank": stage.owned.rank,
                            "host_name": stage.owned.host_name,
                            "ownership_verified": False,
                            "terminated": False,
                            "forced": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    if failure is None:
                        failure = error
            cleanup.sort(
                key=lambda item: cast(int, cast(dict[str, object], item)["rank"])
            )
            cleanup_complete = len(cleanup) == len(running) and all(
                cast(dict[str, object], item).get("ownership_verified") is True
                and cast(dict[str, object], item).get("terminated") is True
                for item in cleanup
            )
            if journal_created and cleanup_complete:
                try:
                    _clear_ownership_journal(config)
                    journal_cleared = True
                except BaseException as error:
                    if failure is None:
                        failure = error
    finally:
        for managed_signal, previous_handler in previous_handlers.items():
            signal.signal(managed_signal, previous_handler)

    all_planned_stages_started = len(running) == len(specs)
    journal_path = _ownership_journal_path(config)
    journal_retained = journal_path.exists()
    payload: JsonObject = {
        "schema_version": 1,
        "kind": "glm47_flash_pp2_local_engineering_diagnostic",
        "status": (
            "passed"
            if failure is None
            and all_planned_stages_started
            and cleanup_complete
            and not journal_retained
            else "failed"
        ),
        "performance_comparable": False,
        "profiler": "none",
        "instrumentation": "nccl_info_logging",
        "run_id": config.run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "topology": {
            "scope": "dwagon_local",
            "host_count": 1,
            "pipeline_parallel_size": 2,
            "tensor_parallel_size": 1,
            "cross_host_transport": False,
            "nvlink_p2p_allowed": True,
        },
        "configuration": _configuration_receipt(config),
        "runtime_contract": runtime_contract,
        "plan": cast(JsonObject, specs[0].plan.model_dump(mode="json")),
        "process_specs": [
            cast(JsonObject, spec.model_dump(mode="json")) for spec in specs
        ],
        "processes": [_process_receipt(stage) for stage in running],
        "readiness": list(readiness),
        "server_info": server_info,
        "sanity": sanity,
        "workloads": workloads,
        "benchmark_summary": summaries,
        "planned_rank_count": len(specs),
        "started_rank_count": len(running),
        "all_planned_stages_started": all_planned_stages_started,
        "cleanup": cleanup,
        "cleanup_complete": cleanup_complete,
        "managed_signal": signal_state.signal_number,
        "ownership_journal": {
            "path": str(journal_path),
            "created": journal_created,
            "cleared_after_verified_cleanup": journal_cleared,
            "retained": journal_retained,
        },
        "logs": _log_receipts(config),
        "failure": (
            None if failure is None else f"{type(failure).__name__}: {failure}"
        ),
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    _write_receipt(config, payload)
    if failure is not None:
        raise Pp2LocalDiagnosticError(
            "local PP2 diagnostic failed; evidence is in "
            f"{config.result_directory}: {type(failure).__name__}: {failure}"
        ) from failure
    if not cleanup_complete:
        raise Pp2LocalDiagnosticError(
            "local PP2 cleanup was incomplete; evidence is in "
            f"{config.result_directory}"
        )
    return payload


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _sha256_argument(raw: str) -> str:
    if _SHA256_PATTERN.fullmatch(raw) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return raw


def _pipeline_layer_partition(raw: str) -> tuple[int, int]:
    try:
        partition = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "pipeline layer partition must be 24,23 or 23,24"
        ) from error
    if partition not in GLM_4_7_FLASH_PP2_LOCAL_PIPELINE_LAYER_PARTITIONS:
        raise argparse.ArgumentTypeError(
            "pipeline layer partition must be 24,23 or 23,24"
        )
    return cast(tuple[int, int], partition)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--dwagon-runtime-python", required=True)
    parser.add_argument(
        "--dwagon-runtime-install-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dwagon-runtime-install-receipt-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument(
        "--dwagon-model-path",
        default=pipeline.DEFAULT_DWAGON_MODEL_PATH,
    )
    parser.add_argument(
        "--local-source-directory",
        default=pipeline.DEFAULT_SOURCE_DIRECTORY,
    )
    parser.add_argument("--dwagon-ip", default=DEFAULT_DWAGON_IP)
    parser.add_argument(
        "--dwagon-socket-interface",
        default=pipeline.DEFAULT_DWAGON_SOCKET_INTERFACE,
    )
    parser.add_argument(
        "--distributed-port",
        type=_positive_int,
        default=DEFAULT_DISTRIBUTED_PORT,
    )
    parser.add_argument(
        "--rank-zero-port",
        type=_positive_int,
        default=DEFAULT_STAGE_PORTS[0],
    )
    parser.add_argument(
        "--rank-one-port",
        type=_positive_int,
        default=DEFAULT_STAGE_PORTS[1],
    )
    parser.add_argument(
        "--pipeline-layer-partition",
        type=_pipeline_layer_partition,
        default=GLM_4_7_FLASH_PP2_LOCAL_PIPELINE_LAYER_PARTITION,
        metavar="LAYERS",
        help="two-stage layer counts: 24,23 (default) or 23,24",
    )
    parser.add_argument(
        "--resident-gpu-experts",
        type=_positive_int,
        default=GLM_4_7_FLASH_PP2_LOCAL_DEFAULT_RESIDENT_GPU_EXPERTS,
        help="resident experts per stage, at most 44 (default: 40)",
    )
    parser.add_argument(
        "--readiness-timeout-seconds",
        type=_positive_float,
        default=1800.0,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=900.0,
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument("--warmups", type=_positive_int, default=2)
    parser.add_argument("--samples", type=_positive_int, default=3)
    return parser


def _config_from_arguments(arguments: argparse.Namespace) -> Pp2LocalDiagnosticConfig:
    result_directory = cast(Path, arguments.result_directory).resolve()
    ports = (
        cast(int, arguments.distributed_port),
        cast(int, arguments.rank_zero_port),
        cast(int, arguments.rank_one_port),
    )
    if len(set(ports)) != len(ports) or any(port > 65535 for port in ports):
        raise Pp2LocalDiagnosticError(
            "distributed and service ports must be unique TCP ports"
        )
    run_id = cast(str, arguments.run_id)
    if not run_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_id
    ):
        raise Pp2LocalDiagnosticError(
            "run_id must contain only safe identifier characters"
        )
    resident_gpu_experts = cast(int, arguments.resident_gpu_experts)
    if resident_gpu_experts > GLM_4_7_FLASH_PP2_LOCAL_MAX_RESIDENT_GPU_EXPERTS:
        raise Pp2LocalDiagnosticError("resident_gpu_experts must be between 1 and 44")
    runtime_install_receipt = cast(Path, arguments.dwagon_runtime_install_receipt)
    if not runtime_install_receipt.is_absolute() or runtime_install_receipt != Path(
        os.path.normpath(runtime_install_receipt)
    ):
        raise Pp2LocalDiagnosticError(
            "dwagon_runtime_install_receipt must be an absolute normalized path"
        )
    return Pp2LocalDiagnosticConfig(
        run_id=run_id,
        result_directory=result_directory,
        dwagon_runtime_python=cast(str, arguments.dwagon_runtime_python),
        dwagon_runtime_install_receipt=runtime_install_receipt,
        dwagon_runtime_install_receipt_sha256=cast(
            str, arguments.dwagon_runtime_install_receipt_sha256
        ),
        dwagon_model_path=cast(str, arguments.dwagon_model_path),
        local_source_directory=cast(str, arguments.local_source_directory),
        dwagon_ip=cast(str, arguments.dwagon_ip),
        dwagon_socket_interface=cast(str, arguments.dwagon_socket_interface),
        distributed_port=ports[0],
        stage_ports=ports[1:],
        pipeline_layer_partition=cast(
            tuple[int, int], arguments.pipeline_layer_partition
        ),
        resident_gpu_experts=resident_gpu_experts,
        readiness_timeout_seconds=cast(float, arguments.readiness_timeout_seconds),
        request_timeout_seconds=cast(float, arguments.request_timeout_seconds),
        cleanup_timeout_seconds=cast(float, arguments.cleanup_timeout_seconds),
        warmup_count=cast(int, arguments.warmups),
        sample_count=cast(int, arguments.samples),
    )


def main() -> int:
    try:
        config = _config_from_arguments(_parser().parse_args())
        payload = run_diagnostic(config)
    except (Pp2LocalDiagnosticError, OSError, ValueError) as error:
        print(f"local PP2 diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload["benchmark_summary"], indent=2, sort_keys=True))
    print(config.result_directory / "pp2-local-diagnostic-result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
