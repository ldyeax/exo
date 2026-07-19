import hashlib
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast, final

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from exo.shared.types.common import ModelId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    KTransformersMethod,
    ResourceIndex,
)
from exo.worker.sglang_kt.preflight import Sha256Digest
from exo.worker.sglang_kt.receipt_io import (
    SglangKtBoundFile,
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    hash_sglang_kt_bound_file,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)

MODEL_CONTRACT_SCHEMA_VERSION = 1
MODEL_CONTRACT_CANONICALIZATION = "exo-sglang-kt-model-contract-v1"
_CONFIG_MAXIMUM_BYTES = 1024 * 1024
_INDEX_MAXIMUM_BYTES = 64 * 1024 * 1024
_MANIFEST_MAXIMUM_BYTES = 1024 * 1024
_RUNTIME_FILE_MAXIMUM_BYTES = 64 * 1024 * 1024
_HUGGING_FACE_METADATA_MAXIMUM_BYTES = 4096
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")

ModelContractFileRole = Literal[
    "chat_template",
    "config",
    "generation_config",
    "safetensors_index",
    "tokenizer",
    "tokenizer_config",
    "weight_shard",
]
HuggingFaceEtagAlgorithm = Literal["git_blob_sha1", "sha256"]
RelativeModelFilePath = Annotated[str, StringConstraints(min_length=1, max_length=255)]
HuggingFaceEtagDigest = Annotated[
    str,
    StringConstraints(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"),
]
_SINGLETON_RUNTIME_FILES: tuple[tuple[str, ModelContractFileRole], ...] = (
    ("chat_template.jinja", "chat_template"),
    ("config.json", "config"),
    ("generation_config.json", "generation_config"),
    ("model.safetensors.index.json", "safetensors_index"),
    ("tokenizer.json", "tokenizer"),
    ("tokenizer_config.json", "tokenizer_config"),
)
_MODEL_CONTRACT_ROLES: tuple[ModelContractFileRole, ...] = (
    "chat_template",
    "config",
    "generation_config",
    "safetensors_index",
    "tokenizer",
    "tokenizer_config",
    "weight_shard",
)


class SglangKtModelContractError(ValueError):
    """Raised when a checkpoint does not satisfy an exact model contract."""


def _validate_relative_model_file_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or path.as_posix() != value
    ):
        raise ValueError("model contract paths must be normalized root filenames")
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@final
class SglangKtModelContractFile(_StrictModel):
    path: RelativeModelFilePath
    role: ModelContractFileRole
    size_bytes: PositiveInt
    sha256: Sha256Digest
    huggingface_etag_algorithm: HuggingFaceEtagAlgorithm
    huggingface_etag_digest: HuggingFaceEtagDigest

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_relative_model_file_path(value)

    @model_validator(mode="after")
    def validate_huggingface_etag(self) -> "SglangKtModelContractFile":
        expected_length = (
            40 if self.huggingface_etag_algorithm == "git_blob_sha1" else 64
        )
        if len(self.huggingface_etag_digest) != expected_length:
            raise ValueError(
                "Hugging Face ETag digest length disagrees with its algorithm"
            )
        if self.role == "weight_shard" and self.huggingface_etag_algorithm != "sha256":
            raise ValueError("weight shards require Hugging Face LFS SHA-256 ETags")
        return self


@final
class SglangKtModelContract(_StrictModel):
    schema_version: Literal[1]
    canonicalization: Literal["exo-sglang-kt-model-contract-v1"]
    model_id: ModelId
    revision: GitRevision
    weight_format: Literal["safetensors"]
    ktransformers_method: KTransformersMethod
    full_indexer_layer_starts: tuple[ResourceIndex, ...]
    weight_map_entries: PositiveInt
    index_metadata_total_size: PositiveInt
    physical_weight_bytes: PositiveInt
    files: tuple[SglangKtModelContractFile, ...]

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: ModelId) -> ModelId:
        if _MODEL_ID_PATTERN.fullmatch(str(value)) is None:
            raise ValueError("model_id must be an exact owner/repository identifier")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "SglangKtModelContract":
        starts = self.full_indexer_layer_starts
        if not starts or starts[0] != 0 or tuple(sorted(set(starts))) != starts:
            raise ValueError(
                "full_indexer_layer_starts must be sorted, unique, and begin at zero"
            )
        paths = tuple(file.path for file in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("model contract files must have sorted, unique paths")
        role_counts = {
            role: sum(file.role == role for file in self.files)
            for role in _MODEL_CONTRACT_ROLES
        }
        if any(role_counts[role] != 1 for _path, role in _SINGLETON_RUNTIME_FILES):
            raise ValueError("model contract requires every singleton runtime file")
        if role_counts["weight_shard"] <= 0:
            raise ValueError("model contract requires indexed weight shards")
        physical_bytes = sum(
            file.size_bytes for file in self.files if file.role == "weight_shard"
        )
        if physical_bytes != self.physical_weight_bytes:
            raise ValueError("physical_weight_bytes disagrees with shard entries")
        return self


@final
class SglangKtLoadedModelContract(_StrictModel):
    path: AbsoluteRuntimePath
    receipt_sha256: Sha256Digest
    contract_sha256: Sha256Digest
    contract: SglangKtModelContract


@final
class SglangKtVerifiedModelSnapshot(_StrictModel):
    model_path: AbsoluteRuntimePath
    model_id: ModelId
    revision: GitRevision
    weight_format: Literal["safetensors"]
    ktransformers_method: KTransformersMethod
    full_indexer_layer_starts: tuple[ResourceIndex, ...]
    contract_path: AbsoluteRuntimePath
    contract_receipt_sha256: Sha256Digest
    contract_sha256: Sha256Digest
    config_sha256: Sha256Digest
    index_sha256: Sha256Digest
    weight_map_entries: PositiveInt
    shard_count: PositiveInt
    physical_weight_bytes: PositiveInt


def calculate_sglang_kt_model_contract_sha256(
    contract: SglangKtModelContract,
) -> str:
    payload = contract.model_dump(mode="json")
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()


def _required_json_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SglangKtModelContractError(f"{description} must be a JSON object")
    object_mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in object_mapping):
        raise SglangKtModelContractError(f"{description} must be a JSON object")
    return cast(dict[str, object], object_mapping)


def _parse_safetensors_index(
    index_file: SglangKtBoundFile,
) -> tuple[dict[str, str], int]:
    root = _required_json_object(
        parse_sglang_kt_strict_json(index_file.contents),
        "safetensors index",
    )
    if set(root) != {"metadata", "weight_map"}:
        raise SglangKtModelContractError(
            "safetensors index must contain exactly metadata and weight_map"
        )
    metadata = _required_json_object(root["metadata"], "safetensors index metadata")
    if set(metadata) != {"total_size"}:
        raise SglangKtModelContractError(
            "safetensors index metadata must contain exactly total_size"
        )
    total_size = metadata["total_size"]
    if type(total_size) is not int or total_size <= 0:
        raise SglangKtModelContractError(
            "safetensors index metadata total_size must be positive"
        )
    weight_map_raw = _required_json_object(
        root["weight_map"], "safetensors index weight_map"
    )
    if not weight_map_raw:
        raise SglangKtModelContractError("safetensors index weight_map is empty")
    weight_map: dict[str, str] = {}
    for tensor_name, shard_name in weight_map_raw.items():
        if not tensor_name or not isinstance(shard_name, str):
            raise SglangKtModelContractError(
                "safetensors index tensor and shard names must be strings"
            )
        try:
            _validate_relative_model_file_path(shard_name)
        except ValueError as error:
            raise SglangKtModelContractError(
                f"safetensors index contains unsafe shard path {shard_name!r}"
            ) from error
        if not shard_name.endswith(".safetensors"):
            raise SglangKtModelContractError(
                "safetensors index contains a non-safetensors shard"
            )
        weight_map[tensor_name] = shard_name
    return weight_map, total_size


def _git_blob_sha1(contents: bytes) -> str:
    prefix = f"blob {len(contents)}\0".encode()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(prefix)
    digest.update(contents)
    return digest.hexdigest()


def _observe_hugging_face_metadata(
    snapshot_path: Path,
    *,
    relative_path: str,
    revision: GitRevision,
    content_sha256: str,
    contents: bytes | None,
) -> tuple[HuggingFaceEtagAlgorithm, str]:
    metadata_path = (
        snapshot_path
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative_path}.metadata"
    )
    metadata = read_sglang_kt_bound_file(
        metadata_path,
        maximum_bytes=_HUGGING_FACE_METADATA_MAXIMUM_BYTES,
    )
    try:
        lines = metadata.contents.decode("utf-8").splitlines()
        timestamp = float(lines[2])
    except (UnicodeDecodeError, ValueError, IndexError) as error:
        raise SglangKtModelContractError(
            f"invalid Hugging Face metadata for {relative_path}"
        ) from error
    if len(lines) != 3 or lines[0] != revision or not math.isfinite(timestamp):
        raise SglangKtModelContractError(
            f"Hugging Face metadata does not bind {relative_path} to {revision}"
        )
    etag = lines[1]
    if re.fullmatch(r"[0-9a-f]{64}", etag) is not None:
        if etag != content_sha256:
            raise SglangKtModelContractError(
                f"Hugging Face LFS ETag disagrees with {relative_path}"
            )
        return "sha256", etag
    if re.fullmatch(r"[0-9a-f]{40}", etag) is not None and contents is not None:
        if etag != _git_blob_sha1(contents):
            raise SglangKtModelContractError(
                f"Hugging Face Git ETag disagrees with {relative_path}"
            )
        return "git_blob_sha1", etag
    raise SglangKtModelContractError(
        f"Hugging Face metadata has an unsupported ETag for {relative_path}"
    )


def _contract_file_from_bound_contents(
    snapshot_path: Path,
    *,
    relative_path: str,
    role: ModelContractFileRole,
    revision: GitRevision,
    bound_file: SglangKtBoundFile,
) -> SglangKtModelContractFile:
    etag_algorithm, etag_digest = _observe_hugging_face_metadata(
        snapshot_path,
        relative_path=relative_path,
        revision=revision,
        content_sha256=bound_file.sha256,
        contents=bound_file.contents,
    )
    return SglangKtModelContractFile(
        path=relative_path,
        role=role,
        size_bytes=len(bound_file.contents),
        sha256=bound_file.sha256,
        huggingface_etag_algorithm=etag_algorithm,
        huggingface_etag_digest=etag_digest,
    )


def _read_snapshot_metadata(
    snapshot_path: Path,
) -> tuple[SglangKtBoundFile, SglangKtBoundFile, dict[str, str], int]:
    config = read_sglang_kt_bound_file(
        snapshot_path / "config.json",
        maximum_bytes=_CONFIG_MAXIMUM_BYTES,
    )
    parse_sglang_kt_strict_json(config.contents)
    index = read_sglang_kt_bound_file(
        snapshot_path / "model.safetensors.index.json",
        maximum_bytes=_INDEX_MAXIMUM_BYTES,
    )
    weight_map, total_size = _parse_safetensors_index(index)
    return config, index, weight_map, total_size


def _snapshot_safetensors_files(snapshot_path: Path) -> tuple[str, ...]:
    try:
        names: list[str] = []
        with os.scandir(snapshot_path) as entries:
            for entry in entries:
                if entry.name.endswith(".safetensors"):
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise SglangKtModelContractError(
                            f"safetensors shard is not a regular file: {entry.name}"
                        )
                    names.append(entry.name)
        return tuple(sorted(names))
    except OSError as error:
        raise SglangKtModelContractError(
            f"cannot enumerate model snapshot {snapshot_path}"
        ) from error


def _reject_uncontracted_executable_files(snapshot_path: Path) -> None:
    executable_suffixes = (".py", ".pyc", ".pyd", ".so")
    try:
        for directory, directory_names, filenames in os.walk(
            snapshot_path,
            followlinks=False,
        ):
            directory_path = Path(directory)
            if directory_path == snapshot_path:
                directory_names[:] = [
                    name for name in directory_names if name != ".cache"
                ]
            for directory_name in directory_names:
                if (directory_path / directory_name).is_symlink():
                    raise SglangKtModelContractError(
                        "model snapshot contains a symlinked directory"
                    )
            for filename in filenames:
                artifact = directory_path / filename
                relative_path = artifact.relative_to(snapshot_path)
                if artifact.is_symlink():
                    raise SglangKtModelContractError(
                        f"model snapshot contains a symlink: {relative_path}"
                    )
                if filename.endswith(executable_suffixes):
                    raise SglangKtModelContractError(
                        f"model snapshot contains uncontracted executable code: "
                        f"{relative_path}"
                    )
                if filename.endswith(".safetensors") and relative_path.parent != Path(
                    "."
                ):
                    raise SglangKtModelContractError(
                        f"model snapshot contains a nested weight shard: {relative_path}"
                    )
    except OSError as error:
        raise SglangKtModelContractError(
            f"cannot inspect model snapshot tree {snapshot_path}"
        ) from error


def _create_sglang_kt_model_contract(
    snapshot_path: Path,
    *,
    model_id: ModelId,
    revision: GitRevision,
    ktransformers_method: KTransformersMethod,
    full_indexer_layer_starts: tuple[ResourceIndex, ...],
) -> SglangKtModelContract:
    """Hash the exact config, index, and indexed shards in one snapshot."""

    _reject_uncontracted_executable_files(snapshot_path)
    config, index, weight_map, total_size = _read_snapshot_metadata(snapshot_path)
    shard_names = tuple(sorted(set(weight_map.values())))
    if _snapshot_safetensors_files(snapshot_path) != shard_names:
        raise SglangKtModelContractError(
            "snapshot safetensors files do not exactly match the index"
        )
    files: list[SglangKtModelContractFile] = [
        _contract_file_from_bound_contents(
            snapshot_path,
            relative_path="config.json",
            role="config",
            revision=revision,
            bound_file=config,
        ),
        _contract_file_from_bound_contents(
            snapshot_path,
            relative_path="model.safetensors.index.json",
            role="safetensors_index",
            revision=revision,
            bound_file=index,
        ),
    ]
    for runtime_path, role in _SINGLETON_RUNTIME_FILES:
        if role in {"config", "safetensors_index"}:
            continue
        artifact = read_sglang_kt_bound_file(
            snapshot_path / runtime_path,
            maximum_bytes=_RUNTIME_FILE_MAXIMUM_BYTES,
        )
        files.append(
            _contract_file_from_bound_contents(
                snapshot_path,
                relative_path=runtime_path,
                role=role,
                revision=revision,
                bound_file=artifact,
            )
        )
    for shard_name in shard_names:
        shard = hash_sglang_kt_bound_file(snapshot_path / shard_name)
        etag_algorithm, etag_digest = _observe_hugging_face_metadata(
            snapshot_path,
            relative_path=shard_name,
            revision=revision,
            content_sha256=shard.sha256,
            contents=None,
        )
        files.append(
            SglangKtModelContractFile(
                path=shard_name,
                role="weight_shard",
                size_bytes=shard.size_bytes,
                sha256=shard.sha256,
                huggingface_etag_algorithm=etag_algorithm,
                huggingface_etag_digest=etag_digest,
            )
        )
    return SglangKtModelContract(
        schema_version=MODEL_CONTRACT_SCHEMA_VERSION,
        canonicalization=MODEL_CONTRACT_CANONICALIZATION,
        model_id=model_id,
        revision=revision,
        weight_format="safetensors",
        ktransformers_method=ktransformers_method,
        full_indexer_layer_starts=full_indexer_layer_starts,
        weight_map_entries=len(weight_map),
        index_metadata_total_size=total_size,
        physical_weight_bytes=sum(
            file.size_bytes for file in files if file.role == "weight_shard"
        ),
        files=tuple(sorted(files, key=lambda file: file.path)),
    )


def create_sglang_kt_model_contract(
    snapshot_path: Path,
    *,
    model_id: ModelId,
    revision: GitRevision,
    ktransformers_method: KTransformersMethod,
    full_indexer_layer_starts: tuple[ResourceIndex, ...],
) -> SglangKtModelContract:
    try:
        return _create_sglang_kt_model_contract(
            snapshot_path,
            model_id=model_id,
            revision=revision,
            ktransformers_method=ktransformers_method,
            full_indexer_layer_starts=full_indexer_layer_starts,
        )
    except SglangKtReceiptFileError as error:
        raise SglangKtModelContractError(
            f"cannot verify model snapshot files below {snapshot_path}"
        ) from error


def load_sglang_kt_model_contract(
    path: Path,
    *,
    expected_contract_sha256: Sha256Digest,
) -> SglangKtLoadedModelContract:
    try:
        bound_file = read_sglang_kt_bound_file(
            path,
            maximum_bytes=_MANIFEST_MAXIMUM_BYTES,
        )
        parse_sglang_kt_strict_json(bound_file.contents)
        contract = SglangKtModelContract.model_validate_json(bound_file.contents)
    except (SglangKtReceiptFileError, ValidationError) as error:
        raise SglangKtModelContractError(f"invalid model contract {path}") from error
    contract_sha256 = calculate_sglang_kt_model_contract_sha256(contract)
    if contract_sha256 != expected_contract_sha256:
        raise SglangKtModelContractError(
            "model contract does not match the pinned canonical SHA-256"
        )
    return SglangKtLoadedModelContract(
        path=str(path),
        receipt_sha256=bound_file.sha256,
        contract_sha256=contract_sha256,
        contract=contract,
    )


def verify_sglang_kt_model_snapshot(
    snapshot_path: Path,
    contract_path: Path,
    *,
    expected_contract_sha256: Sha256Digest,
    expected_model_id: ModelId,
    expected_revision: GitRevision,
    expected_ktransformers_method: KTransformersMethod,
) -> SglangKtVerifiedModelSnapshot:
    loaded_contract = load_sglang_kt_model_contract(
        contract_path,
        expected_contract_sha256=expected_contract_sha256,
    )
    try:
        return _verify_sglang_kt_model_snapshot(
            snapshot_path,
            loaded_contract,
            expected_model_id=expected_model_id,
            expected_revision=expected_revision,
            expected_ktransformers_method=expected_ktransformers_method,
        )
    except SglangKtReceiptFileError as error:
        raise SglangKtModelContractError(
            f"cannot verify model snapshot files below {snapshot_path}"
        ) from error


def _verify_sglang_kt_model_snapshot(
    snapshot_path: Path,
    loaded_contract: SglangKtLoadedModelContract,
    *,
    expected_model_id: ModelId,
    expected_revision: GitRevision,
    expected_ktransformers_method: KTransformersMethod,
) -> SglangKtVerifiedModelSnapshot:
    contract = loaded_contract.contract
    if (
        contract.model_id != expected_model_id
        or contract.revision != expected_revision
        or contract.ktransformers_method != expected_ktransformers_method
    ):
        raise SglangKtModelContractError(
            "model contract identity does not match the launch contract"
        )
    _reject_uncontracted_executable_files(snapshot_path)
    config, index, weight_map, total_size = _read_snapshot_metadata(snapshot_path)
    shard_names = tuple(sorted(set(weight_map.values())))
    if (
        len(weight_map) != contract.weight_map_entries
        or total_size != contract.index_metadata_total_size
        or _snapshot_safetensors_files(snapshot_path) != shard_names
    ):
        raise SglangKtModelContractError(
            "snapshot index does not match the pinned model contract"
        )

    files_by_role: dict[
        ModelContractFileRole, tuple[SglangKtModelContractFile, ...]
    ] = {
        role: tuple(file for file in contract.files if file.role == role)
        for role in _MODEL_CONTRACT_ROLES
    }
    config_contract = files_by_role["config"][0]
    index_contract = files_by_role["safetensors_index"][0]
    if (
        config_contract.path != "config.json"
        or index_contract.path != "model.safetensors.index.json"
        or config.sha256 != config_contract.sha256
        or len(config.contents) != config_contract.size_bytes
        or index.sha256 != index_contract.sha256
        or len(index.contents) != index_contract.size_bytes
    ):
        raise SglangKtModelContractError(
            "snapshot config or index does not match the pinned model contract"
        )
    for bound_file, contract_file in (
        (config, config_contract),
        (index, index_contract),
    ):
        etag_algorithm, etag_digest = _observe_hugging_face_metadata(
            snapshot_path,
            relative_path=contract_file.path,
            revision=contract.revision,
            content_sha256=bound_file.sha256,
            contents=bound_file.contents,
        )
        if (
            etag_algorithm != contract_file.huggingface_etag_algorithm
            or etag_digest != contract_file.huggingface_etag_digest
        ):
            raise SglangKtModelContractError(
                f"Hugging Face metadata changed for {contract_file.path}"
            )
    for runtime_path, role in _SINGLETON_RUNTIME_FILES:
        if role in {"config", "safetensors_index"}:
            continue
        runtime_contract = files_by_role[role][0]
        if runtime_contract.path != runtime_path:
            raise SglangKtModelContractError(
                f"model contract has the wrong path for {role}"
            )
        artifact = read_sglang_kt_bound_file(
            snapshot_path / runtime_path,
            maximum_bytes=_RUNTIME_FILE_MAXIMUM_BYTES,
        )
        if (
            len(artifact.contents) != runtime_contract.size_bytes
            or artifact.sha256 != runtime_contract.sha256
        ):
            raise SglangKtModelContractError(
                f"model runtime file does not match pinned SHA-256: {runtime_path}"
            )
        etag_algorithm, etag_digest = _observe_hugging_face_metadata(
            snapshot_path,
            relative_path=runtime_path,
            revision=contract.revision,
            content_sha256=artifact.sha256,
            contents=artifact.contents,
        )
        if (
            etag_algorithm != runtime_contract.huggingface_etag_algorithm
            or etag_digest != runtime_contract.huggingface_etag_digest
        ):
            raise SglangKtModelContractError(
                f"Hugging Face metadata changed for {runtime_path}"
            )
    shard_contracts = files_by_role["weight_shard"]
    if tuple(file.path for file in shard_contracts) != shard_names:
        raise SglangKtModelContractError(
            "model contract shard set does not match the safetensors index"
        )
    for file in shard_contracts:
        artifact = hash_sglang_kt_bound_file(
            snapshot_path / file.path,
            expected_size_bytes=file.size_bytes,
        )
        if artifact.sha256 != file.sha256:
            raise SglangKtModelContractError(
                f"model shard does not match pinned SHA-256: {file.path}"
            )
        etag_algorithm, etag_digest = _observe_hugging_face_metadata(
            snapshot_path,
            relative_path=file.path,
            revision=contract.revision,
            content_sha256=artifact.sha256,
            contents=None,
        )
        if (
            etag_algorithm != file.huggingface_etag_algorithm
            or etag_digest != file.huggingface_etag_digest
        ):
            raise SglangKtModelContractError(
                f"Hugging Face metadata changed for {file.path}"
            )
    return SglangKtVerifiedModelSnapshot(
        model_path=str(snapshot_path),
        model_id=contract.model_id,
        revision=contract.revision,
        weight_format=contract.weight_format,
        ktransformers_method=contract.ktransformers_method,
        full_indexer_layer_starts=contract.full_indexer_layer_starts,
        contract_path=loaded_contract.path,
        contract_receipt_sha256=loaded_contract.receipt_sha256,
        contract_sha256=loaded_contract.contract_sha256,
        config_sha256=config.sha256,
        index_sha256=index.sha256,
        weight_map_entries=contract.weight_map_entries,
        shard_count=len(shard_contracts),
        physical_weight_bytes=contract.physical_weight_bytes,
    )
