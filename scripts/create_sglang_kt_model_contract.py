#!/usr/bin/env python3
"""Create an exact config/index/shard contract for an SGLang-KT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exo.shared.types.common import ModelId
from exo.shared.types.worker.sglang_kt import KTransformersMethod
from exo.worker.sglang_kt.model_contract import (
    SglangKtModelContractError,
    calculate_sglang_kt_model_contract_sha256,
    create_sglang_kt_model_contract,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    read_sglang_kt_bound_file,
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_MODEL_ID_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_METHODS: tuple[KTransformersMethod, ...] = (
    "AMXINT4",
    "AMXINT8",
    "BF16",
    "FP8",
    "FP8_PERCHANNEL",
    "LLAMAFILE",
    "MOE_INT4",
    "MOE_INT8",
    "MXFP4",
    "RAWINT4",
)


class ContractCreationError(RuntimeError):
    """Raised when a contract cannot be published without ambiguity."""


class _CliArguments(argparse.Namespace):
    snapshot: Path
    output: Path
    model_id: str
    revision: str
    ktransformers_method: KTransformersMethod
    full_indexer_layer_starts: tuple[int, ...]


def _absolute_normalized_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise argparse.ArgumentTypeError("path must be absolute and normalized")
    return path


def _model_id(value: str) -> str:
    components = value.split("/")
    if len(components) != 2 or any(
        _MODEL_ID_COMPONENT_PATTERN.fullmatch(component) is None
        for component in components
    ):
        raise argparse.ArgumentTypeError("model ID must be an exact owner/repository")
    return value


def _revision(value: str) -> str:
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("revision must be lowercase 40-hex")
    return value


def _layer_starts(value: str) -> tuple[int, ...]:
    try:
        starts = tuple(int(component) for component in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "layer starts must be comma-separated integers"
        ) from error
    if not starts or starts[0] != 0 or tuple(sorted(set(starts))) != starts:
        raise argparse.ArgumentTypeError(
            "layer starts must be sorted, unique, and begin at zero"
        )
    return starts


def parse_arguments(arguments: list[str] | None = None) -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=_absolute_normalized_path)
    parser.add_argument("--output", required=True, type=_absolute_normalized_path)
    parser.add_argument("--model-id", required=True, type=_model_id)
    parser.add_argument("--revision", required=True, type=_revision)
    parser.add_argument(
        "--ktransformers-method",
        required=True,
        choices=_METHODS,
    )
    parser.add_argument(
        "--full-indexer-layer-starts",
        required=True,
        type=_layer_starts,
    )
    namespace = _CliArguments()
    parser.parse_args(arguments, namespace=namespace)
    return namespace


def _publish_contract(path: Path, contents: bytes) -> None:
    if not path.parent.is_dir():
        raise ContractCreationError(f"output parent is not a directory: {path.parent}")
    if path.exists():
        try:
            existing = read_sglang_kt_bound_file(
                path,
                maximum_bytes=max(len(contents), 1),
            )
        except SglangKtReceiptFileError as error:
            raise ContractCreationError(
                f"existing output is not reusable evidence: {path}"
            ) from error
        if existing.contents != contents:
            raise ContractCreationError(
                f"refusing to overwrite a different model contract: {path}"
            )
        return

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractCreationError("short write while publishing contract")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except FileExistsError as error:
        raise ContractCreationError(
            f"contract output or temporary path already exists: {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)


def create_contract(arguments: _CliArguments) -> dict[str, object]:
    if arguments.output == arguments.snapshot or arguments.snapshot in (
        arguments.output.parents
    ):
        raise ContractCreationError("contract output must be outside the snapshot")
    contract = create_sglang_kt_model_contract(
        arguments.snapshot,
        model_id=ModelId(arguments.model_id),
        revision=arguments.revision,
        ktransformers_method=arguments.ktransformers_method,
        full_indexer_layer_starts=arguments.full_indexer_layer_starts,
    )
    contents = canonical_sglang_kt_json(contract.model_dump(mode="json"))
    _publish_contract(arguments.output, contents)
    return {
        "schema_version": 1,
        "path": str(arguments.output),
        "receipt_sha256": hashlib.sha256(contents).hexdigest(),
        "contract_sha256": calculate_sglang_kt_model_contract_sha256(contract),
        "shard_count": sum(file.role == "weight_shard" for file in contract.files),
        "physical_weight_bytes": contract.physical_weight_bytes,
    }


def main(arguments: list[str] | None = None) -> int:
    try:
        result = create_contract(parse_arguments(arguments))
    except (ContractCreationError, SglangKtModelContractError, OSError) as error:
        print(f"model contract creation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
