#!/usr/bin/env python3
"""Publish trusted OLMoE snapshot and EP1/EP2 sanity evidence.

The workflow is intentionally two-step because dwagon has only two GPUs:

1. Use the OLMoE benchmark harness in capture mode for native TP2/EP1.
2. Restart through the harness in TP2/EP2 capture mode.
3. Run ``publish`` with both captures. Publication succeeds only when their
   tokenized prompt, output IDs, output text, model, runtime, and server launch
   identities are equal.

Every command full-rehashes the three LFS shards and validates Hugging Face's
per-file revision/ETag metadata against hard-pinned identities. The live model
tree is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, cast, final
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)

sys.dont_write_bytecode = True

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.worker.sglang_kt.launch_spec import (  # noqa: E402
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import (  # noqa: E402
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    hash_sglang_kt_bound_file,
    parse_sglang_kt_strict_json,
    read_sglang_kt_bound_file,
)
from scripts.sglang_olmoe_serving_client import (  # noqa: E402
    OlmoeNativeGenerateRequest,
    OlmoeNativeServingClient,
    OlmoeSamplingParameters,
    token_ids_sha256,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type ExpertParallelSize = Literal[1, 2]

OLMOE_MODEL_ID: Final = "allenai/OLMoE-1B-7B-0924"
OLMOE_MODEL_REVISION: Final = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
OLMOE_SGLANG_REVISION: Final = GLM_4_7_FLASH_SGLANG_REVISION
OLMOE_MODEL_PATH: Final = (
    "/var/lib/exo/models/"
    "allenai--OLMoE-1B-7B-0924--6d84c48581ece794365f2b8e9cfb043c68ade9c5"
)
OLMOE_PHYSICAL_WEIGHT_BYTES: Final = 13_838_721_960
OLMOE_INDEXED_WEIGHT_BYTES: Final = 13_838_323_712
OLMOE_WEIGHT_MAP_ENTRIES: Final = 3_219
OLMOE_SANITY_PROMPT: Final = "17 + 25 ="
OLMOE_SANITY_MARKER: Final = "42"
OLMOE_SANITY_MAX_NEW_TOKENS: Final = 1
OLMOE_SANITY_SAMPLING_SEED: Final = 20_260_720
PINNED_SERVER_VERSION: Final = "0.0.0.dev0"
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)
_CAPTURE_MAXIMUM_BYTES: Final = 4 * 1024 * 1024
_METADATA_MAXIMUM_BYTES: Final = 4_096
_MANAGED_NAMESPACE_PATTERN: Final = re.compile(
    r"exo-olmoe-ep-[A-Za-z0-9_.-]{1,128}-[0-9a-f]{32}", re.ASCII
)
PUBLISHED_STAGE_CONTRACT_V2_SHA256: Final = (
    "d54c92f60483f3786c85a2e92f2dbb8d3898bb924902fd9a87c3093912976881"
)
PUBLISHED_STAGE_CONTRACT_V2_HARNESS_SHA256: Final = (
    "83c55233742e3d82957f99687f6064c086459cbecaae1326226a6f629fe847e1"
)
PUBLISHED_STAGE_CONTRACT_V2_PRODUCER_SHA256: Final = (
    "9e89751a7b955c2cde1aa66e2bac998adf0cf1435eff770e23d74505299e6549"
)
_STAGE_CONTRACT_RECEIPT_CONTEXT_KEY: Final = "olmoe_stage_contract_receipt_sha256"


def _is_published_v2_validation(info: ValidationInfo) -> bool:
    raw_context: object = info.context
    if not isinstance(raw_context, Mapping):
        return False
    context = cast(Mapping[object, object], raw_context)
    return (
        context.get(_STAGE_CONTRACT_RECEIPT_CONTEXT_KEY)
        == PUBLISHED_STAGE_CONTRACT_V2_SHA256
    )


def _stage_contract_validation_context(receipt_sha256: str) -> Mapping[str, str] | None:
    if receipt_sha256 != PUBLISHED_STAGE_CONTRACT_V2_SHA256:
        return None
    return {_STAGE_CONTRACT_RECEIPT_CONTEXT_KEY: receipt_sha256}


class OlmoeStageContractError(RuntimeError):
    """Raised when a trusted OLMoE publication cannot be produced."""


@dataclass(frozen=True, slots=True)
class PinnedSnapshotFile:
    path: str
    role: Literal["auxiliary", "config", "index", "tokenizer", "weight_shard"]
    size_bytes: int
    sha256: str
    huggingface_etag: str


PINNED_SNAPSHOT_FILES: Final = (
    PinnedSnapshotFile(
        ".gitattributes",
        "auxiliary",
        2_140,
        "668f82ed000734d13cf41e1e7122c17b5f2c08b9da3e4e52595b5785b3335fa7",
        "b076fca7b47a6fbdbee743446d33d3f5fdd9d6e3",
    ),
    PinnedSnapshotFile(
        "README.md",
        "auxiliary",
        7_270,
        "a2896dce1cf6b9f1ed32c09bda64eb7be3db3ad49ac4bceef15996a197806608",
        "fc44f9eeece5f3c6afaee154d68f74874c369c3e",
    ),
    PinnedSnapshotFile(
        "config.json",
        "config",
        759,
        "3643aa880d2f1c9b418156269ae791c73e5612d6b6b6fde0724d927cf89b6335",
        "f5c2a6dc189553bbddabb7cbe96b0c03f97ca4ab",
    ),
    PinnedSnapshotFile(
        "generation_config.json",
        "auxiliary",
        120,
        "d77272ffaa7e62a904e8e130bb25ab11585bd4a5026e388d6d682e4b82892ce2",
        "92e53a4411ee46fe1a2ea88b38925fe83192210a",
    ),
    PinnedSnapshotFile(
        "model-00001-of-00003.safetensors",
        "weight_shard",
        4_997_744_872,
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e",
        "5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e",
    ),
    PinnedSnapshotFile(
        "model-00002-of-00003.safetensors",
        "weight_shard",
        4_997_235_176,
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7",
        "15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7",
    ),
    PinnedSnapshotFile(
        "model-00003-of-00003.safetensors",
        "weight_shard",
        3_843_741_912,
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04",
        "a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04",
    ),
    PinnedSnapshotFile(
        "model.safetensors.index.json",
        "index",
        287_214,
        "0e2e1e0d8d357ac7af817cff28410c3dbad398f060c517a433e4076b2aae5579",
        "4c7929b85bfb7ebfeb83cfd2398a4fe0ecfdf90e",
    ),
    PinnedSnapshotFile(
        "olmoe-logo.png",
        "auxiliary",
        23_827,
        "482000c8d723923ba1c2fd39c4334fcf95c9d88102576a3f956fa264172626de",
        "1a55f78bf5158836daa6860de4b65068c163cfb3",
    ),
    PinnedSnapshotFile(
        "special_tokens_map.json",
        "auxiliary",
        65,
        "b77491e270c6fcc5b2ecf22370f7318a6a18d3cabea09ba7bab92e9bf12656c2",
        "3e1d70c16f2640555ba8fd31d3ecd1c63425c6a2",
    ),
    PinnedSnapshotFile(
        "tokenizer.json",
        "tokenizer",
        2_115_417,
        "a094266ac6c4982efba277bc251349a5a6d6ad37efb39a2a90f53d8be2a40a40",
        "d861718426ec675fc9a629e91c30a3a427ff31bc",
    ),
    PinnedSnapshotFile(
        "tokenizer_config.json",
        "auxiliary",
        5_372,
        "78a839c7851f14f9fb30e664c2b46166dc0628f2900679e5ec160656f702edff",
        "00797e612182918e8fd07c15fc58321399cce6d8",
    ),
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@final
class SnapshotFileObservation(_StrictModel):
    path: str = Field(min_length=1, max_length=256)
    role: Literal["auxiliary", "config", "index", "tokenizer", "weight_shard"]
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    huggingface_etag: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@final
class SnapshotObservation(_StrictModel):
    model_id: Literal["allenai/OLMoE-1B-7B-0924"]
    revision: Literal["6d84c48581ece794365f2b8e9cfb043c68ade9c5"]
    model_path: Literal[
        "/var/lib/exo/models/allenai--OLMoE-1B-7B-0924--6d84c48581ece794365f2b8e9cfb043c68ade9c5"
    ]
    files: tuple[SnapshotFileObservation, ...]
    physical_weight_bytes: Literal[13_838_721_960]
    indexed_weight_bytes: Literal[13_838_323_712]
    weight_map_entries: Literal[3_219]
    shard_count: Literal[3]
    full_file_content_rehash: Literal[True]
    huggingface_revision_metadata_verified: Literal[True]
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@final
class ManagedLaunchEvidence(_StrictModel):
    schema_version: Literal[1]
    status: Literal["owned_runtime_listener_verified"]
    harness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_admission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pid: int = Field(gt=0)
    process_group_id: int = Field(gt=0)
    start_time_ticks: int = Field(gt=0)
    owner_token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ownership_namespace: str = Field(min_length=1, max_length=256)
    command: tuple[str, ...] = Field(min_length=1, max_length=256)
    launch_environment: tuple[tuple[str, str], ...] = Field(
        min_length=1, max_length=256
    )
    listener_socket_inodes: tuple[int, ...] = Field(min_length=1, max_length=32)
    listener_owner_pids: tuple[int, ...] = Field(min_length=1, max_length=32)
    rank_local_numa_observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_at_utc: str

    @model_validator(mode="after")
    def validate_managed_launch(self, info: ValidationInfo) -> "ManagedLaunchEvidence":
        harness_path = (
            Path(__file__)
            .resolve(strict=True)
            .with_name("run_sglang_olmoe_ep_local_benchmark.py")
        )
        expected_harness_sha256 = (
            PUBLISHED_STAGE_CONTRACT_V2_HARNESS_SHA256
            if _is_published_v2_validation(info)
            else hash_sglang_kt_bound_file(harness_path).sha256
        )
        environment_names = tuple(name for name, _value in self.launch_environment)
        if (
            self.harness_sha256 != expected_harness_sha256
            or self.process_group_id != self.pid
            or _MANAGED_NAMESPACE_PATTERN.fullmatch(self.ownership_namespace) is None
            or any(not value or "\0" in value for value in self.command)
            or tuple(sorted(self.launch_environment)) != self.launch_environment
            or len(set(environment_names)) != len(environment_names)
            or "EXO_OLMOE_EP_OWNER_TOKEN" in environment_names
            or dict(self.launch_environment).get("EXO_OLMOE_EP_NAMESPACE")
            != self.ownership_namespace
            or tuple(sorted(set(self.listener_socket_inodes)))
            != self.listener_socket_inodes
            or any(inode <= 0 for inode in self.listener_socket_inodes)
            or tuple(sorted(set(self.listener_owner_pids))) != self.listener_owner_pids
            or any(pid <= 0 for pid in self.listener_owner_pids)
        ):
            raise ValueError("managed launch evidence is not bound or coherent")
        _aware_timestamp(self.verified_at_utc, "managed launch verification")
        return self


@final
class SanityCapture(_StrictModel):
    schema_version: Literal[1]
    status: Literal["captured"]
    expert_parallel_size: Literal[1, 2]
    model_id: Literal["allenai/OLMoE-1B-7B-0924"]
    model_revision: Literal["6d84c48581ece794365f2b8e9cfb043c68ade9c5"]
    model_path: Literal[
        "/var/lib/exo/models/allenai--OLMoE-1B-7B-0924--6d84c48581ece794365f2b8e9cfb043c68ade9c5"
    ]
    sglang_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_install_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_text: Literal["17 + 25 ="]
    tokenizer_class: str = Field(min_length=1, max_length=256)
    input_ids: tuple[int, ...]
    input_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_ids: tuple[int, ...]
    output_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_text: str = Field(min_length=1, max_length=4_096)
    output_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_new_tokens: Literal[1]
    sampling_seed: Literal[20_260_720]
    static_memory_fraction: float = Field(ge=0.5, le=0.95)
    server_host: Literal["127.0.0.1", "localhost"]
    server_port: int = Field(ge=1, le=65_535)
    server_info_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    managed_launch: ManagedLaunchEvidence
    producer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at_utc: str

    @model_validator(mode="after")
    def validate_capture(self, info: ValidationInfo) -> "SanityCapture":
        expected_producer_sha256 = (
            PUBLISHED_STAGE_CONTRACT_V2_PRODUCER_SHA256
            if _is_published_v2_validation(info)
            else hash_sglang_kt_bound_file(Path(__file__).resolve(strict=True)).sha256
        )
        if (
            self.sglang_revision != OLMOE_SGLANG_REVISION
            or token_ids_sha256(self.input_ids) != self.input_ids_sha256
            or token_ids_sha256(self.output_ids) != self.output_ids_sha256
            or len(self.output_ids) != self.max_new_tokens
            or hashlib.sha256(self.output_text.encode()).hexdigest()
            != self.output_text_sha256
            or self.output_text.strip() != OLMOE_SANITY_MARKER
            or self.producer_sha256 != expected_producer_sha256
        ):
            raise ValueError("sanity capture is not bound or coherent")
        try:
            captured_at = datetime.fromisoformat(self.captured_at_utc)
        except ValueError as error:
            raise ValueError("sanity capture timestamp is invalid") from error
        if captured_at.utcoffset() is None:
            raise ValueError("sanity capture timestamp lacks timezone")
        verified_at = _aware_timestamp(
            self.managed_launch.verified_at_utc, "managed launch verification"
        )
        if verified_at > captured_at:
            raise ValueError("managed launch verification follows sanity capture")
        _verify_managed_launch_binding(self, self.managed_launch)
        return self


@final
class SanityCaptureBinding(_StrictModel):
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture: SanityCapture


def _command_option(command: tuple[str, ...], option: str) -> str:
    positions = tuple(index for index, value in enumerate(command) if value == option)
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise ValueError(f"managed launch command does not bind {option}")
    return command[positions[0] + 1]


def _normalized_ep_command(command: tuple[str, ...]) -> tuple[str, ...]:
    normalized = list(command)
    position = normalized.index("--ep-size")
    normalized[position + 1] = "<expert-parallel-size>"
    return tuple(normalized)


def _stable_launch_environment(
    evidence: ManagedLaunchEvidence,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        item
        for item in evidence.launch_environment
        if item[0] != "EXO_OLMOE_EP_NAMESPACE"
    )


def _verify_managed_launch_binding(
    capture: SanityCapture, evidence: ManagedLaunchEvidence
) -> None:
    command = evidence.command
    environment = dict(evidence.launch_environment)
    required_environment = {
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
        "EXO_OLMOE_EP_NAMESPACE": evidence.ownership_namespace,
    }
    allowed_environment = frozenset(
        {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            *required_environment,
        }
    )
    try:
        memory_fraction = float(_command_option(command, "--mem-fraction-static"))
    except ValueError as error:
        raise ValueError("managed launch memory fraction is invalid") from error
    if (
        len(command) < 7
        or command[1:5] != ("--physcpubind", "0-111", "--interleave", "0,1")
        or not Path(command[5]).is_absolute()
        or command[6:8] != ("-m", "sglang.launch_server")
        or _command_option(command, "--model-path") != OLMOE_MODEL_PATH
        or _command_option(command, "--host") != capture.server_host
        or _command_option(command, "--port") != str(capture.server_port)
        or _command_option(command, "--tp-size") != "2"
        or _command_option(command, "--pp-size") != "1"
        or _command_option(command, "--ep-size") != str(capture.expert_parallel_size)
        or _command_option(command, "--dtype") != "bfloat16"
        or _command_option(command, "--context-length") != "4096"
        or _command_option(command, "--max-total-tokens") != "4096"
        or _command_option(command, "--max-running-requests") != "1"
        or _command_option(command, "--random-seed") != "20260720"
        or _command_option(command, "--moe-a2a-backend") != "none"
        or _command_option(command, "--moe-runner-backend") != "triton"
        or not math.isclose(
            memory_fraction,
            capture.static_memory_fraction,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or "--disable-radix-cache" not in command
        or "--disable-custom-all-reduce" not in command
        or tuple(command[command.index("--numa-node") + 1 :][:2]) != ("0", "1")
        or frozenset(environment) != allowed_environment
        or any(
            environment.get(key) != value for key, value in required_environment.items()
        )
    ):
        raise ValueError("managed launch does not match the sanity capture")


@final
class OlmoeStageContract(_StrictModel):
    schema_version: Literal[1]
    status: Literal["published"]
    canonicalization: Literal["exo-olmoe-stage-contract-v1"]
    snapshot: SnapshotObservation
    ep1: SanityCaptureBinding
    ep2: SanityCaptureBinding
    published_at_utc: str
    producer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_equivalence(self) -> "OlmoeStageContract":
        first = self.ep1.capture
        second = self.ep2.capture
        _verify_managed_launch_binding(first, first.managed_launch)
        _verify_managed_launch_binding(second, second.managed_launch)
        if (
            first.expert_parallel_size != 1
            or second.expert_parallel_size != 2
            or first.runtime_install_receipt_sha256
            != second.runtime_install_receipt_sha256
            or first.snapshot_canonical_sha256 != self.snapshot.canonical_sha256
            or second.snapshot_canonical_sha256 != self.snapshot.canonical_sha256
            or first.prompt_text != second.prompt_text
            or first.input_ids != second.input_ids
            or first.output_ids != second.output_ids
            or first.output_text != second.output_text
            or first.static_memory_fraction != second.static_memory_fraction
            or first.server_host != second.server_host
            or first.server_port != second.server_port
            or first.managed_launch.harness_sha256
            != second.managed_launch.harness_sha256
            or first.managed_launch.runtime_admission_sha256
            != second.managed_launch.runtime_admission_sha256
            or _normalized_ep_command(first.managed_launch.command)
            != _normalized_ep_command(second.managed_launch.command)
            or _stable_launch_environment(first.managed_launch)
            != _stable_launch_environment(second.managed_launch)
            or first.producer_sha256 != self.producer_sha256
            or second.producer_sha256 != self.producer_sha256
        ):
            raise ValueError("EP1 and EP2 sanity captures are not exactly equivalent")
        first_time = _aware_timestamp(first.captured_at_utc, "EP1 capture")
        second_time = _aware_timestamp(second.captured_at_utc, "EP2 capture")
        published_time = _aware_timestamp(self.published_at_utc, "contract publication")
        if first_time > second_time or second_time > published_time:
            raise ValueError("stage contract timestamps are not ordered EP1 then EP2")
        return self


@dataclass(frozen=True, slots=True)
class LoadedStageContract:
    path: str
    receipt_sha256: str
    contract: OlmoeStageContract


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class _TokenizerFactory(Protocol):
    @staticmethod
    def from_pretrained(
        path: str, *, local_files_only: bool, trust_remote_code: bool
    ) -> object: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _aware_timestamp(raw: str, description: str) -> datetime:
    try:
        observed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{description} timestamp is invalid") from error
    if observed.utcoffset() is None:
        raise ValueError(f"{description} timestamp lacks timezone")
    return observed


def _git_blob_sha1(contents: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(contents)}\0".encode())
    digest.update(contents)
    return digest.hexdigest()


def _strict_json(
    path: Path, maximum_bytes: int, description: str
) -> tuple[JsonObject, str]:
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=maximum_bytes)
        parsed = parse_sglang_kt_strict_json(bound.contents)
    except SglangKtReceiptFileError as error:
        raise OlmoeStageContractError(f"cannot read {description}: {error}") from error
    if not isinstance(parsed, dict):
        raise OlmoeStageContractError(f"{description} must be a JSON object")
    return cast(JsonObject, parsed), bound.sha256


def verify_pinned_snapshot(
    snapshot_path: Path = Path(OLMOE_MODEL_PATH),
) -> SnapshotObservation:
    """Full-rehash the exact HF snapshot and validate every revision/ETag record."""

    if str(snapshot_path) != OLMOE_MODEL_PATH:
        raise OlmoeStageContractError("snapshot path is not the pinned OLMoE location")
    try:
        root_status = snapshot_path.lstat()
    except OSError as error:
        raise OlmoeStageContractError(f"snapshot is unavailable: {error}") from error
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise OlmoeStageContractError("snapshot path is not an exact directory")
    expected_names = {file.path for file in PINNED_SNAPSHOT_FILES}
    observed_names = {
        entry.name for entry in snapshot_path.iterdir() if entry.name != ".cache"
    }
    if observed_names != expected_names:
        raise OlmoeStageContractError("snapshot root file set is not exact")
    metadata_root = snapshot_path / ".cache/huggingface/download"
    expected_metadata_names = {
        f"{file.path}.metadata" for file in PINNED_SNAPSHOT_FILES
    }
    observed_metadata_names = {entry.name for entry in metadata_root.iterdir()}
    if observed_metadata_names != expected_metadata_names:
        raise OlmoeStageContractError("Hugging Face metadata file set is not exact")

    file_evidence: list[SnapshotFileObservation] = []
    for expected in PINNED_SNAPSHOT_FILES:
        path = snapshot_path / expected.path
        try:
            artifact = hash_sglang_kt_bound_file(
                path, expected_size_bytes=expected.size_bytes
            )
            metadata = read_sglang_kt_bound_file(
                metadata_root / f"{expected.path}.metadata",
                maximum_bytes=_METADATA_MAXIMUM_BYTES,
            )
        except SglangKtReceiptFileError as error:
            raise OlmoeStageContractError(
                f"pinned snapshot file changed: {expected.path}: {error}"
            ) from error
        if artifact.sha256 != expected.sha256:
            raise OlmoeStageContractError(
                f"pinned snapshot SHA-256 changed: {expected.path}"
            )
        try:
            lines = metadata.contents.decode().splitlines()
            timestamp = float(lines[2])
        except (UnicodeDecodeError, ValueError, IndexError) as error:
            raise OlmoeStageContractError(
                f"invalid HF metadata for {expected.path}"
            ) from error
        if (
            len(lines) != 3
            or lines[0] != OLMOE_MODEL_REVISION
            or lines[1] != expected.huggingface_etag
            or not math.isfinite(timestamp)
        ):
            raise OlmoeStageContractError(
                f"HF metadata does not bind {expected.path} to the pinned revision"
            )
        if len(expected.huggingface_etag) == 64:
            if expected.huggingface_etag != artifact.sha256:
                raise OlmoeStageContractError("LFS ETag and content hash disagree")
        else:
            contents = path.read_bytes()
            if _git_blob_sha1(contents) != expected.huggingface_etag:
                raise OlmoeStageContractError("Git blob ETag and content disagree")
        file_evidence.append(
            SnapshotFileObservation(
                path=expected.path,
                role=expected.role,
                size_bytes=expected.size_bytes,
                sha256=expected.sha256,
                huggingface_etag=expected.huggingface_etag,
                metadata_sha256=metadata.sha256,
            )
        )

    config_document, _config_receipt_sha = _strict_json(
        snapshot_path / "config.json", 1024 * 1024, "OLMoE config"
    )
    index_document, _index_receipt_sha = _strict_json(
        snapshot_path / "model.safetensors.index.json",
        64 * 1024 * 1024,
        "OLMoE index",
    )
    metadata_value = index_document.get("metadata")
    weight_map_value = index_document.get("weight_map")
    if not isinstance(metadata_value, dict) or not isinstance(weight_map_value, dict):
        raise OlmoeStageContractError("OLMoE index structure is invalid")
    shard_names = set(cast(dict[str, object], weight_map_value).values())
    expected_shards = {
        file.path for file in PINNED_SNAPSHOT_FILES if file.role == "weight_shard"
    }
    if (
        config_document.get("model_type") != "olmoe"
        or config_document.get("vocab_size") != 50_304
        or config_document.get("num_experts") != 64
        or config_document.get("num_experts_per_tok") != 8
        or config_document.get("hidden_size") != 2_048
        or config_document.get("num_hidden_layers") != 16
        or config_document.get("num_attention_heads") != 16
        or config_document.get("num_key_value_heads") != 16
        or config_document.get("max_position_embeddings") != 4_096
        or cast(dict[str, object], metadata_value).get("total_size")
        != OLMOE_INDEXED_WEIGHT_BYTES
        or len(cast(dict[str, object], weight_map_value)) != OLMOE_WEIGHT_MAP_ENTRIES
        or shard_names != expected_shards
    ):
        raise OlmoeStageContractError("OLMoE config/index topology is not pinned")
    payload = SnapshotObservation(
        model_id=OLMOE_MODEL_ID,
        revision=OLMOE_MODEL_REVISION,
        model_path=OLMOE_MODEL_PATH,
        files=tuple(file_evidence),
        physical_weight_bytes=OLMOE_PHYSICAL_WEIGHT_BYTES,
        indexed_weight_bytes=OLMOE_INDEXED_WEIGHT_BYTES,
        weight_map_entries=OLMOE_WEIGHT_MAP_ENTRIES,
        shard_count=3,
        full_file_content_rehash=True,
        huggingface_revision_metadata_verified=True,
        canonical_sha256="0" * 64,
    )
    identity = cast(JsonObject, payload.model_dump(mode="json"))
    identity.pop("canonical_sha256")
    return payload.model_copy(
        update={
            "canonical_sha256": hashlib.sha256(
                canonical_sglang_kt_json(identity)
            ).hexdigest()
        }
    )


def _atomic_publish(path: Path, payload: BaseModel) -> str:
    try:
        parent_status = path.parent.lstat()
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise OlmoeStageContractError(
            f"output parent is unavailable: {error}"
        ) from error
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(path))
        or resolved_parent != path.parent
        or not stat.S_ISDIR(parent_status.st_mode)
        or stat.S_ISLNK(parent_status.st_mode)
        or path.exists()
    ):
        raise OlmoeStageContractError("output must be a new absolute path")
    contents = canonical_sglang_kt_json(payload.model_dump(mode="json"))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    published = False
    try:
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OlmoeStageContractError("atomic publication made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(contents).hexdigest()


def _load_tokenizer(snapshot_path: Path) -> tuple[_Tokenizer, str]:
    import transformers

    factory = cast(_TokenizerFactory, cast(object, transformers.AutoTokenizer))
    tokenizer = cast(
        _Tokenizer,
        factory.from_pretrained(
            str(snapshot_path), local_files_only=True, trust_remote_code=False
        ),
    )
    return tokenizer, type(tokenizer).__name__


def _verify_capture_server_info(
    response: Mapping[str, object],
    ep_size: ExpertParallelSize,
    static_memory_fraction: float,
    server_host: str,
    server_port: int,
) -> None:
    expected: dict[str, object] = {
        "version": PINNED_SERVER_VERSION,
        "model_path": OLMOE_MODEL_PATH,
        "host": server_host,
        "port": server_port,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": ep_size,
        "nnodes": 1,
        "node_rank": 0,
        "dtype": "bfloat16",
        "context_length": 4_096,
        "max_total_tokens": 4_096,
        "max_running_requests": 1,
        "random_seed": OLMOE_SANITY_SAMPLING_SEED,
        "mem_fraction_static": static_memory_fraction,
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "numa_node": [0, 1],
    }
    mismatches = {
        key: {"expected": value, "actual": response.get(key)}
        for key, value in expected.items()
        if response.get(key) != value
    }
    if mismatches:
        raise OlmoeStageContractError(f"capture server_info mismatch: {mismatches}")


def collect_sanity_capture(
    *,
    snapshot_path: Path,
    expert_parallel_size: ExpertParallelSize,
    base_url: str,
    runtime_install_receipt_sha256: str,
    static_memory_fraction: float,
    managed_launch: ManagedLaunchEvidence,
    client_factory: Callable[..., OlmoeNativeServingClient] = OlmoeNativeServingClient,
    tokenizer_loader: Callable[[Path], tuple[_Tokenizer, str]] = _load_tokenizer,
) -> SanityCapture:
    snapshot = verify_pinned_snapshot(snapshot_path)
    if _SHA256_PATTERN.fullmatch(runtime_install_receipt_sha256) is None:
        raise OlmoeStageContractError("runtime install receipt SHA-256 is invalid")
    if (
        not math.isfinite(static_memory_fraction)
        or not 0.5 <= static_memory_fraction <= 0.95
    ):
        raise OlmoeStageContractError("static memory fraction is invalid")
    parsed_url = urlsplit(base_url)
    try:
        server_port = parsed_url.port
    except ValueError as error:
        raise OlmoeStageContractError("capture base URL has an invalid port") from error
    server_host = parsed_url.hostname
    if (
        parsed_url.scheme != "http"
        or server_host not in {"127.0.0.1", "localhost"}
        or server_port is None
        or parsed_url.path not in {"", "/"}
        or parsed_url.query
        or parsed_url.fragment
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise OlmoeStageContractError("capture base URL must be an exact loopback URL")
    tokenizer, tokenizer_class = tokenizer_loader(snapshot_path)
    input_ids = tuple(tokenizer.encode(OLMOE_SANITY_PROMPT, add_special_tokens=True))
    request = OlmoeNativeGenerateRequest(
        input_ids=input_ids,
        sampling_params=OlmoeSamplingParameters(
            max_new_tokens=OLMOE_SANITY_MAX_NEW_TOKENS,
            temperature=0.0,
            ignore_eos=False,
            sampling_seed=OLMOE_SANITY_SAMPLING_SEED,
        ),
        stream=False,
    )
    with client_factory(base_url, timeout_seconds=900.0) as client:
        server_info = client.server_info()
        _verify_capture_server_info(
            server_info.response,
            expert_parallel_size,
            static_memory_fraction,
            server_host,
            server_port,
        )
        response = client.generate_sanity(request)
        client.flush_cache()
    capture = SanityCapture(
        schema_version=1,
        status="captured",
        expert_parallel_size=expert_parallel_size,
        model_id=OLMOE_MODEL_ID,
        model_revision=OLMOE_MODEL_REVISION,
        model_path=OLMOE_MODEL_PATH,
        sglang_revision=OLMOE_SGLANG_REVISION,
        runtime_install_receipt_sha256=runtime_install_receipt_sha256,
        snapshot_canonical_sha256=snapshot.canonical_sha256,
        prompt_text=OLMOE_SANITY_PROMPT,
        tokenizer_class=tokenizer_class,
        input_ids=input_ids,
        input_ids_sha256=token_ids_sha256(input_ids),
        output_ids=response.output_ids,
        output_ids_sha256=token_ids_sha256(response.output_ids),
        output_text=response.text,
        output_text_sha256=hashlib.sha256(response.text.encode()).hexdigest(),
        max_new_tokens=OLMOE_SANITY_MAX_NEW_TOKENS,
        sampling_seed=OLMOE_SANITY_SAMPLING_SEED,
        static_memory_fraction=static_memory_fraction,
        server_host=cast(Literal["127.0.0.1", "localhost"], server_host),
        server_port=server_port,
        server_info_sha256=server_info.canonical_response_sha256,
        managed_launch=managed_launch,
        producer_sha256=hash_sglang_kt_bound_file(
            Path(__file__).resolve(strict=True)
        ).sha256,
        captured_at_utc=_utc_now(),
    )
    return capture


def publish_sanity_capture(*, capture: SanityCapture, output: Path) -> str:
    return _atomic_publish(output, capture)


def _load_capture(path: Path) -> SanityCaptureBinding:
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=_CAPTURE_MAXIMUM_BYTES)
        capture = SanityCapture.model_validate_json(bound.contents)
    except (SglangKtReceiptFileError, ValidationError) as error:
        raise OlmoeStageContractError(f"invalid sanity capture: {path}") from error
    return SanityCaptureBinding(receipt_sha256=bound.sha256, capture=capture)


def publish_contract(
    *, snapshot_path: Path, ep1_capture: Path, ep2_capture: Path, output: Path
) -> str:
    snapshot = verify_pinned_snapshot(snapshot_path)
    first = _load_capture(ep1_capture)
    second = _load_capture(ep2_capture)
    producer_path = Path(__file__).resolve(strict=True)
    contract = OlmoeStageContract(
        schema_version=1,
        status="published",
        canonicalization="exo-olmoe-stage-contract-v1",
        snapshot=snapshot,
        ep1=first,
        ep2=second,
        published_at_utc=_utc_now(),
        producer_sha256=hash_sglang_kt_bound_file(producer_path).sha256,
    )
    return _atomic_publish(output, contract)


def load_stage_contract(
    path: Path,
    *,
    snapshot_path: Path = Path(OLMOE_MODEL_PATH),
    tokenizer_loader: Callable[[Path], tuple[_Tokenizer, str]] = _load_tokenizer,
) -> LoadedStageContract:
    try:
        bound = read_sglang_kt_bound_file(path, maximum_bytes=_CAPTURE_MAXIMUM_BYTES)
        contract = OlmoeStageContract.model_validate_json(
            bound.contents,
            context=_stage_contract_validation_context(bound.sha256),
        )
    except (SglangKtReceiptFileError, ValidationError) as error:
        raise OlmoeStageContractError(
            f"invalid OLMoE stage contract: {path}"
        ) from error
    observed_snapshot = verify_pinned_snapshot(snapshot_path)
    producer_path = Path(__file__).resolve(strict=True)
    expected_producer_sha256 = (
        PUBLISHED_STAGE_CONTRACT_V2_PRODUCER_SHA256
        if bound.sha256 == PUBLISHED_STAGE_CONTRACT_V2_SHA256
        else hash_sglang_kt_bound_file(producer_path).sha256
    )
    if contract.producer_sha256 != expected_producer_sha256:
        raise OlmoeStageContractError("stage contract producer source changed")
    if contract.snapshot != observed_snapshot:
        raise OlmoeStageContractError("stage contract snapshot changed")
    if tuple(
        (item.path, item.role, item.size_bytes, item.sha256, item.huggingface_etag)
        for item in contract.snapshot.files
    ) != tuple(
        (item.path, item.role, item.size_bytes, item.sha256, item.huggingface_etag)
        for item in PINNED_SNAPSHOT_FILES
    ):
        raise OlmoeStageContractError(
            "stage contract snapshot identities are not pinned"
        )
    tokenizer, tokenizer_class = tokenizer_loader(snapshot_path)
    observed_input_ids = tuple(
        tokenizer.encode(OLMOE_SANITY_PROMPT, add_special_tokens=True)
    )
    if (
        contract.ep1.capture.tokenizer_class != tokenizer_class
        or contract.ep2.capture.tokenizer_class != tokenizer_class
        or contract.ep1.capture.input_ids != observed_input_ids
        or contract.ep2.capture.input_ids != observed_input_ids
    ):
        raise OlmoeStageContractError("stage contract tokenizer binding changed")
    return LoadedStageContract(
        path=str(path), receipt_sha256=bound.sha256, contract=contract
    )


def _absolute_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("path must be absolute and normalized")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument(
        "--snapshot", type=_absolute_path, default=Path(OLMOE_MODEL_PATH)
    )
    publish.add_argument("--ep1-capture", required=True, type=_absolute_path)
    publish.add_argument("--ep2-capture", required=True, type=_absolute_path)
    publish.add_argument("--output", required=True, type=_absolute_path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        receipt_sha256 = publish_contract(
            snapshot_path=cast(Path, parsed.snapshot),
            ep1_capture=cast(Path, parsed.ep1_capture),
            ep2_capture=cast(Path, parsed.ep2_capture),
            output=cast(Path, parsed.output),
        )
    except (OlmoeStageContractError, OSError, ValueError) as error:
        print(f"OLMoE stage contract failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"receipt_sha256": receipt_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
