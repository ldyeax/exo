#!/usr/bin/env python3
"""Create one pinned, inert GLM-4.7 validation process specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import suppress
from ipaddress import IPv4Address
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.dont_write_bytecode = True
    _repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repository_root))
    sys.path.insert(0, str(_repository_root / "src"))

from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import SglangKtLaunchPlan, SglangKtStageSpec
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CONTEXT_LENGTH,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_LAYER_COUNT,
    GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs,
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
)

SCHEMA_VERSION: Final = 1
STATIC_MEMORY_FRACTION: Final = 0.8
MAX_CONCURRENT_REQUESTS: Final = 1
MAX_VALIDATION_RESIDENT_GPU_EXPERTS: Final = 4

_GPU_UUID_PATTERN: Final = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_NODE_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ProcessSpecCreationError(RuntimeError):
    """Raised when a process specification cannot be created unambiguously."""


class Glm47ValidationProcessSpecArguments(argparse.Namespace):
    model_path: Path
    runtime_python: Path
    output: Path
    node_id: str
    gpu_uuid: str
    cpu_cores: tuple[int, ...]
    memory_nodes: tuple[int, ...]
    cpu_infer_threads: int
    threadpool_count: int
    distributed_coordinator: Host
    service_endpoint: Host
    resident_gpu_experts: int


def _absolute_normalized_path(value: str) -> Path:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or not path.is_absolute()
        or value.startswith("//")
        or path == PurePosixPath("/")
        or path.as_posix() != value
        or os.path.normpath(value) != value
    ):
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return Path(value)


def _node_id(value: str) -> str:
    if _NODE_ID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "node ID must contain only letters, digits, dots, underscores, or hyphens"
        )
    return value


def _gpu_uuid(value: str) -> str:
    if _GPU_UUID_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("GPU UUID must be a complete NVIDIA GPU UUID")
    return value


def _canonical_nonnegative_integer(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    parsed = int(value)
    if str(parsed) != value:
        raise argparse.ArgumentTypeError("integer must use canonical decimal notation")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = _canonical_nonnegative_integer(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _resident_gpu_experts(value: str) -> int:
    parsed = _canonical_nonnegative_integer(value)
    if parsed > MAX_VALIDATION_RESIDENT_GPU_EXPERTS:
        raise argparse.ArgumentTypeError("resident GPU experts must be between 0 and 4")
    return parsed


def _resource_indices(value: str) -> tuple[int, ...]:
    if not value or value.strip() != value:
        raise argparse.ArgumentTypeError("resource list must not be empty or padded")

    indices: list[int] = []
    for component in value.split(","):
        if not component or component.count("-") > 1:
            raise argparse.ArgumentTypeError("resource list has an invalid component")
        bounds = component.split("-")
        try:
            start = _canonical_nonnegative_integer(bounds[0])
            end = (
                start if len(bounds) == 1 else _canonical_nonnegative_integer(bounds[1])
            )
        except argparse.ArgumentTypeError as error:
            raise argparse.ArgumentTypeError(
                "resource list must use canonical nonnegative integers"
            ) from error
        if len(bounds) == 2 and end <= start:
            raise argparse.ArgumentTypeError(
                "resource ranges must end after their start"
            )
        indices.extend(range(start, end + 1))

    result = tuple(indices)
    if result != tuple(sorted(set(result))):
        raise argparse.ArgumentTypeError(
            "resource list must be sorted, unique, and non-overlapping"
        )
    return result


def _endpoint(value: str) -> Host:
    ip_text, separator, port_text = value.partition(":")
    if not separator or ":" in port_text:
        raise argparse.ArgumentTypeError("endpoint must use canonical IPv4:port form")
    try:
        address = IPv4Address(ip_text)
        port = _positive_integer(port_text)
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError(
            "endpoint must use canonical IPv4:port form"
        ) from error
    if str(address) != ip_text or not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            "endpoint must use canonical IPv4 and a port between 1 and 65535"
        )
    if address.is_unspecified or address.is_multicast:
        raise argparse.ArgumentTypeError("endpoint must use a concrete IPv4 address")
    return Host(ip=ip_text, port=port)


def parse_arguments(
    arguments: list[str] | None = None,
) -> Glm47ValidationProcessSpecArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=_absolute_normalized_path)
    parser.add_argument(
        "--runtime-python", required=True, type=_absolute_normalized_path
    )
    parser.add_argument("--output", required=True, type=_absolute_normalized_path)
    parser.add_argument("--node-id", required=True, type=_node_id)
    parser.add_argument("--gpu-uuid", required=True, type=_gpu_uuid)
    parser.add_argument("--cpu-cores", required=True, type=_resource_indices)
    parser.add_argument("--memory-nodes", required=True, type=_resource_indices)
    parser.add_argument("--cpu-infer-threads", required=True, type=_positive_integer)
    parser.add_argument("--threadpool-count", required=True, type=_positive_integer)
    parser.add_argument("--distributed-coordinator", required=True, type=_endpoint)
    parser.add_argument("--service-endpoint", required=True, type=_endpoint)
    parser.add_argument(
        "--resident-gpu-experts", required=True, type=_resident_gpu_experts
    )
    namespace = Glm47ValidationProcessSpecArguments()
    parser.parse_args(arguments, namespace=namespace)
    return namespace


def _argument_value(arguments: tuple[str, ...], option: str) -> str:
    try:
        index = arguments.index(option)
    except ValueError as error:
        raise ProcessSpecCreationError(
            f"generated process arguments omit {option}"
        ) from error
    if index + 1 >= len(arguments):
        raise ProcessSpecCreationError(
            f"generated process argument {option} has no value"
        )
    return arguments[index + 1]


def _validate_generated_spec(process_spec: SglangKtProcessLaunchSpec) -> None:
    plan = process_spec.plan
    stage = process_spec.stage
    expected_profile = (
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        if stage.resident_gpu_experts == 0
        else GLM_4_7_FLASH_TARGET_PROFILE
    )
    if (
        plan.target_profile != expected_profile
        or plan.model_id != GLM_4_7_FLASH_BF16_MODEL_ID
        or plan.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
        or plan.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION
        or plan.ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
        or plan.total_layers != GLM_4_7_FLASH_LAYER_COUNT
        or plan.context_length != GLM_4_7_FLASH_CONTEXT_LENGTH
        or plan.max_total_tokens != GLM_4_7_FLASH_MAX_TOTAL_TOKENS
        or plan.static_memory_fraction != STATIC_MEMORY_FRACTION
        or plan.max_concurrent_requests != MAX_CONCURRENT_REQUESTS
        or len(plan.stages) != 1
        or process_spec.pipeline_rank != 0
        or stage.start_layer != 0
        or stage.end_layer != GLM_4_7_FLASH_LAYER_COUNT
        or stage.model_path != stage.ktransformers_weight_path
        or stage.ktransformers_method != "BF16"
        or stage.max_deferred_experts_per_token != 0
        or stage.hca_devices
        or process_spec.attention_backend != "flashinfer"
        or process_spec.kv_cache_dtype != "bfloat16"
        or _argument_value(process_spec.arguments, "--pp-size") != "1"
        or _argument_value(process_spec.arguments, "--tp-size") != "1"
        or _argument_value(process_spec.arguments, "--kt-expert-placement-strategy")
        != "uniform"
    ):
        raise ProcessSpecCreationError(
            "generated GLM-4.7 validation process invariants are not pinned"
        )


def create_process_spec(
    arguments: Glm47ValidationProcessSpecArguments,
) -> SglangKtProcessLaunchSpec:
    if arguments.output == arguments.model_path or arguments.model_path in (
        arguments.output.parents
    ):
        raise ProcessSpecCreationError("output must be outside the model snapshot")
    if arguments.output == arguments.runtime_python:
        raise ProcessSpecCreationError("output must differ from the runtime executable")

    target_profile = (
        GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
        if arguments.resident_gpu_experts == 0
        else GLM_4_7_FLASH_TARGET_PROFILE
    )
    stage = SglangKtStageSpec(
        pipeline_rank=0,
        start_layer=0,
        end_layer=GLM_4_7_FLASH_LAYER_COUNT,
        node_id=NodeId(arguments.node_id),
        gpu_uuid=arguments.gpu_uuid,
        service_endpoint=arguments.service_endpoint,
        model_path=str(arguments.model_path),
        ktransformers_weight_path=str(arguments.model_path),
        cpu_cores=arguments.cpu_cores,
        memory_nodes=arguments.memory_nodes,
        cpu_infer_threads=arguments.cpu_infer_threads,
        threadpool_count=arguments.threadpool_count,
        ktransformers_method="BF16",
        resident_gpu_experts=arguments.resident_gpu_experts,
        max_deferred_experts_per_token=0,
        hca_devices=(),
    )
    plan = SglangKtLaunchPlan(
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        target_profile=target_profile,
        total_layers=GLM_4_7_FLASH_LAYER_COUNT,
        context_length=GLM_4_7_FLASH_CONTEXT_LENGTH,
        max_total_tokens=GLM_4_7_FLASH_MAX_TOTAL_TOKENS,
        static_memory_fraction=STATIC_MEMORY_FRACTION,
        max_concurrent_requests=MAX_CONCURRENT_REQUESTS,
        distributed_coordinator=arguments.distributed_coordinator,
        rank_zero_endpoint=arguments.service_endpoint,
        stages=(stage,),
    )
    if arguments.resident_gpu_experts == 0:
        process_specs = (
            build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
                plan, str(arguments.runtime_python)
            )
        )
    else:
        process_specs = build_glm_4_7_flash_bf16_process_launch_specs(
            plan, str(arguments.runtime_python)
        )
    if len(process_specs) != 1:
        raise ProcessSpecCreationError(
            "GLM-4.7 validation builder did not create exactly one process"
        )
    process_spec = process_specs[0]
    _validate_generated_spec(process_spec)
    return process_spec


def _open_absolute_directory(path: Path) -> int:
    descriptor = os.open("/", os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ProcessSpecCreationError(
            f"cannot open output parent without following symlinks: {path}"
        ) from error


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise ProcessSpecCreationError("output parent is not a directory")
    return observed.st_dev, observed.st_ino, observed.st_mode


def _require_current_directory_binding(
    path: Path, identity: tuple[int, int, int]
) -> None:
    current_descriptor = _open_absolute_directory(path)
    try:
        if _directory_identity(current_descriptor) != identity:
            raise ProcessSpecCreationError(
                f"output parent was replaced while publishing: {path}"
            )
    finally:
        os.close(current_descriptor)


def publish_new_process_spec(path: Path, contents: bytes) -> None:
    """Publish one new file without following links or replacing any name."""

    directory_descriptor = _open_absolute_directory(path.parent)
    directory_identity = _directory_identity(directory_descriptor)
    temporary_name = f".{path.name}.{os.getpid()}.tmp"
    descriptor = -1
    final_linked = False
    publication_complete = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_descriptor,
        )
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProcessSpecCreationError("short process-spec write")
            view = view[written:]
        os.fsync(descriptor)
        written_stat = os.fstat(descriptor)
        if not stat.S_ISREG(written_stat.st_mode) or written_stat.st_nlink != 1:
            raise ProcessSpecCreationError(
                "temporary process spec is not a singly linked regular file"
            )
        os.close(descriptor)
        descriptor = -1

        _require_current_directory_binding(path.parent, directory_identity)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final_linked = True
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        _require_current_directory_binding(path.parent, directory_identity)
        publication_complete = True
    except FileExistsError as error:
        raise ProcessSpecCreationError(
            f"refusing to replace an existing output: {path}"
        ) from error
    except OSError as error:
        raise ProcessSpecCreationError(
            f"cannot publish process spec: {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if final_linked and not publication_complete:
            with suppress(FileNotFoundError):
                os.unlink(path.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
        if not final_linked:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=directory_descriptor)
        os.close(directory_descriptor)


def canonical_process_spec_contents(
    process_spec: SglangKtProcessLaunchSpec,
) -> bytes:
    """Serialize and strictly reparse a process spec before publication."""

    contents = canonical_sglang_kt_json(process_spec.model_dump(mode="json"))
    parse_sglang_kt_strict_json(contents)
    reparsed_spec = SglangKtProcessLaunchSpec.model_validate_json(contents)
    if reparsed_spec != process_spec:
        raise ProcessSpecCreationError(
            "canonical process spec changed while it was reparsed"
        )
    if canonical_sglang_kt_json(reparsed_spec.model_dump(mode="json")) != contents:
        raise ProcessSpecCreationError("process spec serialization is not canonical")
    return contents


def create_and_publish_process_spec(
    arguments: Glm47ValidationProcessSpecArguments,
) -> dict[str, object]:
    process_spec = create_process_spec(arguments)
    contents = canonical_process_spec_contents(process_spec)
    publish_new_process_spec(arguments.output, contents)
    return {
        "schema_version": SCHEMA_VERSION,
        "output": str(arguments.output),
        "receipt_sha256": hashlib.sha256(contents).hexdigest(),
        "process_spec_sha256": calculate_sglang_kt_process_launch_spec_sha256(
            process_spec
        ),
    }


def main(arguments: list[str] | None = None) -> int:
    try:
        result = create_and_publish_process_spec(parse_arguments(arguments))
    except (
        ProcessSpecCreationError,
        SglangKtReceiptFileError,
        ValidationError,
        ValueError,
        OSError,
    ) as error:
        print(f"process spec creation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
