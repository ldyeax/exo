from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo.shared.types.worker.sglang_kt import SglangKtLaunchPlan
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
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
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.receipt_io import (
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
)
from scripts import create_sglang_kt_glm47_validation_process_spec as creator

GPU_UUID = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
MODEL_PATH = "/mnt/sanic/exo/models/glm-4.7-flash"
RUNTIME_PYTHON = "/var/lib/exo/runtimes/glm47/venv/bin/python"


def cli_arguments(
    tmp_path: Path,
    *,
    resident_gpu_experts: str = "4",
    cpu_cores: str = "0-3,8",
    memory_nodes: str = "0",
    cpu_infer_threads: str = "5",
    threadpool_count: str = "1",
    distributed_coordinator: str = "192.0.2.10:29510",
    service_endpoint: str = "192.0.2.10:30100",
    output: Path | None = None,
) -> list[str]:
    return [
        "--model-path",
        MODEL_PATH,
        "--runtime-python",
        RUNTIME_PYTHON,
        "--output",
        str(output or tmp_path / "process-spec.json"),
        "--node-id",
        "dwagon",
        "--gpu-uuid",
        GPU_UUID,
        "--cpu-cores",
        cpu_cores,
        "--memory-nodes",
        memory_nodes,
        "--cpu-infer-threads",
        cpu_infer_threads,
        "--threadpool-count",
        threadpool_count,
        "--distributed-coordinator",
        distributed_coordinator,
        "--service-endpoint",
        service_endpoint,
        "--resident-gpu-experts",
        resident_gpu_experts,
    ]


def parsed_arguments(
    tmp_path: Path,
    *,
    resident_gpu_experts: str = "4",
    cpu_cores: str = "0-3,8",
    memory_nodes: str = "0",
    cpu_infer_threads: str = "5",
    threadpool_count: str = "1",
    distributed_coordinator: str = "192.0.2.10:29510",
    service_endpoint: str = "192.0.2.10:30100",
    output: Path | None = None,
) -> creator.Glm47ValidationProcessSpecArguments:
    return creator.parse_arguments(
        cli_arguments(
            tmp_path,
            resident_gpu_experts=resident_gpu_experts,
            cpu_cores=cpu_cores,
            memory_nodes=memory_nodes,
            cpu_infer_threads=cpu_infer_threads,
            threadpool_count=threadpool_count,
            distributed_coordinator=distributed_coordinator,
            service_endpoint=service_endpoint,
            output=output,
        )
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_import_is_inert_and_does_not_require_hardware_or_model_runtimes() -> None:
    repository = Path(__file__).resolve().parents[2]
    program = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(repository)!r})

class BlockRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        blocked = ("torch", "sglang", "ktransformers", "kt_kernel", "pynvml")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("runtime import is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockRuntimeImports())
import scripts.create_sglang_kt_glm47_validation_process_spec
"""

    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("resident_gpu_experts", ("1", "2", "3", "4"))
def test_builds_exact_hybrid_pp1_validation_spec(
    tmp_path: Path, resident_gpu_experts: str
) -> None:
    process_spec = creator.create_process_spec(
        parsed_arguments(tmp_path, resident_gpu_experts=resident_gpu_experts)
    )
    plan = process_spec.plan
    stage = process_spec.stage

    assert plan.target_profile == GLM_4_7_FLASH_TARGET_PROFILE
    assert plan.model_id == GLM_4_7_FLASH_BF16_MODEL_ID
    assert plan.model_revision == GLM_4_7_FLASH_BF16_MODEL_REVISION
    assert plan.sglang_revision == GLM_4_7_FLASH_SGLANG_REVISION
    assert plan.ktransformers_revision == GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    assert plan.total_layers == GLM_4_7_FLASH_LAYER_COUNT
    assert plan.context_length == GLM_4_7_FLASH_CONTEXT_LENGTH
    assert plan.max_total_tokens == GLM_4_7_FLASH_MAX_TOTAL_TOKENS
    assert plan.static_memory_fraction == creator.STATIC_MEMORY_FRACTION
    assert plan.max_concurrent_requests == creator.MAX_CONCURRENT_REQUESTS
    assert len(plan.stages) == 1
    assert process_spec.pipeline_rank == 0
    assert process_spec.executable == RUNTIME_PYTHON
    assert process_spec.model_contract_sha256 == (
        GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    )
    assert stage.start_layer == 0
    assert stage.end_layer == GLM_4_7_FLASH_LAYER_COUNT
    assert stage.node_id == "dwagon"
    assert stage.gpu_uuid == GPU_UUID
    assert stage.model_path == MODEL_PATH
    assert stage.ktransformers_weight_path == MODEL_PATH
    assert stage.cpu_cores == (0, 1, 2, 3, 8)
    assert stage.memory_nodes == (0,)
    assert stage.cpu_infer_threads == 5
    assert stage.threadpool_count == 1
    assert stage.ktransformers_method == "BF16"
    assert stage.resident_gpu_experts == int(resident_gpu_experts)
    assert stage.max_deferred_experts_per_token == 0
    assert stage.hca_devices == ()
    assert process_spec.attention_backend == "flashinfer"
    assert process_spec.kv_cache_dtype == "bfloat16"
    assert argument_value(process_spec.arguments, "--pp-size") == "1"
    assert argument_value(process_spec.arguments, "--tp-size") == "1"
    assert (
        argument_value(process_spec.arguments, "--kt-expert-placement-strategy")
        == "uniform"
    )


def test_zero_residents_selects_cpu_control_profile(tmp_path: Path) -> None:
    process_spec = creator.create_process_spec(
        parsed_arguments(tmp_path, resident_gpu_experts="0")
    )

    assert (
        process_spec.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    )
    assert process_spec.stage.resident_gpu_experts == 0
    assert argument_value(process_spec.arguments, "--kt-num-gpu-experts") == "0"


def test_zero_and_nonzero_residents_call_the_corresponding_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu_builder = (
        creator.build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs
    )
    hybrid_builder = creator.build_glm_4_7_flash_bf16_process_launch_specs
    calls: list[str] = []

    def record_cpu(
        plan: SglangKtLaunchPlan, python_executable: str
    ) -> tuple[SglangKtProcessLaunchSpec, ...]:
        calls.append("cpu")
        return cpu_builder(plan, python_executable)

    def record_hybrid(
        plan: SglangKtLaunchPlan, python_executable: str
    ) -> tuple[SglangKtProcessLaunchSpec, ...]:
        calls.append("hybrid")
        return hybrid_builder(plan, python_executable)

    monkeypatch.setattr(
        creator,
        "build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs",
        record_cpu,
    )
    monkeypatch.setattr(
        creator, "build_glm_4_7_flash_bf16_process_launch_specs", record_hybrid
    )

    creator.create_process_spec(parsed_arguments(tmp_path, resident_gpu_experts="0"))
    creator.create_process_spec(parsed_arguments(tmp_path, resident_gpu_experts="1"))

    assert calls == ["cpu", "hybrid"]


def test_publishes_canonical_strict_json_and_both_digests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "process-spec.json"

    assert creator.main(cli_arguments(tmp_path, output=output)) == 0

    result = parse_sglang_kt_strict_json(capsys.readouterr().out.encode("ascii"))
    contents = output.read_bytes()
    payload = parse_sglang_kt_strict_json(contents)
    process_spec = SglangKtProcessLaunchSpec.model_validate_json(contents)
    assert contents == canonical_sglang_kt_json(payload)
    assert result == {
        "schema_version": creator.SCHEMA_VERSION,
        "output": str(output),
        "receipt_sha256": hashlib.sha256(contents).hexdigest(),
        "process_spec_sha256": calculate_sglang_kt_process_launch_spec_sha256(
            process_spec
        ),
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_canonical_serialization_strictly_reparses_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = creator.parse_sglang_kt_strict_json
    calls = 0

    def record_parse(contents: bytes) -> object:
        nonlocal calls
        calls += 1
        return original(contents)

    monkeypatch.setattr(creator, "parse_sglang_kt_strict_json", record_parse)

    creator.create_and_publish_process_spec(parsed_arguments(tmp_path))

    assert calls == 1


def test_refuses_to_replace_even_an_identical_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = cli_arguments(tmp_path)
    assert creator.main(arguments) == 0
    capsys.readouterr()
    original = (tmp_path / "process-spec.json").read_bytes()

    assert creator.main(arguments) == 1

    captured = capsys.readouterr()
    assert "refusing to replace an existing output" in captured.err
    assert (tmp_path / "process-spec.json").read_bytes() == original


def test_refuses_symlinked_output_name_without_changing_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "target.json"
    target.write_text("operator-owned\n")
    output = tmp_path / "process-spec.json"
    output.symlink_to(target)

    assert creator.main(cli_arguments(tmp_path, output=output)) == 1

    assert "refusing to replace an existing output" in capsys.readouterr().err
    assert target.read_text() == "operator-owned\n"
    assert output.is_symlink()


def test_refuses_symlinked_output_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output = linked_parent / "process-spec.json"

    assert creator.main(cli_arguments(tmp_path, output=output)) == 1

    assert "without following symlinks" in capsys.readouterr().err
    assert not (real_parent / output.name).exists()


def test_removes_final_link_if_output_parent_binding_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    output = output_parent / "process-spec.json"
    original_link = creator.os.link

    def replace_parent_then_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        output_parent.rename(displaced_parent)
        output_parent.mkdir()
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(creator.os, "link", replace_parent_then_link)

    with pytest.raises(creator.ProcessSpecCreationError, match="replaced"):
        creator.create_and_publish_process_spec(
            parsed_arguments(tmp_path, output=output)
        )

    assert not output.exists()
    assert not (displaced_parent / output.name).exists()
    assert not (displaced_parent / f".{output.name}.{os.getpid()}.tmp").exists()


@pytest.mark.parametrize(
    "value",
    (
        "",
        " 0",
        "0 ",
        "00",
        "0,,1",
        "1,0",
        "0,0",
        "0-0",
        "2-1",
        "0-2,2",
        "0-2,1-3",
        "+1",
        "-1",
    ),
)
def test_rejects_noncanonical_resource_lists(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        creator.parse_arguments(cli_arguments(tmp_path, cpu_cores=value))


@pytest.mark.parametrize(
    "value",
    (
        "192.0.2.1",
        "192.0.2.1:0",
        "192.0.2.1:065",
        "192.0.2.1:65536",
        "192.000.2.1:80",
        "0.0.0.0:80",
        "224.0.0.1:80",
        "[::1]:80",
    ),
)
def test_rejects_noncanonical_or_nonconcrete_endpoints(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(SystemExit):
        creator.parse_arguments(cli_arguments(tmp_path, distributed_coordinator=value))


@pytest.mark.parametrize(
    "option,value",
    (
        ("--model-path", "relative/model"),
        ("--model-path", "/mnt/sanic/../model"),
        ("--model-path", "//mnt/sanic/model"),
        ("--runtime-python", "/runtime/bin/python/"),
        ("--output", "relative/output.json"),
        ("--output", "/tmp/./output.json"),
    ),
)
def test_argument_parser_rejects_noncanonical_paths(
    tmp_path: Path, option: str, value: str
) -> None:
    arguments = cli_arguments(tmp_path)
    arguments[arguments.index(option) + 1] = value

    with pytest.raises(SystemExit):
        creator.parse_arguments(arguments)


@pytest.mark.parametrize("value", ("-1", "5", "04", "+1", "1.0"))
def test_argument_parser_rejects_unadmitted_resident_counts(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(SystemExit):
        creator.parse_arguments(cli_arguments(tmp_path, resident_gpu_experts=value))


def test_build_rejects_threads_beyond_cpu_assignment(tmp_path: Path) -> None:
    arguments = parsed_arguments(
        tmp_path,
        cpu_cores="0-3",
        cpu_infer_threads="5",
    )

    with pytest.raises(ValidationError, match="cpu_infer_threads"):
        creator.create_process_spec(arguments)


def test_build_rejects_threadpool_count_not_equal_to_memory_nodes(
    tmp_path: Path,
) -> None:
    arguments = parsed_arguments(
        tmp_path,
        memory_nodes="0-1",
        threadpool_count="1",
    )

    with pytest.raises(ValidationError, match="threadpool_count"):
        creator.create_process_spec(arguments)


def test_build_rejects_coordinator_equal_to_service_endpoint(tmp_path: Path) -> None:
    arguments = parsed_arguments(
        tmp_path,
        distributed_coordinator="192.0.2.10:30100",
        service_endpoint="192.0.2.10:30100",
    )

    with pytest.raises(ValidationError, match="distributed_coordinator"):
        creator.create_process_spec(arguments)


def test_build_rejects_output_inside_model_snapshot(tmp_path: Path) -> None:
    arguments = parsed_arguments(
        tmp_path,
        output=Path(MODEL_PATH) / "process-spec.json",
    )

    with pytest.raises(creator.ProcessSpecCreationError, match="outside"):
        creator.create_process_spec(arguments)
