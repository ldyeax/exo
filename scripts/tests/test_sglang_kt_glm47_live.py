import contextlib
import errno
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import SglangKtLaunchPlan, SglangKtStageSpec
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_process_launch_specs,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    load_sglang_kt_kernel_runtime_validation_receipt,
)
from scripts import sglang_kt_glm47_live as live

GPU_UUID = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
PYTHON_EXECUTABLE = "/var/lib/exo/runtime/overlay/venv/bin/python"


def make_process_spec():
    plan = SglangKtLaunchPlan(
        target_profile=GLM_4_7_FLASH_TARGET_PROFILE,
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        total_layers=47,
        context_length=202_752,
        max_total_tokens=4_096,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(ip="192.0.2.1", port=29510),
        rank_zero_endpoint=Host(ip="192.0.2.1", port=30100),
        stages=(
            SglangKtStageSpec(
                pipeline_rank=0,
                start_layer=0,
                end_layer=47,
                node_id=NodeId("dwagon"),
                gpu_uuid=GPU_UUID,
                service_endpoint=Host(ip="192.0.2.1", port=30100),
                model_path="/mnt/sanic/models/glm47",
                ktransformers_weight_path="/mnt/sanic/models/glm47",
                cpu_cores=(0, 1, 2, 3),
                memory_nodes=(0,),
                cpu_infer_threads=4,
                threadpool_count=1,
                ktransformers_method="BF16",
                resident_gpu_experts=1,
                max_deferred_experts_per_token=0,
                hca_devices=(),
            ),
        ),
    )
    return build_glm_4_7_flash_bf16_process_launch_specs(
        plan,
        PYTHON_EXECUTABLE,
    )[0]


def make_kernel_observation():
    fixture = (
        Path(__file__).parents[2]
        / "src/exo/worker/tests/fixtures/sglang_kt/glm47_kernel_runtime_v1_dwagon_v4.json"
    )
    observed = load_sglang_kt_kernel_runtime_validation_receipt(fixture)
    return observed.model_copy(
        update={
            "executable": PYTHON_EXECUTABLE,
            "cpu_cores": (0, 1, 2, 3),
            "memory_nodes": (0,),
            "threads_per_subpool": (4,),
        }
    )


def test_server_argument_vector_is_exact_launch_suffix() -> None:
    process_spec = make_process_spec()

    arguments = live.sglang_server_argument_vector(process_spec)

    assert arguments == process_spec.arguments[2:]
    assert arguments[:2] == ("--model-path", "/mnt/sanic/models/glm47")
    assert "--disable-cuda-graph" in arguments


def test_child_environment_applies_exact_controls_without_profiler_inheritance() -> (
    None
):
    environment = live.build_live_child_environment(
        make_process_spec(),
        {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "GPU-old",
            "NCCL_DEBUG": "TRACE",
            "SGLANG_TORCH_PROFILER_DIR": "/tmp/profile",
            "AMPLXE_EXPERIMENTAL": "1",
            "VTUNE_PROFILER_DIR": "/tmp/vtune",
            "LD_PRELOAD": "/tmp/profiler.so",
            "PYTHONPATH": "/tmp/injected",
        },
    )

    assert environment == {
        "PATH": "/usr/bin",
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_KT_HYBRID_TIMING": "1",
    }


@pytest.mark.parametrize(
    "name",
    (
        "LD_PRELOAD",
        "LD_AUDIT",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "AMPLXE_COLLECT",
        "VTUNE_PROFILER_DIR",
        "NCCL_DEBUG",
    ),
)
def test_child_environment_rejects_forbidden_process_spec_entries(name: str) -> None:
    base_process_spec = make_process_spec()
    process_spec = Mock(spec=SglangKtProcessLaunchSpec)
    process_spec.environment = ((name, "unsafe"),)
    process_spec.unset_environment_variables = (
        base_process_spec.unset_environment_variables
    )
    process_spec.unset_environment_variable_prefixes = (
        base_process_spec.unset_environment_variable_prefixes
    )

    with pytest.raises(live.Glm47LiveValidationError, match=rf"forbidden.*{name}"):
        live.build_live_child_environment(
            cast(SglangKtProcessLaunchSpec, process_spec),
            {"PATH": "/usr/bin"},
        )


def test_child_environment_rejects_an_explicitly_unset_spec_entry() -> None:
    process_spec = Mock(spec=SglangKtProcessLaunchSpec)
    process_spec.environment = (("EXO_UNSET_CONTROL", "unsafe"),)
    process_spec.unset_environment_variables = ("EXO_UNSET_CONTROL",)
    process_spec.unset_environment_variable_prefixes = ()

    with pytest.raises(
        live.Glm47LiveValidationError,
        match="forbidden.*EXO_UNSET_CONTROL",
    ):
        live.build_live_child_environment(
            cast(SglangKtProcessLaunchSpec, process_spec),
            {"PATH": "/usr/bin"},
        )


def test_kernel_binding_accepts_exact_observation_and_rejects_overlay_loss() -> None:
    process_spec = make_process_spec()
    receipt = make_kernel_observation()

    live.require_kernel_runtime_binding(process_spec, receipt)

    resolved = receipt.model_copy(
        update={"executable": "/var/lib/exo/python/bin/python3.12"}
    )
    with pytest.raises(live.Glm47LiveValidationError, match="does not match"):
        live.require_kernel_runtime_binding(process_spec, resolved)


@pytest.mark.parametrize(
    (
        "executable",
        "hostname",
        "affinity_cpu_ids",
        "memory_policy_nodes",
        "cuda_visible_devices",
    ),
    (
        (PYTHON_EXECUTABLE, "fwuff", (0, 1, 2, 3), (0,), GPU_UUID),
        (PYTHON_EXECUTABLE, "dwagon", (0, 1, 2), (0,), GPU_UUID),
        (PYTHON_EXECUTABLE, "dwagon", (0, 1, 2, 3), (0, 1), GPU_UUID),
        (PYTHON_EXECUTABLE, "dwagon", (0, 1, 2, 3), (0,), "GPU-wrong"),
        ("/resolved/python", "dwagon", (0, 1, 2, 3), (0,), GPU_UUID),
    ),
)
def test_current_process_binding_is_exact(
    executable: str,
    hostname: str,
    affinity_cpu_ids: tuple[int, ...],
    memory_policy_nodes: tuple[int, ...],
    cuda_visible_devices: str,
) -> None:
    with pytest.raises(live.Glm47LiveValidationError, match="not bound"):
        live.require_current_process_binding(
            make_process_spec(),
            executable=executable,
            hostname=hostname,
            affinity_cpu_ids=affinity_cpu_ids,
            memory_policy_nodes=memory_policy_nodes,
            cuda_visible_devices=cuda_visible_devices,
        )


def test_current_process_binding_accepts_exact_values() -> None:
    live.require_current_process_binding(
        make_process_spec(),
        executable=PYTHON_EXECUTABLE,
        hostname="dwagon",
        affinity_cpu_ids=(0, 1, 2, 3),
        memory_policy_nodes=(0,),
        cuda_visible_devices=GPU_UUID,
    )


def test_bound_memory_policy_requires_exact_mpol_bind_nodes() -> None:
    assert live.parse_bound_memory_policy("policy: bind\nmembind: 0 2\n") == (0, 2)

    with pytest.raises(live.Glm47LiveValidationError, match="explicit bound"):
        live.parse_bound_memory_policy("policy: default\nmembind: 0 1\n")
    with pytest.raises(live.Glm47LiveValidationError, match="duplicate"):
        live.parse_bound_memory_policy("policy: bind\nmembind: 0\nmembind: 1\n")


def test_validator_bundle_is_sorted_content_bound_and_domain_separated(
    tmp_path: Path,
) -> None:
    paths = live.validator_bundle_paths(tmp_path)
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source {index}\n".encode())

    bundle = live.calculate_validator_bundle((*reversed(paths), paths[0]))

    assert tuple(source.path for source in bundle.sources) == tuple(map(str, paths))
    source_pairs = tuple((source.path, source.sha256) for source in bundle.sources)
    assert bundle.sha256 == (
        live.calculate_sglang_kt_model_runtime_validator_bundle_sha256(source_pairs)
    )

    paths[0].write_bytes(b"changed\n")
    assert live.calculate_validator_bundle(paths).sha256 != bundle.sha256


def test_validator_bundle_rejects_relative_and_symlink_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        live.calculate_validator_bundle((Path("validator.py"),))
    paths = live.validator_bundle_paths(tmp_path)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    source = paths[0]
    source.unlink()
    target = tmp_path / "real-live.py"
    target.write_text("pass\n")
    source.symlink_to(target)
    with pytest.raises(live.Glm47LiveValidationError, match="cannot bind"):
        live.calculate_validator_bundle(paths)


def test_memory_headroom_and_memavailable_parsing(tmp_path: Path) -> None:
    evidence = live.calculate_memory_headroom(
        available_bytes=100,
        required_bytes=60,
        minimum_headroom_bytes=40,
    )
    assert evidence.remaining_bytes == 40
    with pytest.raises(live.Glm47LiveValidationError, match="39 bytes"):
        live.calculate_memory_headroom(
            available_bytes=99,
            required_bytes=60,
            minimum_headroom_bytes=40,
        )

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 200 kB\nMemAvailable: 123 kB\n")
    assert live.available_host_memory_bytes(meminfo) == 123 * 1024


def test_process_spec_keeps_model_contract_identity() -> None:
    assert make_process_spec().model_contract_sha256 == (
        GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    )


def test_disposable_child_evidence_is_visible_only_after_clean_exit(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child.py"
    child.write_text(
        "import os, sys\n"
        "fd = int(sys.argv[sys.argv.index('--evidence-fd') + 1])\n"
        'os.write(fd, b\'{\\"phase\\":\\"cleaned\\"}\')\n'
        "os.fsync(fd)\n"
    )

    evidence = live.run_disposable_live_child(
        (sys.executable, str(child)),
        evidence_descriptor_argument="--evidence-fd",
        environment=os.environ.copy(),
    )

    assert evidence.contents == b'{"phase":"cleaned"}'
    assert evidence.sha256 == hashlib.sha256(evidence.contents).hexdigest()


def test_forked_descendant_cannot_mutate_sealed_child_evidence(
    tmp_path: Path,
) -> None:
    child = tmp_path / "forking-child.py"
    ready_path = tmp_path / "descendant-ready"
    result_path = tmp_path / "mutation-result"
    descendant_pid_path = tmp_path / "descendant-pid"
    child.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "ready, result, pid_path = map(Path, sys.argv[1:4])\n"
        "fd = int(sys.argv[sys.argv.index('--evidence-fd') + 1])\n"
        "descendant = os.fork()\n"
        "if descendant == 0:\n"
        "    def attempt_mutation(_signal, _frame):\n"
        "        try:\n"
        "            os.pwrite(fd, b'X', 0)\n"
        "        except OSError as error:\n"
        "            result.write_text(str(error.errno))\n"
        "        else:\n"
        "            result.write_text('mutated')\n"
        "        os._exit(0)\n"
        "    signal.signal(signal.SIGTERM, attempt_mutation)\n"
        "    ready.write_text('ready')\n"
        "    while True:\n"
        "        signal.pause()\n"
        "pid_path.write_text(str(descendant))\n"
        "deadline = time.monotonic() + 5\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.005)\n"
        "if not ready.exists():\n"
        "    raise SystemExit(8)\n"
        'os.write(fd, b\'{\\"phase\\":\\"sealed\\"}\')\n'
        "os.fsync(fd)\n"
    )

    evidence = live.run_disposable_live_child(
        (
            sys.executable,
            str(child),
            str(ready_path),
            str(result_path),
            str(descendant_pid_path),
        ),
        evidence_descriptor_argument="--evidence-fd",
        environment=os.environ.copy(),
    )

    deadline = time.monotonic() + 2
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert evidence.contents == b'{"phase":"sealed"}'
    assert result_path.read_text() == str(errno.EPERM)
    descendant_pid = int(descendant_pid_path.read_text())
    assert not _process_is_running(descendant_pid)


def test_interruption_kills_forked_descendant_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "interrupt-child.py"
    ready_path = tmp_path / "interrupt-ready"
    process_ids_path = tmp_path / "process-ids"
    mutation_path = tmp_path / "late-mutation"
    child.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "ready, process_ids, mutation = map(Path, sys.argv[1:4])\n"
        "descendant = os.fork()\n"
        "if descendant == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(0.5)\n"
        "    mutation.write_text('survived')\n"
        "    while True:\n"
        "        time.sleep(1)\n"
        "process_ids.write_text(f'{os.getpid()} {descendant}')\n"
        "ready.write_text('ready')\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )

    class ExpectedInterruption(BaseException):
        pass

    def interrupt_after_fork(_process: object) -> int:
        deadline = time.monotonic() + 5
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        if not ready_path.exists():
            raise AssertionError("forking child did not become ready")
        raise ExpectedInterruption

    monkeypatch.setattr(
        live,
        "_wait_for_direct_child_exit_without_reaping",
        interrupt_after_fork,
    )
    try:
        with pytest.raises(ExpectedInterruption):
            live.run_disposable_live_child(
                (
                    sys.executable,
                    str(child),
                    str(ready_path),
                    str(process_ids_path),
                    str(mutation_path),
                ),
                evidence_descriptor_argument="--evidence-fd",
                environment=os.environ.copy(),
            )
        time.sleep(0.55)
        process_ids = tuple(map(int, process_ids_path.read_text().split()))
        assert not mutation_path.exists()
        assert all(not _process_is_running(process_id) for process_id in process_ids)
    finally:
        if process_ids_path.exists():
            for process_id in map(int, process_ids_path.read_text().split()):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process_id, 9)


def test_child_evidence_sealing_failure_is_not_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "seal-failure-child.py"
    child.write_text(
        "import os, sys\n"
        "fd = int(sys.argv[sys.argv.index('--evidence-fd') + 1])\n"
        "os.write(fd, b'{}')\n"
    )

    def reject_seal(_descriptor: int, _operation: int, _argument: int = 0) -> int:
        raise OSError(errno.EPERM, "sealing rejected")

    monkeypatch.setattr(live.fcntl, "fcntl", reject_seal)
    with pytest.raises(live.Glm47LiveValidationError, match="cannot seal"):
        live.run_disposable_live_child(
            (sys.executable, str(child)),
            evidence_descriptor_argument="--evidence-fd",
            environment=os.environ.copy(),
        )


def test_disposable_child_failure_does_not_admit_written_evidence(
    tmp_path: Path,
) -> None:
    child = tmp_path / "failing-child.py"
    child.write_text(
        "import os, sys\n"
        "fd = int(sys.argv[sys.argv.index('--evidence-fd') + 1])\n"
        'os.write(fd, b\'{\\"status\\":\\"passed\\"}\')\n'
        "raise SystemExit(9)\n"
    )

    with pytest.raises(live.Glm47LiveValidationError, match="status 9"):
        live.run_disposable_live_child(
            (sys.executable, str(child)),
            evidence_descriptor_argument="--evidence-fd",
            environment=os.environ.copy(),
        )


def test_child_evidence_writer_rejects_invalid_descriptor() -> None:
    with pytest.raises(ValueError, match="inherited"):
        live.write_disposable_child_evidence(2, {"status": "passed"})


def _process_is_running(process_id: int) -> bool:
    try:
        fields = Path(f"/proc/{process_id}/stat").read_text().split()
    except FileNotFoundError:
        return False
    return len(fields) > 2 and fields[2] != "Z"
