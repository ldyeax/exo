import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import (
    SglangKtLaunchPlan,
    SglangKtStageSpec,
    SglangKtTargetProfile,
)
from exo.worker.sglang_kt import model_contract as model_contract_module
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs,
    build_glm_4_7_flash_bf16_process_launch_specs,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
)
from exo.worker.sglang_kt.receipt_io import (
    canonical_sglang_kt_json,
    parse_sglang_kt_strict_json,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from scripts import validate_sglang_kt_glm47_model as validator
from scripts.sglang_kt_glm47_live import validator_bundle_paths

GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"
MODEL_PATH = "/var/lib/exo/models/glm-4.7-flash-bf16"
PYTHON_EXECUTABLE = "/var/lib/exo/runtimes/glm47/bin/python"


def make_plan(
    *,
    resident_gpu_experts: int = 4,
    target_profile: SglangKtTargetProfile = GLM_4_7_FLASH_TARGET_PROFILE,
) -> SglangKtLaunchPlan:
    return SglangKtLaunchPlan(
        target_profile=target_profile,
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
        total_layers=47,
        context_length=202_752,
        max_total_tokens=4_096,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(ip="192.168.40.248", port=29_510),
        rank_zero_endpoint=Host(ip="192.168.40.248", port=30_100),
        stages=(
            SglangKtStageSpec(
                pipeline_rank=0,
                start_layer=0,
                end_layer=47,
                node_id=NodeId("dwagon"),
                gpu_uuid=GPU_UUID,
                service_endpoint=Host(ip="192.168.40.248", port=30_100),
                model_path=MODEL_PATH,
                ktransformers_weight_path=MODEL_PATH,
                cpu_cores=(0, 1, 2, 3),
                memory_nodes=(0,),
                cpu_infer_threads=4,
                threadpool_count=1,
                ktransformers_method="BF16",
                resident_gpu_experts=resident_gpu_experts,
                max_deferred_experts_per_token=0,
                hca_devices=(),
            ),
        ),
    )


def make_process_spec(*, resident_gpu_experts: int = 4) -> SglangKtProcessLaunchSpec:
    if resident_gpu_experts == 0:
        plan = make_plan(
            resident_gpu_experts=0,
            target_profile=GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
        )
        return build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
            plan,
            PYTHON_EXECUTABLE,
        )[0]
    return build_glm_4_7_flash_bf16_process_launch_specs(
        make_plan(resident_gpu_experts=resident_gpu_experts),
        PYTHON_EXECUTABLE,
    )[0]


def write_inputs(
    tmp_path: Path,
    *,
    resident_gpu_experts: int = 4,
) -> tuple[list[str], SglangKtProcessLaunchSpec, Path, Path, Path, Path]:
    process_spec = make_process_spec(resident_gpu_experts=resident_gpu_experts)
    process_path = tmp_path / "process-spec.json"
    process_path.write_bytes(
        canonical_sglang_kt_json(process_spec.model_dump(mode="json"))
    )

    packaged_contract_path = (
        Path(model_contract_module.__file__).parent
        / "manifests"
        / GLM_4_7_FLASH_BF16_MODEL_CONTRACT_FILENAME
    )
    contract_path = tmp_path / "model-contract.json"
    contract_contents = packaged_contract_path.read_bytes()
    contract_path.write_bytes(contract_contents)

    kernel_path = tmp_path / "kernel-receipt.json"
    kernel_contents = canonical_sglang_kt_json(
        {"schema_version": 1, "status": "passed", "profiler": "none"}
    )
    kernel_path.write_bytes(kernel_contents)
    output_path = tmp_path / "model-runtime-receipt.json"
    validator_repository_root = tmp_path / "validator-repository"
    for index, validator_path in enumerate(
        validator_bundle_paths(validator_repository_root)
    ):
        validator_path.parent.mkdir(parents=True, exist_ok=True)
        validator_path.write_text(f"# exact validator source {index}\n")

    arguments = [
        "--process-spec",
        str(process_path),
        "--expected-process-spec-sha256",
        calculate_sglang_kt_process_launch_spec_sha256(process_spec),
        "--model-contract",
        str(contract_path),
        "--expected-model-contract-receipt-sha256",
        hashlib.sha256(contract_contents).hexdigest(),
        "--kernel-runtime-receipt",
        str(kernel_path),
        "--expected-kernel-receipt-sha256",
        hashlib.sha256(kernel_contents).hexdigest(),
        "--output",
        str(output_path),
    ]
    return (
        arguments,
        process_spec,
        process_path,
        contract_path,
        kernel_path,
        validator_repository_root,
    )


def replace_argument(arguments: list[str], name: str, value: str) -> list[str]:
    replaced = arguments.copy()
    replaced[replaced.index(name) + 1] = value
    return replaced


def test_cli_parses_exact_required_paths_and_digests(tmp_path: Path) -> None:
    arguments, process_spec, process_path, contract_path, kernel_path, _validator = (
        write_inputs(tmp_path)
    )

    parsed = validator.parse_arguments(arguments)

    assert parsed.process_spec == process_path
    assert parsed.expected_process_spec_sha256 == (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    )
    assert parsed.model_contract == contract_path
    assert parsed.expected_model_contract_receipt_sha256 == (
        GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    )
    assert parsed.kernel_runtime_receipt == kernel_path
    assert parsed.output == tmp_path / "model-runtime-receipt.json"


@pytest.mark.parametrize(
    "argument_name",
    (
        "--process-spec",
        "--model-contract",
        "--kernel-runtime-receipt",
        "--output",
    ),
)
def test_cli_rejects_relative_paths(
    tmp_path: Path,
    argument_name: str,
) -> None:
    arguments, *_rest = write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        validator.parse_arguments(
            replace_argument(arguments, argument_name, "relative.json")
        )


@pytest.mark.parametrize(
    "argument_name",
    (
        "--expected-process-spec-sha256",
        "--expected-model-contract-receipt-sha256",
        "--expected-kernel-receipt-sha256",
    ),
)
@pytest.mark.parametrize("digest", ("a" * 63, "A" * 64, "g" * 64))
def test_cli_rejects_noncanonical_digests(
    tmp_path: Path,
    argument_name: str,
    digest: str,
) -> None:
    arguments, *_rest = write_inputs(tmp_path)

    with pytest.raises(SystemExit):
        validator.parse_arguments(replace_argument(arguments, argument_name, digest))


@pytest.mark.parametrize("resident_gpu_experts", (0, 1, 4))
def test_software_preflight_binds_exact_inputs_without_publishing_capability(
    tmp_path: Path,
    resident_gpu_experts: int,
) -> None:
    (
        arguments,
        process_spec,
        process_path,
        _contract,
        kernel_path,
        validator_repository_root,
    ) = write_inputs(tmp_path, resident_gpu_experts=resident_gpu_experts)
    parsed = validator.parse_arguments(arguments)

    preflight = validator.perform_software_preflight(
        parsed,
        validator_repository_root=validator_repository_root,
    )
    payload = validator.software_preflight_payload(preflight)

    assert preflight.process.path == process_path
    assert preflight.process.process_spec == process_spec
    assert preflight.process.process_spec_sha256 == (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    )
    assert preflight.model_contract.contract_sha256 == (
        GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
    )
    assert preflight.kernel_runtime_receipt.path == kernel_path
    assert tuple(source.path for source in preflight.validator.sources) == tuple(
        map(str, validator_bundle_paths(validator_repository_root))
    )
    assert preflight.output == tmp_path / "model-runtime-receipt.json"
    assert not preflight.output.exists()
    assert payload["runtime_probe_executed"] is False
    assert "capabilities" not in payload


def test_execution_preflight_verifies_the_exact_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, process_spec, *_rest, validator_repository_root = write_inputs(tmp_path)
    parsed = validator.parse_arguments(arguments)
    software = validator.perform_software_preflight(
        parsed,
        validator_repository_root=validator_repository_root,
    )
    contract = software.model_contract.contract
    snapshot = SglangKtVerifiedModelSnapshot(
        model_path=process_spec.model_path,
        model_id=contract.model_id,
        revision=contract.revision,
        weight_format=contract.weight_format,
        ktransformers_method=contract.ktransformers_method,
        full_indexer_layer_starts=contract.full_indexer_layer_starts,
        contract_path=software.model_contract.path,
        contract_receipt_sha256=software.model_contract.receipt_sha256,
        contract_sha256=software.model_contract.contract_sha256,
        config_sha256=next(
            file.sha256 for file in contract.files if file.role == "config"
        ),
        index_sha256=next(
            file.sha256 for file in contract.files if file.role == "safetensors_index"
        ),
        weight_map_entries=contract.weight_map_entries,
        shard_count=sum(file.role == "weight_shard" for file in contract.files),
        physical_weight_bytes=contract.physical_weight_bytes,
    )
    immutable_model_paths: list[Path] = []
    verification_calls: list[tuple[Path, Path, dict[str, object]]] = []
    preflight_order: list[str] = []

    def accept_source_deployment(
        repository_root: Path,
        bundle: validator.ValidatorBundleIdentity,
        *,
        required_uid: int = 0,
    ) -> None:
        assert repository_root == validator_repository_root
        assert bundle == software.validator
        assert required_uid == 0

    def accept_model_snapshot(
        snapshot_path: Path,
        *,
        required_uid: int = 0,
    ) -> None:
        assert required_uid == 0
        preflight_order.append("model_snapshot")
        immutable_model_paths.append(snapshot_path)

    def accept_kernel_receipt(
        observed_process_spec: SglangKtProcessLaunchSpec,
        path: Path,
        *,
        expected_receipt_sha256: str,
    ) -> SglangKtKernelRuntimeValidationReceiptObservation:
        assert observed_process_spec == process_spec
        assert path == software.kernel_runtime_receipt.path
        assert expected_receipt_sha256 == software.kernel_runtime_receipt.sha256
        preflight_order.append("kernel_receipt")
        return cast(
            SglangKtKernelRuntimeValidationReceiptObservation,
            cast(object, SimpleNamespace()),
        )

    def return_verified_snapshot(
        snapshot_path: Path,
        contract_path: Path,
        **expected: object,
    ) -> SglangKtVerifiedModelSnapshot:
        preflight_order.append("model_verification")
        verification_calls.append((snapshot_path, contract_path, expected))
        return snapshot

    monkeypatch.setattr(
        validator,
        "require_immutable_validator_source_deployment",
        accept_source_deployment,
    )
    monkeypatch.setattr(
        validator,
        "require_immutable_model_snapshot",
        accept_model_snapshot,
    )
    monkeypatch.setattr(
        validator,
        "load_bound_kernel_runtime_receipt",
        accept_kernel_receipt,
    )
    monkeypatch.setattr(
        validator,
        "verify_sglang_kt_model_snapshot",
        return_verified_snapshot,
    )

    execution = validator.perform_execution_preflight(
        parsed,
        validator_repository_root=validator_repository_root,
    )

    assert execution.model_snapshot == snapshot
    assert preflight_order == [
        "kernel_receipt",
        "model_snapshot",
        "model_verification",
    ]
    assert immutable_model_paths == [Path(process_spec.model_path)]
    assert verification_calls == [
        (
            Path(process_spec.model_path),
            Path(software.model_contract.path),
            {
                "expected_contract_sha256": software.model_contract.contract_sha256,
                "expected_model_id": process_spec.model_id,
                "expected_revision": process_spec.expected_model_revision,
                "expected_ktransformers_method": process_spec.ktransformers_method,
            },
        )
    ]


def test_execution_preflight_rejects_kernel_before_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, *_rest, validator_repository_root = write_inputs(tmp_path)
    parsed = validator.parse_arguments(arguments)
    model_snapshot_inspected = False

    def accept_source_deployment(*_arguments: object, **_keywords: object) -> None:
        return None

    def reject_kernel_receipt(*_arguments: object, **_keywords: object) -> None:
        raise validator.Glm47LiveValidationError("rejected kernel receipt")

    def inspect_model_snapshot(*_arguments: object, **_keywords: object) -> None:
        nonlocal model_snapshot_inspected
        model_snapshot_inspected = True

    monkeypatch.setattr(
        validator,
        "require_immutable_validator_source_deployment",
        accept_source_deployment,
    )
    monkeypatch.setattr(
        validator,
        "load_bound_kernel_runtime_receipt",
        reject_kernel_receipt,
    )
    monkeypatch.setattr(
        validator,
        "require_immutable_model_snapshot",
        inspect_model_snapshot,
    )
    monkeypatch.setattr(
        validator,
        "verify_sglang_kt_model_snapshot",
        inspect_model_snapshot,
    )

    with pytest.raises(
        validator.Glm47LiveValidationError,
        match="rejected kernel receipt",
    ):
        validator.perform_execution_preflight(
            parsed,
            validator_repository_root=validator_repository_root,
        )

    assert model_snapshot_inspected is False


@pytest.mark.parametrize("resident_gpu_experts", (5, 62, 63))
def test_software_preflight_rejects_unsafe_initial_resident_expert_count(
    tmp_path: Path,
    resident_gpu_experts: int,
) -> None:
    arguments, *_rest = write_inputs(
        tmp_path,
        resident_gpu_experts=resident_gpu_experts,
    )

    with pytest.raises(
        validator.Glm47ModelValidationError,
        match="between 1 and 4",
    ):
        validator.perform_software_preflight(validator.parse_arguments(arguments))


def test_process_spec_loader_rejects_wrong_digest_and_ambiguous_json(
    tmp_path: Path,
) -> None:
    arguments, process_spec, process_path, *_rest = write_inputs(tmp_path)
    parsed = validator.parse_arguments(arguments)

    with pytest.raises(validator.Glm47ModelValidationError, match="canonical"):
        validator.load_bound_process_spec(
            process_path,
            expected_process_spec_sha256="f" * 64,
        )

    process_path.write_bytes(
        b'{"pipeline_rank":0,"pipeline_rank":1,"plan":{},'
        b'"executable":"/bin/python","model_contract_sha256":null}\n'
    )
    with pytest.raises(validator.Glm47ModelValidationError, match="process-spec"):
        validator.load_bound_process_spec(
            process_path,
            expected_process_spec_sha256=(
                calculate_sglang_kt_process_launch_spec_sha256(process_spec)
            ),
        )
    assert parsed.process_spec == process_path


def test_software_preflight_rejects_symlinked_process_spec(tmp_path: Path) -> None:
    arguments, _spec, process_path, *_rest = write_inputs(tmp_path)
    symlink = tmp_path / "process-link.json"
    symlink.symlink_to(process_path)
    parsed = validator.parse_arguments(
        replace_argument(arguments, "--process-spec", str(symlink))
    )

    with pytest.raises(validator.Glm47ModelValidationError, match="process-spec"):
        validator.perform_software_preflight(parsed)


@pytest.mark.parametrize(
    ("argument_name", "message"),
    (
        (
            "--expected-model-contract-receipt-sha256",
            "raw receipt",
        ),
        ("--expected-kernel-receipt-sha256", "raw SHA-256"),
    ),
)
def test_software_preflight_rejects_changed_parent_receipt(
    tmp_path: Path,
    argument_name: str,
    message: str,
) -> None:
    arguments, *_rest = write_inputs(tmp_path)
    parsed = validator.parse_arguments(
        replace_argument(arguments, argument_name, "f" * 64)
    )

    with pytest.raises(validator.Glm47ModelValidationError, match=message):
        validator.perform_software_preflight(parsed)


def test_software_preflight_reserves_a_create_new_output(tmp_path: Path) -> None:
    arguments, *_rest = write_inputs(tmp_path)
    output_path = tmp_path / "model-runtime-receipt.json"
    output_path.write_text("existing evidence\n")

    with pytest.raises(validator.Glm47ModelValidationError, match="replace"):
        validator.perform_software_preflight(validator.parse_arguments(arguments))

    assert output_path.read_text() == "existing evidence\n"


@pytest.mark.parametrize(
    ("resident_gpu_experts", "selected", "gpu", "cpu"),
    (
        (0, (0, 1, 2, 3), (), (0, 1, 2, 3)),
        (1, (0, 1, 2, 3), (0,), (1, 2, 3)),
        (4, (0, 1, 4, 5), (0, 1), (4, 5)),
        (62, (0, 1, 62, 63), (0, 1), (62, 63)),
        (63, (0, 1, 2, 63), (0, 1, 2), (63,)),
    ),
)
def test_resident_expert_route_boundaries(
    resident_gpu_experts: int,
    selected: tuple[int, ...],
    gpu: tuple[int, ...],
    cpu: tuple[int, ...],
) -> None:
    route = validator.select_resident_expert_route(resident_gpu_experts)

    assert route.resident_gpu_expert_ids == tuple(range(resident_gpu_experts))
    assert route.selected_expert_ids == selected
    assert route.gpu_expert_ids == gpu
    assert route.cpu_expert_ids == cpu


@pytest.mark.parametrize("resident_gpu_experts", (-1, 64))
def test_resident_expert_route_rejects_unadmitted_counts(
    resident_gpu_experts: int,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 63"):
        validator.select_resident_expert_route(resident_gpu_experts)
    with pytest.raises(TypeError, match="integer"):
        validator.select_resident_expert_route(True)


def test_resource_headroom_accepts_exact_boundary_and_rejects_one_byte_short() -> None:
    exact = validator.require_resource_headroom(
        "host memory",
        available_bytes=100,
        required_bytes=60,
        minimum_headroom_bytes=40,
    )

    assert exact.remaining_bytes == 40
    assert exact.sufficient is True
    with pytest.raises(validator.Glm47ModelValidationError, match="39 bytes"):
        validator.require_resource_headroom(
            "host memory",
            available_bytes=99,
            required_bytes=60,
            minimum_headroom_bytes=40,
        )


@pytest.mark.parametrize(
    "values",
    (
        {"available_bytes": -1, "required_bytes": 0, "minimum_headroom_bytes": 1},
        {"available_bytes": 1, "required_bytes": -1, "minimum_headroom_bytes": 1},
        {"available_bytes": 1, "required_bytes": 0, "minimum_headroom_bytes": 0},
    ),
)
def test_resource_headroom_rejects_invalid_counts(values: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        validator.calculate_resource_headroom(**values)
    with pytest.raises(TypeError):
        validator.calculate_resource_headroom(
            available_bytes=True,
            required_bytes=0,
            minimum_headroom_bytes=1,
        )


def test_available_numa_memory_uses_only_selected_nodes(tmp_path: Path) -> None:
    node_root = tmp_path / "nodes"
    for node, free_kibibytes in ((0, 123), (1, 456), (2, 789)):
        node_path = node_root / f"node{node}"
        node_path.mkdir(parents=True)
        (node_path / "meminfo").write_text(
            f"Node {node} MemTotal: 999 kB\nNode {node} MemFree: {free_kibibytes} kB\n"
        )

    assert validator.available_numa_memory_bytes((0, 2), node_root) == (
        (123 + 789) * 1024
    )

    with pytest.raises(ValueError):
        validator.available_numa_memory_bytes((2, 0), node_root)
    (node_root / "node0" / "meminfo").write_text("Node 0 MemFree: bad kB\n")
    with pytest.raises(validator.Glm47ModelValidationError, match="invalid"):
        validator.available_numa_memory_bytes((0,), node_root)


def test_validator_source_bundle_uses_every_bound_file_identity(tmp_path: Path) -> None:
    repository_root = tmp_path / "validator-repository"
    paths = validator_bundle_paths(repository_root)
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source {index}\n".encode())

    observed = validator.calculate_validator_source_bundle(repository_root)

    assert tuple(source.path for source in observed.sources) == tuple(map(str, paths))
    assert tuple(source.size_bytes for source in observed.sources) == tuple(
        path.stat().st_size for path in paths
    )
    first_sha256 = observed.sha256
    paths[0].write_bytes(b"changed source\n")
    assert validator.calculate_validator_source_bundle(repository_root).sha256 != (
        first_sha256
    )


def _set_tree_read_only(root: Path, *, read_only: bool) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else (0o444 if read_only else 0o644))
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o555 if read_only else 0o755)
    root.chmod(0o555 if read_only else 0o755)


def test_immutable_validator_deployment_is_exact_and_nonwritable(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "validator-repository"
    for index, path in enumerate(validator_bundle_paths(repository_root)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source {index}\n".encode())
    bundle = validator.calculate_validator_source_bundle(repository_root)

    with pytest.raises(validator.Glm47ModelValidationError, match="read-only"):
        validator.require_immutable_validator_source_deployment(
            repository_root,
            bundle,
            required_uid=os.getuid(),
        )

    _set_tree_read_only(repository_root, read_only=True)
    try:
        validator.require_immutable_validator_source_deployment(
            repository_root,
            bundle,
            required_uid=os.getuid(),
        )
    finally:
        _set_tree_read_only(repository_root, read_only=False)

    extra = repository_root / "unbound.py"
    extra.write_text("raise RuntimeError\n")
    _set_tree_read_only(repository_root, read_only=True)
    try:
        with pytest.raises(validator.Glm47ModelValidationError, match="exactly"):
            validator.require_immutable_validator_source_deployment(
                repository_root,
                bundle,
                required_uid=os.getuid(),
            )
    finally:
        _set_tree_read_only(repository_root, read_only=False)


def test_direct_entrypoint_does_not_mutate_immutable_source_closure(
    tmp_path: Path,
) -> None:
    source_root = Path(validator.__file__).resolve().parents[1]
    repository_root = tmp_path / "validator-repository"
    for relative_path in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS:
        source = source_root / relative_path
        destination = repository_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    expected_files = frozenset(validator_bundle_paths(repository_root))
    environment = os.environ.copy()
    environment.pop("PYTHONDONTWRITEBYTECODE", None)

    _set_tree_read_only(repository_root, read_only=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(repository_root / "scripts/validate_sglang_kt_glm47_model.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
        observed_files = frozenset(
            path for path in repository_root.rglob("*") if path.is_file()
        )
    finally:
        _set_tree_read_only(repository_root, read_only=False)

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert observed_files == expected_files
    assert not tuple(repository_root.rglob("__pycache__"))


def test_immutable_model_snapshot_rejects_writable_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n")

    with pytest.raises(validator.Glm47ModelValidationError, match="writable"):
        validator.require_immutable_model_snapshot(
            snapshot,
            required_uid=os.getuid(),
        )

    _set_tree_read_only(snapshot, read_only=True)
    try:
        validator.require_immutable_model_snapshot(
            snapshot,
            required_uid=os.getuid(),
        )
    finally:
        _set_tree_read_only(snapshot, read_only=False)


def test_validator_source_bundle_covers_local_import_closure() -> None:
    repository_root = Path(validator.__file__).resolve().parents[1]
    program = f"""
from pathlib import Path
import sys

repository_root = Path({str(repository_root)!r})
sys.path.insert(0, str(repository_root / "src"))
sys.path.insert(0, str(repository_root))

import scripts.sglang_kt_glm47_backend
import scripts.validate_sglang_kt_glm47_model

paths = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        continue
    try:
        relative = Path(source).resolve().relative_to(repository_root)
    except (OSError, ValueError):
        continue
    if relative.suffix == ".py" and relative.parts[0] in {{"scripts", "src"}}:
        paths.add(relative.as_posix())
print("\\n".join(sorted(paths)))
"""

    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert frozenset(result.stdout.splitlines()) == frozenset(
        MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    )


def test_disposable_child_command_applies_exact_numa_and_overlay_bindings(
    tmp_path: Path,
) -> None:
    arguments, process_spec, *_rest = write_inputs(tmp_path)
    parsed = validator.parse_arguments(arguments)
    validator_repository_root = tmp_path / "validator-repository"
    preflight = validator.perform_software_preflight(
        parsed,
        validator_repository_root=validator_repository_root,
    )
    script = tmp_path / "validator.py"

    command = validator.build_disposable_child_command(
        parsed,
        preflight,
        validator_script=script,
        numactl_executable=Path("/usr/bin/numactl"),
    )

    assert command[:6] == (
        "/usr/bin/numactl",
        "--physcpubind",
        "0,1,2,3",
        "--membind",
        "0",
        process_spec.executable,
    )
    assert command[6] == str(script)
    assert command[7:] == tuple(arguments)
    assert "--internal-evidence-fd" not in command


def test_internal_child_rejects_binding_before_backend_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, *_rest, validator_repository_root = write_inputs(tmp_path)
    parsed = validator.parse_arguments([*arguments, "--internal-evidence-fd", "9"])
    preflight = validator.perform_software_preflight(
        parsed,
        validator_repository_root=validator_repository_root,
    )
    backend_called = False

    def reject_binding(_process_spec: object) -> None:
        raise validator.Glm47LiveValidationError("wrong binding")

    def fail_if_backend_runs(_preflight: object) -> object:
        nonlocal backend_called
        backend_called = True
        return object()

    def return_preflight(_arguments: object) -> validator.Glm47ModelExecutionPreflight:
        return cast(validator.Glm47ModelExecutionPreflight, preflight)

    def return_kernel_receipt(
        _process_spec: SglangKtProcessLaunchSpec,
        _path: Path,
        *,
        expected_receipt_sha256: str,
    ) -> SglangKtKernelRuntimeValidationReceiptObservation:
        assert expected_receipt_sha256
        return cast(
            SglangKtKernelRuntimeValidationReceiptObservation,
            cast(object, SimpleNamespace()),
        )

    monkeypatch.setattr(
        validator,
        "perform_execution_preflight",
        return_preflight,
    )
    monkeypatch.setattr(
        validator,
        "load_bound_kernel_runtime_receipt",
        return_kernel_receipt,
    )
    monkeypatch.setattr(validator, "require_current_process_binding", reject_binding)
    monkeypatch.setattr(validator, "_run_bound_backend", fail_if_backend_runs)

    with pytest.raises(validator.Glm47LiveValidationError, match="wrong binding"):
        validator.perform_internal_live_validation(parsed)
    assert backend_called is False
    assert not parsed.output.exists()


def test_parent_rejects_evidence_transport_before_model_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, *_rest = write_inputs(tmp_path)
    parsed = validator.parse_arguments(arguments)
    model_preflight_started = False

    def reject_evidence_transport() -> None:
        raise validator.Glm47LiveValidationError("evidence transport unavailable")

    def start_model_preflight(
        _arguments: object,
    ) -> validator.Glm47ModelExecutionPreflight:
        nonlocal model_preflight_started
        model_preflight_started = True
        raise AssertionError("model preflight must not start")

    monkeypatch.setattr(
        validator,
        "require_disposable_child_evidence_transport",
        reject_evidence_transport,
    )
    monkeypatch.setattr(
        validator,
        "perform_execution_preflight",
        start_model_preflight,
    )

    with pytest.raises(
        validator.Glm47LiveValidationError,
        match="evidence transport unavailable",
    ):
        validator.run_parent_live_validation(parsed)

    assert model_preflight_started is False


def test_publisher_creates_one_canonical_receipt_without_replacement(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "receipt.json"
    payload = {"z": [3, 2, 1], "a": {"ready": True}}

    receipt_sha256 = validator.publish_new_canonical_receipt(output_path, payload)
    first_contents = output_path.read_bytes()

    assert first_contents == canonical_sglang_kt_json(payload)
    assert receipt_sha256 == hashlib.sha256(first_contents).hexdigest()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
    assert output_path.stat().st_nlink == 1
    assert not tuple(tmp_path.glob(f".{output_path.name}.*.tmp"))
    with pytest.raises(validator.Glm47ModelValidationError, match="replace"):
        validator.publish_new_canonical_receipt(output_path, {"different": True})
    assert output_path.read_bytes() == first_contents


def test_publisher_rejects_symlink_output_and_nonfinite_json(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("unchanged\n")
    output_path = tmp_path / "receipt.json"
    output_path.symlink_to(target)

    with pytest.raises(validator.Glm47ModelValidationError, match="replace"):
        validator.publish_new_canonical_receipt(output_path, {"safe": True})
    with pytest.raises(validator.Glm47ModelValidationError, match="canonicalize"):
        validator.publish_new_canonical_receipt(
            tmp_path / "nonfinite.json",
            {"value": float("nan")},
        )
    assert target.read_text() == "unchanged\n"


def test_publisher_removes_final_link_if_output_parent_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    output = output_parent / "receipt.json"
    original_link = validator.os.link

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

    monkeypatch.setattr(validator.os, "link", replace_parent_then_link)

    with pytest.raises(validator.Glm47ModelValidationError, match="replaced"):
        validator.publish_new_canonical_receipt(output, {"safe": True})

    assert not output.exists()
    assert not (displaced_parent / output.name).exists()
    assert not (displaced_parent / f".{output.name}.{os.getpid()}.tmp").exists()


def test_main_reports_published_live_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, *_rest = write_inputs(tmp_path)

    def return_live_result(_arguments: object) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "passed",
            "profiler": "none",
            "output": str(tmp_path / "model-runtime-receipt.json"),
            "receipt_sha256": "a" * 64,
            "validator_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        validator,
        "run_parent_live_validation",
        return_live_result,
    )

    assert validator.main(arguments) == 0
    parsed_output = parse_sglang_kt_strict_json(capsys.readouterr().out.encode())
    assert isinstance(parsed_output, dict)
    output = cast(dict[str, object], parsed_output)

    assert output["status"] == "passed"
    assert output["profiler"] == "none"
    assert output["receipt_sha256"] == "a" * 64
