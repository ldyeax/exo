from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import scripts.benchmark_lease as benchmark_lease_module
import scripts.tp1_oracle_capture as oracle


def make_config(tmp_path: Path) -> oracle.OracleConfig:
    run_id = "tp1-test"
    model_path = (
        f"/models/{oracle.MODEL_ID.replace('/', '--')}--{oracle.MODEL_REVISION}"
    )
    manifest = {
        ".exo-huggingface-revision.json": "1" * 64,
        "config.json": "2" * 64,
        "model.safetensors": "3" * 64,
        "model.safetensors.index.json": "4" * 64,
    }
    gpu = oracle.GpuIdentity(
        device_uuid="GPU-11111111-2222-3333-4444-555555555555",
        pci_bus_id="00000000:65:00.0",
        model_name="NVIDIA GeForce RTX 3090",
    )
    cpu = oracle.CpuBinding(cpu_set=(0, 1), numa_nodes=(0,))
    ports = oracle.ServicePorts(api=53101, zenoh=53102, discovery=53103, ring=53104)
    environment = {
        "CUDA_VISIBLE_DEVICES": gpu.device_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "EXO_MODELS_READ_ONLY_DIRS": str(Path(model_path).parent),
        "EXO_OFFLINE": "true",
        "EXO_MAX_CONCURRENT_REQUESTS": "1",
        "ENABLE_DISAGGREGATION": "false",
        "EXO_MLX_VISION_LOADING": "disabled",
        "PYTHONHASHSEED": str(oracle.ORACLE_SEED),
        "EXO_HOME": f"/tmp/exo-{run_id}",
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/test",
        "LD_LIBRARY_PATH": "/usr/lib:/lib",
    }
    launch_argv = (
        "/usr/bin/numactl",
        "--physcpubind=0,1",
        "--membind=0",
        "/venv/bin/python",
        "-m",
        "exo",
        "--namespace",
        f"exo-{run_id}",
        "--api-port",
        str(ports.api),
        "--zenoh-port",
        str(ports.zenoh),
        "--discovery-port",
        str(ports.discovery),
        "--offline",
        "--no-downloads",
        "--force-master",
        "--no-batch",
    )
    return oracle.OracleConfig(
        schema_version=1,
        run_id=run_id,
        namespace=f"exo-{run_id}",
        result_directory=str(tmp_path / "results" / run_id),
        host_name="dwagon",
        source_directory="/source/exo",
        python_executable="/venv/bin/python",
        source=oracle.SourceIdentity(commit="a" * 40, dirty_file_hashes={}),
        model=oracle.ModelSnapshot(
            model_id=oracle.MODEL_ID,
            revision=oracle.MODEL_REVISION,
            local_path=model_path,
            expected_weight_bytes=oracle.MODEL_WEIGHT_BYTES,
            expected_sha256_manifest=manifest,
        ),
        gpu=gpu,
        hca_ports=(
            oracle.HcaPort(
                device="mlx4_0",
                port=1,
                gid="fe80::10:e000:166:3a19",
            ),
            oracle.HcaPort(
                device="mlx4_0",
                port=2,
                gid="fe80::10:e000:166:3a1a",
            ),
        ),
        cpu=cpu,
        ports=ports,
        environment=environment,
        launch_argv=launch_argv,
        request=oracle.DeterministicRequest(
            prompt=oracle.ORACLE_PROMPT,
            repetitions=3,
            max_tokens=oracle.ORACLE_MAX_TOKENS,
            seed=oracle.ORACLE_SEED,
            temperature=0.0,
            stream=False,
            use_prefix_cache=False,
            logprobs=False,
        ),
        timeouts=oracle.Timeouts(
            preflight_seconds=1.0,
            process_start_seconds=1.0,
            api_start_seconds=0.1,
            cluster_seconds=0.1,
            runner_ready_seconds=0.1,
            request_seconds=0.1,
            cleanup_seconds=0.05,
            poll_seconds=0.01,
        ),
        runtime=oracle.RuntimeRequirements(
            cuda_major=13,
            minimum_nvidia_driver_version="570.0",
            exo_rs_sha256="5" * 64,
        ),
    )


def make_preflight(config: oracle.OracleConfig) -> oracle.PreflightReport:
    return oracle.PreflightReport(
        schema_version=1,
        run_id=config.run_id,
        host_name=config.host_name,
        passed=True,
        conflicts=(),
        source=config.source,
        full_gpu_inventory=(config.gpu,),
        selected_gpu=config.gpu,
        gpu_compute_processes=(),
        busy_tcp_ports=(),
        busy_udp_ports=(),
        online_cpu_ids=(0, 1, 2, 3),
        numa_node_ids=(0, 1),
        storage_conflicts=(),
        process_conflicts=(),
        gpu_telemetry_csv="uuid,pci.bus_id,name\n",
        model=oracle.ModelVerification(
            receipt={
                "repo_id": config.model.model_id,
                "revision": config.model.revision,
            },
            indexed_weight_bytes=config.model.expected_weight_bytes,
            physical_weight_bytes=config.model.expected_weight_bytes,
            weight_files=1,
            sha256_manifest=config.model.expected_sha256_manifest,
        ),
        runtime=oracle.RuntimeFacts(
            python_version="3.13.5",
            exo_version="0.0.test",
            mlx_version="0.28.0",
            mlx_cuda_13_version="0.28.0",
            exo_import_origin="/source/exo/src/exo/__init__.py",
            exo_rs_origin="/venv/lib/python3.13/site-packages/exo_rs.so",
            exo_rs_sha256=config.runtime.exo_rs_sha256,
            nvidia_driver_version="580.95.05",
            cuda_driver_major=13,
            nvidia_smi_banner="NVIDIA-SMI 580.95.05",
        ),
    )


def make_cluster_resources(
    config: oracle.OracleConfig, *, include_other_gpu: bool = False
) -> oracle.JsonObject:
    resources: list[oracle.JsonValue] = [
        {
            "NvidiaGpuComputeResource": {
                "deviceUuid": config.gpu.device_uuid,
                "pciBusId": config.gpu.pci_bus_id,
                "modelName": config.gpu.model_name,
            }
        }
    ]
    if include_other_gpu:
        resources.append(
            {
                "NvidiaGpuComputeResource": {
                    "deviceUuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "pciBusId": "00000000:66:00.0",
                    "modelName": "NVIDIA GeForce RTX 3090",
                }
            }
        )
    return {"node-1": resources}


def make_placement(
    config: oracle.OracleConfig, *, explicit_binding: bool = True
) -> oracle.JsonObject:
    resource_to_runner: oracle.JsonObject = {}
    resource_to_node: oracle.JsonObject = {}
    if explicit_binding:
        resource_to_runner[config.gpu.resource_id] = "runner-1"
        resource_to_node[config.gpu.resource_id] = "node-1"
    return {
        "MlxRingInstance": {
            "instanceId": "instance-1",
            "shardAssignments": {
                "modelId": config.model.model_id,
                "runnerToShard": {
                    "runner-1": {
                        "PipelineShardMetadata": {
                            "modelCard": {
                                "modelId": config.model.model_id,
                                "revision": config.model.revision,
                            },
                            "deviceRank": 0,
                            "worldSize": 1,
                            "startLayer": 0,
                            "endLayer": 30,
                            "nLayers": 30,
                        }
                    }
                },
                "nodeToRunner": {"node-1": "runner-1"},
                "computeResourceToRunner": resource_to_runner,
                "computeResourceToNode": resource_to_node,
            },
            "hostsByNode": {"node-1": [{"ip": "0.0.0.0", "port": 49000}]},
            "ephemeralPort": 49000,
        }
    }


@pytest.mark.parametrize(
    ("banner", "major"),
    [
        ("Driver Version: 595.71.05 CUDA Version: 13.2", 13),
        ("KMD Version: 610.43.03 CUDA UMD Version: 13.3", 13),
    ],
)
def test_cuda_driver_major_accepts_legacy_and_umd_banner_labels(
    banner: str, major: int
) -> None:
    assert oracle._cuda_driver_major_from_nvidia_smi(banner) == major


def test_cuda_driver_major_rejects_banner_without_cuda_version() -> None:
    with pytest.raises(oracle.OracleError, match="CUDA driver version"):
        oracle._cuda_driver_major_from_nvidia_smi("KMD Version: 610.43.03")


def make_completion(
    config: oracle.OracleConfig,
    *,
    response_id: str,
    content: str = "Stable synthetic response.",
    prompt_tokens: int = 6,
    completion_tokens: int = 4,
    finish_reason: str = "stop",
) -> oracle.JsonObject:
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": config.model.model_id,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "generation_stats": {
            "prompt_tokens": prompt_tokens,
            "generation_tokens": completion_tokens,
            "prefix_cache_hit": "none",
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class FakeEffects:
    def __init__(
        self,
        config: oracle.OracleConfig,
        *,
        completions: list[oracle.JsonValue] | None = None,
        preflight: oracle.PreflightReport | None = None,
        cleanup_state_clear: bool = True,
        stop_cleanup: oracle.ProcessCleanup | None = None,
        start_error: oracle.StartNodeError | None = None,
        signal_latch: oracle.SignalLatch | None = None,
        signal_at_checkpoint: int | None = None,
    ) -> None:
        self.config = config
        self.preflight = preflight or make_preflight(config)
        self.completions = completions or [
            make_completion(config, response_id=f"completion-{index}")
            for index in range(config.request.repetitions)
        ]
        self.cleanup_state_clear = cleanup_state_clear
        self.stop_cleanup = stop_cleanup
        self.start_error = start_error
        self.signal_latch = signal_latch
        self.signal_at_checkpoint = signal_at_checkpoint
        self.clock = 0.0
        self.checkpoint_count = 0
        self.completion_index = 0
        self.alive = False
        self.started = False
        self.submitted = False
        self.deleted = False
        self.stopped_process: oracle.OwnedProcess | None = None
        self.request_bodies: list[oracle.JsonObject] = []
        self.placement_params: Mapping[str, str] | None = None
        self.submitted_body: oracle.JsonObject | None = None
        self.calls: list[tuple[str, str]] = []
        self.writes: dict[str, oracle.JsonObject] = {}
        self.write_order: list[str] = []

    def checkpoint_lease(self) -> None:
        self.checkpoint_count += 1
        if (
            self.signal_latch is not None
            and self.signal_at_checkpoint == self.checkpoint_count
        ):
            self.signal_latch.handle(signal.SIGTERM, None)

    def run_preflight(self, config: oracle.OracleConfig) -> oracle.PreflightReport:
        del config
        return self.preflight

    def start_node(
        self, config: oracle.OracleConfig, owner_token: str
    ) -> oracle.OwnedProcess:
        self.started = True
        if self.start_error is not None:
            raise self.start_error
        self.alive = True
        return oracle.OwnedProcess(
            host_name=config.host_name,
            pid=12001,
            process_group_id=12001,
            start_time_ticks=998877,
            owner_token=owner_token,
            namespace=config.namespace,
            transport_pid=12001,
            log_path="/tmp/tp1-test-exo.log",
        )

    def process_alive(self, process: oracle.OwnedProcess) -> bool:
        del process
        return self.alive

    def stop_node(
        self, process: oracle.OwnedProcess, timeout_seconds: float
    ) -> oracle.ProcessCleanup:
        del timeout_seconds
        self.stopped_process = process
        self.alive = False
        if self.stop_cleanup is not None:
            return self.stop_cleanup
        return oracle.ProcessCleanup(
            host_name=process.host_name,
            ownership_verified=True,
            terminated=True,
            forced=False,
        )

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: oracle.JsonObject | None = None,
    ) -> oracle.JsonValue:
        self.calls.append((method, path))
        if method == "GET" and path == "/node_id":
            return "node-1"
        if method == "GET" and path == "/state/nodeComputeResources":
            return make_cluster_resources(self.config)
        if method == "GET" and path == "/state/nodeBackends":
            return {"node-1": ["MlxCuda"]}
        if method == "GET" and path == "/instance/placement":
            self.placement_params = None if params is None else dict(params)
            return make_placement(self.config, explicit_binding=False)
        if method == "GET" and path == "/state/instances/instance-1":
            if not self.submitted or self.deleted:
                raise oracle.HttpResponseError(404, "Not Found", "missing")
            if self.submitted_body is None:
                raise AssertionError("submitted placement body is missing")
            return self.submitted_body["instance"]
        if method == "POST" and path == "/instance":
            if body is None:
                raise AssertionError("placement POST must include a body")
            self.submitted = True
            self.submitted_body = copy.deepcopy(body)
            return {"accepted": True}
        if method == "GET" and path == "/state/runners/runner-1":
            return {"RunnerReady": {}}
        if method == "POST" and path == "/bench/chat/completions":
            if body is None:
                raise AssertionError("completion POST must include a body")
            self.request_bodies.append(copy.deepcopy(body))
            response = self.completions[self.completion_index]
            self.completion_index += 1
            return response
        if method == "DELETE" and path == "/instance/instance-1":
            self.deleted = True
            return {"deleted": True}
        if method == "GET" and path == "/state":
            if self.cleanup_state_clear:
                return {
                    "instances": {},
                    "runners": {},
                    "retiringComputeResources": {},
                    "prefillServerPorts": {},
                }
            return {
                "instances": {"instance-1": {}},
                "runners": {"runner-1": {}},
                "retiringComputeResources": {self.config.gpu.resource_id: "runner-1"},
                "prefillServerPorts": {"runner-1": 60000},
            }
        raise AssertionError(f"unexpected fake request {method} {path}")

    def monotonic(self) -> float:
        self.clock += 0.001
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def write_result_json(self, filename: str, value: oracle.JsonObject) -> None:
        self.write_order.append(filename)
        self.writes[filename] = copy.deepcopy(value)


class FragmentValidatingLease(benchmark_lease_module.BenchmarkLease):
    def validate_fragments(
        self,
        runtime_metadata: Mapping[str, object],
        benchmark_result: Mapping[str, object],
    ) -> dict[str, object]:
        self._runtime_metadata_cache = self._validate_runtime_metadata(runtime_metadata)
        return self._validate_benchmark_result(benchmark_result)


class FakeLeaseGuard(oracle.LeaseGuard):
    def __init__(self) -> None:
        pass

    def checkpoint(self) -> None:
        pass


class InspectableSystemEffects(oracle.SystemEffects):
    def gpu_inventory_for_test(self) -> tuple[oracle.GpuIdentity, ...]:
        return self._gpu_inventory()

    def gpu_compute_processes_for_test(self) -> tuple[str, ...]:
        return self._gpu_compute_processes()

    def abandon_handle_for_test(self, process_id: int) -> None:
        handle = self._running.pop(process_id, None)
        if handle is not None:
            handle.log_file.close()


class StubPopen:
    def __init__(self, *, wait_times_out: bool = False) -> None:
        self.pid = 4321
        self.returncode: int | None = None
        self.wait_times_out = wait_times_out
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.wait_times_out and self.returncode is None:
            raise subprocess.TimeoutExpired(
                ("exo",), 0.0 if timeout is None else timeout
            )
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def open_system_effects(
    config: oracle.OracleConfig,
) -> tuple[InspectableSystemEffects, int]:
    result_directory = Path(config.result_directory)
    result_directory.mkdir(parents=True)
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    return (
        InspectableSystemEffects(config, FakeLeaseGuard(), descriptor),
        descriptor,
    )


def model_data(config: oracle.OracleConfig) -> dict[str, object]:
    return cast(dict[str, object], cast(object, config.model_dump(mode="python")))


def make_lease_context(
    config: oracle.OracleConfig, now: datetime
) -> tuple[tuple[str, ...], Path, Path, Path, dict[str, object]]:
    config_path = Path("/tmp/tp1-oracle-config.json")
    lease_path = Path("/tmp/tp1-oracle-lease.json")
    lock_path = Path("/tmp/tp1-oracle.lock")
    command = (
        config.python_executable,
        str(Path(oracle.__file__).resolve()),
        "--config",
        str(config_path),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    )
    metadata = oracle.lease_static_metadata(config, command)
    metadata["generated_at"] = now.isoformat()
    record: dict[str, object] = {
        "lease_id": "lease-1",
        "run_id": config.run_id,
        "wrapper_pid": 4100,
        "child_pid": 4101,
        "command": list(command),
        "exo_namespace": config.namespace,
        "ports": list(config.reserved_ports),
        "heartbeat": now.isoformat(),
        "result_directory": config.result_directory,
        "metadata": metadata,
        "cleanup_grace_seconds": oracle.minimum_cleanup_grace_seconds(config),
        "child_cleanup_confirmation_required": True,
        "fragment_errors": {},
        "manual_clearance_required": False,
    }
    return command, config_path, lease_path, lock_path, record


def validate_record(
    config: oracle.OracleConfig,
    now: datetime,
    context: tuple[tuple[str, ...], Path, Path, Path, dict[str, object]],
    *,
    required_lease_id: str | None = None,
) -> str:
    command, config_path, lease_path, lock_path, record = context
    return oracle.validate_lease_record(
        config,
        record,
        command=command,
        config_path=config_path,
        lease_path=lease_path,
        lock_path=lock_path,
        process_id=4101,
        parent_process_id=4100,
        now=now,
        required_lease_id=required_lease_id,
    )


@dataclass(frozen=True)
class LeasePreparationFixture:
    config: oracle.OracleConfig
    config_path: Path
    metadata_output: Path
    wrapper_python: Path
    child_python: Path
    benchmark_lease_script: Path
    harness_script: Path
    lease_path: Path
    lock_path: Path
    result_root: Path


def make_lease_preparation_fixture(tmp_path: Path) -> LeasePreparationFixture:
    result_root = (tmp_path / "prepared-results").resolve()
    result_root.mkdir()
    source_directory = Path(oracle.__file__).resolve().parents[1]
    raw = cast(
        dict[str, object],
        cast(
            object,
            make_config(tmp_path / "unused-result").model_dump(mode="json"),
        ),
    )
    raw["result_directory"] = str(result_root / cast(str, raw["run_id"]))
    raw["source_directory"] = str(source_directory)
    raw["python_executable"] = sys.executable
    launch_argv = cast(list[str], raw["launch_argv"])
    launch_argv[3] = sys.executable
    config = oracle.OracleConfig.model_validate_json(json.dumps(raw))
    config_path = (tmp_path / "strict-tp1-config.json").resolve()
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    return LeasePreparationFixture(
        config=config,
        config_path=config_path,
        metadata_output=(tmp_path / "tp1-lease-metadata.json").resolve(),
        wrapper_python=Path(sys.executable),
        child_python=Path(sys.executable),
        benchmark_lease_script=source_directory / "scripts" / "benchmark_lease.py",
        harness_script=Path(oracle.__file__).resolve(),
        lease_path=(tmp_path / "coordination" / "benchmark-lease.json").resolve(),
        lock_path=(tmp_path / "coordination" / "benchmark.lock").resolve(),
        result_root=result_root,
    )


def prepare_lease_fixture(
    fixture: LeasePreparationFixture,
    *,
    metadata_output: Path | None = None,
    cleanup_grace_seconds: float | None = None,
    clock: Callable[[], datetime] | None = None,
    source_identity: Callable[[str], oracle.SourceIdentity] | None = None,
) -> oracle.LeasePreparation:
    return oracle.prepare_lease_metadata(
        config_path=fixture.config_path,
        metadata_output=metadata_output or fixture.metadata_output,
        wrapper_python=fixture.wrapper_python,
        child_python=fixture.child_python,
        benchmark_lease_script=fixture.benchmark_lease_script,
        harness_script=fixture.harness_script,
        owner="/root/tp1-oracle-test",
        purpose="strict deterministic TP1 oracle",
        expected_duration_seconds=600.0,
        heartbeat_seconds=10.0,
        cleanup_grace_seconds=(
            cleanup_grace_seconds
            if cleanup_grace_seconds is not None
            else oracle.minimum_cleanup_grace_seconds(fixture.config)
        ),
        lease_path=fixture.lease_path,
        lock_path=fixture.lock_path,
        result_root=fixture.result_root,
        now=clock or (lambda: datetime.now(timezone.utc)),
        source_identity=(
            source_identity or (lambda _source_directory: fixture.config.source)
        ),
    )


def test_prepare_lease_cli_invokes_exact_wrapper_child_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)

    def source_identity_override(_source_directory: str) -> oracle.SourceIdentity:
        return fixture.config.source

    monkeypatch.setattr(oracle, "_read_source_identity", source_identity_override)
    preparation_arguments = [
        "prepare-lease",
        "--config",
        str(fixture.config_path),
        "--metadata-output",
        str(fixture.metadata_output),
        "--wrapper-python",
        str(fixture.wrapper_python),
        "--child-python",
        str(fixture.child_python),
        "--benchmark-lease-script",
        str(fixture.benchmark_lease_script),
        "--harness-script",
        str(fixture.harness_script),
        "--owner=/root/tp1-oracle-test",
        "--purpose=strict deterministic TP1 oracle",
        "--expected-duration-seconds",
        "600",
        "--heartbeat-seconds",
        "10",
        "--cleanup-grace-seconds",
        str(oracle.minimum_cleanup_grace_seconds(fixture.config)),
        "--lease-path",
        str(fixture.lease_path),
        "--lock-path",
        str(fixture.lock_path),
        "--result-root",
        str(fixture.result_root),
    ]

    assert oracle.main(preparation_arguments) == 0
    machine_output = cast(dict[str, object], json.loads(capsys.readouterr().out))
    child_argv = tuple(cast(list[str], machine_output["child_argv"]))
    wrapper_argv = tuple(cast(list[str], machine_output["benchmark_lease_argv"]))
    expected_child_argv = (
        str(fixture.child_python),
        str(fixture.harness_script),
        "--config",
        str(fixture.config_path),
        "--lease-path",
        str(fixture.lease_path),
        "--lock-path",
        str(fixture.lock_path),
        "--result-dir",
        fixture.config.result_directory,
    )
    assert child_argv == expected_child_argv
    written = cast(
        dict[str, object],
        json.loads(fixture.metadata_output.read_text(encoding="utf-8")),
    )
    assert written["command"] == list(expected_child_argv)
    assert written["hca_bindings"] == {
        "dwagon": [
            {"device": "mlx4_0", "port": 1, "gid": "fe80::10:e000:166:3a19"},
            {"device": "mlx4_0", "port": 2, "gid": "fe80::10:e000:166:3a1a"},
        ]
    }
    generated_at = datetime.fromisoformat(cast(str, written["generated_at"]))
    assert (
        benchmark_lease_module.validate_run_metadata(written, now=generated_at)
        == written
    )

    captured_leases: list[benchmark_lease_module.BenchmarkLease] = []

    def invoke_exact_child(
        lease: benchmark_lease_module.BenchmarkLease,
    ) -> int:
        captured_leases.append(lease)
        return 0

    monkeypatch.setattr(
        benchmark_lease_module,
        "run_with_lease",
        invoke_exact_child,
    )
    assert benchmark_lease_module.main(wrapper_argv[2:]) == 0
    assert len(captured_leases) == 1
    assert captured_leases[0].command == expected_child_argv
    assert captured_leases[0].metadata == written
    assert not Path(fixture.config.result_directory).exists()
    assert fixture.metadata_output.stat().st_mode & 0o777 == 0o644


def test_prepare_lease_rejects_stale_or_changing_source_identity(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    stale = oracle.SourceIdentity(commit="b" * 40, dirty_file_hashes={})
    with pytest.raises(oracle.OracleError, match="source identity is stale"):
        prepare_lease_fixture(
            fixture,
            source_identity=lambda _source_directory: stale,
        )

    identities = iter((fixture.config.source, stale))
    with pytest.raises(oracle.OracleError, match="source identity changed"):
        prepare_lease_fixture(
            fixture,
            source_identity=lambda _source_directory: next(identities),
        )
    assert not fixture.metadata_output.exists()


@pytest.mark.parametrize("existing_path", ["metadata", "result"])
def test_prepare_lease_never_reuses_existing_outputs(
    tmp_path: Path, existing_path: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    if existing_path == "metadata":
        fixture.metadata_output.write_text("operator-owned", encoding="utf-8")
        expected_message = "metadata output already exists"
    else:
        Path(fixture.config.result_directory).mkdir()
        expected_message = "result directory already exists"

    with pytest.raises(oracle.OracleError, match=expected_message):
        prepare_lease_fixture(fixture)

    if existing_path == "metadata":
        assert fixture.metadata_output.read_text(encoding="utf-8") == "operator-owned"


def test_prepare_lease_rejects_short_cleanup_and_naive_clock(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    with pytest.raises(oracle.OracleError, match="shorter than the TP1 cleanup bound"):
        prepare_lease_fixture(fixture, cleanup_grace_seconds=1.0)
    with pytest.raises(oracle.OracleError, match="offset-aware"):
        prepare_lease_fixture(
            fixture,
            clock=lambda: datetime(2026, 7, 18, 12, 0),
        )
    assert not fixture.metadata_output.exists()


def test_success_captures_only_consensus_hash_and_cleans_up(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)

    result = oracle.run_oracle_capture(config, effects)

    assert result["status"] == "completed"
    assert result["reportable"] is True
    assert result["cleanup_succeeded"] is True
    assert effects.write_order == [
        oracle.RUNTIME_METADATA_FILENAME,
        oracle.BENCHMARK_RESULT_FILENAME,
    ]
    assert len(effects.request_bodies) == config.request.repetitions
    assert all(
        body == oracle.deterministic_request(config) for body in effects.request_bodies
    )
    result_oracle = cast(dict[str, object], result["oracle"])
    assert (
        result_oracle["content_sha256"]
        == hashlib.sha256(b"Stable synthetic response.").hexdigest()
    )
    assert result["resource_binding_mode"] == "explicit_resource"
    assert result["api_explicit_resource_binding"] is True
    assert effects.placement_params == {
        "model_id": config.model.model_id,
        "sharding": "Pipeline",
        "instance_meta": "MlxRing",
        "min_nodes": "1",
        "use_all_compute_resources": "false",
    }
    assert effects.submitted_body is not None
    submitted = cast(dict[str, object], effects.submitted_body["instance"])
    submitted_instance = cast(dict[str, object], submitted["MlxRingInstance"])
    submitted_assignments = cast(
        dict[str, object], submitted_instance["shardAssignments"]
    )
    assert submitted_assignments["computeResourceToRunner"] == {
        config.gpu.resource_id: "runner-1"
    }
    assert submitted_assignments["computeResourceToNode"] == {
        config.gpu.resource_id: "node-1"
    }
    assert submitted_instance["ephemeralPort"] == config.ports.ring
    submitted_hosts = cast(dict[str, object], submitted_instance["hostsByNode"])
    submitted_host_list = cast(list[object], submitted_hosts["node-1"])
    submitted_host = cast(dict[str, object], submitted_host_list[0])
    assert submitted_host["port"] == config.ports.ring
    assert "Stable synthetic response." not in json.dumps(result, sort_keys=True)
    assert effects.deleted is True
    assert effects.stopped_process is not None


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (("content", "Different response."), ("completion_tokens", 5)),
)
def test_consensus_mismatch_is_nonreportable_but_still_cleans_up(
    tmp_path: Path, changed_field: str, changed_value: str | int
) -> None:
    config = make_config(tmp_path)
    completion_arguments: dict[str, str | int] = {
        "response_id": "completion-1",
        "content": "Stable synthetic response.",
        "completion_tokens": 4,
    }
    completion_arguments[changed_field] = changed_value
    changed = make_completion(
        config,
        response_id=cast(str, completion_arguments["response_id"]),
        content=cast(str, completion_arguments["content"]),
        completion_tokens=cast(int, completion_arguments["completion_tokens"]),
    )
    effects = FakeEffects(
        config,
        completions=[
            make_completion(config, response_id="completion-0"),
            changed,
            make_completion(config, response_id="completion-2"),
        ],
    )

    result = oracle.run_oracle_capture(config, effects)

    assert result["status"] == "capture_failed"
    assert result["reportable"] is False
    assert result["cleanup_succeeded"] is True
    assert "repeated deterministic completions differ" in cast(str, result["error"])
    assert effects.deleted is True
    assert effects.stopped_process is not None


def test_preflight_failure_never_starts_node_and_is_labeled_preflight(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    failed = make_preflight(config).model_copy(
        update={
            "passed": False,
            "conflicts": ("reserved process conflict",),
            "process_conflicts": ("pid 999",),
        }
    )
    effects = FakeEffects(config, preflight=failed)

    result = oracle.run_oracle_capture(config, effects)

    assert result["status"] == "preflight_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert effects.started is False
    assert effects.stopped_process is None


def test_signal_after_start_stops_only_the_owned_process(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    latch = oracle.SignalLatch()
    effects = FakeEffects(
        config,
        signal_latch=latch,
        signal_at_checkpoint=4,
    )

    result = oracle.run_oracle_capture(config, effects, latch)

    assert result["status"] == "capture_failed"
    assert result["interrupted_signal"] == signal.SIGTERM
    assert result["cleanup_succeeded"] is True
    assert effects.stopped_process is not None
    assert effects.stopped_process.pid == 12001


def test_instance_cleanup_timeout_makes_result_nonreportable(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config, cleanup_state_clear=False)

    result = oracle.run_oracle_capture(config, effects)

    assert result["status"] == "cleanup_failed"
    assert result["reportable"] is False
    assert result["instance_cleanup_succeeded"] is False
    assert result["cleanup_succeeded"] is False
    assert effects.stopped_process is not None


def test_start_failure_cleanup_receipt_is_counted_once(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    start_cleanup = oracle.ProcessCleanup(
        host_name=config.host_name,
        ownership_verified=True,
        terminated=True,
        forced=False,
    )
    effects = FakeEffects(
        config,
        start_error=oracle.StartNodeError(
            oracle.OracleError("synthetic startup failure"),
            start_cleanup,
            None,
        ),
    )

    result = oracle.run_oracle_capture(config, effects)

    assert result["status"] == "capture_failed"
    assert result["cleanup_succeeded"] is True
    assert result["owned_processes"] == []
    assert effects.stopped_process is None


def test_inherited_result_descriptor_is_required_and_identity_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    result_directory = Path(config.result_directory)
    result_directory.mkdir(parents=True)
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    monkeypatch.delenv(oracle.RESULT_DIRECTORY_FD_ENVIRONMENT, raising=False)
    try:
        with pytest.raises(oracle.OracleError, match="lease wrapper did not pass"):
            oracle.inherited_result_directory_descriptor(result_directory)
        monkeypatch.setenv(oracle.RESULT_DIRECTORY_FD_ENVIRONMENT, str(descriptor))
        assert (
            oracle.inherited_result_directory_descriptor(result_directory) == descriptor
        )
        moved = tmp_path / "moved-result"
        outside = tmp_path / "outside-result"
        outside.mkdir()
        result_directory.rename(moved)
        result_directory.symlink_to(outside, target_is_directory=True)
        with pytest.raises(oracle.OracleError, match="identity changed"):
            oracle.inherited_result_directory_descriptor(result_directory)
    finally:
        os.close(descriptor)


def test_result_fragments_and_log_stay_on_trusted_descriptor_after_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects, descriptor = open_system_effects(config)
    process = StubPopen()
    receipt: oracle.OwnedProcess | None = None
    result_directory = Path(config.result_directory)
    moved = tmp_path / "moved-owned-result"
    outside = tmp_path / "outside-result"
    outside.mkdir()
    result_directory.rename(moved)
    result_directory.symlink_to(outside, target_is_directory=True)

    def fake_popen(*arguments: object, **keywords: object) -> StubPopen:
        del arguments, keywords
        return process

    def fake_group(
        _receipt: oracle.OwnedProcess,
    ) -> tuple[bool, tuple[int, ...]]:
        return (True, (process.pid,) if process.poll() is None else ())

    def fake_killpg(_process_group_id: int, _signal_number: int) -> None:
        process.returncode = 0

    def fake_process_identity(_process_id: int) -> tuple[int, int]:
        return process.pid, 123456

    monkeypatch.setattr(oracle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(effects, "_read_process_identity", fake_process_identity)
    monkeypatch.setattr(effects, "_group_ownership", fake_group)
    monkeypatch.setattr(oracle.os, "killpg", fake_killpg)

    try:
        effects.write_result_json(
            oracle.RUNTIME_METADATA_FILENAME,
            {"schema_version": 1, "run_id": config.run_id},
        )
        receipt = effects.start_node(config, "owner-token")
        cleanup = effects.stop_node(receipt, 0.1)
        assert cleanup.terminated is True

        attacker_target = outside / "attacker-result.json"
        (moved / oracle.BENCHMARK_RESULT_FILENAME).symlink_to(attacker_target)
        with pytest.raises(oracle.OracleError, match="refusing to replace"):
            effects.write_result_json(
                oracle.BENCHMARK_RESULT_FILENAME,
                {"schema_version": 1, "cleanup_succeeded": False},
            )
        assert not attacker_target.exists()
    finally:
        if receipt is not None:
            effects.abandon_handle_for_test(receipt.pid)
        os.close(descriptor)

    assert (moved / oracle.RUNTIME_METADATA_FILENAME).is_file()
    assert (moved / "exo-tp1.log").is_file()
    assert not (outside / oracle.RUNTIME_METADATA_FILENAME).exists()
    assert not (outside / "exo-tp1.log").exists()


def test_compute_process_query_failure_is_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects, descriptor = open_system_effects(config)

    def failed_run(
        arguments: object, **keywords: object
    ) -> subprocess.CompletedProcess[str]:
        del keywords
        return subprocess.CompletedProcess(
            cast(list[str], arguments),
            17,
            stdout="",
            stderr="injected nvidia-smi query failure",
        )

    monkeypatch.setattr(oracle.subprocess, "run", failed_run)
    try:
        with pytest.raises(oracle.OracleError, match="failed with 17"):
            effects.gpu_compute_processes_for_test()
    finally:
        os.close(descriptor)


def test_gpu_inventory_fails_closed_on_malformed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects, descriptor = open_system_effects(config)

    def malformed_run(
        arguments: object, **keywords: object
    ) -> subprocess.CompletedProcess[str]:
        del keywords
        return subprocess.CompletedProcess(
            cast(list[str], arguments),
            0,
            stdout=(
                "GPU-01234567-89ab-cdef-0123-456789abcdef, 00000000:01:00.0, "
                "NVIDIA GeForce RTX 3090\n"
                "GPU-malformed, 00000000:02:00.0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(oracle.subprocess, "run", malformed_run)
    try:
        with pytest.raises(oracle.OracleError, match="malformed GPU inventory rows: 2"):
            effects.gpu_inventory_for_test()
    finally:
        os.close(descriptor)


def test_stop_node_rejects_empty_group_scan_while_popen_leader_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects, descriptor = open_system_effects(config)
    process = StubPopen(wait_times_out=True)
    group_observations = 0

    def fake_popen(*arguments: object, **keywords: object) -> StubPopen:
        del arguments, keywords
        return process

    def fake_group(
        _receipt: oracle.OwnedProcess,
    ) -> tuple[bool, tuple[int, ...]]:
        nonlocal group_observations
        group_observations += 1
        if group_observations == 1:
            return True, (process.pid,)
        return True, ()

    def fake_process_identity(_process_id: int) -> tuple[int, int]:
        return process.pid, 123456

    def ignore_killpg(_process_group_id: int, _signal_number: int) -> None:
        pass

    monkeypatch.setattr(oracle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(effects, "_read_process_identity", fake_process_identity)
    monkeypatch.setattr(effects, "_group_ownership", fake_group)
    monkeypatch.setattr(oracle.os, "killpg", ignore_killpg)
    receipt: oracle.OwnedProcess | None = None
    try:
        receipt = effects.start_node(config, "owner-token")
        cleanup = effects.stop_node(receipt, 0.001)
        assert cleanup.ownership_verified is True
        assert cleanup.terminated is False
        assert cleanup.error is not None and "Popen leader survived" in cleanup.error
        assert process.poll() is None
        assert process.wait_calls
    finally:
        if receipt is not None:
            effects.abandon_handle_for_test(receipt.pid)
        os.close(descriptor)


def test_strict_config_rejects_source_ports_gpu_and_request_tampering(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    dirty = model_data(config)
    dirty_source = cast(dict[str, object], dirty["source"])
    dirty_source["dirty_file_hashes"] = {"src/exo/main.py": "6" * 64}
    with pytest.raises(ValidationError, match="source must be clean"):
        oracle.OracleConfig.model_validate(dirty)

    duplicate_ports = model_data(config)
    ports = cast(dict[str, object], duplicate_ports["ports"])
    ports["discovery"] = ports["api"]
    with pytest.raises(ValidationError, match="ports must be unique"):
        oracle.OracleConfig.model_validate(duplicate_ports)

    wrong_gpu = model_data(config)
    environment = cast(dict[str, object], wrong_gpu["environment"])
    environment["CUDA_VISIBLE_DEVICES"] = "GPU-wrong"
    with pytest.raises(ValidationError, match="CUDA_VISIBLE_DEVICES"):
        oracle.OracleConfig.model_validate(wrong_gpu)

    missing_hca = model_data(config)
    missing_hca["hca_ports"] = ()
    with pytest.raises(ValidationError, match="hca_ports must be nonempty"):
        oracle.OracleConfig.model_validate(missing_hca)

    wrong_hca = model_data(config)
    wrong_hca_ports = cast(list[dict[str, object]], wrong_hca["hca_ports"])
    wrong_hca_ports[0]["gid"] = "fe80::1"
    with pytest.raises(ValidationError, match="current dwagon mlx4_0 port GIDs"):
        oracle.OracleConfig.model_validate(wrong_hca)

    invalid_gid = model_data(config)
    invalid_hca_ports = cast(list[dict[str, object]], invalid_gid["hca_ports"])
    invalid_hca_ports[0]["gid"] = "192.0.2.1"
    with pytest.raises(ValidationError, match="non-unspecified IPv6"):
        oracle.OracleConfig.model_validate(invalid_gid)

    nondeterministic = model_data(config)
    request = cast(dict[str, object], nondeterministic["request"])
    request["temperature"] = 0.1
    request["repetitions"] = 2
    with pytest.raises(ValidationError, match="greater than or equal to 3"):
        oracle.OracleConfig.model_validate(nondeterministic)

    for field_name, replacement in (
        ("prompt", "Reply with something else."),
        ("max_tokens", 31),
        ("seed", 41),
    ):
        unpinned = model_data(config)
        unpinned_request = cast(dict[str, object], unpinned["request"])
        unpinned_request[field_name] = replacement
        with pytest.raises(ValidationError, match="Input should be"):
            oracle.OracleConfig.model_validate(unpinned)


def test_lease_metadata_binds_full_config_request_and_manifest_digest(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    command = (config.python_executable, str(Path(oracle.__file__).resolve()))
    metadata = oracle.lease_static_metadata(config, command)
    contract = cast(dict[str, object], metadata["tp1_oracle_contract"])
    assert contract == {
        "config_sha256": oracle.canonical_config_sha256(config),
        "request": oracle.deterministic_request_contract(config),
        "request_sha256": oracle.deterministic_request_contract_sha256(config),
        "expected_model_manifest_sha256": oracle.expected_manifest_sha256(config),
    }
    wrapper_metadata = dict(metadata)
    generated_at = datetime.now(timezone.utc)
    wrapper_metadata["generated_at"] = generated_at.isoformat()
    validated = benchmark_lease_module.validate_run_metadata(
        wrapper_metadata, now=generated_at
    )
    assert validated["hca_bindings"] == {
        config.host_name: [
            {"device": "mlx4_0", "port": 1, "gid": "fe80::10:e000:166:3a19"},
            {"device": "mlx4_0", "port": 2, "gid": "fe80::10:e000:166:3a1a"},
        ]
    }

    repeated_data = model_data(config)
    repeated_request = cast(dict[str, object], repeated_data["request"])
    repeated_request["repetitions"] = 4
    repeated_config = oracle.OracleConfig.model_validate(repeated_data)
    repeated_contract = cast(
        dict[str, object],
        oracle.lease_static_metadata(repeated_config, command)["tp1_oracle_contract"],
    )
    assert repeated_contract["config_sha256"] != contract["config_sha256"]
    assert repeated_contract["request"] != contract["request"]
    assert repeated_contract["request_sha256"] != contract["request_sha256"]

    manifest_data = model_data(config)
    model = cast(dict[str, object], manifest_data["model"])
    manifest = cast(dict[str, object], model["expected_sha256_manifest"])
    manifest["config.json"] = "9" * 64
    manifest_config = oracle.OracleConfig.model_validate(manifest_data)
    manifest_contract = cast(
        dict[str, object],
        oracle.lease_static_metadata(manifest_config, command)["tp1_oracle_contract"],
    )
    assert manifest_contract["config_sha256"] != contract["config_sha256"]
    assert (
        manifest_contract["expected_model_manifest_sha256"]
        != contract["expected_model_manifest_sha256"]
    )


def test_exact_cluster_inventory_and_tp1_placement_binding(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    extra_gpu = oracle.GpuIdentity(
        device_uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210",
        pci_bus_id="00000000:02:00.0",
        model_name="NVIDIA GeForce RTX 3090",
    )
    oracle.validate_preflight(
        make_preflight(config).model_copy(
            update={"full_gpu_inventory": (config.gpu, extra_gpu)}
        ),
        config,
    )
    node_id = oracle.validate_cluster_inventory(
        make_cluster_resources(config, include_other_gpu=True),
        {"node-1": ["MlxCuda"]},
        config,
    )
    assert node_id == "node-1"
    raw_placement = make_placement(config)
    explicit = oracle.validate_tp1_placement(raw_placement, config, node_id)
    assert explicit.binding_mode == "explicit_resource"
    assert (
        cast(dict[str, object], raw_placement["MlxRingInstance"])["ephemeralPort"]
        == 49000
    )
    patched_instance = cast(dict[str, object], explicit.instance["MlxRingInstance"])
    assert patched_instance["ephemeralPort"] == config.ports.ring

    unbound = make_placement(config, explicit_binding=False)
    rebound = oracle.validate_tp1_placement(
        unbound,
        config,
        node_id,
        allow_unbound_template=True,
    )
    rebound_instance = cast(dict[str, object], rebound.instance["MlxRingInstance"])
    rebound_assignments = cast(dict[str, object], rebound_instance["shardAssignments"])
    assert rebound_assignments["computeResourceToRunner"] == {
        config.gpu.resource_id: "runner-1"
    }
    assert rebound_assignments["computeResourceToNode"] == {
        config.gpu.resource_id: "node-1"
    }
    with pytest.raises(oracle.OracleError, match="unexpected GPU resource"):
        oracle.validate_tp1_placement(unbound, config, node_id)

    nccl = make_placement(config)
    nccl["MlxNcclInstance"] = nccl.pop("MlxRingInstance")
    with pytest.raises(oracle.OracleError, match="MlxRingInstance"):
        oracle.validate_tp1_placement(nccl, config, node_id)

    wrong_resource = make_placement(config)
    instance = cast(dict[str, object], wrong_resource["MlxRingInstance"])
    assignments = cast(dict[str, object], instance["shardAssignments"])
    assignments["computeResourceToRunner"] = {"nvidia-gpu:GPU-wrong": "runner-1"}
    with pytest.raises(oracle.OracleError, match="unexpected GPU resource"):
        oracle.validate_tp1_placement(wrong_resource, config, node_id)

    missing_resource = make_placement(config)
    missing_instance = cast(dict[str, object], missing_resource["MlxRingInstance"])
    missing_assignments = cast(dict[str, object], missing_instance["shardAssignments"])
    missing_assignments["computeResourceToRunner"] = {}
    missing_assignments["computeResourceToNode"] = {}
    with pytest.raises(oracle.OracleError, match="unexpected GPU resource"):
        oracle.validate_tp1_placement(missing_resource, config, node_id)

    wrong_shape = make_placement(config)
    wrong_shape_instance = cast(dict[str, object], wrong_shape["MlxRingInstance"])
    wrong_shape_instance["hostsByNode"] = {
        "node-1": [
            {"ip": "0.0.0.0", "port": 49000},
            {"ip": "0.0.0.0", "port": 49000},
        ]
    }
    with pytest.raises(oracle.OracleError, match="exactly one ring host"):
        oracle.validate_tp1_placement(wrong_shape, config, node_id)

    wrong_port = make_placement(config)
    wrong_port_instance = cast(dict[str, object], wrong_port["MlxRingInstance"])
    wrong_port_hosts = cast(dict[str, object], wrong_port_instance["hostsByNode"])
    wrong_port_list = cast(list[object], wrong_port_hosts["node-1"])
    cast(dict[str, object], wrong_port_list[0])["port"] = 49001
    with pytest.raises(oracle.OracleError, match="differs from ephemeralPort"):
        oracle.validate_tp1_placement(wrong_port, config, node_id)


def test_active_lease_record_binds_exact_child_and_static_metadata(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    now = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    context = make_lease_context(config, now)

    assert validate_record(config, now, context) == "lease-1"
    assert validate_record(config, now, context, required_lease_id="lease-1") == (
        "lease-1"
    )


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    (
        ("child_pid", 9999, "different child"),
        ("wrapper_pid", 9998, "not this process's parent"),
        ("run_id", "other-run", "run_id differs"),
        ("ports", [1, 2, 3], "ports differs"),
        ("cleanup_grace_seconds", 1.0, "shorter than cleanup bound"),
    ),
)
def test_active_lease_rejects_top_level_tampering(
    tmp_path: Path,
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    config = make_config(tmp_path)
    now = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
    command, config_path, lease_path, lock_path, record = make_lease_context(
        config, now
    )
    record[field_name] = replacement

    with pytest.raises(oracle.OracleError, match=message):
        validate_record(
            config,
            now,
            (command, config_path, lease_path, lock_path, record),
        )


def test_active_lease_rejects_stale_heartbeat_command_and_metadata_tampering(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    now = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)

    stale_context = make_lease_context(config, now)
    stale_context[4]["heartbeat"] = (
        now - oracle.LEASE_HEARTBEAT_MAX_AGE - timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(oracle.OracleError, match="heartbeat is stale"):
        validate_record(config, now, stale_context)

    command_context = make_lease_context(config, now)
    cast(list[str], command_context[4]["command"])[0] = "/other/python"
    with pytest.raises(oracle.OracleError, match="exact child command"):
        validate_record(config, now, command_context)

    metadata_context = make_lease_context(config, now)
    metadata = cast(dict[str, object], metadata_context[4]["metadata"])
    gpu_bindings = cast(dict[str, object], metadata["gpu_bindings"])
    gpu_bindings[config.host_name] = [
        {"uuid": "GPU-wrong", "pci_address": config.gpu.pci_bus_id}
    ]
    with pytest.raises(oracle.OracleError, match="metadata.gpu_bindings"):
        validate_record(config, now, metadata_context)

    contract_context = make_lease_context(config, now)
    contract_metadata = cast(dict[str, object], contract_context[4]["metadata"])
    contract = cast(dict[str, object], contract_metadata["tp1_oracle_contract"])
    contract["config_sha256"] = "0" * 64
    with pytest.raises(oracle.OracleError, match="metadata.tp1_oracle_contract"):
        validate_record(config, now, contract_context)

    fragment_context = make_lease_context(config, now)
    fragment_context[4]["fragment_errors"] = {"runtime-metadata.json": "bad"}
    with pytest.raises(oracle.OracleError, match="invalid child result fragments"):
        validate_record(config, now, fragment_context)

    identity_context = make_lease_context(config, now)
    with pytest.raises(oracle.OracleError, match="identity changed"):
        validate_record(
            config, now, identity_context, required_lease_id="different-lease"
        )


def test_runtime_and_result_fragments_match_benchmark_wrapper_contract(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    result = oracle.run_oracle_capture(config, effects)
    runtime_metadata = effects.writes[oracle.RUNTIME_METADATA_FILENAME]
    now = datetime.now(timezone.utc)
    command, _config_path, lease_path, lock_path, record = make_lease_context(
        config, now
    )
    metadata = cast(dict[str, object], record["metadata"])
    validated_metadata = benchmark_lease_module.validate_run_metadata(metadata, now=now)
    lease = FragmentValidatingLease(
        lock_path=lock_path,
        lease_path=lease_path,
        result_directory=Path(config.result_directory),
        owner="codex:tp1-test",
        purpose="validate TP1 child fragments",
        run_id=config.run_id,
        namespace=config.namespace,
        ports=config.reserved_ports,
        command=command,
        metadata=validated_metadata,
        cleanup_grace_seconds=oracle.minimum_cleanup_grace_seconds(config),
    )
    assert (
        lease.validate_fragments(runtime_metadata, result)["cleanup_succeeded"] is True
    )
