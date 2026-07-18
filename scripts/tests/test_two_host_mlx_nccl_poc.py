from __future__ import annotations

import copy
import fcntl
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest
from pydantic import ValidationError

import scripts.benchmark_lease as benchmark_lease
import scripts.two_host_mlx_nccl_poc as poc
from scripts.two_host_mlx_nccl_poc import (
    ApiConfig,
    BenchmarkConfig,
    GpuIdentity,
    HarnessConfig,
    HarnessError,
    HcaPort,
    HcaPortObservation,
    HostConfig,
    HostPreflightReport,
    HostPreflightRequest,
    HttpResponseError,
    JsonObject,
    JsonValue,
    ModelProbeResult,
    ModelSnapshot,
    OwnedProcess,
    ProcessCleanup,
    RuntimeRequirements,
    SignalLatch,
    SourceIdentity,
    SystemEffects,
    TimeoutConfig,
    cleanup_state_is_clear,
    collect_host_preflight,
    run_harness,
    validate_and_patch_placement,
    wait_for_owned_runners_ready,
)

MODEL_ID = "mlx-community/SmolLM2-135M-Instruct-8bit"
REVISION = "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
WEIGHT_BYTES = 142_955_136
RUNTIME_NODE_IDS = {"dwagon": "a1b2c3", "fwuff": "d4e5f6"}

DWAGON_GPUS = (
    GpuIdentity(
        device_uuid="GPU-dwagon-0",
        pci_bus_id="00000000:01:00.0",
        model_name="NVIDIA GeForce RTX 3090",
    ),
    GpuIdentity(
        device_uuid="GPU-dwagon-1",
        pci_bus_id="00000000:02:00.0",
        model_name="NVIDIA GeForce RTX 3090",
    ),
)
FWUFF_GPUS = (
    GpuIdentity(
        device_uuid="GPU-fwuff-0",
        pci_bus_id="00000000:03:00.0",
        model_name="NVIDIA GeForce RTX 3090",
    ),
)


def _preflight_facts() -> dict[str, object]:
    return {
        "load_average": "0.01 0.02 0.03 1/100 1",
        "memory": "MemTotal: 1 kB",
        "kernel": "Linux test 6.0",
        "gpu_telemetry_csv": "GPU-test, 0 %, 0 MiB, 24 GiB, 30 C",
        "nvidia_smi_banner": "Driver Version: 580.1 CUDA Version: 13.0",
        "nvidia_driver_versions": "580.1",
        "ip_addresses_json": "[]",
        "ip_routes_json": "[]",
        "cpu_frequency_policy": {"policy0.scaling_governor": "performance"},
        "runtime_versions": {
            "python": "3.13",
            "exo": "1.0.0",
            "mlx": "0.32.0",
            "mlx-cuda-12": None,
            "mlx-cuda-13": "0.32.0",
            "nvidia-nccl-cu12": None,
            "nvidia-nccl-cu13": "2.28.9",
        },
        "exo_rs_artifact": {"path": "/opt/exo_rs.so", "sha256": "a" * 64},
        "exo_import_origin": "/opt/exo/src/exo/__init__.py",
    }


def _launch_arguments(
    *,
    namespace: str,
    zenoh_port: int,
    discovery_port: int,
    coordinator: bool,
) -> tuple[str, ...]:
    arguments = [
        "/usr/bin/numactl",
        "--physcpubind=0,1" if coordinator else "--physcpubind=4,5",
        "--membind=0",
        "/opt/exo/.venv/bin/python",
        "-m",
        "exo",
        "--namespace",
        namespace,
        "--zenoh-port",
        str(zenoh_port),
        "--discovery-port",
        str(discovery_port),
        "--offline",
        "--no-downloads",
    ]
    if coordinator:
        arguments.extend(("--force-master", "--api-port", "6100"))
    else:
        arguments.append("--no-api")
    return tuple(arguments)


def _host_environment(
    model_parent: str,
    gpus: tuple[GpuIdentity, ...],
    hca_port: int,
    run_id: str,
    host_name: str,
) -> dict[str, str]:
    return {
        "EXO_MODELS_READ_ONLY_DIRS": model_parent,
        "EXO_OFFLINE": "true",
        "CUDA_VISIBLE_DEVICES": ",".join(gpu.device_uuid for gpu in gpus),
        "NCCL_NET": "IB",
        "NCCL_GIN_ENABLE": "0",
        "NCCL_GIN_TYPE": "0",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_IB_MERGE_NICS": "1",
        "NCCL_IB_HCA": f"=mlx4_0:{hca_port}",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "EXO_MAX_CONCURRENT_REQUESTS": "1",
        "ENABLE_DISAGGREGATION": "false",
        "EXO_MLX_VISION_LOADING": "disabled",
        "PYTHONHASHSEED": "42",
        "EXO_HOME": f"/var/lib/exo/runs/{run_id}/{host_name}",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/root",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    }


def make_config(tmp_path: Path) -> HarnessConfig:
    run_id = "smollm-tp3-test"
    namespace = f"fwuffydwagon-{run_id}"
    source = SourceIdentity(commit="a" * 40, dirty_file_hashes={})
    dwagon_parent = "/var/lib/exo/models"
    fwuff_parent = "/mnt/sanic/exo/models"
    dwagon = HostConfig(
        name="dwagon",
        role="coordinator",
        transport="local",
        ssh_target=None,
        python_executable="/opt/exo/.venv/bin/python",
        source_directory="/opt/exo",
        source=source,
        model_path=f"{dwagon_parent}/{MODEL_ID.replace('/', '--')}--{REVISION}",
        gpus=DWAGON_GPUS,
        hca_ports=(HcaPort(device="mlx4_0", port=1, gid="fe80::1"),),
        cpu_set=(0, 1),
        numa_nodes=(0,),
        launch_argv=_launch_arguments(
            namespace=namespace,
            zenoh_port=6101,
            discovery_port=6102,
            coordinator=True,
        ),
        environment=_host_environment(dwagon_parent, DWAGON_GPUS, 1, run_id, "dwagon"),
        zenoh_port=6101,
        discovery_port=6102,
        launch_order=0,
    )
    fwuff = HostConfig(
        name="fwuff",
        role="worker",
        transport="ssh",
        ssh_target="fwuff",
        python_executable="/opt/exo/.venv/bin/python",
        source_directory="/opt/exo",
        source=source,
        model_path=f"{fwuff_parent}/{MODEL_ID.replace('/', '--')}--{REVISION}",
        gpus=FWUFF_GPUS,
        hca_ports=(HcaPort(device="mlx4_0", port=2, gid="fe80::2"),),
        cpu_set=(4, 5),
        numa_nodes=(0,),
        launch_argv=_launch_arguments(
            namespace=namespace,
            zenoh_port=6103,
            discovery_port=6102,
            coordinator=False,
        ),
        environment=_host_environment(fwuff_parent, FWUFF_GPUS, 2, run_id, "fwuff"),
        zenoh_port=6103,
        discovery_port=6102,
        launch_order=1,
    )
    return HarnessConfig(
        schema_version=1,
        run_id=run_id,
        namespace=namespace,
        result_directory=str(tmp_path),
        api=ApiConfig(host="127.0.0.1", port=6100),
        nccl_coordinator_port=6200,
        reserved_ports=(6100, 6101, 6102, 6103, 6200),
        model=ModelSnapshot(
            model_id=MODEL_ID,
            revision=REVISION,
            expected_weight_bytes=WEIGHT_BYTES,
        ),
        hosts=(dwagon, fwuff),
        benchmark=BenchmarkConfig(
            prompt="Reply with exactly: NCCL proof complete.",
            expected_content_sha256=(
                "7b6643ad1dc722043097271b4d9e337d64fac848b8f219d1e65988db98cb25c9"
            ),
            warmup_count=2,
            sample_count=3,
            max_tokens=32,
            seed=42,
            temperature=0.0,
        ),
        timeouts=TimeoutConfig(
            process_start_seconds=1.0,
            api_start_seconds=1.0,
            cluster_seconds=1.0,
            runner_ready_seconds=1.0,
            request_seconds=1.0,
            cleanup_seconds=1.0,
            poll_seconds=0.01,
        ),
        runtime=RuntimeRequirements(
            cuda_major=13,
            minimum_nvidia_driver_version="580.0",
        ),
    )


@dataclass(frozen=True)
class LeasePreparationFixture:
    config: HarnessConfig
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
    result_root = (tmp_path / "results").resolve()
    result_root.mkdir()
    source_directory = Path(poc.__file__).resolve().parents[1]
    raw = make_config(tmp_path / "unused-result").model_dump(mode="json")
    raw["result_directory"] = str(result_root / raw["run_id"])
    raw["hosts"][0]["source_directory"] = str(source_directory)
    raw["hosts"][0]["python_executable"] = sys.executable
    raw["hosts"][0]["launch_argv"][3] = sys.executable
    config = HarnessConfig.model_validate_json(json.dumps(raw))
    config_path = (tmp_path / "strict-poc-config.json").resolve()
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    return LeasePreparationFixture(
        config=config,
        config_path=config_path,
        metadata_output=(tmp_path / "lease-metadata.json").resolve(),
        wrapper_python=Path(sys.executable),
        child_python=Path(sys.executable),
        benchmark_lease_script=(source_directory / "scripts" / "benchmark_lease.py"),
        harness_script=Path(poc.__file__).resolve(),
        lease_path=(tmp_path / "coordination" / "benchmark-lease.json").resolve(),
        lock_path=(tmp_path / "coordination" / "benchmark.lock").resolve(),
        result_root=result_root,
    )


def prepare_lease_fixture(
    fixture: LeasePreparationFixture,
    *,
    config_path: Path | None = None,
    metadata_output: Path | None = None,
    wrapper_python: Path | None = None,
    child_python: Path | None = None,
    benchmark_lease_script: Path | None = None,
    harness_script: Path | None = None,
    lease_path: Path | None = None,
    lock_path: Path | None = None,
    result_root: Path | None = None,
    owner: str = "/root/poc-metadata-cli",
    purpose: str = "strict TP3 proof",
    expected_duration_seconds: float = 600.0,
    heartbeat_seconds: float = 10.0,
    cleanup_grace_seconds: float = 300.0,
    clock: Callable[[], datetime] | None = None,
    source_identity: Callable[[str], SourceIdentity] | None = None,
) -> poc.LeasePreparation:
    identity = fixture.config.hosts[0].source
    return poc.prepare_lease_metadata(
        config_path=config_path or fixture.config_path,
        metadata_output=metadata_output or fixture.metadata_output,
        wrapper_python=wrapper_python or fixture.wrapper_python,
        child_python=child_python or fixture.child_python,
        benchmark_lease_script=(
            benchmark_lease_script or fixture.benchmark_lease_script
        ),
        harness_script=harness_script or fixture.harness_script,
        owner=owner,
        purpose=purpose,
        expected_duration_seconds=expected_duration_seconds,
        heartbeat_seconds=heartbeat_seconds,
        cleanup_grace_seconds=cleanup_grace_seconds,
        lease_path=lease_path or fixture.lease_path,
        lock_path=lock_path or fixture.lock_path,
        result_root=result_root or fixture.result_root,
        now=clock or (lambda: datetime.now(timezone.utc)),
        source_identity=source_identity or (lambda _source_directory: identity),
    )


def _active_lease_fixture(
    config: HarnessConfig, tmp_path: Path
) -> tuple[Path, Path, Path, tuple[str, ...], JsonObject]:
    config_path = (tmp_path / "poc-config.json").resolve()
    lease_path = (tmp_path / "benchmark-lease.json").resolve()
    lock_path = (tmp_path / "benchmark.lock").resolve()
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    lock_path.touch()
    command = (
        sys.executable,
        str(Path(poc.__file__).resolve()),
        "--config",
        str(config_path),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
    )
    now = datetime.now(timezone.utc).isoformat()
    metadata = poc._lease_static_metadata(config, command)
    metadata["generated_at"] = now
    record: JsonObject = {
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "ports": [port for port in config.reserved_ports],
        "result_directory": config.result_directory,
        "wrapper_pid": os.getppid(),
        "child_pid": os.getpid(),
        "command": [argument for argument in command],
        "cleanup_grace_seconds": 300.0,
        "child_cleanup_confirmation_required": True,
        "heartbeat": now,
        "metadata": metadata,
    }
    return config_path, lease_path, lock_path, command, record


def make_placement(
    config: HarnessConfig,
    runtime_node_ids: Mapping[str, str] = RUNTIME_NODE_IDS,
) -> JsonObject:
    runner_ids = ("runner-0", "runner-1", "runner-2")
    node_ids = (
        runtime_node_ids[config.hosts[0].name],
        runtime_node_ids[config.hosts[1].name],
    )
    resource_ids = tuple(gpu.resource_id for host in config.hosts for gpu in host.gpus)
    runner_to_shard: JsonObject = {}
    for rank, runner_id in enumerate(runner_ids):
        runner_to_shard[runner_id] = {
            "TensorShardMetadata": {
                "modelCard": {"modelId": MODEL_ID, "revision": REVISION},
                "deviceRank": rank,
                "worldSize": 3,
                "startLayer": 0,
                "endLayer": 30,
                "nLayers": 30,
            }
        }
    return {
        "MlxNcclInstance": {
            "instanceId": "owned-instance",
            "shardAssignments": {
                "modelId": MODEL_ID,
                "runnerToShard": runner_to_shard,
                "nodeToRunner": {
                    node_ids[0]: runner_ids[0],
                    node_ids[1]: runner_ids[2],
                },
                "computeResourceToRunner": {
                    resource_ids[0]: runner_ids[0],
                    resource_ids[1]: runner_ids[1],
                    resource_ids[2]: runner_ids[2],
                },
                "computeResourceToNode": {
                    resource_ids[0]: node_ids[0],
                    resource_ids[1]: node_ids[0],
                    resource_ids[2]: node_ids[1],
                },
            },
            "ncclCoordinator": {"ip": "192.168.40.248", "port": 59999},
        }
    }


def _resource_json(gpu: GpuIdentity) -> JsonObject:
    return {
        "NvidiaGpuComputeResource": {
            "resourceId": gpu.resource_id,
            "deviceUuid": gpu.device_uuid,
            "pciBusId": gpu.pci_bus_id,
            "modelName": gpu.model_name,
            "totalMemory": {"inBytes": 24 * 1024**3},
        }
    }


class FakeEffects:
    def __init__(
        self,
        config: HarnessConfig,
        *,
        fail_benchmark: bool = False,
        vary_samples: bool = False,
        signal_latch: SignalLatch | None = None,
    ) -> None:
        self.config = config
        self.placement = make_placement(config)
        self.fail_benchmark = fail_benchmark
        self.vary_samples = vary_samples
        self.signal_latch = signal_latch
        self.clock = 0.0
        self.created = False
        self.started: list[OwnedProcess] = []
        self.stopped: list[OwnedProcess] = []
        self.deleted_paths: list[str] = []
        self.posted_instances: list[JsonObject] = []
        self.completion_requests: list[JsonObject] = []
        self.writes: dict[str, JsonObject] = {}

    def run_preflight(
        self, host: HostConfig, config: HarnessConfig
    ) -> HostPreflightReport:
        return HostPreflightReport(
            schema_version=1,
            run_id=config.run_id,
            host_name=host.name,
            passed=True,
            conflicts=(),
            source_commit=host.source.commit,
            dirty_file_hashes=host.source.dirty_file_hashes,
            gpus=host.gpus,
            checked_tcp_ports=config.reserved_ports,
            checked_udp_ports=config.reserved_ports,
            hca_ports=tuple(
                HcaPortObservation(
                    device=port.device,
                    port=port.port,
                    state="4: ACTIVE",
                    rate="40 Gb/sec",
                    physical_state="5: LinkUp",
                    link_layer="InfiniBand",
                    lid="1",
                    gids=(port.gid,),
                    net_devices=(),
                    ip_addresses=(),
                    counters={"port_rcv_data": "1"},
                )
                for port in host.hca_ports
            ),
            amx_flags=("amx_bf16", "amx_int8", "amx_tile"),
            facts=_preflight_facts(),
        )

    def probe_model(self, host: HostConfig, model: ModelSnapshot) -> ModelProbeResult:
        return ModelProbeResult(
            host_name=host.name,
            path=host.model_path,
            model_id=model.model_id,
            revision=model.revision,
            weight_bytes=model.expected_weight_bytes,
            physical_weight_bytes=model.expected_weight_bytes + 1024,
            weight_files=1,
            sha256_manifest={
                "config.json": "1" * 64,
                "model.safetensors.index.json": "2" * 64,
                "model-00001-of-00001.safetensors": "3" * 64,
            },
            receipt_kind="exo",
            verified=True,
        )

    def start_node(
        self, host: HostConfig, config: HarnessConfig, owner_token: str
    ) -> OwnedProcess:
        process = OwnedProcess(
            host_name=host.name,
            pid=1000 + len(self.started),
            process_group_id=1000 + len(self.started),
            start_time_ticks=5000 + len(self.started),
            owner_token=owner_token,
            namespace=config.namespace,
            transport_pid=2000 + len(self.started),
            log_path=f"/results/{host.name}.log",
        )
        self.started.append(process)
        return process

    def process_alive(self, process: OwnedProcess) -> bool:
        return process in self.started and process not in self.stopped

    def stop_node(
        self, process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        del timeout_seconds
        self.stopped.append(process)
        return ProcessCleanup(process.host_name, True, True, False)

    def read_owned_log(self, process: OwnedProcess) -> str:
        host = next(
            host for host in self.config.hosts if host.name == process.host_name
        )
        launch_receipt = {
            "host_name": host.name,
            "environment": {
                **host.environment,
                "EXO_BENCHMARK_OWNER_TOKEN": process.owner_token,
            },
            "argv": list(host.launch_argv),
        }
        rank_offset = 0
        ranks_by_host: dict[str, list[int]] = {}
        for configured_host in self.config.hosts:
            ranks_by_host[configured_host.name] = list(
                range(rank_offset, rank_offset + len(configured_host.gpus))
            )
            rank_offset += len(configured_host.gpus)
        hca_text = " ".join(f"{port.device}:{port.port}" for port in host.hca_ports)
        lines = ["EXO_POC_LAUNCH " + json.dumps(launch_receipt, sort_keys=True)]
        lines.append(f"test NCCL INFO NET/IB : Using {hca_text}")
        lines.extend(
            f"test NCCL INFO comm 0x1 rank {rank} nranks 3 - Init COMPLETE"
            for rank in ranks_by_host[host.name]
        )
        if host.transport == "ssh":
            lines.extend(("EXO_SUPERVISOR_LOG_COMPLETE", "EXO_SUPERVISOR_CLEAN"))
        return "\n".join(lines) + "\n"

    def request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue:
        if method == "GET" and path == "/node_id":
            return RUNTIME_NODE_IDS["dwagon"]
        if method == "GET" and path == "/state/nodeComputeResources":
            return {
                RUNTIME_NODE_IDS[host.name]: [_resource_json(gpu) for gpu in host.gpus]
                for host in self.config.hosts
            }
        if method == "GET" and path == "/state/nodeBackends":
            return {
                RUNTIME_NODE_IDS[host.name]: ["MlxCuda"] for host in self.config.hosts
            }
        if method == "GET" and path == "/instance/placement":
            assert params == {
                "model_id": MODEL_ID,
                "sharding": "Tensor",
                "instance_meta": "MlxNccl",
                "min_nodes": "2",
                "use_all_compute_resources": "true",
            }
            return copy.deepcopy(self.placement)
        if method == "GET" and path == "/state/instances/owned-instance":
            if not self.created:
                raise HttpResponseError(404, "Not Found", "")
            return copy.deepcopy(self.posted_instances[-1])
        if method == "POST" and path == "/instance":
            assert body is not None
            instance = body["instance"]
            assert isinstance(instance, dict)
            self.posted_instances.append(copy.deepcopy(instance))
            self.created = True
            return {"message": "Command received"}
        if method == "GET" and path.startswith("/state/runners/runner-"):
            if not self.created:
                raise HttpResponseError(404, "Not Found", "")
            return {"RunnerReady": {}}
        if method == "POST" and path == "/bench/chat/completions":
            if self.fail_benchmark:
                raise HarnessError("injected completion failure")
            assert body is not None
            self.completion_requests.append(copy.deepcopy(body))
            if self.signal_latch is not None:
                self.signal_latch.handle(signal.SIGTERM, None)
            content = (
                "different deterministic output"
                if self.vary_samples and len(self.completion_requests) == 4
                else "NCCL proof complete."
            )
            return {
                "id": f"completion-{len(self.completion_requests)}",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "generation_stats": {
                    "prompt_tps": 100.0,
                    "generation_tps": 50.0,
                    "prompt_tokens": 10,
                    "generation_tokens": 5,
                },
            }
        if method == "DELETE" and path == "/instance/owned-instance":
            self.deleted_paths.append(path)
            self.created = False
            return {"message": "Command received"}
        if method == "GET" and path == "/state":
            return {
                "instances": {},
                "runners": {},
                "retiringComputeResources": {},
                "prefillServerPorts": {},
            }
        raise AssertionError(f"unexpected request: {method} {path}")

    def monotonic(self) -> float:
        self.clock += 0.1
        return self.clock

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def write_result_json(self, filename: str, value: JsonObject) -> None:
        self.writes[filename] = copy.deepcopy(value)


def _owned_nccl_logs(config: HarnessConfig, owner_token: str) -> dict[str, str]:
    effects = FakeEffects(config)
    logs: dict[str, str] = {}
    for index, host in enumerate(config.hosts):
        process = OwnedProcess(
            host_name=host.name,
            pid=1000 + index,
            process_group_id=1000 + index,
            start_time_ticks=2000 + index,
            owner_token=owner_token,
            namespace=config.namespace,
            transport_pid=3000 + index,
            log_path=f"/results/{host.name}.log",
        )
        logs[host.name] = effects.read_owned_log(process)
    return logs


def test_config_is_strict_and_requires_the_reserved_nccl_port(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    raw = config.model_dump(mode="json")
    raw["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HarnessConfig.model_validate_json(json.dumps(raw))

    raw = config.model_dump(mode="json")
    raw["reserved_ports"].remove(config.nccl_coordinator_port)
    with pytest.raises(ValidationError, match="must be reserved"):
        HarnessConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("gid", ["10.0.0.1", "::", "fe80::", "not-a-gid"])
def test_config_rejects_invalid_raw_verbs_gid(tmp_path: Path, gid: str) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["hosts"][0]["hca_ports"][0]["gid"] = gid

    with pytest.raises(ValidationError, match="gid"):
        HarnessConfig.model_validate_json(json.dumps(raw))


def test_config_requires_one_shared_multicast_discovery_port(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    raw = config.model_dump(mode="json")
    worker = raw["hosts"][1]
    worker["discovery_port"] = 6104
    discovery_index = worker["launch_argv"].index("--discovery-port") + 1
    worker["launch_argv"][discovery_index] = "6104"
    raw["reserved_ports"].append(6104)

    with pytest.raises(ValidationError, match="shared multicast discovery port"):
        HarnessConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("mutation", ["cpu_binding", "exo_home"])
def test_config_enforces_declared_process_bindings(
    tmp_path: Path, mutation: str
) -> None:
    config = make_config(tmp_path)
    raw = config.model_dump(mode="json")
    coordinator = raw["hosts"][0]
    if mutation == "cpu_binding":
        coordinator["launch_argv"][1] = "--physcpubind=2,3"
        message = "exact numactl"
    else:
        coordinator["environment"]["EXO_HOME"] = "/var/lib/exo/shared"
        message = "EXO_HOME"

    with pytest.raises(ValidationError, match=message):
        HarnessConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("sibling_model", "exact revision-suffixed snapshot"),
        ("extra_model_root", "exactly one canonical read-only model root"),
        ("missing_no_downloads", "disable model downloads"),
        ("other_executable", "configured Python -m exo"),
    ],
)
def test_config_binds_exact_model_and_executable_contract(
    tmp_path: Path, mutation: str, message: str
) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    coordinator = raw["hosts"][0]
    if mutation == "sibling_model":
        coordinator["model_path"] = "/var/lib/exo/models/sibling-snapshot"
    elif mutation == "extra_model_root":
        coordinator["environment"]["EXO_MODELS_READ_ONLY_DIRS"] = (
            "/var/lib/exo/models:/mnt/other"
        )
    elif mutation == "missing_no_downloads":
        coordinator["launch_argv"].remove("--no-downloads")
    else:
        coordinator["launch_argv"][3] = "/other/venv/bin/python"

    with pytest.raises(ValidationError, match=message):
        HarnessConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    "extra_argument",
    [
        "--namespace=unleased",
        "--zenoh-port=65001",
        "--discovery-port=65002",
        "--api-port=65003",
        "--offline=true",
    ],
)
def test_config_rejects_duplicate_or_valued_binding_flags(
    tmp_path: Path, extra_argument: str
) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["hosts"][0]["launch_argv"].append(extra_argument)

    with pytest.raises(ValidationError):
        HarnessConfig.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("name", "value"),
    [("BAD-NAME", "value"), ("GOOD_NAME", "bad\0value")],
)
def test_config_rejects_ambiguous_environment_entries(
    tmp_path: Path, name: str, value: str
) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["hosts"][0]["environment"][name] = value

    with pytest.raises(ValidationError, match="environment variable"):
        HarnessConfig.model_validate_json(json.dumps(raw))


def test_active_lease_binding_requires_held_lock_and_exact_static_proof(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config_path, lease_path, lock_path, command, record = _active_lease_fixture(
        config, tmp_path
    )
    lease_path.write_text(json.dumps(record), encoding="utf-8")

    with lock_path.open("r+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        validated = poc.validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
        )

    assert validated["child_pid"] == os.getpid()
    metadata = validated["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["command"] == list(command)
    oracle = metadata["correctness_oracle"]
    assert isinstance(oracle, dict)
    assert oracle["expected_content_sha256"] == config.benchmark.expected_content_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cleanup_confirmation", "cleanup confirmation"),
        ("cleanup_grace", "cleanup grace"),
        ("gpu_binding", "gpu_bindings"),
        ("child_pid", "different child process"),
    ],
)
def test_active_lease_binding_fails_closed_on_contract_mismatch(
    tmp_path: Path, mutation: str, message: str
) -> None:
    config = make_config(tmp_path)
    config_path, lease_path, lock_path, _command, record = _active_lease_fixture(
        config, tmp_path
    )
    if mutation == "cleanup_confirmation":
        record["child_cleanup_confirmation_required"] = False
    elif mutation == "cleanup_grace":
        record["cleanup_grace_seconds"] = 299.0
    elif mutation == "gpu_binding":
        metadata = record["metadata"]
        assert isinstance(metadata, dict)
        gpu_bindings = metadata["gpu_bindings"]
        assert isinstance(gpu_bindings, dict)
        gpu_bindings["dwagon"] = []
    else:
        record["child_pid"] = os.getpid() + 1
    lease_path.write_text(json.dumps(record), encoding="utf-8")

    with lock_path.open("r+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(HarnessError, match=message):
            poc.validate_active_lease(
                config,
                config_path=config_path,
                lease_path=lease_path,
                lock_path=lock_path,
            )


def test_active_lease_rejects_unheld_coordination_lock(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config_path, lease_path, lock_path, _command, record = _active_lease_fixture(
        config, tmp_path
    )
    lease_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(HarnessError, match="lock is not held"):
        poc.validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
        )


def test_active_lease_grace_covers_two_remote_start_markers(tmp_path: Path) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["timeouts"]["process_start_seconds"] = 1000.0
    config = HarnessConfig.model_validate_json(json.dumps(raw))
    config_path, lease_path, lock_path, _command, record = _active_lease_fixture(
        config, tmp_path
    )
    record["cleanup_grace_seconds"] = 1500.0
    lease_path.write_text(json.dumps(record), encoding="utf-8")

    with lock_path.open("r+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(HarnessError, match="cleanup grace"):
            poc.validate_active_lease(
                config,
                config_path=config_path,
                lease_path=lease_path,
                lock_path=lock_path,
            )


def test_prepare_lease_writes_canonical_metadata_and_exact_wrapper_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    generated_at = datetime(2026, 7, 18, 12, 34, 56, tzinfo=timezone.utc)

    prepared = prepare_lease_fixture(fixture, clock=lambda: generated_at)

    written = json.loads(fixture.metadata_output.read_text(encoding="utf-8"))
    assert written == prepared.metadata
    assert written["generated_at"] == "2026-07-18T12:34:56+00:00"
    assert written["command"] == list(prepared.child_argv)
    assert prepared.child_argv == (
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
    parsed = benchmark_lease.build_parser().parse_args(
        list(prepared.benchmark_lease_argv[2:])
    )
    parsed_command = tuple(parsed.command)
    if parsed_command and parsed_command[0] == "--":
        parsed_command = parsed_command[1:]
    assert parsed_command == prepared.child_argv
    assert parsed.run_id == fixture.config.run_id
    assert parsed.namespace == fixture.config.namespace
    assert benchmark_lease.parse_ports(parsed.port) == fixture.config.reserved_ports
    assert parsed.metadata_json == fixture.metadata_output
    assert parsed.result_root == fixture.result_root
    assert parsed.cleanup_grace_seconds == 300.0
    assert benchmark_lease.validate_run_metadata(written, now=generated_at) == written

    def fixed_utc_now() -> datetime:
        return generated_at

    monkeypatch.setattr(benchmark_lease, "utc_now", fixed_utc_now)
    lease = benchmark_lease.BenchmarkLease(
        lock_path=parsed.lock_path,
        lease_path=parsed.lease_path,
        result_directory=parsed.result_root / parsed.run_id,
        owner=parsed.owner,
        purpose=parsed.purpose,
        run_id=parsed.run_id,
        namespace=parsed.namespace,
        ports=benchmark_lease.parse_ports(parsed.port),
        command=parsed_command,
        metadata=written,
        heartbeat_seconds=parsed.heartbeat_seconds,
        expected_duration_seconds=parsed.expected_duration_seconds,
        cleanup_grace_seconds=parsed.cleanup_grace_seconds,
    )
    assert lease.command == prepared.child_argv
    assert lease.metadata == written
    assert fixture.metadata_output.stat().st_mode & 0o777 == 0o644
    assert not tuple(tmp_path.glob(".lease-metadata.json.*.tmp"))


def test_prepare_lease_canonicalizes_symlinked_lock_parent_once(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    real_lock_parent = tmp_path / "real-lock-parent"
    real_lock_parent.mkdir()
    linked_lock_parent = tmp_path / "linked-lock-parent"
    linked_lock_parent.symlink_to(real_lock_parent, target_is_directory=True)
    requested_lock_path = linked_lock_parent / "benchmark.lock"
    canonical_lock_path = real_lock_parent / "benchmark.lock"

    prepared = prepare_lease_fixture(fixture, lock_path=requested_lock_path)

    assert str(requested_lock_path) not in prepared.child_argv
    assert str(canonical_lock_path) in prepared.child_argv
    parsed = benchmark_lease.build_parser().parse_args(
        list(prepared.benchmark_lease_argv[2:])
    )
    assert parsed.lock_path == canonical_lock_path
    assert poc._canonical_prospective_path(
        poc._DEFAULT_LOCK_PATH, "default lock path"
    ) == poc._DEFAULT_LOCK_PATH.resolve(strict=False)


def test_prepare_lease_emits_dash_prefixed_owner_values_without_argparse_ambiguity(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)

    prepared = prepare_lease_fixture(
        fixture,
        owner="-owner-thread",
        purpose="-strict-proof",
    )

    parsed = benchmark_lease.build_parser().parse_args(
        list(prepared.benchmark_lease_argv[2:])
    )
    assert parsed.owner == "-owner-thread"
    assert parsed.purpose == "-strict-proof"


def test_prepare_lease_cli_emits_machine_readable_exact_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    identity = fixture.config.hosts[0].source

    def source_identity_override(
        _probe: poc.LinuxHostProbe, _source_directory: str
    ) -> SourceIdentity:
        return identity

    monkeypatch.setattr(
        poc.LinuxHostProbe,
        "source_identity",
        source_identity_override,
    )

    result = poc.main(
        [
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
            "--owner",
            "/root/poc-metadata-cli",
            "--purpose",
            "strict TP3 proof",
            "--expected-duration-seconds",
            "600",
            "--heartbeat-seconds",
            "10",
            "--cleanup-grace-seconds",
            "300",
            "--lease-path",
            str(fixture.lease_path),
            "--lock-path",
            str(fixture.lock_path),
            "--result-root",
            str(fixture.result_root),
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == 1
    assert output["metadata_output"] == str(fixture.metadata_output)
    assert (
        output["child_argv"]
        == json.loads(fixture.metadata_output.read_text(encoding="utf-8"))["command"]
    )
    wrapper_argv = output["benchmark_lease_argv"]
    assert isinstance(wrapper_argv, list)
    assert wrapper_argv[-len(output["child_argv"]) :] == output["child_argv"]


@pytest.mark.parametrize(
    "path_name",
    [
        "config",
        "metadata_output",
        "wrapper_python",
        "child_python",
        "benchmark_lease_script",
        "harness_script",
        "lease_path",
        "lock_path",
        "result_root",
    ],
)
def test_prepare_lease_rejects_every_relative_operator_path(
    tmp_path: Path, path_name: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    relative = Path("relative-path")
    config_path = relative if path_name == "config" else fixture.config_path
    metadata_output = (
        relative if path_name == "metadata_output" else fixture.metadata_output
    )
    wrapper_python = (
        relative if path_name == "wrapper_python" else fixture.wrapper_python
    )
    child_python = relative if path_name == "child_python" else fixture.child_python
    benchmark_script = (
        relative
        if path_name == "benchmark_lease_script"
        else fixture.benchmark_lease_script
    )
    harness_script = (
        relative if path_name == "harness_script" else fixture.harness_script
    )
    lease_path = relative if path_name == "lease_path" else fixture.lease_path
    lock_path = relative if path_name == "lock_path" else fixture.lock_path
    result_root = relative if path_name == "result_root" else fixture.result_root

    with pytest.raises(HarnessError, match="absolute path"):
        prepare_lease_fixture(
            fixture,
            config_path=config_path,
            metadata_output=metadata_output,
            wrapper_python=wrapper_python,
            child_python=child_python,
            benchmark_lease_script=benchmark_script,
            harness_script=harness_script,
            lease_path=lease_path,
            lock_path=lock_path,
            result_root=result_root,
        )


@pytest.mark.parametrize("result_kind", ["empty", "nonempty", "symlink"])
def test_prepare_lease_refuses_any_existing_result_path(
    tmp_path: Path, result_kind: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    result_directory = Path(fixture.config.result_directory)
    if result_kind == "symlink":
        target = tmp_path / "existing-result-target"
        target.mkdir()
        result_directory.symlink_to(target, target_is_directory=True)
    else:
        result_directory.mkdir()
        if result_kind == "nonempty":
            (result_directory / "partial.json").write_text("{}", encoding="utf-8")

    with pytest.raises(HarnessError, match="result directory already exists"):
        prepare_lease_fixture(fixture)


@pytest.mark.parametrize("output_kind", ["file", "symlink", "broken-symlink"])
def test_prepare_lease_never_replaces_an_existing_output(
    tmp_path: Path, output_kind: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    if output_kind == "file":
        fixture.metadata_output.write_text("operator-owned", encoding="utf-8")
    else:
        target = tmp_path / "operator-owned-target"
        if output_kind == "symlink":
            target.write_text("operator-owned", encoding="utf-8")
        fixture.metadata_output.symlink_to(target)

    with pytest.raises(HarnessError, match="metadata output already exists"):
        prepare_lease_fixture(fixture)

    assert fixture.metadata_output.is_symlink() == (output_kind != "file")
    if output_kind == "file":
        assert fixture.metadata_output.read_text(encoding="utf-8") == "operator-owned"


def test_atomic_metadata_install_loses_collision_race_without_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_output = (tmp_path / "metadata.json").resolve()
    operator_target = (tmp_path / "operator-target.json").resolve()
    operator_target.write_text("operator-owned", encoding="utf-8")
    original_link = os.link

    def race_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        assert dst_dir_fd is not None
        os.symlink(operator_target, destination, dir_fd=dst_dir_fd)
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(poc.os, "link", race_link)
    with pytest.raises(HarnessError, match="metadata output already exists"):
        poc._atomic_write_new_json(metadata_output, {"schema_version": 1})

    assert metadata_output.is_symlink()
    assert metadata_output.resolve() == operator_target
    assert operator_target.read_text(encoding="utf-8") == "operator-owned"
    assert not tuple(tmp_path.glob(".metadata.json.*.tmp"))


def test_prepare_lease_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(HarnessError, match="must be canonical"):
        prepare_lease_fixture(fixture, metadata_output=linked_parent / "metadata.json")

    assert not (real_parent / "metadata.json").exists()


def test_atomic_metadata_install_rejects_ancestor_replaced_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_ancestor = tmp_path / "trusted"
    trusted_parent = trusted_ancestor / "metadata-parent"
    trusted_parent.mkdir(parents=True)
    moved_trusted_ancestor = tmp_path / "moved-trusted"
    outside_ancestor = tmp_path / "outside"
    outside_parent = outside_ancestor / "metadata-parent"
    outside_parent.mkdir(parents=True)
    metadata_output = trusted_parent / "metadata.json"
    original_validation = poc._require_canonical_directory
    swapped = False

    def validate_then_swap(path: Path, description: str) -> None:
        nonlocal swapped
        original_validation(path, description)
        if description == "metadata output parent" and not swapped:
            trusted_ancestor.rename(moved_trusted_ancestor)
            trusted_ancestor.symlink_to(outside_ancestor, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(poc, "_require_canonical_directory", validate_then_swap)
    with pytest.raises(HarnessError, match="without symlinks"):
        poc._atomic_write_new_json(metadata_output, {"schema_version": 1})

    assert swapped is True
    assert not (outside_parent / metadata_output.name).exists()
    assert not (
        moved_trusted_ancestor / "metadata-parent" / metadata_output.name
    ).exists()


def test_prepare_lease_rejects_symlinked_config(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    linked_config = tmp_path / "linked-config.json"
    linked_config.symlink_to(fixture.config_path)

    with pytest.raises(HarnessError, match="must be canonical"):
        prepare_lease_fixture(fixture, config_path=linked_config)


@pytest.mark.parametrize("generated_path", ["metadata", "result", "lease", "lock"])
def test_prepare_lease_keeps_generated_paths_outside_the_source_identity(
    tmp_path: Path, generated_path: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    source_directory = Path(poc.__file__).resolve().parents[1]
    protected_path = source_directory / (
        fixture.config.run_id
        if generated_path == "result"
        else f"poc-test-must-not-create-{generated_path}"
    )
    config_path = fixture.config_path
    result_root = fixture.result_root
    if generated_path == "result":
        result_root = source_directory
        raw = fixture.config.model_dump(mode="json")
        raw["result_directory"] = str(result_root / fixture.config.run_id)
        config = HarnessConfig.model_validate_json(json.dumps(raw))
        config_path = (tmp_path / "source-result-config.json").resolve()
        config_path.write_text(config.model_dump_json(), encoding="utf-8")

    with pytest.raises(HarnessError, match="outside the source deployment"):
        prepare_lease_fixture(
            fixture,
            config_path=config_path,
            metadata_output=(
                protected_path
                if generated_path == "metadata"
                else fixture.metadata_output
            ),
            lease_path=(
                protected_path if generated_path == "lease" else fixture.lease_path
            ),
            lock_path=(
                protected_path if generated_path == "lock" else fixture.lock_path
            ),
            result_root=result_root,
        )

    assert not protected_path.exists()


def test_prepare_lease_rejects_config_result_root_mismatch(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    raw = fixture.config.model_dump(mode="json")
    raw["result_directory"] = str(fixture.result_root / "different-run")
    mismatched_config = HarnessConfig.model_validate_json(json.dumps(raw))
    mismatched_path = (tmp_path / "mismatched-result-config.json").resolve()
    mismatched_path.write_text(mismatched_config.model_dump_json(), encoding="utf-8")

    with pytest.raises(HarnessError, match="result_root/run_id"):
        prepare_lease_fixture(fixture, config_path=mismatched_path)


@pytest.mark.parametrize(
    "collision",
    [
        "metadata-result",
        "metadata-lease",
        "metadata-lock",
        "result-lease",
        "result-lock",
        "lease-lock",
    ],
)
def test_prepare_lease_requires_pairwise_distinct_generated_paths(
    tmp_path: Path, collision: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    result_directory = Path(fixture.config.result_directory)
    metadata_output = fixture.metadata_output
    lease_path = fixture.lease_path
    lock_path = fixture.lock_path
    if collision == "metadata-result":
        metadata_output = result_directory
    elif collision == "metadata-lease":
        metadata_output = lease_path
    elif collision == "metadata-lock":
        metadata_output = lock_path
    elif collision == "result-lease":
        lease_path = result_directory
    elif collision == "result-lock":
        lock_path = result_directory
    else:
        lock_path = lease_path
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(HarnessError, match="pairwise distinct"):
        prepare_lease_fixture(
            fixture,
            metadata_output=metadata_output,
            lease_path=lease_path,
            lock_path=lock_path,
        )

    assert not metadata_output.exists()


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("wrapper", "benchmark lease script does not match"),
        ("harness", "harness script does not match"),
        ("python", "child Python does not match"),
    ],
)
def test_prepare_lease_rejects_wrapper_config_invocation_mismatch(
    tmp_path: Path, mismatch: str, message: str
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    other_script = (tmp_path / "other-script.py").resolve()
    other_script.write_text("pass\n", encoding="utf-8")

    with pytest.raises(HarnessError, match=message):
        prepare_lease_fixture(
            fixture,
            benchmark_lease_script=(
                other_script
                if mismatch == "wrapper"
                else fixture.benchmark_lease_script
            ),
            harness_script=(
                other_script if mismatch == "harness" else fixture.harness_script
            ),
            child_python=(
                Path("/bin/true") if mismatch == "python" else fixture.child_python
            ),
        )


def test_prepare_lease_rejects_non_python_wrapper_executable(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)

    with pytest.raises(HarnessError, match="wrapper Python must match"):
        prepare_lease_fixture(fixture, wrapper_python=Path("/bin/true"))

    assert not fixture.metadata_output.exists()


def test_prepare_lease_rejects_configured_non_python_interpreter(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    raw = fixture.config.model_dump(mode="json")
    raw["hosts"][0]["python_executable"] = "/bin/true"
    raw["hosts"][0]["launch_argv"][3] = "/bin/true"
    config = HarnessConfig.model_validate_json(json.dumps(raw))
    config_path = (tmp_path / "non-python-config.json").resolve()
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    with pytest.raises(HarnessError, match="identity probe"):
        prepare_lease_fixture(
            fixture,
            config_path=config_path,
            wrapper_python=Path("/bin/true"),
            child_python=Path("/bin/true"),
        )

    assert not fixture.metadata_output.exists()


def test_prepare_lease_rejects_incompatible_host_source_identities(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    raw = fixture.config.model_dump(mode="json")
    raw["hosts"][1]["source"]["commit"] = "b" * 40
    incompatible = HarnessConfig.model_validate_json(json.dumps(raw))
    incompatible_path = (tmp_path / "incompatible-source-config.json").resolve()
    incompatible_path.write_text(incompatible.model_dump_json(), encoding="utf-8")

    with pytest.raises(HarnessError, match="identical source identities"):
        prepare_lease_fixture(fixture, config_path=incompatible_path)


def test_prepare_lease_rejects_stale_or_changing_source_identity(
    tmp_path: Path,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    stale = SourceIdentity(commit="b" * 40, dirty_file_hashes={})
    with pytest.raises(HarnessError, match="source identity is stale"):
        prepare_lease_fixture(fixture, source_identity=lambda _source_directory: stale)

    identities = iter((fixture.config.hosts[0].source, stale))
    with pytest.raises(HarnessError, match="source identity changed"):
        prepare_lease_fixture(
            fixture, source_identity=lambda _source_directory: next(identities)
        )
    assert not fixture.metadata_output.exists()


def test_prepare_lease_enforces_the_shared_cleanup_bound(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    raw = fixture.config.model_dump(mode="json")
    raw["timeouts"]["process_start_seconds"] = 1000.0
    slow_config = HarnessConfig.model_validate_json(json.dumps(raw))
    slow_config_path = (tmp_path / "slow-config.json").resolve()
    slow_config_path.write_text(slow_config.model_dump_json(), encoding="utf-8")
    assert poc.minimum_cleanup_grace_seconds(slow_config) > 1500.0

    with pytest.raises(HarnessError, match="shorter than the proof cleanup bound"):
        prepare_lease_fixture(
            fixture,
            config_path=slow_config_path,
            cleanup_grace_seconds=1500.0,
        )


def test_cleanup_bound_counts_sequential_signal_and_delete_poll_delays(
    tmp_path: Path,
) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    for timeout_name in (
        "api_start_seconds",
        "request_seconds",
        "process_start_seconds",
        "cleanup_seconds",
    ):
        raw["timeouts"][timeout_name] = 1.0
    raw["timeouts"]["poll_seconds"] = 1000.0
    slow_poll_config = HarnessConfig.model_validate_json(json.dumps(raw))

    assert poc.minimum_cleanup_grace_seconds(slow_poll_config) > 2000.0


@pytest.mark.parametrize(
    ("duration", "heartbeat", "cleanup", "message"),
    [
        (0.0, 10.0, 300.0, "expected duration"),
        (600.0, float("inf"), 300.0, "heartbeat interval"),
        (600.0, 10.0, float("nan"), "cleanup grace"),
    ],
)
def test_prepare_lease_rejects_nonpositive_or_nonfinite_timing(
    tmp_path: Path,
    duration: float,
    heartbeat: float,
    cleanup: float,
    message: str,
) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    with pytest.raises(HarnessError, match=message):
        prepare_lease_fixture(
            fixture,
            expected_duration_seconds=duration,
            heartbeat_seconds=heartbeat,
            cleanup_grace_seconds=cleanup,
        )


def test_prepare_lease_requires_offset_aware_clock(tmp_path: Path) -> None:
    fixture = make_lease_preparation_fixture(tmp_path)
    with pytest.raises(HarnessError, match="offset-aware"):
        prepare_lease_fixture(fixture, clock=lambda: datetime(2026, 7, 18, 12, 34, 56))
    assert not fixture.metadata_output.exists()


def test_placement_validation_patches_only_the_reserved_nccl_port(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    placement = make_placement(config)
    original = copy.deepcopy(placement)

    patched, instance_id, runners, resources = validate_and_patch_placement(
        placement, config, RUNTIME_NODE_IDS
    )

    assert placement == original
    expected = copy.deepcopy(original)
    expected["MlxNcclInstance"]["ncclCoordinator"]["port"] = 6200
    assert patched == expected
    assert instance_id == "owned-instance"
    assert runners == ("runner-0", "runner-1", "runner-2")
    assert set(resources) == {
        gpu.resource_id for host in config.hosts for gpu in host.gpus
    }


@pytest.mark.parametrize("mutation", ["revision", "resource_owner", "rank"])
def test_placement_validation_rejects_wrong_exact_topology(
    tmp_path: Path, mutation: str
) -> None:
    config = make_config(tmp_path)
    placement = make_placement(config)
    inner = placement["MlxNcclInstance"]
    assert isinstance(inner, dict)
    assignments = inner["shardAssignments"]
    assert isinstance(assignments, dict)
    if mutation == "revision":
        runner_to_shard = assignments["runnerToShard"]
        assert isinstance(runner_to_shard, dict)
        shard = runner_to_shard["runner-1"]
        assert isinstance(shard, dict)
        tensor = shard["TensorShardMetadata"]
        assert isinstance(tensor, dict)
        card = tensor["modelCard"]
        assert isinstance(card, dict)
        card["revision"] = "b" * 40
    elif mutation == "resource_owner":
        owners = assignments["computeResourceToNode"]
        assert isinstance(owners, dict)
        owners[DWAGON_GPUS[1].resource_id] = "node-fwuff"
    else:
        runner_to_shard = assignments["runnerToShard"]
        assert isinstance(runner_to_shard, dict)
        shard = runner_to_shard["runner-2"]
        assert isinstance(shard, dict)
        tensor = shard["TensorShardMetadata"]
        assert isinstance(tensor, dict)
        tensor["deviceRank"] = 1

    with pytest.raises(HarnessError):
        validate_and_patch_placement(placement, config, RUNTIME_NODE_IDS)


def test_nccl_log_evidence_binds_initialized_ranks_to_each_host(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    owner_token = "owned-log-token"
    logs = _owned_nccl_logs(config, owner_token)

    evidence = poc.validate_nccl_logs(config, make_placement(config), logs, owner_token)

    hosts = evidence["hosts"]
    assert isinstance(hosts, dict)
    assert hosts["dwagon"]["initialized_ranks"] == [0, 1]
    assert hosts["fwuff"]["initialized_ranks"] == [2]


@pytest.mark.parametrize("mutation", ["socket", "missing_rank", "wrong_host"])
def test_nccl_log_evidence_rejects_fallback_or_bad_host_rank_proof(
    tmp_path: Path, mutation: str
) -> None:
    config = make_config(tmp_path)
    owner_token = "owned-log-token"
    logs = _owned_nccl_logs(config, owner_token)
    rank_two_line = "test NCCL INFO comm 0x1 rank 2 nranks 3 - Init COMPLETE\n"
    if mutation == "socket":
        logs["fwuff"] += "test NCCL INFO NET/Socket : fallback\n"
    elif mutation == "missing_rank":
        logs["fwuff"] = logs["fwuff"].replace(rank_two_line, "")
    else:
        logs["fwuff"] = logs["fwuff"].replace(rank_two_line, "")
        logs["dwagon"] += rank_two_line

    with pytest.raises(HarnessError):
        poc.validate_nccl_logs(config, make_placement(config), logs, owner_token)


def test_nccl_log_evidence_requires_one_line_proving_merged_rails(
    tmp_path: Path,
) -> None:
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["hosts"][0]["hca_ports"].append(
        {"device": "mlx4_1", "port": 1, "gid": "fe80::3"}
    )
    raw["hosts"][0]["environment"]["NCCL_IB_HCA"] = "=mlx4_0:1,mlx4_1:1"
    config = HarnessConfig.model_validate_json(json.dumps(raw))
    owner_token = "owned-log-token"
    logs = _owned_nccl_logs(config, owner_token)
    logs["dwagon"] = logs["dwagon"].replace(
        "test NCCL INFO NET/IB : Using mlx4_0:1 mlx4_1:1",
        "test NCCL INFO NET/IB : Using mlx4_0:1\n"
        "test NCCL INFO NET/IB : Using mlx4_1:1",
    )

    with pytest.raises(HarnessError, match="merged HCA rails"):
        poc.validate_nccl_logs(config, make_placement(config), logs, owner_token)


def test_runner_failed_aborts_readiness_immediately(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.created = True
    effects.posted_instances.append(make_placement(config))
    process = effects.start_node(config.hosts[0], config, "owner-token")
    original_request = effects.request_json

    def request_with_failure(
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: JsonObject | None = None,
    ) -> JsonValue:
        if path == "/state/runners/runner-1":
            return {"RunnerFailed": {"errorMessage": "NCCL init failed"}}
        return original_request(method, path, params=params, body=body)

    effects.request_json = request_with_failure  # type: ignore[method-assign]

    with pytest.raises(HarnessError, match="NCCL init failed"):
        wait_for_owned_runners_ready(
            effects,
            config,
            [process],
            "owned-instance",
            ("runner-0", "runner-1", "runner-2"),
        )


def test_cleanup_requires_runners_resource_leases_and_assignments_to_be_gone(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    placement = make_placement(config)
    resources = [gpu.resource_id for host in config.hosts for gpu in host.gpus]
    base_state: JsonObject = {
        "instances": {},
        "runners": {},
        "retiringComputeResources": {},
        "prefillServerPorts": {},
    }
    assert cleanup_state_is_clear(
        base_state,
        "owned-instance",
        ("runner-0", "runner-1", "runner-2"),
        resources,
    )

    retiring = copy.deepcopy(base_state)
    retiring["retiringComputeResources"] = {resources[0]: "runner-0"}
    assert not cleanup_state_is_clear(
        retiring,
        "owned-instance",
        ("runner-0", "runner-1", "runner-2"),
        resources,
    )

    reassigned = copy.deepcopy(base_state)
    reassigned["instances"] = {"other-instance": placement}
    assert not cleanup_state_is_clear(
        reassigned,
        "owned-instance",
        ("runner-0", "runner-1", "runner-2"),
        resources,
    )


def test_full_harness_uses_deterministic_requests_and_owned_cleanup(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)

    result = run_harness(config, effects)

    assert result["status"] == "completed"
    assert result["reportable"] is True
    assert result["cleanup_succeeded"] is True
    assert effects.deleted_paths == ["/instance/owned-instance"]
    assert effects.stopped == list(reversed(effects.started))
    assert len(effects.completion_requests) == 5
    assert all(
        request
        == {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: NCCL proof complete.",
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
            "use_prefix_cache": False,
            "logprobs": False,
        }
        for request in effects.completion_requests
    )
    posted = effects.posted_instances[0]
    inner = posted["MlxNcclInstance"]
    assert isinstance(inner, dict)
    coordinator = inner["ncclCoordinator"]
    assert isinstance(coordinator, dict)
    assert coordinator["port"] == config.nccl_coordinator_port
    assert result["warmup_count"] == 2
    assert result["sample_count"] == 3
    assert len(result["samples"]) == 3
    assert result["performance_comparable"] is False
    assert result["requires_exact_physical_gpu_inventory"] is True
    oracle = result["correctness_oracle"]
    assert isinstance(oracle, dict)
    assert oracle["expected_content_sha256"] == config.benchmark.expected_content_sha256
    log_evidence = result["nccl_log_evidence"]
    assert isinstance(log_evidence, dict)
    assert log_evidence["hca_payload_counter_deltas_verified"] is False
    runtime_metadata = effects.writes["runtime-metadata.json"]
    assert "owner_processes" not in runtime_metadata
    assert len(runtime_metadata["owned_processes"]) == 2
    assert effects.writes["benchmark-result.json"] == result


def test_different_supported_driver_versions_do_not_break_runtime_identity(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    original_preflight = effects.run_preflight

    def differing_driver_preflight(
        host: HostConfig, harness_config: HarnessConfig
    ) -> HostPreflightReport:
        report = original_preflight(host, harness_config)
        if host.name != "fwuff":
            return report
        facts = dict(report.facts)
        facts["nvidia_driver_versions"] = "610.2"
        facts["nvidia_smi_banner"] = "Driver Version: 610.2 CUDA Version: 13.0"
        return report.model_copy(update={"facts": facts})

    effects.run_preflight = differing_driver_preflight  # type: ignore[method-assign]

    result = run_harness(config, effects)

    assert result["status"] == "completed"


def test_different_mlx_runtime_versions_fail_before_process_start(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    original_preflight = effects.run_preflight

    def differing_runtime_preflight(
        host: HostConfig, harness_config: HarnessConfig
    ) -> HostPreflightReport:
        report = original_preflight(host, harness_config)
        if host.name != "fwuff":
            return report
        facts = dict(report.facts)
        versions = dict(facts["runtime_versions"])
        versions["mlx"] = "999.0"
        facts["runtime_versions"] = versions
        return report.model_copy(update={"facts": facts})

    effects.run_preflight = differing_runtime_preflight  # type: ignore[method-assign]

    result = run_harness(config, effects)

    assert result["status"] == "preflight_failed"
    assert "runtime identity differs" in str(result["error"])
    assert effects.started == []


def test_failure_after_submission_still_deletes_only_owned_instance(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config, fail_benchmark=True)

    result = run_harness(config, effects)

    assert result["status"] == "benchmark_failed"
    assert result["reportable"] is False
    assert result["cleanup_succeeded"] is True
    assert "injected completion failure" in str(result["error"])
    assert effects.deleted_paths == ["/instance/owned-instance"]
    assert effects.stopped == list(reversed(effects.started))


def test_measured_output_mismatch_is_non_reportable_and_cleans_up(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config, vary_samples=True)

    result = run_harness(config, effects)

    assert result["status"] == "benchmark_failed"
    assert "differs from the correctness oracle" in str(result["error"])
    assert result["cleanup_succeeded"] is True
    assert effects.deleted_paths == ["/instance/owned-instance"]


def test_managed_signal_runs_owned_cleanup_before_reporting_interrupt(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    latch = SignalLatch()
    effects = FakeEffects(config, signal_latch=latch)

    result = run_harness(config, effects, latch)

    assert result["status"] == "benchmark_failed"
    assert result["interrupted_signal"] == signal.SIGTERM
    assert result["cleanup_succeeded"] is True
    assert effects.deleted_paths == ["/instance/owned-instance"]
    assert effects.stopped == list(reversed(effects.started))


class CleanHostProbe:
    def __init__(self, host: HostConfig) -> None:
        self.host = host

    def source_identity(self, source_directory: str) -> SourceIdentity:
        assert source_directory == self.host.source_directory
        return self.host.source

    def gpu_identities(self) -> tuple[GpuIdentity, ...]:
        return self.host.gpus

    def gpu_compute_processes(self) -> tuple[str, ...]:
        return ()

    def busy_ports(self, ports: tuple[int, ...], socket_type: str) -> tuple[int, ...]:
        assert ports
        assert socket_type in {"tcp", "udp"}
        return ()

    def hca_port_observations(
        self, ports: tuple[HcaPort, ...]
    ) -> tuple[HcaPortObservation, ...]:
        return tuple(
            HcaPortObservation(
                device=port.device,
                port=port.port,
                state="4: ACTIVE",
                rate="40 Gb/sec (4X QDR)",
                physical_state="5: LinkUp",
                link_layer="InfiniBand",
                lid="1",
                gids=(port.gid,),
                net_devices=(),
                ip_addresses=(),
                counters={"port_rcv_data": "1"},
            )
            for port in ports
        )

    def amx_flags(self) -> tuple[str, ...]:
        return ("amx_bf16", "amx_int8", "amx_tile")

    def online_cpu_ids(self) -> tuple[int, ...]:
        return self.host.cpu_set

    def numa_node_ids(self) -> tuple[int, ...]:
        return self.host.numa_nodes

    def process_conflicts(self, substrings: tuple[str, ...]) -> tuple[str, ...]:
        assert substrings
        return ()

    def raid_operations(self) -> tuple[str, ...]:
        return ()

    def facts(self) -> dict[str, object]:
        return _preflight_facts()


def test_linux_hca_probe_accepts_raw_verbs_without_ipoib(tmp_path: Path) -> None:
    port_root = tmp_path / "mlx4_0" / "ports" / "1"
    (port_root / "gids").mkdir(parents=True)
    (port_root / "gid_attrs" / "ndevs" / "0").mkdir(parents=True)
    (port_root / "counters").mkdir()
    for name, value in (
        ("state", "4: ACTIVE\n"),
        ("rate", "40 Gb/sec (4X QDR)\n"),
        ("phys_state", "5: LinkUp\n"),
        ("link_layer", "InfiniBand\n"),
        ("lid", "0x1\n"),
    ):
        (port_root / name).write_text(value, encoding="utf-8")
    (port_root / "gids" / "0").write_text(
        "fe80:0000:0000:0000:0010:e000:0166:3a19\n", encoding="utf-8"
    )
    (port_root / "gids" / "1").write_text("fe80::\n", encoding="utf-8")
    (port_root / "counters" / "port_rcv_data").write_text("1\n", encoding="utf-8")

    observed = poc.LinuxHostProbe(tmp_path).hca_port_observations(
        (HcaPort(device="mlx4_0", port=1, gid="fe80::10:e000:166:3a19"),)
    )

    assert len(observed) == 1
    assert observed[0].gids == ("fe80::10:e000:166:3a19",)
    assert observed[0].net_devices == ()
    assert observed[0].ip_addresses == ()
    assert observed[0].counters == {"port_rcv_data": "1"}


def test_builtin_preflight_contract_is_strict_and_self_contained(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    host = config.hosts[0]
    request = HostPreflightRequest(
        schema_version=1,
        run_id=config.run_id,
        host=host,
        reserved_ports=config.reserved_ports,
    )

    report = collect_host_preflight(request, CleanHostProbe(host))

    assert report.passed is True
    assert report.conflicts == ()
    assert report.checked_tcp_ports == config.reserved_ports
    assert report.checked_udp_ports == config.reserved_ports


@pytest.mark.parametrize("mutation", ["inactive", "wrong_gid", "zero_lid"])
def test_preflight_accepts_raw_verbs_without_ipoib_and_rejects_bad_identity(
    tmp_path: Path, mutation: str
) -> None:
    config = make_config(tmp_path)
    host = config.hosts[0]

    class MutatedHcaProbe(CleanHostProbe):
        def hca_port_observations(
            self, ports: tuple[HcaPort, ...]
        ) -> tuple[HcaPortObservation, ...]:
            observations = list(super().hca_port_observations(ports))
            first = observations[0]
            if mutation == "inactive":
                observations[0] = first.model_copy(update={"state": "4: INACTIVE"})
            elif mutation == "wrong_gid":
                observations[0] = first.model_copy(update={"gids": ("fe80::ffff",)})
            else:
                observations[0] = first.model_copy(update={"lid": "0x0000"})
            return tuple(observations)

    report = collect_host_preflight(
        HostPreflightRequest(
            schema_version=1,
            run_id=config.run_id,
            host=host,
            reserved_ports=config.reserved_ports,
        ),
        MutatedHcaProbe(host),
    )

    assert report.passed is False
    assert any("InfiniBand" in conflict for conflict in report.conflicts)


@pytest.mark.parametrize(
    "arguments",
    [
        ("/opt/exo/.venv/bin/exo", "--namespace", "other"),
        ("/opt/exo/.venv/bin/python", "-m", "exo", "--namespace", "other"),
        ("/usr/bin/uv", "run", "exo", "--namespace", "other"),
        ("/opt/exo/.venv/bin/python", "mlx_nccl_smoke.py"),
    ],
)
def test_process_conflict_matcher_covers_exo_and_nccl_launch_forms(
    arguments: tuple[str, ...],
) -> None:
    assert poc.LinuxHostProbe._command_is_conflict(arguments, ("mlx_nccl_smoke",))


def test_system_effects_preflight_uses_deployed_script_and_remote_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    host = config.hosts[1]
    effects = SystemEffects(config)
    expected_report = FakeEffects(config).run_preflight(host, config)
    observed: dict[str, object] = {}

    def fake_run_text_command(
        command_host: HostConfig,
        command: tuple[str, ...],
        *,
        stdin: str | None,
        timeout: float,
    ) -> str:
        observed.update(
            host=command_host,
            command=command,
            stdin=stdin,
            timeout=timeout,
        )
        return expected_report.model_dump_json()

    monkeypatch.setattr(effects, "_run_text_command", fake_run_text_command)

    report = effects.run_preflight(host, config)

    assert report == expected_report
    command = observed["command"]
    assert command == (
        host.python_executable,
        f"{host.source_directory}/scripts/two_host_mlx_nccl_poc.py",
        "host-preflight",
    )
    request = HostPreflightRequest.model_validate_json(str(observed["stdin"]))
    assert request.host == host
    transported = SystemEffects._transport_argv(
        host, ("python", "-c", "print('argument with spaces')")
    )
    assert transported[:2] == ["ssh", "-T"]
    assert "BatchMode=yes" in transported
    assert "StrictHostKeyChecking=yes" in transported
    assert "ConnectTimeout=10" in transported
    assert "ConnectionAttempts=1" in transported
    assert "ServerAliveInterval=5" in transported
    assert "ServerAliveCountMax=2" in transported
    assert "ControlMaster=no" in transported
    assert "ControlPath=none" in transported
    assert transported[-3:] == [
        "--",
        "fwuff",
        shlex.join(("python", "-c", "print('argument with spaces')")),
    ]
    assert "-n" not in transported
    assert transported[-1] == shlex.join(
        ("python", "-c", "print('argument with spaces')")
    )


def test_system_effects_result_output_stays_on_trusted_directory_descriptor(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "owned-result"
    result_directory.mkdir()
    raw = make_config(tmp_path).model_dump(mode="json")
    raw["result_directory"] = str(result_directory)
    config = HarnessConfig.model_validate_json(json.dumps(raw))
    result_descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    effects = SystemEffects(config, result_descriptor)
    moved_owned_directory = tmp_path / "moved-owned-result"
    outside_directory = tmp_path / "outside-result"
    outside_directory.mkdir()
    result_directory.rename(moved_owned_directory)
    result_directory.symlink_to(outside_directory, target_is_directory=True)

    try:
        effects.write_result_json(
            "benchmark-result.json",
            {"schema_version": 1, "cleanup_succeeded": False},
        )
    finally:
        os.close(result_descriptor)

    assert (moved_owned_directory / "benchmark-result.json").is_file()
    assert not (outside_directory / "benchmark-result.json").exists()


def test_inherited_result_descriptor_must_match_configured_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_directory = tmp_path / "owned-result"
    result_directory.mkdir()
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    monkeypatch.setenv("EXO_BENCHMARK_RESULT_DIRECTORY_FD", str(descriptor))
    try:
        assert (
            poc._inherited_result_directory_descriptor(result_directory) == descriptor
        )
        moved = tmp_path / "moved-result"
        outside = tmp_path / "outside-result"
        outside.mkdir()
        result_directory.rename(moved)
        result_directory.symlink_to(outside, target_is_directory=True)
        with pytest.raises(HarnessError, match="identity changed"):
            poc._inherited_result_directory_descriptor(result_directory)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("host_index", [0, 1])
def test_probe_commands_use_exact_configured_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_index: int
) -> None:
    config = make_config(tmp_path)
    host = config.hosts[host_index]
    effects = SystemEffects(config)
    captured: dict[str, object] = {}
    monkeypatch.setenv("NCCL_UNDECLARED_PARENT_VALUE", "must-not-leak")

    def fake_run(
        arguments: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured.update(arguments=arguments, kwargs=kwargs)
        return subprocess.CompletedProcess(arguments, 0, stdout="ok", stderr="")

    monkeypatch.setattr(poc.subprocess, "run", fake_run)

    assert (
        effects._run_text_command(
            host,
            (host.python_executable, "-c", "print('ok')"),
            timeout=1.0,
        )
        == "ok"
    )

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["env"] == host.environment
    assert "NCCL_UNDECLARED_PARENT_VALUE" not in kwargs["env"]
    arguments = captured["arguments"]
    assert isinstance(arguments, list)
    if host.transport == "ssh":
        remote_command = shlex.split(arguments[-1])
        assert remote_command[:2] == ["/usr/bin/env", "-i"]
        assert "NCCL_UNDECLARED_PARENT_VALUE=must-not-leak" not in remote_command
        assert set(remote_command[2 : 2 + len(host.environment)]) == {
            f"{name}={value}" for name, value in host.environment.items()
        }


class StubPopen:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode: int | None = None
        self.wait_calls: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0


def test_local_start_failure_terminates_unregistered_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects = SystemEffects(config)
    process = StubPopen()
    signals: list[tuple[int, signal.Signals]] = []
    popen_kwargs: dict[str, object] = {}
    monkeypatch.setenv("NCCL_UNDECLARED_PARENT_VALUE", "must-not-leak")

    def fake_popen(*_args: object, **kwargs: object) -> StubPopen:
        popen_kwargs.update(kwargs)
        return process

    monkeypatch.setattr(poc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        effects,
        "_read_process_identity",
        lambda _pid: (_ for _ in ()).throw(OSError("injected stat failure")),
    )
    monkeypatch.setattr(
        poc.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal))
    )

    with pytest.raises(poc.StartNodeError, match="injected stat failure") as caught:
        effects.start_node(config.hosts[0], config, "owner-token")

    assert caught.value.cleanup.ownership_verified is False
    assert caught.value.cleanup.terminated is False
    assert popen_kwargs["env"] == config.hosts[0].environment | {
        "EXO_BENCHMARK_OWNER_TOKEN": "owner-token"
    }
    assert "NCCL_UNDECLARED_PARENT_VALUE" not in popen_kwargs["env"]
    assert signals == []
    assert process.wait_calls == []


def test_remote_malformed_receipt_closes_supervisor_stdin_before_transport_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects = SystemEffects(config)
    process = StubPopen()
    popen_call: dict[str, object] = {}
    markers = iter(("EXO_SUPERVISOR_READY", "malformed receipt"))
    signals: list[tuple[int, signal.Signals]] = []

    def fake_popen(arguments: list[str], **kwargs: object) -> StubPopen:
        popen_call.update(arguments=arguments, kwargs=kwargs)
        return process

    monkeypatch.setattr(poc.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(effects, "_read_start_marker", lambda *_args: next(markers))
    monkeypatch.setattr(
        poc.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal))
    )

    with pytest.raises(poc.StartNodeError, match="invalid ownership receipt") as caught:
        effects.start_node(config.hosts[1], config, "owner-token")

    assert caught.value.cleanup.ownership_verified is False
    assert caught.value.cleanup.terminated is False
    assert process.stdin.closed is True
    assert signals == []
    assert process.wait_calls == [1.0]
    remote_command = str(popen_call["arguments"][-1])
    assert "EXO_SUPERVISOR_READY" in remote_command
    assert "start_new_session=True" in remote_command
    assert 'sys.stdin.readline() == ""' in remote_command
    assert "EXO_BENCHMARK_OWNER_TOKEN" in remote_command


def test_unregistered_local_cleanup_fails_closed_when_signal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects = SystemEffects(config)
    process = StubPopen()
    monkeypatch.setattr(
        effects, "_read_process_identity", lambda _pid: (process.pid, 12345)
    )
    monkeypatch.setattr(
        effects, "_local_group_ownership", lambda _receipt: (True, (process.pid,))
    )
    monkeypatch.setattr(
        poc.os,
        "killpg",
        lambda _pid, _signal: (_ for _ in ()).throw(OSError("injected kill failure")),
    )

    cleanup = effects._terminate_unregistered_transport(
        config.hosts[0],
        process,
        owner_token="owner-token",
        namespace=config.namespace,
        log_path=tmp_path / "local.log",
        timeout_seconds=1.0,
    )

    assert cleanup.ownership_verified is False
    assert cleanup.terminated is False
    assert "injected kill failure" in str(cleanup.error)


def test_parsed_failed_start_receipt_is_persisted_and_cleanup_fails_closed(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    class FailedSecondStartEffects(FakeEffects):
        def start_node(
            self, host: HostConfig, harness_config: HarnessConfig, owner_token: str
        ) -> OwnedProcess:
            if host.name == "fwuff":
                receipt = OwnedProcess(
                    host_name=host.name,
                    pid=7777,
                    process_group_id=7777,
                    start_time_ticks=8888,
                    owner_token=owner_token,
                    namespace=harness_config.namespace,
                    transport_pid=9999,
                    log_path="/results/fwuff.log",
                )
                raise poc.StartNodeError(
                    host.name,
                    RuntimeError("post-receipt failure"),
                    ProcessCleanup(
                        host.name,
                        ownership_verified=True,
                        terminated=False,
                        forced=False,
                        error="owned child survived",
                    ),
                    receipt,
                )
            return super().start_node(host, harness_config, owner_token)

    effects = FailedSecondStartEffects(config)
    result = run_harness(config, effects)

    assert result["status"] == "cleanup_failed"
    owned_processes = result["owned_processes"]
    assert isinstance(owned_processes, list)
    assert {process["host_name"] for process in owned_processes} == {
        "dwagon",
        "fwuff",
    }
    runtime_metadata = effects.writes["runtime-metadata.json"]
    metadata_processes = runtime_metadata["owned_processes"]
    assert isinstance(metadata_processes, list)
    assert {process["host_name"] for process in metadata_processes} == {
        "dwagon",
        "fwuff",
    }
    assert len(result["process_cleanup"]) == 2


def test_stop_node_fails_if_ssh_output_pump_does_not_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    effects = SystemEffects(config)
    host = config.hosts[1]
    transport = StubPopen(pid=9000)
    log_file = io.StringIO()

    class StuckPump:
        def __init__(self) -> None:
            self.join_calls: list[float] = []

        def join(self, timeout: float) -> None:
            self.join_calls.append(timeout)

        def is_alive(self) -> bool:
            return True

    pump = StuckPump()
    process = OwnedProcess(
        host_name=host.name,
        pid=7000,
        process_group_id=7000,
        start_time_ticks=8000,
        owner_token="owner-token",
        namespace=config.namespace,
        transport_pid=transport.pid,
        log_path=str(tmp_path / "exo-fwuff.log"),
    )
    effects._running[host.name] = poc._RunningHandle(  # type: ignore[arg-type]
        transport,
        log_file,
        pump,  # type: ignore[arg-type]
    )
    cleanup_receipt = poc.ProcessCleanupReceipt(
        host_name=host.name,
        ownership_verified=True,
        terminated=True,
        forced=False,
    )
    monkeypatch.setattr(
        effects,
        "_run_text_command",
        lambda *_args, **_kwargs: cleanup_receipt.model_dump_json(),
    )

    cleanup = effects.stop_node(process, timeout_seconds=1.0)

    assert cleanup.ownership_verified is True
    assert cleanup.terminated is False
    assert "output pump did not finish" in str(cleanup.error)
    assert pump.join_calls == [5.0]
    assert log_file.closed is False


def test_remote_supervisor_cleans_child_on_stdin_eof(tmp_path: Path) -> None:
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "LD_LIBRARY_PATH": "",
    }
    owner_token = "supervisor-contract-token"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            poc._REMOTE_LAUNCH_PROGRAM,
            owner_token,
            "supervisor-contract-namespace",
            str(tmp_path),
            json.dumps(environment, sort_keys=True),
            json.dumps(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                sort_keys=True,
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "EXO_SUPERVISOR_READY"
    process.stdin.write("START\n")
    process.stdin.flush()
    owner_line = process.stdout.readline().strip()
    assert owner_line.startswith("EXO_OWNER ")
    receipt = poc.RemoteOwnerReceipt.model_validate_json(
        owner_line.removeprefix("EXO_OWNER ")
    )

    process.stdin.close()
    remaining_output = process.stdout.read()
    process.wait(timeout=15.0)

    assert "EXO_SUPERVISOR_CLEAN" in remaining_output.splitlines()
    assert "EXO_SUPERVISOR_LOG_COMPLETE" in remaining_output.splitlines()
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{receipt.pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{receipt.pid}").exists()


def test_model_probe_uses_indexed_bytes_and_hashes_full_snapshot(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"llama"}')
    shard = model_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"physical bytes include a safetensors header")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 10},
                "weight_map": {
                    "model.embed.weight": shard.name,
                    "model.layer.weight": shard.name,
                },
            }
        )
    )
    (model_path / ".exo-huggingface-revision.json").write_text(
        json.dumps({"repo_id": MODEL_ID, "revision": REVISION})
    )
    unrelated = model_path / "unrelated.bin"
    unrelated.write_bytes(b"tokenizer-side files must also be hashed")

    completed = subprocess.run(
        [
            poc.sys.executable,
            "-c",
            poc._MODEL_PROBE_PROGRAM,
            "dwagon",
            str(model_path),
            MODEL_ID,
            REVISION,
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = ModelProbeResult.model_validate_json(completed.stdout)

    assert result.verified is True
    assert result.weight_bytes == 10
    assert result.physical_weight_bytes == shard.stat().st_size
    assert set(result.sha256_manifest) == {
        "config.json",
        "model.safetensors.index.json",
        shard.name,
        unrelated.name,
    }


def test_model_probe_rejects_symlinked_snapshot_root(tmp_path: Path) -> None:
    real_model_path = tmp_path / "real-model"
    real_model_path.mkdir()
    symlink_path = tmp_path / "linked-model"
    symlink_path.symlink_to(real_model_path, target_is_directory=True)

    completed = subprocess.run(
        [
            poc.sys.executable,
            "-c",
            poc._MODEL_PROBE_PROGRAM,
            "dwagon",
            str(symlink_path),
            MODEL_ID,
            REVISION,
            "10",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = ModelProbeResult.model_validate_json(completed.stdout)

    assert result.verified is False
    assert "symlink" in str(result.error).lower()


@pytest.mark.parametrize(
    ("mutation", "elapsed_seconds"),
    [
        ("finish_reason", 0.1),
        ("empty_content", 0.1),
        ("float_prompt_tokens", 0.1),
        ("zero_generation_tokens", 0.1),
        ("nan_prompt_rate", 0.1),
        ("infinite_generation_rate", 0.1),
        ("elapsed", 0.0),
    ],
)
def test_completion_validation_rejects_vacuous_or_nonfinite_success(
    mutation: str, elapsed_seconds: float
) -> None:
    response: JsonObject = {
        "id": "completion",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "complete"},
                "finish_reason": "stop",
            }
        ],
        "generation_stats": {
            "prompt_tps": 10.0,
            "generation_tps": 5.0,
            "prompt_tokens": 2,
            "generation_tokens": 1,
        },
    }
    choice = response["choices"][0]
    assert isinstance(choice, dict)
    message = choice["message"]
    assert isinstance(message, dict)
    statistics = response["generation_stats"]
    assert isinstance(statistics, dict)
    if mutation == "finish_reason":
        choice["finish_reason"] = None
    elif mutation == "empty_content":
        message["content"] = ""
    elif mutation == "float_prompt_tokens":
        statistics["prompt_tokens"] = 2.5
    elif mutation == "zero_generation_tokens":
        statistics["generation_tokens"] = 0
    elif mutation == "nan_prompt_rate":
        statistics["prompt_tps"] = float("nan")
    elif mutation == "infinite_generation_rate":
        statistics["generation_tps"] = float("inf")

    with pytest.raises(HarnessError):
        poc._completion_result(
            response,
            elapsed_seconds=elapsed_seconds,
            iteration=0,
            expected_model_id=MODEL_ID,
            expected_content_sha256=(
                "eebbf6457e46a7f63acdf9b97390f790ba443d60cfa44b607da7e5c40aa1cc1d"
            ),
        )


def test_model_manifest_mismatch_aborts_before_process_start(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    original_probe = effects.probe_model

    def mismatched_probe(host: HostConfig, model: ModelSnapshot) -> ModelProbeResult:
        result = original_probe(host, model)
        if host.name == "fwuff":
            manifest = dict(result.sha256_manifest)
            manifest["config.json"] = "f" * 64
            return result.model_copy(update={"sha256_manifest": manifest})
        return result

    effects.probe_model = mismatched_probe  # type: ignore[method-assign]

    result = run_harness(config, effects)

    assert result["status"] == "preflight_failed"
    assert "SHA-256 manifests differ" in str(result["error"])
    assert effects.started == []


def test_stop_failure_does_not_skip_other_owned_process_cleanup(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    original_stop = effects.stop_node
    attempted_hosts: list[str] = []

    def stop_with_one_failure(
        process: OwnedProcess, timeout_seconds: float
    ) -> ProcessCleanup:
        attempted_hosts.append(process.host_name)
        if len(attempted_hosts) == 1:
            raise RuntimeError("injected stop adapter failure")
        return original_stop(process, timeout_seconds)

    effects.stop_node = stop_with_one_failure  # type: ignore[method-assign]

    result = run_harness(config, effects)

    assert attempted_hosts == ["fwuff", "dwagon"]
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False
    assert len(result["process_cleanup"]) == 2
    assert "injected stop adapter failure" in str(result["process_cleanup"][0])


def test_failed_preflight_never_starts_a_process(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    original_preflight = effects.run_preflight

    def failed_preflight(
        host: HostConfig, harness_config: HarnessConfig
    ) -> HostPreflightReport:
        report = original_preflight(host, harness_config)
        if host.name == "fwuff":
            return report.model_copy(
                update={"passed": False, "conflicts": ("unowned GPU process",)}
            )
        return report

    effects.run_preflight = failed_preflight  # type: ignore[method-assign]

    result = run_harness(config, effects)

    assert result["status"] == "preflight_failed"
    assert result["reportable"] is False
    assert effects.started == []
    assert effects.deleted_paths == []
    assert "unowned GPU process" in str(result["error"])
