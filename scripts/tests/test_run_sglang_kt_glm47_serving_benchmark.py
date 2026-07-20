from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from exo.shared.types.common import ModelId, NodeId
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
    GLM_4_7_FLASH_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
    calculate_sglang_kt_process_launch_spec_sha256,
)
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
    SglangKtModelRuntimeValidationReceiptObservation,
    calculate_sglang_kt_model_runtime_validator_bundle_sha256,
)
from exo.worker.sglang_kt.receipt_io import SglangKtBoundFile
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptObservation,
)
from exo.worker.sglang_kt.serving_benchmark_receipt import (
    SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
    SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH,
    ServingWorkloadKind,
    SglangKtServingAdmissionBinding,
    SglangKtServingClientIdentity,
    SglangKtServingCoordinationGuardEvidence,
    SglangKtServingFileIdentity,
    SglangKtServingInvocationEvidence,
    SglangKtServingJitCacheEvidence,
    SglangKtServingModelFilesystemEvidence,
    SglangKtServingModelIdentity,
    SglangKtServingOwnedServerProcessIdentity,
    SglangKtServingProcessSpecIdentity,
    SglangKtServingRuntimeIdentity,
    SglangKtServingSanityEvidence,
    SglangKtServingServerInfoIdentity,
    SglangKtServingSetupEvidence,
    SglangKtServingSourceFileIdentity,
    SglangKtServingSourceIdentity,
    SglangKtServingTopologyIdentity,
    SglangKtServingTopologyStage,
    SglangKtServingTuningIdentity,
    SglangKtServingWorkloadEvidence,
    SglangKtWarmServingRunIdentity,
    calculate_sglang_kt_length_finish_reason_sha256,
    calculate_sglang_kt_serving_coordination_guard_evidence_sha256,
    calculate_sglang_kt_serving_source_bundle_sha256,
    calculate_sglang_kt_token_ids_sha256,
    calculate_sglang_kt_warm_serving_run_identity_sha256,
    load_sglang_kt_warm_serving_run_receipt,
)
from scripts import run_sglang_kt_glm47_serving_benchmark as harness
from scripts import run_sglang_kt_glm47_validation as validation
from scripts.create_sglang_kt_glm47_validation_process_spec import (
    create_process_spec,
)
from scripts.create_sglang_kt_glm47_validation_process_spec import (
    parse_arguments as parse_process_spec_arguments,
)
from scripts.sglang_kt_glm47_serving_client import (
    Glm47NativeServingClient,
    PreparedServingWorkload,
)

GPU_UUID = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
OTHER_GPU_UUID = "GPU-63a7760a-6164-0758-9228-03dbf35d721c"
SHA = "a" * 64


def config_payload(tmp_path: Path) -> dict[str, object]:
    run_id = "glm47-serving-test"
    deployment = tmp_path / "deployment"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "namespace": f"exo-{run_id}",
        "phase": "serving_baseline",
        "profiler": "none",
        "hca_requirement": "metadata_only",
        "result_directory": f"/var/lib/exo/benchmarks/{run_id}",
        "scratch_directory": f"/var/lib/exo/validation-scratch/{run_id}",
        "source": {
            "repository": "/root/exo",
            "deployment_root": str(deployment),
        },
        "runtime_python": {
            "path": "/runtime/venv/bin/python",
            "sha256": SHA,
            "symlink_chain": ["python3.12", "/runtime/base/bin/python3.12"],
        },
        "numactl_executable": "/usr/bin/numactl",
        "build_receipt": {
            "path": "/runtime/build-receipt.json",
            "sha256": "b" * 64,
        },
        "model_path": "/var/lib/exo/models/glm47",
        "model_contract": {
            "path": str(
                deployment / "orchestrator" / validation.MODEL_CONTRACT_RELATIVE_PATH
            ),
            "sha256": validation.GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        },
        "host": {
            "hostname": "dwagon",
            "node_id": "dwagon",
            "gpu": {
                "uuid": GPU_UUID,
                "pci_address": "00000000:01:00.0",
            },
            "cpu_cores": [0, 1, 2, 3],
            "memory_nodes": [0],
            "threads_per_subpool": [4],
            "cpu_infer_threads": 4,
            "threadpool_count": 1,
            "hca_bindings": [
                {
                    "device": "mlx4_0",
                    "port": 1,
                    "gid_index": 0,
                    "gid": "fe80:0000:0000:0000:0210:e000:0166:3a19",
                },
                {
                    "device": "mlx4_0",
                    "port": 2,
                    "gid_index": 0,
                    "gid": "fe80:0000:0000:0000:0210:e000:0166:3a1a",
                },
            ],
        },
        "distributed_coordinator": {"ip": "127.0.0.1", "port": 29510},
        "service_endpoint": {"ip": "127.0.0.1", "port": 30100},
        "reserved_ports": [29510, 30100],
        "resident_gpu_experts": 4,
        "timeouts": {
            "generator_seconds": 30.0,
            "kernel_seconds": 120.0,
            "model_seconds": 3600.0,
            "cleanup_seconds": 120.0,
        },
        "admission": {
            "validator_sha256": "c" * 64,
            "process_spec_sha256": "d" * 64,
            "model_runtime_validation_receipt": {
                "path": "/receipts/model.json",
                "sha256": "e" * 64,
            },
            "kernel_runtime_validation_receipt": {
                "path": "/receipts/kernel.json",
                "sha256": "f" * 64,
            },
            "model_contract_receipt": {
                "path": "/receipts/contract.json",
                "sha256": validation.GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
            },
        },
        "tools": {
            "numactl": {"path": "/usr/bin/numactl", "sha256": "1" * 64},
            "nvidia_smi": {
                "path": "/usr/bin/nvidia-smi",
                "sha256": "2" * 64,
            },
            "systemctl": {"path": "/usr/bin/systemctl", "sha256": "3" * 64},
        },
        "lease_execution": {
            "owner": "serving-test-owner",
            "purpose": "serving-test-purpose",
            "expected_duration_seconds": 3600.0,
            "cleanup_grace_seconds": 300.0,
            "heartbeat_seconds": 30.0,
            "metadata_output": f"/var/lib/exo/benchmarks/{run_id}-metadata.json",
            "lease_path": "/run/exo/benchmark-lease.json",
            "lock_path": "/run/exo/benchmark.lock",
            "result_root": "/var/lib/exo/benchmarks",
        },
        "coordination_guard": {
            "model_filesystem": {
                "mount_point": "/",
                "mount_source": "/dev/nvme0n1p3",
                "filesystem_type": "xfs",
                "device_major": 259,
                "device_minor": 3,
            },
            "local_hca": {
                "device": "mlx4_0",
                "node_guid": "e41d:2d03:004d:32d1",
                "ports": [
                    {
                        "port": 1,
                        "gid_index": 0,
                        "gid": "fe80:0000:0000:0000:0210:e000:0166:3a19",
                        "expected_rate": "40 Gb/sec (4X QDR)",
                        "health_counter_maximums": {"link_downed": 0},
                        "idle_data_counter_maximum_deltas": {
                            "port_rcv_data": 1000,
                            "port_rcv_packets": 1000,
                            "port_xmit_data": 1000,
                            "port_xmit_packets": 1000,
                        },
                    },
                    {
                        "port": 2,
                        "gid_index": 0,
                        "gid": "fe80:0000:0000:0000:0210:e000:0166:3a1a",
                        "expected_rate": "40 Gb/sec (4X QDR)",
                        "health_counter_maximums": {"link_downed": 0},
                        "idle_data_counter_maximum_deltas": {
                            "port_rcv_data": 1000,
                            "port_rcv_packets": 1000,
                            "port_xmit_data": 1000,
                            "port_xmit_packets": 1000,
                        },
                    },
                ],
            },
            "idle_peer": {
                "schema_version": 1,
                "peer_role": "idle_nonparticipant",
                "peer": {
                    "schema_version": 1,
                    "hostname": "fwuff",
                    "ssh": {
                        "executable": {
                            "path": "/usr/bin/ssh",
                            "resolved_path": "/usr/bin/ssh",
                            "sha256": "4" * 64,
                        },
                        "target": "fwuff",
                        "user": "root",
                        "port": 22,
                        "known_hosts_file": {
                            "path": "/root/.ssh/known_hosts",
                            "resolved_path": "/root/.ssh/known_hosts",
                            "sha256": "5" * 64,
                        },
                        "identity_file": {
                            "path": "/root/.ssh/id_ed25519",
                            "resolved_path": "/root/.ssh/id_ed25519",
                            "sha256": "6" * 64,
                        },
                        "connect_timeout_seconds": 5,
                        "server_alive_interval_seconds": 5,
                        "server_alive_count_max": 2,
                    },
                    "remote_probe": {
                        "python": {
                            "path": "/usr/bin/python3",
                            "resolved_path": "/usr/bin/python3",
                            "sha256": "7" * 64,
                        },
                        "script": {
                            "path": "/opt/exo/benchmark_host_guard.py",
                            "resolved_path": "/opt/exo/benchmark_host_guard.py",
                            "sha256": "8" * 64,
                        },
                    },
                    "tools": {
                        "nvidia_smi": {
                            "path": "/usr/bin/nvidia-smi",
                            "resolved_path": "/usr/bin/nvidia-smi",
                            "sha256": "9" * 64,
                        },
                        "systemctl": {
                            "path": "/usr/bin/systemctl",
                            "resolved_path": "/usr/bin/systemctl",
                            "sha256": "a" * 64,
                        },
                    },
                    "cpu_memory": {
                        "minimum_online_cpu_count": 4,
                        "numa_cpu_sets": {"0": [0, 1], "1": [2, 3]},
                        "minimum_total_memory_bytes": 137438953472,
                    },
                    "gpus": [
                        {
                            "uuid": "GPU-11111111-2222-3333-4444-555555555555",
                            "pci_bus_id": "00000000:01:00.0",
                            "name": "NVIDIA GeForce RTX 3090",
                            "memory_total_bytes": 25769803776,
                        }
                    ],
                    "hca": {
                        "device": "mlx4_0",
                        "node_guid": "e41d:2d03:004d:32e1",
                        "ports": [
                            {
                                "port": 1,
                                "gid_index": 0,
                                "gid": "fe80:0000:0000:0000:0210:e000:0166:3b19",
                                "expected_rate": "40 Gb/sec (4X QDR)",
                                "health_counter_maximums": {"link_downed": 0},
                                "idle_data_counter_maximum_deltas": {
                                    "port_rcv_data": 1000,
                                    "port_rcv_packets": 1000,
                                    "port_xmit_data": 1000,
                                    "port_xmit_packets": 1000,
                                },
                            },
                            {
                                "port": 2,
                                "gid_index": 0,
                                "gid": "fe80:0000:0000:0000:0210:e000:0166:3b1a",
                                "expected_rate": "40 Gb/sec (4X QDR)",
                                "health_counter_maximums": {"link_downed": 0},
                                "idle_data_counter_maximum_deltas": {
                                    "port_rcv_data": 1000,
                                    "port_rcv_packets": 1000,
                                    "port_xmit_data": 1000,
                                    "port_xmit_packets": 1000,
                                },
                            },
                        ],
                    },
                    "opensm_units": [
                        {
                            "unit": "opensm-port1.service",
                            "port": 1,
                            "guid": "0xe41d2d03004d32e1",
                            "executable": {
                                "path": "/usr/sbin/opensm",
                                "resolved_path": "/usr/sbin/opensm",
                                "sha256": "b" * 64,
                            },
                            "argv": [
                                "/usr/sbin/opensm",
                                "--guid",
                                "0xe41d2d03004d32e1",
                            ],
                            "version": "OpenSM 3.3.24",
                        },
                        {
                            "unit": "opensm-port2.service",
                            "port": 2,
                            "guid": "0xe41d2d03004d32e2",
                            "executable": {
                                "path": "/usr/sbin/opensm",
                                "resolved_path": "/usr/sbin/opensm",
                                "sha256": "b" * 64,
                            },
                            "argv": [
                                "/usr/sbin/opensm",
                                "--guid",
                                "0xe41d2d03004d32e2",
                            ],
                            "version": "OpenSM 3.3.24",
                        },
                    ],
                    "reserved_ports": [29510, 30100],
                    "policy": {
                        "maximum_load_1m_per_online_cpu": 0.5,
                        "minimum_available_memory_bytes": 68719476736,
                        "maximum_gpu_memory_used_bytes": 134217728,
                        "maximum_gpu_utilization_percent": 2,
                        "maximum_gpu_memory_utilization_percent": 2,
                        "maximum_gpu_temperature_celsius": 70,
                        "maximum_clock_skew_ns": 1000000000,
                    },
                },
                "cross_host_fabric": {
                    "rails": [
                        {
                            "local_port": 1,
                            "remote_port": 1,
                            "local_gid": "fe80:0000:0000:0000:0210:e000:0166:3a19",
                            "remote_gid": "fe80:0000:0000:0000:0210:e000:0166:3b19",
                            "rate": "40 Gb/sec (4X QDR)",
                            "subnet_manager_host": "remote",
                            "subnet_manager_unit": "opensm-port1.service",
                            "subnet_manager_guid": "0xe41d2d03004d32e1",
                            "subnet_manager_argv": [
                                "/usr/sbin/opensm",
                                "--guid",
                                "0xe41d2d03004d32e1",
                            ],
                        },
                        {
                            "local_port": 2,
                            "remote_port": 2,
                            "local_gid": "fe80:0000:0000:0000:0210:e000:0166:3a1a",
                            "remote_gid": "fe80:0000:0000:0000:0210:e000:0166:3b1a",
                            "rate": "40 Gb/sec (4X QDR)",
                            "subnet_manager_host": "remote",
                            "subnet_manager_unit": "opensm-port2.service",
                            "subnet_manager_guid": "0xe41d2d03004d32e2",
                            "subnet_manager_argv": [
                                "/usr/sbin/opensm",
                                "--guid",
                                "0xe41d2d03004d32e2",
                            ],
                        },
                    ]
                },
            },
        },
        "request_timeout_seconds": 900.0,
        "readiness_timeout_seconds": 900.0,
    }


def make_config(tmp_path: Path) -> harness.ServingBenchmarkConfig:
    return harness.ServingBenchmarkConfig.model_validate_json(
        json.dumps(config_payload(tmp_path))
    )


def validator_sources(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(root / relative_path),
            hashlib.sha256(relative_path.encode()).hexdigest(),
        )
        for relative_path in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    )


def make_process_spec(
    config: harness.ServingBenchmarkConfig,
) -> SglangKtProcessLaunchSpec:
    arguments = parse_process_spec_arguments(
        [
            "--launch-mode",
            "serving_baseline",
            "--model-path",
            config.model_path,
            "--runtime-python",
            config.runtime_python.path,
            "--output",
            "/tmp/process-spec.json",
            "--node-id",
            config.host.node_id,
            "--gpu-uuid",
            config.host.gpu.uuid,
            "--cpu-cores",
            ",".join(str(core) for core in config.host.cpu_cores),
            "--memory-nodes",
            ",".join(str(node) for node in config.host.memory_nodes),
            "--cpu-infer-threads",
            str(config.host.cpu_infer_threads),
            "--threadpool-count",
            str(config.host.threadpool_count),
            "--distributed-coordinator",
            config.distributed_coordinator.argument,
            "--service-endpoint",
            config.service_endpoint.argument,
            "--resident-gpu-experts",
            str(config.resident_gpu_experts),
        ]
    )
    return create_process_spec(arguments)


def admitted_capability_receipts(
    config: harness.ServingBenchmarkConfig,
    *,
    admitted_gpu_uuid: str | None = None,
    kernel_gpu_uuid: str | None = None,
    resident_gpu_experts: int | None = None,
) -> tuple[
    SglangKtModelRuntimeValidationReceiptObservation,
    SglangKtKernelRuntimeValidationReceiptObservation,
]:
    observed_admitted_gpu_uuid = admitted_gpu_uuid or config.host.gpu.uuid
    observed_kernel_gpu_uuid = kernel_gpu_uuid or observed_admitted_gpu_uuid
    admitted_cpu_cores = (56, 57, 58, 59)
    admitted_memory_nodes = (1,)
    model = cast(
        SglangKtModelRuntimeValidationReceiptObservation,
        SimpleNamespace(
            process_spec_sha256=config.admission.process_spec_sha256,
            target_profile=GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
            model_id=harness.GLM_4_7_FLASH_BF16_MODEL_ID,
            model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
            model_path=config.model_path,
            model_config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
            gpu_uuid=observed_admitted_gpu_uuid,
            gpu_compute_capability=(8, 6),
            cpu_cores=admitted_cpu_cores,
            memory_nodes=admitted_memory_nodes,
            resident_gpu_experts=(
                resident_gpu_experts
                if resident_gpu_experts is not None
                else config.resident_gpu_experts
            ),
            executed_cpu_backend="AMX_BF16",
            sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
            ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
            kernel_runtime_validation_receipt_path=(
                config.admission.kernel_runtime_validation_receipt.path
            ),
            kernel_runtime_validation_receipt_sha256=(
                config.admission.kernel_runtime_validation_receipt.sha256
            ),
            model_contract_path=config.admission.model_contract_receipt.path,
            model_contract_receipt_sha256=(
                config.admission.model_contract_receipt.sha256
            ),
            model_contract_sha256=harness.GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
            sgl_kernel_build_id="4" * 64,
            deep_gemm_build_id="5" * 64,
            kt_kernel_build_id="6" * 64,
            torch_version="2.9.1+cu128",
            cuda_version="12.8",
        ),
    )
    kernel = cast(
        SglangKtKernelRuntimeValidationReceiptObservation,
        SimpleNamespace(
            receipt_path=config.admission.kernel_runtime_validation_receipt.path,
            receipt_sha256=config.admission.kernel_runtime_validation_receipt.sha256,
            executable=config.runtime_python.path,
            hostname=config.host.hostname,
            gpu_uuid=observed_kernel_gpu_uuid,
            gpu_compute_capability=(8, 6),
            cpu_cores=admitted_cpu_cores,
            memory_nodes=admitted_memory_nodes,
            build_receipt_path=config.build_receipt.path,
            build_receipt_sha256=config.build_receipt.sha256,
            sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
            ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
            sgl_kernel_build_id="4" * 64,
            deep_gemm_build_id="5" * 64,
            kt_kernel_build_id="6" * 64,
            torch_version="2.9.1+cu128",
            cuda_version="12.8",
        ),
    )
    return model, kernel


def fake_scratch(tmp_path: Path) -> validation.OwnedScratchDirectory:
    tmp_path.mkdir(exist_ok=True)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    observed = os.fstat(descriptor)
    return validation.OwnedScratchDirectory(
        path=tmp_path,
        parent_descriptor=os.dup(descriptor),
        descriptor=descriptor,
        device=observed.st_dev,
        inode=observed.st_ino,
    )


def invocation(
    kind: ServingWorkloadKind, ordinal: int
) -> SglangKtServingInvocationEvidence:
    output_count, input_hash, output_hash = (
        (
            32,
            harness.prepare_glm47_serving_workload(
                "prefill"
            ).receipt_request.input_ids_sha256,
            "1" * 64,
        )
        if kind == "prefill"
        else (
            128,
            harness.prepare_glm47_serving_workload(
                "decode"
            ).receipt_request.input_ids_sha256,
            "2" * 64,
        )
    )
    generation_seconds = 1.0
    return SglangKtServingInvocationEvidence(
        ordinal=ordinal,
        input_ids_sha256=input_hash,
        cache_flush_status_code=200,
        cache_flush_response_sha256="3" * 64,
        prompt_tokens=1_024 if kind == "prefill" else 128,
        completion_tokens=output_count,
        cached_tokens=0,
        output_ids_sha256=output_hash,
        finish_reason_sha256=calculate_sglang_kt_length_finish_reason_sha256(
            output_count
        ),
        stream_line_count=output_count + 1,
        stream_event_count=output_count,
        output_bearing_event_count=output_count,
        maximum_stream_line_bytes=512,
        first_stream_event_output_tokens=1,
        total_client_seconds=3.0,
        client_observed_ttft_seconds=1.0,
        client_observed_generation_window_seconds=generation_seconds,
        client_observed_decode_tokens_per_second=(output_count - 1)
        / generation_seconds,
        ttft_semantics="client_stream_first_output_event_including_http_and_queue_v1",
    )


def serving_identity(
    config: harness.ServingBenchmarkConfig,
) -> SglangKtWarmServingRunIdentity:
    def file_identity(path: str, marker: str) -> SglangKtServingFileIdentity:
        return SglangKtServingFileIdentity(
            path=path,
            size_bytes=123,
            sha256=marker * 64,
        )

    source_files = (
        SglangKtServingSourceFileIdentity(
            relative_path=SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
            size_bytes=111,
            sha256="1" * 64,
        ),
        SglangKtServingSourceFileIdentity(
            relative_path=SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH,
            size_bytes=222,
            sha256="2" * 64,
        ),
    )
    source_sha256 = calculate_sglang_kt_serving_source_bundle_sha256(source_files)
    return SglangKtWarmServingRunIdentity(
        admission=SglangKtServingAdmissionBinding(
            model_runtime_validation_receipt=file_identity("/receipts/model.json", "3"),
            kernel_runtime_validation_receipt=file_identity(
                "/receipts/kernel.json", "4"
            ),
            model_contract_receipt=file_identity("/receipts/contract.json", "5"),
        ),
        runtime=SglangKtServingRuntimeIdentity(
            executable=config.runtime_python.path,
            runtime_build_receipt=file_identity("/receipts/build.json", "6"),
            numactl_executable=file_identity("/usr/bin/numactl", "7"),
            nvidia_smi_executable=file_identity("/usr/bin/nvidia-smi", "8"),
            systemctl_executable=file_identity("/usr/bin/systemctl", "9"),
            runtime_build_id="a" * 64,
            python_version="3.12.11",
            torch_version="2.9.1+cu128",
            cuda_version="12.8",
            sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
            ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
            sgl_kernel_build_id="b" * 64,
            deep_gemm_build_id="c" * 64,
            kt_kernel_build_id="d" * 64,
        ),
        model=SglangKtServingModelIdentity(
            model_id=ModelId("zai-org/GLM-4.7-Flash"),
            model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
            model_path=config.model_path,
            model_config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
            model_index_sha256="b" * 64,
            physical_weight_bytes=62_444_175_504,
        ),
        process_spec=SglangKtServingProcessSpecIdentity(
            receipt=file_identity("/receipts/process-spec.json", "c"),
            process_spec_sha256="d" * 64,
            launch_argv_sha256="e" * 64,
            launch_environment_sha256="f" * 64,
            target_profile=GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
            resident_gpu_experts=4,
            cpu_cores=config.host.cpu_cores,
            memory_nodes=config.host.memory_nodes,
        ),
        tuning=SglangKtServingTuningIdentity(
            mode="untuned",
            config_directory=None,
            manifest_sha256=None,
            config_files=(),
        ),
        client=SglangKtServingClientIdentity(
            source_file=SglangKtServingFileIdentity(
                path=f"/source/{SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH}",
                size_bytes=source_files[0].size_bytes,
                sha256=source_files[0].sha256,
            ),
            source_bundle_sha256=source_sha256,
            protocol_version=1,
            http_library="httpx",
            http_library_version="0.28.1",
            request_timeout_seconds=config.request_timeout_seconds,
        ),
        source=SglangKtServingSourceIdentity(
            repository_root="/source",
            commit="1" * 40,
            source_bundle_sha256=source_sha256,
            files=source_files,
            dirty_files=(),
        ),
        topology=SglangKtServingTopologyIdentity(
            deployment="local",
            interconnect="none",
            stages=(
                SglangKtServingTopologyStage(
                    pipeline_rank=0,
                    node_id=NodeId(config.host.node_id),
                    host=config.service_endpoint.ip,
                    port=config.service_endpoint.port,
                    gpu_uuid=config.host.gpu.uuid,
                    hca_devices=(),
                ),
            ),
        ),
        server_info=(
            SglangKtServingServerInfoIdentity(
                node_id=NodeId(config.host.node_id),
                host=config.service_endpoint.ip,
                port=config.service_endpoint.port,
                canonical_response_sha256="2" * 64,
                version="0.0.0.dev0",
                model_path=config.model_path,
                tp_size=1,
                pp_size=1,
                nnodes=1,
                node_rank=0,
                disable_radix_cache=True,
            ),
        ),
    )


def serving_measurement(
    config: harness.ServingBenchmarkConfig,
    cgroup_path: Path,
    handoff: harness.ServingHandoffIdentity | None = None,
) -> harness.WarmServingMeasurementV3:
    identity = serving_identity(config)
    prefill = harness.prepare_glm47_serving_workload("prefill").receipt_request
    decode = harness.prepare_glm47_serving_workload("decode").receipt_request
    workloads = (
        SglangKtServingWorkloadEvidence(
            request=prefill,
            warmups=tuple(invocation("prefill", item) for item in range(1, 3)),
            samples=tuple(invocation("prefill", item) for item in range(1, 4)),
        ),
        SglangKtServingWorkloadEvidence(
            request=decode,
            warmups=tuple(invocation("decode", item) for item in range(1, 3)),
            samples=tuple(invocation("decode", item) for item in range(1, 4)),
        ),
    )
    sanity_output_ids = (3257, 46, 62674, 3333, 8374)
    sanity = SglangKtServingSanityEvidence(
        prompt="Reply with exactly EXO_SANITY_OK and nothing else.",
        marker="EXO_SANITY_OK",
        chat_template_sha256=(
            "d63ad536c3c81880043e22ec7fd08db42b4d8fb7c89c7138bc562bfa25281375"
        ),
        rendered_prompt_sha256=(
            "62acda2056933064acbc3211ff3474871d746256bc57e3cd195032e9507b7604"
        ),
        tokenizer_class="TokenizersBackend",
        input_token_count=17,
        input_ids_sha256=(
            "0506b57087c67f3f4a3c92b488e60484f2fc6ffa5d32609df1d8e07ed438f876"
        ),
        max_new_tokens=16,
        sampling_seed=20_260_719,
        temperature=0.0,
        ignore_eos=False,
        stream=False,
        return_logprob=False,
        log_metrics=False,
        prompt_tokens=17,
        completion_tokens=len(sanity_output_ids),
        output_ids=sanity_output_ids,
        output_ids_sha256=calculate_sglang_kt_token_ids_sha256(sanity_output_ids),
        server_output_text="EXO_SANITY_OK",
        locally_decoded_output_text="EXO_SANITY_OK",
        finish_reason_type="stop",
        finish_reason_sha256="6" * 64,
        total_client_seconds=0.5,
        post_sanity_cache_flush_status_code=200,
        post_sanity_cache_flush_response_sha256="7" * 64,
    )
    coordination_guard = coordination_evidence(config)
    return harness.WarmServingMeasurementV3(
        schema_version=3,
        status="passed",
        generated_at_utc="2026-07-19T20:00:00+00:00",
        profiler="none",
        instrumentation="none",
        radix_cache_disabled=True,
        max_concurrent_requests=1,
        lease_id="a" * 32,
        identity_sha256=calculate_sglang_kt_warm_serving_run_identity_sha256(identity),
        identity=identity,
        setup=SglangKtServingSetupEvidence(
            process_launch_seconds=1.0,
            health_ready_seconds=2.0,
            admission_seconds=3.0,
            server_info_fetch_seconds=0.1,
            health_generate_status_code=200,
            health_generate_response_sha256="1" * 64,
        ),
        sanity=sanity,
        jit_cache=SglangKtServingJitCacheEvidence(
            cache_directories=("/cache/triton",),
            after_penultimate_warmup_manifest_sha256="2" * 64,
            after_final_warmup_manifest_sha256="2" * 64,
            after_measurement_manifest_sha256="2" * 64,
        ),
        workloads=workloads,
        coordination_guard=coordination_guard,
        owner_token="owner-token",
        server_process=SglangKtServingOwnedServerProcessIdentity(
            pid=1234,
            proc_start_time_ticks=5678,
            executable=identity.runtime.executable,
            argv_sha256=identity.process_spec.launch_argv_sha256,
            cpu_affinity=identity.process_spec.cpu_cores,
            memory_nodes=identity.process_spec.memory_nodes,
        ),
        server_return_code=-15,
        termination_signal="SIGTERM",
        forced=False,
        owned_processes_absent=True,
        delegated_cgroup_path=str(cgroup_path),
        delegated_cgroup_removed=True,
        transient_unit_name=validation.systemd_unit_name(config),
        handoff=handoff
        or harness.ServingHandoffIdentity(
            result_directory=harness.FilesystemObjectIdentity(
                path=config.result_directory,
                kind="directory",
                device=1,
                inode=2,
                owner_uid=0,
            ),
            coordination_lock=harness.FilesystemObjectIdentity(
                path="/run/exo/benchmark.lock",
                kind="regular_file",
                device=1,
                inode=3,
                owner_uid=0,
            ),
        ),
    )


def coordination_evidence(
    config: harness.ServingBenchmarkConfig,
) -> SglangKtServingCoordinationGuardEvidence:
    guard_config = config.coordination_guard.idle_peer
    config_sha256 = harness.host_guard.calculate_host_guard_config_sha256(guard_config)
    binding_sha256 = harness.host_guard.calculate_coordination_peer_binding_sha256(
        guard_config.peer
    )
    preflight: dict[str, object] = {
        "phase": "preflight",
        "binding_sha256": binding_sha256,
        "config_sha256": config_sha256,
        "snapshot_sha256": "a" * 64,
    }
    postflight: dict[str, object] = {
        "phase": "postflight",
        "binding_sha256": binding_sha256,
        "config_sha256": config_sha256,
        "snapshot_sha256": "b" * 64,
    }
    comparison: dict[str, object] = {
        "stable": True,
        "failures": [],
        "binding_sha256": binding_sha256,
        "config_sha256": config_sha256,
        "preflight_snapshot_sha256": "a" * 64,
        "postflight_snapshot_sha256": "b" * 64,
    }
    local_preflight: dict[str, object] = {"device": "mlx4_0", "marker": 1}
    local_postflight: dict[str, object] = {"device": "mlx4_0", "marker": 2}
    filesystem = SglangKtServingModelFilesystemEvidence(
        model_path=config.model_path,
        mount_point="/",
        mount_source="/dev/nvme0n1p3",
        filesystem_type="xfs",
        device_major=259,
        device_minor=3,
        local_block_filesystem=True,
    )
    evidence_sha256 = calculate_sglang_kt_serving_coordination_guard_evidence_sha256(
        host_guard_config_sha256=config_sha256,
        peer_binding_sha256=binding_sha256,
        remote_preflight=preflight,
        remote_postflight=postflight,
        comparison=comparison,
        local_hca_preflight=local_preflight,
        local_hca_postflight=local_postflight,
        model_filesystem=filesystem,
    )
    return SglangKtServingCoordinationGuardEvidence(
        peer_role="idle_nonparticipant",
        host_guard_config_sha256=config_sha256,
        peer_binding_sha256=binding_sha256,
        remote_preflight=preflight,
        remote_postflight=postflight,
        comparison=comparison,
        local_hca_preflight=local_preflight,
        local_hca_postflight=local_postflight,
        model_filesystem=filesystem,
        remote_peer_unchanged=True,
        cross_host_fabric_validated=True,
        evidence_sha256=evidence_sha256,
    )


def local_result_config(tmp_path: Path) -> harness.ServingBenchmarkConfig:
    payload = config_payload(tmp_path)
    run_id = cast(str, payload["run_id"])
    payload["result_directory"] = str(tmp_path / run_id)
    execution = cast(dict[str, object], payload["lease_execution"])
    execution["result_root"] = str(tmp_path)
    execution["metadata_output"] = str(tmp_path / f"{run_id}-metadata.json")
    return harness.ServingBenchmarkConfig.model_validate_json(json.dumps(payload))


def open_handoff(result_path: Path, lock_path: Path) -> harness.PreservedServingHandoff:
    lock_file = lock_path.open("r+b")
    descriptor = os.open(result_path, os.O_RDONLY | os.O_DIRECTORY)
    results = validation.ResultDirectory(result_path, descriptor)
    os.close(descriptor)
    return harness.PreservedServingHandoff(
        lock_file=lock_file,
        results=results,
        identity=harness.ServingHandoffIdentity(
            result_directory=harness.filesystem_object_identity(
                result_path, results.descriptor, "directory"
            ),
            coordination_lock=harness.filesystem_object_identity(
                lock_path, lock_file.fileno(), "regular_file"
            ),
        ),
    )


def mock_successful_finalization_proofs(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrapper_manifest(
        _config: harness.ServingBenchmarkConfig,
        _deployment: validation.DeploymentIdentity,
        _results: validation.ResultDirectory,
        _snapshot: harness.BoundMeasurementSnapshot,
    ) -> harness.JsonObject:
        return {}

    def identity_files(
        _config: harness.ServingBenchmarkConfig,
        _deployment: validation.DeploymentIdentity,
        _measurement: harness.WarmServingMeasurementV3,
    ) -> None:
        return None

    def processes_absent(
        _manifest: harness.JsonObject,
        _measurement: harness.WarmServingMeasurementV3,
    ) -> None:
        return None

    def unit_removed(_config: harness.ServingBenchmarkConfig, _unit_name: str) -> None:
        return None

    def port_clear(_host: str, _port: int) -> None:
        return None

    def gpu_clear(_config: harness.ServingBenchmarkConfig) -> None:
        return None

    monkeypatch.setattr(harness, "_validate_wrapper_manifest", wrapper_manifest)
    monkeypatch.setattr(harness, "_validate_identity_files", identity_files)
    monkeypatch.setattr(harness, "_require_processes_absent", processes_absent)
    monkeypatch.setattr(harness, "_require_unit_removed", unit_removed)
    monkeypatch.setattr(harness, "require_port_clear", port_clear)
    monkeypatch.setattr(harness, "require_gpu_clear", gpu_clear)


def test_serving_config_requires_exact_phase_and_distinct_admission_paths(
    tmp_path: Path,
) -> None:
    wrong_phase = config_payload(tmp_path)
    wrong_phase["phase"] = "hybrid"
    with pytest.raises(ValidationError, match="warm serving"):
        harness.ServingBenchmarkConfig.model_validate_json(json.dumps(wrong_phase))

    duplicate = config_payload(tmp_path)
    admission = cast(dict[str, object], duplicate["admission"])
    admission["kernel_runtime_validation_receipt"] = admission[
        "model_runtime_validation_receipt"
    ]
    with pytest.raises(ValidationError, match="distinct paths"):
        harness.ServingBenchmarkConfig.model_validate_json(json.dumps(duplicate))


def test_admitted_capabilities_allow_a_recorded_performance_variant(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    model, kernel = admitted_capability_receipts(config)

    assert (
        calculate_sglang_kt_process_launch_spec_sha256(process_spec)
        != config.admission.process_spec_sha256
    )
    assert model.cpu_cores != process_spec.cpu_cores
    assert model.memory_nodes != process_spec.memory_nodes

    harness._validate_admission_cross_bindings(config, process_spec, model, kernel)


def test_admitted_capabilities_allow_full_two_numa_cpu_variant(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    host = cast(dict[str, object], payload["host"])
    host.update(
        {
            "cpu_cores": list(range(112)),
            "memory_nodes": [0, 1],
            "threads_per_subpool": [56, 56],
            "cpu_infer_threads": 112,
            "threadpool_count": 2,
        }
    )
    config = harness.ServingBenchmarkConfig.model_validate_json(json.dumps(payload))
    process_spec = make_process_spec(config)
    model, kernel = admitted_capability_receipts(config)

    assert process_spec.cpu_cores == tuple(range(112))
    assert process_spec.memory_nodes == (0, 1)
    assert process_spec.stage.cpu_infer_threads == 112
    assert process_spec.stage.threadpool_count == 2
    harness._validate_admission_cross_bindings(config, process_spec, model, kernel)


def test_performance_variant_process_spec_must_match_current_config(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    model, kernel = admitted_capability_receipts(config)
    changed = config_payload(tmp_path)
    changed["resident_gpu_experts"] = 3
    changed_config = harness.ServingBenchmarkConfig.model_validate_json(
        json.dumps(changed)
    )

    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="configured variant",
    ):
        harness._validate_admission_cross_bindings(
            changed_config, process_spec, model, kernel
        )


def test_reused_model_and_kernel_admission_must_describe_one_baseline(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    model, kernel = admitted_capability_receipts(config, kernel_gpu_uuid=OTHER_GPU_UUID)

    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="reusable capability evidence",
    ):
        harness._validate_admission_cross_bindings(config, process_spec, model, kernel)


@pytest.mark.parametrize(
    ("admitted_gpu_uuid", "resident_gpu_experts"),
    (
        (OTHER_GPU_UUID, None),
        (None, 3),
    ),
)
def test_reused_admission_remains_bound_to_gpu_and_expert_count(
    tmp_path: Path,
    admitted_gpu_uuid: str | None,
    resident_gpu_experts: int | None,
) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    model, kernel = admitted_capability_receipts(
        config,
        admitted_gpu_uuid=admitted_gpu_uuid,
        resident_gpu_experts=resident_gpu_experts,
    )

    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="reusable capability evidence",
    ):
        harness._validate_admission_cross_bindings(config, process_spec, model, kernel)


def test_recorded_process_spec_identity_binds_variant_digest_and_placement(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    identity = serving_identity(config)
    variant_sha256 = calculate_sglang_kt_process_launch_spec_sha256(process_spec)
    recorded = identity.process_spec.model_copy(
        update={
            "process_spec_sha256": variant_sha256,
            "launch_argv_sha256": harness._canonical_sha256(
                list(harness._server_command(config, process_spec))
            ),
        }
    )
    variant_identity = identity.model_copy(update={"process_spec": recorded})

    harness._validate_recorded_process_spec_identity(
        config, variant_identity, process_spec
    )

    admitted_identity = identity.model_copy(
        update={
            "process_spec": recorded.model_copy(
                update={"process_spec_sha256": config.admission.process_spec_sha256}
            )
        }
    )
    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="generated process spec",
    ):
        harness._validate_recorded_process_spec_identity(
            config, admitted_identity, process_spec
        )

    wrong_placement = variant_identity.model_copy(
        update={"process_spec": recorded.model_copy(update={"cpu_cores": (0, 1, 2)})}
    )
    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="generated process spec",
    ):
        harness._validate_recorded_process_spec_identity(
            config, wrong_placement, process_spec
        )

    wrong_stage = variant_identity.topology.stages[0].model_copy(
        update={"host": "127.0.0.2"}
    )
    wrong_topology = variant_identity.model_copy(
        update={
            "topology": variant_identity.topology.model_copy(
                update={"stages": (wrong_stage,)}
            )
        }
    )
    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="generated process spec",
    ):
        harness._validate_recorded_process_spec_identity(
            config, wrong_topology, process_spec
        )

    wrong_argv = variant_identity.model_copy(
        update={
            "process_spec": recorded.model_copy(update={"launch_argv_sha256": "0" * 64})
        }
    )
    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="generated process spec",
    ):
        harness._validate_recorded_process_spec_identity(
            config, wrong_argv, process_spec
        )


def test_measurement_timestamp_must_be_utc(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = serving_measurement(config, tmp_path / "cgroup").model_dump(mode="json")
    payload["generated_at_utc"] = "2026-07-19T16:00:00-04:00"
    with pytest.raises(ValidationError, match="must be UTC"):
        harness.WarmServingMeasurementV3.model_validate_json(json.dumps(payload))


def test_measurement_accepts_sglang_self_sigkill_after_sigterm(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    payload = serving_measurement(config, tmp_path / "cgroup").model_dump(mode="json")
    payload["server_return_code"] = -9

    measurement = harness.WarmServingMeasurementV3.model_validate_json(
        json.dumps(payload)
    )

    assert measurement.server_return_code == -9
    assert measurement.termination_signal == "SIGTERM"
    assert measurement.forced is False


def test_measurement_v3_rejects_v2_payload(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = serving_measurement(config, tmp_path / "cgroup").model_dump(mode="json")
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        harness.WarmServingMeasurementV3.model_validate_json(json.dumps(payload))


def test_measurement_v3_rejects_v1_payload(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = serving_measurement(config, tmp_path / "cgroup").model_dump(mode="json")
    payload["schema_version"] = 1
    with pytest.raises(ValidationError):
        harness.WarmServingMeasurementV3.model_validate_json(json.dumps(payload))


def test_measurement_v3_requires_sanity(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = serving_measurement(config, tmp_path / "cgroup").model_dump(mode="json")
    del payload["sanity"]
    with pytest.raises(ValidationError):
        harness.WarmServingMeasurementV3.model_validate_json(json.dumps(payload))


def relocated_validator_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    change_current_content: bool,
) -> tuple[harness.ServingBenchmarkConfig, validation.DeploymentIdentity]:
    admitted_sources = validator_sources(tmp_path / "admitted" / "validator")
    current_sources = validator_sources(tmp_path / "deployment" / "validator")
    if change_current_content:
        current_sources = ((current_sources[0][0], "f" * 64), *current_sources[1:])

    payload = config_payload(tmp_path)
    admission = cast(dict[str, object], payload["admission"])
    admission["validator_sha256"] = (
        calculate_sglang_kt_model_runtime_validator_bundle_sha256(admitted_sources)
    )
    admitted_contents = json.dumps(
        {
            "validator_sources": [
                {"path": path, "sha256": sha256} for path, sha256 in admitted_sources
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    admitted_receipt_sha256 = hashlib.sha256(admitted_contents).hexdigest()
    model_receipt_binding = cast(
        dict[str, object], admission["model_runtime_validation_receipt"]
    )
    model_receipt_binding["sha256"] = admitted_receipt_sha256
    config = harness.ServingBenchmarkConfig.model_validate_json(json.dumps(payload))
    deployment = validation.DeploymentIdentity(
        root=config.source.deployment_root,
        orchestrator_sha256="1" * 64,
        validator_sha256=(
            calculate_sglang_kt_model_runtime_validator_bundle_sha256(current_sources)
        ),
        validator_files=tuple(
            {
                "path": path,
                "size_bytes": 0 if path.endswith("/__init__.py") else 123,
                "sha256": sha256,
            }
            for path, sha256 in current_sources
        ),
        source=validation.SourceIdentity("3" * 40, {}),
    )

    def load_admitted_receipt(
        path: Path,
        *,
        expected_validator_sha256: str,
        expected_process_spec_sha256: str,
        expected_model_contract_receipt_sha256: str,
        expected_kernel_receipt_sha256: str,
        expected_receipt_sha256: str | None = None,
    ) -> SglangKtModelRuntimeValidationReceiptObservation:
        assert path == Path(config.admission.model_runtime_validation_receipt.path)
        assert expected_validator_sha256 == config.admission.validator_sha256
        assert expected_process_spec_sha256 == config.admission.process_spec_sha256
        assert (
            expected_model_contract_receipt_sha256
            == config.admission.model_contract_receipt.sha256
        )
        assert (
            expected_kernel_receipt_sha256
            == config.admission.kernel_runtime_validation_receipt.sha256
        )
        assert (
            expected_receipt_sha256
            == config.admission.model_runtime_validation_receipt.sha256
        )
        return cast(
            SglangKtModelRuntimeValidationReceiptObservation,
            SimpleNamespace(receipt_sha256=admitted_receipt_sha256),
        )

    def read_admitted_receipt(path: Path, *, maximum_bytes: int) -> SglangKtBoundFile:
        assert path == Path(config.admission.model_runtime_validation_receipt.path)
        assert maximum_bytes >= len(admitted_contents)
        return SglangKtBoundFile(
            path=path,
            contents=admitted_contents,
            sha256=admitted_receipt_sha256,
        )

    monkeypatch.setattr(
        harness,
        "load_sglang_kt_model_runtime_validation_receipt",
        load_admitted_receipt,
    )
    monkeypatch.setattr(
        harness,
        "read_sglang_kt_bound_file",
        read_admitted_receipt,
    )
    return config, deployment


def test_relocated_validator_content_matches_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deployment = relocated_validator_admission(
        tmp_path, monkeypatch, change_current_content=False
    )
    assert deployment.validator_sha256 != config.admission.validator_sha256

    harness.require_admitted_validator(config, deployment)


def test_relocated_validator_rejects_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, deployment = relocated_validator_admission(
        tmp_path, monkeypatch, change_current_content=True
    )

    with pytest.raises(harness.Glm47ServingHarnessError, match="content differs"):
        harness.require_admitted_validator(config, deployment)


def test_bound_helper_executable_rejects_content_replacement(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"first")
    executable.chmod(0o755)
    binding = validation.ArtifactBinding(
        path=str(executable), sha256=hashlib.sha256(b"first").hexdigest()
    )
    identity = harness.bound_executable_identity(binding, "test tool")
    assert identity.sha256 == binding.sha256
    executable.write_bytes(b"second")
    with pytest.raises(harness.Glm47ServingHarnessError, match="identity changed"):
        harness.bound_executable_identity(binding, "test tool")


@pytest.mark.parametrize("subcommand", ["execute", "finalize"])
def test_outer_entry_points_reject_relative_paths(subcommand: str) -> None:
    arguments = [
        "--config",
        "relative-config.json",
        "--lease-path",
        "/run/exo/benchmark-lease.json",
        "--lock-path",
        "/run/exo/benchmark.lock",
    ]
    if subcommand == "execute":
        arguments.extend(["--", "/bin/true"])
        entry_point = harness.execute_main
    else:
        entry_point = harness.finalize_main
    with pytest.raises(harness.Glm47ServingHarnessError, match="must be absolute"):
        entry_point(arguments)


def test_open_or_create_lock_bootstraps_missing_runtime_parent(
    tmp_path: Path,
) -> None:
    runtime_parent = tmp_path / "run" / "exo"
    lock_path = runtime_parent / "benchmark.lock"

    with harness._open_or_create_lock(lock_path) as first_lock:
        first_identity = os.fstat(first_lock.fileno())

    assert runtime_parent.is_dir()
    assert lock_path.is_file()
    with harness._open_or_create_lock(lock_path) as reopened_lock:
        reopened_identity = os.fstat(reopened_lock.fileno())

    assert (reopened_identity.st_dev, reopened_identity.st_ino) == (
        first_identity.st_dev,
        first_identity.st_ino,
    )


@pytest.mark.parametrize("unsafe_kind", ["file", "symlink"])
def test_open_or_create_lock_rejects_unsafe_existing_runtime_parent(
    tmp_path: Path, unsafe_kind: str
) -> None:
    runtime_root = tmp_path / "run"
    runtime_root.mkdir()
    runtime_parent = runtime_root / "exo"
    if unsafe_kind == "file":
        runtime_parent.write_text("not a directory", encoding="ascii")
    else:
        target = tmp_path / "untrusted"
        target.mkdir()
        runtime_parent.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        harness.Glm47ServingHarnessError,
        match="cannot safely create coordination lock parent",
    ):
        harness._open_or_create_lock(runtime_parent / "benchmark.lock")

    if unsafe_kind == "symlink":
        assert not (tmp_path / "untrusted" / "benchmark.lock").exists()


def test_serving_environment_is_sanitized_and_owns_all_caches(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    process_spec = make_process_spec(config)
    scratch = fake_scratch(tmp_path / "scratch")
    try:
        caches = harness.create_cache_directories(scratch)
        environment = harness.build_serving_environment(
            config,
            process_spec,
            "owner-token",
            scratch,
            caches,
            parent_environment={
                "PATH": "/untrusted/bin",
                "HOME": "/untrusted/home",
                "SPT_NOENV": "",
                "SGLANG_MOE_CONFIG_DIR": "/foreign",
                "NCCL_DEBUG": "TRACE",
            },
        )
    finally:
        os.close(scratch.descriptor)
        os.close(scratch.parent_descriptor)
    assert environment["PATH"] == harness.FIXED_CHILD_PATH
    assert environment["HOME"] == str(scratch.path / "home")
    assert environment["SPT_NOENV"] == "1"
    assert environment["TMPDIR"] == str(scratch.path / "tmp")
    assert environment["TEMP"] == environment["TMPDIR"]
    assert environment["TMP"] == environment["TMPDIR"]
    assert environment["CUDA_VISIBLE_DEVICES"] == config.host.gpu.uuid
    assert environment["EXO_BENCHMARK_OWNER_TOKEN"] == "owner-token"
    assert not any(name.startswith(("SGLANG_", "NCCL_")) for name in environment)
    assert {
        Path(environment[name]).parent
        for name in (
            "CUDA_CACHE_PATH",
            "HF_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
        )
    } == {scratch.path / "cache"}
    assert {scratch.path, scratch.path / "home", scratch.path / "tmp"}.issubset(
        set(caches)
    )


def test_cache_manifest_accepts_owned_socket_and_rejects_unsafe_entries(
    tmp_path: Path,
) -> None:
    first = tmp_path / "cache" / "triton"
    second = tmp_path / "cache" / "cuda"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "kernel.bin").write_bytes(b"first")
    initial = harness.stable_cache_manifest(
        (second, first), timeout_seconds=0.1, quiet_seconds=0.001
    )
    (first / "kernel.bin").write_bytes(b"second")
    changed = harness.stable_cache_manifest(
        (second, first), timeout_seconds=0.1, quiet_seconds=0.001
    )
    assert initial != changed

    with socket.socket(socket.AF_UNIX) as endpoint:
        endpoint.bind(str(second / "endpoint.sock"))
        socket_manifest = harness.stable_cache_manifest(
            (second, first), timeout_seconds=0.1, quiet_seconds=0.001
        )
    assert socket_manifest != changed

    hardlink = second / "hardlink"
    os.link(first / "kernel.bin", hardlink)
    with pytest.raises(
        harness.Glm47ServingHarnessError, match="special or multiply linked"
    ):
        harness.stable_cache_manifest(
            (second, first), timeout_seconds=0.1, quiet_seconds=0.001
        )
    hardlink.unlink()

    os.mkfifo(second / "named-pipe")
    with pytest.raises(
        harness.Glm47ServingHarnessError, match="special or multiply linked"
    ):
        harness.stable_cache_manifest(
            (second, first), timeout_seconds=0.1, quiet_seconds=0.001
        )

    (second / "named-pipe").unlink()
    (second / "foreign").symlink_to(first / "kernel.bin")
    with pytest.raises(harness.Glm47ServingHarnessError, match="foreign link"):
        harness.stable_cache_manifest(
            (second, first), timeout_seconds=0.1, quiet_seconds=0.001
        )


@pytest.mark.parametrize("relative_path", ["home/state.json", "tmp/kernel.tmp"])
def test_writable_home_and_temp_mutations_change_stability_manifest(
    tmp_path: Path, relative_path: str
) -> None:
    scratch = fake_scratch(tmp_path / "scratch")
    try:
        writable_roots = harness.create_cache_directories(scratch)
        initial = harness.stable_cache_manifest(
            writable_roots, timeout_seconds=0.1, quiet_seconds=0.001
        )
        changed_path = scratch.path / relative_path
        changed_path.write_bytes(b"late write")
        changed = harness.stable_cache_manifest(
            writable_roots, timeout_seconds=0.1, quiet_seconds=0.001
        )
        assert changed != initial
    finally:
        os.close(scratch.descriptor)
        os.close(scratch.parent_descriptor)


def test_interleaved_workload_order_and_manifest_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    calls: list[tuple[str, int]] = []
    manifests = iter(("a" * 64, "a" * 64, "a" * 64))

    def run_invocation(
        _client: Glm47NativeServingClient,
        workload: PreparedServingWorkload,
        ordinal: int,
    ) -> SglangKtServingInvocationEvidence:
        kind = workload.receipt_request.kind
        calls.append((kind, ordinal))
        return invocation(kind, ordinal)

    monkeypatch.setattr(harness, "run_glm47_serving_invocation", run_invocation)

    def cache_manifest(_paths: tuple[Path, ...]) -> str:
        return next(manifests)

    process_identity = serving_measurement(config, tmp_path / "cgroup").server_process

    def validate_server(
        _config: harness.ServingBenchmarkConfig,
        _server: harness.ServerProcess,
    ) -> SglangKtServingOwnedServerProcessIdentity:
        return process_identity

    monkeypatch.setattr(harness, "stable_cache_manifest", cache_manifest)
    monkeypatch.setattr(harness, "observe_running_server_process", validate_server)
    server = cast(harness.ServerProcess, object())
    workloads, cache = harness.collect_interleaved_workloads(
        cast(harness.Glm47NativeServingClient, object()),
        server,
        config,
        (tmp_path,),
        validation.SignalLatch(),
        process_identity,
    )
    assert calls == [
        ("prefill", 1),
        ("decode", 1),
        ("prefill", 2),
        ("decode", 2),
        ("prefill", 1),
        ("decode", 1),
        ("prefill", 2),
        ("decode", 2),
        ("prefill", 3),
        ("decode", 3),
    ]
    assert tuple(item.request.kind for item in workloads) == ("prefill", "decode")
    assert cache.after_measurement_manifest_sha256 == "a" * 64


def test_interleaved_workloads_reject_second_warmup_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    manifests = iter(("a" * 64, "b" * 64))

    def run_invocation(
        _client: Glm47NativeServingClient,
        workload: PreparedServingWorkload,
        ordinal: int,
    ) -> SglangKtServingInvocationEvidence:
        return invocation(workload.receipt_request.kind, ordinal)

    def cache_manifest(_paths: tuple[Path, ...]) -> str:
        return next(manifests)

    process_identity = serving_measurement(config, tmp_path / "cgroup").server_process

    def validate_server(
        _config: harness.ServingBenchmarkConfig,
        _server: harness.ServerProcess,
    ) -> SglangKtServingOwnedServerProcessIdentity:
        return process_identity

    monkeypatch.setattr(
        harness,
        "run_glm47_serving_invocation",
        run_invocation,
    )
    monkeypatch.setattr(harness, "stable_cache_manifest", cache_manifest)
    monkeypatch.setattr(harness, "observe_running_server_process", validate_server)
    with pytest.raises(harness.Glm47ServingHarnessError, match="between the two"):
        harness.collect_interleaved_workloads(
            cast(harness.Glm47NativeServingClient, object()),
            cast(harness.ServerProcess, object()),
            config,
            (tmp_path,),
            validation.SignalLatch(),
            process_identity,
        )


def test_server_placement_is_observed_from_proc_and_affinity(tmp_path: Path) -> None:
    runtime = tmp_path / "python"
    runtime.write_bytes(b"runtime")
    runtime.chmod(0o755)
    payload = config_payload(tmp_path)
    runtime_payload = cast(dict[str, object], payload["runtime_python"])
    runtime_payload["path"] = str(runtime)
    host_payload = cast(dict[str, object], payload["host"])
    host_payload["memory_nodes"] = [0, 1]
    host_payload["threads_per_subpool"] = [2, 2]
    host_payload["threadpool_count"] = 2
    config = harness.ServingBenchmarkConfig.model_validate_json(json.dumps(payload))
    working_directory = tmp_path / "scratch"
    working_directory.mkdir()
    process_id = 4242
    process_root = tmp_path / "proc" / str(process_id)
    process_root.mkdir(parents=True)
    (process_root / "exe").symlink_to(runtime)
    (process_root / "cwd").symlink_to(working_directory)
    command = (
        config.tools.numactl.path,
        "--physcpubind",
        "0,1,2,3",
        "--membind",
        "0,1",
        str(runtime),
        "server.py",
    )
    (process_root / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in command[5:]) + b"\0"
    )
    stat_fields = ["0"] * 20
    stat_fields[0] = "S"
    stat_fields[19] = "5678"
    (process_root / "stat").write_text(
        f"{process_id} (server) {' '.join(stat_fields)}\n", encoding="ascii"
    )
    (process_root / "status").write_text(
        "Cpus_allowed_list:\t0-3\nMems_allowed_list:\t0-1\n",
        encoding="ascii",
    )
    (process_root / "numa_maps").write_text(
        "00400000 bind:0-1 file=/runtime\n"
        "00600000 bind:0,1 heap anon=8 dirty=8 N0=4 N1=4\n"
        "7f000000 default file=/dev/nvidiactl\n"
        "7f100000 default file=/dev/shm/torch_4242_0 shmem N0=1\n"
        "7f200000 local file=/dev/nvidia-uvm\n",
        encoding="ascii",
    )

    class RunningProcess:
        pid = process_id
        returncode: int | None = None

        def poll(self) -> None:
            return None

    server = harness.ServerProcess(
        process=cast(subprocess.Popen[bytes], cast(object, RunningProcess())),
        owned=validation.OwnedProcess(
            host_name=config.host.hostname,
            pid=process_id,
            process_group_id=process_id,
            start_time_ticks=5678,
            transport_pid=process_id,
            namespace=config.namespace,
            owner_token="owner-token",
            log_path="/tmp/server.log",
        ),
        command=command,
        environment={},
        working_directory=str(working_directory),
        launch_argv_sha256="4" * 64,
        launch_seconds=1.0,
    )
    observed = harness.observe_running_server_process(
        config,
        server,
        proc_root=tmp_path / "proc",
        affinity_reader=lambda _process_id: {0, 1, 2, 3},
    )
    assert observed.cpu_affinity == config.host.cpu_cores
    assert observed.memory_nodes == (0, 1)

    with pytest.raises(harness.Glm47ServingHarnessError, match="placement"):
        harness.observe_running_server_process(
            config,
            server,
            proc_root=tmp_path / "proc",
            affinity_reader=lambda _process_id: {0, 1, 2},
        )
    (process_root / "numa_maps").write_text(
        "00400000 bind:0-1 file=/runtime\n00600000 bind:1 heap\n",
        encoding="ascii",
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="placement"):
        harness.observe_running_server_process(
            config,
            server,
            proc_root=tmp_path / "proc",
            affinity_reader=lambda _process_id: {0, 1, 2, 3},
        )
    (process_root / "numa_maps").write_text(
        "00400000 default file=/runtime\n", encoding="ascii"
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="placement"):
        harness.observe_running_server_process(
            config,
            server,
            proc_root=tmp_path / "proc",
            affinity_reader=lambda _process_id: {0, 1, 2, 3},
        )


def test_already_exited_zero_server_cannot_claim_sigterm_delivery(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    class ExitedProcess:
        pid = 4242
        returncode: int | None = 0

        def poll(self) -> int:
            return 0

    identity = serving_measurement(config, tmp_path / "cgroup").server_process
    server = harness.ServerProcess(
        process=cast(subprocess.Popen[bytes], cast(object, ExitedProcess())),
        owned=validation.OwnedProcess(
            host_name=config.host.hostname,
            pid=4242,
            process_group_id=4242,
            start_time_ticks=identity.proc_start_time_ticks,
            transport_pid=4242,
            namespace=config.namespace,
            owner_token="owner-token",
            log_path="/tmp/server.log",
        ),
        command=("/usr/bin/numactl",),
        environment={},
        working_directory=str(tmp_path),
        launch_argv_sha256=identity.argv_sha256,
        launch_seconds=1.0,
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="exited early"):
        harness.terminate_server_unforced(config, server, "owner-token", identity)


@pytest.mark.parametrize(
    ("return_code", "accepted"), ((-9, True), (-6, False), (137, False))
)
def test_server_shutdown_accepts_only_canonical_codes_after_verified_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    return_code: int,
    accepted: bool,
) -> None:
    config = make_config(tmp_path)
    identity = serving_measurement(config, tmp_path / "cgroup").server_process

    class ShutdownProcess:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                raise subprocess.TimeoutExpired("server", 0)
            return self.returncode

    process = ShutdownProcess()
    server = harness.ServerProcess(
        process=cast(subprocess.Popen[bytes], cast(object, process)),
        owned=validation.OwnedProcess(
            host_name=config.host.hostname,
            pid=process.pid,
            process_group_id=process.pid,
            start_time_ticks=identity.proc_start_time_ticks,
            transport_pid=process.pid,
            namespace=config.namespace,
            owner_token="owner-token",
            log_path="/tmp/server.log",
        ),
        command=("/usr/bin/numactl",),
        environment={},
        working_directory=str(tmp_path),
        launch_argv_sha256=identity.argv_sha256,
        launch_seconds=1.0,
    )
    delivered_signals: list[tuple[int, int]] = []

    def observe_server(
        _config: harness.ServingBenchmarkConfig,
        _server: harness.ServerProcess,
    ) -> SglangKtServingOwnedServerProcessIdentity:
        return identity

    def group_ownership(_process_group_id: int, _owner_token: str) -> str:
        return "owned" if process.returncode is None else "absent"

    def send_signal(process_group_id: int, signal_number: int) -> None:
        delivered_signals.append((process_group_id, signal_number))
        process.returncode = return_code

    monkeypatch.setattr(harness, "observe_running_server_process", observe_server)
    monkeypatch.setattr(validation, "live_group_ownership", group_ownership)
    monkeypatch.setattr(validation, "owned_token_processes", lambda _token: {})
    monkeypatch.setattr(validation, "reap_adopted_children", lambda: None)
    monkeypatch.setattr(harness.os, "killpg", send_signal)

    if accepted:
        assert (
            harness.terminate_server_unforced(config, server, "owner-token", identity)
            == return_code
        )
    else:
        with pytest.raises(harness.Glm47ServingHarnessError, match="noncanonical code"):
            harness.terminate_server_unforced(config, server, "owner-token", identity)
    assert delivered_signals == [(process.pid, signal.SIGTERM)]


def test_finalization_lock_fails_closed_when_another_owner_wins(tmp_path: Path) -> None:
    result_path = tmp_path / "results"
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(harness.Glm47ServingHarnessError, match="won"):
            harness.acquire_finalization_lock(handoff)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        handoff.close()
    assert not (result_path / harness.PERFORMANCE_RECEIPT_FILENAME).exists()


def test_preserved_lock_rejects_path_replacement(tmp_path: Path) -> None:
    result_path = tmp_path / "results"
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    original_lock = tmp_path / "original.lock"
    lock_path.rename(original_lock)
    lock_path.touch()
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="identity changed"):
            harness.acquire_finalization_lock(handoff)
    finally:
        handoff.close()
    assert not (result_path / harness.PERFORMANCE_RECEIPT_FILENAME).exists()


def test_preserved_result_directory_rejects_path_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    measurement = serving_measurement(config, tmp_path / "cgroup", handoff.identity)
    handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(dict[str, object], cast(object, measurement.model_dump(mode="json"))),
        replace=False,
    )
    displaced = tmp_path / "displaced-results"
    result_path.rename(displaced)
    result_path.mkdir()
    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    mock_successful_finalization_proofs(monkeypatch)
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="identity changed"):
            harness.finalize_serving_benchmark(
                config,
                deployment,
                handoff=handoff,
                lease_path=tmp_path / "absent-lease.json",
                wrapper_return_code=0,
            )
    finally:
        handoff.close()
    assert not (result_path / harness.PERFORMANCE_RECEIPT_FILENAME).exists()
    assert not (displaced / harness.PERFORMANCE_RECEIPT_FILENAME).exists()


def test_port_clear_rejects_a_live_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = cast(tuple[str, int], listener.getsockname())[1]
        with pytest.raises(harness.Glm47ServingHarnessError, match="reachable"):
            harness.require_port_clear("127.0.0.1", port)
    harness.require_port_clear("127.0.0.1", port)


def test_gpu_clear_is_exact_to_selected_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)

    outputs = iter(("", f"{config.host.gpu.uuid}, 1234\n"))

    def bound_tool(
        _binding: validation.ArtifactBinding,
        _arguments: Sequence[str],
        _description: str,
        **_kwargs: object,
    ) -> str:
        return next(outputs)

    monkeypatch.setattr(harness, "run_bound_tool_text", bound_tool)
    harness.require_gpu_clear(config)
    with pytest.raises(harness.Glm47ServingHarnessError, match="1234"):
        harness.require_gpu_clear(config)


def test_new_byte_publication_never_replaces(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    results = validation.ResultDirectory(tmp_path, descriptor)
    os.close(descriptor)
    try:
        harness.write_new_bytes(results, "receipt.json", b"first")
        with pytest.raises(harness.Glm47ServingHarnessError, match="refusing"):
            harness.write_new_bytes(results, "receipt.json", b"second")
        assert (tmp_path / "receipt.json").read_bytes() == b"first"
    finally:
        results.close()


def test_only_outer_finalizer_publishes_single_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    cgroup_path = tmp_path / "removed-cgroup"
    measurement = serving_measurement(config, cgroup_path, handoff.identity)
    handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(
            dict[str, object],
            cast(object, measurement.model_dump(mode="json")),
        ),
        replace=False,
    )
    assert not (result_path / harness.PERFORMANCE_RECEIPT_FILENAME).exists()
    mock_successful_finalization_proofs(monkeypatch)

    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    try:
        receipt = harness.finalize_serving_benchmark(
            config,
            deployment,
            handoff=handoff,
            lease_path=tmp_path / "absent-lease.json",
            wrapper_return_code=0,
        )
    finally:
        handoff.close()
    receipt_path = result_path / harness.PERFORMANCE_RECEIPT_FILENAME
    assert receipt.performance_comparable
    assert receipt_path.exists()
    loaded = load_sglang_kt_warm_serving_run_receipt(
        receipt_path,
        expected_identity_sha256=measurement.identity_sha256,
    )
    assert loaded.receipt.evidence_class == "performance"
    assert {path.name for path in result_path.glob("warm-serving-*")} == {
        harness.MEASUREMENT_FILENAME,
        harness.PERFORMANCE_RECEIPT_FILENAME,
    }


def test_nonzero_wrapper_return_cannot_publish_passed_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    measurement = serving_measurement(config, tmp_path / "cgroup", handoff.identity)
    handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(dict[str, object], cast(object, measurement.model_dump(mode="json"))),
        replace=False,
    )
    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    mock_successful_finalization_proofs(monkeypatch)
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="returned 75"):
            harness.finalize_serving_benchmark(
                config,
                deployment,
                handoff=handoff,
                lease_path=tmp_path / "absent-lease.json",
                wrapper_return_code=75,
            )
    finally:
        handoff.close()
    terminal = cast(
        dict[str, object],
        json.loads((result_path / harness.PERFORMANCE_RECEIPT_FILENAME).read_bytes()),
    )
    assert terminal["status"] == "failed_closed"
    assert terminal["performance_comparable"] is False
    assert terminal["retry_permitted"] is False


@pytest.mark.parametrize("crash_after_commit", [False, True])
def test_terminal_publication_crash_points_remain_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    crash_after_commit: bool,
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    measurement = serving_measurement(config, tmp_path / "cgroup", handoff.identity)
    handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(dict[str, object], cast(object, measurement.model_dump(mode="json"))),
        replace=False,
    )
    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    mock_successful_finalization_proofs(monkeypatch)
    original_writer = harness.write_new_bytes

    def crashing_writer(
        results: validation.ResultDirectory, name: str, contents: bytes
    ) -> None:
        if crash_after_commit:
            original_writer(results, name, contents)
        raise SystemExit("simulated process crash")

    monkeypatch.setattr(harness, "write_new_bytes", crashing_writer)
    try:
        with pytest.raises(SystemExit, match="simulated"):
            harness.finalize_serving_benchmark(
                config,
                deployment,
                handoff=handoff,
                lease_path=tmp_path / "absent-lease.json",
                wrapper_return_code=0,
            )
    finally:
        handoff.close()
    terminal = result_path / harness.PERFORMANCE_RECEIPT_FILENAME
    assert terminal.exists() is crash_after_commit
    if crash_after_commit:
        loaded = load_sglang_kt_warm_serving_run_receipt(
            terminal,
            expected_identity_sha256=measurement.identity_sha256,
        )
        assert loaded.receipt.performance_comparable


def test_concurrent_finalizers_commit_exactly_one_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    initial_handoff = open_handoff(result_path, lock_path)
    measurement = serving_measurement(
        config, tmp_path / "cgroup", initial_handoff.identity
    )
    initial_handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(dict[str, object], cast(object, measurement.model_dump(mode="json"))),
        replace=False,
    )
    initial_handoff.close()
    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    mock_successful_finalization_proofs(monkeypatch)
    start_read, start_write = os.pipe()
    children: list[int] = []
    for _index in range(2):
        child = os.fork()
        if child == 0:
            os.close(start_write)
            os.read(start_read, 1)
            child_handoff = open_handoff(result_path, lock_path)
            try:
                harness.finalize_serving_benchmark(
                    config,
                    deployment,
                    handoff=child_handoff,
                    lease_path=tmp_path / "absent-lease.json",
                    wrapper_return_code=0,
                )
            except harness.Glm47ServingHarnessError as error:
                expected = "won" in str(error) or "terminal" in str(error)
                exit_code = 10 if expected else 20
            else:
                exit_code = 0
            finally:
                child_handoff.close()
            os._exit(exit_code)
        children.append(child)
    os.close(start_read)
    os.write(start_write, b"12")
    os.close(start_write)
    exit_codes: list[int] = []
    for child in children:
        _process_id, status = os.waitpid(child, 0)
        exit_codes.append(os.waitstatus_to_exitcode(status))
    assert sorted(exit_codes) == [0, 10]
    terminal = result_path / harness.PERFORMANCE_RECEIPT_FILENAME
    loaded = load_sglang_kt_warm_serving_run_receipt(
        terminal,
        expected_identity_sha256=measurement.identity_sha256,
    )
    assert loaded.receipt.performance_comparable


def test_failed_finalization_commits_one_irreversible_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = local_result_config(tmp_path)
    result_path = Path(config.result_directory)
    result_path.mkdir()
    lock_path = tmp_path / "benchmark.lock"
    lock_path.touch()
    handoff = open_handoff(result_path, lock_path)
    measurement = serving_measurement(config, tmp_path / "cgroup", handoff.identity)
    handoff.results.write_json(
        harness.MEASUREMENT_FILENAME,
        cast(dict[str, object], cast(object, measurement.model_dump(mode="json"))),
        replace=False,
    )
    deployment = validation.DeploymentIdentity(
        root=str(tmp_path / "deployment"),
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )

    def reject_manifest(
        _config: harness.ServingBenchmarkConfig,
        _deployment: validation.DeploymentIdentity,
        _results: validation.ResultDirectory,
        _snapshot: harness.BoundMeasurementSnapshot,
    ) -> harness.JsonObject:
        raise harness.Glm47ServingHarnessError("adversarial manifest")

    monkeypatch.setattr(harness, "_validate_wrapper_manifest", reject_manifest)
    terminal = result_path / harness.PERFORMANCE_RECEIPT_FILENAME
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="adversarial"):
            harness.finalize_serving_benchmark(
                config,
                deployment,
                handoff=handoff,
                lease_path=tmp_path / "absent-lease.json",
                wrapper_return_code=0,
            )
        first_contents = terminal.read_bytes()
        assert json.loads(first_contents)["status"] == "failed_closed"
        with pytest.raises(harness.Glm47ServingHarnessError, match="terminal"):
            harness.finalize_serving_benchmark(
                config,
                deployment,
                handoff=handoff,
                lease_path=tmp_path / "absent-lease.json",
                wrapper_return_code=0,
            )
        assert terminal.read_bytes() == first_contents
    finally:
        handoff.close()


def test_execute_rejects_command_outside_immutable_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    deployment = validation.DeploymentIdentity(
        root=config.source.deployment_root,
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    expected = ("/usr/bin/systemd-run", "--", "/bin/true")
    monkeypatch.setattr(
        harness, "_load_immutable_context", lambda _path: (config, deployment)
    )
    monkeypatch.setattr(
        harness, "_prepared_systemd_argv", lambda _config, _deployment: expected
    )
    arguments = [
        "--config",
        "/immutable/orchestrator/run-config.json",
        "--lease-path",
        config.lease_execution.lease_path,
        "--lock-path",
        config.lease_execution.lock_path,
        "--",
        *expected,
        "--foreign-argument",
    ]
    with pytest.raises(harness.Glm47ServingHarnessError, match="exact prepared"):
        harness.execute_main(arguments)


@pytest.mark.parametrize("tool_name", ["numactl", "nvidia-smi", "systemctl"])
def test_bound_tool_executes_retained_inode_across_path_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_name: str
) -> None:
    executable = tmp_path / tool_name
    original_contents = b"#!/bin/sh\nprintf 'original\\n'\n"
    executable.write_bytes(original_contents)
    executable.chmod(0o755)
    replacement = tmp_path / f"{tool_name}.replacement"
    replacement.write_bytes(b"#!/bin/sh\nprintf 'replacement\\n'\n")
    replacement.chmod(0o755)
    binding = validation.ArtifactBinding(
        path=str(executable), sha256=hashlib.sha256(original_contents).hexdigest()
    )
    actual_run = subprocess.run

    def replace_then_run(
        arguments: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        os.replace(replacement, executable)
        return actual_run(arguments, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(harness.subprocess, "run", replace_then_run)
    assert harness.run_bound_tool_text(binding, (), tool_name) == "original\n"
    assert executable.read_bytes().endswith(b"replacement\\n'\n")


def test_model_path_on_nfs_is_never_local_performance_evidence(
    tmp_path: Path,
) -> None:
    payload = config_payload(tmp_path)
    payload["model_path"] = "/mnt/sanic/models/glm47"
    coordination = cast(dict[str, object], payload["coordination_guard"])
    filesystem = cast(dict[str, object], coordination["model_filesystem"])
    filesystem["mount_point"] = "/mnt/sanic"
    config = harness.ServingBenchmarkConfig.model_validate_json(json.dumps(payload))
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "41 25 0:42 / /mnt/sanic ro - nfs4 fwuff:/mnt/sanic ro\n",
        encoding="ascii",
    )
    fake_stat = cast(
        os.stat_result,
        cast(object, SimpleNamespace(st_dev=os.makedev(0, 42))),
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="local block"):
        harness.observe_local_model_filesystem(
            config, mountinfo_path=mountinfo, stat_path=lambda _path: fake_stat
        )


def test_local_block_model_provenance_is_exactly_config_bound(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "41 25 259:3 / / rw - xfs /dev/nvme0n1p3 rw\n",
        encoding="ascii",
    )
    fake_stat = cast(
        os.stat_result,
        cast(object, SimpleNamespace(st_dev=os.makedev(259, 3))),
    )
    observed = harness.observe_local_model_filesystem(
        config, mountinfo_path=mountinfo, stat_path=lambda _path: fake_stat
    )
    assert observed.local_block_filesystem is True
    assert (observed.device_major, observed.device_minor) == (259, 3)


def test_nfs_overmount_cannot_hide_behind_matching_block_mount(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "41 25 259:3 / / rw - xfs /dev/nvme0n1p3 rw\n"
        "42 25 0:42 / / ro - nfs4 fwuff:/mnt/sanic ro\n",
        encoding="ascii",
    )
    fake_stat = cast(
        os.stat_result,
        cast(object, SimpleNamespace(st_dev=os.makedev(0, 42))),
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="local block"):
        harness.observe_local_model_filesystem(
            config, mountinfo_path=mountinfo, stat_path=lambda _path: fake_stat
        )


def test_unstable_idle_peer_postflight_cannot_become_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    preflight = cast(
        harness.CoordinationPreflight,
        cast(
            object,
            SimpleNamespace(
                remote=object(),
                local_hca=object(),
                model_filesystem=coordination_evidence(config).model_filesystem,
            ),
        ),
    )
    postflight = object()
    comparison = SimpleNamespace(stable=False, failures=("GPU became busy",))
    monkeypatch.setattr(
        harness.host_guard,
        "collect_remote_snapshot",
        lambda _config, _phase: postflight,
    )
    monkeypatch.setattr(
        harness.host_guard,
        "compare_snapshots",
        lambda _before, _after: comparison,
    )
    with pytest.raises(harness.Glm47ServingHarnessError, match="GPU became busy"):
        harness.collect_coordination_postflight(config, preflight)


def test_coordination_preflight_requires_remote_guard_and_cross_fabric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    filesystem = coordination_evidence(config).model_filesystem
    remote = object()
    local_hca = object()
    fabric_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        harness, "observe_local_model_filesystem", lambda _config: filesystem
    )
    monkeypatch.setattr(
        harness.host_guard,
        "collect_remote_snapshot",
        lambda _config, phase: remote if phase == "preflight" else None,
    )
    monkeypatch.setattr(
        harness.host_guard, "collect_hca_observation", lambda _binding: local_hca
    )
    monkeypatch.setattr(
        harness, "_validate_local_hca_observation", lambda _local, _binding: None
    )
    monkeypatch.setattr(
        harness,
        "_validate_cross_host_fabric",
        lambda _config, local, peer: fabric_calls.append((local, peer)),
    )
    observed = harness.collect_coordination_preflight(config)
    assert observed.remote is remote
    assert observed.local_hca is local_hca
    assert fabric_calls == [(local_hca, remote)]


def test_measurement_snapshot_rejects_path_replacement_after_descriptor_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = local_result_config(tmp_path)
    measurement = serving_measurement(config, tmp_path / "cgroup")
    contents = harness.canonical_sglang_kt_json(measurement.model_dump(mode="json"))
    (tmp_path / harness.MEASUREMENT_FILENAME).write_bytes(contents)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    results = validation.ResultDirectory(tmp_path, descriptor)
    os.close(descriptor)
    original_parser = harness.parse_sglang_kt_strict_json

    def replace_after_parse(value: bytes) -> object:
        parsed = original_parser(value)
        displaced = tmp_path / "original-measurement.json"
        (tmp_path / harness.MEASUREMENT_FILENAME).rename(displaced)
        (tmp_path / harness.MEASUREMENT_FILENAME).write_bytes(b"{}\n")
        return parsed

    monkeypatch.setattr(harness, "parse_sglang_kt_strict_json", replace_after_parse)
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="path changed"):
            harness._load_measurement(results)
    finally:
        results.close()
    assert (tmp_path / "original-measurement.json").read_bytes() == contents
    assert (tmp_path / harness.MEASUREMENT_FILENAME).read_bytes() == b"{}\n"


def _valid_wrapper_manifest(
    config: harness.ServingBenchmarkConfig,
    deployment: validation.DeploymentIdentity,
    snapshot: harness.BoundMeasurementSnapshot,
) -> harness.JsonObject:
    command = harness._serving_child_argv(
        config,
        lease_path=Path(config.lease_execution.lease_path),
        lock_path=Path(config.lease_execution.lock_path),
    )
    immutable_config = Path(deployment.root) / validation.IMMUTABLE_CONFIG_RELATIVE_PATH
    metadata = harness.build_serving_static_metadata(
        config,
        command,
        validation.config_sha256(immutable_config),
        deployment,
    )
    metadata["generated_at"] = "2026-07-19T20:00:00+00:00"
    containment: dict[str, object] = {"scope": "owned"}
    coordination_sha256 = snapshot.measurement.coordination_guard.evidence_sha256
    benchmark_result: dict[str, object] = {
        "status": "completed",
        "run_id": config.run_id,
        "namespace": config.namespace,
        "cleanup_succeeded": True,
        "performance_comparable": False,
        "outer_finalization_required": True,
        "measurement_file": harness.MEASUREMENT_FILENAME,
        "measurement_sha256": snapshot.sha256,
        "coordination_guard_evidence_sha256": coordination_sha256,
        "containment": containment,
    }
    child_manifest = {
        **benchmark_result,
        "config": config.model_dump(mode="json"),
        "deployment": asdict(deployment),
    }
    return cast(
        harness.JsonObject,
        cast(
            object,
            {
                "manifest_writer": "benchmark_lease.py",
                "lease_id": "a" * 32,
                "owner": config.lease_execution.owner,
                "purpose": config.lease_execution.purpose,
                "run_id": config.run_id,
                "exo_namespace": config.namespace,
                "ports": list(config.reserved_ports),
                "result_directory": config.result_directory,
                "command": list(command),
                "cleanup_grace_seconds": (config.lease_execution.cleanup_grace_seconds),
                "child_cleanup_confirmation_required": True,
                "metadata": metadata,
                "status": "completed",
                "return_code": 0,
                "command_return_code": 0,
                "cleanup_succeeded": True,
                "cleanup_forced": False,
                "benchmark_result": benchmark_result,
                "child_manifest": child_manifest,
                "runtime_metadata": {
                    "owner_token": snapshot.measurement.owner_token,
                    "run_id": config.run_id,
                    "namespace": config.namespace,
                    "containment": containment,
                },
            },
        ),
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("command",), ["/bin/false"]),
        (("lease_id",), "b" * 32),
        (("run_id",), "foreign-run"),
        (("exo_namespace",), "foreign-namespace"),
        (("ports",), [1, 2]),
        (("owner",), "foreign-owner"),
        (("cleanup_grace_seconds",), 1.0),
        (("metadata", "command"), ["/bin/false"]),
        (("runtime_metadata", "containment"), {"scope": "foreign"}),
    ],
)
def test_wrapper_manifest_rejects_authorization_and_containment_substitution(
    tmp_path: Path, path: tuple[str, ...], replacement: object
) -> None:
    config = local_result_config(tmp_path)
    immutable_config = (
        Path(config.source.deployment_root) / validation.IMMUTABLE_CONFIG_RELATIVE_PATH
    )
    immutable_config.parent.mkdir(parents=True)
    immutable_config.write_bytes(
        harness.canonical_sglang_kt_json(config.model_dump(mode="json"))
    )
    deployment = validation.DeploymentIdentity(
        root=config.source.deployment_root,
        orchestrator_sha256="1" * 64,
        validator_sha256="2" * 64,
        validator_files=(),
        source=validation.SourceIdentity("3" * 40, {}),
    )
    measurement = serving_measurement(config, tmp_path / "cgroup")
    contents = harness.canonical_sglang_kt_json(measurement.model_dump(mode="json"))
    snapshot = harness.BoundMeasurementSnapshot(
        measurement=measurement,
        contents=contents,
        sha256=hashlib.sha256(contents).hexdigest(),
        device=1,
        inode=2,
    )
    manifest = _valid_wrapper_manifest(config, deployment, snapshot)
    mutated = copy.deepcopy(manifest)
    target: dict[str, object] = cast(dict[str, object], mutated)
    for component in path[:-1]:
        target = cast(dict[str, object], target[component])
    target[path[-1]] = replacement
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    results = validation.ResultDirectory(tmp_path, descriptor)
    os.close(descriptor)
    results.write_json(validation.CHILD_MANIFEST_FILENAME, mutated, replace=False)
    try:
        with pytest.raises(harness.Glm47ServingHarnessError, match="transaction"):
            harness._validate_wrapper_manifest(config, deployment, results, snapshot)
    finally:
        results.close()
