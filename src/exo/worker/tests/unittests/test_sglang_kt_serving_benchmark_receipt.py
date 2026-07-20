import copy
import hashlib
from pathlib import Path

import pytest

from exo.shared.types.common import ModelId, NodeId
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.receipt_io import canonical_sglang_kt_json
from exo.worker.sglang_kt.serving_benchmark_receipt import (
    GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256,
    SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
    SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH,
    ServingWorkloadKind,
    SglangKtServingAdmissionBinding,
    SglangKtServingCleanupEvidence,
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
    SglangKtServingWorkloadRequest,
    SglangKtWarmServingRunIdentity,
    SglangKtWarmServingRunReceiptError,
    WarmServingRunReceiptV3,
    calculate_sglang_kt_length_finish_reason_sha256,
    calculate_sglang_kt_serving_coordination_guard_evidence_sha256,
    calculate_sglang_kt_serving_source_bundle_sha256,
    calculate_sglang_kt_token_ids_sha256,
    calculate_sglang_kt_warm_serving_run_identity_sha256,
    canonicalize_sglang_kt_warm_serving_run_receipt,
    load_sglang_kt_warm_serving_run_receipt,
)

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type JsonObject = dict[str, JsonValue]

GPU_UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
MODEL_PATH = "/mnt/models/glm47"


def _file(path: str, marker: str) -> SglangKtServingFileIdentity:
    return SglangKtServingFileIdentity(
        path=path,
        size_bytes=123,
        sha256=marker * 64,
    )


def _identity() -> SglangKtWarmServingRunIdentity:
    source_files = (
        SglangKtServingSourceFileIdentity(
            relative_path=SGLANG_KT_SERVING_CLIENT_RELATIVE_PATH,
            size_bytes=111,
            sha256="e" * 64,
        ),
        SglangKtServingSourceFileIdentity(
            relative_path="src/exo/__init__.py",
            size_bytes=0,
            sha256="0" * 64,
        ),
        SglangKtServingSourceFileIdentity(
            relative_path=SGLANG_KT_SERVING_RECEIPT_RELATIVE_PATH,
            size_bytes=222,
            sha256="f" * 64,
        ),
    )
    source_bundle_sha256 = calculate_sglang_kt_serving_source_bundle_sha256(
        source_files
    )
    return SglangKtWarmServingRunIdentity(
        admission=SglangKtServingAdmissionBinding(
            model_runtime_validation_receipt=_file("/receipts/model.json", "1"),
            kernel_runtime_validation_receipt=_file("/receipts/kernel.json", "2"),
            model_contract_receipt=_file("/receipts/contract.json", "3"),
        ),
        runtime=SglangKtServingRuntimeIdentity(
            executable="/runtime/bin/python",
            runtime_build_receipt=_file("/receipts/build.json", "4"),
            numactl_executable=_file("/usr/bin/numactl", "5"),
            nvidia_smi_executable=_file("/usr/bin/nvidia-smi", "6"),
            systemctl_executable=_file("/usr/bin/systemctl", "7"),
            runtime_build_id="8" * 64,
            python_version="3.12.11",
            torch_version="2.9.1+cu128",
            cuda_version="12.8",
            sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
            ktransformers_revision=GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
            sgl_kernel_build_id="a" * 64,
            deep_gemm_build_id="b" * 64,
            kt_kernel_build_id="c" * 64,
        ),
        model=SglangKtServingModelIdentity(
            model_id=ModelId("zai-org/GLM-4.7-Flash"),
            model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
            model_path=MODEL_PATH,
            model_config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
            model_index_sha256="9" * 64,
            physical_weight_bytes=62_444_175_504,
        ),
        process_spec=SglangKtServingProcessSpecIdentity(
            receipt=_file("/spec/process.json", "a"),
            process_spec_sha256="b" * 64,
            launch_argv_sha256="c" * 64,
            launch_environment_sha256="d" * 64,
            target_profile=GLM_4_7_FLASH_SERVING_BASELINE_TARGET_PROFILE,
            resident_gpu_experts=4,
            cpu_cores=(0, 1, 2, 3),
            memory_nodes=(0,),
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
            source_bundle_sha256=source_bundle_sha256,
            protocol_version=1,
            http_library="httpx",
            http_library_version="0.28.1",
            request_timeout_seconds=900.0,
        ),
        source=SglangKtServingSourceIdentity(
            repository_root="/source",
            commit="1" * 40,
            source_bundle_sha256=source_bundle_sha256,
            files=source_files,
            dirty_files=(),
        ),
        topology=SglangKtServingTopologyIdentity(
            deployment="local",
            interconnect="none",
            stages=(
                SglangKtServingTopologyStage(
                    pipeline_rank=0,
                    node_id=NodeId("dwagon"),
                    host="127.0.0.1",
                    port=62075,
                    gpu_uuid=GPU_UUID,
                    hca_devices=(),
                ),
            ),
        ),
        server_info=(
            SglangKtServingServerInfoIdentity(
                node_id=NodeId("dwagon"),
                host="127.0.0.1",
                port=62075,
                canonical_response_sha256="1" * 64,
                version="0.0.0.dev0",
                model_path=MODEL_PATH,
                tp_size=1,
                pp_size=1,
                nnodes=1,
                node_rank=0,
                disable_radix_cache=True,
            ),
        ),
    )


def _request(kind: ServingWorkloadKind) -> SglangKtServingWorkloadRequest:
    token_count, output_count = (1_024, 32) if kind == "prefill" else (128, 128)
    input_ids_sha256 = (
        GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256
        if kind == "prefill"
        else GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256
    )
    return SglangKtServingWorkloadRequest(
        kind=kind,
        input_token_count=token_count,
        input_ids_sha256=input_ids_sha256,
        max_new_tokens=output_count,
        sampling_seed=20_260_719,
        temperature=0.0,
        ignore_eos=True,
        stream=True,
        return_logprob=False,
        log_metrics=True,
    )


def _invocation(
    request: SglangKtServingWorkloadRequest,
    ordinal: int,
) -> SglangKtServingInvocationEvidence:
    generation_window_seconds = 2.0
    return SglangKtServingInvocationEvidence(
        ordinal=ordinal,
        input_ids_sha256=request.input_ids_sha256,
        cache_flush_status_code=200,
        cache_flush_response_sha256="2" * 64,
        prompt_tokens=request.input_token_count,
        completion_tokens=request.max_new_tokens,
        cached_tokens=0,
        output_ids_sha256=("3" if request.kind == "prefill" else "4") * 64,
        finish_reason_sha256=calculate_sglang_kt_length_finish_reason_sha256(
            request.max_new_tokens
        ),
        stream_line_count=request.max_new_tokens * 2 + 2,
        stream_event_count=request.max_new_tokens,
        output_bearing_event_count=request.max_new_tokens,
        maximum_stream_line_bytes=512,
        first_stream_event_output_tokens=1,
        total_client_seconds=3.0,
        client_observed_ttft_seconds=1.0,
        client_observed_generation_window_seconds=generation_window_seconds,
        client_observed_decode_tokens_per_second=(request.max_new_tokens - 1)
        / generation_window_seconds,
        ttft_semantics=("client_stream_first_output_event_including_http_and_queue_v1"),
    )


def _workload(kind: ServingWorkloadKind) -> SglangKtServingWorkloadEvidence:
    request = _request(kind)
    return SglangKtServingWorkloadEvidence(
        request=request,
        warmups=tuple(_invocation(request, index) for index in range(1, 3)),
        samples=tuple(_invocation(request, index) for index in range(1, 4)),
    )


def _sanity() -> SglangKtServingSanityEvidence:
    output_ids = (3257, 46, 62674, 3333, 8374)
    return SglangKtServingSanityEvidence(
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
        completion_tokens=len(output_ids),
        output_ids=output_ids,
        output_ids_sha256=calculate_sglang_kt_token_ids_sha256(output_ids),
        server_output_text="EXO_SANITY_OK",
        locally_decoded_output_text="EXO_SANITY_OK",
        finish_reason_type="stop",
        finish_reason_sha256="5" * 64,
        total_client_seconds=1.0,
        post_sanity_cache_flush_status_code=200,
        post_sanity_cache_flush_response_sha256="6" * 64,
    )


def _coordination_guard() -> SglangKtServingCoordinationGuardEvidence:
    config_sha256 = "8" * 64
    binding_sha256 = "9" * 64
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
    local_preflight: dict[str, object] = {"device": "mlx5_0", "marker": 1}
    local_postflight: dict[str, object] = {"device": "mlx5_0", "marker": 2}
    filesystem = SglangKtServingModelFilesystemEvidence(
        model_path=MODEL_PATH,
        mount_point="/mnt",
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


def _receipt() -> WarmServingRunReceiptV3:
    identity = _identity()
    return WarmServingRunReceiptV3(
        schema_version=3,
        status="passed",
        generated_at_utc="2026-07-19T20:31:00+00:00",
        evidence_class="performance",
        performance_comparable=True,
        profiler="none",
        instrumentation="none",
        radix_cache_disabled=True,
        max_concurrent_requests=1,
        measurement_sha256="d" * 64,
        identity_sha256=calculate_sglang_kt_warm_serving_run_identity_sha256(identity),
        identity=identity,
        setup=SglangKtServingSetupEvidence(
            process_launch_seconds=180.0,
            health_ready_seconds=2.0,
            admission_seconds=1.0,
            server_info_fetch_seconds=0.01,
            health_generate_status_code=200,
            health_generate_response_sha256="6" * 64,
        ),
        sanity=_sanity(),
        jit_cache=SglangKtServingJitCacheEvidence(
            cache_directories=("/cache/triton",),
            after_penultimate_warmup_manifest_sha256="7" * 64,
            after_final_warmup_manifest_sha256="7" * 64,
            after_measurement_manifest_sha256="7" * 64,
        ),
        workloads=(_workload("prefill"), _workload("decode")),
        coordination_guard=_coordination_guard(),
        cleanup=SglangKtServingCleanupEvidence(
            lease_id="a" * 32,
            benchmark_completed_normally=True,
            server_process=SglangKtServingOwnedServerProcessIdentity(
                pid=12_345,
                proc_start_time_ticks=987_654,
                executable=identity.runtime.executable,
                argv_sha256="c" * 64,
                cpu_affinity=identity.process_spec.cpu_cores,
                memory_nodes=identity.process_spec.memory_nodes,
            ),
            termination_signal="SIGTERM",
            server_return_code=-15,
            forced=False,
            owned_processes_absent=True,
            service_host="127.0.0.1",
            service_port=62075,
            service_port_clear=True,
            gpu_uuid=GPU_UUID,
            gpu_process_clear=True,
            delegated_cgroup_path="/sys/fs/cgroup/system.slice/exo-serving.scope",
            delegated_cgroup_removed=True,
            transient_unit_name="exo-serving.scope",
            transient_unit_removed=True,
            cleanup_completed_at_utc="2026-07-19T20:30:00+00:00",
            lease_cleanup_scope="outer_benchmark_wrapper",
        ),
    )


def _payload() -> JsonObject:
    return _receipt().model_dump(mode="json")


def _object(parent: JsonObject, key: str) -> JsonObject:
    value = parent[key]
    assert isinstance(value, dict)
    return value


def _array(parent: JsonObject, key: str) -> list[JsonValue]:
    value = parent[key]
    assert isinstance(value, list)
    return value


def test_canonical_receipt_round_trips_and_loads(tmp_path: Path) -> None:
    payload = _payload()
    contents = canonicalize_sglang_kt_warm_serving_run_receipt(payload)
    assert contents == canonical_sglang_kt_json(payload)

    path = tmp_path / "serving.json"
    path.write_bytes(contents)
    receipt_sha256 = hashlib.sha256(contents).hexdigest()
    observation = load_sglang_kt_warm_serving_run_receipt(
        path,
        expected_identity_sha256=_receipt().identity_sha256,
        expected_receipt_sha256=receipt_sha256,
    )
    assert observation.receipt.performance_comparable
    assert observation.receipt_sha256 == receipt_sha256


def test_canonical_receipt_rejects_v1_payload_with_sanity() -> None:
    payload = _payload()
    payload["schema_version"] = 1
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_canonical_receipt_rejects_v2_payload() -> None:
    payload = _payload()
    payload["schema_version"] = 2
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_canonical_receipt_rejects_v3_payload_without_sanity() -> None:
    payload = _payload()
    del payload["sanity"]
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    "invalid_shape", ["v1_with_sanity", "v2_with_sanity", "v3_without_sanity"]
)
def test_receipt_loader_rejects_non_v3_shape(
    tmp_path: Path, invalid_shape: str
) -> None:
    payload = _payload()
    if invalid_shape == "v1_with_sanity":
        payload["schema_version"] = 1
    elif invalid_shape == "v2_with_sanity":
        payload["schema_version"] = 2
    else:
        del payload["sanity"]
    path = tmp_path / "invalid-serving.json"
    path.write_bytes(canonical_sglang_kt_json(payload))
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        load_sglang_kt_warm_serving_run_receipt(
            path,
            expected_identity_sha256=_receipt().identity_sha256,
        )


def test_receipt_rejects_debug_instrumentation_as_performance() -> None:
    payload = _payload()
    payload["instrumentation"] = "debug_timing"
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_performance_receipt_requires_coordination_guard_evidence() -> None:
    payload = _payload()
    payload["coordination_guard"] = None
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("comparison", "stable", False),
        ("comparison", "failures", ["peer changed"]),
        ("remote_postflight", "binding_sha256", "f" * 64),
        ("remote_postflight", "config_sha256", "f" * 64),
        ("coordination_guard", "evidence_sha256", "f" * 64),
    ],
)
def test_receipt_rejects_changed_coordination_guard_evidence(
    section: str, field: str, value: JsonValue
) -> None:
    payload = _payload()
    guard = _object(payload, "coordination_guard")
    target = guard if section == "coordination_guard" else _object(guard, section)
    target[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_changed_output_ids() -> None:
    payload = _payload()
    workloads = _array(payload, "workloads")
    prefill = workloads[0]
    assert isinstance(prefill, dict)
    samples = _array(prefill, "samples")
    sample = samples[-1]
    assert isinstance(sample, dict)
    sample["output_ids_sha256"] = "f" * 64
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    "field",
    ["server_output_text", "locally_decoded_output_text"],
)
def test_receipt_rejects_incoherent_sanity_marker(field: str) -> None:
    payload = _payload()
    _object(payload, "sanity")[field] = "WRONG"
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_noncanonical_finish_reason_hash() -> None:
    payload = _payload()
    workloads = _array(payload, "workloads")
    prefill = workloads[0]
    assert isinstance(prefill, dict)
    samples = _array(prefill, "samples")
    sample = samples[-1]
    assert isinstance(sample, dict)
    sample["finish_reason_sha256"] = "5" * 64
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_cache_hits_and_too_few_samples() -> None:
    cached_payload = _payload()
    workloads = _array(cached_payload, "workloads")
    prefill = workloads[0]
    assert isinstance(prefill, dict)
    samples = _array(prefill, "samples")
    first_sample = samples[0]
    assert isinstance(first_sample, dict)
    first_sample["cached_tokens"] = 1
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(cached_payload)

    short_payload = _payload()
    short_workloads = _array(short_payload, "workloads")
    short_prefill = short_workloads[0]
    assert isinstance(short_prefill, dict)
    short_prefill["samples"] = copy.deepcopy(_array(short_prefill, "samples")[:2])
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(short_payload)


def test_receipt_rejects_unstable_jit_cache_and_identity_hash() -> None:
    jit_payload = _payload()
    _object(jit_payload, "jit_cache")["after_measurement_manifest_sha256"] = "8" * 64
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(jit_payload)

    identity_payload = _payload()
    process_spec = _object(_object(identity_payload, "identity"), "process_spec")
    process_spec["resident_gpu_experts"] = 1
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(identity_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_profile", "glm47_flash_bf16_sm86_smoke_v1"),
        ("resident_gpu_experts", 0),
    ],
)
def test_receipt_rejects_nonbaseline_process_identity(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    process_spec = _object(_object(payload, "identity"), "process_spec")
    process_spec[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_tuning_until_a_tuned_profile_is_pinned() -> None:
    payload = _payload()
    tuning = _object(_object(payload, "identity"), "tuning")
    tuning.update(
        {
            "mode": "tuned",
            "config_directory": "/configs/triton",
            "manifest_sha256": "8" * 64,
            "config_files": [
                {
                    "relative_path": "configs/E=4,N=1536.json",
                    "size_bytes": 123,
                    "sha256": "9" * 64,
                }
            ],
        }
    )
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_duplicate_topology_gpu() -> None:
    payload = _payload()
    topology = _object(_object(payload, "identity"), "topology")
    stages = _array(topology, "stages")
    duplicate_stage = copy.deepcopy(stages[0])
    assert isinstance(duplicate_stage, dict)
    duplicate_stage["pipeline_rank"] = 1
    duplicate_stage["port"] = 62076
    stages.append(duplicate_stage)
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_two_host_topology_in_schema_v1() -> None:
    payload = _payload()
    topology = _object(_object(payload, "identity"), "topology")
    topology["deployment"] = "two_host"
    topology["interconnect"] = "infiniband"
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_ids_sha256", "0" * 64),
        ("sampling_seed", 1),
        ("return_logprob", True),
        ("log_metrics", False),
    ],
)
def test_receipt_rejects_noncanonical_request_fields(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    prefill = _array(payload, "workloads")[0]
    assert isinstance(prefill, dict)
    _object(prefill, "request")[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "0.6.3.post1"),
        ("tp_size", 2),
        ("pp_size", 2),
        ("nnodes", 2),
        ("node_rank", 1),
        ("disable_radix_cache", False),
    ],
)
def test_receipt_rejects_unpinned_server_info_fields(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    identity = _object(payload, "identity")
    server_info = _array(identity, "server_info")[0]
    assert isinstance(server_info, dict)
    server_info[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_bundle_sha256", "0" * 64),
        ("source_file.path", "/source/scripts/not-the-client.py"),
        ("source_file.sha256", "0" * 64),
    ],
)
def test_receipt_rejects_client_source_bundle_mismatch(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    client = _object(_object(payload, "identity"), "client")
    if field.startswith("source_file."):
        _object(client, "source_file")[field.removeprefix("source_file.")] = value
    else:
        client[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_source_bundle_without_required_client_member() -> None:
    payload = _payload()
    source = _object(_object(payload, "identity"), "source")
    files = _array(source, "files")
    source["files"] = files[1:]
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_rejects_ttft_plus_generation_window_over_total() -> None:
    payload = _payload()
    prefill = _array(payload, "workloads")[0]
    assert isinstance(prefill, dict)
    sample = _array(prefill, "samples")[0]
    assert isinstance(sample, dict)
    sample["total_client_seconds"] = 2.9
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_bearing_event_count", 1),
        ("stream_event_count", 41),
        ("stream_line_count", 137),
        ("maximum_stream_line_bytes", 262_145),
    ],
)
def test_receipt_rejects_unbounded_stream_evidence(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    prefill = _array(payload, "workloads")[0]
    assert isinstance(prefill, dict)
    sample = _array(prefill, "samples")[0]
    assert isinstance(sample, dict)
    sample[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_performance_receipt_requires_cleanup_evidence() -> None:
    missing = _payload()
    del missing["cleanup"]
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(missing)

    incomplete = _payload()
    incomplete["cleanup"] = None
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(incomplete)


def test_performance_receipt_accepts_sglang_self_sigkill_after_sigterm() -> None:
    payload = _payload()
    cleanup = _object(payload, "cleanup")
    cleanup["server_return_code"] = -9

    receipt = WarmServingRunReceiptV3.model_validate_json(
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)
    )

    assert receipt.cleanup is not None
    assert receipt.cleanup.server_return_code == -9
    assert receipt.cleanup.termination_signal == "SIGTERM"
    assert receipt.cleanup.forced is False


def test_receipt_is_published_only_after_cleanup_completes() -> None:
    payload = _payload()
    payload["generated_at_utc"] = "2026-07-19T20:30:00+00:00"
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_completed_normally", False),
        ("termination_signal", "SIGKILL"),
        ("server_return_code", 137),
        ("forced", True),
        ("owned_processes_absent", False),
        ("service_port_clear", False),
        ("gpu_process_clear", False),
        ("delegated_cgroup_removed", False),
        ("transient_unit_removed", False),
        ("lease_cleanup_scope", "inside_server_process"),
    ],
)
def test_performance_receipt_rejects_forced_or_unclear_cleanup(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    _object(payload, "cleanup")[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("server_process.executable", "/runtime/bin/other-python"),
        ("server_process.argv_sha256", "0" * 64),
        ("service_port", 62076),
        ("gpu_uuid", "GPU-ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ],
)
def test_performance_receipt_binds_cleanup_to_owned_server(
    field: str,
    value: JsonValue,
) -> None:
    payload = _payload()
    cleanup = _object(payload, "cleanup")
    if field.startswith("server_process."):
        _object(cleanup, "server_process")[field.removeprefix("server_process.")] = (
            value
        )
    else:
        cleanup[field] = value
    with pytest.raises(SglangKtWarmServingRunReceiptError):
        canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_receipt_accepts_decode_rate_for_multitoken_first_event() -> None:
    payload = _payload()
    workloads = _array(payload, "workloads")
    prefill = workloads[0]
    assert isinstance(prefill, dict)
    sample = _array(prefill, "samples")[0]
    assert isinstance(sample, dict)
    sample["first_stream_event_output_tokens"] = 2
    sample["client_observed_decode_tokens_per_second"] = 15.0
    canonicalize_sglang_kt_warm_serving_run_receipt(payload)


def test_loader_rejects_wrong_independent_identity(tmp_path: Path) -> None:
    contents = canonicalize_sglang_kt_warm_serving_run_receipt(_payload())
    path = tmp_path / "serving.json"
    path.write_bytes(contents)
    with pytest.raises(
        SglangKtWarmServingRunReceiptError,
        match="identity does not match",
    ):
        load_sglang_kt_warm_serving_run_receipt(
            path,
            expected_identity_sha256="f" * 64,
        )
