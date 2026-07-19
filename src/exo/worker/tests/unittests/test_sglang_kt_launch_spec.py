import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import (
    KTransformersMethod,
    SglangKtLaunchPlan,
    SglangKtStageSpec,
)
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
    GLM_4_7_FLASH_TARGET_PROFILE,
    GLM_5_2_FULL_INDEXER_LAYER_STARTS,
    GLM_5_2_TARGET_PROFILE,
    REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION,
    REQUIRED_TRANSFORMERS_MODULE_VERSION,
    REQUIRED_TRANSFORMERS_VERSION,
    SUPPORTED_KTRANSFORMERS_REVISION,
    SUPPORTED_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
    build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs,
    build_glm_4_7_flash_bf16_process_launch_specs,
    build_glm_5_2_fp8_process_launch_specs,
)

MODEL_REVISION = "4" * 40
PYTHON_EXECUTABLE = "/opt/sglang-kt/bin/python"


def make_stage(
    pipeline_rank: int,
    start_layer: int,
    end_layer: int,
    *,
    node_id: str,
    gpu_suffix: int,
    service_ip: str,
    service_port: int,
    cpu_cores: tuple[int, ...],
    memory_node: int,
    hca_devices: tuple[str, ...],
    ktransformers_method: KTransformersMethod = "FP8",
) -> SglangKtStageSpec:
    return SglangKtStageSpec(
        pipeline_rank=pipeline_rank,
        start_layer=start_layer,
        end_layer=end_layer,
        node_id=NodeId(node_id),
        gpu_uuid=f"GPU-00000000-0000-0000-0000-{gpu_suffix:012x}",
        service_endpoint=Host(ip=service_ip, port=service_port),
        model_path="/var/lib/exo/models/glm-5.2-fp8",
        ktransformers_weight_path="/var/lib/exo/models/glm-5.2-fp8",
        cpu_cores=cpu_cores,
        memory_nodes=(memory_node,),
        cpu_infer_threads=len(cpu_cores),
        threadpool_count=1,
        ktransformers_method=ktransformers_method,
        resident_gpu_experts=0,
        max_deferred_experts_per_token=0,
        hca_devices=hca_devices,
    )


def make_plan(
    *,
    model_id: str = "zai-org/GLM-5.2-FP8",
    sglang_revision: str = SUPPORTED_SGLANG_REVISION,
    ktransformers_revision: str = SUPPORTED_KTRANSFORMERS_REVISION,
    ktransformers_method: KTransformersMethod = "FP8",
    hca_devices: tuple[str, ...] = ("mlx4_0:1",),
) -> SglangKtLaunchPlan:
    return SglangKtLaunchPlan(
        model_id=ModelId(model_id),
        model_revision=MODEL_REVISION,
        sglang_revision=sglang_revision,
        ktransformers_revision=ktransformers_revision,
        target_profile=GLM_5_2_TARGET_PROFILE,
        total_layers=78,
        context_length=262_144,
        max_total_tokens=4_096,
        static_memory_fraction=0.8,
        max_concurrent_requests=1,
        distributed_coordinator=Host(ip="192.168.40.248", port=29_500),
        rank_zero_endpoint=Host(ip="192.168.40.248", port=30_000),
        stages=(
            make_stage(
                0,
                0,
                30,
                node_id="dwagon",
                gpu_suffix=1,
                service_ip="192.168.40.248",
                service_port=30_000,
                cpu_cores=tuple(range(30)),
                memory_node=0,
                hca_devices=hca_devices,
                ktransformers_method=ktransformers_method,
            ),
            make_stage(
                1,
                30,
                58,
                node_id="dwagon",
                gpu_suffix=2,
                service_ip="192.168.40.248",
                service_port=30_001,
                cpu_cores=tuple(range(30, 58)),
                memory_node=1,
                hca_devices=("mlx4_0:2",),
                ktransformers_method=ktransformers_method,
            ),
            make_stage(
                2,
                58,
                78,
                node_id="fwuff",
                gpu_suffix=3,
                service_ip="192.168.40.249",
                service_port=30_002,
                cpu_cores=tuple(range(20)),
                memory_node=0,
                hca_devices=("mlx4_0:1",),
                ktransformers_method=ktransformers_method,
            ),
        ),
    )


def make_glm_4_7_flash_bf16_plan(
    *,
    resident_gpu_experts: int = 4,
    ktransformers_method: KTransformersMethod = "BF16",
    hca_devices: tuple[str, ...] = (),
) -> SglangKtLaunchPlan:
    model_path = "/var/lib/exo/models/glm-4.7-flash-bf16"
    return SglangKtLaunchPlan(
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
        distributed_coordinator=Host(ip="192.168.40.248", port=29_510),
        rank_zero_endpoint=Host(ip="192.168.40.248", port=30_100),
        stages=(
            SglangKtStageSpec(
                pipeline_rank=0,
                start_layer=0,
                end_layer=47,
                node_id=NodeId("dwagon"),
                gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
                service_endpoint=Host(ip="192.168.40.248", port=30_100),
                model_path=model_path,
                ktransformers_weight_path=model_path,
                cpu_cores=tuple(range(56)),
                memory_nodes=(0,),
                cpu_infer_threads=56,
                threadpool_count=1,
                ktransformers_method=ktransformers_method,
                resident_gpu_experts=resident_gpu_experts,
                max_deferred_experts_per_token=0,
                hca_devices=hca_devices,
            ),
        ),
    )


def make_glm_4_7_flash_bf16_cpu_routed_experts_plan() -> SglangKtLaunchPlan:
    plan = make_glm_4_7_flash_bf16_plan(resident_gpu_experts=0)
    return plan.model_copy(
        update={
            "target_profile": GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE,
        }
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_builds_three_logical_nodes_for_two_physical_hosts() -> None:
    plan = make_plan()

    specs = build_glm_5_2_fp8_process_launch_specs(plan, PYTHON_EXECUTABLE)

    assert plan.target_profile == GLM_5_2_TARGET_PROFILE
    assert len(specs) == 3
    assert tuple(spec.pipeline_rank for spec in specs) == (0, 1, 2)
    assert tuple(spec.node_id for spec in specs) == (
        NodeId("dwagon"),
        NodeId("dwagon"),
        NodeId("fwuff"),
    )
    assert specs[0].service_endpoint == plan.rank_zero_endpoint
    assert specs[0].service_endpoint != specs[1].service_endpoint
    assert specs[0].command[:3] == (
        PYTHON_EXECUTABLE,
        "-m",
        "sglang.launch_server",
    )

    for expected_rank, spec in enumerate(specs):
        assert argument_value(spec.arguments, "--pp-size") == "3"
        assert argument_value(spec.arguments, "--tp-size") == "1"
        assert argument_value(spec.arguments, "--nnodes") == "3"
        assert argument_value(spec.arguments, "--node-rank") == str(expected_rank)
        assert (
            argument_value(spec.arguments, "--dist-init-addr") == "192.168.40.248:29500"
        )
        assert argument_value(spec.arguments, "--host") == spec.service_endpoint.ip
        assert argument_value(spec.arguments, "--port") == str(
            spec.service_endpoint.port
        )
        assert "--nccl-port" not in spec.arguments


def test_builds_pinned_glm_5_2_ktransformers_arguments() -> None:
    plan = make_plan()

    spec = build_glm_5_2_fp8_process_launch_specs(plan, PYTHON_EXECUTABLE)[1]

    assert argument_value(spec.arguments, "--model-path") == plan.stages[1].model_path
    assert (
        argument_value(spec.arguments, "--kt-weight-path")
        == plan.stages[1].ktransformers_weight_path
    )
    assert argument_value(spec.arguments, "--kt-cpuinfer") == "28"
    assert argument_value(spec.arguments, "--kt-threadpool-count") == "1"
    assert argument_value(spec.arguments, "--kt-numa-nodes") == "1"
    assert argument_value(spec.arguments, "--kt-num-gpu-experts") == "0"
    assert argument_value(spec.arguments, "--kt-method") == "FP8"
    assert argument_value(spec.arguments, "--kt-max-deferred-experts-per-token") == "0"
    assert argument_value(spec.arguments, "--context-length") == "262144"
    assert argument_value(spec.arguments, "--max-total-tokens") == "4096"
    assert argument_value(spec.arguments, "--mem-fraction-static") == "0.8"
    assert argument_value(spec.arguments, "--max-running-requests") == "1"
    assert argument_value(spec.arguments, "--attention-backend") == "nsa"
    assert argument_value(spec.arguments, "--kv-cache-dtype") == "fp8_e4m3"
    assert argument_value(spec.arguments, "--tool-call-parser") == "glm47"
    assert argument_value(spec.arguments, "--reasoning-parser") == "glm45"
    assert "--disable-shared-experts-fusion" in spec.arguments
    assert "--trust-remote-code" in spec.arguments
    assert spec.unset_environment_variables == (
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ALLOC_CONF",
    )
    assert spec.unset_environment_variable_prefixes == ("NCCL_", "SGLANG_")

    assert spec.model_path == plan.stages[1].model_path
    assert spec.ktransformers_weight_path == plan.stages[1].ktransformers_weight_path
    assert spec.hca_devices == plan.stages[1].hca_devices
    assert spec.distributed_coordinator == plan.distributed_coordinator
    assert spec.expected_model_revision == MODEL_REVISION
    assert spec.expected_sglang_revision == SUPPORTED_SGLANG_REVISION
    assert spec.expected_ktransformers_revision == SUPPORTED_KTRANSFORMERS_REVISION
    assert spec.required_transformers_version == REQUIRED_TRANSFORMERS_VERSION
    assert (
        spec.required_transformers_distribution_version
        == REQUIRED_TRANSFORMERS_DISTRIBUTION_VERSION
        == "5.6.0.post1"
    )
    assert (
        spec.required_transformers_module_version
        == REQUIRED_TRANSFORMERS_MODULE_VERSION
        == "5.6.0"
    )
    assert "--revision" not in spec.arguments


def test_builds_ampere_and_connectx_3_environment_without_unvalidated_paths() -> None:
    specs = build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)

    assert specs[0].environment == (
        (
            "CUDA_VISIBLE_DEVICES",
            "GPU-00000000-0000-0000-0000-000000000001",
        ),
        ("NCCL_NET", "IB"),
        ("NCCL_IB_HCA", "=mlx4_0:1"),
        ("NCCL_GIN_ENABLE", "0"),
        ("NCCL_GIN_TYPE", "0"),
        ("NCCL_NET_GDR_LEVEL", "LOC"),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"),
        ("SGLANG_PP_LAYER_PARTITION", "30,28,20"),
    )
    assert dict(specs[1].environment)["NCCL_IB_HCA"] == "=mlx4_0:2"
    for spec in specs:
        assert "--fp8-gemm-backend" not in spec.arguments
        assert "--kt-gpu-prefill-token-threshold" not in spec.arguments
        assert "--kt-enable-dynamic-expert-update" not in spec.arguments


def test_builds_fail_closed_glm_4_7_flash_bf16_hybrid_smoke() -> None:
    plan = make_glm_4_7_flash_bf16_plan()

    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(plan, PYTHON_EXECUTABLE)

    assert spec.target_profile == GLM_4_7_FLASH_TARGET_PROFILE
    assert spec.pipeline_rank == 0
    assert argument_value(spec.arguments, "--pp-size") == "1"
    assert argument_value(spec.arguments, "--tp-size") == "1"
    assert argument_value(spec.arguments, "--nnodes") == "1"
    assert argument_value(spec.arguments, "--node-rank") == "0"
    assert argument_value(spec.arguments, "--kt-method") == "BF16"
    assert argument_value(spec.arguments, "--kt-num-gpu-experts") == "4"
    assert argument_value(spec.arguments, "--kt-max-deferred-experts-per-token") == "0"
    assert argument_value(spec.arguments, "--kt-expert-placement-strategy") == "uniform"
    assert argument_value(spec.arguments, "--chunked-prefill-size") == "1024"
    assert argument_value(spec.arguments, "--attention-backend") == "flashinfer"
    assert argument_value(spec.arguments, "--kv-cache-dtype") == "bfloat16"
    assert argument_value(spec.arguments, "--served-model-name") == "GLM-4.7-Flash"
    assert "--disable-cuda-graph" in spec.arguments
    assert "--record-kt-gpu-expert-distribution" in spec.arguments
    assert "--disable-shared-experts-fusion" in spec.arguments
    assert "--kt-enable-dynamic-expert-update" not in spec.arguments
    assert "--kt-gpu-prefill-token-threshold" not in spec.arguments
    assert spec.environment == (
        (
            "CUDA_VISIBLE_DEVICES",
            "GPU-00000000-0000-0000-0000-000000000001",
        ),
        ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
        ("SGLANG_KT_HYBRID_TIMING", "1"),
    )
    assert all(not name.startswith("NCCL_") for name, _value in spec.environment)
    assert SglangKtProcessLaunchSpec.model_validate_json(spec.model_dump_json()) == spec


def test_builds_fail_closed_glm_4_7_flash_cpu_routed_experts_control() -> None:
    plan = make_glm_4_7_flash_bf16_cpu_routed_experts_plan()

    (spec,) = build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
        plan, PYTHON_EXECUTABLE
    )

    assert spec.target_profile == GLM_4_7_FLASH_CPU_ROUTED_EXPERTS_TARGET_PROFILE
    assert spec.model_id == GLM_4_7_FLASH_BF16_MODEL_ID
    assert spec.expected_model_revision == GLM_4_7_FLASH_BF16_MODEL_REVISION
    assert spec.expected_sglang_revision == GLM_4_7_FLASH_SGLANG_REVISION
    assert spec.expected_ktransformers_revision == GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    assert len(spec.plan.stages) == 1
    assert spec.hca_devices == ()
    assert argument_value(spec.arguments, "--pp-size") == "1"
    assert argument_value(spec.arguments, "--tp-size") == "1"
    assert argument_value(spec.arguments, "--nnodes") == "1"
    assert argument_value(spec.arguments, "--kt-method") == "BF16"
    assert argument_value(spec.arguments, "--kt-num-gpu-experts") == "0"
    assert argument_value(spec.arguments, "--attention-backend") == "flashinfer"
    assert argument_value(spec.arguments, "--kv-cache-dtype") == "bfloat16"
    assert dict(spec.environment)["CUDA_VISIBLE_DEVICES"] == spec.gpu_uuid
    assert all(not name.startswith("NCCL_") for name, _value in spec.environment)
    assert SglangKtProcessLaunchSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize("resident_gpu_experts", (1, 63))
def test_glm_4_7_flash_mixed_profile_keeps_resident_expert_boundaries(
    resident_gpu_experts: int,
) -> None:
    (spec,) = build_glm_4_7_flash_bf16_process_launch_specs(
        make_glm_4_7_flash_bf16_plan(
            resident_gpu_experts=resident_gpu_experts,
        ),
        PYTHON_EXECUTABLE,
    )

    assert spec.stage.resident_gpu_experts == resident_gpu_experts


@pytest.mark.parametrize(
    ("stage_update", "error_message"),
    (
        ({"resident_gpu_experts": 1}, "exactly 0 resident GPU experts"),
        ({"resident_gpu_experts": 63}, "exactly 0 resident GPU experts"),
        ({"hca_devices": ("mlx4_0:1",)}, "does not admit HCA"),
        ({"ktransformers_method": "FP8"}, "method BF16"),
    ),
)
def test_glm_4_7_flash_cpu_routed_experts_control_rejects_other_resources(
    stage_update: dict[str, object],
    error_message: str,
) -> None:
    plan = make_glm_4_7_flash_bf16_cpu_routed_experts_plan()
    plan = plan.model_copy(
        update={"stages": (plan.stages[0].model_copy(update=stage_update),)}
    )

    with pytest.raises(ValueError, match=error_message):
        build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
            plan, PYTHON_EXECUTABLE
        )


@pytest.mark.parametrize(
    ("plan_update", "stage_update", "error_message"),
    (
        ({"model_id": ModelId("zai-org/GLM-4.7")}, {}, "only supports"),
        ({"model_revision": "1" * 40}, {}, "model revision must be"),
        ({"sglang_revision": "2" * 40}, {}, "SGLang revision must be"),
        (
            {"ktransformers_revision": "3" * 40},
            {},
            "KTransformers revision must be",
        ),
        ({"total_layers": 46}, {}, "exactly 47 layers"),
        ({"context_length": 4_096}, {}, "context length must be"),
        ({"max_total_tokens": 2_048}, {}, "max_total_tokens must be"),
        ({"max_concurrent_requests": 2}, {}, "one running request"),
        ({"static_memory_fraction": 0.75}, {}, "memory fraction must be"),
        ({}, {"ktransformers_method": "FP8"}, "method BF16"),
        ({}, {"resident_gpu_experts": 0}, "between 1 and 63"),
        ({}, {"resident_gpu_experts": 64}, "between 1 and 63"),
        (
            {},
            {"ktransformers_weight_path": "/var/lib/exo/models/other"},
            "model_path and ktransformers_weight_path",
        ),
        ({}, {"max_deferred_experts_per_token": 1}, "deferred experts disabled"),
        ({}, {"hca_devices": ("mlx4_0:1",)}, "does not admit HCA"),
    ),
)
def test_glm_4_7_flash_smoke_rejects_unvalidated_combinations(
    plan_update: dict[str, object],
    stage_update: dict[str, object],
    error_message: str,
) -> None:
    plan = make_glm_4_7_flash_bf16_plan()
    if stage_update:
        plan = plan.model_copy(
            update={"stages": (plan.stages[0].model_copy(update=stage_update),)}
        )
    plan = plan.model_copy(update=plan_update)

    with pytest.raises(ValueError, match=error_message):
        build_glm_4_7_flash_bf16_process_launch_specs(plan, PYTHON_EXECUTABLE)


def test_glm_4_7_flash_smoke_rejects_multiple_pipeline_stages() -> None:
    plan = make_glm_4_7_flash_bf16_plan()
    plan = plan.model_copy(update={"stages": (plan.stages[0], plan.stages[0])})

    with pytest.raises(ValueError, match="requires PP=1 and TP=1"):
        build_glm_4_7_flash_bf16_process_launch_specs(plan, PYTHON_EXECUTABLE)


def test_model_specific_builders_reject_the_other_target_profile() -> None:
    with pytest.raises(ValueError, match="GLM-5.2 builder requires"):
        build_glm_5_2_fp8_process_launch_specs(
            make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
        )
    with pytest.raises(ValueError, match="GLM-4.7-Flash builder requires"):
        build_glm_4_7_flash_bf16_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)
    with pytest.raises(ValueError, match="GLM-4.7-Flash builder requires"):
        build_glm_4_7_flash_bf16_process_launch_specs(
            make_glm_4_7_flash_bf16_cpu_routed_experts_plan(), PYTHON_EXECUTABLE
        )
    with pytest.raises(ValueError, match="CPU-routed-experts builder requires"):
        build_glm_4_7_flash_bf16_cpu_routed_experts_process_launch_specs(
            make_glm_4_7_flash_bf16_plan(), PYTHON_EXECUTABLE
        )


def test_process_launch_spec_roundtrip() -> None:
    spec = build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)[0]

    assert SglangKtProcessLaunchSpec.model_validate_json(spec.model_dump_json()) == spec


def test_launch_plan_requires_an_explicit_target_profile() -> None:
    payload = make_plan().model_dump(mode="python")
    del payload["target_profile"]

    with pytest.raises(ValidationError, match="targetProfile"):
        SglangKtLaunchPlan.model_validate(payload)


def test_process_launch_spec_rejects_free_form_command_or_environment() -> None:
    spec = build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)[0]
    serialized = spec.model_dump()

    with pytest.raises(ValidationError, match="arguments"):
        SglangKtProcessLaunchSpec.model_validate(
            {**serialized, "arguments": ("-m", "untrusted.module")}
        )
    with pytest.raises(ValidationError, match="environment"):
        SglangKtProcessLaunchSpec.model_validate(
            {
                **serialized,
                "environment": (("CUDA_VISIBLE_DEVICES", "GPU-wrong"),),
            }
        )
    with pytest.raises(ValidationError, match="unset_environment_variables"):
        SglangKtProcessLaunchSpec.model_validate(
            {
                **serialized,
                "unset_environment_variables": ("UNTRUSTED_VARIABLE",),
            }
        )
    with pytest.raises(ValidationError, match="unset_environment_variable_prefixes"):
        SglangKtProcessLaunchSpec.model_validate(
            {
                **serialized,
                "unset_environment_variable_prefixes": ("UNTRUSTED_",),
            }
        )


@pytest.mark.parametrize(
    ("plan", "error_message"),
    [
        (make_plan(model_id="zai-org/GLM-5-FP8"), "only supports"),
        (make_plan(sglang_revision="5" * 40), "SGLang revision must be"),
        (
            make_plan(ktransformers_revision="6" * 40),
            "KTransformers revision must be",
        ),
        (make_plan(ktransformers_method="AMXINT8"), "require KTransformers method"),
        (make_plan(hca_devices=()), "require hca_devices"),
    ],
)
def test_rejects_unverified_runtime_combinations(
    plan: SglangKtLaunchPlan, error_message: str
) -> None:
    with pytest.raises(ValueError, match=error_message):
        build_glm_5_2_fp8_process_launch_specs(plan, PYTHON_EXECUTABLE)


@pytest.mark.parametrize(
    ("stages", "error_message"),
    (
        (
            (make_plan().stages[0].model_copy(update={"end_layer": 78}),),
            "requires exactly three stages",
        ),
        (
            (
                make_plan().stages[0],
                make_plan().stages[1].model_copy(update={"end_layer": 54}),
                make_plan().stages[2].model_copy(update={"start_layer": 54}),
            ),
            "requires the 30,28,20 layer partition",
        ),
    ),
)
def test_glm_5_2_pp3_profile_rejects_other_topologies(
    stages: tuple[SglangKtStageSpec, ...], error_message: str
) -> None:
    with pytest.raises(ValueError, match=error_message):
        build_glm_5_2_fp8_process_launch_specs(
            make_plan().model_copy(update={"stages": stages}),
            PYTHON_EXECUTABLE,
        )


def test_rejects_non_absolute_python_executable() -> None:
    with pytest.raises(ValidationError, match="executable"):
        build_glm_5_2_fp8_process_launch_specs(make_plan(), "python")


@pytest.mark.parametrize(
    ("stage_update", "error_message"),
    [
        (
            {"ktransformers_weight_path": "/var/lib/exo/models/separate-weights"},
            "model_path and ktransformers_weight_path to match",
        ),
        (
            {"max_deferred_experts_per_token": 1},
            "deferred expert work to be disabled",
        ),
    ],
)
def test_rejects_unaudited_pipeline_runtime_options(
    stage_update: dict[str, object], error_message: str
) -> None:
    plan = make_plan()
    stages = (
        plan.stages[0].model_copy(update=stage_update),
        *plan.stages[1:],
    )

    with pytest.raises(ValueError, match=error_message):
        build_glm_5_2_fp8_process_launch_specs(
            plan.model_copy(update={"stages": stages}),
            PYTHON_EXECUTABLE,
        )


def test_rejects_pipeline_start_on_shared_indexer_layer() -> None:
    plan = make_plan()
    invalid_stages = (
        plan.stages[0].model_copy(update={"end_layer": 39}),
        plan.stages[1].model_copy(update={"start_layer": 39}),
        plan.stages[2],
    )
    invalid_plan = plan.model_copy(update={"stages": invalid_stages})

    assert 30 in GLM_5_2_FULL_INDEXER_LAYER_STARTS
    assert 38 in GLM_5_2_FULL_INDEXER_LAYER_STARTS
    assert 58 in GLM_5_2_FULL_INDEXER_LAYER_STARTS
    assert 39 not in GLM_5_2_FULL_INDEXER_LAYER_STARTS
    with pytest.raises(ValueError, match="full IndexShare layers"):
        build_glm_5_2_fp8_process_launch_specs(
            invalid_plan,
            PYTHON_EXECUTABLE,
        )
