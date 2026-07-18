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
    REQUIRED_TRANSFORMERS_VERSION,
    SUPPORTED_KTRANSFORMERS_REVISION,
    SUPPORTED_SGLANG_REVISION,
    SglangKtProcessLaunchSpec,
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
    nccl_port: int,
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
        nccl_port=nccl_port,
        model_path="/var/lib/exo/models/glm-5.2-fp8",
        ktransformers_weight_path="/var/lib/exo/models/glm-5.2-fp8-kt",
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
        total_layers=78,
        context_length=262_144,
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
                nccl_port=31_000,
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
                nccl_port=31_001,
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
                nccl_port=31_000,
                cpu_cores=tuple(range(20)),
                memory_node=0,
                hca_devices=("mlx4_0:1",),
                ktransformers_method=ktransformers_method,
            ),
        ),
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_builds_three_logical_nodes_for_two_physical_hosts() -> None:
    plan = make_plan()

    specs = build_glm_5_2_fp8_process_launch_specs(plan, PYTHON_EXECUTABLE)

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
        assert argument_value(spec.arguments, "--nccl-port") == str(spec.nccl_port)


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
    assert argument_value(spec.arguments, "--max-running-requests") == "1"
    assert argument_value(spec.arguments, "--attention-backend") == "nsa"
    assert argument_value(spec.arguments, "--kv-cache-dtype") == "fp8_e4m3"
    assert argument_value(spec.arguments, "--tool-call-parser") == "glm47"
    assert argument_value(spec.arguments, "--reasoning-parser") == "glm45"
    assert "--disable-shared-experts-fusion" in spec.arguments
    assert "--trust-remote-code" in spec.arguments

    assert spec.model_path == plan.stages[1].model_path
    assert spec.ktransformers_weight_path == plan.stages[1].ktransformers_weight_path
    assert spec.hca_devices == plan.stages[1].hca_devices
    assert spec.distributed_coordinator == plan.distributed_coordinator
    assert spec.expected_model_revision == MODEL_REVISION
    assert spec.expected_sglang_revision == SUPPORTED_SGLANG_REVISION
    assert spec.expected_ktransformers_revision == SUPPORTED_KTRANSFORMERS_REVISION
    assert spec.required_transformers_version == REQUIRED_TRANSFORMERS_VERSION
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


def test_process_launch_spec_roundtrip() -> None:
    spec = build_glm_5_2_fp8_process_launch_specs(make_plan(), PYTHON_EXECUTABLE)[0]

    assert SglangKtProcessLaunchSpec.model_validate_json(spec.model_dump_json()) == spec


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


def test_rejects_non_absolute_python_executable() -> None:
    with pytest.raises(ValidationError, match="executable"):
        build_glm_5_2_fp8_process_launch_specs(make_plan(), "python")
