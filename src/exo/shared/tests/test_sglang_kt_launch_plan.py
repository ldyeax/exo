import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Host, NodeId
from exo.shared.types.worker.sglang_kt import (
    SglangKtLaunchPlan,
    SglangKtStageSpec,
)

REVISION = "1" * 40


def make_stage(
    pipeline_rank: int,
    start_layer: int,
    end_layer: int,
    *,
    node_id: str,
    gpu_suffix: int,
    cpu_cores: tuple[int, ...],
    memory_node: int,
    service_ip: str | None = None,
    service_port: int | None = None,
) -> SglangKtStageSpec:
    if service_ip is None:
        service_ip = "192.168.40.248" if node_id == "dwagon" else "192.168.40.249"
    return SglangKtStageSpec(
        pipeline_rank=pipeline_rank,
        start_layer=start_layer,
        end_layer=end_layer,
        node_id=NodeId(node_id),
        gpu_uuid=f"GPU-00000000-0000-0000-0000-{gpu_suffix:012x}",
        service_endpoint=Host(
            ip=service_ip,
            port=30_000 + pipeline_rank if service_port is None else service_port,
        ),
        model_path="/var/lib/exo/models/glm-5.2-fp8",
        ktransformers_weight_path="/var/lib/exo/models/glm-5.2-fp8",
        cpu_cores=cpu_cores,
        memory_nodes=(memory_node,),
        cpu_infer_threads=len(cpu_cores),
        threadpool_count=1,
        ktransformers_method="FP8",
        resident_gpu_experts=0,
        max_deferred_experts_per_token=0,
        hca_devices=("mlx5_0:1",),
    )


def make_plan(
    stages: tuple[SglangKtStageSpec, ...] | None = None,
    updates: dict[str, object] | None = None,
) -> SglangKtLaunchPlan:
    if stages is None:
        stages = (
            make_stage(
                0,
                0,
                30,
                node_id="dwagon",
                gpu_suffix=1,
                cpu_cores=tuple(range(30)),
                memory_node=0,
            ),
            make_stage(
                1,
                30,
                58,
                node_id="dwagon",
                gpu_suffix=2,
                cpu_cores=tuple(range(30, 58)),
                memory_node=1,
            ),
            make_stage(
                2,
                58,
                78,
                node_id="fwuff",
                gpu_suffix=3,
                cpu_cores=tuple(range(20)),
                memory_node=0,
            ),
        )
    values: dict[str, object] = {
        "target_profile": "glm52_fp8_pp3_sm86_v1",
        "model_id": ModelId("zai-org/GLM-5.2-FP8"),
        "model_revision": REVISION,
        "sglang_revision": "2" * 40,
        "ktransformers_revision": "3" * 40,
        "total_layers": 78,
        "context_length": 262_144,
        "max_total_tokens": 4_096,
        "static_memory_fraction": 0.8,
        "max_concurrent_requests": 1,
        "distributed_coordinator": Host(ip="192.168.40.248", port=29500),
        "rank_zero_endpoint": Host(ip="192.168.40.248", port=30000),
        "stages": stages,
    }
    if updates is not None:
        values.update(updates)
    return SglangKtLaunchPlan.model_validate(values)


def test_target_launch_plan_roundtrip_and_partition() -> None:
    plan = make_plan()

    assert plan.pipeline_layer_partition == (30, 28, 20)
    assert SglangKtLaunchPlan.model_validate_json(plan.model_dump_json()) == plan


def test_stage_rejects_oversubscribed_cpu_threads() -> None:
    with pytest.raises(ValidationError, match="cpu_infer_threads exceeds"):
        SglangKtStageSpec(
            pipeline_rank=0,
            start_layer=0,
            end_layer=1,
            node_id=NodeId("dwagon"),
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            service_endpoint=Host(ip="192.168.40.248", port=30_000),
            model_path="/model",
            ktransformers_weight_path="/weights",
            cpu_cores=(0,),
            memory_nodes=(0,),
            cpu_infer_threads=2,
            threadpool_count=1,
            ktransformers_method="AMXINT8",
        )


@pytest.mark.parametrize(
    ("stages", "error_message"),
    [
        (
            (
                make_stage(
                    1,
                    0,
                    78,
                    node_id="dwagon",
                    gpu_suffix=1,
                    cpu_cores=(0,),
                    memory_node=0,
                ),
            ),
            "ranks must be ordered and contiguous",
        ),
        (
            (
                make_stage(
                    0,
                    0,
                    30,
                    node_id="dwagon",
                    gpu_suffix=1,
                    cpu_cores=(0,),
                    memory_node=0,
                ),
                make_stage(
                    1,
                    31,
                    78,
                    node_id="fwuff",
                    gpu_suffix=2,
                    cpu_cores=(0,),
                    memory_node=0,
                ),
            ),
            "layer ranges must be ordered and contiguous",
        ),
        (
            (
                make_stage(
                    0,
                    0,
                    30,
                    node_id="dwagon",
                    gpu_suffix=1,
                    cpu_cores=(0,),
                    memory_node=0,
                ),
                make_stage(
                    1,
                    30,
                    78,
                    node_id="fwuff",
                    gpu_suffix=1,
                    cpu_cores=(0,),
                    memory_node=0,
                ),
            ),
            "must use distinct GPUs",
        ),
        (
            (
                make_stage(
                    0,
                    0,
                    30,
                    node_id="dwagon",
                    gpu_suffix=1,
                    cpu_cores=(0, 1),
                    memory_node=0,
                ),
                make_stage(
                    1,
                    30,
                    78,
                    node_id="dwagon",
                    gpu_suffix=2,
                    cpu_cores=(1, 2),
                    memory_node=1,
                ),
            ),
            "same node must use disjoint cpu_cores",
        ),
    ],
)
def test_launch_plan_rejects_invalid_pipeline(
    stages: tuple[SglangKtStageSpec, ...], error_message: str
) -> None:
    with pytest.raises(ValidationError, match=error_message):
        make_plan(stages)


@pytest.mark.parametrize(
    ("updates", "error_message"),
    [
        ({"total_layers": 79}, "cover total_layers exactly"),
        (
            {"distributed_coordinator": Host(ip="0.0.0.0", port=29500)},
            "distributed_coordinator must use a concrete IPv4",
        ),
        (
            {"rank_zero_endpoint": Host(ip="192.168.40.248", port=0)},
            "rank_zero_endpoint must use a concrete IPv4",
        ),
        ({"model_revision": "main"}, "model_revision"),
        ({"max_total_tokens": 262_145}, "cannot exceed context_length"),
        ({"static_memory_fraction": 1.0}, "static_memory_fraction"),
    ],
)
def test_launch_plan_rejects_invalid_cluster_fields(
    updates: dict[str, object], error_message: str
) -> None:
    with pytest.raises(ValidationError, match=error_message):
        make_plan(updates=updates)


def test_launch_plan_rejects_rank_zero_service_endpoint_mismatch() -> None:
    with pytest.raises(ValidationError, match="rank_zero_endpoint must equal"):
        make_plan(
            updates={"rank_zero_endpoint": Host(ip="192.168.40.248", port=30_010)}
        )


def test_launch_plan_rejects_duplicate_service_endpoints() -> None:
    stages = (
        make_stage(
            0,
            0,
            30,
            node_id="dwagon",
            gpu_suffix=1,
            cpu_cores=(0,),
            memory_node=0,
        ),
        make_stage(
            1,
            30,
            78,
            node_id="fwuff",
            gpu_suffix=2,
            cpu_cores=(0,),
            memory_node=0,
            service_ip="192.168.40.248",
            service_port=30_000,
        ),
    )

    with pytest.raises(ValidationError, match="distinct service endpoints"):
        make_plan(stages)
