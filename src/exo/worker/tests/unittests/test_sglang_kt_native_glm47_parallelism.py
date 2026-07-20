from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo.shared.types.common import ModelId, NodeId
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.native_glm47_parallelism import (
    DWAGON_NODE_ID,
    DWAGON_RANK_ONE_CPUS,
    DWAGON_RANK_ONE_GPU,
    DWAGON_RANK_ZERO_CPUS,
    DWAGON_RANK_ZERO_GPU,
    FWUFF_NODE_ID,
    FWUFF_RANK_TWO_CPUS,
    FWUFF_RANK_TWO_GPU,
    GLM_4_7_FLASH_NATIVE_INDEX_SHA256,
    GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS,
    GLM_4_7_FLASH_NATIVE_MAX_TOTAL_TOKENS,
    GLM_4_7_FLASH_NATIVE_PHYSICAL_WEIGHT_BYTES,
    GLM_4_7_FLASH_NATIVE_SHARD_COUNT,
    GLM_4_7_FLASH_NATIVE_WEIGHT_MAP_ENTRIES,
    Glm47NativeParallelism,
    Glm47NativeParallelismError,
    Glm47NativeSourceSupport,
    assess_native_glm47_source_support,
    build_dwagon_fwuff_native_glm47_plan,
    build_native_glm47_planned_diagnostic_receipt,
    build_native_glm47_process_specs,
    calculate_native_glm47_planned_receipt_sha256,
    calculate_native_glm47_process_spec_sha256,
    inspect_native_glm47_source_support,
)

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "sglang_kt"
MODEL_CONFIG_CONTENTS = (FIXTURE_DIRECTORY / "glm47_flash_config.json").read_bytes()

DWAGON_RUNTIME = "/runtime/dwagon/venv/bin/python"
FWUFF_RUNTIME = "/runtime/fwuff/venv/bin/python"
DWAGON_SOURCE = "/source/dwagon/sglang"
FWUFF_SOURCE = "/source/fwuff/sglang"
DWAGON_MODEL = "/models/dwagon/glm47"
FWUFF_MODEL = "/models/fwuff/glm47"

REQUIRED_OPTIONS = (
    "--attention-backend",
    "--chunked-prefill-size",
    "--disable-cuda-graph",
    "--disable-custom-all-reduce",
    "--disable-radix-cache",
    "--disable-shared-experts-fusion",
    "--dist-init-addr",
    "--dp-size",
    "--enable-dp-attention",
    "--ep-num-redundant-experts",
    "--ep-size",
    "--host",
    "--kv-cache-dtype",
    "--max-running-requests",
    "--max-total-tokens",
    "--mem-fraction-static",
    "--model-path",
    "--moe-a2a-backend",
    "--moe-dense-tp-size",
    "--moe-runner-backend",
    "--nnodes",
    "--node-rank",
    "--pp-size",
    "--port",
    "--reasoning-parser",
    "--served-model-name",
    "--tool-call-parser",
    "--tp-size",
    "--trust-remote-code",
    "--context-length",
)


def make_plan(
    mode: Glm47NativeParallelism = "tp3_ep1",
    *,
    distributed_port: int = 62500,
    rank_ports: tuple[int, int, int] = (62510, 62511, 62512),
):
    return build_dwagon_fwuff_native_glm47_plan(
        mode,
        dwagon_runtime_python=DWAGON_RUNTIME,
        fwuff_runtime_python=FWUFF_RUNTIME,
        dwagon_sglang_source_directory=DWAGON_SOURCE,
        fwuff_sglang_source_directory=FWUFF_SOURCE,
        dwagon_model_path=DWAGON_MODEL,
        fwuff_model_path=FWUFF_MODEL,
        dwagon_ip="192.168.40.24",
        fwuff_ip="192.168.40.248",
        distributed_port=distributed_port,
        rank_ports=rank_ports,
        hca_devices=("mlx4_0:1", "mlx4_0:2"),
        dwagon_socket_interface="ens13f0np0",
        fwuff_socket_interface="ens17f0",
    )


def argument_value(arguments: tuple[str, ...], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def server_args_source(*, omit: str | None = None) -> bytes:
    options = tuple(option for option in REQUIRED_OPTIONS if option != omit)
    calls = "\n".join(f'parser.add_argument("{option}")' for option in options)
    return (
        "MOE_A2A_BACKEND_CHOICES = ['none', 'deepep']\n"
        "MOE_RUNNER_BACKEND_CHOICES = ['auto', 'triton']\n"
        f"{calls}\n"
    ).encode()


def model_source(*, hook: str = "missing") -> bytes:
    if hook == "missing":
        method = ""
    elif hook == "exact":
        method = """
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )
"""
    else:
        method = """
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_shared_experts,
            num_groups=config.n_group,
        )
"""
    return (
        "class Glm4MoeLiteForCausalLM(DeepseekV2ForCausalLM):\n"
        f"{method or '    pass\n'}"
    ).encode()


def deepseek_source(*, hook: str = "exact") -> bytes:
    if hook == "missing":
        method = ""
    elif hook == "exact":
        method = """
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )
"""
    else:
        method = """
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_shared_experts,
            num_groups=config.n_group,
        )
"""
    return (f"class DeepseekV2ForCausalLM:\n{method or '    pass\n'}").encode()


def make_support(
    node_id: NodeId,
    *,
    hook: str = "missing",
    base_hook: str = "exact",
    omit_option: str | None = None,
    revision: str = GLM_4_7_FLASH_SGLANG_REVISION,
    clean: bool = True,
    config_contents: bytes = MODEL_CONFIG_CONTENTS,
    runtime_mismatch: bool = False,
) -> Glm47NativeSourceSupport:
    source_directory = DWAGON_SOURCE if node_id == DWAGON_NODE_ID else FWUFF_SOURCE
    model_path = DWAGON_MODEL if node_id == DWAGON_NODE_ID else FWUFF_MODEL
    runtime = DWAGON_RUNTIME if node_id == DWAGON_NODE_ID else FWUFF_RUNTIME
    server_contents = server_args_source(omit=omit_option)
    model_contents = model_source(hook=hook)
    base_model_contents = deepseek_source(hook=base_hook)
    return assess_native_glm47_source_support(
        node_id=node_id,
        source_directory=source_directory,
        model_config_path=f"{model_path}/config.json",
        runtime_executable=runtime,
        sglang_revision=revision,
        source_tree_clean=clean,
        server_args_contents=server_contents,
        model_implementation_contents=model_contents,
        base_model_implementation_contents=base_model_contents,
        runtime_server_args_contents=(
            server_contents + b"# runtime drift\n"
            if runtime_mismatch
            else server_contents
        ),
        runtime_model_implementation_contents=model_contents,
        runtime_base_model_implementation_contents=base_model_contents,
        model_config_contents=config_contents,
    )


def make_model_snapshot(node_id: NodeId) -> SglangKtVerifiedModelSnapshot:
    model_path = DWAGON_MODEL if node_id == DWAGON_NODE_ID else FWUFF_MODEL
    return SglangKtVerifiedModelSnapshot(
        model_path=model_path,
        model_id=ModelId("zai-org/GLM-4.7-Flash"),
        revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        weight_format="safetensors",
        ktransformers_method="BF16",
        full_indexer_layer_starts=(0,),
        contract_path="/contracts/glm47_flash_bf16_7dd20894.json",
        contract_receipt_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        contract_sha256=GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
        config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        index_sha256=GLM_4_7_FLASH_NATIVE_INDEX_SHA256,
        weight_map_entries=GLM_4_7_FLASH_NATIVE_WEIGHT_MAP_ENTRIES,
        shard_count=GLM_4_7_FLASH_NATIVE_SHARD_COUNT,
        physical_weight_bytes=GLM_4_7_FLASH_NATIVE_PHYSICAL_WEIGHT_BYTES,
    )


def make_model_snapshots() -> dict[NodeId, SglangKtVerifiedModelSnapshot]:
    return {
        DWAGON_NODE_ID: make_model_snapshot(DWAGON_NODE_ID),
        FWUFF_NODE_ID: make_model_snapshot(FWUFF_NODE_ID),
    }


def test_fixture_is_exact_admitted_model_config() -> None:
    assert hashlib.sha256(MODEL_CONFIG_CONTENTS).hexdigest() == (
        GLM_4_7_FLASH_BF16_CONFIG_SHA256
    )


@pytest.mark.parametrize(
    ("mode", "ep_size", "redundant_experts"),
    (("tp3_ep1", "1", "0"), ("tp3_ep3", "3", "2")),
)
def test_builds_exact_three_logical_node_tp_dp_ep_commands(
    mode: Glm47NativeParallelism,
    ep_size: str,
    redundant_experts: str,
) -> None:
    specs = build_native_glm47_process_specs(make_plan(mode))

    assert tuple(spec.world_rank for spec in specs) == (0, 1, 2)
    assert tuple(spec.rank.node_id for spec in specs) == (
        DWAGON_NODE_ID,
        DWAGON_NODE_ID,
        FWUFF_NODE_ID,
    )
    assert tuple(spec.rank.gpu_uuid for spec in specs) == (
        DWAGON_RANK_ZERO_GPU,
        DWAGON_RANK_ONE_GPU,
        FWUFF_RANK_TWO_GPU,
    )
    assert tuple(spec.rank.cpu_cores for spec in specs) == (
        DWAGON_RANK_ZERO_CPUS,
        DWAGON_RANK_ONE_CPUS,
        FWUFF_RANK_TWO_CPUS,
    )
    assert tuple(spec.rank.memory_nodes for spec in specs) == ((0,), (1,), (0,))
    for rank, spec in enumerate(specs):
        assert spec.arguments[:2] == ("-P", "-m")
        assert argument_value(spec.arguments, "--tp-size") == "3"
        assert argument_value(spec.arguments, "--pp-size") == "1"
        assert argument_value(spec.arguments, "--dp-size") == "3"
        assert argument_value(spec.arguments, "--ep-size") == ep_size
        assert (
            argument_value(spec.arguments, "--ep-num-redundant-experts")
            == redundant_experts
        )
        assert argument_value(spec.arguments, "--nnodes") == "3"
        assert argument_value(spec.arguments, "--node-rank") == str(rank)
        assert argument_value(spec.arguments, "--moe-dense-tp-size") == "1"
        assert argument_value(spec.arguments, "--moe-a2a-backend") == "none"
        assert argument_value(spec.arguments, "--moe-runner-backend") == "triton"
        assert "--enable-dp-attention" in spec.arguments
        assert "--disable-custom-all-reduce" in spec.arguments
        assert "--disable-shared-experts-fusion" in spec.arguments
        assert "--disable-cuda-graph" in spec.arguments
        assert argument_value(spec.arguments, "--max-total-tokens") == str(
            GLM_4_7_FLASH_NATIVE_MAX_TOTAL_TOKENS
        )
        assert argument_value(spec.arguments, "--max-running-requests") == str(
            GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS
        )
        assert GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS == 3


def test_builds_full_numa_binding_and_host_staged_dual_rail_environment() -> None:
    specs = build_native_glm47_process_specs(make_plan())

    for spec in specs:
        environment = dict(spec.environment)
        assert spec.cpu_bound_command[:5] == (
            "/usr/bin/numactl",
            "--physcpubind",
            ",".join(str(core) for core in spec.rank.cpu_cores),
            "--membind",
            str(spec.rank.memory_nodes[0]),
        )
        assert environment["CUDA_VISIBLE_DEVICES"] == spec.rank.gpu_uuid
        assert environment["NCCL_NET"] == "IB"
        assert environment["NCCL_IB_HCA"] == "=mlx4_0:1,mlx4_0:2"
        assert environment["NCCL_IB_MERGE_NICS"] == "1"
        assert environment["NCCL_NET_GDR_LEVEL"] == "LOC"
        assert environment["NCCL_GIN_ENABLE"] == "0"
        assert environment["NCCL_GIN_TYPE"] == "0"
        assert environment["NCCL_SOCKET_IFNAME"] == spec.rank.socket_interface
        assert environment["PYTHONPATH"] == (
            f"{spec.rank.sglang_source_directory}/python"
        )
        assert environment["PYTHONSAFEPATH"] == "1"
        assert "SGLANG_TORCH_PROFILER" not in environment
        assert spec.unset_environment_variable_prefixes == (
            "NCCL_",
            "SGLANG_",
            "VTUNE_",
            "NSYS_",
            "KINETO_",
        )


@pytest.mark.parametrize(
    ("plan_update", "error"),
    (
        ({"model_id": "zai-org/GLM-4.7"}, "model ID"),
        ({"max_total_tokens": 4_096}, "token and memory bounds"),
        ({"static_memory_fraction": 0.95}, "token and memory bounds"),
        ({"sglang_revision": "1" * 40}, "SGLang revision"),
    ),
)
def test_plan_rejects_unbounded_or_unpinned_variants(
    plan_update: dict[str, object], error: str
) -> None:
    payload = make_plan().model_dump()
    payload.update(plan_update)

    with pytest.raises(ValidationError, match=error):
        type(make_plan()).model_validate(payload)


@pytest.mark.parametrize(
    ("rank", "update", "error"),
    (
        (0, {"cpu_cores": tuple(range(55))}, "rank placement"),
        (1, {"memory_nodes": (0,)}, "rank placement"),
        (2, {"gpu_uuid": DWAGON_RANK_ONE_GPU}, "rank placement"),
        (2, {"hca_devices": ("mlx4_0:1",)}, "exactly two"),
    ),
)
def test_plan_rejects_inexact_hardware_placement(
    rank: int, update: dict[str, object], error: str
) -> None:
    plan = make_plan()
    ranks = list(plan.ranks)
    ranks[rank] = ranks[rank].model_copy(update=update)

    with pytest.raises((ValidationError, ValueError), match=error):
        type(plan).model_validate({**plan.model_dump(), "ranks": tuple(ranks)})


@pytest.mark.parametrize(
    "endpoint",
    (
        {"ip": "192.168.40.24", "port": 0},
        {"ip": "224.0.0.1", "port": 62510},
    ),
)
def test_rank_rejects_non_routable_service_endpoint(
    endpoint: dict[str, object],
) -> None:
    rank = make_plan().ranks[0]

    with pytest.raises(ValidationError, match="unicast IPv4 with a port"):
        type(rank).model_validate({**rank.model_dump(), "service_endpoint": endpoint})


@pytest.mark.parametrize(
    "coordinator",
    (
        {"ip": "192.168.40.24", "port": 0},
        {"ip": "224.0.0.1", "port": 62500},
    ),
)
def test_plan_rejects_non_routable_distributed_coordinator(
    coordinator: dict[str, object],
) -> None:
    plan = make_plan()

    with pytest.raises(ValidationError, match="unicast IPv4 with a port"):
        type(plan).model_validate(
            {**plan.model_dump(), "distributed_coordinator": coordinator}
        )


@pytest.mark.parametrize("reserved_offset", (0, 1, 5, 13))
def test_plan_rejects_http_collision_with_coordinator_reserved_ports(
    reserved_offset: int,
) -> None:
    with pytest.raises(ValidationError, match="reserved ports overlap"):
        make_plan(rank_ports=(62500 + reserved_offset, 62520, 62521))


def test_plan_rejects_coordinator_derived_port_overflow() -> None:
    with pytest.raises(ValidationError, match="derived ports exceed 65535"):
        make_plan(distributed_port=65_523)

    assert make_plan(distributed_port=65_522).distributed_coordinator.port == 65_522


def test_pinned_glm_lite_inherits_exact_deepseek_mapping_and_admits_ep3() -> None:
    support = make_support(DWAGON_NODE_ID, hook="missing")

    assert support.admitted_modes == ("tp3_ep1", "tp3_ep3")
    assert support.expert_location_hook_supported is True
    assert support.expert_location_hook_source == "deepseek_inherited"
    assert support.blockers == ()
    support.require_mode("tp3_ep3")


def test_exact_upstream_style_expert_location_hook_admits_ep3() -> None:
    support = make_support(DWAGON_NODE_ID, hook="exact")

    assert support.admitted_modes == ("tp3_ep1", "tp3_ep3")
    assert support.expert_location_hook_supported is True
    assert support.expert_location_hook_source == "glm_lite_override"
    assert support.blockers == ()
    support.require_mode("tp3_ep3")


def test_wrong_expert_mapping_hook_fails_closed() -> None:
    support = make_support(DWAGON_NODE_ID, hook="wrong")

    assert support.admitted_modes == ("tp3_ep1",)
    with pytest.raises(Glm47NativeParallelismError, match="expert-location"):
        support.require_mode("tp3_ep3")


@pytest.mark.parametrize("base_hook", ("missing", "wrong"))
def test_missing_or_wrong_inherited_expert_mapping_fails_closed(
    base_hook: str,
) -> None:
    support = make_support(DWAGON_NODE_ID, hook="missing", base_hook=base_hook)

    assert support.admitted_modes == ("tp3_ep1",)
    assert support.expert_location_hook_source == "unsupported"
    with pytest.raises(Glm47NativeParallelismError, match=r"64\+2 padded expert"):
        support.require_mode("tp3_ep3")


@pytest.mark.parametrize(
    (
        "omit_option",
        "revision",
        "clean",
        "config_contents",
        "runtime_mismatch",
        "blocker",
    ),
    (
        (
            "--enable-dp-attention",
            GLM_4_7_FLASH_SGLANG_REVISION,
            True,
            MODEL_CONFIG_CONTENTS,
            False,
            "missing options",
        ),
        (None, "1" * 40, True, MODEL_CONFIG_CONTENTS, False, "SGLang revision"),
        (
            None,
            GLM_4_7_FLASH_SGLANG_REVISION,
            False,
            MODEL_CONFIG_CONTENTS,
            False,
            "modifications or untracked files",
        ),
        (
            None,
            GLM_4_7_FLASH_SGLANG_REVISION,
            True,
            b"{}\n",
            False,
            "exact admitted snapshot",
        ),
        (
            None,
            GLM_4_7_FLASH_SGLANG_REVISION,
            True,
            MODEL_CONFIG_CONTENTS,
            True,
            "runtime modules do not match",
        ),
    ),
)
def test_source_support_rejects_missing_or_unbound_inputs(
    omit_option: str | None,
    revision: str,
    clean: bool,
    config_contents: bytes,
    runtime_mismatch: bool,
    blocker: str,
) -> None:
    support = make_support(
        DWAGON_NODE_ID,
        omit_option=omit_option,
        revision=revision,
        clean=clean,
        config_contents=config_contents,
        runtime_mismatch=runtime_mismatch,
    )

    assert support.admitted_modes == ()
    with pytest.raises(Glm47NativeParallelismError, match=blocker):
        support.require_mode("tp3_ep1")


def test_source_inspection_rejects_untracked_python_startup_code(
    tmp_path: Path,
) -> None:
    source_directory = tmp_path / "sglang"
    source_module_directory = source_directory / "python/sglang/srt"
    source_model_directory = source_module_directory / "models"
    source_model_directory.mkdir(parents=True)
    (source_module_directory / "server_args.py").write_bytes(server_args_source())
    (source_model_directory / "glm4_moe_lite.py").write_bytes(model_source())
    (source_model_directory / "deepseek_v2.py").write_bytes(deepseek_source())
    subprocess.run(("git", "init", "-q"), cwd=source_directory, check=True)
    subprocess.run(("git", "add", "."), cwd=source_directory, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Exo Test",
            "-c",
            "user.email=exo-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=source_directory,
        check=True,
    )
    (source_directory / "python/sitecustomize.py").write_text("raise RuntimeError\n")

    runtime_executable = tmp_path / "runtime/venv/bin/python"
    runtime_executable.parent.mkdir(parents=True)
    runtime_executable.write_text("#!/bin/sh\n")
    runtime_executable.chmod(0o755)
    runtime_site_packages = (
        runtime_executable.parent.parent / "lib/python3.13/site-packages/sglang/srt"
    )
    runtime_model_directory = runtime_site_packages / "models"
    runtime_model_directory.mkdir(parents=True)
    (runtime_site_packages / "server_args.py").write_bytes(server_args_source())
    (runtime_model_directory / "glm4_moe_lite.py").write_bytes(model_source())
    (runtime_model_directory / "deepseek_v2.py").write_bytes(deepseek_source())
    model_config_path = tmp_path / "model/config.json"
    model_config_path.parent.mkdir()
    model_config_path.write_bytes(MODEL_CONFIG_CONTENTS)

    support = inspect_native_glm47_source_support(
        node_id=DWAGON_NODE_ID,
        source_directory=source_directory,
        model_config_path=model_config_path,
        runtime_executable=runtime_executable,
    )

    assert support.source_tree_clean is False
    assert support.admitted_modes == ()
    assert any("untracked files" in blocker for blocker in support.blockers)


@pytest.mark.parametrize(
    "evidence_update",
    (
        {"admitted_modes": ("tp3_ep1",)},
        {"runtime_matches_source": False},
        {"expert_location_hook_source": "unsupported"},
    ),
)
def test_source_support_rejects_forged_derived_evidence(
    evidence_update: dict[str, object],
) -> None:
    support = make_support(DWAGON_NODE_ID)

    with pytest.raises(ValidationError):
        type(support).model_validate({**support.model_dump(), **evidence_update})


@pytest.mark.parametrize("mode", ("tp3_ep1", "tp3_ep3"))
def test_planned_receipt_binds_sanity_first_benchmark_and_support(
    mode: Glm47NativeParallelism,
) -> None:
    plan = make_plan(mode)
    support = {
        DWAGON_NODE_ID: make_support(DWAGON_NODE_ID),
        FWUFF_NODE_ID: make_support(FWUFF_NODE_ID),
    }

    receipt = build_native_glm47_planned_diagnostic_receipt(
        plan, support, make_model_snapshots()
    )
    digest = calculate_native_glm47_planned_receipt_sha256(receipt)

    assert receipt.status == "ready"
    assert receipt.performance_comparable is False
    assert receipt.profiler == "none"
    assert receipt.benchmark_protocol.execution_order == (
        "semantic_sanity",
        "hca_counters_before",
        "prefill_workload",
        "decode_workload",
        "hca_counters_after",
        "ownership_verified_cleanup",
    )
    assert receipt.benchmark_protocol.sanity_required is True
    assert receipt.benchmark_protocol.sanity_marker == "EXO_SANITY_OK"
    assert tuple(
        workload.warmup_count for workload in receipt.benchmark_protocol.workloads
    ) == (
        2,
        2,
    )
    assert tuple(
        workload.sample_count for workload in receipt.benchmark_protocol.workloads
    ) == (
        3,
        3,
    )
    assert len(digest) == 64
    assert (
        calculate_native_glm47_planned_receipt_sha256(
            type(receipt).model_validate_json(receipt.model_dump_json())
        )
        == digest
    )


@pytest.mark.parametrize("mutation", ("process_spec", "protocol", "snapshot"))
def test_planned_receipt_rejects_forged_derived_contract(mutation: str) -> None:
    receipt = build_native_glm47_planned_diagnostic_receipt(
        make_plan(),
        {
            DWAGON_NODE_ID: make_support(DWAGON_NODE_ID),
            FWUFF_NODE_ID: make_support(FWUFF_NODE_ID),
        },
        make_model_snapshots(),
    )
    payload = receipt.model_dump()
    if mutation == "process_spec":
        payload["process_specs"][1]["world_rank"] = 0
    elif mutation == "protocol":
        payload["benchmark_protocol"]["sanity_max_new_tokens"] += 1
    else:
        payload["model_snapshots"][1]["shard_count"] = 47

    with pytest.raises(ValidationError):
        type(receipt).model_validate(payload)


def test_planned_ep3_receipt_rejects_one_unpatched_node() -> None:
    with pytest.raises(Glm47NativeParallelismError, match=r"fwuff.*64\+2"):
        build_native_glm47_planned_diagnostic_receipt(
            make_plan("tp3_ep3"),
            {
                DWAGON_NODE_ID: make_support(DWAGON_NODE_ID),
                FWUFF_NODE_ID: make_support(FWUFF_NODE_ID, hook="wrong"),
            },
            make_model_snapshots(),
        )


@pytest.mark.parametrize(
    ("evidence_update", "error"),
    (
        ({"source_directory": "/source/other/sglang"}, "launch source path"),
        ({"runtime_executable": "/runtime/other/bin/python"}, "launch executable"),
        ({"model_config_path": "/models/other/config.json"}, "model config"),
        ({"sglang_revision": "1" * 40}, "launch revision"),
    ),
)
def test_planned_receipt_binds_support_to_launch_paths_and_revision(
    evidence_update: dict[str, object], error: str
) -> None:
    fwuff_support = make_support(FWUFF_NODE_ID).model_copy(update=evidence_update)

    with pytest.raises(Glm47NativeParallelismError, match=error):
        build_native_glm47_planned_diagnostic_receipt(
            make_plan(),
            {
                DWAGON_NODE_ID: make_support(DWAGON_NODE_ID),
                FWUFF_NODE_ID: fwuff_support,
            },
            make_model_snapshots(),
        )


@pytest.mark.parametrize(
    ("snapshot_update", "error"),
    (
        ({"model_path": "/models/other/glm47"}, "launch model path"),
        ({"revision": "1" * 40}, "pinned GLM-4.7 BF16 contract"),
        ({"shard_count": 47}, "pinned GLM-4.7 BF16 contract"),
    ),
)
def test_planned_receipt_requires_verified_snapshot_contract(
    snapshot_update: dict[str, object], error: str
) -> None:
    snapshots = make_model_snapshots()
    snapshots[FWUFF_NODE_ID] = snapshots[FWUFF_NODE_ID].model_copy(
        update=snapshot_update
    )

    with pytest.raises(Glm47NativeParallelismError, match=error):
        build_native_glm47_planned_diagnostic_receipt(
            make_plan(),
            {
                DWAGON_NODE_ID: make_support(DWAGON_NODE_ID),
                FWUFF_NODE_ID: make_support(FWUFF_NODE_ID),
            },
            snapshots,
        )


def test_process_spec_roundtrip_and_digest_bind_mode() -> None:
    ep1_spec = build_native_glm47_process_specs(make_plan("tp3_ep1"))[0]
    ep3_spec = build_native_glm47_process_specs(make_plan("tp3_ep3"))[0]

    assert type(ep1_spec).model_validate_json(ep1_spec.model_dump_json()) == ep1_spec
    assert calculate_native_glm47_process_spec_sha256(ep1_spec) != (
        calculate_native_glm47_process_spec_sha256(ep3_spec)
    )
