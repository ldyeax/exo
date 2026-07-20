from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from ipaddress import IPv4Address, ip_address
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, cast, final

from pydantic import StringConstraints, model_validator

from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    GpuUuid,
    HcaDevice,
    ResourceIndex,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from exo.worker.sglang_kt.model_contract import SglangKtVerifiedModelSnapshot
from exo.worker.sglang_kt.receipt_io import canonical_sglang_kt_json
from exo.worker.sglang_kt.serving_benchmark_receipt import (
    GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_DECODE_INPUT_TOKENS,
    GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS,
    GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_PREFILL_INPUT_TOKENS,
    GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS,
    GLM_4_7_FLASH_SANITY_CHAT_TEMPLATE_SHA256,
    GLM_4_7_FLASH_SANITY_INPUT_IDS_SHA256,
    GLM_4_7_FLASH_SANITY_INPUT_TOKENS,
    GLM_4_7_FLASH_SANITY_MARKER,
    GLM_4_7_FLASH_SANITY_MAX_NEW_TOKENS,
    GLM_4_7_FLASH_SANITY_RENDERED_PROMPT_SHA256,
    GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
    WARM_SERVING_MINIMUM_SAMPLES,
    WARM_SERVING_MINIMUM_WARMUPS,
)

Glm47NativeParallelism = Literal["tp3_ep1", "tp3_ep3"]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SocketInterface = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$"),
]

DWAGON_NODE_ID: Final = NodeId("dwagon")
FWUFF_NODE_ID: Final = NodeId("fwuff")
DWAGON_RANK_ZERO_GPU: Final = "GPU-63a7760a-6164-0758-9228-03dbf35d721c"
DWAGON_RANK_ONE_GPU: Final = "GPU-a442b72e-6727-6322-ba5d-5a9512b79886"
FWUFF_RANK_TWO_GPU: Final = "GPU-93e47864-13c3-0211-f3a9-ccee1a00d618"
DWAGON_RANK_ZERO_CPUS: Final = tuple(range(56))
DWAGON_RANK_ONE_CPUS: Final = tuple(range(56, 112))
FWUFF_RANK_TWO_CPUS: Final = tuple(range(60))

GLM_4_7_FLASH_NATIVE_CONTEXT_LENGTH: Final = 2_048
GLM_4_7_FLASH_NATIVE_MAX_TOTAL_TOKENS: Final = 2_048
GLM_4_7_FLASH_NATIVE_STATIC_MEMORY_FRACTION: Final = 0.92
GLM_4_7_FLASH_NATIVE_CHUNKED_PREFILL_SIZE: Final = 1_024
GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS: Final = 3
GLM_4_7_FLASH_NATIVE_EP3_REDUNDANT_EXPERTS: Final = 2
GLM_4_7_FLASH_NATIVE_INDEX_SHA256: Final = (
    "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
)
GLM_4_7_FLASH_NATIVE_WEIGHT_MAP_ENTRIES: Final = 9_703
GLM_4_7_FLASH_NATIVE_SHARD_COUNT: Final = 48
GLM_4_7_FLASH_NATIVE_PHYSICAL_WEIGHT_BYTES: Final = 62_444_175_504
GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS: Final = (0, 1, 2, 3, 4, 5, 13)
GLM_4_7_FLASH_NATIVE_PROCESS_SPEC_CANONICALIZATION: Final = (
    "exo-glm47-native-tp3-ep-process-spec-v1"
)
GLM_4_7_FLASH_NATIVE_PLANNED_RECEIPT_CANONICALIZATION: Final = (
    "exo-glm47-native-tp3-ep-planned-receipt-v1"
)
_EP3_EXPERT_LOCATION_BLOCKER: Final = (
    "EP3 requires an exact GLM-Lite override or inherited DeepSeek "
    "expert-location method for the 64+2 padded expert mapping"
)

_REQUIRED_SERVER_OPTIONS: Final = frozenset(
    {
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
    }
)


class Glm47NativeParallelismError(ValueError):
    """Raised when native TP3/EP evidence is not admitted exactly."""


@final
class Glm47NativeRankSpec(FrozenModel):
    """One logical one-GPU SGLang node in the three-rank process group."""

    world_rank: ResourceIndex
    node_id: NodeId
    executable: AbsoluteRuntimePath
    sglang_source_directory: AbsoluteRuntimePath
    model_path: AbsoluteRuntimePath
    gpu_uuid: GpuUuid
    cpu_cores: tuple[ResourceIndex, ...]
    memory_nodes: tuple[ResourceIndex, ...]
    service_endpoint: Host
    socket_interface: SocketInterface
    hca_devices: tuple[HcaDevice, ...]

    @model_validator(mode="after")
    def validate_rank_resources(self) -> "Glm47NativeRankSpec":
        if not self.cpu_cores or len(set(self.cpu_cores)) != len(self.cpu_cores):
            raise ValueError("native rank cpu_cores must be nonempty and unique")
        if len(self.memory_nodes) != 1:
            raise ValueError("native rank requires exactly one NUMA memory node")
        if len(self.hca_devices) != 2 or len(set(self.hca_devices)) != 2:
            raise ValueError("native rank requires exactly two distinct HCA rails")
        try:
            endpoint_ip = ip_address(self.service_endpoint.ip)
        except ValueError as error:
            raise ValueError("native rank endpoint must be concrete IPv4") from error
        if (
            not isinstance(endpoint_ip, IPv4Address)
            or endpoint_ip.is_unspecified
            or endpoint_ip.is_multicast
            or self.service_endpoint.port == 0
        ):
            raise ValueError("native rank endpoint must be unicast IPv4 with a port")
        return self


@final
class Glm47NativeTpEpPlan(FrozenModel):
    """Exact native SGLang TP3 control or TP3/EP3 experiment."""

    mode: Glm47NativeParallelism
    model_id: ModelId
    model_revision: GitRevision
    model_config_sha256: Sha256Digest
    sglang_revision: GitRevision
    distributed_coordinator: Host
    ranks: tuple[Glm47NativeRankSpec, ...]
    context_length: int = GLM_4_7_FLASH_NATIVE_CONTEXT_LENGTH
    max_total_tokens: int = GLM_4_7_FLASH_NATIVE_MAX_TOTAL_TOKENS
    static_memory_fraction: float = GLM_4_7_FLASH_NATIVE_STATIC_MEMORY_FRACTION
    max_running_requests: int = GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS
    chunked_prefill_size: int = GLM_4_7_FLASH_NATIVE_CHUNKED_PREFILL_SIZE

    @property
    def expert_parallel_size(self) -> Literal[1, 3]:
        return 1 if self.mode == "tp3_ep1" else 3

    @property
    def redundant_expert_count(self) -> Literal[0, 2]:
        return (
            0 if self.mode == "tp3_ep1" else GLM_4_7_FLASH_NATIVE_EP3_REDUNDANT_EXPERTS
        )

    @model_validator(mode="after")
    def validate_pinned_plan(self) -> "Glm47NativeTpEpPlan":
        if self.model_id != GLM_4_7_FLASH_BF16_MODEL_ID:
            raise ValueError("native GLM-4.7 model ID is not pinned")
        if self.model_revision != GLM_4_7_FLASH_BF16_MODEL_REVISION:
            raise ValueError("native GLM-4.7 model revision is not pinned")
        if self.model_config_sha256 != GLM_4_7_FLASH_BF16_CONFIG_SHA256:
            raise ValueError("native GLM-4.7 config SHA-256 is not pinned")
        if self.sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
            raise ValueError("native GLM-4.7 SGLang revision is not pinned")
        if (
            self.context_length != GLM_4_7_FLASH_NATIVE_CONTEXT_LENGTH
            or self.max_total_tokens != GLM_4_7_FLASH_NATIVE_MAX_TOTAL_TOKENS
            or self.static_memory_fraction
            != GLM_4_7_FLASH_NATIVE_STATIC_MEMORY_FRACTION
            or self.max_running_requests != GLM_4_7_FLASH_NATIVE_MAX_RUNNING_REQUESTS
            or self.chunked_prefill_size != GLM_4_7_FLASH_NATIVE_CHUNKED_PREFILL_SIZE
        ):
            raise ValueError("native GLM-4.7 token and memory bounds are not pinned")
        if tuple(rank.world_rank for rank in self.ranks) != (0, 1, 2):
            raise ValueError("native GLM-4.7 requires logical ranks 0,1,2")
        expected_placement = (
            (
                DWAGON_NODE_ID,
                DWAGON_RANK_ZERO_GPU,
                DWAGON_RANK_ZERO_CPUS,
                (0,),
            ),
            (
                DWAGON_NODE_ID,
                DWAGON_RANK_ONE_GPU,
                DWAGON_RANK_ONE_CPUS,
                (1,),
            ),
            (FWUFF_NODE_ID, FWUFF_RANK_TWO_GPU, FWUFF_RANK_TWO_CPUS, (0,)),
        )
        observed_placement = tuple(
            (rank.node_id, rank.gpu_uuid, rank.cpu_cores, rank.memory_nodes)
            for rank in self.ranks
        )
        if observed_placement != expected_placement:
            raise ValueError(
                "native GLM-4.7 rank placement does not match dwagon/fwuff"
            )
        if self.ranks[0].service_endpoint.ip != self.ranks[1].service_endpoint.ip:
            raise ValueError("dwagon logical ranks must use the same host IP")
        if self.ranks[2].service_endpoint.ip == self.ranks[0].service_endpoint.ip:
            raise ValueError("fwuff logical rank must use a distinct host IP")
        if len({str(rank.service_endpoint) for rank in self.ranks}) != 3:
            raise ValueError("native GLM-4.7 service endpoints must be distinct")
        try:
            coordinator_ip = ip_address(self.distributed_coordinator.ip)
        except ValueError as error:
            raise ValueError(
                "distributed coordinator must be unicast IPv4 with a port"
            ) from error
        if (
            not isinstance(coordinator_ip, IPv4Address)
            or coordinator_ip.is_unspecified
            or coordinator_ip.is_multicast
            or self.distributed_coordinator.port == 0
        ):
            raise ValueError("distributed coordinator must be unicast IPv4 with a port")
        if self.distributed_coordinator.ip != self.ranks[0].service_endpoint.ip:
            raise ValueError(
                "distributed coordinator must be hosted by logical rank zero"
            )
        highest_internal_port = (
            self.distributed_coordinator.port
            + GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS[-1]
        )
        if highest_internal_port > 65_535:
            raise ValueError("distributed coordinator derived ports exceed 65535")
        coordinator_ports = {
            self.distributed_coordinator.port + offset
            for offset in GLM_4_7_FLASH_NATIVE_COORDINATOR_PORT_OFFSETS
        }
        if any(
            rank.service_endpoint.ip == self.distributed_coordinator.ip
            and rank.service_endpoint.port in coordinator_ports
            for rank in self.ranks
        ):
            raise ValueError(
                "distributed coordinator reserved ports overlap an HTTP endpoint"
            )
        if len({rank.model_path for rank in self.ranks[:2]}) != 1:
            raise ValueError("dwagon logical ranks must share one model snapshot")
        if len({rank.executable for rank in self.ranks[:2]}) != 1:
            raise ValueError("dwagon logical ranks must share one runtime executable")
        if len({rank.sglang_source_directory for rank in self.ranks[:2]}) != 1:
            raise ValueError("dwagon logical ranks must share one SGLang source")
        if len({rank.hca_devices for rank in self.ranks}) != 1:
            raise ValueError("all native ranks must select the same two HCA rails")
        return self


@final
class Glm47NativeProcessSpec(FrozenModel):
    """Inert command, environment, and NUMA binding for one logical rank."""

    plan: Glm47NativeTpEpPlan
    world_rank: ResourceIndex

    @property
    def rank(self) -> Glm47NativeRankSpec:
        return self.plan.ranks[self.world_rank]

    @property
    def arguments(self) -> tuple[str, ...]:
        rank = self.rank
        return (
            "-P",
            "-m",
            "sglang.launch_server",
            "--model-path",
            rank.model_path,
            "--tp-size",
            "3",
            "--pp-size",
            "1",
            "--dp-size",
            "3",
            "--enable-dp-attention",
            "--moe-dense-tp-size",
            "1",
            "--ep-size",
            str(self.plan.expert_parallel_size),
            "--ep-num-redundant-experts",
            str(self.plan.redundant_expert_count),
            "--moe-a2a-backend",
            "none",
            "--moe-runner-backend",
            "triton",
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--nnodes",
            "3",
            "--node-rank",
            str(rank.world_rank),
            "--dist-init-addr",
            str(self.plan.distributed_coordinator),
            "--host",
            rank.service_endpoint.ip,
            "--port",
            str(rank.service_endpoint.port),
            "--context-length",
            str(self.plan.context_length),
            "--max-total-tokens",
            str(self.plan.max_total_tokens),
            "--mem-fraction-static",
            str(self.plan.static_memory_fraction),
            "--max-running-requests",
            str(self.plan.max_running_requests),
            "--chunked-prefill-size",
            str(self.plan.chunked_prefill_size),
            "--disable-cuda-graph",
            "--disable-radix-cache",
            "--attention-backend",
            "flashinfer",
            "--kv-cache-dtype",
            "bfloat16",
            "--tool-call-parser",
            "glm47",
            "--reasoning-parser",
            "glm45",
            "--served-model-name",
            "GLM-4.7-Flash",
            "--trust-remote-code",
        )

    @property
    def environment(self) -> tuple[tuple[str, str], ...]:
        rank = self.rank
        return (
            ("CUDA_VISIBLE_DEVICES", rank.gpu_uuid),
            ("GLOO_SOCKET_IFNAME", rank.socket_interface),
            ("NCCL_DEBUG", "INFO"),
            ("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV"),
            ("NCCL_GIN_ENABLE", "0"),
            ("NCCL_GIN_TYPE", "0"),
            ("NCCL_IB_HCA", f"={','.join(rank.hca_devices)}"),
            ("NCCL_IB_MERGE_NICS", "1"),
            ("NCCL_NET", "IB"),
            ("NCCL_NET_GDR_LEVEL", "LOC"),
            ("NCCL_SOCKET_IFNAME", rank.socket_interface),
            ("PYTHONDONTWRITEBYTECODE", "1"),
            ("PYTHONHASHSEED", "0"),
            ("PYTHONSAFEPATH", "1"),
            (
                "PYTHONPATH",
                str(PurePosixPath(rank.sglang_source_directory) / "python"),
            ),
            ("PYTORCH_ALLOC_CONF", "expandable_segments:True"),
            ("TOKENIZERS_PARALLELISM", "false"),
        )

    @property
    def unset_environment_variables(self) -> tuple[str, ...]:
        return ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTORCH_ALLOC_CONF")

    @property
    def unset_environment_variable_prefixes(self) -> tuple[str, ...]:
        return ("NCCL_", "SGLANG_", "VTUNE_", "NSYS_", "KINETO_")

    @property
    def command(self) -> tuple[str, ...]:
        return (self.rank.executable, *self.arguments)

    @property
    def cpu_bound_command(self) -> tuple[str, ...]:
        return (
            "/usr/bin/numactl",
            "--physcpubind",
            ",".join(str(core) for core in self.rank.cpu_cores),
            "--membind",
            str(self.rank.memory_nodes[0]),
            *self.command,
        )

    @model_validator(mode="after")
    def validate_rank_exists(self) -> "Glm47NativeProcessSpec":
        if self.world_rank >= len(self.plan.ranks):
            raise ValueError("native process rank is absent from its plan")
        return self


@final
class Glm47NativeSourceSupport(FrozenModel):
    """Static, hash-bound support evidence from one physical node."""

    node_id: NodeId
    source_directory: AbsoluteRuntimePath
    model_config_path: AbsoluteRuntimePath
    runtime_executable: AbsoluteRuntimePath
    sglang_revision: GitRevision
    source_tree_clean: bool
    server_args_sha256: Sha256Digest
    model_implementation_sha256: Sha256Digest
    base_model_implementation_sha256: Sha256Digest
    runtime_server_args_sha256: Sha256Digest
    runtime_model_implementation_sha256: Sha256Digest
    runtime_base_model_implementation_sha256: Sha256Digest
    runtime_matches_source: bool
    model_config_sha256: Sha256Digest
    parser_options: tuple[str, ...]
    moe_a2a_backend_choices: tuple[str, ...]
    moe_runner_backend_choices: tuple[str, ...]
    glm_lite_inherits_deepseek: bool
    expert_location_hook_supported: bool
    expert_location_hook_source: Literal[
        "glm_lite_override", "deepseek_inherited", "unsupported"
    ]
    admitted_modes: tuple[Glm47NativeParallelism, ...]
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_support_evidence(self) -> "Glm47NativeSourceSupport":
        hashes_match = (
            self.runtime_server_args_sha256 == self.server_args_sha256
            and self.runtime_model_implementation_sha256
            == self.model_implementation_sha256
            and self.runtime_base_model_implementation_sha256
            == self.base_model_implementation_sha256
        )
        if self.runtime_matches_source != hashes_match:
            raise ValueError("runtime/source match flag contradicts module hashes")
        hook_source_supported = self.expert_location_hook_source != "unsupported"
        if self.expert_location_hook_supported != hook_source_supported:
            raise ValueError("expert-location support contradicts its source")
        if (
            self.expert_location_hook_source == "deepseek_inherited"
            and not self.glm_lite_inherits_deepseek
        ):
            raise ValueError(
                "inherited expert-location hook requires the DeepSeek base"
            )

        general_blockers = tuple(
            blocker
            for blocker in self.blockers
            if blocker != _EP3_EXPERT_LOCATION_BLOCKER
        )
        generally_supported = (
            self.sglang_revision == GLM_4_7_FLASH_SGLANG_REVISION
            and self.source_tree_clean
            and self.runtime_matches_source
            and self.model_config_sha256 == GLM_4_7_FLASH_BF16_CONFIG_SHA256
            and self.glm_lite_inherits_deepseek
            and _REQUIRED_SERVER_OPTIONS.issubset(self.parser_options)
            and "none" in self.moe_a2a_backend_choices
            and "triton" in self.moe_runner_backend_choices
            and not general_blockers
        )
        expected_modes: tuple[Glm47NativeParallelism, ...]
        if generally_supported and self.expert_location_hook_supported:
            expected_modes = ("tp3_ep1", "tp3_ep3")
        elif generally_supported:
            expected_modes = ("tp3_ep1",)
        else:
            expected_modes = ()
        if self.admitted_modes != expected_modes:
            raise ValueError("admitted modes contradict support evidence")
        if (
            not self.expert_location_hook_supported
            and _EP3_EXPERT_LOCATION_BLOCKER not in self.blockers
        ):
            raise ValueError("unsupported EP3 evidence lacks its blocker")
        if generally_supported and self.expert_location_hook_supported:
            if self.blockers:
                raise ValueError("fully admitted evidence cannot contain blockers")
        elif generally_supported:
            if self.blockers != (_EP3_EXPERT_LOCATION_BLOCKER,):
                raise ValueError("EP1-only evidence has unexpected blockers")
        elif not self.blockers:
            raise ValueError("rejected evidence must identify at least one blocker")
        return self

    def require_mode(self, mode: Glm47NativeParallelism) -> None:
        if mode not in self.admitted_modes:
            detail = "; ".join(self.blockers) or "mode was not admitted"
            raise Glm47NativeParallelismError(
                f"{self.node_id} does not support {mode}: {detail}"
            )


@final
class Glm47NativeBenchmarkWorkloadProtocol(FrozenModel):
    kind: Literal["prefill", "decode"]
    input_tokens: int
    input_ids_sha256: Sha256Digest
    output_tokens: int
    warmup_count: int
    sample_count: int


@final
class Glm47NativeBenchmarkProtocol(FrozenModel):
    """Sanity-first measurement order reused from the serving benchmark."""

    protocol_version: Literal["glm47-native-tp3-ep-diagnostic-v1"]
    execution_order: tuple[
        Literal["semantic_sanity"],
        Literal["hca_counters_before"],
        Literal["prefill_workload"],
        Literal["decode_workload"],
        Literal["hca_counters_after"],
        Literal["ownership_verified_cleanup"],
    ]
    sanity_required: Literal[True]
    sanity_marker: Literal["EXO_SANITY_OK"]
    sanity_input_tokens: int
    sanity_max_new_tokens: int
    sanity_input_ids_sha256: Sha256Digest
    sanity_chat_template_sha256: Sha256Digest
    sanity_rendered_prompt_sha256: Sha256Digest
    sampling_seed: int
    workloads: tuple[
        Glm47NativeBenchmarkWorkloadProtocol,
        Glm47NativeBenchmarkWorkloadProtocol,
    ]


@final
class Glm47NativePlannedDiagnosticReceipt(FrozenModel):
    schema_version: Literal[1]
    kind: Literal["glm47_flash_native_tp3_ep_planned_diagnostic"]
    status: Literal["ready"]
    performance_comparable: Literal[False]
    profiler: Literal["none"]
    plan: Glm47NativeTpEpPlan
    process_specs: tuple[
        Glm47NativeProcessSpec,
        Glm47NativeProcessSpec,
        Glm47NativeProcessSpec,
    ]
    source_support: tuple[Glm47NativeSourceSupport, Glm47NativeSourceSupport]
    model_snapshots: tuple[SglangKtVerifiedModelSnapshot, SglangKtVerifiedModelSnapshot]
    benchmark_protocol: Glm47NativeBenchmarkProtocol

    @model_validator(mode="after")
    def validate_planned_diagnostic_contract(
        self,
    ) -> "Glm47NativePlannedDiagnosticReceipt":
        if self.process_specs != build_native_glm47_process_specs(self.plan):
            raise ValueError("planned process specs do not derive from the plan")
        _validate_native_glm47_support_bindings(self.plan, self.source_support)
        _validate_native_glm47_model_snapshot_bindings(self.plan, self.model_snapshots)
        if self.benchmark_protocol != build_native_glm47_benchmark_protocol():
            raise ValueError(
                "benchmark protocol is not the pinned sanity-first protocol"
            )
        return self


def build_dwagon_fwuff_native_glm47_plan(
    mode: Glm47NativeParallelism,
    *,
    dwagon_runtime_python: str,
    fwuff_runtime_python: str,
    dwagon_sglang_source_directory: str,
    fwuff_sglang_source_directory: str,
    dwagon_model_path: str,
    fwuff_model_path: str,
    dwagon_ip: str,
    fwuff_ip: str,
    distributed_port: int,
    rank_ports: tuple[int, int, int],
    hca_devices: tuple[str, str],
    dwagon_socket_interface: str,
    fwuff_socket_interface: str,
) -> Glm47NativeTpEpPlan:
    rank_inputs = (
        (
            DWAGON_NODE_ID,
            dwagon_runtime_python,
            dwagon_sglang_source_directory,
            dwagon_model_path,
            DWAGON_RANK_ZERO_GPU,
            DWAGON_RANK_ZERO_CPUS,
            (0,),
            dwagon_ip,
            dwagon_socket_interface,
        ),
        (
            DWAGON_NODE_ID,
            dwagon_runtime_python,
            dwagon_sglang_source_directory,
            dwagon_model_path,
            DWAGON_RANK_ONE_GPU,
            DWAGON_RANK_ONE_CPUS,
            (1,),
            dwagon_ip,
            dwagon_socket_interface,
        ),
        (
            FWUFF_NODE_ID,
            fwuff_runtime_python,
            fwuff_sglang_source_directory,
            fwuff_model_path,
            FWUFF_RANK_TWO_GPU,
            FWUFF_RANK_TWO_CPUS,
            (0,),
            fwuff_ip,
            fwuff_socket_interface,
        ),
    )
    ranks = tuple(
        Glm47NativeRankSpec(
            world_rank=world_rank,
            node_id=node_id,
            executable=executable,
            sglang_source_directory=source_directory,
            model_path=model_path,
            gpu_uuid=gpu_uuid,
            cpu_cores=cpu_cores,
            memory_nodes=memory_nodes,
            service_endpoint=Host(ip=host_ip, port=rank_ports[world_rank]),
            socket_interface=socket_interface,
            hca_devices=hca_devices,
        )
        for world_rank, (
            node_id,
            executable,
            source_directory,
            model_path,
            gpu_uuid,
            cpu_cores,
            memory_nodes,
            host_ip,
            socket_interface,
        ) in enumerate(rank_inputs)
    )
    return Glm47NativeTpEpPlan(
        mode=mode,
        model_id=GLM_4_7_FLASH_BF16_MODEL_ID,
        model_revision=GLM_4_7_FLASH_BF16_MODEL_REVISION,
        model_config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        sglang_revision=GLM_4_7_FLASH_SGLANG_REVISION,
        distributed_coordinator=Host(ip=dwagon_ip, port=distributed_port),
        ranks=ranks,
    )


def build_native_glm47_process_specs(
    plan: Glm47NativeTpEpPlan,
) -> tuple[Glm47NativeProcessSpec, Glm47NativeProcessSpec, Glm47NativeProcessSpec]:
    specs = tuple(
        Glm47NativeProcessSpec(plan=plan, world_rank=rank.world_rank)
        for rank in plan.ranks
    )
    if len(specs) != 3:
        raise Glm47NativeParallelismError("native GLM-4.7 requires three ranks")
    return specs


def calculate_native_glm47_process_spec_sha256(
    process_spec: Glm47NativeProcessSpec,
) -> str:
    payload = {
        "canonicalization": GLM_4_7_FLASH_NATIVE_PROCESS_SPEC_CANONICALIZATION,
        "process_spec": process_spec.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()


def _server_argument_options(module: ast.Module) -> frozenset[str]:
    options: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
            continue
        for argument in node.args:
            if (
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            ):
                options.add(argument.value)
    return frozenset(options)


def _literal_string_sequence_assignment(
    module: ast.Module, name: str
) -> tuple[str, ...]:
    for node in module.body:
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name != name or not isinstance(value, (ast.List, ast.Tuple)):
            continue
        values: list[str] = []
        for element in value.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value, str
            ):
                return ()
            values.append(element.value)
        return tuple(values)
    return ()


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _expert_location_method_is_exact(model_class: ast.ClassDef) -> bool | None:
    method = next(
        (
            node
            for node in model_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_model_config_for_expert_location"
        ),
        None,
    )
    if method is None:
        return None
    if not isinstance(method, ast.FunctionDef):
        return False
    if not any(
        _name(decorator) == "classmethod" for decorator in method.decorator_list
    ):
        return False
    if tuple(argument.arg for argument in method.args.args) != ("cls", "config"):
        return False
    returned_calls = tuple(
        node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    )
    if (
        len(returned_calls) != 1
        or _name(returned_calls[0].func) != "ModelConfigForExpertLocation"
    ):
        return False
    returned_call = returned_calls[0]
    if len(returned_call.keywords) != 3 or returned_call.args:
        return False
    expected_attributes = {
        "num_layers": "num_hidden_layers",
        "num_logical_experts": "n_routed_experts",
        "num_groups": "n_group",
    }
    observed_attributes: dict[str, str] = {}
    for keyword in returned_call.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Attribute):
            return False
        if not (
            isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == "config"
        ):
            return False
        observed_attributes[keyword.arg] = keyword.value.attr
    return observed_attributes == expected_attributes


def _glm_lite_source_support(
    module: ast.Module, base_model_module: ast.Module
) -> tuple[
    bool,
    bool,
    Literal["glm_lite_override", "deepseek_inherited", "unsupported"],
]:
    model_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Glm4MoeLiteForCausalLM"
        ),
        None,
    )
    if model_class is None:
        return False, False, "unsupported"
    inherits_deepseek = any(
        _name(base) == "DeepseekV2ForCausalLM" for base in model_class.bases
    )
    override_is_exact = _expert_location_method_is_exact(model_class)
    if override_is_exact is not None:
        return (
            inherits_deepseek,
            override_is_exact,
            "glm_lite_override" if override_is_exact else "unsupported",
        )
    base_model_class = next(
        (
            node
            for node in base_model_module.body
            if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2ForCausalLM"
        ),
        None,
    )
    inherited_is_exact = (
        inherits_deepseek
        and base_model_class is not None
        and _expert_location_method_is_exact(base_model_class) is True
    )
    return (
        inherits_deepseek,
        inherited_is_exact,
        "deepseek_inherited" if inherited_is_exact else "unsupported",
    )


def _validate_model_config(contents: bytes) -> tuple[bool, str]:
    digest = hashlib.sha256(contents).hexdigest()
    try:
        value = cast(object, json.loads(contents))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, digest
    if not isinstance(value, dict):
        return False, digest
    config = cast(dict[str, object], value)
    expected = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "model_type": "glm4_moe_lite",
        "num_hidden_layers": 47,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "n_routed_experts": 64,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "n_group": 1,
        "dtype": "bfloat16",
    }
    return digest == GLM_4_7_FLASH_BF16_CONFIG_SHA256 and all(
        config.get(name) == expected_value for name, expected_value in expected.items()
    ), digest


def assess_native_glm47_source_support(
    *,
    node_id: NodeId,
    source_directory: str,
    model_config_path: str,
    runtime_executable: str,
    sglang_revision: str,
    source_tree_clean: bool,
    server_args_contents: bytes,
    model_implementation_contents: bytes,
    base_model_implementation_contents: bytes,
    runtime_server_args_contents: bytes,
    runtime_model_implementation_contents: bytes,
    runtime_base_model_implementation_contents: bytes,
    model_config_contents: bytes,
) -> Glm47NativeSourceSupport:
    blockers: list[str] = []
    try:
        server_args_module = ast.parse(server_args_contents)
    except (SyntaxError, ValueError):
        server_args_module = ast.Module(body=[], type_ignores=[])
        blockers.append("server_args.py is not parseable Python")
    try:
        model_module = ast.parse(model_implementation_contents)
    except (SyntaxError, ValueError):
        model_module = ast.Module(body=[], type_ignores=[])
        blockers.append("glm4_moe_lite.py is not parseable Python")
    try:
        base_model_module = ast.parse(base_model_implementation_contents)
    except (SyntaxError, ValueError):
        base_model_module = ast.Module(body=[], type_ignores=[])
        blockers.append("deepseek_v2.py is not parseable Python")

    parser_options = _server_argument_options(server_args_module)
    missing_options = sorted(_REQUIRED_SERVER_OPTIONS - parser_options)
    if missing_options:
        blockers.append(f"SGLang parser is missing options {missing_options}")
    a2a_choices = _literal_string_sequence_assignment(
        server_args_module, "MOE_A2A_BACKEND_CHOICES"
    )
    runner_choices = _literal_string_sequence_assignment(
        server_args_module, "MOE_RUNNER_BACKEND_CHOICES"
    )
    if "none" not in a2a_choices:
        blockers.append("SGLang does not admit the host-staged none MoE A2A backend")
    if "triton" not in runner_choices:
        blockers.append("SGLang does not admit the Triton MoE runner backend")

    inherits_deepseek, hook_supported, hook_source = _glm_lite_source_support(
        model_module, base_model_module
    )
    if not inherits_deepseek:
        blockers.append(
            "Glm4MoeLiteForCausalLM is absent or lacks the admitted DeepSeek base"
        )
    model_config_valid, model_config_sha256 = _validate_model_config(
        model_config_contents
    )
    if not model_config_valid:
        blockers.append("GLM-4.7 model config is not the exact admitted snapshot")
    if sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        blockers.append(
            f"SGLang revision is {sglang_revision}, expected "
            f"{GLM_4_7_FLASH_SGLANG_REVISION}"
        )
    if not source_tree_clean:
        blockers.append("SGLang source tree has modifications or untracked files")
    runtime_matches_source = (
        runtime_server_args_contents == server_args_contents
        and runtime_model_implementation_contents == model_implementation_contents
        and runtime_base_model_implementation_contents
        == base_model_implementation_contents
    )
    if not runtime_matches_source:
        blockers.append("installed SGLang runtime modules do not match pinned source")

    admitted_modes: list[Glm47NativeParallelism] = []
    if not blockers:
        admitted_modes.append("tp3_ep1")
        if hook_supported:
            admitted_modes.append("tp3_ep3")
    if not hook_supported:
        blockers.append(_EP3_EXPERT_LOCATION_BLOCKER)
    return Glm47NativeSourceSupport(
        node_id=node_id,
        source_directory=source_directory,
        model_config_path=model_config_path,
        runtime_executable=runtime_executable,
        sglang_revision=sglang_revision,
        source_tree_clean=source_tree_clean,
        server_args_sha256=hashlib.sha256(server_args_contents).hexdigest(),
        model_implementation_sha256=hashlib.sha256(
            model_implementation_contents
        ).hexdigest(),
        base_model_implementation_sha256=hashlib.sha256(
            base_model_implementation_contents
        ).hexdigest(),
        runtime_server_args_sha256=hashlib.sha256(
            runtime_server_args_contents
        ).hexdigest(),
        runtime_model_implementation_sha256=hashlib.sha256(
            runtime_model_implementation_contents
        ).hexdigest(),
        runtime_base_model_implementation_sha256=hashlib.sha256(
            runtime_base_model_implementation_contents
        ).hexdigest(),
        runtime_matches_source=runtime_matches_source,
        model_config_sha256=model_config_sha256,
        parser_options=tuple(sorted(parser_options)),
        moe_a2a_backend_choices=a2a_choices,
        moe_runner_backend_choices=runner_choices,
        glm_lite_inherits_deepseek=inherits_deepseek,
        expert_location_hook_supported=hook_supported,
        expert_location_hook_source=hook_source,
        admitted_modes=tuple(admitted_modes),
        blockers=tuple(blockers),
    )


def inspect_native_glm47_source_support(
    *,
    node_id: NodeId,
    source_directory: Path,
    model_config_path: Path,
    runtime_executable: Path,
) -> Glm47NativeSourceSupport:
    """Inspect one local clean source checkout without importing its runtime."""

    revision_result = subprocess.run(
        ("git", "-C", str(source_directory), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        (
            "git",
            "-C",
            str(source_directory),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if revision_result.returncode != 0 or status_result.returncode != 0:
        detail = (
            revision_result.stderr.strip()
            or status_result.stderr.strip()
            or "git source inspection failed"
        )
        raise Glm47NativeParallelismError(detail)
    server_args_path = source_directory / "python/sglang/srt/server_args.py"
    model_implementation_path = (
        source_directory / "python/sglang/srt/models/glm4_moe_lite.py"
    )
    base_model_implementation_path = (
        source_directory / "python/sglang/srt/models/deepseek_v2.py"
    )
    if not runtime_executable.is_file() or not os.access(runtime_executable, os.X_OK):
        raise Glm47NativeParallelismError(
            f"native GLM-4.7 runtime is not executable: {runtime_executable}"
        )
    runtime_site_packages = tuple(
        (runtime_executable.parent.parent / "lib").glob("python*/site-packages")
    )
    if len(runtime_site_packages) != 1:
        raise Glm47NativeParallelismError(
            "native GLM-4.7 runtime must contain exactly one Python site-packages"
        )
    runtime_server_args_path = runtime_site_packages[0] / "sglang/srt/server_args.py"
    runtime_model_implementation_path = (
        runtime_site_packages[0] / "sglang/srt/models/glm4_moe_lite.py"
    )
    runtime_base_model_implementation_path = (
        runtime_site_packages[0] / "sglang/srt/models/deepseek_v2.py"
    )
    try:
        server_args_contents = server_args_path.read_bytes()
        model_implementation_contents = model_implementation_path.read_bytes()
        base_model_implementation_contents = base_model_implementation_path.read_bytes()
        runtime_server_args_contents = runtime_server_args_path.read_bytes()
        runtime_model_implementation_contents = (
            runtime_model_implementation_path.read_bytes()
        )
        runtime_base_model_implementation_contents = (
            runtime_base_model_implementation_path.read_bytes()
        )
        model_config_contents = model_config_path.read_bytes()
    except OSError as error:
        raise Glm47NativeParallelismError(
            f"native GLM-4.7 support input cannot be read: {error}"
        ) from error
    return assess_native_glm47_source_support(
        node_id=node_id,
        source_directory=str(source_directory),
        model_config_path=str(model_config_path),
        runtime_executable=str(runtime_executable),
        sglang_revision=revision_result.stdout.strip().lower(),
        source_tree_clean=not status_result.stdout.strip(),
        server_args_contents=server_args_contents,
        model_implementation_contents=model_implementation_contents,
        base_model_implementation_contents=base_model_implementation_contents,
        runtime_server_args_contents=runtime_server_args_contents,
        runtime_model_implementation_contents=runtime_model_implementation_contents,
        runtime_base_model_implementation_contents=(
            runtime_base_model_implementation_contents
        ),
        model_config_contents=model_config_contents,
    )


def build_native_glm47_benchmark_protocol() -> Glm47NativeBenchmarkProtocol:
    return Glm47NativeBenchmarkProtocol(
        protocol_version="glm47-native-tp3-ep-diagnostic-v1",
        execution_order=(
            "semantic_sanity",
            "hca_counters_before",
            "prefill_workload",
            "decode_workload",
            "hca_counters_after",
            "ownership_verified_cleanup",
        ),
        sanity_required=True,
        sanity_marker=GLM_4_7_FLASH_SANITY_MARKER,
        sanity_input_tokens=GLM_4_7_FLASH_SANITY_INPUT_TOKENS,
        sanity_max_new_tokens=GLM_4_7_FLASH_SANITY_MAX_NEW_TOKENS,
        sanity_input_ids_sha256=GLM_4_7_FLASH_SANITY_INPUT_IDS_SHA256,
        sanity_chat_template_sha256=GLM_4_7_FLASH_SANITY_CHAT_TEMPLATE_SHA256,
        sanity_rendered_prompt_sha256=GLM_4_7_FLASH_SANITY_RENDERED_PROMPT_SHA256,
        sampling_seed=GLM_4_7_FLASH_SERVING_SAMPLING_SEED,
        workloads=(
            Glm47NativeBenchmarkWorkloadProtocol(
                kind="prefill",
                input_tokens=GLM_4_7_FLASH_PREFILL_INPUT_TOKENS,
                input_ids_sha256=GLM_4_7_FLASH_PREFILL_INPUT_IDS_SHA256,
                output_tokens=GLM_4_7_FLASH_PREFILL_OUTPUT_TOKENS,
                warmup_count=WARM_SERVING_MINIMUM_WARMUPS,
                sample_count=WARM_SERVING_MINIMUM_SAMPLES,
            ),
            Glm47NativeBenchmarkWorkloadProtocol(
                kind="decode",
                input_tokens=GLM_4_7_FLASH_DECODE_INPUT_TOKENS,
                input_ids_sha256=GLM_4_7_FLASH_DECODE_INPUT_IDS_SHA256,
                output_tokens=GLM_4_7_FLASH_DECODE_OUTPUT_TOKENS,
                warmup_count=WARM_SERVING_MINIMUM_WARMUPS,
                sample_count=WARM_SERVING_MINIMUM_SAMPLES,
            ),
        ),
    )


def _validate_native_glm47_support_bindings(
    plan: Glm47NativeTpEpPlan,
    support: tuple[Glm47NativeSourceSupport, Glm47NativeSourceSupport],
) -> None:
    expected_nodes = (DWAGON_NODE_ID, FWUFF_NODE_ID)
    if tuple(evidence.node_id for evidence in support) != expected_nodes:
        raise Glm47NativeParallelismError(
            "support evidence must be ordered as dwagon then fwuff"
        )
    for node_id, evidence in zip(expected_nodes, support, strict=True):
        node_ranks = tuple(rank for rank in plan.ranks if rank.node_id == node_id)
        expected_source_directories = {
            rank.sglang_source_directory for rank in node_ranks
        }
        expected_runtime_executables = {rank.executable for rank in node_ranks}
        expected_config_paths = {
            str(PurePosixPath(rank.model_path) / "config.json") for rank in node_ranks
        }
        if expected_source_directories != {evidence.source_directory}:
            raise Glm47NativeParallelismError(
                f"{node_id} support evidence does not bind the launch source path"
            )
        if expected_runtime_executables != {evidence.runtime_executable}:
            raise Glm47NativeParallelismError(
                f"{node_id} support evidence does not bind the launch executable"
            )
        if expected_config_paths != {evidence.model_config_path}:
            raise Glm47NativeParallelismError(
                f"{node_id} support evidence does not bind the launch model config"
            )
        if evidence.sglang_revision != plan.sglang_revision:
            raise Glm47NativeParallelismError(
                f"{node_id} support evidence does not bind the launch revision"
            )
        evidence.require_mode(plan.mode)


def _validate_native_glm47_model_snapshot_bindings(
    plan: Glm47NativeTpEpPlan,
    snapshots: tuple[SglangKtVerifiedModelSnapshot, SglangKtVerifiedModelSnapshot],
) -> None:
    expected_nodes = (DWAGON_NODE_ID, FWUFF_NODE_ID)
    for node_id, snapshot in zip(expected_nodes, snapshots, strict=True):
        expected_paths = {
            rank.model_path for rank in plan.ranks if rank.node_id == node_id
        }
        if expected_paths != {snapshot.model_path}:
            raise Glm47NativeParallelismError(
                f"{node_id} verified snapshot does not bind the launch model path"
            )
        if (
            snapshot.model_id != plan.model_id
            or snapshot.revision != plan.model_revision
            or snapshot.weight_format != "safetensors"
            or snapshot.ktransformers_method != "BF16"
            or snapshot.full_indexer_layer_starts != (0,)
            or snapshot.contract_receipt_sha256
            != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
            or snapshot.contract_sha256 != GLM_4_7_FLASH_BF16_MODEL_CONTRACT_SHA256
            or snapshot.config_sha256 != plan.model_config_sha256
            or snapshot.index_sha256 != GLM_4_7_FLASH_NATIVE_INDEX_SHA256
            or snapshot.weight_map_entries != GLM_4_7_FLASH_NATIVE_WEIGHT_MAP_ENTRIES
            or snapshot.shard_count != GLM_4_7_FLASH_NATIVE_SHARD_COUNT
            or snapshot.physical_weight_bytes
            != GLM_4_7_FLASH_NATIVE_PHYSICAL_WEIGHT_BYTES
        ):
            raise Glm47NativeParallelismError(
                f"{node_id} verified snapshot is not the pinned GLM-4.7 BF16 contract"
            )


def build_native_glm47_planned_diagnostic_receipt(
    plan: Glm47NativeTpEpPlan,
    support_by_node: Mapping[NodeId, Glm47NativeSourceSupport],
    model_snapshot_by_node: Mapping[NodeId, SglangKtVerifiedModelSnapshot],
) -> Glm47NativePlannedDiagnosticReceipt:
    expected_nodes = (DWAGON_NODE_ID, FWUFF_NODE_ID)
    if frozenset(support_by_node) != frozenset(expected_nodes):
        raise Glm47NativeParallelismError(
            "support evidence must contain exactly dwagon and fwuff"
        )
    if frozenset(model_snapshot_by_node) != frozenset(expected_nodes):
        raise Glm47NativeParallelismError(
            "model snapshot evidence must contain exactly dwagon and fwuff"
        )
    support = tuple(support_by_node[node_id] for node_id in expected_nodes)
    typed_support = cast(
        tuple[Glm47NativeSourceSupport, Glm47NativeSourceSupport], support
    )
    _validate_native_glm47_support_bindings(plan, typed_support)
    model_snapshots = cast(
        tuple[SglangKtVerifiedModelSnapshot, SglangKtVerifiedModelSnapshot],
        tuple(model_snapshot_by_node[node_id] for node_id in expected_nodes),
    )
    _validate_native_glm47_model_snapshot_bindings(plan, model_snapshots)
    specs = build_native_glm47_process_specs(plan)
    return Glm47NativePlannedDiagnosticReceipt(
        schema_version=1,
        kind="glm47_flash_native_tp3_ep_planned_diagnostic",
        status="ready",
        performance_comparable=False,
        profiler="none",
        plan=plan,
        process_specs=specs,
        source_support=typed_support,
        model_snapshots=model_snapshots,
        benchmark_protocol=build_native_glm47_benchmark_protocol(),
    )


def calculate_native_glm47_planned_receipt_sha256(
    receipt: Glm47NativePlannedDiagnosticReceipt,
) -> str:
    payload = {
        "canonicalization": GLM_4_7_FLASH_NATIVE_PLANNED_RECEIPT_CANONICALIZATION,
        "receipt": receipt.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_sglang_kt_json(payload)).hexdigest()
