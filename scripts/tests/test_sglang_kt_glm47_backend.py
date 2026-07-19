from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import struct
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from scripts import sglang_kt_glm47_backend as backend
from scripts.sglang_kt_glm47_reference import (
    GLM47_BF16_DTYPE,
    GLM47_FLOAT32_DTYPE,
    Glm47LayerOneReferenceOutputs,
    Glm47TensorDtype,
    Glm47TensorEvidenceBackend,
    Glm47TensorSnapshot,
)
from scripts.sglang_kt_glm47_trace import (
    GPU_METHOD_APPLY,
    MODEL_FORWARD,
    SGLANG_CPU_SYNC,
    TraceProbe,
)

MODEL_PATH = "/mnt/sanic/models/glm-4.7-flash-bf16"
SERVICE_ENDPOINT_IP = "127.0.0.1"
SERVICE_ENDPOINT_PORT = 29_500
DISTRIBUTED_COORDINATOR_IP = "127.0.0.1"
DISTRIBUTED_COORDINATOR_PORT = 29_501


def test_import_does_not_require_torch_or_sglang() -> None:
    source = Path(backend.__file__).resolve()
    repository = source.parents[1]
    program = f"""
import importlib.abc
import importlib.util
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {{"torch", "sglang"}} or fullname.startswith(("torch.", "sglang.")):
            raise ModuleNotFoundError(f"{{fullname}} is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockRuntime())
sys.path.insert(0, {str(repository)!r})
spec = importlib.util.spec_from_file_location("isolated_glm47_backend", {str(source)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert "sglang" not in sys.modules
"""
    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


class FakeTensor:
    def __init__(
        self,
        value: float,
        shape: tuple[int, ...],
        dtype: str,
        *,
        integer_values: tuple[int, ...] | None = None,
        device: str = "cpu",
    ) -> None:
        self.value = value
        self.shape = shape
        self.dtype = dtype
        self.integer_values = integer_values
        self.device = device
        self.clone_count = 0

    def detach(self) -> FakeTensor:
        return self

    def clone(self) -> FakeTensor:
        self.clone_count += 1
        return FakeTensor(
            self.value,
            self.shape,
            self.dtype,
            integer_values=self.integer_values,
            device=self.device,
        )

    def cpu(self) -> FakeTensor:
        return self.to(device="cpu")

    def reshape(self, *shape: int) -> FakeTensor:
        assert shape == (-1,)
        element_count = (
            len(self.integer_values)
            if self.integer_values is not None
            else math.prod(self.shape)
        )
        return FakeTensor(
            self.value,
            (element_count,),
            self.dtype,
            integer_values=self.integer_values,
            device=self.device,
        )

    def tolist(self) -> list[int] | list[float]:
        if self.integer_values is not None:
            return list(self.integer_values)
        return [self.value] * math.prod(self.shape)

    def to(
        self,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> FakeTensor:
        return FakeTensor(
            self.value,
            self.shape,
            self.dtype if dtype is None else str(dtype),
            integer_values=self.integer_values,
            device=self.device if device is None else str(device),
        )

    def argmax(self, *, dim: int) -> FakeTensor:
        assert dim == -1
        return FakeTensor(
            42.0,
            (1,),
            "torch.long",
            integer_values=(42,),
            device=self.device,
        )


def _snapshot_bytes(tensor: FakeTensor) -> bytes:
    element_count = math.prod(tensor.shape)
    if tensor.dtype == GLM47_FLOAT32_DTYPE:
        return struct.pack("<f", tensor.value) * element_count
    if tensor.dtype == GLM47_BF16_DTYPE:
        float_bits = cast(int, struct.unpack("<I", struct.pack("<f", tensor.value))[0])
        return struct.pack("<H", float_bits >> 16) * element_count
    raise AssertionError(f"unsupported fake snapshot dtype {tensor.dtype}")


class FakeTensorEvidenceBackend(Glm47TensorEvidenceBackend[object]):
    def snapshot(
        self,
        tensor: object,
        *,
        expected_dtype: Glm47TensorDtype,
        expected_shape: tuple[int, ...],
    ) -> Glm47TensorSnapshot:
        assert isinstance(tensor, FakeTensor)
        assert tensor.dtype == expected_dtype
        assert tensor.shape == expected_shape
        return Glm47TensorSnapshot(
            dtype=expected_dtype,
            shape=expected_shape,
            little_endian_contiguous_bytes=_snapshot_bytes(tensor),
        )

    def merge_bfloat16(self, left: object, right: object) -> object:
        assert isinstance(left, FakeTensor)
        assert isinstance(right, FakeTensor)
        return FakeTensor(
            left.value + right.value,
            (1, 2_048),
            GLM47_BF16_DTYPE,
        )

    def merge_float32(self, left: object, right: object) -> object:
        assert isinstance(left, FakeTensor)
        assert isinstance(right, FakeTensor)
        return FakeTensor(
            left.value + right.value,
            (1, 2_048),
            GLM47_FLOAT32_DTYPE,
        )


class FakeMask:
    def __init__(self, rows: tuple[tuple[bool, ...], ...]) -> None:
        self.rows = rows

    def cpu(self) -> FakeMask:
        return self

    def tolist(self) -> list[list[bool]]:
        return [list(row) for row in self.rows]


class FakeGenerator:
    def __init__(self, *, device: str) -> None:
        assert device == "cpu"
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeCuda:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def synchronize(self) -> None:
        self.events.append(("cuda_synchronize",))


class FakeInferenceMode:
    def __init__(self, torch: FakeTorch) -> None:
        self.torch = torch
        self.previous_grad_enabled: bool | None = None

    def __enter__(self) -> None:
        assert self.previous_grad_enabled is None
        self.previous_grad_enabled = self.torch.grad_enabled
        self.torch.grad_enabled = False
        self.torch.events.append(("inference_mode_enter",))

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        assert self.previous_grad_enabled is not None
        self.torch.grad_enabled = self.previous_grad_enabled
        self.torch.events.append(("inference_mode_exit",))


class FakeTorch:
    bfloat16 = GLM47_BF16_DTYPE
    float32 = GLM47_FLOAT32_DTYPE
    long = "torch.long"

    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.cuda = FakeCuda(events)
        self.grad_enabled = True
        self.execution_grad_states: list[tuple[str, bool]] = []

    def inference_mode(self) -> FakeInferenceMode:
        return FakeInferenceMode(self)

    def require_grad_disabled(self, operation: str) -> None:
        self.execution_grad_states.append((operation, self.grad_enabled))
        assert not self.grad_enabled

    def Generator(self, *, device: str) -> FakeGenerator:  # noqa: N802
        self.events.append(("generator", device))
        return FakeGenerator(device=device)

    def randn(
        self,
        shape: tuple[int, ...],
        *,
        generator: FakeGenerator,
        dtype: object,
        device: str,
    ) -> FakeTensor:
        assert generator.seed == backend.GLM47_LAYER_ONE_RANDOM_SEED
        self.events.append(("randn", shape, str(dtype), device))
        return FakeTensor(0.5, shape, str(dtype), device=device)

    def tensor(
        self,
        values: tuple[tuple[int, ...], ...] | tuple[tuple[float, ...], ...],
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        flat = values[0]
        integers = tuple(flat) if str(dtype) == "torch.long" else None
        return FakeTensor(
            float(flat[0]),
            (1, len(flat)),
            str(dtype),
            integer_values=cast(tuple[int, ...] | None, integers),
            device=str(device),
        )

    def zeros(
        self,
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        return FakeTensor(0.0, shape, str(dtype), device=str(device))


class FakeTopKOutput:
    def __init__(self, **values: object) -> None:
        vars(self).update(values)


class FakeDispatchOutput:
    def __init__(
        self,
        *,
        hidden_states: FakeTensor,
        hidden_states_scale: object | None = None,
        topk_output: object | None = None,
    ) -> None:
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.topk_output = topk_output


class FakeHiddenStates:
    def __init__(self, hidden_states: FakeTensor) -> None:
        self.hidden_states = hidden_states


class AMXBF16_MOE:  # noqa: N801 - pinned runtime class identity
    pass


class NativeMoEWrapper:
    def __init__(self, cpu_value: float) -> None:
        self.moe = AMXBF16_MOE()
        self.cpu_value = cpu_value

    def submit_forward(self, *_args: object, **_kwargs: object) -> None:
        return None

    def sync_forward(self, *_args: object, **_kwargs: object) -> FakeTensor:
        return FakeTensor(self.cpu_value, (1, 2_048), GLM47_BF16_DTYPE)


class FakeGpuMethod:
    def __init__(self, gpu_value: float, torch: FakeTorch) -> None:
        self.gpu_value = gpu_value
        self.torch = torch

    def apply(self, *_args: object, **_kwargs: object) -> FakeHiddenStates:
        self.torch.require_grad_disabled("gpu_apply")
        return FakeHiddenStates(
            FakeTensor(
                self.gpu_value,
                (1, 2_048),
                GLM47_BF16_DTYPE,
                device="cuda:0",
            )
        )


class KTEPWrapperMethod:
    def __init__(
        self,
        layer_id: int,
        resident_count: int,
        torch: FakeTorch,
        *,
        fail_forward: bool,
    ) -> None:
        self.layer_id = layer_id
        self.num_gpu_experts = resident_count
        self._quant_wrapper_id = "kt_ep"
        self.kt_config = SimpleNamespace(layer_idx=layer_id)
        self.wrapper = NativeMoEWrapper(1.0 if resident_count else 3.0)
        self.gpu_method = FakeGpuMethod(2.0, torch)
        self.torch = torch
        self.fail_forward = fail_forward

    def _submit_cpu_forward(self, *args: object) -> None:
        self.wrapper.submit_forward(*args)

    def _sync_cpu_forward(self, *args: object) -> FakeTensor:
        return self.wrapper.sync_forward(*args)

    def apply(
        self,
        _expert_module: object,
        dispatch_output: FakeDispatchOutput,
    ) -> FakeHiddenStates:
        self.torch.require_grad_disabled("ktep_apply")
        if self.fail_forward:
            raise RuntimeError("injected forward failure")
        if self.layer_id != 1:
            return FakeHiddenStates(dispatch_output.hidden_states)
        handle = self._submit_cpu_forward(dispatch_output)
        gpu_output = (
            self.gpu_method.apply(dispatch_output) if self.num_gpu_experts > 0 else None
        )
        cpu_output = self._sync_cpu_forward(handle)
        value = cpu_output.value + (
            0.0 if gpu_output is None else gpu_output.hidden_states.value
        )
        return FakeHiddenStates(
            FakeTensor(value, (1, 2_048), GLM47_BF16_DTYPE, device="cuda:0")
        )


KTEPWrapperMethod.__module__ = "sglang.srt.layers.moe.kt_ep_wrapper"


class FakeExperts:
    def __init__(
        self,
        layer_id: int,
        resident_count: int,
        torch: FakeTorch,
        *,
        fail_forward: bool,
    ) -> None:
        self._registry_prefix = f"model.layers.{layer_id}.mlp.experts"
        self.layer_id = layer_id
        self.moe_runner_config = SimpleNamespace(layer_id=layer_id)
        self.quant_method = KTEPWrapperMethod(
            layer_id,
            resident_count,
            torch,
            fail_forward=fail_forward,
        )


class FakeLayer:
    def __init__(
        self,
        layer_id: int,
        resident_count: int,
        torch: FakeTorch,
        *,
        fail_forward: bool,
    ) -> None:
        self.mlp = SimpleNamespace(
            experts=FakeExperts(
                layer_id,
                resident_count,
                torch,
                fail_forward=fail_forward,
            )
        )


def _mask_rows(resident_ids: tuple[int, ...]) -> tuple[tuple[bool, ...], ...]:
    row_zero = (True,) * 64
    routed_row = tuple(expert_id in resident_ids for expert_id in range(64))
    return (row_zero, *(routed_row for _ in range(46)))


def _mask_sha256(rows: tuple[tuple[bool, ...], ...]) -> str:
    return hashlib.sha256(
        bytes(int(value) for row in rows for value in row)
    ).hexdigest()


class FakeLogitsProcessorOutput:
    def __init__(self) -> None:
        self.next_token_logits = FakeTensor(
            0.25,
            (1, 154_880),
            GLM47_FLOAT32_DTYPE,
            device="cuda:0",
        )
        self.hidden_states: FakeTensor | None = None


class FakeCausalModel:
    def __init__(
        self,
        resident_ids: tuple[int, ...],
        torch: FakeTorch,
        *,
        fail_forward: bool,
        bad_coverage: bool,
    ) -> None:
        resident_count = len(resident_ids)
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(),
                *(
                    FakeLayer(
                        layer_id,
                        resident_count,
                        torch,
                        fail_forward=fail_forward,
                    )
                    for layer_id in range(1, 47)
                ),
            ]
        )
        rows = _mask_rows(resident_ids)
        layers = [
            {
                "layer_id": layer_id,
                "module_path": f"model.layers.{layer_id}.mlp.experts",
                "wrapper_id": "kt_ep",
                "gpu_resident_experts": resident_count,
            }
            for layer_id in range(1, 47)
        ]
        if bad_coverage:
            layers[-1] = {**layers[-1], "gpu_resident_experts": 63}
        self.kt_ep_coverage_receipt = {
            "schema_version": 1,
            "model_architecture": "Glm4MoeLiteForCausalLM",
            "pipeline_parallel_size": 1,
            "layer_count": 47,
            "routed_layer_ids": list(range(1, 47)),
            "routed_expert_count": 64,
            "gpu_expert_mask_shape": [47, 64],
            "gpu_expert_mask_sha256": _mask_sha256(rows),
            "configured_gpu_resident_experts_per_layer": resident_count,
            "ktransformers_method": "BF16",
            "wrapper_id": "kt_ep",
            "layers": layers,
        }
        self.torch = torch

    def forward(self, _forward_batch: object) -> FakeLogitsProcessorOutput:
        self.torch.require_grad_disabled("causal_model_forward")
        hidden = FakeTensor(0.5, (1, 2_048), GLM47_BF16_DTYPE, device="cuda:0")
        for layer in self.model.layers[1:]:
            experts = layer.mlp.experts
            result = experts.quant_method.apply(
                experts,
                FakeDispatchOutput(hidden_states=hidden),
            )
            hidden = result.hidden_states
        return FakeLogitsProcessorOutput()


class FakeRunnerOutput:
    def __init__(self, logits_output: FakeLogitsProcessorOutput) -> None:
        self.logits_output = logits_output


class FakeModelRunner:
    def __init__(
        self,
        server_args: object,
        model_config: object,
        resident_ids: tuple[int, ...],
        torch: FakeTorch,
        *,
        fail_forward: bool,
        bad_coverage: bool,
    ) -> None:
        self.server_args = server_args
        self.model_config = model_config
        self.device = "cuda:0"
        self.req_to_token_pool = object()
        self.token_to_kv_pool_allocator = object()
        self.model = FakeCausalModel(
            resident_ids,
            torch,
            fail_forward=fail_forward,
            bad_coverage=bad_coverage,
        )
        self.torch = torch

    def forward(self, forward_batch: object) -> FakeRunnerOutput:
        self.torch.require_grad_disabled("model_runner_forward")
        return FakeRunnerOutput(self.model.forward(forward_batch))

    def sample(
        self, logits_output: FakeLogitsProcessorOutput, _batch: object
    ) -> FakeTensor:
        return logits_output.next_token_logits.argmax(dim=-1)


class FakeServerArgs:
    def __init__(self, namespace: argparse.Namespace) -> None:
        self.tp_size = 1
        self.pp_size = 1
        self.ep_size = 1
        self.nnodes = 1
        self.node_rank = 0
        self.dp_size = 1
        self.kt_method = "BF16"
        self.kt_expert_placement_strategy = "uniform"
        self.kt_max_deferred_experts_per_token = 0
        self.disable_cuda_graph = True
        self.disable_shared_experts_fusion = True
        self.kt_num_gpu_experts = namespace.resident_count
        self.model_path = namespace.model_path
        self.kt_weight_path = namespace.kt_weight_path
        self.mem_fraction_static = 0.8
        self.tokenizer_path = namespace.model_path
        self.tokenizer_mode = "auto"
        self.trust_remote_code = True
        self.page_size = 1
        self.host = namespace.host
        self.port = namespace.port
        self.dist_init_addr = namespace.dist_init_addr
        self.nccl_port: int | None = None


class FakeServerArgsType:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        resident_count: int,
        *,
        fail_parse: bool,
    ) -> None:
        self.events = events
        self.resident_count = resident_count
        self.fail_parse = fail_parse

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        self.events.append(("server_add_cli_args",))
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--kt-weight-path", required=True)
        parser.add_argument("--argument-parity-marker", required=True)
        parser.add_argument("--host", required=True)
        parser.add_argument("--port", required=True, type=int)
        parser.add_argument("--dist-init-addr", required=True)

    def from_cli_args(self, namespace: argparse.Namespace) -> FakeServerArgs:
        self.events.append(
            (
                "server_from_cli_args",
                namespace.model_path,
                namespace.kt_weight_path,
                namespace.argument_parity_marker,
            )
        )
        if self.fail_parse:
            raise RuntimeError("injected parse failure")
        namespace.resident_count = self.resident_count
        return FakeServerArgs(namespace)


class FakePortArgsType:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        return_mismatched_port: bool,
    ) -> None:
        self.events = events
        self.return_mismatched_port = return_mismatched_port

    def init_new(self, server_args: FakeServerArgs) -> object:
        self.events.append(("port_args",))
        assert server_args.nccl_port is not None
        port = server_args.nccl_port
        if self.return_mismatched_port:
            port += 1
        return SimpleNamespace(nccl_port=port)


class FakeModelConfigType:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_model_config: bool,
    ) -> None:
        self.events = events
        self.fail_model_config = fail_model_config

    def from_server_args(self, _server_args: object) -> object:
        self.events.append(("model_config",))
        if self.fail_model_config:
            raise RuntimeError("injected model config failure")
        return SimpleNamespace(name="fake-glm47-config")


class FakeModelRunnerType:
    def __init__(self, environment: FakeEnvironment) -> None:
        self.environment = environment

    def __call__(self, **arguments: object) -> FakeModelRunner:
        self.environment.events.append(("model_runner",))
        self.environment.model_runner_arguments = arguments
        if self.environment.fail_stage == "model_load":
            raise RuntimeError("injected model load failure")
        model_config = arguments["model_config"]
        server_args = arguments["server_args"]
        runner = FakeModelRunner(
            server_args,
            model_config,
            self.environment.route.resident_gpu_expert_ids,
            self.environment.torch,
            fail_forward=self.environment.fail_stage == "forward",
            bad_coverage=self.environment.fail_stage == "coverage",
        )
        self.environment.runner = runner
        return runner


class FakeSamplingParams:
    def __init__(self, *, temperature: int, max_new_tokens: int) -> None:
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens


class FakeRequest:
    def __init__(
        self,
        *,
        rid: str,
        origin_input_text: str,
        origin_input_ids: list[int],
        sampling_params: FakeSamplingParams,
    ) -> None:
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids = origin_input_ids
        self.sampling_params = sampling_params
        self.fill_ids: list[int] = []
        self.logprob_start_len = 0
        self.prefix_indices: list[int] = []
        self.kv_committed_len = 0
        self.extend_input_len = 0

    def set_extend_input_len(self, length: int) -> None:
        self.extend_input_len = length


class FakeWorkerBatch:
    def __init__(self, input_ids: FakeTensor, positions: FakeTensor) -> None:
        self.input_ids = input_ids
        self.positions = positions


class FakeScheduleBatch:
    def __init__(self, request: FakeRequest) -> None:
        self.request = request
        self.output_ids: FakeTensor | None = None
        self.worker_batch: FakeWorkerBatch | None = None

    @classmethod
    def init_new(
        cls, *, reqs: list[FakeRequest], **_kwargs: object
    ) -> FakeScheduleBatch:
        assert len(reqs) == 1
        return cls(reqs[0])

    def prepare_for_extend(self) -> None:
        tokens = tuple(self.request.fill_ids)
        self.worker_batch = FakeWorkerBatch(
            FakeTensor(
                0.0,
                (len(tokens),),
                "torch.long",
                integer_values=tokens,
                device="cuda:0",
            ),
            FakeTensor(
                0.0,
                (len(tokens),),
                "torch.long",
                integer_values=tuple(range(len(tokens))),
                device="cuda:0",
            ),
        )
        self.request.kv_committed_len = len(tokens)

    def prepare_for_decode(self) -> None:
        assert self.output_ids is not None
        assert self.output_ids.integer_values is not None
        token = self.output_ids.integer_values[0]
        self.worker_batch = FakeWorkerBatch(
            FakeTensor(
                float(token),
                (1,),
                "torch.long",
                integer_values=(token,),
                device="cuda:0",
            ),
            FakeTensor(
                8.0,
                (1,),
                "torch.long",
                integer_values=(8,),
                device="cuda:0",
            ),
        )
        self.request.kv_committed_len = 9

    def get_model_worker_batch(self) -> FakeWorkerBatch:
        assert self.worker_batch is not None
        return self.worker_batch


class FakeForwardBatchType:
    @staticmethod
    def init_new(worker_batch: FakeWorkerBatch, _runner: object) -> FakeWorkerBatch:
        return worker_batch


@dataclass(frozen=True)
class FakeStage:
    resident_gpu_experts: int


@dataclass(frozen=True)
class FakeEndpoint:
    ip: str
    port: int


@dataclass(frozen=True)
class FakeProcessSpec:
    arguments: tuple[str, ...]
    model_path: str
    service_endpoint: FakeEndpoint
    distributed_coordinator: FakeEndpoint
    stage: FakeStage


@dataclass(frozen=True)
class FakeRoute:
    resident_gpu_expert_ids: tuple[int, ...]
    selected_expert_ids: tuple[int, ...]
    cpu_expert_ids: tuple[int, ...]
    gpu_expert_ids: tuple[int, ...]


def _route(resident_count: int) -> FakeRoute:
    assert resident_count in (0, 2)
    selected = (0, 1, 4, 5)
    resident = () if resident_count == 0 else (0, 1)
    return FakeRoute(
        resident_gpu_expert_ids=resident,
        selected_expert_ids=selected,
        cpu_expert_ids=tuple(item for item in selected if item not in resident),
        gpu_expert_ids=tuple(item for item in selected if item in resident),
    )


def _process_spec(resident_count: int) -> FakeProcessSpec:
    suffix = (
        "--model-path",
        MODEL_PATH,
        "--kt-weight-path",
        MODEL_PATH,
        "--argument-parity-marker",
        "exact-suffix",
        "--host",
        SERVICE_ENDPOINT_IP,
        "--port",
        str(SERVICE_ENDPOINT_PORT),
        "--dist-init-addr",
        f"{DISTRIBUTED_COORDINATOR_IP}:{DISTRIBUTED_COORDINATOR_PORT}",
    )
    return FakeProcessSpec(
        arguments=("-m", "sglang.launch_server", *suffix),
        model_path=MODEL_PATH,
        service_endpoint=FakeEndpoint(
            ip=SERVICE_ENDPOINT_IP,
            port=SERVICE_ENDPOINT_PORT,
        ),
        distributed_coordinator=FakeEndpoint(
            ip=DISTRIBUTED_COORDINATOR_IP,
            port=DISTRIBUTED_COORDINATOR_PORT,
        ),
        stage=FakeStage(resident_gpu_experts=resident_count),
    )


class FakeEnvironment:
    def __init__(self, resident_count: int, *, fail_stage: str | None = None) -> None:
        self.events: list[tuple[object, ...]] = []
        self.torch = FakeTorch(self.events)
        self.route = _route(resident_count)
        self.fail_stage = fail_stage
        self.runner: FakeModelRunner | None = None
        self.model_runner_arguments: dict[str, object] | None = None
        self.rows = _mask_rows(self.route.resident_gpu_expert_ids)
        if fail_stage == "dense_mask":
            self.rows = ((False,) * 64, *self.rows[1:])

    def runtime(self) -> backend.Glm47RuntimeBindings:
        server_args_type = FakeServerArgsType(
            self.events,
            len(self.route.resident_gpu_expert_ids),
            fail_parse=self.fail_stage == "parse",
        )

        def event(name: str) -> Any:
            def record(_argument: object | None = None) -> None:
                self.events.append((name,))
                if self.fail_stage == name:
                    raise RuntimeError(f"injected {name} failure")

            return record

        def tokenizer(*args: object, **kwargs: object) -> object:
            self.events.append(("tokenizer", args, kwargs))
            return SimpleNamespace(name="fake-tokenizer")

        def cleanup(*, shutdown_ray: bool) -> None:
            self.events.append(("cleanup", shutdown_ray))

        return backend.Glm47RuntimeBindings(
            torch=self.torch,
            inference_mode=self.torch.inference_mode,
            server_args_type=server_args_type,
            port_args_type=FakePortArgsType(
                self.events,
                return_mismatched_port=self.fail_stage == "port_args_mismatch",
            ),
            model_config_type=FakeModelConfigType(
                self.events,
                fail_model_config=self.fail_stage == "model_config",
            ),
            model_runner_type=FakeModelRunnerType(self),
            sampling_params_type=FakeSamplingParams,
            request_type=FakeRequest,
            schedule_batch_type=FakeScheduleBatch,
            forward_batch_type=FakeForwardBatchType,
            standard_topk_output_type=FakeTopKOutput,
            standard_dispatch_output_type=FakeDispatchOutput,
            speculative_algorithm_none="NONE",
            set_envs_and_config=event("set_env"),
            initialize_moe_config=event("init_moe"),
            initialize_fp8_gemm_config=event("init_fp8"),
            initialize_fp4_gemm_config=event("init_fp4"),
            get_tokenizer=tokenizer,
            suppress_other_loggers=event("suppress_loggers"),
            require_mlp_sync=lambda _server_args: False,
            get_kt_ep_gpu_experts_masks=lambda: FakeMask(self.rows),
            cleanup_dist_env_and_memory=cleanup,
            tensor_evidence_backend=FakeTensorEvidenceBackend(),
        )

    def reference_runner(
        self,
        *,
        model_path: Path,
        selected_expert_ids: tuple[int, ...],
        cpu_expert_ids: tuple[int, ...],
        gpu_expert_ids: tuple[int, ...],
        hidden_states: object,
    ) -> Glm47LayerOneReferenceOutputs[object]:
        self.events.append(
            (
                "reference",
                model_path,
                selected_expert_ids,
                cpu_expert_ids,
                gpu_expert_ids,
                hidden_states,
            )
        )
        if self.fail_stage == "reference":
            raise RuntimeError("injected reference failure")
        gpu_output = (
            FakeTensor(2.0, (1, 2_048), GLM47_FLOAT32_DTYPE) if gpu_expert_ids else None
        )
        return Glm47LayerOneReferenceOutputs(
            combined=FakeTensor(3.0, (1, 2_048), GLM47_FLOAT32_DTYPE),
            cpu_only=FakeTensor(
                1.0 if gpu_expert_ids else 3.0,
                (1, 2_048),
                GLM47_FLOAT32_DTYPE,
            ),
            gpu_only=gpu_output,
        )


@pytest.mark.parametrize("resident_count", (0, 2))
def test_backend_runs_exact_probe_and_cleans_up(resident_count: int) -> None:
    environment = FakeEnvironment(resident_count)
    process_spec = _process_spec(resident_count)

    evidence = backend.run_glm47_backend(
        process_spec,
        environment.route,
        runtime_loader=environment.runtime,
        reference_runner=environment.reference_runner,
    )

    assert evidence.cleanup_completed is True
    assert evidence.runtime.server_arguments == process_spec.arguments[2:]
    assert evidence.runtime.service_endpoint == (
        f"{SERVICE_ENDPOINT_IP}:{SERVICE_ENDPOINT_PORT}"
    )
    assert evidence.runtime.distributed_coordinator == (
        f"{DISTRIBUTED_COORDINATOR_IP}:{DISTRIBUTED_COORDINATOR_PORT}"
    )
    assert evidence.runtime.model_runner_nccl_port == SERVICE_ENDPOINT_PORT
    assert environment.events[:11] == [
        ("server_add_cli_args",),
        (
            "server_from_cli_args",
            MODEL_PATH,
            MODEL_PATH,
            "exact-suffix",
        ),
        ("set_env",),
        ("init_moe",),
        ("init_fp8",),
        ("init_fp4",),
        ("port_args",),
        ("suppress_loggers",),
        ("model_config",),
        ("model_runner",),
        (
            "tokenizer",
            (MODEL_PATH,),
            {"tokenizer_mode": "auto", "trust_remote_code": True},
        ),
    ]
    assert environment.events[-1] == ("cleanup", False)
    assert environment.events.count(("inference_mode_enter",)) == 1
    assert environment.events.count(("inference_mode_exit",)) == 1
    assert environment.torch.grad_enabled is True
    operation_names = [
        operation for operation, _enabled in environment.torch.execution_grad_states
    ]
    assert all(
        not enabled for _operation, enabled in environment.torch.execution_grad_states
    )
    assert operation_names.count("ktep_apply") == 94
    assert operation_names.count("gpu_apply") == (0 if resident_count == 0 else 4)
    assert operation_names.count("model_runner_forward") == 2
    assert operation_names.count("causal_model_forward") == 2

    arguments = environment.model_runner_arguments
    assert arguments is not None
    assert set(arguments) == {
        "model_config",
        "mem_fraction_static",
        "gpu_id",
        "tp_rank",
        "tp_size",
        "moe_ep_rank",
        "moe_ep_size",
        "pp_rank",
        "pp_size",
        "nccl_port",
        "server_args",
    }
    assert {
        key: arguments[key]
        for key in (
            "mem_fraction_static",
            "gpu_id",
            "tp_rank",
            "tp_size",
            "moe_ep_rank",
            "moe_ep_size",
            "pp_rank",
            "pp_size",
            "nccl_port",
        )
    } == {
        "mem_fraction_static": 0.8,
        "gpu_id": 0,
        "tp_rank": 0,
        "tp_size": 1,
        "moe_ep_rank": 0,
        "moe_ep_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "nccl_port": SERVICE_ENDPOINT_PORT,
    }

    assert len(evidence.wrapper_coverage.layers) == 46
    assert tuple(layer.layer_index for layer in evidence.wrapper_coverage.layers) == (
        tuple(range(1, 47))
    )
    assert all(
        layer.resident_gpu_expert_ids == environment.route.resident_gpu_expert_ids
        for layer in evidence.wrapper_coverage.layers
    )
    probe = evidence.layer_one_expert_probe
    assert probe.probe_invocation_count == 2
    assert probe.sglang_cpu_submit_count == 2
    assert probe.sglang_cpu_sync_count == 2
    assert probe.native_cpu_submit_count == 2
    assert probe.native_cpu_sync_count == 2
    assert probe.output_merge_count == 2
    assert probe.gpu_forward_count == (0 if resident_count == 0 else 2)
    assert (probe.gpu_output is None) is (resident_count == 0)
    assert (probe.hybrid_merge is None) is (resident_count == 0)

    assert evidence.short_forward.extend.input_token_ids == tuple(range(1, 9))
    assert evidence.short_forward.extend.positions == tuple(range(8))
    assert evidence.short_forward.extend.kv_cache_length_before == 0
    assert evidence.short_forward.extend.kv_cache_length_after == 8
    assert evidence.short_forward.decode.input_token_ids == (42,)
    assert evidence.short_forward.decode.positions == (8,)
    assert evidence.short_forward.decode.kv_cache_length_before == 8
    assert evidence.short_forward.decode.kv_cache_length_after == 9
    assert len(evidence.short_forward.wrapper_invocations) == 46
    assert all(
        invocation.extend_invocation_count == 1
        and invocation.decode_invocation_count == 1
        for invocation in evidence.short_forward.wrapper_invocations
    )
    assert len(evidence.trace_events) == (116 if resident_count == 0 else 120)
    assert len(evidence.trace_counters) == (110 if resident_count == 0 else 113)

    runner = environment.runner
    assert runner is not None
    assert "forward" not in vars(runner.model)
    for layer in runner.model.model.layers[1:]:
        quant_method = layer.mlp.experts.quant_method
        assert "apply" not in vars(quant_method)
    layer_one = runner.model.model.layers[1].mlp.experts.quant_method
    assert "_submit_cpu_forward" not in vars(layer_one)
    assert "_sync_cpu_forward" not in vars(layer_one)
    assert "submit_forward" not in vars(layer_one.wrapper)
    assert "sync_forward" not in vars(layer_one.wrapper)
    assert "apply" not in vars(layer_one.gpu_method)


@pytest.mark.parametrize(
    ("operation", "output_attribute"),
    (
        (SGLANG_CPU_SYNC, "layer_probe_cpu_outputs"),
        (GPU_METHOD_APPLY, "layer_probe_gpu_outputs"),
    ),
)
def test_trace_collector_clones_reused_layer_probe_buffers(
    operation: str,
    output_attribute: str,
) -> None:
    runtime = FakeEnvironment(2).runtime()
    collector = backend._TraceOutputCollector(runtime)
    reused = FakeTensor(1.0, (1, 2_048), GLM47_BF16_DTYPE, device="cuda:0")
    output: object = (
        FakeHiddenStates(reused) if operation == GPU_METHOD_APPLY else reused
    )

    collector(TraceProbe(operation, 1), "layer_probe", output)
    reused.value = 9.0
    collector(TraceProbe(operation, 1), "layer_probe", output)

    captured = cast(list[FakeTensor], getattr(collector, output_attribute))
    assert reused.clone_count == 2
    assert captured[0] is not reused
    assert captured[1] is not reused
    assert captured[0] is not captured[1]
    assert (captured[0].value, captured[1].value) == (1.0, 9.0)


def test_trace_collector_captures_model_logits_when_hidden_states_are_none() -> None:
    runtime = FakeEnvironment(0).runtime()
    collector = backend._TraceOutputCollector(runtime)

    captured = collector(
        TraceProbe(MODEL_FORWARD),
        "extend",
        FakeLogitsProcessorOutput(),
    )

    assert captured.kind == "next_token_logits"
    assert captured.tensor is not None
    assert captured.tensor.dtype == GLM47_FLOAT32_DTYPE
    assert captured.tensor.shape == (1, 154_880)


@pytest.mark.parametrize(
    "fail_stage",
    (
        "parse",
        "set_env",
        "port_args_mismatch",
        "model_config",
        "model_load",
        "coverage",
        "dense_mask",
        "reference",
        "forward",
    ),
)
def test_backend_always_cleans_up_and_restores_trace_on_failure(
    fail_stage: str,
) -> None:
    environment = FakeEnvironment(2, fail_stage=fail_stage)

    with pytest.raises(
        Exception,
        match="injected|not exact|does not mark|did not preserve",
    ):
        backend.run_glm47_backend(
            _process_spec(2),
            environment.route,
            runtime_loader=environment.runtime,
            reference_runner=environment.reference_runner,
        )

    assert environment.events[-1] == ("cleanup", False)
    assert environment.torch.grad_enabled is True
    runner = environment.runner
    if runner is not None:
        assert "forward" not in vars(runner.model)
        for layer in runner.model.model.layers[1:]:
            quant_method = layer.mlp.experts.quant_method
            assert "apply" not in vars(quant_method)
        layer_one = runner.model.model.layers[1].mlp.experts.quant_method
        assert "_submit_cpu_forward" not in vars(layer_one)
        assert "_sync_cpu_forward" not in vars(layer_one)
        assert "submit_forward" not in vars(layer_one.wrapper)
        assert "sync_forward" not in vars(layer_one.wrapper)
        assert "apply" not in vars(layer_one.gpu_method)


def test_invalid_route_fails_closed_after_runtime_load_and_cleanup() -> None:
    environment = FakeEnvironment(2)
    invalid_route = FakeRoute(
        resident_gpu_expert_ids=(0, 1),
        selected_expert_ids=(0, 1, 4, 5),
        cpu_expert_ids=(4,),
        gpu_expert_ids=(0, 1),
    )

    with pytest.raises(backend.Glm47BackendError, match="route is not exact"):
        backend.run_glm47_backend(
            _process_spec(2),
            invalid_route,
            runtime_loader=environment.runtime,
            reference_runner=environment.reference_runner,
        )

    assert environment.events == [("cleanup", False)]


def test_backend_rejects_endpoint_properties_that_disagree_with_arguments() -> None:
    environment = FakeEnvironment(2)
    process_spec = _process_spec(2)
    mismatched = replace(
        process_spec,
        service_endpoint=replace(
            process_spec.service_endpoint,
            port=SERVICE_ENDPOINT_PORT + 1,
        ),
    )

    with pytest.raises(backend.Glm47BackendError, match="reserved process endpoints"):
        backend.run_glm47_backend(
            mismatched,
            environment.route,
            runtime_loader=environment.runtime,
            reference_runner=environment.reference_runner,
        )

    assert environment.events[-1] == ("cleanup", False)
    assert environment.runner is None


def test_receipt_evidence_shapes_are_exact() -> None:
    environment = FakeEnvironment(2)
    evidence = backend.run_glm47_backend(
        _process_spec(2),
        environment.route,
        runtime_loader=environment.runtime,
        reference_runner=environment.reference_runner,
    )

    coverage = evidence.wrapper_coverage.as_receipt_json()
    assert set(coverage) == {"global_expert_mask_sha256", "layers"}
    layers = cast(list[dict[str, object]], coverage["layers"])
    assert len(layers) == 46
    assert set(layers[0]) == {
        "layer_index",
        "expert_module_name",
        "quant_method_wrapper",
        "expert_count",
        "resident_gpu_expert_ids",
        "cpu_backend_wrapper_class",
        "cpu_kernel_class",
        "global_expert_mask_sha256",
    }
    probe = evidence.layer_one_expert_probe.as_receipt_json()
    assert probe["selected_expert_ids"] == [0, 1, 4, 5]
    assert probe["cpu_expert_ids"] == [4, 5]
    assert probe["gpu_expert_ids"] == [0, 1]
    assert probe["gpu_forward_count"] == 2
    forward = evidence.short_forward.as_receipt_json()
    assert cast(dict[str, object], forward["extend"])["input_token_ids"] == list(
        range(1, 9)
    )
    assert cast(dict[str, object], forward["decode"])["input_token_ids"] == [42]
    assert len(cast(list[object], forward["wrapper_invocations"])) == 46


def test_runtime_loader_imports_only_the_direct_pinned_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    sentinel_backend = cast(Glm47TensorEvidenceBackend[object], object())

    def sentinel_inference_mode() -> None:
        return None

    symbols: dict[str, dict[str, object]] = {
        "torch": {"inference_mode": sentinel_inference_mode},
        "sglang.srt.server_args": {"ServerArgs": object(), "PortArgs": object()},
        "sglang.srt.entrypoints.engine": {"_set_envs_and_config": lambda _: None},
        "sglang.srt.layers.moe": {"initialize_moe_config": lambda _: None},
        "sglang.srt.layers.quantization.fp8_utils": {
            "initialize_fp8_gemm_config": lambda _: None
        },
        "sglang.srt.layers.quantization.fp4_utils": {
            "initialize_fp4_gemm_config": lambda _: None
        },
        "sglang.srt.configs.model_config": {"ModelConfig": object()},
        "sglang.srt.model_executor.model_runner": {"ModelRunner": object()},
        "sglang.srt.utils.hf_transformers_utils": {"get_tokenizer": lambda *_: None},
        "sglang.srt.utils.common": {
            "suppress_other_loggers": lambda: None,
            "require_mlp_sync": lambda _: False,
        },
        "sglang.srt.managers.schedule_batch": {
            "Req": object(),
            "ScheduleBatch": object(),
        },
        "sglang.srt.model_executor.forward_batch_info": {"ForwardBatch": object()},
        "sglang.srt.sampling.sampling_params": {"SamplingParams": object()},
        "sglang.srt.speculative.spec_info": {
            "SpeculativeAlgorithm": SimpleNamespace(NONE="NONE")
        },
        "sglang.srt.layers.moe.topk": {"StandardTopKOutput": object()},
        "sglang.srt.layers.moe.token_dispatcher.standard": {
            "StandardDispatchOutput": object()
        },
        "sglang.srt.layers.moe.kt_ep_wrapper": {
            "get_kt_ep_gpu_experts_masks": lambda: None
        },
        "sglang.srt.distributed.parallel_state": {
            "cleanup_dist_env_and_memory": lambda **_: None
        },
    }

    def fake_import(name: str) -> ModuleType:
        imported.append(name)
        module = ModuleType(name)
        for key, value in symbols.get(name, {}).items():
            setattr(module, key, value)
        return module

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(
        backend,
        "create_torch_glm47_tensor_evidence_backend",
        lambda: sentinel_backend,
    )

    bindings = backend.load_glm47_runtime_bindings()

    assert imported == [
        "torch",
        "sglang.srt.server_args",
        "sglang.srt.entrypoints.engine",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.quantization.fp8_utils",
        "sglang.srt.layers.quantization.fp4_utils",
        "sglang.srt.configs.model_config",
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.utils.hf_transformers_utils",
        "sglang.srt.utils.common",
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.model_executor.forward_batch_info",
        "sglang.srt.sampling.sampling_params",
        "sglang.srt.speculative.spec_info",
        "sglang.srt.layers.moe.topk",
        "sglang.srt.layers.moe.token_dispatcher.standard",
        "sglang.srt.layers.moe.kt_ep_wrapper",
        "sglang.srt.distributed.parallel_state",
    ]
    assert bindings.tensor_evidence_backend is sentinel_backend
    assert bindings.inference_mode is sentinel_inference_mode
    assert all("bench" not in name and "profiler" not in name for name in imported)
