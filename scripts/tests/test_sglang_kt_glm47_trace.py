from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Mapping
from types import MethodType

import pytest

import scripts.sglang_kt_glm47_trace as trace_module
from scripts.sglang_kt_glm47_trace import (
    GLM47_ROUTED_LAYER_IDS,
    GPU_METHOD_APPLY,
    LAYER_ONE_KTEP_OUTER_APPLY,
    MODEL_FORWARD,
    NATIVE_WRAPPER_SUBMIT,
    NATIVE_WRAPPER_SYNC,
    QUANT_METHOD_APPLY,
    SGLANG_CPU_SUBMIT,
    SGLANG_CPU_SYNC,
    Glm47TraceConfigurationError,
    Glm47TraceCoverageError,
    Glm47TraceSession,
    Glm47TraceTargets,
    InstanceMethodPatchError,
    InstanceMethodPatchStack,
    TraceProbe,
)


def test_import_does_not_require_torch_or_sglang() -> None:
    repository = trace_module.__file__
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
spec = importlib.util.spec_from_file_location("isolated_glm47_trace", {repository!r})
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


class ArithmeticTarget:
    def calculate(self, value: int) -> int:
        return value + 1


def record_result(
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]],
):
    def observer(
        arguments: tuple[object, ...],
        keyword_arguments: Mapping[str, object],
        result: object,
    ) -> None:
        calls.append((arguments, keyword_arguments, result))

    return observer


def test_patch_stack_restores_inherited_descriptor() -> None:
    target = ArithmeticTarget()
    descriptor = inspect.getattr_static(target, "calculate")
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]] = []

    with InstanceMethodPatchStack() as patches:
        patches.patch_success(target, "calculate", record_result(calls))
        assert "calculate" in vars(target)
        assert target.calculate(4) == 5

    assert calls == [((4,), {}, 5)]
    assert "calculate" not in vars(target)
    assert inspect.getattr_static(target, "calculate") is descriptor
    assert target.calculate(5) == 6


def test_patch_stack_restores_preexisting_instance_method_exactly() -> None:
    target = ArithmeticTarget()

    def override(_target: ArithmeticTarget, value: int) -> int:
        return value * 3

    original_override = MethodType(override, target)
    target.calculate = original_override  # type: ignore[method-assign]
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]] = []
    patches = InstanceMethodPatchStack()
    patches.patch_success(target, "calculate", record_result(calls))

    assert target.calculate(4) == 12
    patches.restore()

    assert vars(target)["calculate"] is original_override
    assert target.calculate(5) == 15
    assert calls == [((4,), {}, 12)]


def test_nested_patches_count_one_success_each_and_restore() -> None:
    target = ArithmeticTarget()
    calls: list[str] = []
    descriptor = inspect.getattr_static(target, "calculate")
    patches = InstanceMethodPatchStack()

    def inner(
        arguments: tuple[object, ...],
        keyword_arguments: Mapping[str, object],
        result: object,
    ) -> None:
        del arguments, keyword_arguments, result
        calls.append("inner")

    def outer(
        arguments: tuple[object, ...],
        keyword_arguments: Mapping[str, object],
        result: object,
    ) -> None:
        del arguments, keyword_arguments, result
        calls.append("outer")

    patches.patch_success(target, "calculate", inner)
    first_wrapper = vars(target)["calculate"]
    patches.patch_success(target, "calculate", outer)

    assert patches.patch_count == 2
    assert target.calculate(2) == 3
    assert calls == ["inner", "outer"]
    patches.restore()

    assert patches.patch_count == 0
    assert patches.closed
    assert "calculate" not in vars(target)
    assert inspect.getattr_static(target, "calculate") is descriptor
    assert first_wrapper is not descriptor


def test_patch_stack_records_only_successful_returns() -> None:
    class ExplodingTarget:
        def run(self) -> None:
            raise RuntimeError("expected failure")

    target = ExplodingTarget()
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]] = []

    with InstanceMethodPatchStack() as patches:
        patches.patch_success(target, "run", record_result(calls))
        with pytest.raises(RuntimeError, match="expected failure"):
            target.run()

    assert calls == []
    assert "run" not in vars(target)


def test_patch_stack_restores_after_observer_error() -> None:
    target = ArithmeticTarget()

    def fail_observer(
        arguments: tuple[object, ...],
        keyword_arguments: Mapping[str, object],
        result: object,
    ) -> None:
        del arguments, keyword_arguments, result
        raise RuntimeError("copy failed")

    with (
        pytest.raises(RuntimeError, match="copy failed"),
        InstanceMethodPatchStack() as patches,
    ):
        patches.patch_success(target, "calculate", fail_observer)
        target.calculate(1)

    assert "calculate" not in vars(target)
    assert target.calculate(1) == 2


def test_patch_stack_restores_multiple_targets_in_reverse_order() -> None:
    restore_order: list[str] = []

    class LoggingTarget:
        def __init__(self, name: str) -> None:
            object.__setattr__(self, "name", name)

        def run(self) -> str:
            return self.name

        def __delattr__(self, name: str) -> None:
            if name == "run":
                restore_order.append(self.name)
            object.__delattr__(self, name)

    first = LoggingTarget("first")
    second = LoggingTarget("second")
    patches = InstanceMethodPatchStack()
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]] = []
    patches.patch_success(first, "run", record_result(calls))
    patches.patch_success(second, "run", record_result(calls))

    patches.restore()

    assert restore_order == ["second", "first"]


def test_closed_patch_stack_rejects_new_patch() -> None:
    patches = InstanceMethodPatchStack()
    patches.restore()
    target = ArithmeticTarget()
    calls: list[tuple[tuple[object, ...], Mapping[str, object], object]] = []

    with pytest.raises(InstanceMethodPatchError, match="already closed"):
        patches.patch_success(target, "calculate", record_result(calls))


class FakeNativeWrapper:
    def __init__(self) -> None:
        self.submit_count = 0
        self.sync_count = 0

    def submit_forward(
        self,
        hidden_states: object,
        topk_ids: object,
        topk_weights: object,
        cuda_stream: object,
    ) -> None:
        del hidden_states, topk_ids, topk_weights, cuda_stream
        self.submit_count += 1

    def sync_forward(
        self, hidden_states: object, cuda_stream: object
    ) -> dict[str, int]:
        del hidden_states, cuda_stream
        self.sync_count += 1
        return {"cpu": self.sync_count}


class FakeGpuMethod:
    def __init__(self) -> None:
        self.apply_count = 0

    def apply(self, layer: object, dispatch_output: object) -> dict[str, int]:
        del layer, dispatch_output
        self.apply_count += 1
        return {"gpu": self.apply_count}


class FakeQuantMethod:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.fail = False

    def apply(self, layer: object, dispatch_output: object) -> dict[str, object]:
        del layer
        if self.fail:
            raise RuntimeError(f"layer {self.layer_id} failed")
        return {"layer": self.layer_id, "dispatch": dispatch_output}


class FakeKtepMethod(FakeQuantMethod):
    def __init__(self) -> None:
        super().__init__(1)
        self.wrapper = FakeNativeWrapper()
        self.gpu_method = FakeGpuMethod()

    def _submit_cpu_forward(
        self,
        hidden_states: object,
        topk_ids: object,
        topk_weights: object,
    ) -> None:
        self.wrapper.submit_forward(
            hidden_states,
            topk_ids,
            topk_weights,
            "cpu-stream",
        )

    def _sync_cpu_forward(self, hidden_states: object) -> dict[str, int]:
        return self.wrapper.sync_forward(hidden_states, "cpu-stream")

    def apply(self, layer: object, dispatch_output: object) -> dict[str, object]:
        if self.fail:
            raise RuntimeError("layer 1 failed")
        self._submit_cpu_forward(dispatch_output, (0, 4, 5, 6), (0.4, 0.3, 0.2, 0.1))
        gpu_output = self.gpu_method.apply(layer, dispatch_output)
        cpu_output = self._sync_cpu_forward(dispatch_output)
        return {"layer": 1, "gpu": gpu_output, "cpu": cpu_output}


class FakeExperts:
    def __init__(self, quant_method: FakeQuantMethod) -> None:
        self.quant_method = quant_method


class FakeMlp:
    def __init__(self, quant_method: FakeQuantMethod | None = None) -> None:
        if quant_method is not None:
            self.experts = FakeExperts(quant_method)


class FakeLayer:
    def __init__(self, quant_method: FakeQuantMethod | None = None) -> None:
        self.mlp = FakeMlp(quant_method)


class FakeDecoder:
    def __init__(self, quant_methods: tuple[FakeQuantMethod, ...]) -> None:
        self.layers = [FakeLayer()]
        self.layers.extend(FakeLayer(method) for method in quant_methods)


class FakeModel:
    def __init__(self, quant_methods: tuple[FakeQuantMethod, ...]) -> None:
        self.model = FakeDecoder(quant_methods)
        self.outputs: list[dict[str, object]] = []

    def forward(self, dispatch_output: object) -> dict[str, object]:
        layer_outputs = []
        for layer_id in GLM47_ROUTED_LAYER_IDS:
            quant_method = self.model.layers[layer_id].mlp.experts.quant_method
            layer_outputs.append(quant_method.apply(None, dispatch_output))
        output = {"dispatch": dispatch_output, "layer_outputs": layer_outputs}
        self.outputs.append(output)
        return output


class FakeModelRunner:
    def __init__(self) -> None:
        quant_methods: list[FakeQuantMethod] = [FakeKtepMethod()]
        quant_methods.extend(FakeQuantMethod(layer_id) for layer_id in range(2, 47))
        self.model = FakeModel(tuple(quant_methods))


def fake_output_copier(
    probe: TraceProbe,
    phase: str | None,
    output: object,
) -> object:
    del phase
    if probe.operation == MODEL_FORWARD:
        assert isinstance(output, dict)
        return output["dispatch"]
    if isinstance(output, dict) and "layer" in output:
        return output["layer"]
    if output is None:
        return "returned-none"
    return type(output).__name__


def all_traced_owners(targets: Glm47TraceTargets) -> tuple[tuple[object, str], ...]:
    owners = tuple((method, "apply") for _, method in targets.layer_quant_methods)
    return (
        *owners,
        (targets.layer_one_ktep_method, "_submit_cpu_forward"),
        (targets.layer_one_ktep_method, "_sync_cpu_forward"),
        (targets.native_wrapper, "submit_forward"),
        (targets.native_wrapper, "sync_forward"),
        (targets.gpu_method, "apply"),
        (targets.model, "forward"),
    )


def test_discovery_finds_exact_pinned_runtime_graph() -> None:
    runner = FakeModelRunner()

    targets = Glm47TraceTargets.discover(runner)

    assert tuple(layer_id for layer_id, _ in targets.layer_quant_methods) == (
        GLM47_ROUTED_LAYER_IDS
    )
    assert targets.model is runner.model
    assert targets.layer_one_ktep_method is (
        runner.model.model.layers[1].mlp.experts.quant_method
    )


def test_discovery_rejects_wrong_layer_count() -> None:
    runner = FakeModelRunner()
    runner.model.model.layers.pop()

    with pytest.raises(Glm47TraceConfigurationError, match="expected 47"):
        Glm47TraceTargets.discover(runner)


def test_discovery_rejects_shared_quant_method_instances() -> None:
    runner = FakeModelRunner()
    first_method = runner.model.model.layers[1].mlp.experts.quant_method
    runner.model.model.layers[2].mlp.experts.quant_method = first_method

    with pytest.raises(Glm47TraceConfigurationError, match="distinct"):
        Glm47TraceTargets.discover(runner)


def test_trace_session_counts_nested_probes_and_exact_forward_coverage() -> None:
    runner = FakeModelRunner()
    targets = Glm47TraceTargets.discover(runner)
    descriptors = {
        (id(owner), method_name): inspect.getattr_static(owner, method_name)
        for owner, method_name in all_traced_owners(targets)
    }
    trace = Glm47TraceSession(targets, output_copier=fake_output_copier)

    with trace:
        with trace.phase("layer_probe"):
            targets.layer_one_ktep_method.apply(None, "direct")
        with trace.phase("extend"):
            runner.model.forward("prompt")
        with trace.phase("decode"):
            runner.model.forward("token")

        trace.require_exact_layer_apply_coverage(("extend", "decode"))
        assert (
            trace.successful_return_count(TraceProbe(MODEL_FORWARD), phase="extend")
            == 1
        )
        assert (
            trace.successful_return_count(TraceProbe(MODEL_FORWARD), phase="decode")
            == 1
        )
        for operation in (
            QUANT_METHOD_APPLY,
            LAYER_ONE_KTEP_OUTER_APPLY,
            SGLANG_CPU_SUBMIT,
            SGLANG_CPU_SYNC,
            NATIVE_WRAPPER_SUBMIT,
            NATIVE_WRAPPER_SYNC,
            GPU_METHOD_APPLY,
        ):
            assert (
                trace.successful_return_count(
                    TraceProbe(operation, 1), phase="layer_probe"
                )
                == 1
            )

    direct_operations = [
        event.probe.operation for event in trace.events if event.phase == "layer_probe"
    ]
    assert direct_operations == [
        NATIVE_WRAPPER_SUBMIT,
        SGLANG_CPU_SUBMIT,
        GPU_METHOD_APPLY,
        NATIVE_WRAPPER_SYNC,
        SGLANG_CPU_SYNC,
        QUANT_METHOD_APPLY,
        LAYER_ONE_KTEP_OUTER_APPLY,
    ]
    assert [event.sequence for event in trace.events] == list(
        range(1, len(trace.events) + 1)
    )
    assert all(event.output_captured for event in trace.events)
    model_events = [
        event for event in trace.events if event.probe.operation == MODEL_FORWARD
    ]
    assert [event.captured_output for event in model_events] == ["prompt", "token"]

    for owner, method_name in all_traced_owners(targets):
        assert method_name not in vars(owner)
        assert (
            inspect.getattr_static(owner, method_name)
            is descriptors[(id(owner), method_name)]
        )


def test_exact_coverage_rejects_duplicate_and_missing_layer_calls() -> None:
    runner = FakeModelRunner()
    trace = Glm47TraceSession.discover(runner)

    with trace:
        with trace.phase("extend"):
            runner.model.forward("prompt")
            runner.model.model.layers[1].mlp.experts.quant_method.apply(None, "extra")
        with trace.phase("decode"):
            for layer_id in range(1, 46):
                method = runner.model.model.layers[layer_id].mlp.experts.quant_method
                method.apply(None, "token")

        with pytest.raises(Glm47TraceCoverageError, match="not exact"):
            trace.require_exact_layer_apply_coverage(("extend", "decode"))

    assert trace.layer_apply_counts(phase="extend")[1] == 2
    assert 46 not in trace.layer_apply_counts(phase="decode")


def test_trace_session_counts_no_failed_return_and_restores_on_error() -> None:
    runner = FakeModelRunner()
    targets = Glm47TraceTargets.discover(runner)
    failing_method = runner.model.model.layers[10].mlp.experts.quant_method
    failing_method.fail = True
    trace = Glm47TraceSession(targets)

    with (
        pytest.raises(RuntimeError, match="layer 10 failed"),
        trace,
        trace.phase("decode"),
    ):
        runner.model.forward("token")

    counts = trace.layer_apply_counts(phase="decode")
    assert counts == {layer_id: 1 for layer_id in range(1, 10)}
    assert trace.successful_return_count(TraceProbe(MODEL_FORWARD), phase="decode") == 0
    assert trace.active_phase is None
    for owner, method_name in all_traced_owners(targets):
        assert method_name not in vars(owner)


def test_output_copier_failure_does_not_create_event_and_restores() -> None:
    runner = FakeModelRunner()
    targets = Glm47TraceTargets.discover(runner)

    def failing_copier(
        probe: TraceProbe,
        phase: str | None,
        output: object,
    ) -> object:
        del phase, output
        if probe.operation == GPU_METHOD_APPLY:
            raise RuntimeError("copy failed")
        return probe.operation

    trace = Glm47TraceSession(targets, output_copier=failing_copier)
    with (
        pytest.raises(RuntimeError, match="copy failed"),
        trace,
        trace.phase("layer_probe"),
    ):
        targets.layer_one_ktep_method.apply(None, "direct")

    assert [event.probe.operation for event in trace.events] == [
        NATIVE_WRAPPER_SUBMIT,
        SGLANG_CPU_SUBMIT,
    ]
    for owner, method_name in all_traced_owners(targets):
        assert method_name not in vars(owner)


def test_output_copier_result_is_retained_instead_of_runtime_output() -> None:
    target = ArithmeticTarget()
    copied: list[int] = []

    def copier(
        probe: TraceProbe,
        phase: str | None,
        output: object,
    ) -> object:
        del probe, phase
        assert isinstance(output, list)
        copied.extend(output)
        return tuple(output)

    class ListQuantMethod:
        def apply(self, layer: object, dispatch_output: object) -> list[int]:
            del layer, dispatch_output
            return [1, 2]

    del target
    runner = FakeModelRunner()
    list_method = ListQuantMethod()
    runner.model.model.layers[2].mlp.experts.quant_method = list_method
    trace = Glm47TraceSession.discover(runner, output_copier=copier)

    with trace, trace.phase("capture"):
        output = list_method.apply(None, None)
        output.append(3)

    event = next(
        event
        for event in trace.events
        if event.probe == TraceProbe(QUANT_METHOD_APPLY, 2)
    )
    assert copied == [1, 2]
    assert event.captured_output == (1, 2)


def test_phase_context_rejects_nesting_and_resets_after_error() -> None:
    runner = FakeModelRunner()
    trace = Glm47TraceSession.discover(runner)

    with trace:
        with (
            pytest.raises(Glm47TraceConfigurationError, match="already active"),
            trace.phase("extend"),
            trace.phase("decode"),
        ):
            pass
        assert trace.active_phase is None

        with (
            pytest.raises(RuntimeError, match="phase failed"),
            trace.phase("decode"),
        ):
            raise RuntimeError("phase failed")
        assert trace.active_phase is None
