"""Torch-free tracing helpers for the pinned GLM-4.7 SGLang runtime probe.

The live validator imports this module before it imports Torch or SGLang.  The
helpers therefore operate on a discovered object graph and deliberately avoid
runtime-specific imports.  Only successful method returns become trace events.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MethodType, TracebackType
from typing import Final, Protocol, cast

GLM47_ROUTED_LAYER_IDS: Final[tuple[int, ...]] = tuple(range(1, 47))

QUANT_METHOD_APPLY: Final = "quant_method.apply"
LAYER_ONE_KTEP_OUTER_APPLY: Final = "layer_1_ktep.apply"
SGLANG_CPU_SUBMIT: Final = "ktep._submit_cpu_forward"
SGLANG_CPU_SYNC: Final = "ktep._sync_cpu_forward"
NATIVE_WRAPPER_SUBMIT: Final = "native_wrapper.submit_forward"
NATIVE_WRAPPER_SYNC: Final = "native_wrapper.sync_forward"
GPU_METHOD_APPLY: Final = "gpu_method.apply"
MODEL_FORWARD: Final = "model.forward"

_BoundMethod = Callable[..., object]


class Glm47TraceError(RuntimeError):
    """Base error for trace configuration and validation failures."""


class Glm47TraceConfigurationError(Glm47TraceError):
    """Raised when the pinned runtime graph cannot be traced safely."""


class Glm47TraceCoverageError(Glm47TraceError):
    """Raised when a phase does not execute every routed layer exactly once."""


class InstanceMethodPatchError(Glm47TraceError):
    """Raised when an instance method cannot be patched or restored."""


class MethodSuccessObserver(Protocol):
    """Observe one method only after its wrapped invocation returns."""

    def __call__(
        self,
        arguments: tuple[object, ...],
        keyword_arguments: Mapping[str, object],
        result: object,
        /,
    ) -> None: ...


class TraceOutputCopier(Protocol):
    """Copy a small, detached value from a successful runtime output."""

    def __call__(
        self,
        probe: TraceProbe,
        phase: str | None,
        output: object,
        /,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class TraceProbe:
    operation: str
    layer_id: int | None = None

    def __post_init__(self) -> None:
        if not self.operation or self.operation.strip() != self.operation:
            raise ValueError("trace operation must be nonempty and trimmed")
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("trace layer ID must be nonnegative")


@dataclass(frozen=True, slots=True)
class SuccessfulTraceEvent:
    sequence: int
    phase: str | None
    probe: TraceProbe
    output_captured: bool
    captured_output: object | None


@dataclass(frozen=True, slots=True)
class Glm47TraceTargets:
    """The exact runtime objects whose instance methods are traced."""

    model: object
    layer_quant_methods: tuple[tuple[int, object], ...]
    layer_one_ktep_method: object
    native_wrapper: object
    gpu_method: object

    @classmethod
    def discover(cls, model_runner: object) -> Glm47TraceTargets:
        """Discover and validate the pinned PP1 GLM-4.7 object graph."""

        model = _require_attribute(model_runner, "model")
        decoder = _require_attribute(model, "model")
        layers_value = _require_attribute(decoder, "layers")
        if not hasattr(layers_value, "__len__") or not hasattr(
            layers_value, "__getitem__"
        ):
            raise Glm47TraceConfigurationError("model layers are not indexable")
        layers = cast(_IndexableObjects, layers_value)
        if len(layers) != 47:
            raise Glm47TraceConfigurationError(
                f"expected 47 GLM layers, observed {len(layers)}"
            )

        layer_quant_methods: list[tuple[int, object]] = []
        for layer_id in GLM47_ROUTED_LAYER_IDS:
            layer = layers[layer_id]
            mlp = _require_attribute(layer, "mlp")
            experts = _require_attribute(mlp, "experts")
            quant_method = _require_attribute(experts, "quant_method")
            _require_callable_method(quant_method, "apply")
            layer_quant_methods.append((layer_id, quant_method))

        quant_method_ids = {id(method) for _, method in layer_quant_methods}
        if len(quant_method_ids) != len(GLM47_ROUTED_LAYER_IDS):
            raise Glm47TraceConfigurationError(
                "routed layers do not have distinct quantization methods"
            )

        layer_one_ktep_method = layer_quant_methods[0][1]
        native_wrapper = _require_attribute(layer_one_ktep_method, "wrapper")
        gpu_method = _require_attribute(layer_one_ktep_method, "gpu_method")
        for owner, method_names in (
            (model, ("forward",)),
            (
                layer_one_ktep_method,
                ("apply", "_submit_cpu_forward", "_sync_cpu_forward"),
            ),
            (native_wrapper, ("submit_forward", "sync_forward")),
            (gpu_method, ("apply",)),
        ):
            for method_name in method_names:
                _require_callable_method(owner, method_name)

        return cls(
            model=model,
            layer_quant_methods=tuple(layer_quant_methods),
            layer_one_ktep_method=layer_one_ktep_method,
            native_wrapper=native_wrapper,
            gpu_method=gpu_method,
        )


class _IndexableObjects(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


@dataclass(slots=True)
class _InstanceMethodPatch:
    target: object
    method_name: str
    had_instance_attribute: bool
    original_instance_value: object | None

    def restore(self) -> None:
        if self.had_instance_attribute:
            setattr(self.target, self.method_name, self.original_instance_value)
        else:
            delattr(self.target, self.method_name)


class InstanceMethodPatchStack:
    """Apply reversible instance-only method patches in stack order."""

    def __init__(self) -> None:
        self._patches: list[_InstanceMethodPatch] = []
        self._closed = False

    @property
    def patch_count(self) -> int:
        return len(self._patches)

    @property
    def closed(self) -> bool:
        return self._closed

    def patch_success(
        self,
        target: object,
        method_name: str,
        observer: MethodSuccessObserver,
    ) -> None:
        """Wrap one bound method and notify only after a successful return."""

        if self._closed:
            raise InstanceMethodPatchError("patch stack is already closed")
        if not method_name or method_name.strip() != method_name:
            raise InstanceMethodPatchError("method name must be nonempty and trimmed")
        _require_callable_method(target, method_name)
        original_method = cast(_BoundMethod, getattr(target, method_name))

        instance_namespace = getattr(target, "__dict__", None)
        if not isinstance(instance_namespace, dict):
            raise InstanceMethodPatchError(
                f"{type(target).__name__}.{method_name} has no writable instance state"
            )
        typed_namespace = cast(dict[str, object], instance_namespace)
        had_instance_attribute = method_name in typed_namespace
        original_instance_value = typed_namespace.get(method_name)
        patch = _InstanceMethodPatch(
            target=target,
            method_name=method_name,
            had_instance_attribute=had_instance_attribute,
            original_instance_value=original_instance_value,
        )

        def wrapped(
            _instance: object, *arguments: object, **keyword_arguments: object
        ) -> object:
            result = original_method(*arguments, **keyword_arguments)
            observer(arguments, keyword_arguments, result)
            return result

        try:
            setattr(target, method_name, MethodType(wrapped, target))
        except Exception as error:
            try:
                patch.restore()
            except Exception as restore_error:
                raise InstanceMethodPatchError(
                    f"failed to patch and restore {type(target).__name__}.{method_name}"
                ) from restore_error
            raise InstanceMethodPatchError(
                f"failed to patch {type(target).__name__}.{method_name}"
            ) from error
        self._patches.append(patch)

    def restore(self) -> None:
        """Restore every raw instance slot in reverse patch order."""

        if self._closed:
            return
        self._closed = True
        failures: list[Exception] = []
        while self._patches:
            patch = self._patches.pop()
            try:
                patch.restore()
            except Exception as error:
                failures.append(error)
        if failures:
            raise InstanceMethodPatchError(
                f"failed to restore {len(failures)} instance method patch(es)"
            ) from failures[0]

    def __enter__(self) -> InstanceMethodPatchStack:
        if self._closed:
            raise InstanceMethodPatchError("patch stack is already closed")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, exception, traceback
        self.restore()
        return False


class _TracePhaseContext:
    def __init__(self, trace: Glm47TraceSession, phase: str) -> None:
        self._trace = trace
        self._phase = phase
        self._entered = False

    def __enter__(self) -> Glm47TraceSession:
        if self._entered:
            raise Glm47TraceConfigurationError("trace phase context cannot be reused")
        self._trace.enter_phase(self._phase)
        self._entered = True
        return self._trace

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, exception, traceback
        self._trace.leave_phase(self._phase)
        return False


class Glm47TraceSession:
    """Trace successful calls in one loaded, pinned GLM-4.7 model runner."""

    def __init__(
        self,
        targets: Glm47TraceTargets,
        *,
        output_copier: TraceOutputCopier | None = None,
    ) -> None:
        self._targets = targets
        self._output_copier = output_copier
        self._patches = InstanceMethodPatchStack()
        self._events: list[SuccessfulTraceEvent] = []
        self._active_phase: str | None = None
        self._installed = False
        self._restored = False

    @classmethod
    def discover(
        cls,
        model_runner: object,
        *,
        output_copier: TraceOutputCopier | None = None,
    ) -> Glm47TraceSession:
        return cls(
            Glm47TraceTargets.discover(model_runner),
            output_copier=output_copier,
        )

    @property
    def events(self) -> tuple[SuccessfulTraceEvent, ...]:
        return tuple(self._events)

    @property
    def active_phase(self) -> str | None:
        return self._active_phase

    @property
    def installed(self) -> bool:
        return self._installed and not self._restored

    def install(self) -> None:
        if self._installed:
            raise Glm47TraceConfigurationError("trace session was already installed")
        if self._restored:
            raise Glm47TraceConfigurationError("trace session was already restored")
        self._installed = True
        try:
            for layer_id, quant_method in self._targets.layer_quant_methods:
                self._install_probe(
                    quant_method,
                    "apply",
                    TraceProbe(QUANT_METHOD_APPLY, layer_id),
                )

            # Deliberately wrap layer 1 twice.  The outer probe must return only
            # after the generic layer probe and every nested backend probe.
            self._install_probe(
                self._targets.layer_one_ktep_method,
                "apply",
                TraceProbe(LAYER_ONE_KTEP_OUTER_APPLY, 1),
            )
            self._install_probe(
                self._targets.layer_one_ktep_method,
                "_submit_cpu_forward",
                TraceProbe(SGLANG_CPU_SUBMIT, 1),
            )
            self._install_probe(
                self._targets.layer_one_ktep_method,
                "_sync_cpu_forward",
                TraceProbe(SGLANG_CPU_SYNC, 1),
            )
            self._install_probe(
                self._targets.native_wrapper,
                "submit_forward",
                TraceProbe(NATIVE_WRAPPER_SUBMIT, 1),
            )
            self._install_probe(
                self._targets.native_wrapper,
                "sync_forward",
                TraceProbe(NATIVE_WRAPPER_SYNC, 1),
            )
            self._install_probe(
                self._targets.gpu_method,
                "apply",
                TraceProbe(GPU_METHOD_APPLY, 1),
            )
            self._install_probe(
                self._targets.model,
                "forward",
                TraceProbe(MODEL_FORWARD),
            )
        except Exception:
            self._patches.restore()
            self._restored = True
            raise

    def restore(self) -> None:
        if self._restored:
            return
        if self._active_phase is not None:
            raise Glm47TraceConfigurationError(
                "cannot restore a trace session inside an active phase"
            )
        self._patches.restore()
        self._restored = True

    def phase(self, phase: str) -> _TracePhaseContext:
        if not phase or phase.strip() != phase:
            raise Glm47TraceConfigurationError(
                "trace phase must be nonempty and trimmed"
            )
        return _TracePhaseContext(self, phase)

    def successful_return_count(
        self,
        probe: TraceProbe,
        *,
        phase: str | None,
    ) -> int:
        return sum(
            event.probe == probe and event.phase == phase for event in self._events
        )

    def layer_apply_counts(self, *, phase: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for event in self._events:
            if event.phase != phase or event.probe.operation != QUANT_METHOD_APPLY:
                continue
            layer_id = event.probe.layer_id
            if layer_id is None:
                raise Glm47TraceCoverageError(
                    "quantization apply event omitted its layer ID"
                )
            counts[layer_id] = counts.get(layer_id, 0) + 1
        return counts

    def require_exact_layer_apply_coverage(self, phases: Iterable[str]) -> None:
        phase_names = tuple(phases)
        if not phase_names or len(set(phase_names)) != len(phase_names):
            raise ValueError("coverage phases must be nonempty and unique")
        expected = {layer_id: 1 for layer_id in GLM47_ROUTED_LAYER_IDS}
        failures = {
            phase: self.layer_apply_counts(phase=phase)
            for phase in phase_names
            if self.layer_apply_counts(phase=phase) != expected
        }
        if failures:
            raise Glm47TraceCoverageError(
                f"routed layer apply coverage is not exact: {failures}"
            )

    def _install_probe(
        self,
        owner: object,
        method_name: str,
        probe: TraceProbe,
    ) -> None:
        def observe(
            arguments: tuple[object, ...],
            keyword_arguments: Mapping[str, object],
            result: object,
        ) -> None:
            del arguments, keyword_arguments
            self._record_success(probe, result)

        self._patches.patch_success(owner, method_name, observe)

    def _record_success(self, probe: TraceProbe, result: object) -> None:
        captured_output: object | None = None
        output_captured = self._output_copier is not None
        if self._output_copier is not None:
            captured_output = self._output_copier(
                probe,
                self._active_phase,
                result,
            )
        self._events.append(
            SuccessfulTraceEvent(
                sequence=len(self._events) + 1,
                phase=self._active_phase,
                probe=probe,
                output_captured=output_captured,
                captured_output=captured_output,
            )
        )

    def enter_phase(self, phase: str) -> None:
        """Enter one non-nested phase through a phase context."""

        if not self.installed:
            raise Glm47TraceConfigurationError("trace session is not installed")
        if self._active_phase is not None:
            raise Glm47TraceConfigurationError(
                f"trace phase {self._active_phase!r} is already active"
            )
        self._active_phase = phase

    def leave_phase(self, phase: str) -> None:
        """Leave the currently active phase through its phase context."""

        if self._active_phase != phase:
            raise Glm47TraceConfigurationError(
                f"cannot leave inactive trace phase {phase!r}"
            )
        self._active_phase = None

    def __enter__(self) -> Glm47TraceSession:
        self.install()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exception_type, exception, traceback
        self.restore()
        return False


def _require_attribute(owner: object, attribute_name: str) -> object:
    try:
        return cast(object, getattr(owner, attribute_name))
    except AttributeError as error:
        raise Glm47TraceConfigurationError(
            f"{type(owner).__name__} is missing {attribute_name}"
        ) from error


def _require_callable_method(owner: object, method_name: str) -> None:
    try:
        method = cast(object, getattr(owner, method_name))
    except AttributeError as error:
        raise Glm47TraceConfigurationError(
            f"{type(owner).__name__} is missing method {method_name}"
        ) from error
    if not callable(method):
        raise Glm47TraceConfigurationError(
            f"{type(owner).__name__}.{method_name} is not callable"
        )
