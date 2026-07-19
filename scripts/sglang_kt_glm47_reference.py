"""Memory-bounded GLM-4.7 layer-one BF16 expert reference computation."""

from __future__ import annotations

import hashlib
import importlib
import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast, final

GLM47_ROUTED_EXPERT_COUNT: Final = 64
GLM47_LAYER_ONE_ROUTE_WEIGHTS: Final = (0.4, 0.3, 0.2, 0.1)
GLM47_BF16_DTYPE: Final = "torch.bfloat16"
GLM47_GATE_UP_WEIGHT_SHAPE: Final = (1_536, 2_048)
GLM47_DOWN_WEIGHT_SHAPE: Final = (2_048, 1_536)
GLM47_LAYER_ONE_OUTPUT_SHAPE: Final = (1, 2_048)
GLM47_FLOAT32_DTYPE: Final = "torch.float32"
GLM47_LAYER_ONE_RELATIVE_L1_TOLERANCE: Final = 0.02

_TENSOR_SHA256_DOMAIN = b"exo.sglang-kt.tensor-sha256.v1\0"
_MAXIMUM_UINT64 = (1 << 64) - 1

type Glm47TensorDtype = Literal["torch.bfloat16", "torch.float32"]


class Glm47ReferenceError(ValueError):
    """Raised when a layer-one expert reference is not exact."""


def _tensor_element_size(dtype: Glm47TensorDtype) -> int:
    if dtype == GLM47_BF16_DTYPE:
        return 2
    if dtype == GLM47_FLOAT32_DTYPE:
        return 4
    raise Glm47ReferenceError(f"unsupported tensor dtype: {dtype}")


def _validate_tensor_shape(shape: tuple[int, ...]) -> int:
    if not shape:
        raise Glm47ReferenceError("tensor shape must have at least one dimension")
    element_count = 1
    for dimension in shape:
        if type(dimension) is not int:
            raise TypeError("tensor dimensions must be integers")
        if dimension <= 0 or dimension > _MAXIMUM_UINT64:
            raise Glm47ReferenceError(
                "tensor dimensions must be positive unsigned 64-bit integers"
            )
        element_count *= dimension
        if element_count > _MAXIMUM_UINT64:
            raise Glm47ReferenceError("tensor element count exceeds unsigned 64-bit")
    return element_count


def calculate_glm47_tensor_sha256(
    *,
    dtype: Glm47TensorDtype,
    shape: tuple[int, ...],
    little_endian_contiguous_bytes: bytes,
) -> str:
    """Hash a tensor using Exo's stable v1 tensor convention.

    The SHA-256 preimage is the ASCII domain
    ``exo.sglang-kt.tensor-sha256.v1\\0`` followed by: a uint16 little-endian
    dtype-name length and the ASCII Torch dtype name; a uint16 little-endian
    rank; each dimension as uint64 little-endian; the raw byte length as
    uint64 little-endian; and contiguous row-major element bytes in canonical
    little-endian order. BF16 elements occupy two bytes and FP32 elements four.
    """

    if dtype not in (GLM47_BF16_DTYPE, GLM47_FLOAT32_DTYPE):
        raise Glm47ReferenceError(f"unsupported tensor dtype: {dtype}")
    element_count = _validate_tensor_shape(shape)
    if type(little_endian_contiguous_bytes) is not bytes:
        raise TypeError("tensor contents must be immutable bytes")
    expected_bytes = element_count * _tensor_element_size(dtype)
    if len(little_endian_contiguous_bytes) != expected_bytes:
        raise Glm47ReferenceError(
            f"tensor contents have {len(little_endian_contiguous_bytes)} bytes, "
            f"expected {expected_bytes}"
        )
    if len(shape) > 0xFFFF:
        raise Glm47ReferenceError("tensor rank exceeds unsigned 16-bit")

    dtype_bytes = dtype.encode("ascii")
    digest = hashlib.sha256()
    digest.update(_TENSOR_SHA256_DOMAIN)
    digest.update(struct.pack("<H", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack("<H", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<Q", dimension))
    digest.update(struct.pack("<Q", len(little_endian_contiguous_bytes)))
    digest.update(little_endian_contiguous_bytes)
    return digest.hexdigest()


@final
@dataclass(frozen=True)
class Glm47TensorSnapshot:
    """A dtype/shape-bound copy of one small tensor in canonical byte order."""

    dtype: Glm47TensorDtype
    shape: tuple[int, ...]
    little_endian_contiguous_bytes: bytes

    def __post_init__(self) -> None:
        calculate_glm47_tensor_sha256(
            dtype=self.dtype,
            shape=self.shape,
            little_endian_contiguous_bytes=self.little_endian_contiguous_bytes,
        )

    @property
    def sha256(self) -> str:
        return calculate_glm47_tensor_sha256(
            dtype=self.dtype,
            shape=self.shape,
            little_endian_contiguous_bytes=self.little_endian_contiguous_bytes,
        )

    def finite_values(self) -> tuple[float, ...]:
        """Decode canonical bytes and reject all non-finite elements."""

        if self.dtype == GLM47_FLOAT32_DTYPE:
            values = cast(
                tuple[float, ...],
                struct.unpack(
                    f"<{len(self.little_endian_contiguous_bytes) // 4}f",
                    self.little_endian_contiguous_bytes,
                ),
            )
        else:
            bit_patterns = cast(
                tuple[int, ...],
                struct.unpack(
                    f"<{len(self.little_endian_contiguous_bytes) // 2}H",
                    self.little_endian_contiguous_bytes,
                ),
            )
            values = tuple(
                cast(
                    float,
                    struct.unpack("<f", struct.pack("<I", bits << 16))[0],
                )
                for bits in bit_patterns
            )
        if not all(math.isfinite(value) for value in values):
            raise Glm47ReferenceError("tensor contains non-finite values")
        return tuple(values)


@final
@dataclass(frozen=True)
class Glm47NumericalEvidence:
    """Receipt-ready numerical comparison for one 1x2048 output."""

    shape: tuple[int, int]
    execution_dtype: Literal["torch.bfloat16"]
    reference_dtype: Literal["torch.float32"]
    output_finite: Literal[True]
    reference_finite: Literal[True]
    mean_absolute_error: float
    maximum_absolute_error: float
    reference_mean_absolute: float
    relative_l1_error: float
    relative_l1_tolerance: float

    def __post_init__(self) -> None:
        metrics = (
            self.mean_absolute_error,
            self.maximum_absolute_error,
            self.reference_mean_absolute,
            self.relative_l1_error,
            self.relative_l1_tolerance,
        )
        if (
            self.shape != GLM47_LAYER_ONE_OUTPUT_SHAPE
            or self.execution_dtype != GLM47_BF16_DTYPE
            or self.reference_dtype != GLM47_FLOAT32_DTYPE
            or self.output_finite is not True
            or self.reference_finite is not True
            or not all(math.isfinite(metric) for metric in metrics)
            or self.mean_absolute_error < 0
            or self.maximum_absolute_error < 0
            or self.reference_mean_absolute <= 0
            or self.relative_l1_error < 0
            or self.relative_l1_tolerance != GLM47_LAYER_ONE_RELATIVE_L1_TOLERANCE
            or self.relative_l1_error > self.relative_l1_tolerance
        ):
            raise Glm47ReferenceError("numerical evidence is not admissible")

    def as_json(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "execution_dtype": self.execution_dtype,
            "reference_dtype": self.reference_dtype,
            "output_finite": self.output_finite,
            "reference_finite": self.reference_finite,
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_absolute_error": self.maximum_absolute_error,
            "reference_mean_absolute": self.reference_mean_absolute,
            "relative_l1_error": self.relative_l1_error,
            "relative_l1_tolerance": self.relative_l1_tolerance,
        }


@final
@dataclass(frozen=True)
class Glm47LayerOneOutputEvidence:
    """Receipt-ready repeat and oracle evidence for one backend output."""

    actual_output_sha256: str
    repeat_actual_output_sha256: str
    reference_output_sha256: str
    numerical: Glm47NumericalEvidence

    def __post_init__(self) -> None:
        for digest in (
            self.actual_output_sha256,
            self.repeat_actual_output_sha256,
            self.reference_output_sha256,
        ):
            _validate_sha256(digest)
        if self.repeat_actual_output_sha256 != self.actual_output_sha256:
            raise Glm47ReferenceError("repeated output hash does not match")

    def as_json(self) -> dict[str, object]:
        return {
            "actual_output_sha256": self.actual_output_sha256,
            "repeat_actual_output_sha256": self.repeat_actual_output_sha256,
            "reference_output_sha256": self.reference_output_sha256,
            "numerical": self.numerical.as_json(),
        }


@final
@dataclass(frozen=True)
class Glm47LayerOneHybridMergeEvidence:
    """Receipt-ready exact BF16 merge and FP32 merge-reference evidence."""

    combined_output_sha256: str
    cpu_output_sha256: str
    gpu_output_sha256: str
    merged_backend_output_sha256: str
    repeat_merged_backend_output_sha256: str
    reference_merged_backend_output_sha256: str
    numerical: Glm47NumericalEvidence

    def __post_init__(self) -> None:
        for digest in (
            self.combined_output_sha256,
            self.cpu_output_sha256,
            self.gpu_output_sha256,
            self.merged_backend_output_sha256,
            self.repeat_merged_backend_output_sha256,
            self.reference_merged_backend_output_sha256,
        ):
            _validate_sha256(digest)
        if self.merged_backend_output_sha256 != self.combined_output_sha256:
            raise Glm47ReferenceError(
                "BF16 CPU/GPU merge does not match combined output"
            )
        if (
            self.repeat_merged_backend_output_sha256
            != self.merged_backend_output_sha256
        ):
            raise Glm47ReferenceError("repeated BF16 CPU/GPU merge does not match")

    def as_json(self) -> dict[str, object]:
        return {
            "combined_output_sha256": self.combined_output_sha256,
            "cpu_output_sha256": self.cpu_output_sha256,
            "gpu_output_sha256": self.gpu_output_sha256,
            "merged_backend_output_sha256": self.merged_backend_output_sha256,
            "repeat_merged_backend_output_sha256": (
                self.repeat_merged_backend_output_sha256
            ),
            "reference_merged_backend_output_sha256": (
                self.reference_merged_backend_output_sha256
            ),
            "numerical": self.numerical.as_json(),
        }


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise Glm47ReferenceError("tensor SHA-256 must be lowercase 64-hex")


@final
@dataclass(frozen=True)
class Glm47LayerOneExpertTensorKeys:
    """Canonical BF16 checkpoint keys for one routed expert."""

    expert_id: int
    gate: str
    up: str
    down: str

    @property
    def all(self) -> tuple[str, str, str]:
        return (self.gate, self.up, self.down)


@final
@dataclass(frozen=True)
class Glm47LayerOneReferenceOutputs[TensorT]:
    """Weighted reference totals for the complete and backend-partitioned route."""

    combined: TensorT
    cpu_only: TensorT | None
    gpu_only: TensorT | None


class Glm47Bf16TensorLoader[TensorT](Protocol):
    """Narrow loading surface required from a BF16 safetensor loader."""

    def load_tensor(self, key: str, device: str = "cpu") -> TensorT: ...

    def close_all_handles(self) -> None: ...


class Glm47Bf16TensorLoaderFactory[TensorT](Protocol):
    """Factory used to defer heavyweight loader imports and construction."""

    def __call__(self, model_path: Path) -> Glm47Bf16TensorLoader[TensorT]: ...


class Glm47ReferenceComputationBackend[TensorT](Protocol):
    """Tensor validation and arithmetic used by the reference computation."""

    def validate_bf16_weight(
        self,
        tensor: TensorT,
        *,
        key: str,
        expected_shape: tuple[int, int],
        expected_dtype: str,
    ) -> None: ...

    def compute_expert_output(
        self,
        hidden_states: TensorT,
        gate_weight: TensorT,
        up_weight: TensorT,
        down_weight: TensorT,
    ) -> TensorT: ...

    def accumulate_weighted_output(
        self,
        accumulated: TensorT | None,
        expert_output: TensorT,
        *,
        route_weight: float,
    ) -> TensorT: ...


class Glm47TensorEvidenceBackend[TensorT](Protocol):
    """Capture and merge small tensors without coupling pure evidence to Torch."""

    def snapshot(
        self,
        tensor: TensorT,
        *,
        expected_dtype: Glm47TensorDtype,
        expected_shape: tuple[int, ...],
    ) -> Glm47TensorSnapshot: ...

    def merge_bfloat16(self, left: TensorT, right: TensorT) -> TensorT: ...

    def merge_float32(self, left: TensorT, right: TensorT) -> TensorT: ...


def calculate_glm47_numerical_evidence(
    actual_output: Glm47TensorSnapshot,
    reference_output: Glm47TensorSnapshot,
) -> Glm47NumericalEvidence:
    """Compare an exact BF16 output with its finite FP32 1x2048 reference."""

    if (
        actual_output.dtype != GLM47_BF16_DTYPE
        or reference_output.dtype != GLM47_FLOAT32_DTYPE
        or actual_output.shape != GLM47_LAYER_ONE_OUTPUT_SHAPE
        or reference_output.shape != GLM47_LAYER_ONE_OUTPUT_SHAPE
    ):
        raise Glm47ReferenceError(
            "numerical evidence requires BF16 and FP32 1x2048 tensors"
        )
    actual_values = actual_output.finite_values()
    reference_values = reference_output.finite_values()
    absolute_errors = tuple(
        abs(actual - reference)
        for actual, reference in zip(
            actual_values,
            reference_values,
            strict=True,
        )
    )
    error_l1 = math.fsum(absolute_errors)
    reference_l1 = math.fsum(abs(value) for value in reference_values)
    if reference_l1 <= 0 or not math.isfinite(reference_l1):
        raise Glm47ReferenceError("FP32 reference must have positive finite L1 norm")
    relative_l1_error = error_l1 / reference_l1
    metrics = (
        error_l1,
        max(absolute_errors),
        relative_l1_error,
    )
    if not all(math.isfinite(metric) for metric in metrics):
        raise Glm47ReferenceError("numerical comparison produced non-finite metrics")
    if relative_l1_error > GLM47_LAYER_ONE_RELATIVE_L1_TOLERANCE:
        raise Glm47ReferenceError(
            "BF16 output exceeds the layer-one relative L1 tolerance"
        )
    element_count = len(reference_values)
    return Glm47NumericalEvidence(
        shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
        execution_dtype=GLM47_BF16_DTYPE,
        reference_dtype=GLM47_FLOAT32_DTYPE,
        output_finite=True,
        reference_finite=True,
        mean_absolute_error=error_l1 / element_count,
        maximum_absolute_error=max(absolute_errors),
        reference_mean_absolute=reference_l1 / element_count,
        relative_l1_error=relative_l1_error,
        relative_l1_tolerance=GLM47_LAYER_ONE_RELATIVE_L1_TOLERANCE,
    )


def require_matching_glm47_tensor_sha256(
    original: Glm47TensorSnapshot,
    repeat: Glm47TensorSnapshot,
    *,
    description: str,
) -> str:
    """Return a common tensor hash or reject a nondeterministic repeat."""

    if not description or description.strip() != description:
        raise ValueError("repeat description must be nonempty and trimmed")
    original_sha256 = original.sha256
    repeat_sha256 = repeat.sha256
    if repeat_sha256 != original_sha256:
        raise Glm47ReferenceError(f"repeated {description} hash does not match")
    return original_sha256


def build_glm47_layer_one_output_evidence[TensorT](
    *,
    actual_output: TensorT,
    repeat_actual_output: TensorT,
    reference_output: TensorT,
    backend: Glm47TensorEvidenceBackend[TensorT],
) -> Glm47LayerOneOutputEvidence:
    """Build strict repeat and FP32-oracle evidence for one BF16 output."""

    actual_snapshot = backend.snapshot(
        actual_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    repeat_snapshot = backend.snapshot(
        repeat_actual_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    reference_snapshot = backend.snapshot(
        reference_output,
        expected_dtype=GLM47_FLOAT32_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    actual_sha256 = require_matching_glm47_tensor_sha256(
        actual_snapshot,
        repeat_snapshot,
        description="layer-one backend output",
    )
    return Glm47LayerOneOutputEvidence(
        actual_output_sha256=actual_sha256,
        repeat_actual_output_sha256=repeat_snapshot.sha256,
        reference_output_sha256=reference_snapshot.sha256,
        numerical=calculate_glm47_numerical_evidence(
            actual_snapshot,
            reference_snapshot,
        ),
    )


def build_glm47_layer_one_hybrid_merge_evidence[TensorT](
    *,
    combined_output: TensorT,
    cpu_output: TensorT,
    gpu_output: TensorT,
    repeat_cpu_output: TensorT,
    repeat_gpu_output: TensorT,
    backend: Glm47TensorEvidenceBackend[TensorT],
) -> Glm47LayerOneHybridMergeEvidence:
    """Prove an exact repeatable BF16 merge and its FP32 consistency."""

    combined_snapshot = backend.snapshot(
        combined_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    cpu_snapshot = backend.snapshot(
        cpu_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    gpu_snapshot = backend.snapshot(
        gpu_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    cpu_snapshot.finite_values()
    gpu_snapshot.finite_values()
    merged_output = backend.merge_bfloat16(cpu_output, gpu_output)
    repeat_merged_output = backend.merge_bfloat16(
        repeat_cpu_output,
        repeat_gpu_output,
    )
    reference_merged_output = backend.merge_float32(cpu_output, gpu_output)
    merged_snapshot = backend.snapshot(
        merged_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    repeat_merged_snapshot = backend.snapshot(
        repeat_merged_output,
        expected_dtype=GLM47_BF16_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    reference_merged_snapshot = backend.snapshot(
        reference_merged_output,
        expected_dtype=GLM47_FLOAT32_DTYPE,
        expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
    )
    merged_sha256 = require_matching_glm47_tensor_sha256(
        merged_snapshot,
        repeat_merged_snapshot,
        description="layer-one BF16 CPU/GPU merge",
    )
    if merged_sha256 != combined_snapshot.sha256:
        raise Glm47ReferenceError(
            "layer-one BF16 CPU/GPU merge does not match combined output"
        )
    return Glm47LayerOneHybridMergeEvidence(
        combined_output_sha256=combined_snapshot.sha256,
        cpu_output_sha256=cpu_snapshot.sha256,
        gpu_output_sha256=gpu_snapshot.sha256,
        merged_backend_output_sha256=merged_sha256,
        repeat_merged_backend_output_sha256=repeat_merged_snapshot.sha256,
        reference_merged_backend_output_sha256=reference_merged_snapshot.sha256,
        numerical=calculate_glm47_numerical_evidence(
            combined_snapshot,
            reference_merged_snapshot,
        ),
    )


def _validate_selected_expert_ids(
    selected_expert_ids: tuple[int, ...],
) -> None:
    if len(selected_expert_ids) != len(GLM47_LAYER_ONE_ROUTE_WEIGHTS):
        raise Glm47ReferenceError("exactly four selected expert IDs are required")
    if any(type(expert_id) is not int for expert_id in selected_expert_ids):
        raise TypeError("selected expert IDs must be integers")
    if len(set(selected_expert_ids)) != len(selected_expert_ids):
        raise Glm47ReferenceError("selected expert IDs must be unique")
    if any(
        expert_id < 0 or expert_id >= GLM47_ROUTED_EXPERT_COUNT
        for expert_id in selected_expert_ids
    ):
        raise Glm47ReferenceError("selected expert IDs must be between 0 and 63")


def build_glm47_layer_one_bf16_expert_tensor_keys(
    selected_expert_ids: tuple[int, ...],
) -> tuple[Glm47LayerOneExpertTensorKeys, ...]:
    """Build the canonical three-key group for each of four selected experts."""

    _validate_selected_expert_ids(selected_expert_ids)
    return tuple(
        Glm47LayerOneExpertTensorKeys(
            expert_id=expert_id,
            gate=(f"model.layers.1.mlp.experts.{expert_id}.gate_proj.weight"),
            up=f"model.layers.1.mlp.experts.{expert_id}.up_proj.weight",
            down=f"model.layers.1.mlp.experts.{expert_id}.down_proj.weight",
        )
        for expert_id in selected_expert_ids
    )


def build_glm47_layer_one_bf16_tensor_keys(
    selected_expert_ids: tuple[int, ...],
) -> tuple[str, ...]:
    """Build all 12 canonical layer-one BF16 tensor keys in route order."""

    return tuple(
        key
        for expert_keys in build_glm47_layer_one_bf16_expert_tensor_keys(
            selected_expert_ids
        )
        for key in expert_keys.all
    )


def _validate_route_partition(
    selected_expert_ids: tuple[int, ...],
    cpu_expert_ids: tuple[int, ...],
    gpu_expert_ids: tuple[int, ...],
) -> None:
    for description, expert_ids in (
        ("CPU", cpu_expert_ids),
        ("GPU", gpu_expert_ids),
    ):
        if any(type(expert_id) is not int for expert_id in expert_ids):
            raise TypeError(f"{description} expert IDs must be integers")
        if len(set(expert_ids)) != len(expert_ids):
            raise Glm47ReferenceError(f"{description} expert IDs must be unique")

    cpu_set = frozenset(cpu_expert_ids)
    gpu_set = frozenset(gpu_expert_ids)
    selected_set = frozenset(selected_expert_ids)
    if cpu_set & gpu_set or cpu_set | gpu_set != selected_set:
        raise Glm47ReferenceError(
            "CPU and GPU expert IDs must exactly partition the selected route"
        )
    expected_cpu_order = tuple(
        expert_id for expert_id in selected_expert_ids if expert_id in cpu_set
    )
    expected_gpu_order = tuple(
        expert_id for expert_id in selected_expert_ids if expert_id in gpu_set
    )
    if cpu_expert_ids != expected_cpu_order or gpu_expert_ids != expected_gpu_order:
        raise Glm47ReferenceError(
            "CPU and GPU expert IDs must preserve selected-route order"
        )


def compute_glm47_layer_one_reference[TensorT](
    *,
    model_path: Path,
    selected_expert_ids: tuple[int, ...],
    cpu_expert_ids: tuple[int, ...],
    gpu_expert_ids: tuple[int, ...],
    hidden_states: TensorT,
    loader_factory: Glm47Bf16TensorLoaderFactory[TensorT],
    backend: Glm47ReferenceComputationBackend[TensorT],
) -> Glm47LayerOneReferenceOutputs[TensorT]:
    """Compute a four-expert reference while retaining one expert at a time."""

    expert_key_groups = build_glm47_layer_one_bf16_expert_tensor_keys(
        selected_expert_ids
    )
    _validate_route_partition(
        selected_expert_ids,
        cpu_expert_ids,
        gpu_expert_ids,
    )
    cpu_set = frozenset(cpu_expert_ids)
    gpu_set = frozenset(gpu_expert_ids)

    loader = loader_factory(model_path)
    combined_output: TensorT | None = None
    cpu_output: TensorT | None = None
    gpu_output: TensorT | None = None
    try:
        for expert_keys, route_weight in zip(
            expert_key_groups,
            GLM47_LAYER_ONE_ROUTE_WEIGHTS,
            strict=True,
        ):
            gate_weight = loader.load_tensor(expert_keys.gate, "cpu")
            backend.validate_bf16_weight(
                gate_weight,
                key=expert_keys.gate,
                expected_shape=GLM47_GATE_UP_WEIGHT_SHAPE,
                expected_dtype=GLM47_BF16_DTYPE,
            )
            up_weight = loader.load_tensor(expert_keys.up, "cpu")
            backend.validate_bf16_weight(
                up_weight,
                key=expert_keys.up,
                expected_shape=GLM47_GATE_UP_WEIGHT_SHAPE,
                expected_dtype=GLM47_BF16_DTYPE,
            )
            down_weight = loader.load_tensor(expert_keys.down, "cpu")
            backend.validate_bf16_weight(
                down_weight,
                key=expert_keys.down,
                expected_shape=GLM47_DOWN_WEIGHT_SHAPE,
                expected_dtype=GLM47_BF16_DTYPE,
            )

            expert_output = backend.compute_expert_output(
                hidden_states,
                gate_weight,
                up_weight,
                down_weight,
            )
            combined_output = backend.accumulate_weighted_output(
                combined_output,
                expert_output,
                route_weight=route_weight,
            )
            if expert_keys.expert_id in cpu_set:
                cpu_output = backend.accumulate_weighted_output(
                    cpu_output,
                    expert_output,
                    route_weight=route_weight,
                )
            elif expert_keys.expert_id in gpu_set:
                gpu_output = backend.accumulate_weighted_output(
                    gpu_output,
                    expert_output,
                    route_weight=route_weight,
                )
            else:
                raise AssertionError("validated route partition lost an expert")

            del gate_weight, up_weight, down_weight, expert_output
    finally:
        loader.close_all_handles()

    if combined_output is None:
        raise AssertionError("four-expert reference produced no combined output")
    return Glm47LayerOneReferenceOutputs(
        combined=combined_output,
        cpu_only=cpu_output,
        gpu_only=gpu_output,
    )


class _TorchTensor(Protocol):
    @property
    def shape(self) -> Iterable[int]: ...

    @property
    def dtype(self) -> object: ...

    def float(self) -> _TorchTensor: ...

    def detach(self) -> _TorchTensor: ...

    def contiguous(self) -> _TorchTensor: ...

    def cpu(self) -> _TorchTensor: ...

    def view(self, dtype: object) -> _TorchTensor: ...

    def reshape(self, *shape: int) -> _TorchTensor: ...

    def tolist(self) -> object: ...

    def __mul__(self, other: object) -> _TorchTensor: ...

    def __add__(self, other: object) -> _TorchTensor: ...


type _LinearOperation = Callable[[_TorchTensor, _TorchTensor], _TorchTensor]
type _UnaryTensorOperation = Callable[[_TorchTensor], _TorchTensor]


@final
class TorchGlm47ReferenceComputationBackend(Glm47ReferenceComputationBackend[object]):
    """Float32 Torch reference math with exact BF16 checkpoint validation."""

    def __init__(
        self,
        *,
        bfloat16_dtype: object,
        linear: _LinearOperation,
        silu: _UnaryTensorOperation,
    ) -> None:
        self._bfloat16_dtype = bfloat16_dtype
        self._linear = linear
        self._silu = silu

    def validate_bf16_weight(
        self,
        tensor: object,
        *,
        key: str,
        expected_shape: tuple[int, int],
        expected_dtype: str,
    ) -> None:
        torch_tensor = cast(_TorchTensor, tensor)
        try:
            actual_shape = tuple(torch_tensor.shape)
        except (TypeError, ValueError) as error:
            raise Glm47ReferenceError(f"weight has an invalid shape: {key}") from error
        if actual_shape != expected_shape:
            raise Glm47ReferenceError(
                f"weight {key} has shape {actual_shape}, expected {expected_shape}"
            )
        if (
            torch_tensor.dtype != self._bfloat16_dtype
            or str(torch_tensor.dtype) != expected_dtype
        ):
            raise Glm47ReferenceError(
                f"weight {key} has dtype {torch_tensor.dtype}, "
                f"expected {expected_dtype}"
            )

    def compute_expert_output(
        self,
        hidden_states: object,
        gate_weight: object,
        up_weight: object,
        down_weight: object,
    ) -> object:
        hidden_float = cast(_TorchTensor, hidden_states).float()
        gate_projection = self._linear(
            hidden_float,
            cast(_TorchTensor, gate_weight).float(),
        )
        up_projection = self._linear(
            hidden_float,
            cast(_TorchTensor, up_weight).float(),
        )
        intermediate = self._silu(gate_projection) * up_projection
        return self._linear(
            intermediate,
            cast(_TorchTensor, down_weight).float(),
        )

    def accumulate_weighted_output(
        self,
        accumulated: object | None,
        expert_output: object,
        *,
        route_weight: float,
    ) -> object:
        weighted = cast(_TorchTensor, expert_output) * route_weight
        if accumulated is None:
            return weighted
        return cast(_TorchTensor, accumulated) + weighted


@final
class TorchGlm47TensorEvidenceBackend(Glm47TensorEvidenceBackend[object]):
    """Copy small Torch tensors into stable, host-independent evidence."""

    def __init__(
        self,
        *,
        bfloat16_dtype: object,
        float32_dtype: object,
        int16_dtype: object,
        int32_dtype: object,
    ) -> None:
        self._bfloat16_dtype = bfloat16_dtype
        self._float32_dtype = float32_dtype
        self._int16_dtype = int16_dtype
        self._int32_dtype = int32_dtype

    def _expected_torch_dtype(self, dtype: Glm47TensorDtype) -> object:
        if dtype == GLM47_BF16_DTYPE:
            return self._bfloat16_dtype
        if dtype == GLM47_FLOAT32_DTYPE:
            return self._float32_dtype
        raise Glm47ReferenceError(f"unsupported tensor dtype: {dtype}")

    def _validate_identity(
        self,
        tensor: object,
        *,
        expected_dtype: Glm47TensorDtype,
        expected_shape: tuple[int, ...],
    ) -> _TorchTensor:
        torch_tensor = cast(_TorchTensor, tensor)
        try:
            actual_shape = tuple(torch_tensor.shape)
            actual_dtype = torch_tensor.dtype
        except (AttributeError, TypeError, ValueError) as error:
            raise Glm47ReferenceError(
                "value is not a supported Torch tensor"
            ) from error
        if actual_shape != expected_shape:
            raise Glm47ReferenceError(
                f"tensor has shape {actual_shape}, expected {expected_shape}"
            )
        if (
            actual_dtype != self._expected_torch_dtype(expected_dtype)
            or str(actual_dtype) != expected_dtype
        ):
            raise Glm47ReferenceError(
                f"tensor has dtype {actual_dtype}, expected {expected_dtype}"
            )
        return torch_tensor

    def snapshot(
        self,
        tensor: object,
        *,
        expected_dtype: Glm47TensorDtype,
        expected_shape: tuple[int, ...],
    ) -> Glm47TensorSnapshot:
        torch_tensor = self._validate_identity(
            tensor,
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
        )
        host_tensor = torch_tensor.detach().contiguous().cpu()
        integer_dtype = (
            self._int16_dtype
            if expected_dtype == GLM47_BF16_DTYPE
            else self._int32_dtype
        )
        raw_values = host_tensor.view(integer_dtype).reshape(-1).tolist()
        if not isinstance(raw_values, list):
            raise Glm47ReferenceError("Torch tensor bit view is not a flat list")
        typed_values = cast(list[object], raw_values)
        if any(type(value) is not int for value in typed_values):
            raise Glm47ReferenceError("Torch tensor bit view is not integral")
        byte_width = _tensor_element_size(expected_dtype)
        mask = (1 << (byte_width * 8)) - 1
        little_endian_bytes = b"".join(
            (cast(int, value) & mask).to_bytes(byte_width, "little")
            for value in typed_values
        )
        return Glm47TensorSnapshot(
            dtype=expected_dtype,
            shape=expected_shape,
            little_endian_contiguous_bytes=little_endian_bytes,
        )

    def merge_bfloat16(self, left: object, right: object) -> object:
        left_tensor = self._validate_identity(
            left,
            expected_dtype=GLM47_BF16_DTYPE,
            expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
        )
        right_tensor = self._validate_identity(
            right,
            expected_dtype=GLM47_BF16_DTYPE,
            expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
        )
        return left_tensor + right_tensor

    def merge_float32(self, left: object, right: object) -> object:
        left_tensor = self._validate_identity(
            left,
            expected_dtype=GLM47_BF16_DTYPE,
            expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
        )
        right_tensor = self._validate_identity(
            right,
            expected_dtype=GLM47_BF16_DTYPE,
            expected_shape=GLM47_LAYER_ONE_OUTPUT_SHAPE,
        )
        return left_tensor.float() + right_tensor.float()


class _LoaderConstructor(Protocol):
    def __call__(self, model_path: str) -> object: ...


@final
class LazyBf16SafeTensorLoaderFactory(Glm47Bf16TensorLoaderFactory[object]):
    """Construct the pinned KT BF16 loader without import-time Torch coupling."""

    def __call__(self, model_path: Path) -> Glm47Bf16TensorLoader[object]:
        importlib.import_module("torch")
        loader_module = importlib.import_module("kt_kernel.utils.loader")
        constructor = cast(
            _LoaderConstructor,
            loader_module.BF16SafeTensorLoader,
        )
        return cast(
            Glm47Bf16TensorLoader[object],
            constructor(str(model_path)),
        )


def create_torch_glm47_reference_backend() -> Glm47ReferenceComputationBackend[object]:
    """Lazily bind the Torch operations used by the concrete reference backend."""

    torch_module = importlib.import_module("torch")
    functional_module = importlib.import_module("torch.nn.functional")
    return TorchGlm47ReferenceComputationBackend(
        bfloat16_dtype=cast(object, torch_module.bfloat16),
        linear=cast(_LinearOperation, functional_module.linear),
        silu=cast(_UnaryTensorOperation, functional_module.silu),
    )


def create_torch_glm47_tensor_evidence_backend() -> Glm47TensorEvidenceBackend[object]:
    """Lazily bind Torch dtypes used for stable tensor evidence."""

    torch_module = importlib.import_module("torch")
    return TorchGlm47TensorEvidenceBackend(
        bfloat16_dtype=cast(object, torch_module.bfloat16),
        float32_dtype=cast(object, torch_module.float32),
        int16_dtype=cast(object, torch_module.int16),
        int32_dtype=cast(object, torch_module.int32),
    )


def compute_torch_glm47_layer_one_reference(
    *,
    model_path: Path,
    selected_expert_ids: tuple[int, ...],
    cpu_expert_ids: tuple[int, ...],
    gpu_expert_ids: tuple[int, ...],
    hidden_states: object,
) -> Glm47LayerOneReferenceOutputs[object]:
    """Compute the concrete Torch reference using the pinned KT BF16 loader."""

    return compute_glm47_layer_one_reference(
        model_path=model_path,
        selected_expert_ids=selected_expert_ids,
        cpu_expert_ids=cpu_expert_ids,
        gpu_expert_ids=gpu_expert_ids,
        hidden_states=hidden_states,
        loader_factory=LazyBf16SafeTensorLoaderFactory(),
        backend=create_torch_glm47_reference_backend(),
    )
