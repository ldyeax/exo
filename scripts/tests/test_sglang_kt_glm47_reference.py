from __future__ import annotations

import math
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast, final

import pytest

from scripts import sglang_kt_glm47_reference as reference

MODEL_PATH = Path("/var/lib/exo/models/glm-4.7-flash-bf16")
SELECTED_EXPERTS = (0, 1, 4, 5)


@final
@dataclass(frozen=True)
class FakeTensor:
    value: float
    expert_id: int
    projection: str


@final
class FakeLoader(reference.Glm47Bf16TensorLoader[FakeTensor]):
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_on_load: str | None = None,
    ) -> None:
        self.events = events
        self.fail_on_load = fail_on_load
        self.close_count = 0

    def load_tensor(self, key: str, device: str = "cpu") -> FakeTensor:
        self.events.append(("load", key, device))
        if key == self.fail_on_load:
            raise RuntimeError("load failed")
        components = key.split(".")
        expert_id = int(components[5])
        projection = components[6]
        return FakeTensor(
            value=float(expert_id + 1),
            expert_id=expert_id,
            projection=projection,
        )

    def close_all_handles(self) -> None:
        self.close_count += 1
        self.events.append(("close",))


@final
class FakeLoaderFactory(reference.Glm47Bf16TensorLoaderFactory[FakeTensor]):
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_on_load: str | None = None,
    ) -> None:
        self.events = events
        self.loader = FakeLoader(events, fail_on_load=fail_on_load)
        self.paths: list[Path] = []

    def __call__(self, model_path: Path) -> FakeLoader:
        self.paths.append(model_path)
        self.events.append(("open", model_path))
        return self.loader


@final
class FakeBackend(reference.Glm47ReferenceComputationBackend[FakeTensor]):
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        fail_stage: str | None = None,
    ) -> None:
        self.events = events
        self.fail_stage = fail_stage
        self.validations: list[tuple[str, tuple[int, int], str, int, str]] = []

    def validate_bf16_weight(
        self,
        tensor: FakeTensor,
        *,
        key: str,
        expected_shape: tuple[int, int],
        expected_dtype: str,
    ) -> None:
        self.events.append(("validate", tensor.expert_id, tensor.projection))
        self.validations.append(
            (
                key,
                expected_shape,
                expected_dtype,
                tensor.expert_id,
                tensor.projection,
            )
        )
        if self.fail_stage == "validate":
            raise RuntimeError("validation failed")

    def compute_expert_output(
        self,
        hidden_states: FakeTensor,
        gate_weight: FakeTensor,
        up_weight: FakeTensor,
        down_weight: FakeTensor,
    ) -> FakeTensor:
        assert hidden_states.projection == "hidden"
        assert gate_weight.expert_id == up_weight.expert_id == down_weight.expert_id
        self.events.append(("compute", gate_weight.expert_id))
        if self.fail_stage == "compute":
            raise RuntimeError("compute failed")
        return FakeTensor(
            value=gate_weight.value,
            expert_id=gate_weight.expert_id,
            projection="output",
        )

    def accumulate_weighted_output(
        self,
        accumulated: FakeTensor | None,
        expert_output: FakeTensor,
        *,
        route_weight: float,
    ) -> FakeTensor:
        self.events.append(
            (
                "accumulate",
                expert_output.expert_id,
                route_weight,
                None if accumulated is None else accumulated.value,
            )
        )
        if self.fail_stage == "accumulate":
            raise RuntimeError("accumulation failed")
        prior = 0.0 if accumulated is None else accumulated.value
        return FakeTensor(
            value=prior + expert_output.value * route_weight,
            expert_id=-1,
            projection="aggregate",
        )


@final
class FakeTorchDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


@final
@dataclass(frozen=True)
class FakeTorchTensor:
    value: float
    shape: tuple[int, ...]
    dtype: object

    def float(self) -> FakeTorchTensor:
        return FakeTorchTensor(self.value, self.shape, "torch.float32")

    def __mul__(self, other: object) -> FakeTorchTensor:
        multiplier = other.value if isinstance(other, FakeTorchTensor) else other
        if not isinstance(multiplier, (int, float)):
            return NotImplemented
        return FakeTorchTensor(
            self.value * multiplier,
            self.shape,
            self.dtype,
        )

    def __add__(self, other: object) -> FakeTorchTensor:
        if not isinstance(other, FakeTorchTensor):
            return NotImplemented
        return FakeTorchTensor(
            self.value + other.value,
            self.shape,
            self.dtype,
        )


@final
@dataclass(frozen=True)
class FakeEvidenceTensor:
    snapshot: reference.Glm47TensorSnapshot


@final
class FakeEvidenceBackend(reference.Glm47TensorEvidenceBackend[FakeEvidenceTensor]):
    def snapshot(
        self,
        tensor: FakeEvidenceTensor,
        *,
        expected_dtype: reference.Glm47TensorDtype,
        expected_shape: tuple[int, ...],
    ) -> reference.Glm47TensorSnapshot:
        if tensor.snapshot.dtype != expected_dtype:
            raise reference.Glm47ReferenceError("unexpected fake dtype")
        if tensor.snapshot.shape != expected_shape:
            raise reference.Glm47ReferenceError("unexpected fake shape")
        return tensor.snapshot

    def merge_bfloat16(
        self,
        left: FakeEvidenceTensor,
        right: FakeEvidenceTensor,
    ) -> FakeEvidenceTensor:
        left_value = left.snapshot.finite_values()[0]
        right_value = right.snapshot.finite_values()[0]
        return FakeEvidenceTensor(make_bfloat16_snapshot(left_value + right_value))

    def merge_float32(
        self,
        left: FakeEvidenceTensor,
        right: FakeEvidenceTensor,
    ) -> FakeEvidenceTensor:
        left_value = left.snapshot.finite_values()[0]
        right_value = right.snapshot.finite_values()[0]
        return FakeEvidenceTensor(make_float32_snapshot(left_value + right_value))


@final
@dataclass(frozen=True)
class FakeTorchBitTensor:
    bit_values: tuple[object, ...]
    shape: tuple[int, ...]
    dtype: object

    def detach(self) -> FakeTorchBitTensor:
        return self

    def contiguous(self) -> FakeTorchBitTensor:
        return self

    def cpu(self) -> FakeTorchBitTensor:
        return self

    def view(self, dtype: object) -> FakeTorchBitTensor:
        return FakeTorchBitTensor(self.bit_values, self.shape, dtype)

    def reshape(self, *shape: int) -> FakeTorchBitTensor:
        assert shape == (-1,)
        return FakeTorchBitTensor(
            self.bit_values,
            (len(self.bit_values),),
            self.dtype,
        )

    def tolist(self) -> object:
        return list(self.bit_values)


def make_bfloat16_snapshot(
    value: float,
    *,
    shape: tuple[int, ...] = reference.GLM47_LAYER_ONE_OUTPUT_SHAPE,
) -> reference.Glm47TensorSnapshot:
    element_count = math.prod(shape)
    float32_bits = cast(
        int,
        struct.unpack("<I", struct.pack("<f", value))[0],
    )
    bfloat16_bits = float32_bits >> 16
    return reference.Glm47TensorSnapshot(
        dtype=reference.GLM47_BF16_DTYPE,
        shape=shape,
        little_endian_contiguous_bytes=(
            struct.pack("<H", bfloat16_bits) * element_count
        ),
    )


def make_float32_snapshot(
    value: float,
    *,
    shape: tuple[int, ...] = reference.GLM47_LAYER_ONE_OUTPUT_SHAPE,
) -> reference.Glm47TensorSnapshot:
    return reference.Glm47TensorSnapshot(
        dtype=reference.GLM47_FLOAT32_DTYPE,
        shape=shape,
        little_endian_contiguous_bytes=(struct.pack("<f", value) * math.prod(shape)),
    )


def hidden_states() -> FakeTensor:
    return FakeTensor(value=1.0, expert_id=-1, projection="hidden")


def compute_reference(
    *,
    cpu_expert_ids: tuple[int, ...] = (4, 5),
    gpu_expert_ids: tuple[int, ...] = (0, 1),
    fail_on_load: str | None = None,
    fail_stage: str | None = None,
) -> tuple[
    reference.Glm47LayerOneReferenceOutputs[FakeTensor],
    FakeLoaderFactory,
    FakeBackend,
    list[tuple[object, ...]],
]:
    events: list[tuple[object, ...]] = []
    factory = FakeLoaderFactory(events, fail_on_load=fail_on_load)
    backend = FakeBackend(events, fail_stage=fail_stage)
    outputs = reference.compute_glm47_layer_one_reference(
        model_path=MODEL_PATH,
        selected_expert_ids=SELECTED_EXPERTS,
        cpu_expert_ids=cpu_expert_ids,
        gpu_expert_ids=gpu_expert_ids,
        hidden_states=hidden_states(),
        loader_factory=factory,
        backend=backend,
    )
    return outputs, factory, backend, events


def test_builds_exact_twelve_canonical_layer_one_bf16_keys() -> None:
    keys = reference.build_glm47_layer_one_bf16_tensor_keys(SELECTED_EXPERTS)

    assert len(keys) == 12
    assert keys == tuple(
        f"model.layers.1.mlp.experts.{expert_id}.{projection}_proj.weight"
        for expert_id in SELECTED_EXPERTS
        for projection in ("gate", "up", "down")
    )
    key_groups = reference.build_glm47_layer_one_bf16_expert_tensor_keys(
        SELECTED_EXPERTS
    )
    assert tuple(group.expert_id for group in key_groups) == SELECTED_EXPERTS
    assert tuple(key for group in key_groups for key in group.all) == keys


@pytest.mark.parametrize(
    "expert_ids",
    (
        (),
        (0, 1, 2),
        (0, 1, 2, 3, 4),
        (0, 1, 1, 2),
        (-1, 0, 1, 2),
        (0, 1, 2, 64),
    ),
)
def test_key_builder_rejects_noncanonical_expert_sets(
    expert_ids: tuple[int, ...],
) -> None:
    with pytest.raises(reference.Glm47ReferenceError):
        reference.build_glm47_layer_one_bf16_tensor_keys(expert_ids)

    with pytest.raises(TypeError):
        reference.build_glm47_layer_one_bf16_tensor_keys((0, 1, 2, True))


def test_computes_combined_cpu_and_gpu_totals_one_expert_at_a_time() -> None:
    outputs, factory, backend, events = compute_reference()

    assert math.isclose(outputs.combined.value, 2.6)
    assert outputs.cpu_only is not None
    assert math.isclose(outputs.cpu_only.value, 1.6)
    assert outputs.gpu_only is not None
    assert math.isclose(outputs.gpu_only.value, 1.0)
    assert factory.paths == [MODEL_PATH]
    assert factory.loader.close_count == 1
    assert events[-1] == ("close",)

    for expert_index, expert_id in enumerate(SELECTED_EXPERTS):
        next_expert_id = (
            SELECTED_EXPERTS[expert_index + 1]
            if expert_index + 1 < len(SELECTED_EXPERTS)
            else None
        )
        compute_index = events.index(("compute", expert_id))
        expert_accumulations = [
            index
            for index, event in enumerate(events)
            if event[0] == "accumulate" and event[1] == expert_id
        ]
        assert len(expert_accumulations) == 2
        assert compute_index < min(expert_accumulations)
        if next_expert_id is not None:
            next_load_index = next(
                index
                for index, event in enumerate(events)
                if event[0] == "load"
                and f"experts.{next_expert_id}." in cast(str, event[1])
            )
            assert max(expert_accumulations) < next_load_index

    assert len(backend.validations) == 12
    for key, shape, dtype, _expert_id, projection in backend.validations:
        assert dtype == reference.GLM47_BF16_DTYPE
        assert shape == (
            reference.GLM47_DOWN_WEIGHT_SHAPE
            if projection == "down_proj"
            else reference.GLM47_GATE_UP_WEIGHT_SHAPE
        )
        assert key.endswith(f".{projection}.weight")


@pytest.mark.parametrize(
    ("cpu_expert_ids", "gpu_expert_ids", "cpu_present", "gpu_present"),
    (
        (SELECTED_EXPERTS, (), True, False),
        ((), SELECTED_EXPERTS, False, True),
    ),
)
def test_computes_backend_only_reference_totals(
    cpu_expert_ids: tuple[int, ...],
    gpu_expert_ids: tuple[int, ...],
    cpu_present: bool,
    gpu_present: bool,
) -> None:
    outputs, _factory, _backend, _events = compute_reference(
        cpu_expert_ids=cpu_expert_ids,
        gpu_expert_ids=gpu_expert_ids,
    )

    assert (outputs.cpu_only is not None) is cpu_present
    assert (outputs.gpu_only is not None) is gpu_present
    partition_output = outputs.cpu_only if cpu_present else outputs.gpu_only
    assert partition_output is not None
    assert math.isclose(partition_output.value, outputs.combined.value)


@pytest.mark.parametrize(
    ("cpu_expert_ids", "gpu_expert_ids"),
    (
        ((0, 1), (1, 4, 5)),
        ((0, 1), (4,)),
        ((1, 0), (4, 5)),
        ((0, 1, True), (4,)),
    ),
)
def test_rejects_invalid_route_partition_before_opening_loader(
    cpu_expert_ids: tuple[int, ...],
    gpu_expert_ids: tuple[int, ...],
) -> None:
    events: list[tuple[object, ...]] = []
    factory = FakeLoaderFactory(events)

    with pytest.raises((TypeError, reference.Glm47ReferenceError)):
        reference.compute_glm47_layer_one_reference(
            model_path=MODEL_PATH,
            selected_expert_ids=SELECTED_EXPERTS,
            cpu_expert_ids=cpu_expert_ids,
            gpu_expert_ids=gpu_expert_ids,
            hidden_states=hidden_states(),
            loader_factory=factory,
            backend=FakeBackend(events),
        )

    assert events == []
    assert factory.paths == []


@pytest.mark.parametrize("failure_stage", ("load", "validate", "compute", "accumulate"))
def test_closes_all_loader_handles_after_every_failure(failure_stage: str) -> None:
    events: list[tuple[object, ...]] = []
    first_key = reference.build_glm47_layer_one_bf16_tensor_keys(SELECTED_EXPERTS)[0]
    factory = FakeLoaderFactory(
        events,
        fail_on_load=first_key if failure_stage == "load" else None,
    )
    backend = FakeBackend(
        events,
        fail_stage=None if failure_stage == "load" else failure_stage,
    )

    with pytest.raises(RuntimeError, match="failed"):
        reference.compute_glm47_layer_one_reference(
            model_path=MODEL_PATH,
            selected_expert_ids=SELECTED_EXPERTS,
            cpu_expert_ids=(4, 5),
            gpu_expert_ids=(0, 1),
            hidden_states=hidden_states(),
            loader_factory=factory,
            backend=backend,
        )

    assert factory.loader.close_count == 1
    assert events[-1] == ("close",)


def test_module_import_does_not_import_torch_or_kt_kernel() -> None:
    repository = Path(__file__).resolve().parents[2]
    program = f"""
import importlib.abc
import sys

sys.path.insert(0, {str(repository)!r})

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch is intentionally unavailable")
        if fullname == "kt_kernel" or fullname.startswith("kt_kernel."):
            raise ModuleNotFoundError("kt-kernel is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockHeavyImports())
import scripts.sglang_kt_glm47_reference
"""

    result = subprocess.run(
        (sys.executable, "-I", "-c", program),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_lazy_loader_factory_imports_dependencies_only_when_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    constructed_paths: list[str] = []
    loader = FakeLoader([])

    class LoaderConstructor:
        def __call__(self, model_path: str) -> FakeLoader:
            constructed_paths.append(model_path)
            return loader

    class FakeLoaderModule(ModuleType):
        BF16SafeTensorLoader: LoaderConstructor

    loader_module = FakeLoaderModule("kt_kernel.utils.loader")
    loader_module.BF16SafeTensorLoader = LoaderConstructor()

    def fake_import_module(name: str) -> ModuleType:
        imported.append(name)
        if name == "torch":
            return ModuleType("torch")
        if name == "kt_kernel.utils.loader":
            return loader_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(reference.importlib, "import_module", fake_import_module)

    factory = reference.LazyBf16SafeTensorLoaderFactory()
    assert imported == []
    assert factory(MODEL_PATH) is loader
    assert imported == ["torch", "kt_kernel.utils.loader"]
    assert constructed_paths == [str(MODEL_PATH)]


def test_concrete_backend_enforces_exact_bf16_shape_and_dtype() -> None:
    bfloat16 = FakeTorchDtype(reference.GLM47_BF16_DTYPE)
    backend = reference.TorchGlm47ReferenceComputationBackend(
        bfloat16_dtype=bfloat16,
        linear=lambda inputs, weights: inputs * weights,
        silu=lambda tensor: tensor,
    )
    valid = FakeTorchTensor(
        value=1.0,
        shape=reference.GLM47_GATE_UP_WEIGHT_SHAPE,
        dtype=bfloat16,
    )

    backend.validate_bf16_weight(
        valid,
        key="gate",
        expected_shape=reference.GLM47_GATE_UP_WEIGHT_SHAPE,
        expected_dtype=reference.GLM47_BF16_DTYPE,
    )
    with pytest.raises(reference.Glm47ReferenceError, match="shape"):
        backend.validate_bf16_weight(
            FakeTorchTensor(1.0, (1, 2), bfloat16),
            key="gate",
            expected_shape=reference.GLM47_GATE_UP_WEIGHT_SHAPE,
            expected_dtype=reference.GLM47_BF16_DTYPE,
        )
    with pytest.raises(reference.Glm47ReferenceError, match="dtype"):
        backend.validate_bf16_weight(
            FakeTorchTensor(
                1.0,
                reference.GLM47_GATE_UP_WEIGHT_SHAPE,
                FakeTorchDtype("torch.float32"),
            ),
            key="gate",
            expected_shape=reference.GLM47_GATE_UP_WEIGHT_SHAPE,
            expected_dtype=reference.GLM47_BF16_DTYPE,
        )


def test_concrete_backend_computes_and_accumulates_float_reference() -> None:
    bfloat16 = FakeTorchDtype(reference.GLM47_BF16_DTYPE)
    backend = reference.TorchGlm47ReferenceComputationBackend(
        bfloat16_dtype=bfloat16,
        linear=lambda inputs, weights: inputs * weights,
        silu=lambda tensor: tensor,
    )
    hidden = FakeTorchTensor(2.0, (1, 2_048), bfloat16)
    gate = FakeTorchTensor(3.0, reference.GLM47_GATE_UP_WEIGHT_SHAPE, bfloat16)
    up = FakeTorchTensor(4.0, reference.GLM47_GATE_UP_WEIGHT_SHAPE, bfloat16)
    down = FakeTorchTensor(5.0, reference.GLM47_DOWN_WEIGHT_SHAPE, bfloat16)

    expert_output = cast(
        FakeTorchTensor,
        backend.compute_expert_output(hidden, gate, up, down),
    )
    assert math.isclose(expert_output.value, 240.0)
    weighted = cast(
        FakeTorchTensor,
        backend.accumulate_weighted_output(
            None,
            expert_output,
            route_weight=0.4,
        ),
    )
    combined = cast(
        FakeTorchTensor,
        backend.accumulate_weighted_output(
            weighted,
            expert_output,
            route_weight=0.1,
        ),
    )
    assert math.isclose(weighted.value, 96.0)
    assert math.isclose(combined.value, 120.0)


def test_stable_tensor_hash_binds_dtype_shape_and_little_endian_bytes() -> None:
    raw_bytes = bytes.fromhex("803f00c0")

    digest = reference.calculate_glm47_tensor_sha256(
        dtype=reference.GLM47_BF16_DTYPE,
        shape=(1, 2),
        little_endian_contiguous_bytes=raw_bytes,
    )

    assert digest == "a6f412124c432f8c08f870bfed122f6432e35c81b5681dc9a4c985346e0c3d3c"
    assert digest != reference.calculate_glm47_tensor_sha256(
        dtype=reference.GLM47_BF16_DTYPE,
        shape=(2, 1),
        little_endian_contiguous_bytes=raw_bytes,
    )
    assert digest != reference.calculate_glm47_tensor_sha256(
        dtype=reference.GLM47_FLOAT32_DTYPE,
        shape=(1,),
        little_endian_contiguous_bytes=raw_bytes,
    )
    assert digest != reference.calculate_glm47_tensor_sha256(
        dtype=reference.GLM47_BF16_DTYPE,
        shape=(1, 2),
        little_endian_contiguous_bytes=bytes.fromhex("00c0803f"),
    )


@pytest.mark.parametrize("shape", ((), (0,), (-1,), (True,), (1 << 64,)))
def test_stable_tensor_hash_rejects_invalid_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises((TypeError, reference.Glm47ReferenceError)):
        reference.calculate_glm47_tensor_sha256(
            dtype=reference.GLM47_BF16_DTYPE,
            shape=shape,
            little_endian_contiguous_bytes=b"",
        )


def test_stable_tensor_hash_rejects_invalid_dtype_and_byte_count() -> None:
    with pytest.raises(reference.Glm47ReferenceError, match="dtype"):
        reference.calculate_glm47_tensor_sha256(
            dtype=cast(reference.Glm47TensorDtype, cast(object, "torch.float16")),
            shape=(1,),
            little_endian_contiguous_bytes=b"\0\0",
        )
    with pytest.raises(reference.Glm47ReferenceError, match="expected 4"):
        reference.calculate_glm47_tensor_sha256(
            dtype=reference.GLM47_FLOAT32_DTYPE,
            shape=(1,),
            little_endian_contiguous_bytes=b"\0\0",
        )
    with pytest.raises(TypeError, match="immutable"):
        reference.calculate_glm47_tensor_sha256(
            dtype=reference.GLM47_BF16_DTYPE,
            shape=(1,),
            little_endian_contiguous_bytes=cast(bytes, cast(object, bytearray(2))),
        )


def test_tensor_snapshot_decodes_exact_bfloat16_and_float32_values() -> None:
    bfloat16 = reference.Glm47TensorSnapshot(
        dtype=reference.GLM47_BF16_DTYPE,
        shape=(1, 2),
        little_endian_contiguous_bytes=bytes.fromhex("803f00c0"),
    )
    float32 = reference.Glm47TensorSnapshot(
        dtype=reference.GLM47_FLOAT32_DTYPE,
        shape=(1, 2),
        little_endian_contiguous_bytes=struct.pack("<ff", 1.25, -2.5),
    )

    assert bfloat16.finite_values() == (1.0, -2.0)
    assert float32.finite_values() == (1.25, -2.5)
    assert bfloat16.sha256 == reference.calculate_glm47_tensor_sha256(
        dtype=bfloat16.dtype,
        shape=bfloat16.shape,
        little_endian_contiguous_bytes=bfloat16.little_endian_contiguous_bytes,
    )


@pytest.mark.parametrize(
    "snapshot",
    (
        make_bfloat16_snapshot(float("nan"), shape=(1,)),
        make_float32_snapshot(float("inf"), shape=(1,)),
    ),
)
def test_tensor_snapshot_rejects_nonfinite_values_when_consumed(
    snapshot: reference.Glm47TensorSnapshot,
) -> None:
    with pytest.raises(reference.Glm47ReferenceError, match="non-finite"):
        snapshot.finite_values()


def test_calculates_strict_receipt_ready_numerical_evidence() -> None:
    actual = make_bfloat16_snapshot(1.0)
    expected_reference_value = cast(
        float,
        struct.unpack("<f", struct.pack("<f", 1.01))[0],
    )
    oracle = make_float32_snapshot(1.01)

    evidence = reference.calculate_glm47_numerical_evidence(actual, oracle)

    expected_error = abs(1.0 - expected_reference_value)
    assert evidence.shape == reference.GLM47_LAYER_ONE_OUTPUT_SHAPE
    assert math.isclose(evidence.mean_absolute_error, expected_error)
    assert math.isclose(evidence.maximum_absolute_error, expected_error)
    assert math.isclose(evidence.reference_mean_absolute, expected_reference_value)
    assert math.isclose(
        evidence.relative_l1_error,
        expected_error / expected_reference_value,
    )
    assert evidence.relative_l1_tolerance == 0.02
    assert evidence.as_json()["shape"] == [1, 2_048]


@pytest.mark.parametrize(
    ("actual", "oracle", "message"),
    (
        (make_bfloat16_snapshot(1.0), make_float32_snapshot(2.0), "tolerance"),
        (make_bfloat16_snapshot(1.0), make_float32_snapshot(0.0), "L1 norm"),
        (
            make_bfloat16_snapshot(float("nan")),
            make_float32_snapshot(1.0),
            "non-finite",
        ),
        (
            make_bfloat16_snapshot(1.0),
            make_float32_snapshot(float("inf")),
            "non-finite",
        ),
    ),
)
def test_numerical_evidence_rejects_inadmissible_values(
    actual: reference.Glm47TensorSnapshot,
    oracle: reference.Glm47TensorSnapshot,
    message: str,
) -> None:
    with pytest.raises(reference.Glm47ReferenceError, match=message):
        reference.calculate_glm47_numerical_evidence(actual, oracle)


def test_numerical_evidence_rejects_wrong_shape_and_dtype() -> None:
    with pytest.raises(reference.Glm47ReferenceError, match="BF16 and FP32"):
        reference.calculate_glm47_numerical_evidence(
            make_bfloat16_snapshot(1.0, shape=(2_048, 1)),
            make_float32_snapshot(1.0, shape=(2_048, 1)),
        )
    with pytest.raises(reference.Glm47ReferenceError, match="BF16 and FP32"):
        reference.calculate_glm47_numerical_evidence(
            make_float32_snapshot(1.0),
            make_float32_snapshot(1.0),
        )


def test_repeat_hash_helper_requires_exact_tensor_identity() -> None:
    original = make_bfloat16_snapshot(1.0)
    repeat = make_bfloat16_snapshot(1.0)

    assert (
        reference.require_matching_glm47_tensor_sha256(
            original,
            repeat,
            description="backend output",
        )
        == original.sha256
    )
    with pytest.raises(reference.Glm47ReferenceError, match="does not match"):
        reference.require_matching_glm47_tensor_sha256(
            original,
            make_bfloat16_snapshot(2.0),
            description="backend output",
        )
    with pytest.raises(ValueError, match="description"):
        reference.require_matching_glm47_tensor_sha256(
            original,
            repeat,
            description=" untrimmed",
        )


def test_builds_output_evidence_from_exact_repeat_and_fp32_reference() -> None:
    backend = FakeEvidenceBackend()
    actual = FakeEvidenceTensor(make_bfloat16_snapshot(1.0))
    oracle = FakeEvidenceTensor(make_float32_snapshot(1.0))

    evidence = reference.build_glm47_layer_one_output_evidence(
        actual_output=actual,
        repeat_actual_output=actual,
        reference_output=oracle,
        backend=backend,
    )

    assert evidence.actual_output_sha256 == actual.snapshot.sha256
    assert evidence.repeat_actual_output_sha256 == actual.snapshot.sha256
    assert evidence.reference_output_sha256 == oracle.snapshot.sha256
    assert evidence.numerical.relative_l1_error == 0
    assert evidence.as_json()["numerical"] == evidence.numerical.as_json()

    with pytest.raises(reference.Glm47ReferenceError, match="does not match"):
        reference.build_glm47_layer_one_output_evidence(
            actual_output=actual,
            repeat_actual_output=FakeEvidenceTensor(make_bfloat16_snapshot(2.0)),
            reference_output=oracle,
            backend=backend,
        )


def test_builds_exact_hybrid_merge_hash_and_fp32_consistency() -> None:
    backend = FakeEvidenceBackend()
    combined = FakeEvidenceTensor(make_bfloat16_snapshot(3.0))
    cpu = FakeEvidenceTensor(make_bfloat16_snapshot(1.0))
    gpu = FakeEvidenceTensor(make_bfloat16_snapshot(2.0))

    evidence = reference.build_glm47_layer_one_hybrid_merge_evidence(
        combined_output=combined,
        cpu_output=cpu,
        gpu_output=gpu,
        repeat_cpu_output=cpu,
        repeat_gpu_output=gpu,
        backend=backend,
    )

    reference_merge = make_float32_snapshot(3.0)
    assert evidence.combined_output_sha256 == combined.snapshot.sha256
    assert evidence.cpu_output_sha256 == cpu.snapshot.sha256
    assert evidence.gpu_output_sha256 == gpu.snapshot.sha256
    assert evidence.merged_backend_output_sha256 == combined.snapshot.sha256
    assert evidence.repeat_merged_backend_output_sha256 == combined.snapshot.sha256
    assert evidence.reference_merged_backend_output_sha256 == reference_merge.sha256
    assert evidence.numerical.relative_l1_error == 0
    assert evidence.as_json()["reference_merged_backend_output_sha256"] == (
        reference_merge.sha256
    )


def test_hybrid_merge_rejects_changed_combined_or_repeat_output() -> None:
    backend = FakeEvidenceBackend()
    cpu = FakeEvidenceTensor(make_bfloat16_snapshot(1.0))
    gpu = FakeEvidenceTensor(make_bfloat16_snapshot(2.0))

    with pytest.raises(reference.Glm47ReferenceError, match="combined"):
        reference.build_glm47_layer_one_hybrid_merge_evidence(
            combined_output=FakeEvidenceTensor(make_bfloat16_snapshot(4.0)),
            cpu_output=cpu,
            gpu_output=gpu,
            repeat_cpu_output=cpu,
            repeat_gpu_output=gpu,
            backend=backend,
        )
    with pytest.raises(reference.Glm47ReferenceError, match="does not match"):
        reference.build_glm47_layer_one_hybrid_merge_evidence(
            combined_output=FakeEvidenceTensor(make_bfloat16_snapshot(3.0)),
            cpu_output=cpu,
            gpu_output=gpu,
            repeat_cpu_output=cpu,
            repeat_gpu_output=FakeEvidenceTensor(make_bfloat16_snapshot(3.0)),
            backend=backend,
        )


def test_torch_evidence_backend_captures_signed_bits_as_little_endian() -> None:
    bfloat16 = FakeTorchDtype(reference.GLM47_BF16_DTYPE)
    float32 = FakeTorchDtype(reference.GLM47_FLOAT32_DTYPE)
    int16 = FakeTorchDtype("torch.int16")
    int32 = FakeTorchDtype("torch.int32")
    backend = reference.TorchGlm47TensorEvidenceBackend(
        bfloat16_dtype=bfloat16,
        float32_dtype=float32,
        int16_dtype=int16,
        int32_dtype=int32,
    )
    tensor = FakeTorchBitTensor(
        bit_values=(16_256, -16_384),
        shape=(1, 2),
        dtype=bfloat16,
    )

    snapshot = backend.snapshot(
        tensor,
        expected_dtype=reference.GLM47_BF16_DTYPE,
        expected_shape=(1, 2),
    )

    assert snapshot.little_endian_contiguous_bytes == bytes.fromhex("803f00c0")
    assert snapshot.finite_values() == (1.0, -2.0)
    with pytest.raises(reference.Glm47ReferenceError, match="shape"):
        backend.snapshot(
            tensor,
            expected_dtype=reference.GLM47_BF16_DTYPE,
            expected_shape=(2, 1),
        )
    with pytest.raises(reference.Glm47ReferenceError, match="integral"):
        backend.snapshot(
            FakeTorchBitTensor((1.5, 2.5), (1, 2), bfloat16),
            expected_dtype=reference.GLM47_BF16_DTYPE,
            expected_shape=(1, 2),
        )


def test_torch_evidence_backend_factory_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    class FakeTorchModule(ModuleType):
        bfloat16: object
        float32: object
        int16: object
        int32: object

    torch_module = FakeTorchModule("torch")
    torch_module.bfloat16 = FakeTorchDtype(reference.GLM47_BF16_DTYPE)
    torch_module.float32 = FakeTorchDtype(reference.GLM47_FLOAT32_DTYPE)
    torch_module.int16 = FakeTorchDtype("torch.int16")
    torch_module.int32 = FakeTorchDtype("torch.int32")

    def fake_import_module(name: str) -> ModuleType:
        imported.append(name)
        if name == "torch":
            return torch_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(reference.importlib, "import_module", fake_import_module)

    assert imported == []
    backend = reference.create_torch_glm47_tensor_evidence_backend()
    assert isinstance(backend, reference.TorchGlm47TensorEvidenceBackend)
    assert imported == ["torch"]
