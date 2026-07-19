import hashlib
import importlib.machinery
import importlib.util
import os
import re
import socket
import subprocess
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path
from typing import Callable, Literal, Protocol, cast, final

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from exo.download.download_utils import is_model_directory_complete
from exo.shared.types.common import Host, ModelId, NodeId
from exo.shared.types.compute_resources import NvidiaGpuComputeResource
from exo.shared.types.worker.sglang_kt import (
    AbsoluteRuntimePath,
    GitRevision,
    HcaDevice,
    KTransformersMethod,
    NetworkPort,
    ResourceIndex,
)
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_BF16_CONFIG_SHA256,
    GLM_4_7_FLASH_BF16_MODEL_ID,
    GLM_4_7_FLASH_BF16_MODEL_REVISION,
    GLM_5_2_FP8_MODEL_ID,
    SglangKtProcessLaunchSpec,
)
from exo.worker.sglang_kt.preflight import (
    SglangKtHostPreflightObservation,
    SglangKtModelSnapshotReceiptObservation,
    SglangKtRuntimeObservation,
    SglangKtRuntimeValidationReceiptObservation,
    Sha256Digest,
)

_DEFAULT_NUMA_NODES_PATH = Path("/sys/devices/system/node")
_DEFAULT_INFINIBAND_DEVICES_PATH = Path("/sys/class/infiniband")
_HCA_DEVICE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_RUNTIME_PROBE_TIMEOUT_SECONDS = 30.0


def calculate_sglang_kt_artifact_build_id(
    module_name: str,
    required_native_fragment: str,
    additional_native_module_names: tuple[str, ...] = (),
) -> str | None:
    """Hash installed kernel sources and native extensions without absolute paths."""

    try:
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return None
        roots: list[Path] = []
        if module_spec.submodule_search_locations:
            roots.extend(
                Path(location).resolve()
                for location in module_spec.submodule_search_locations
            )
        elif module_spec.origin is not None:
            roots.append(Path(module_spec.origin).resolve())
        if not roots:
            return None

        native_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        source_suffixes = (
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".json",
            ".py",
            ".pyi",
        )
        artifact_files: list[tuple[str, Path]] = []
        native_names: list[str] = []
        for root_index, root in enumerate(sorted(roots, key=str)):
            candidates = (root,) if root.is_file() else root.rglob("*")
            for candidate in candidates:
                if not candidate.is_file() or "__pycache__" in candidate.parts:
                    continue
                candidate_name = candidate.name
                is_native = candidate_name.endswith(native_suffixes)
                if not is_native and not candidate_name.endswith(source_suffixes):
                    continue
                relative_name = (
                    candidate.name
                    if root.is_file()
                    else candidate.relative_to(root).as_posix()
                )
                artifact_files.append(
                    (f"package:{root_index}:{relative_name}", candidate.resolve())
                )
                if is_native:
                    native_names.append(relative_name)

        for additional_module_name in additional_native_module_names:
            try:
                additional_spec = importlib.util.find_spec(additional_module_name)
            except ModuleNotFoundError:
                additional_spec = None
            if additional_spec is None or additional_spec.origin is None:
                continue
            additional_path = Path(additional_spec.origin).resolve()
            if not additional_path.is_file() or not additional_path.name.endswith(
                native_suffixes
            ):
                continue
            if any(
                artifact_path == additional_path
                for _relative_name, artifact_path in artifact_files
            ):
                native_names.append(f"{additional_module_name}:{additional_path.name}")
                continue
            artifact_files.append(
                (
                    f"module:{additional_module_name}:{additional_path.name}",
                    additional_path,
                )
            )
            native_names.append(f"{additional_module_name}:{additional_path.name}")

        if not artifact_files or not any(
            required_native_fragment in name for name in native_names
        ):
            return None

        digest = hashlib.sha256()
        digest.update(b"exo-sglang-kt-artifact-v1\0")
        for relative_name, artifact_path in sorted(artifact_files):
            encoded_name = relative_name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            with artifact_path.open("rb") as artifact_file:
                while chunk := artifact_file.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE = r"""
def calculate_sglang_kt_artifact_build_id(
    module_name,
    required_native_fragment,
    additional_native_module_names=(),
):
    try:
        module_spec = importlib.util.find_spec(module_name)
        if module_spec is None:
            return None
        roots = []
        if module_spec.submodule_search_locations:
            roots.extend(
                pathlib.Path(location).resolve()
                for location in module_spec.submodule_search_locations
            )
        elif module_spec.origin is not None:
            roots.append(pathlib.Path(module_spec.origin).resolve())
        if not roots:
            return None

        native_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
        source_suffixes = (
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".json",
            ".py",
            ".pyi",
        )
        artifact_files = []
        native_names = []
        for root_index, root in enumerate(sorted(roots, key=str)):
            candidates = (root,) if root.is_file() else root.rglob("*")
            for candidate in candidates:
                if not candidate.is_file() or "__pycache__" in candidate.parts:
                    continue
                candidate_name = candidate.name
                is_native = candidate_name.endswith(native_suffixes)
                if not is_native and not candidate_name.endswith(source_suffixes):
                    continue
                relative_name = (
                    candidate.name
                    if root.is_file()
                    else candidate.relative_to(root).as_posix()
                )
                artifact_files.append(
                    (f"package:{root_index}:{relative_name}", candidate.resolve())
                )
                if is_native:
                    native_names.append(relative_name)

        for additional_module_name in additional_native_module_names:
            try:
                additional_spec = importlib.util.find_spec(additional_module_name)
            except ModuleNotFoundError:
                additional_spec = None
            if additional_spec is None or additional_spec.origin is None:
                continue
            additional_path = pathlib.Path(additional_spec.origin).resolve()
            if not additional_path.is_file() or not additional_path.name.endswith(
                native_suffixes
            ):
                continue
            if any(
                artifact_path == additional_path
                for _relative_name, artifact_path in artifact_files
            ):
                native_names.append(
                    f"{additional_module_name}:{additional_path.name}"
                )
                continue
            artifact_files.append(
                (
                    f"module:{additional_module_name}:{additional_path.name}",
                    additional_path,
                )
            )
            native_names.append(f"{additional_module_name}:{additional_path.name}")

        if not artifact_files or not any(
            required_native_fragment in name for name in native_names
        ):
            return None

        digest = hashlib.sha256()
        digest.update(b"exo-sglang-kt-artifact-v1\0")
        for relative_name, artifact_path in sorted(artifact_files):
            encoded_name = relative_name.encode("utf-8")
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            with artifact_path.open("rb") as artifact_file:
                while chunk := artifact_file.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None
""".strip()


# This script runs only under the exact external Python named by the launch spec.
# It deliberately uses only the standard library and reports partial facts when a
# package is absent. The evaluator turns every absent fact into a failed check.
SGLANG_KT_RUNTIME_PROBE_SCRIPT = (
    r"""
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import pathlib
import platform
import sys


"""
    + SGLANG_KT_ARTIFACT_BUILD_ID_FUNCTION_SOURCE
    + r"""
def embedded_source_revisions():
    try:
        observations = []
        for package_name in ("sglang", "kt_kernel"):
            package_spec = importlib.util.find_spec(package_name)
            provenance_name = package_name + "._exo_build_provenance"
            provenance_spec = importlib.util.find_spec(provenance_name)
            if (
                package_spec is None
                or package_spec.origin is None
                or provenance_spec is None
                or provenance_spec.origin is None
            ):
                return None, None
            package_directory = pathlib.Path(package_spec.origin).resolve().parent
            provenance_path = pathlib.Path(provenance_spec.origin).resolve()
            if (
                provenance_path.parent != package_directory
                or provenance_path.name != "_exo_build_provenance.py"
            ):
                return None, None
            provenance = importlib.import_module(provenance_name)
            schema_version = getattr(provenance, "SCHEMA_VERSION", None)
            ktransformers_revision = getattr(
                provenance, "KTRANSFORMERS_REVISION", None
            )
            sglang_revision = getattr(provenance, "SGLANG_REVISION", None)
            revisions = (ktransformers_revision, sglang_revision)
            if schema_version != 1 or any(
                not isinstance(revision, str)
                or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)
                for revision in revisions
            ):
                return None, None
            observations.append(revisions)
        if len(observations) == 2 and observations[0] == observations[1]:
            return observations[0]
    except Exception:
        pass
    return None, None


def distribution_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_version(module_name):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return version if isinstance(version, str) and version else None
    except Exception:
        return None


def torch_runtime_versions():
    try:
        torch = importlib.import_module("torch")
        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if not isinstance(torch_version, str) or not torch_version:
            torch_version = None
        if not isinstance(cuda_version, str) or not cuda_version:
            cuda_version = None
        return torch_version, cuda_version
    except Exception:
        return None, None


torch_version, cuda_version = torch_runtime_versions()
ktransformers_revision, sglang_revision = embedded_source_revisions()
print(json.dumps({
    "executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": {
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "patch": sys.version_info.micro,
    },
    "sglang_revision": sglang_revision,
    "ktransformers_revision": ktransformers_revision,
    "transformers_distribution_version": distribution_version("transformers-kt"),
    "transformers_module_version": module_version("transformers"),
    "torch_version": torch_version,
    "cuda_version": cuda_version,
    "sgl_kernel_build_id": calculate_sglang_kt_artifact_build_id(
        "sgl_kernel", "common_ops"
    ),
    "deep_gemm_build_id": calculate_sglang_kt_artifact_build_id(
        "deep_gemm", "deep_gemm"
    ),
    "kt_kernel_build_id": calculate_sglang_kt_artifact_build_id(
        "kt_kernel", "kt_kernel_ext", ("kt_kernel_ext",)
    ),
}))
"""
).strip()


@final
class SglangKtRuntimeCommandResult(FrozenModel):
    return_code: int
    stdout: str
    stderr: str


class SglangKtRuntimeCommandRunner(Protocol):
    def __call__(
        self,
        command: tuple[str, ...],
        timeout_seconds: float,
    ) -> SglangKtRuntimeCommandResult: ...


class SglangKtRuntimeProbe(Protocol):
    def observe_runtime(
        self, executable: AbsoluteRuntimePath
    ) -> SglangKtRuntimeObservation: ...


class SglangKtFilesystemProbe(Protocol):
    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool: ...

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation | None: ...


@final
class SglangKtLocalHostInventory(FrozenModel):
    gpu_resources: tuple[NvidiaGpuComputeResource, ...] = ()
    cpu_cores: tuple[ResourceIndex, ...] = ()
    memory_nodes: tuple[ResourceIndex, ...] = ()
    hca_devices: tuple[HcaDevice, ...] = ()

    @model_validator(mode="after")
    def validate_unique_resources(self) -> "SglangKtLocalHostInventory":
        resources: tuple[tuple[str, tuple[object, ...]], ...] = (
            (
                "gpu_resources",
                tuple(resource.device_uuid for resource in self.gpu_resources),
            ),
            ("cpu_cores", self.cpu_cores),
            ("memory_nodes", self.memory_nodes),
            ("hca_devices", self.hca_devices),
        )
        for resource_name, values in resources:
            if len(set(values)) != len(values):
                raise ValueError(f"{resource_name} must be unique")
        return self


class SglangKtHostInventoryProbe(Protocol):
    def observe_inventory(
        self,
        gpu_resources: tuple[NvidiaGpuComputeResource, ...],
    ) -> SglangKtLocalHostInventory: ...


class SglangKtPortProbe(Protocol):
    def can_bind_endpoint(self, endpoint: Host) -> bool: ...


def _run_runtime_command(
    command: tuple[str, ...], timeout_seconds: float
) -> SglangKtRuntimeCommandResult:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return SglangKtRuntimeCommandResult(
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


class ExternalPythonSglangKtRuntimeProbe:
    """Observe package facts inside the planned external Python runtime."""

    def __init__(
        self,
        command_runner: SglangKtRuntimeCommandRunner = _run_runtime_command,
    ) -> None:
        self._command_runner = command_runner

    def observe_runtime(
        self, executable: AbsoluteRuntimePath
    ) -> SglangKtRuntimeObservation:
        try:
            result = self._command_runner(
                (executable, "-I", "-c", SGLANG_KT_RUNTIME_PROBE_SCRIPT),
                _RUNTIME_PROBE_TIMEOUT_SECONDS,
            )
            if result.return_code != 0:
                return SglangKtRuntimeObservation()
            return SglangKtRuntimeObservation.model_validate_json(result.stdout)
        except Exception:
            return SglangKtRuntimeObservation()


type ReadableDirectoryChecker = Callable[[Path], bool]
type ModelSnapshotCompletenessChecker = Callable[[Path, ModelId, GitRevision], bool]


@final
class SglangKtModelSnapshotCompatibility(FrozenModel):
    """Artifact facts established independently of the revision receipt."""

    weight_format: Literal["safetensors"]
    ktransformers_method: KTransformersMethod
    config_sha256: Sha256Digest
    full_indexer_layer_starts: tuple[ResourceIndex, ...]


type ModelSnapshotCompatibilityVerifier = Callable[
    [Path, ModelId, GitRevision], SglangKtModelSnapshotCompatibility | None
]


@final
class _Glm52Fp8QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    activation_scheme: Literal["dynamic"]
    fmt: Literal["e4m3"]
    quant_method: Literal["fp8"]
    weight_block_size: tuple[Literal[128], Literal[128]]


@final
class _Glm52Fp8ModelConfig(BaseModel):
    """Raw checkpoint fields required by the pinned GLM-5.2 runtime."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    architectures: tuple[Literal["GlmMoeDsaForCausalLM"]]
    model_type: Literal["glm_moe_dsa"]
    num_hidden_layers: Literal[78]
    n_routed_experts: Literal[256]
    num_experts_per_tok: Literal[8]
    n_shared_experts: Literal[1]
    first_k_dense_replace: Literal[3]
    num_nextn_predict_layers: Literal[1]
    index_topk_freq: Literal[4]
    index_topk_pattern: None
    index_skip_topk_offset: Literal[3]
    index_share_for_mtp_iteration: Literal[True]
    indexer_types: tuple[Literal["full", "shared"], ...]
    quantization_config: _Glm52Fp8QuantizationConfig

    @model_validator(mode="after")
    def validate_indexer_types(self) -> "_Glm52Fp8ModelConfig":
        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError("indexer_types must describe every hidden layer")
        full_starts = tuple(
            index
            for index, indexer_type in enumerate(self.indexer_types)
            if indexer_type == "full"
        )
        expected_full_starts = (0, 1, 2, *range(6, self.num_hidden_layers, 4))
        if full_starts != expected_full_starts:
            raise ValueError("indexer_types does not match audited GLM-5.2 IndexShare")
        return self


@final
class _Glm47FlashBf16ModelConfig(BaseModel):
    """Exact architecture fields for the pinned official Flash BF16 snapshot."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    architectures: tuple[Literal["Glm4MoeLiteForCausalLM"]]
    attention_bias: Literal[False]
    attention_dropout: float
    hidden_act: Literal["silu"]
    hidden_size: Literal[2048]
    intermediate_size: Literal[10240]
    max_position_embeddings: Literal[202752]
    model_type: Literal["glm4_moe_lite"]
    moe_intermediate_size: Literal[1536]
    topk_method: Literal["noaux_tc"]
    norm_topk_prob: Literal[True]
    num_attention_heads: Literal[20]
    n_group: Literal[1]
    topk_group: Literal[1]
    n_routed_experts: Literal[64]
    n_shared_experts: Literal[1]
    routed_scaling_factor: float
    num_experts_per_tok: Literal[4]
    first_k_dense_replace: Literal[1]
    num_hidden_layers: Literal[47]
    num_key_value_heads: Literal[20]
    num_nextn_predict_layers: Literal[1]
    partial_rotary_factor: float
    rms_norm_eps: float
    rope_scaling: None
    rope_theta: Literal[1000000]
    tie_word_embeddings: Literal[False]
    dtype: Literal["bfloat16"]
    q_lora_rank: Literal[768]
    kv_lora_rank: Literal[512]
    qk_nope_head_dim: Literal[192]
    qk_rope_head_dim: Literal[64]
    v_head_dim: Literal[256]
    vocab_size: Literal[154880]

    @model_validator(mode="after")
    def validate_float_constants(self) -> "_Glm47FlashBf16ModelConfig":
        expected_values = (
            ("attention_dropout", self.attention_dropout, 0.0),
            ("routed_scaling_factor", self.routed_scaling_factor, 1.8),
            ("partial_rotary_factor", self.partial_rotary_factor, 1.0),
            ("rms_norm_eps", self.rms_norm_eps, 0.00001),
        )
        for field_name, actual, expected in expected_values:
            if actual != expected:
                raise ValueError(f"{field_name} must equal {expected}")
        return self


def verify_glm_5_2_fp8_model_snapshot_compatibility(
    path: Path,
    model_id: ModelId,
    revision: GitRevision,
) -> SglangKtModelSnapshotCompatibility | None:
    """Verify the GLM-5.2-FP8 artifact contract from raw ``config.json``."""

    del revision
    if model_id != GLM_5_2_FP8_MODEL_ID:
        return None
    try:
        config_bytes = (path / "config.json").read_bytes()
        config = _Glm52Fp8ModelConfig.model_validate_json(config_bytes)
    except (OSError, UnicodeError, ValidationError):
        return None
    return SglangKtModelSnapshotCompatibility(
        weight_format="safetensors",
        ktransformers_method="FP8",
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        full_indexer_layer_starts=tuple(
            index
            for index, indexer_type in enumerate(config.indexer_types)
            if indexer_type == "full"
        ),
    )


def verify_glm_4_7_flash_bf16_model_snapshot_compatibility(
    path: Path,
    model_id: ModelId,
    revision: GitRevision,
) -> SglangKtModelSnapshotCompatibility | None:
    """Verify the exact official GLM-4.7-Flash BF16 smoke artifact."""

    if (
        model_id != GLM_4_7_FLASH_BF16_MODEL_ID
        or revision != GLM_4_7_FLASH_BF16_MODEL_REVISION
    ):
        return None
    try:
        config_bytes = (path / "config.json").read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != GLM_4_7_FLASH_BF16_CONFIG_SHA256:
            return None
        _Glm47FlashBf16ModelConfig.model_validate_json(config_bytes)
    except (OSError, UnicodeError, ValidationError):
        return None
    return SglangKtModelSnapshotCompatibility(
        weight_format="safetensors",
        ktransformers_method="BF16",
        config_sha256=GLM_4_7_FLASH_BF16_CONFIG_SHA256,
        # The first profile is deliberately PP=1. Multi-stage Flash requires a
        # separate runtime validation contract.
        full_indexer_layer_starts=(0,),
    )


def verify_sglang_kt_model_snapshot_compatibility(
    path: Path,
    model_id: ModelId,
    revision: GitRevision,
) -> SglangKtModelSnapshotCompatibility | None:
    """Dispatch only exact model families admitted by the launch profiles."""

    if model_id == GLM_5_2_FP8_MODEL_ID:
        return verify_glm_5_2_fp8_model_snapshot_compatibility(path, model_id, revision)
    if model_id == GLM_4_7_FLASH_BF16_MODEL_ID:
        return verify_glm_4_7_flash_bf16_model_snapshot_compatibility(
            path, model_id, revision
        )
    return None


def _is_readable_directory(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.R_OK | os.X_OK)


def _is_model_snapshot_complete(
    path: Path,
    model_id: ModelId,
    revision: GitRevision,
) -> bool:
    return is_model_directory_complete(
        path,
        model_id=model_id,
        revision=revision,
    )


class LocalSglangKtFilesystemProbe:
    """Check local paths using Exo's exact revision-receipt semantics.

    The compatibility verifier must inspect the snapshot and return only facts
    it established. Returning ``None`` withholds the entire receipt.
    """

    def __init__(
        self,
        *,
        model_snapshot_compatibility_verifier: ModelSnapshotCompatibilityVerifier = (
            verify_sglang_kt_model_snapshot_compatibility
        ),
        readable_directory_checker: ReadableDirectoryChecker = _is_readable_directory,
        model_snapshot_completeness_checker: ModelSnapshotCompletenessChecker = (
            _is_model_snapshot_complete
        ),
    ) -> None:
        self._readable_directory_checker = readable_directory_checker
        self._model_snapshot_completeness_checker = model_snapshot_completeness_checker
        self._model_snapshot_compatibility_verifier = (
            model_snapshot_compatibility_verifier
        )

    def is_readable_directory(self, path: AbsoluteRuntimePath) -> bool:
        try:
            return self._readable_directory_checker(Path(path)) is True
        except Exception:
            return False

    def observe_model_snapshot(
        self,
        path: AbsoluteRuntimePath,
        model_id: ModelId,
        revision: GitRevision,
    ) -> SglangKtModelSnapshotReceiptObservation | None:
        try:
            snapshot_complete = self._model_snapshot_completeness_checker(
                Path(path), model_id, revision
            )
            compatibility = self._model_snapshot_compatibility_verifier(
                Path(path), model_id, revision
            )
        except Exception:
            return None
        if compatibility is None:
            return None
        return SglangKtModelSnapshotReceiptObservation(
            model_path=path,
            model_id=model_id,
            revision=revision,
            weight_format=compatibility.weight_format,
            ktransformers_method=compatibility.ktransformers_method,
            config_sha256=compatibility.config_sha256,
            full_indexer_layer_starts=compatibility.full_indexer_layer_starts,
            receipt_verified=True,
            snapshot_complete=snapshot_complete is True,
        )


type CpuAffinityReader = Callable[[], set[int]]
type DirectoryNamesReader = Callable[[Path], tuple[str, ...]]
type SysfsTextReader = Callable[[Path], str]


def _read_cpu_affinity() -> set[int]:
    affinity_reader_attribute = cast(
        object,
        getattr(os, "sched_getaffinity", None),
    )
    if not callable(affinity_reader_attribute):
        raise OSError("sched_getaffinity is unavailable on this platform")
    affinity_reader = cast(Callable[[int], set[int]], affinity_reader_attribute)
    return affinity_reader(0)


def _read_directory_names(path: Path) -> tuple[str, ...]:
    return tuple(sorted(entry.name for entry in path.iterdir()))


def _read_sysfs_text(path: Path) -> str:
    return path.read_text()


class LinuxSglangKtHostInventoryProbe:
    """Collect process-visible CPUs, online NUMA nodes, and active HCA ports."""

    def __init__(
        self,
        *,
        cpu_affinity_reader: CpuAffinityReader = _read_cpu_affinity,
        directory_names_reader: DirectoryNamesReader = _read_directory_names,
        text_reader: SysfsTextReader = _read_sysfs_text,
        numa_nodes_path: Path = _DEFAULT_NUMA_NODES_PATH,
        infiniband_devices_path: Path = _DEFAULT_INFINIBAND_DEVICES_PATH,
    ) -> None:
        self._cpu_affinity_reader = cpu_affinity_reader
        self._directory_names_reader = directory_names_reader
        self._text_reader = text_reader
        self._numa_nodes_path = numa_nodes_path
        self._infiniband_devices_path = infiniband_devices_path

    def observe_inventory(
        self,
        gpu_resources: tuple[NvidiaGpuComputeResource, ...],
    ) -> SglangKtLocalHostInventory:
        return SglangKtLocalHostInventory(
            gpu_resources=gpu_resources,
            cpu_cores=self._observe_cpu_cores(),
            memory_nodes=self._observe_memory_nodes(),
            hca_devices=self._observe_active_hca_ports(),
        )

    def _observe_cpu_cores(self) -> tuple[int, ...]:
        try:
            cpu_cores = self._cpu_affinity_reader()
        except Exception:
            return ()
        if any(core < 0 for core in cpu_cores):
            return ()
        return tuple(sorted(cpu_cores))

    def _observe_memory_nodes(self) -> tuple[int, ...]:
        try:
            online_nodes = self._text_reader(self._numa_nodes_path / "online")
        except Exception:
            return ()
        return _parse_linux_index_list(online_nodes)

    def _observe_active_hca_ports(self) -> tuple[str, ...]:
        try:
            device_names = self._directory_names_reader(self._infiniband_devices_path)
        except Exception:
            return ()

        active_ports: list[str] = []
        for device_name in sorted(set(device_names)):
            if _HCA_DEVICE_NAME.fullmatch(device_name) is None:
                continue
            ports_path = self._infiniband_devices_path / device_name / "ports"
            try:
                port_names = self._directory_names_reader(ports_path)
            except Exception:
                continue
            for port_name in sorted(set(port_names), key=_numeric_text_sort_key):
                if not port_name.isdecimal() or int(port_name) <= 0:
                    continue
                try:
                    state = self._text_reader(ports_path / port_name / "state")
                except Exception:
                    continue
                if _is_active_infiniband_port_state(state):
                    active_ports.append(f"{device_name}:{int(port_name)}")
        return tuple(active_ports)


type TcpBindProbe = Callable[[str, NetworkPort], bool]


def _can_bind_tcp_endpoint(ip: str, port: NetworkPort) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
            probe_socket.bind((ip, port))
    except OSError:
        return False
    return True


class SocketSglangKtPortProbe:
    """Check planned TCP bindings without retaining any socket reservation.

    These are momentary availability facts. A future executor must still bind
    and start the process group transactionally under the coordination lease.
    """

    def __init__(self, bind_probe: TcpBindProbe = _can_bind_tcp_endpoint) -> None:
        self._bind_probe = bind_probe

    def can_bind_endpoint(self, endpoint: Host) -> bool:
        try:
            return self._bind_probe(endpoint.ip, endpoint.port) is True
        except Exception:
            return False


def collect_sglang_kt_local_host_preflight_observation(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
    *,
    gpu_resources: Sequence[NvidiaGpuComputeResource],
    runtime_probe: SglangKtRuntimeProbe,
    filesystem_probe: SglangKtFilesystemProbe,
    inventory_probe: SglangKtHostInventoryProbe,
    port_probe: SglangKtPortProbe,
    runtime_validation_receipts: Sequence[
        SglangKtRuntimeValidationReceiptObservation
    ] = (),
) -> SglangKtHostPreflightObservation:
    """Collect one host observation from explicitly supplied local effects."""

    node_id, executable = _validate_local_process_specs(process_specs)
    try:
        runtime = runtime_probe.observe_runtime(executable)
    except Exception:
        runtime = SglangKtRuntimeObservation()

    try:
        supplied_gpu_resources = tuple(gpu_resources)
        inventory = inventory_probe.observe_inventory(supplied_gpu_resources)
        if inventory.gpu_resources != supplied_gpu_resources:
            inventory = SglangKtLocalHostInventory()
    except Exception:
        inventory = SglangKtLocalHostInventory()

    readable_directories = tuple(
        path
        for path in _planned_directories(process_specs)
        if _is_observed_readable_directory(filesystem_probe, path)
    )
    model_snapshot_receipts = _observe_model_snapshots(
        filesystem_probe,
        _planned_model_snapshots(process_specs),
    )
    available_bind_endpoints = tuple(
        endpoint
        for endpoint in _planned_bind_endpoints(process_specs)
        if _is_observed_bind_endpoint_available(port_probe, endpoint)
    )
    supplied_validation_receipts = tuple(runtime_validation_receipts)
    planned_gpu_uuids = {spec.gpu_uuid for spec in process_specs}
    observed_gpu_uuids = {resource.device_uuid for resource in inventory.gpu_resources}
    receipt_gpu_uuids = tuple(
        receipt.gpu_uuid for receipt in supplied_validation_receipts
    )
    if len(set(receipt_gpu_uuids)) != len(receipt_gpu_uuids) or any(
        gpu_uuid not in planned_gpu_uuids or gpu_uuid not in observed_gpu_uuids
        for gpu_uuid in receipt_gpu_uuids
    ):
        supplied_validation_receipts = ()
    return SglangKtHostPreflightObservation(
        node_id=node_id,
        runtime=runtime,
        runtime_validation_receipts=supplied_validation_receipts,
        readable_directories=readable_directories,
        model_snapshot_receipts=model_snapshot_receipts,
        gpu_uuids=tuple(resource.device_uuid for resource in inventory.gpu_resources),
        cpu_cores=inventory.cpu_cores,
        memory_nodes=inventory.memory_nodes,
        hca_devices=inventory.hca_devices,
        available_bind_endpoints=available_bind_endpoints,
    )


def _validate_local_process_specs(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[NodeId, AbsoluteRuntimePath]:
    if not process_specs:
        raise ValueError("local SGLang-KT collection requires process specs")
    node_ids = {spec.node_id for spec in process_specs}
    if len(node_ids) != 1:
        raise ValueError("local SGLang-KT process specs must target one node")
    launch_plan = process_specs[0].plan
    if any(spec.plan != launch_plan for spec in process_specs):
        raise ValueError("local SGLang-KT process specs must share one launch plan")
    pipeline_ranks = tuple(spec.pipeline_rank for spec in process_specs)
    if len(set(pipeline_ranks)) != len(pipeline_ranks):
        raise ValueError("local SGLang-KT process ranks must be unique")
    executables = {spec.executable for spec in process_specs}
    if len(executables) != 1:
        raise ValueError("local SGLang-KT process specs must share one executable")
    return next(iter(node_ids)), next(iter(executables))


def _planned_directories(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[AbsoluteRuntimePath, ...]:
    return _unique(
        path
        for spec in process_specs
        for path in (spec.model_path, spec.ktransformers_weight_path)
    )


def _planned_model_snapshots(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[tuple[AbsoluteRuntimePath, ModelId, GitRevision, KTransformersMethod], ...]:
    snapshots_by_path: dict[
        AbsoluteRuntimePath, tuple[ModelId, GitRevision, KTransformersMethod]
    ] = {}
    for spec in process_specs:
        identity = (
            spec.model_id,
            spec.expected_model_revision,
            spec.ktransformers_method,
        )
        for snapshot_path in (spec.model_path, spec.ktransformers_weight_path):
            previous_identity = snapshots_by_path.setdefault(snapshot_path, identity)
            if previous_identity != identity:
                raise ValueError(
                    "one model path cannot represent multiple model snapshot identities"
                )
    return tuple(
        (path, model_id, revision, ktransformers_method)
        for path, (
            model_id,
            revision,
            ktransformers_method,
        ) in snapshots_by_path.items()
    )


def _planned_bind_endpoints(
    process_specs: tuple[SglangKtProcessLaunchSpec, ...],
) -> tuple[Host, ...]:
    endpoints = [spec.service_endpoint for spec in process_specs]
    rank_zero_spec = next(
        (spec for spec in process_specs if spec.pipeline_rank == 0),
        None,
    )
    if rank_zero_spec is not None:
        endpoints.append(rank_zero_spec.distributed_coordinator)
    endpoints_by_key: dict[tuple[str, int], Host] = {}
    for endpoint in endpoints:
        endpoints_by_key.setdefault((endpoint.ip, endpoint.port), endpoint)
    return tuple(endpoints_by_key.values())


def _is_observed_readable_directory(
    filesystem_probe: SglangKtFilesystemProbe,
    path: AbsoluteRuntimePath,
) -> bool:
    try:
        return filesystem_probe.is_readable_directory(path) is True
    except Exception:
        return False


def _observe_model_snapshots(
    filesystem_probe: SglangKtFilesystemProbe,
    planned_snapshots: tuple[
        tuple[AbsoluteRuntimePath, ModelId, GitRevision, KTransformersMethod], ...
    ],
) -> tuple[SglangKtModelSnapshotReceiptObservation, ...]:
    receipts: list[SglangKtModelSnapshotReceiptObservation] = []
    for path, model_id, revision, ktransformers_method in planned_snapshots:
        receipt = _observe_model_snapshot(
            filesystem_probe,
            path,
            model_id,
            revision,
            ktransformers_method,
        )
        if receipt is not None:
            receipts.append(receipt)
    return tuple(receipts)


def _observe_model_snapshot(
    filesystem_probe: SglangKtFilesystemProbe,
    path: AbsoluteRuntimePath,
    model_id: ModelId,
    revision: GitRevision,
    ktransformers_method: KTransformersMethod,
) -> SglangKtModelSnapshotReceiptObservation | None:
    try:
        receipt = filesystem_probe.observe_model_snapshot(path, model_id, revision)
    except Exception:
        return None
    if receipt is None:
        return None
    if (
        receipt.model_path != path
        or receipt.model_id != model_id
        or receipt.revision != revision
        or receipt.weight_format != "safetensors"
        or receipt.ktransformers_method != ktransformers_method
    ):
        return None
    return receipt


def _is_observed_bind_endpoint_available(
    port_probe: SglangKtPortProbe,
    endpoint: Host,
) -> bool:
    try:
        return port_probe.can_bind_endpoint(endpoint) is True
    except Exception:
        return False


def _parse_linux_index_list(value: str) -> tuple[int, ...]:
    stripped_value = value.strip()
    if not stripped_value:
        return ()

    indices: set[int] = set()
    for item in stripped_value.split(","):
        bounds = item.strip().split("-")
        if not 1 <= len(bounds) <= 2 or not all(bound.isdecimal() for bound in bounds):
            return ()
        first_index = int(bounds[0])
        last_index = int(bounds[-1])
        if first_index > last_index:
            return ()
        indices.update(range(first_index, last_index + 1))
    return tuple(sorted(indices))


def _is_active_infiniband_port_state(value: str) -> bool:
    state_code, separator, state_name = value.strip().partition(":")
    return (
        separator == ":"
        and state_code.strip() == "4"
        and state_name.strip().upper() == "ACTIVE"
    )


def _numeric_text_sort_key(value: str) -> tuple[bool, int | str]:
    return (not value.isdecimal(), int(value) if value.isdecimal() else value)


def _unique[T: Hashable](values: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))
