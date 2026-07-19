"""Fail-closed orchestration primitives for the live GLM-4.7 model probe.

The heavyweight SGLang/Torch execution is intentionally loaded lazily in a
disposable child process.  This module contains the parent/child invariants
that can be tested without importing either runtime.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from exo.worker.sglang_kt.launch_spec import SglangKtProcessLaunchSpec
from exo.worker.sglang_kt.model_runtime_validation_receipt import (
    MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION,
    MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS,
    calculate_sglang_kt_model_runtime_validator_bundle_sha256,
)
from exo.worker.sglang_kt.receipt_io import (
    SglangKtReceiptFileError,
    canonical_sglang_kt_json,
    hash_sglang_kt_bound_file,
)
from exo.worker.sglang_kt.runtime_validation_receipt import (
    SglangKtKernelRuntimeValidationReceiptError,
    SglangKtKernelRuntimeValidationReceiptObservation,
    load_sglang_kt_kernel_runtime_validation_receipt,
)

VALIDATOR_BUNDLE_CANONICALIZATION: Final = (
    MODEL_RUNTIME_VALIDATOR_SOURCE_BUNDLE_CANONICALIZATION
)
MINIMUM_HOST_MEMORY_HEADROOM_BYTES: Final = 64 * 1024**3
MINIMUM_GPU_MEMORY_HEADROOM_BYTES: Final = 2 * 1024**3
MAXIMUM_CHILD_EVIDENCE_BYTES: Final = 4 * 1024 * 1024
_NUMACTL_EXECUTABLE: Final = Path("/usr/bin/numactl")
_UNINHERITED_LIVE_ENVIRONMENT_NAMES: Final = frozenset(
    {
        "LD_AUDIT",
        "LD_PRELOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
    }
)
_PROFILER_ENVIRONMENT_PREFIXES: Final = ("AMPLXE_", "VTUNE_")
_ALLOWED_REINTRODUCED_LIVE_ENVIRONMENT_NAMES: Final = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_ALLOC_CONF",
        "SGLANG_KT_HYBRID_TIMING",
    }
)
# Stable Linux UAPI values remain valid when a portable CPython omits wrappers.
_LINUX_MEMFD_CONSTANTS: Final = (
    ("MFD_CLOEXEC", 0x0001),
    ("MFD_ALLOW_SEALING", 0x0002),
)
_LINUX_FCNTL_CONSTANTS: Final = (
    ("F_ADD_SEALS", 1024 + 9),
    ("F_GET_SEALS", 1024 + 10),
    ("F_SEAL_SEAL", 0x0001),
    ("F_SEAL_SHRINK", 0x0002),
    ("F_SEAL_GROW", 0x0004),
    ("F_SEAL_WRITE", 0x0008),
)
_MISSING_UAPI_CONSTANT: Final = object()
_PROCESS_GROUP_TERMINATION_GRACE_SECONDS: Final = 0.1


class Glm47LiveValidationError(RuntimeError):
    """Raised before live evidence can be admitted or published."""


class _WaitIdResult(Protocol):
    si_pid: int
    si_status: int
    si_code: int


class _CtypesMemfdCreate(Protocol):
    argtypes: object
    restype: object

    def __call__(self, name: bytes, flags: int, /) -> int: ...


@dataclass(frozen=True, slots=True)
class ValidatorSourceIdentity:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatorBundleIdentity:
    canonicalization: str
    sources: tuple[ValidatorSourceIdentity, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class MemoryHeadroomEvidence:
    available_bytes: int
    required_bytes: int
    minimum_headroom_bytes: int
    remaining_bytes: int


@dataclass(frozen=True, slots=True)
class DisposableChildEvidence:
    contents: bytes
    sha256: str


def _parse_linux_index_list(value: str) -> tuple[int, ...]:
    indices: list[int] = []
    for part in value.strip().split(","):
        if not part:
            raise Glm47LiveValidationError("Linux resource list contains an empty item")
        bounds = part.split("-", maxsplit=1)
        try:
            start = int(bounds[0])
            end = start if len(bounds) == 1 else int(bounds[1])
        except ValueError as error:
            raise Glm47LiveValidationError(
                "Linux resource list contains a non-integer item"
            ) from error
        if start < 0 or end < start:
            raise Glm47LiveValidationError("Linux resource list has invalid bounds")
        indices.extend(range(start, end + 1))
    result = tuple(indices)
    if not result or result != tuple(sorted(set(result))):
        raise Glm47LiveValidationError(
            "Linux resource list must be nonempty, sorted, and unique"
        )
    return result


def parse_bound_memory_policy(output: str) -> tuple[int, ...]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        name, separator, value = line.partition(":")
        normalized_name = name.strip()
        if separator and normalized_name in {"policy", "membind"}:
            if normalized_name in values:
                raise Glm47LiveValidationError(
                    f"numactl reported duplicate {normalized_name} evidence"
                )
            values[normalized_name] = value.strip()
    if values.get("policy") != "bind" or not values.get("membind"):
        raise Glm47LiveValidationError(
            "process does not have an explicit bound NUMA memory policy"
        )
    return _parse_linux_index_list(",".join(values["membind"].split()))


def current_bound_memory_nodes(
    numactl_executable: Path = _NUMACTL_EXECUTABLE,
) -> tuple[int, ...]:
    """Read the effective MPOL_BIND nodes rather than the cpuset allowance."""

    if not numactl_executable.is_absolute():
        raise ValueError("numactl executable must be absolute")
    try:
        result = subprocess.run(
            (str(numactl_executable), "--show"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Glm47LiveValidationError("cannot inspect NUMA memory policy") from error
    if result.returncode != 0 or result.stderr:
        raise Glm47LiveValidationError("cannot inspect NUMA memory policy")
    return parse_bound_memory_policy(result.stdout)


def calculate_memory_headroom(
    *,
    available_bytes: int,
    required_bytes: int,
    minimum_headroom_bytes: int,
) -> MemoryHeadroomEvidence:
    values = (available_bytes, required_bytes, minimum_headroom_bytes)
    if any(type(value) is not int for value in values):
        raise TypeError("memory byte counts must be integers")
    if available_bytes < 0 or required_bytes < 0 or minimum_headroom_bytes <= 0:
        raise ValueError(
            "available and required bytes must be nonnegative and headroom positive"
        )
    remaining_bytes = available_bytes - required_bytes
    if remaining_bytes < minimum_headroom_bytes:
        raise Glm47LiveValidationError(
            f"allocation would retain {remaining_bytes} bytes, below the required "
            f"{minimum_headroom_bytes}-byte headroom"
        )
    return MemoryHeadroomEvidence(
        available_bytes=available_bytes,
        required_bytes=required_bytes,
        minimum_headroom_bytes=minimum_headroom_bytes,
        remaining_bytes=remaining_bytes,
    )


def available_host_memory_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    """Return Linux MemAvailable in bytes without estimating missing fields."""

    try:
        lines = meminfo_path.read_text().splitlines()
    except OSError as error:
        raise Glm47LiveValidationError(
            f"cannot read memory information: {meminfo_path}"
        ) from error
    row = next((line for line in lines if line.startswith("MemAvailable:")), None)
    if row is None:
        raise Glm47LiveValidationError("MemAvailable is absent from memory information")
    fields = row.split()
    if len(fields) != 3 or fields[2] != "kB":
        raise Glm47LiveValidationError("MemAvailable has an unexpected representation")
    try:
        kibibytes = int(fields[1])
    except ValueError as error:
        raise Glm47LiveValidationError("MemAvailable is not an integer") from error
    if kibibytes <= 0:
        raise Glm47LiveValidationError("MemAvailable must be positive")
    return kibibytes * 1024


def calculate_validator_bundle(
    paths: Iterable[Path],
) -> ValidatorBundleIdentity:
    """Hash every executable validator source into one auditable identity."""

    normalized_paths = tuple(sorted({str(path) for path in paths}))
    if not normalized_paths:
        raise ValueError("validator bundle requires at least one source")
    if any(not Path(path).is_absolute() for path in normalized_paths):
        raise ValueError("validator bundle source paths must be absolute")
    identities: list[ValidatorSourceIdentity] = []
    try:
        for source_path in normalized_paths:
            identity = hash_sglang_kt_bound_file(Path(source_path))
            identities.append(
                ValidatorSourceIdentity(
                    path=str(identity.path),
                    size_bytes=identity.size_bytes,
                    sha256=identity.sha256,
                )
            )
    except SglangKtReceiptFileError as error:
        raise Glm47LiveValidationError("cannot bind validator source bundle") from error
    sources = tuple(identities)
    source_pairs = tuple((source.path, source.sha256) for source in sources)
    return ValidatorBundleIdentity(
        canonicalization=VALIDATOR_BUNDLE_CANONICALIZATION,
        sources=sources,
        sha256=calculate_sglang_kt_model_runtime_validator_bundle_sha256(source_pairs),
    )


def validator_bundle_paths(repository_root: Path) -> tuple[Path, ...]:
    """Return the exact source paths admitted by the model receipt schema."""

    if not repository_root.is_absolute():
        raise ValueError("validator repository root must be absolute")
    return tuple(
        repository_root / relative_path
        for relative_path in MODEL_RUNTIME_VALIDATOR_SOURCE_RELATIVE_PATHS
    )


def write_disposable_child_evidence(file_descriptor: int, payload: object) -> None:
    """Write one bounded canonical payload to a parent-owned anonymous file."""

    if type(file_descriptor) is not int or file_descriptor < 3:
        raise ValueError("child evidence descriptor must be an inherited descriptor")
    contents = canonical_sglang_kt_json(payload)
    if not contents or len(contents) > MAXIMUM_CHILD_EVIDENCE_BYTES:
        raise Glm47LiveValidationError("child evidence has an invalid size")
    view = memoryview(contents)
    while view:
        try:
            written = os.write(file_descriptor, view)
        except OSError as error:
            raise Glm47LiveValidationError("cannot write child evidence") from error
        if written <= 0:
            raise Glm47LiveValidationError("child evidence write made no progress")
        view = view[written:]
    try:
        os.fsync(file_descriptor)
    except OSError as error:
        raise Glm47LiveValidationError("cannot synchronize child evidence") from error


def _wait_for_direct_child_exit_without_reaping(
    process: subprocess.Popen[bytes],
) -> int:
    """Observe direct-child exit while retaining its PID as the owned PGID."""

    waitid_value = getattr(os, "waitid", None)
    required_constants = tuple(
        getattr(os, name, None)
        for name in ("P_PID", "WEXITED", "WNOWAIT", "CLD_EXITED")
    )
    if not callable(waitid_value) or any(
        type(value) is not int for value in required_constants
    ):
        raise Glm47LiveValidationError(
            "non-reaping child wait is unavailable on this platform"
        )
    waitid = cast(
        Callable[[int, int, int], _WaitIdResult | None],
        waitid_value,
    )
    process_id_type, exited_option, nowait_option, child_exited_code = cast(
        tuple[int, int, int, int], required_constants
    )
    try:
        result = waitid(
            process_id_type,
            process.pid,
            exited_option | nowait_option,
        )
    except ChildProcessError as error:
        raise Glm47LiveValidationError(
            "cannot retain disposable child ownership through exit"
        ) from error
    if result is None or result.si_pid != process.pid:
        raise Glm47LiveValidationError(
            "non-reaping child wait returned an unexpected process"
        )
    if result.si_code == child_exited_code:
        return result.si_status
    return -result.si_status


def _require_owned_process_group(process: subprocess.Popen[bytes]) -> int:
    process_group_id = process.pid
    if process_group_id <= 1:
        raise Glm47LiveValidationError("disposable child has an unsafe process ID")
    try:
        observed_process_group_id = os.getpgid(process.pid)
    except ProcessLookupError as error:
        raise Glm47LiveValidationError(
            "disposable child process-group ownership was lost"
        ) from error
    if observed_process_group_id != process_group_id:
        raise Glm47LiveValidationError(
            "disposable child does not lead its owned process group"
        )
    return process_group_id


def _terminate_and_reap_owned_process_group(
    process: subprocess.Popen[bytes],
) -> int:
    """Signal the verified child-owned group before releasing its leader PID."""

    process_group_id = _require_owned_process_group(process)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    time.sleep(_PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired as error:
        raise Glm47LiveValidationError(
            "disposable child process group could not be reaped"
        ) from error


def _require_linux_uapi_constants(
    module: object,
    expected_constants: tuple[tuple[str, int], ...],
) -> tuple[int, ...]:
    if sys.platform != "linux":
        raise Glm47LiveValidationError("sealed memfd support requires Linux")
    values: list[int] = []
    for name, expected in expected_constants:
        observed = cast(object, getattr(module, name, _MISSING_UAPI_CONSTANT))
        if observed is not _MISSING_UAPI_CONSTANT and (
            type(observed) is not int or observed != expected
        ):
            raise Glm47LiveValidationError(
                f"Python exposes an unexpected Linux UAPI value for {name}"
            )
        values.append(expected)
    return tuple(values)


def _memfd_creation_flags() -> int:
    cloexec, allow_sealing = _require_linux_uapi_constants(
        os,
        _LINUX_MEMFD_CONSTANTS,
    )
    return cloexec | allow_sealing


def _create_sealable_memfd(name: str) -> int:
    if not name or not name.isascii() or "\0" in name or len(name) > 249:
        raise ValueError("memfd name must be a bounded nonempty ASCII string")
    flags = _memfd_creation_flags()
    memfd_create_value = getattr(os, "memfd_create", None)
    try:
        if callable(memfd_create_value):
            memfd_create = cast(Callable[[str, int], int], memfd_create_value)
            descriptor = memfd_create(name, flags)
        else:
            try:
                libc = ctypes.CDLL(None, use_errno=True)
            except OSError as error:
                raise Glm47LiveValidationError(
                    "cannot load libc for sealed memfd creation"
                ) from error
            libc_memfd_create_value = getattr(libc, "memfd_create", None)
            if not callable(libc_memfd_create_value):
                raise Glm47LiveValidationError("libc memfd creation is unavailable")
            libc_memfd_create = cast(
                _CtypesMemfdCreate,
                libc_memfd_create_value,
            )
            libc_memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
            libc_memfd_create.restype = ctypes.c_int
            ctypes.set_errno(0)
            descriptor = libc_memfd_create(name.encode("ascii"), flags)
            if descriptor < 0:
                error_number = ctypes.get_errno() or errno.EIO
                raise OSError(error_number, os.strerror(error_number))
    except OSError as error:
        raise Glm47LiveValidationError("cannot create sealed memfd") from error
    if type(descriptor) is not int or descriptor < 3:
        if type(descriptor) is int and descriptor >= 0:
            os.close(descriptor)
        raise Glm47LiveValidationError(
            "sealed memfd creation returned an unsafe descriptor"
        )
    return descriptor


def _seal_child_evidence(file_descriptor: int) -> None:
    (
        add_seals,
        get_seals,
        seal_seal,
        seal_shrink,
        seal_grow,
        seal_write,
    ) = _require_linux_uapi_constants(
        fcntl,
        _LINUX_FCNTL_CONSTANTS,
    )
    seals = seal_seal | seal_shrink | seal_grow | seal_write
    try:
        fcntl.fcntl(file_descriptor, add_seals, seals)
        observed_seals = fcntl.fcntl(file_descriptor, get_seals)
    except OSError as error:
        raise Glm47LiveValidationError("cannot seal child evidence") from error
    if observed_seals & seals != seals:
        raise Glm47LiveValidationError("child evidence seals are incomplete")


def require_disposable_child_evidence_transport() -> None:
    """Fail before model inspection unless anonymous evidence can be sealed."""

    descriptor = _create_sealable_memfd("exo-glm47-evidence-probe")
    try:
        _seal_child_evidence(descriptor)
    finally:
        os.close(descriptor)


def run_disposable_live_child(
    command: tuple[str, ...],
    *,
    evidence_descriptor_argument: str,
    environment: dict[str, str] | None = None,
) -> DisposableChildEvidence:
    """Run one child, then expose its evidence only after successful exit."""

    if (
        not command
        or not Path(command[0]).is_absolute()
        or any(not argument for argument in command)
    ):
        raise ValueError("disposable child command must use an absolute executable")
    if not evidence_descriptor_argument.startswith("--"):
        raise ValueError("child evidence descriptor argument must be a long option")
    descriptor = _create_sealable_memfd("exo-glm47-model-evidence")
    process: subprocess.Popen[bytes] | None = None
    try:
        child_command = (
            *command,
            evidence_descriptor_argument,
            str(descriptor),
        )
        # The memfd carries evidence; reserve parent stdout for its JSON response.
        process = subprocess.Popen(
            child_command,
            env=environment,
            pass_fds=(descriptor,),
            start_new_session=True,
            stdout=sys.stderr,
        )
        try:
            returncode = _wait_for_direct_child_exit_without_reaping(process)
        except BaseException:
            _terminate_and_reap_owned_process_group(process)
            raise
        if returncode != 0:
            reaped_returncode = _terminate_and_reap_owned_process_group(process)
            if reaped_returncode != returncode:
                raise Glm47LiveValidationError(
                    "disposable child exit status changed while being reaped"
                )
            raise Glm47LiveValidationError(
                f"disposable live child exited with status {returncode}"
            )
        try:
            _seal_child_evidence(descriptor)
            reaped_returncode = _terminate_and_reap_owned_process_group(process)
            if reaped_returncode != returncode:
                raise Glm47LiveValidationError(
                    "disposable child exit status changed while being reaped"
                )
            size = os.fstat(descriptor).st_size
            if not 0 < size <= MAXIMUM_CHILD_EVIDENCE_BYTES:
                raise Glm47LiveValidationError(
                    f"disposable child returned {size} evidence bytes"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            contents = bytearray()
            while len(contents) < size:
                chunk = os.read(descriptor, size - len(contents))
                if not chunk:
                    raise Glm47LiveValidationError(
                        "disposable child evidence ended before its recorded size"
                    )
                contents.extend(chunk)
        except OSError as error:
            raise Glm47LiveValidationError(
                "cannot read disposable child evidence"
            ) from error
        immutable_contents = bytes(contents)
        return DisposableChildEvidence(
            contents=immutable_contents,
            sha256=hashlib.sha256(immutable_contents).hexdigest(),
        )
    finally:
        if process is not None and process.returncode is None:
            _terminate_and_reap_owned_process_group(process)
        os.close(descriptor)


def sglang_server_argument_vector(
    process_spec: SglangKtProcessLaunchSpec,
) -> tuple[str, ...]:
    """Return the exact launch-server arguments consumed by ServerArgs."""

    arguments = process_spec.arguments
    if arguments[:2] != ("-m", "sglang.launch_server"):
        raise Glm47LiveValidationError(
            "process spec does not invoke the pinned SGLang launch module"
        )
    server_arguments = arguments[2:]
    if not server_arguments:
        raise Glm47LiveValidationError("process spec has no SGLang server arguments")
    return server_arguments


def build_live_child_environment(
    process_spec: SglangKtProcessLaunchSpec,
    parent_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the launch-spec environment without inherited runtime controls."""

    process_environment = process_spec.environment
    process_environment_names: set[str] = set()
    for name, _value in process_environment:
        if name in process_environment_names:
            raise Glm47LiveValidationError(
                f"process spec repeats environment variable {name}"
            )
        process_environment_names.add(name)
        matches_unset_control = name in process_spec.unset_environment_variables or any(
            name.startswith(prefix)
            for prefix in process_spec.unset_environment_variable_prefixes
        )
        if (
            name in _UNINHERITED_LIVE_ENVIRONMENT_NAMES
            or any(name.startswith(prefix) for prefix in _PROFILER_ENVIRONMENT_PREFIXES)
            or (
                matches_unset_control
                and name not in _ALLOWED_REINTRODUCED_LIVE_ENVIRONMENT_NAMES
            )
        ):
            raise Glm47LiveValidationError(
                f"process spec contains forbidden environment variable {name}"
            )

    source = os.environ if parent_environment is None else parent_environment
    environment = {
        name: value
        for name, value in source.items()
        if name not in _UNINHERITED_LIVE_ENVIRONMENT_NAMES
        and name not in process_spec.unset_environment_variables
        and not any(
            name.startswith(prefix)
            for prefix in (
                *process_spec.unset_environment_variable_prefixes,
                *_PROFILER_ENVIRONMENT_PREFIXES,
            )
        )
    }
    for name, value in process_environment:
        environment[name] = value
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def load_bound_kernel_runtime_receipt(
    process_spec: SglangKtProcessLaunchSpec,
    path: Path,
    *,
    expected_receipt_sha256: str,
) -> SglangKtKernelRuntimeValidationReceiptObservation:
    """Load strict kernel evidence and require exact launch identity parity."""

    try:
        receipt = load_sglang_kt_kernel_runtime_validation_receipt(
            path,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    except SglangKtKernelRuntimeValidationReceiptError as error:
        raise Glm47LiveValidationError(
            "kernel runtime receipt is not admissible"
        ) from error
    require_kernel_runtime_binding(process_spec, receipt)
    return receipt


def require_kernel_runtime_binding(
    process_spec: SglangKtProcessLaunchSpec,
    receipt: SglangKtKernelRuntimeValidationReceiptObservation,
) -> None:
    """Cross-check every kernel fact consumed by the model receipt."""

    stage = process_spec.stage
    expected_hostname = str(process_spec.node_id)
    threads_per_subpool = receipt.threads_per_subpool
    threadpool_count = stage.threadpool_count
    cpu_infer_threads = stage.cpu_infer_threads
    if (
        receipt.gpu_uuid != process_spec.gpu_uuid
        or receipt.gpu_compute_capability != (8, 6)
        or receipt.hostname != expected_hostname
        or receipt.executable != process_spec.executable
        or receipt.cpu_cores != process_spec.cpu_cores
        or receipt.memory_nodes != process_spec.memory_nodes
        or len(threads_per_subpool) != threadpool_count
        or sum(threads_per_subpool) != cpu_infer_threads
        or receipt.sglang_revision != process_spec.expected_sglang_revision
        or receipt.ktransformers_revision
        != process_spec.expected_ktransformers_revision
        or receipt.transformers_distribution_version
        != process_spec.required_transformers_distribution_version
        or receipt.transformers_module_version
        != process_spec.required_transformers_module_version
        or receipt.capabilities != ("kt_bf16_amx_executed_v1",)
    ):
        raise Glm47LiveValidationError(
            "kernel runtime receipt does not match the model process spec"
        )


def require_current_process_binding(
    process_spec: SglangKtProcessLaunchSpec,
    *,
    executable: str | None = None,
    hostname: str | None = None,
    affinity_cpu_ids: tuple[int, ...] | None = None,
    memory_policy_nodes: tuple[int, ...] | None = None,
    cuda_visible_devices: str | None = None,
) -> None:
    """Fail unless the disposable child is constrained exactly as planned."""

    observed_executable = executable if executable is not None else sys.executable
    observed_hostname = (
        hostname
        if hostname is not None
        else socket.gethostname().split(".", maxsplit=1)[0].lower()
    )
    if affinity_cpu_ids is None:
        sched_getaffinity = cast(
            Callable[[int], set[int]],
            vars(os)["sched_getaffinity"],
        )
        observed_affinity = tuple(sorted(sched_getaffinity(0)))
    else:
        observed_affinity = affinity_cpu_ids
    observed_memory_nodes = (
        memory_policy_nodes
        if memory_policy_nodes is not None
        else current_bound_memory_nodes()
    )
    observed_visible_devices = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if (
        observed_executable != process_spec.executable
        or observed_hostname != str(process_spec.node_id)
        or observed_affinity != process_spec.cpu_cores
        or observed_memory_nodes != process_spec.memory_nodes
        or observed_visible_devices != process_spec.gpu_uuid
    ):
        raise Glm47LiveValidationError(
            "live validation process is not bound to the exact launch resources"
        )
