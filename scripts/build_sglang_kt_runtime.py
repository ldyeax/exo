#!/usr/bin/env python3
"""Build Exo's pinned GLM-4.7 SGLang-KTransformers wheels per host.

The default action is a read-only preflight. Pass ``--build`` only after the
JSON plan has been reviewed. Native compilation is performed from tracked Git
archives in an isolated build directory; the prepared source checkout is never
used as a build directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from scripts.prepare_sglang_kt_source import SglangKtSourcePlan

HostProfileName = Literal["dwagon", "fwuff"]
ReceiptStatus = Literal["preflight", "wheel_build_complete"]

SCHEMA_VERSION = 2
EXPECTED_PYTHON_VERSION = (3, 12)
EXPECTED_CUDA_RELEASE = "13.1"
CUDA_ARCHITECTURES = "86"
EXPECTED_SGLANG_WHEEL_ENTRY_COUNT = 2_205
EMBEDDED_PROVENANCE_SCHEMA_VERSION = 1
EMBEDDED_PROVENANCE_MODULE = "_exo_build_provenance.py"
PROVENANCE_DISTRIBUTIONS = frozenset(("sglang-kt", "kt-kernel"))
REQUIRED_CPU_FEATURES = frozenset(
    {
        "amx_bf16",
        "amx_int8",
        "amx_tile",
        "avx512_bf16",
        "avx512_vnni",
        "avx512f",
        "avx512vbmi",
    }
)


@dataclass(frozen=True)
class BootstrapWheelPin:
    distribution: str
    version: str
    filename: str
    sha256: str

    @property
    def requirement(self) -> str:
        return f"{self.distribution}=={self.version}"


BOOTSTRAP_WHEEL_PINS = (
    BootstrapWheelPin(
        distribution="pip",
        version="25.2",
        filename="pip-25.2-py3-none-any.whl",
        sha256="6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717",
    ),
    BootstrapWheelPin(
        distribution="setuptools",
        version="80.9.0",
        filename="setuptools-80.9.0-py3-none-any.whl",
        sha256="062d34222ad13e0cc312a4c02d73f059e86a4acbfbdea8f8f76b28c99f306922",
    ),
    BootstrapWheelPin(
        distribution="wheel",
        version="0.45.1",
        filename="wheel-0.45.1-py3-none-any.whl",
        sha256="708e7481cc80179af0e556bbf0cc00b8444c7321e2700b8d8580231d13017248",
    ),
)
BUILD_TOOL_REQUIREMENTS = tuple(pin.requirement for pin in BOOTSTRAP_WHEEL_PINS)
WHEEL_COMPONENTS = (
    ("sglang-kt", Path("third_party/sglang/python")),
    ("kt-kernel", Path("kt-kernel")),
    ("ktransformers", Path(".")),
)
KTRANSFORMERS_BUILD_SUBMODULE_PATHS = (
    Path("third_party/llama.cpp"),
    Path("third_party/pybind11"),
)


class RuntimeBuildError(RuntimeError):
    """Raised when a runtime build input or artifact is not exact."""


@dataclass(frozen=True)
class HostBuildProfile:
    name: HostProfileName
    cuda_root: Path
    parallel_jobs: int


HOST_BUILD_PROFILES: Mapping[HostProfileName, HostBuildProfile] = {
    "dwagon": HostBuildProfile(
        name="dwagon",
        cuda_root=Path("/opt/cuda"),
        parallel_jobs=16,
    ),
    "fwuff": HostBuildProfile(
        name="fwuff",
        cuda_root=Path("/usr/local/cuda-13.1"),
        parallel_jobs=16,
    ),
}


@dataclass(frozen=True)
class RuntimeSourcePins:
    ktransformers_revision: str
    sglang_revision: str
    sglang_submodule_path: Path

    @classmethod
    def admitted(cls) -> Self:
        source_plan = SglangKtSourcePlan.exo_default()
        if (
            source_plan.ktransformers_result_revision
            != GLM_4_7_FLASH_KTRANSFORMERS_REVISION
            or source_plan.sglang_result_revision != GLM_4_7_FLASH_SGLANG_REVISION
        ):
            raise RuntimeBuildError(
                "source-preparation and launch-spec GLM-4.7 revisions disagree"
            )
        return cls(
            ktransformers_revision=source_plan.ktransformers_result_revision,
            sglang_revision=source_plan.sglang_result_revision,
            sglang_submodule_path=source_plan.sglang_submodule_path,
        )


@dataclass(frozen=True)
class SubmoduleObservation:
    path: str
    revision: str


@dataclass(frozen=True)
class RuntimeSourceObservation:
    ktransformers_source: str
    ktransformers_revision: str
    sglang_revision: str
    package_version: str
    source_date_epoch: int
    submodules: tuple[SubmoduleObservation, ...]


@dataclass(frozen=True)
class ToolObservation:
    name: str
    path: str
    sha256: str
    version: str


@dataclass(frozen=True)
class RuntimeToolchainObservation:
    host_profile: HostProfileName
    hostname: str
    operating_system: str
    machine: str
    python_version: str
    python_implementation: str
    python_soabi: str
    cuda_root: str
    cuda_version: str
    cuda_version_manifest_sha256: str
    cuda_architectures: str
    cpu_features: tuple[str, ...]
    tools: tuple[ToolObservation, ...]

    def tool(self, name: str) -> ToolObservation:
        match = next((tool for tool in self.tools if tool.name == name), None)
        if match is None:
            raise RuntimeBuildError(f"toolchain receipt has no {name} observation")
        return match


@dataclass(frozen=True)
class RuntimeBuildLayout:
    build_root: str
    state: str
    cache: str
    logs: str
    wheels: str
    receipt: str


@dataclass(frozen=True)
class WheelArtifact:
    distribution: str
    version: str
    filename: str
    path: str
    size_bytes: int
    sha256: str
    root_is_purelib: bool
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeBuildReceipt:
    schema_version: int
    status: ReceiptStatus
    build_id: str
    builder_sha256: str
    source: RuntimeSourceObservation
    toolchain: RuntimeToolchainObservation
    layout: RuntimeBuildLayout
    build_environment: tuple[tuple[str, str], ...]
    commands: tuple[tuple[str, ...], ...]
    bootstrap_wheels: tuple[WheelArtifact, ...]
    runtime_wheels: tuple[WheelArtifact, ...]
    completed_at_utc: str | None


@dataclass(frozen=True)
class RuntimeBuildPlan:
    source_path: Path
    python_executable: Path
    profile: HostBuildProfile
    pins: RuntimeSourcePins
    source: RuntimeSourceObservation
    toolchain: RuntimeToolchainObservation
    build_id: str
    builder_sha256: str
    layout: RuntimeBuildLayout
    environment: tuple[tuple[str, str], ...]
    commands: tuple[tuple[str, ...], ...]

    def preflight_receipt(self) -> RuntimeBuildReceipt:
        return RuntimeBuildReceipt(
            schema_version=SCHEMA_VERSION,
            status="preflight",
            build_id=self.build_id,
            builder_sha256=self.builder_sha256,
            source=self.source,
            toolchain=self.toolchain,
            layout=self.layout,
            build_environment=self.environment,
            commands=self.commands,
            bootstrap_wheels=(),
            runtime_wheels=(),
            completed_at_utc=None,
        )


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        tuple(arguments),
        cwd=cwd,
        env=None if environment is None else dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeBuildError(f"{' '.join(arguments)} failed: {detail}")
    return result


def _git_output(repository: Path, *arguments: str) -> str:
    return _run(("git", "-C", str(repository), *arguments)).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _embedded_provenance_contents(source: RuntimeSourceObservation) -> bytes:
    return (
        '"""Generated by Exo\'s pinned SGLang-KTransformers builder."""\n'
        "\n"
        f"SCHEMA_VERSION = {EMBEDDED_PROVENANCE_SCHEMA_VERSION}\n"
        f'KTRANSFORMERS_REVISION = "{source.ktransformers_revision}"\n'
        f'SGLANG_REVISION = "{source.sglang_revision}"\n'
    ).encode("ascii")


def _json_object(contents: str, description: str) -> dict[str, object]:
    try:
        value = cast(object, json.loads(contents))
    except json.JSONDecodeError as error:
        raise RuntimeBuildError(f"invalid {description} JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeBuildError(f"{description} JSON is not an object")
    return cast(dict[str, object], value)


def _required_string(values: Mapping[str, object], key: str, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeBuildError(f"{description} has no string {key}")
    return value


def _require_repository_root(repository: Path, description: str) -> None:
    result = _run(
        ("git", "-C", str(repository), "rev-parse", "--show-toplevel"),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeBuildError(f"{description} is not an initialized Git repository")
    try:
        observed = Path(result.stdout.strip()).resolve(strict=True)
        expected = repository.resolve(strict=True)
    except OSError as error:
        raise RuntimeBuildError(f"cannot resolve {description}: {error}") from error
    if observed != expected:
        raise RuntimeBuildError(
            f"{description} resolves through a different repository root: {observed}"
        )


def _require_clean(
    repository: Path,
    description: str,
    *,
    ignore_submodules: Literal["all", "none"] = "none",
) -> None:
    status = _git_output(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        f"--ignore-submodules={ignore_submodules}",
    )
    if status:
        raise RuntimeBuildError(f"{description} is not clean: {status}")


def _head_revision(repository: Path) -> str:
    return _git_output(repository, "rev-parse", "HEAD").lower()


def _direct_gitlinks(repository: Path) -> tuple[tuple[Path, str], ...]:
    result = subprocess.run(
        ("git", "-C", str(repository), "ls-files", "--stage", "-z"),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip() or "no diagnostic"
        raise RuntimeBuildError(f"cannot inspect submodules in {repository}: {detail}")

    gitlinks: list[tuple[Path, str]] = []
    for raw_record in result.stdout.split(b"\0"):
        if not raw_record:
            continue
        metadata, separator, raw_path = raw_record.partition(b"\t")
        fields = metadata.decode("ascii").split()
        if not separator or len(fields) != 3:
            raise RuntimeBuildError(f"unexpected git index record in {repository}")
        mode, revision, stage = fields
        if mode != "160000":
            continue
        if stage != "0":
            raise RuntimeBuildError(f"unmerged submodule entry in {repository}")
        path = Path(os.fsdecode(raw_path))
        pure_path = PurePosixPath(path.as_posix())
        if pure_path.is_absolute() or ".." in pure_path.parts:
            raise RuntimeBuildError(f"unsafe submodule path in {repository}: {path}")
        gitlinks.append((path, revision.lower()))
    return tuple(sorted(gitlinks, key=lambda item: item[0].as_posix()))


def _observe_build_submodules(
    repository: Path,
    required_paths: Sequence[Path],
    *,
    independently_pinned_paths: frozenset[Path] = frozenset(),
) -> tuple[SubmoduleObservation, ...]:
    unexpected_pins = independently_pinned_paths.difference(required_paths)
    if unexpected_pins:
        raise RuntimeBuildError(
            "independently pinned paths are not required build submodules: "
            + ", ".join(sorted(path.as_posix() for path in unexpected_pins))
        )
    gitlinks = dict(_direct_gitlinks(repository))
    observations: list[SubmoduleObservation] = []
    for direct_path in sorted(required_paths, key=lambda path: path.as_posix()):
        expected_revision = gitlinks.get(direct_path)
        if expected_revision is None:
            raise RuntimeBuildError(
                f"required build submodule has no gitlink: {direct_path.as_posix()}"
            )
        submodule = repository / direct_path
        description = f"build submodule {direct_path.as_posix()}"
        _require_repository_root(submodule, description)
        observed_revision = _head_revision(submodule)
        if (
            direct_path not in independently_pinned_paths
            and observed_revision != expected_revision
        ):
            raise RuntimeBuildError(
                f"{description} revision is {observed_revision}, "
                f"expected gitlink {expected_revision}"
            )
        _require_clean(submodule, description, ignore_submodules="all")
        observations.append(
            SubmoduleObservation(
                path=direct_path.as_posix(),
                revision=observed_revision,
            )
        )
    return tuple(observations)


def _read_package_version(version_file: Path) -> str:
    try:
        module = ast.parse(version_file.read_text(), filename=str(version_file))
    except (OSError, SyntaxError) as error:
        raise RuntimeBuildError(f"cannot parse {version_file}: {error}") from error
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(
            statement.value.value, str
        ):
            return statement.value.value
    raise RuntimeBuildError(f"{version_file} has no literal __version__ assignment")


def _require_build_roots(source: Path, sglang_path: Path) -> None:
    required_paths = (
        source / "pyproject.toml",
        source / "setup.py",
        source / "version.py",
        source / "kt-kernel/pyproject.toml",
        source / "kt-kernel/setup.py",
        source / "kt-kernel/CMakeLists.txt",
        source / sglang_path / "python/pyproject.toml",
        source / sglang_path / "python/setup.py",
    )
    missing = tuple(str(path) for path in required_paths if not path.is_file())
    if missing:
        raise RuntimeBuildError(
            "prepared source lacks required build files: " + ", ".join(missing)
        )


def observe_runtime_source(
    ktransformers_source: Path,
    pins: RuntimeSourcePins | None = None,
) -> RuntimeSourceObservation:
    selected_pins = pins or RuntimeSourcePins.admitted()
    source = ktransformers_source.expanduser().resolve(strict=True)
    _require_repository_root(source, "KTransformers source")
    revision = _head_revision(source)
    if revision != selected_pins.ktransformers_revision:
        raise RuntimeBuildError(
            f"KTransformers revision is {revision}, "
            f"expected {selected_pins.ktransformers_revision}"
        )
    _require_clean(source, "KTransformers source", ignore_submodules="all")

    required_submodules = (
        *KTRANSFORMERS_BUILD_SUBMODULE_PATHS,
        selected_pins.sglang_submodule_path,
    )
    submodules = tuple(
        sorted(
            _observe_build_submodules(
                source,
                required_submodules,
                independently_pinned_paths=frozenset(
                    (selected_pins.sglang_submodule_path,)
                ),
            ),
            key=lambda observation: observation.path,
        )
    )
    sglang_path = selected_pins.sglang_submodule_path.as_posix()
    sglang = next(
        (observation for observation in submodules if observation.path == sglang_path),
        None,
    )
    if sglang is None:
        raise RuntimeBuildError(f"required SGLang submodule is absent: {sglang_path}")
    if sglang.revision != selected_pins.sglang_revision:
        raise RuntimeBuildError(
            f"SGLang revision is {sglang.revision}, "
            f"expected {selected_pins.sglang_revision}"
        )
    _require_build_roots(source, selected_pins.sglang_submodule_path)

    timestamp_text = _git_output(source, "show", "-s", "--format=%ct", "HEAD")
    if not timestamp_text.isdecimal():
        raise RuntimeBuildError(
            f"invalid KTransformers commit timestamp: {timestamp_text}"
        )
    return RuntimeSourceObservation(
        ktransformers_source=str(source),
        ktransformers_revision=revision,
        sglang_revision=sglang.revision,
        package_version=_read_package_version(source / "version.py"),
        source_date_epoch=int(timestamp_text),
        submodules=submodules,
    )


def _resolve_executable(name: str) -> Path:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeBuildError(f"required executable is absent from PATH: {name}")
    try:
        return Path(executable).resolve(strict=True)
    except OSError as error:
        raise RuntimeBuildError(f"cannot resolve executable {name}: {error}") from error


def _observe_tool(
    name: str, path: Path, version_arguments: Sequence[str]
) -> ToolObservation:
    result = _run((str(path), *version_arguments))
    version = (result.stdout.strip() or result.stderr.strip()).splitlines()[0]
    if not version:
        raise RuntimeBuildError(f"{name} returned no version")
    return ToolObservation(
        name=name,
        path=str(path),
        sha256=_sha256(path),
        version=version,
    )


def _observe_file(name: str, path: Path, version: str) -> ToolObservation:
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeBuildError(
            f"cannot resolve required {name} file {path}: {error}"
        ) from error
    if not resolved_path.is_file():
        raise RuntimeBuildError(f"required {name} path is not a file: {resolved_path}")
    return ToolObservation(
        name=name,
        path=str(resolved_path),
        sha256=_sha256(resolved_path),
        version=version,
    )


def _common_cpu_features(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> frozenset[str]:
    try:
        contents = cpuinfo_path.read_text(errors="replace")
    except OSError as error:
        raise RuntimeBuildError(f"cannot read {cpuinfo_path}: {error}") from error
    flag_sets: list[frozenset[str]] = [
        frozenset(line.partition(":")[2].strip().lower().split())
        for line in contents.splitlines()
        if line.lower().startswith("flags") and ":" in line
    ]
    if not flag_sets:
        raise RuntimeBuildError("/proc/cpuinfo contains no x86 CPU flags")
    common_features = set(flag_sets[0])
    for flag_set in flag_sets[1:]:
        common_features.intersection_update(flag_set)
    return frozenset(common_features)


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+(?:\.\d+)+)", text)
    if match is None:
        raise RuntimeBuildError(f"cannot parse version from: {text}")
    return tuple(int(component) for component in match.group(1).split("."))


def observe_runtime_toolchain(
    profile: HostBuildProfile,
    python_executable: Path | None = None,
) -> tuple[Path, RuntimeToolchainObservation]:
    hostname = socket.gethostname().split(".", maxsplit=1)[0].lower()
    if hostname != profile.name:
        raise RuntimeBuildError(
            f"host profile {profile.name} cannot run on observed host {hostname}"
        )
    operating_system = platform.system()
    machine = platform.machine().lower()
    if operating_system != "Linux" or machine != "x86_64":
        raise RuntimeBuildError(
            f"unsupported build platform: {operating_system}/{machine}; "
            "expected Linux/x86_64"
        )

    if python_executable is None:
        python_path = _resolve_executable("python3.12")
    else:
        try:
            python_path = python_executable.expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeBuildError(
                f"cannot resolve Python 3.12 executable {python_executable}: {error}"
            ) from error
    python_probe = _run(
        (
            str(python_path),
            "-c",
            (
                "import json,platform,sys,sysconfig,venv;"
                "print(json.dumps({'executable':sys.executable,"
                "'version':platform.python_version(),"
                "'implementation':platform.python_implementation(),"
                "'soabi':sysconfig.get_config_var('SOABI')}))"
            ),
        )
    )
    try:
        python_facts = _json_object(python_probe.stdout, "Python toolchain probe")
        python_version = _required_string(python_facts, "version", "Python probe")
        python_implementation = _required_string(
            python_facts, "implementation", "Python probe"
        )
        python_soabi = _required_string(python_facts, "soabi", "Python probe")
        probed_python = Path(
            _required_string(python_facts, "executable", "Python probe")
        ).resolve(strict=True)
    except (OSError, ValueError) as error:
        raise RuntimeBuildError(f"invalid Python toolchain probe: {error}") from error
    if probed_python != python_path:
        raise RuntimeBuildError(
            f"Python probe resolved {probed_python}, expected {python_path}"
        )
    if _version_tuple(python_version)[:2] != EXPECTED_PYTHON_VERSION:
        raise RuntimeBuildError(
            f"runtime build requires Python 3.12, observed {python_version}"
        )
    if python_implementation != "CPython" or not python_soabi.startswith("cpython-312"):
        raise RuntimeBuildError(
            f"runtime build requires CPython 3.12 ABI, observed "
            f"{python_implementation}/{python_soabi}"
        )

    try:
        cuda_root = profile.cuda_root.resolve(strict=True)
    except OSError as error:
        raise RuntimeBuildError(
            f"configured CUDA root is unavailable for {profile.name}: "
            f"{profile.cuda_root}: {error}"
        ) from error
    nvcc = (cuda_root / "bin/nvcc").resolve(strict=True)
    try:
        nvcc.relative_to(cuda_root)
    except ValueError as error:
        raise RuntimeBuildError(f"nvcc escapes configured CUDA root: {nvcc}") from error

    cuda_manifest = cuda_root / "version.json"
    try:
        cuda_versions = _json_object(cuda_manifest.read_text(), "CUDA version manifest")
        cuda_entry = cuda_versions.get("cuda")
        if not isinstance(cuda_entry, dict):
            raise RuntimeBuildError("CUDA version manifest has no cuda object")
        cuda_version = _required_string(
            cast(dict[str, object], cuda_entry),
            "version",
            "CUDA version manifest cuda object",
        )
    except (OSError, ValueError) as error:
        raise RuntimeBuildError(
            f"cannot read CUDA version manifest {cuda_manifest}: {error}"
        ) from error
    if not cuda_version.startswith(f"{EXPECTED_CUDA_RELEASE}."):
        raise RuntimeBuildError(
            f"CUDA SDK is {cuda_version}, expected {EXPECTED_CUDA_RELEASE}.x"
        )

    cmake = _resolve_executable("cmake")
    ninja = _resolve_executable("ninja")
    compiler = _resolve_executable("gcc")
    compiler_cxx = _resolve_executable("g++")
    python_tool = _observe_tool(
        "python",
        python_path,
        ("--version",),
    )
    nvcc_tool = _observe_tool("nvcc", nvcc, ("--version",))
    if f"release {EXPECTED_CUDA_RELEASE}" not in _run((str(nvcc), "--version")).stdout:
        raise RuntimeBuildError(
            f"nvcc is not CUDA {EXPECTED_CUDA_RELEASE}: {nvcc_tool.version}"
        )
    cmake_tool = _observe_tool("cmake", cmake, ("--version",))
    if _version_tuple(cmake_tool.version) < (3, 16):
        raise RuntimeBuildError(f"CMake 3.16+ is required: {cmake_tool.version}")
    ninja_tool = _observe_tool("ninja", ninja, ("--version",))
    compiler_tool = _observe_tool("cc", compiler, ("--version",))
    compiler_cxx_tool = _observe_tool("cxx", compiler_cxx, ("--version",))
    cudart_static_tool = _observe_file(
        "cudart_static",
        cuda_root / "lib64/libcudart_static.a",
        cuda_version,
    )
    compiler_version = _run(
        (str(compiler), "-dumpfullversion", "-dumpversion")
    ).stdout.strip()
    compiler_cxx_version = _run(
        (str(compiler_cxx), "-dumpfullversion", "-dumpversion")
    ).stdout.strip()
    if compiler_version != compiler_cxx_version:
        raise RuntimeBuildError(
            f"C/C++ compiler versions differ: {compiler_version} vs "
            f"{compiler_cxx_version}"
        )

    cpu_features = _common_cpu_features()
    missing_features = tuple(sorted(REQUIRED_CPU_FEATURES - cpu_features))
    if missing_features:
        raise RuntimeBuildError(
            "host cannot build the admitted AMX-BF16 runtime; missing common CPU "
            "features: " + ", ".join(missing_features)
        )
    return python_path, RuntimeToolchainObservation(
        host_profile=profile.name,
        hostname=hostname,
        operating_system=operating_system,
        machine=machine,
        python_version=python_version,
        python_implementation=python_implementation,
        python_soabi=python_soabi,
        cuda_root=str(cuda_root),
        cuda_version=cuda_version,
        cuda_version_manifest_sha256=_sha256(cuda_manifest),
        cuda_architectures=CUDA_ARCHITECTURES,
        cpu_features=tuple(sorted(cpu_features)),
        tools=tuple(
            sorted(
                (
                    python_tool,
                    nvcc_tool,
                    cmake_tool,
                    ninja_tool,
                    compiler_tool,
                    compiler_cxx_tool,
                    cudart_static_tool,
                ),
                key=lambda tool: tool.name,
            )
        ),
    )


def _semantic_build_environment(
    source: RuntimeSourceObservation,
    profile: HostBuildProfile,
) -> dict[str, str]:
    return {
        "CMAKE_BUILD_PARALLEL_LEVEL": str(profile.parallel_jobs),
        "CMAKE_GENERATOR": "Ninja",
        "CPUINFER_BUILD_ALL_VARIANTS": "0",
        "CPUINFER_BUILD_TYPE": "Release",
        "CPUINFER_CPU_INSTRUCT": "NATIVE",
        "CPUINFER_CUDA_ARCHS": CUDA_ARCHITECTURES,
        "CPUINFER_CUDA_STATIC_RUNTIME": "ON",
        "CPUINFER_ENABLE_AMX": "ON",
        "CPUINFER_ENABLE_AVX512": "ON",
        "CPUINFER_ENABLE_AVX512_BF16": "ON",
        "CPUINFER_ENABLE_AVX512_VBMI": "ON",
        "CPUINFER_ENABLE_AVX512_VNNI": "ON",
        "CPUINFER_ENABLE_BLIS": "OFF",
        "CPUINFER_ENABLE_CPPTRACE": "OFF",
        "CPUINFER_ENABLE_KML": "OFF",
        "CPUINFER_ENABLE_LTO": "OFF",
        "CPUINFER_ENABLE_MLA": "OFF",
        "CPUINFER_FORCE_REBUILD": "1",
        "CPUINFER_PARALLEL": str(profile.parallel_jobs),
        "CPUINFER_USE_CUDA": "1",
        "CPUINFER_USE_MACA": "0",
        "CPUINFER_USE_MUSA": "0",
        "CPUINFER_USE_ROCM": "0",
        "CPUINFER_VERSION": source.package_version,
        "PYTHONHASHSEED": "0",
        "SGLANG_KT_VERSION": source.package_version,
        "SOURCE_DATE_EPOCH": str(source.source_date_epoch),
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "TZ": "UTC",
    }


def _build_id_payload(
    source: RuntimeSourceObservation,
    toolchain: RuntimeToolchainObservation,
    profile: HostBuildProfile,
    builder_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": builder_sha256,
        "source": {
            "ktransformers_revision": source.ktransformers_revision,
            "sglang_revision": source.sglang_revision,
            "package_version": source.package_version,
            "source_date_epoch": source.source_date_epoch,
            "submodules": [asdict(observation) for observation in source.submodules],
        },
        "toolchain": asdict(toolchain),
        "build": {
            "bootstrap_wheels": tuple(asdict(pin) for pin in BOOTSTRAP_WHEEL_PINS),
            "environment": tuple(
                sorted(_semantic_build_environment(source, profile).items())
            ),
            "wheel_components": tuple(
                (name, path.as_posix()) for name, path in WHEEL_COMPONENTS
            ),
        },
    }


def calculate_runtime_build_id(
    source: RuntimeSourceObservation,
    toolchain: RuntimeToolchainObservation,
    profile: HostBuildProfile,
    builder_sha256: str,
) -> str:
    """Return the content-addressed ID for one exact native build contract."""
    return hashlib.sha256(
        _canonical_json(_build_id_payload(source, toolchain, profile, builder_sha256))
    ).hexdigest()


def _make_layout(output_root: Path, host: str, build_id: str) -> RuntimeBuildLayout:
    build_root = output_root.expanduser().resolve(strict=False) / host / build_id
    return RuntimeBuildLayout(
        build_root=str(build_root),
        state=str(build_root / "state"),
        cache=str(build_root / "cache"),
        logs=str(build_root / "state/logs"),
        wheels=str(build_root / "wheels"),
        receipt=str(build_root / "build-receipt.json"),
    )


def _require_output_outside_git_worktree(output_root: Path) -> None:
    candidate = output_root.expanduser().resolve(strict=False)
    existing_ancestor = candidate
    while (
        not existing_ancestor.exists() and existing_ancestor != existing_ancestor.parent
    ):
        existing_ancestor = existing_ancestor.parent
    result = _run(
        ("git", "-C", str(existing_ancestor), "rev-parse", "--show-toplevel"),
        check=False,
    )
    if result.returncode != 0:
        return
    try:
        worktree = Path(result.stdout.strip()).resolve(strict=True)
        candidate.relative_to(worktree)
    except (OSError, ValueError):
        return
    raise RuntimeBuildError(
        f"output root {candidate} is inside Git worktree {worktree}; KT's CMake "
        "hook detection would target the wrong repository"
    )


def runtime_build_environment(
    source: RuntimeSourceObservation,
    toolchain: RuntimeToolchainObservation,
    profile: HostBuildProfile,
    layout: RuntimeBuildLayout,
) -> tuple[tuple[str, str], ...]:
    state = Path(layout.state)
    cache = Path(layout.cache)
    venv_bin = state / "venv/bin"
    tool_directories = (
        venv_bin,
        Path(toolchain.tool("nvcc").path).parent,
        Path(toolchain.tool("cmake").path).parent,
        Path(toolchain.tool("ninja").path).parent,
        Path(toolchain.tool("cc").path).parent,
        Path(toolchain.tool("cxx").path).parent,
        Path("/usr/bin"),
        Path("/bin"),
    )
    path_entries = tuple(dict.fromkeys(str(path) for path in tool_directories))
    values = _semantic_build_environment(source, profile)
    values.update(
        {
            "CC": toolchain.tool("cc").path,
            "CUDAHOSTCXX": toolchain.tool("cxx").path,
            "CUDA_HOME": toolchain.cuda_root,
            "CUDA_PATH": toolchain.cuda_root,
            "CXX": toolchain.tool("cxx").path,
            "HOME": str(state / "home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.pathsep.join(path_entries),
            "PIP_CACHE_DIR": str(cache / "pip"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "PIP_NO_INPUT": "1",
            "TMPDIR": str(state / "tmp"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
    )
    return tuple(sorted(values.items()))


def runtime_build_commands(
    python_executable: Path,
    layout: RuntimeBuildLayout,
) -> tuple[tuple[str, ...], ...]:
    state = Path(layout.state)
    build_python = state / "venv/bin/python"
    source = state / "source/ktransformers"
    bootstrap = state / "bootstrap-wheels"
    component_output = state / "wheel-output"
    commands: list[tuple[str, ...]] = [
        (str(python_executable), "-m", "venv", str(state / "venv")),
        (
            str(build_python),
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(bootstrap),
            *BUILD_TOOL_REQUIREMENTS,
        ),
        (
            str(build_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(bootstrap),
            *BUILD_TOOL_REQUIREMENTS,
        ),
    ]
    for distribution, relative_source in WHEEL_COMPONENTS:
        commands.append(
            (
                str(build_python),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(component_output / distribution),
                str(source / relative_source),
            )
        )
    return tuple(commands)


def plan_runtime_build(
    ktransformers_source: Path,
    output_root: Path,
    profile: HostBuildProfile,
    python_executable: Path | None = None,
    *,
    pins: RuntimeSourcePins | None = None,
    toolchain: RuntimeToolchainObservation | None = None,
) -> RuntimeBuildPlan:
    selected_pins = pins or RuntimeSourcePins.admitted()
    _require_output_outside_git_worktree(output_root)
    source = observe_runtime_source(ktransformers_source, selected_pins)
    if toolchain is None:
        selected_python, observed_toolchain = observe_runtime_toolchain(
            profile, python_executable
        )
    else:
        observed_toolchain = toolchain
        if observed_toolchain.host_profile != profile.name:
            raise RuntimeBuildError(
                f"injected toolchain profile is {observed_toolchain.host_profile}, "
                f"expected {profile.name}"
            )
        selected_python = Path(observed_toolchain.tool("python").path)
        if python_executable is not None and selected_python != python_executable:
            raise RuntimeBuildError(
                f"injected Python path is {selected_python}, expected "
                f"{python_executable}"
            )

    builder_sha256 = _sha256(Path(__file__).resolve(strict=True))
    build_id = calculate_runtime_build_id(
        source,
        observed_toolchain,
        profile,
        builder_sha256,
    )
    layout = _make_layout(output_root, profile.name, build_id)
    environment = runtime_build_environment(source, observed_toolchain, profile, layout)
    commands = runtime_build_commands(selected_python, layout)
    return RuntimeBuildPlan(
        source_path=Path(source.ktransformers_source),
        python_executable=selected_python,
        profile=profile,
        pins=selected_pins,
        source=source,
        toolchain=observed_toolchain,
        build_id=build_id,
        builder_sha256=builder_sha256,
        layout=layout,
        environment=environment,
        commands=commands,
    )


def _export_git_archive(
    repository: Path, destination: Path, archive_path: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    _run(
        (
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            "HEAD",
        )
    )
    try:
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(destination, filter="data")
    finally:
        archive_path.unlink(missing_ok=True)


def export_runtime_source_snapshot(plan: RuntimeBuildPlan) -> Path:
    state = Path(plan.layout.state)
    snapshot = state / "source/ktransformers"
    archive_directory = state / "archives"
    archive_directory.mkdir(parents=True, exist_ok=True)
    _export_git_archive(
        plan.source_path,
        snapshot,
        archive_directory / "ktransformers.tar",
    )
    for index, submodule in enumerate(plan.source.submodules):
        source = plan.source_path / submodule.path
        destination = snapshot / submodule.path
        if destination.exists():
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            else:
                raise RuntimeBuildError(
                    f"Git archive unexpectedly populated submodule path: {destination}"
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _export_git_archive(
            source,
            destination,
            archive_directory / f"submodule-{index}.tar",
        )
    _require_build_roots(snapshot, plan.pins.sglang_submodule_path)
    provenance_contents = _embedded_provenance_contents(plan.source)
    provenance_paths = (
        snapshot
        / plan.pins.sglang_submodule_path
        / "python/sglang"
        / EMBEDDED_PROVENANCE_MODULE,
        snapshot / "kt-kernel/python" / EMBEDDED_PROVENANCE_MODULE,
    )
    for provenance_path in provenance_paths:
        if provenance_path.exists():
            raise RuntimeBuildError(
                f"source archive already contains generated provenance: "
                f"{provenance_path}"
            )
        provenance_path.write_bytes(provenance_contents)
    return snapshot


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def inspect_wheel(
    wheel_path: Path,
    expected_distribution: str,
    expected_version: str,
    receipt_root: Path,
    *,
    expected_provenance: bytes | None = None,
) -> WheelArtifact:
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel_names = tuple(wheel.namelist())
            if len(set(wheel_names)) != len(wheel_names):
                raise RuntimeBuildError(
                    f"wheel {wheel_path} contains duplicate entries"
                )
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in wheel_names
            ):
                raise RuntimeBuildError(f"wheel {wheel_path} contains an unsafe path")
            metadata_paths = tuple(
                name
                for name in wheel_names
                if len(PurePosixPath(name).parts) == 2
                and PurePosixPath(name).parent.name.endswith(".dist-info")
                and PurePosixPath(name).name == "METADATA"
            )
            wheel_paths = tuple(
                name
                for name in wheel_names
                if len(PurePosixPath(name).parts) == 2
                and PurePosixPath(name).parent.name.endswith(".dist-info")
                and PurePosixPath(name).name == "WHEEL"
            )
            if (
                len(metadata_paths) != 1
                or len(wheel_paths) != 1
                or PurePosixPath(metadata_paths[0]).parent
                != PurePosixPath(wheel_paths[0]).parent
            ):
                raise RuntimeBuildError(
                    f"wheel {wheel_path} has ambiguous dist-info metadata"
                )
            metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
            wheel_metadata = BytesParser().parsebytes(wheel.read(wheel_paths[0]))
            distribution = str(metadata["Name"] or "")
            version = str(metadata["Version"] or "")
            purelib_text = str(wheel_metadata["Root-Is-Purelib"] or "").lower()
            if purelib_text not in {"true", "false"}:
                raise RuntimeBuildError(
                    f"wheel {wheel_path} has invalid Root-Is-Purelib metadata"
                )
            tags = tuple(sorted(str(tag) for tag in wheel_metadata.get_all("Tag", [])))
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeBuildError(
            f"cannot inspect wheel {wheel_path}: {error}"
        ) from error
    if _normalize_distribution_name(distribution) != _normalize_distribution_name(
        expected_distribution
    ):
        raise RuntimeBuildError(
            f"wheel {wheel_path} is {distribution}, expected {expected_distribution}"
        )
    if version != expected_version:
        raise RuntimeBuildError(
            f"wheel {wheel_path} version is {version}, expected {expected_version}"
        )
    root_is_purelib = purelib_text == "true"
    if expected_distribution in PROVENANCE_DISTRIBUTIONS:
        if expected_provenance is None:
            raise RuntimeBuildError(
                f"{expected_distribution} inspection requires exact build provenance"
            )
        package_name = "sglang" if expected_distribution == "sglang-kt" else "kt_kernel"
        provenance_name = f"{package_name}/{EMBEDDED_PROVENANCE_MODULE}"
        if provenance_name not in wheel_names:
            raise RuntimeBuildError(
                f"{expected_distribution} wheel lacks embedded build provenance"
            )
        with zipfile.ZipFile(wheel_path) as wheel:
            observed_provenance = wheel.read(provenance_name)
        if observed_provenance != expected_provenance:
            raise RuntimeBuildError(
                f"{expected_distribution} wheel has unexpected build provenance"
            )
    if expected_distribution == "kt-kernel":
        if root_is_purelib:
            raise RuntimeBuildError(
                "kt-kernel wheel is unexpectedly platform-independent"
            )
        if not any(
            name.startswith("kt_kernel/") and name.endswith(".so")
            for name in wheel_names
        ):
            raise RuntimeBuildError("kt-kernel wheel contains no native extension")
        if not any(tag.startswith("cp312-cp312-") for tag in tags):
            raise RuntimeBuildError("kt-kernel wheel is not tagged for CPython 3.12")
    if expected_distribution == "sglang-kt":
        if len(wheel_names) != EXPECTED_SGLANG_WHEEL_ENTRY_COUNT:
            raise RuntimeBuildError(
                f"sglang-kt wheel contains {len(wheel_names)} entries, expected "
                f"{EXPECTED_SGLANG_WHEEL_ENTRY_COUNT}; the source tree may contain "
                "ignored build output"
            )
        forbidden_prefixes = ("build/", "python/build/", "sglang/build/")
        if any(name.startswith(forbidden_prefixes) for name in wheel_names):
            raise RuntimeBuildError("sglang-kt wheel contains a nested build tree")
        required_entries = {
            "sglang/srt/layers/moe/kt_ep_wrapper.py",
            "sglang/srt/models/deepseek_v2.py",
        }
        missing_entries = sorted(required_entries - set(wheel_names))
        if missing_entries:
            raise RuntimeBuildError(
                "sglang-kt wheel lacks patched runtime modules: "
                + ", ".join(missing_entries)
            )
    try:
        relative_path = wheel_path.relative_to(receipt_root).as_posix()
    except ValueError as error:
        raise RuntimeBuildError(
            f"wheel {wheel_path} is outside receipt root {receipt_root}"
        ) from error
    return WheelArtifact(
        distribution=distribution,
        version=version,
        filename=wheel_path.name,
        path=relative_path,
        size_bytes=wheel_path.stat().st_size,
        sha256=_sha256(wheel_path),
        root_is_purelib=root_is_purelib,
        tags=tags,
    )


def _collect_bootstrap_wheels(plan: RuntimeBuildPlan) -> tuple[WheelArtifact, ...]:
    bootstrap = Path(plan.layout.state) / "bootstrap-wheels"
    wheel_paths = tuple(sorted(bootstrap.glob("*.whl")))
    expected_by_filename = {pin.filename: pin for pin in BOOTSTRAP_WHEEL_PINS}
    if len(wheel_paths) != len(BOOTSTRAP_WHEEL_PINS) or {
        path.name for path in wheel_paths
    } != set(expected_by_filename):
        raise RuntimeBuildError(
            "bootstrap wheel filenames do not match the pinned set: "
            f"observed {sorted(path.name for path in wheel_paths)}, "
            f"expected {sorted(expected_by_filename)}"
        )
    artifacts: list[WheelArtifact] = []
    for path in wheel_paths:
        pin = expected_by_filename[path.name]
        artifact = inspect_wheel(
            path,
            expected_distribution=pin.distribution,
            expected_version=pin.version,
            receipt_root=Path(plan.layout.build_root),
        )
        if artifact.sha256 != pin.sha256:
            raise RuntimeBuildError(
                f"bootstrap wheel {pin.filename} SHA-256 is {artifact.sha256}, "
                f"expected {pin.sha256}"
            )
        if not artifact.root_is_purelib or artifact.tags != ("py3-none-any",):
            raise RuntimeBuildError(
                f"bootstrap wheel {pin.filename} is not the pinned pure-Python wheel"
            )
        artifacts.append(artifact)
    return tuple(sorted(artifacts, key=lambda item: item.distribution))


def _collect_runtime_wheels(plan: RuntimeBuildPlan) -> tuple[WheelArtifact, ...]:
    state = Path(plan.layout.state)
    wheel_directory = Path(plan.layout.wheels)
    artifacts: list[WheelArtifact] = []
    for distribution, _ in WHEEL_COMPONENTS:
        candidates = tuple(
            sorted((state / "wheel-output" / distribution).glob("*.whl"))
        )
        if len(candidates) != 1:
            raise RuntimeBuildError(
                f"{distribution} build produced {len(candidates)} wheels, expected one"
            )
        destination = wheel_directory / candidates[0].name
        if destination.exists():
            raise RuntimeBuildError(
                f"duplicate runtime wheel filename: {destination.name}"
            )
        shutil.move(candidates[0], destination)
        artifacts.append(
            inspect_wheel(
                destination,
                expected_distribution=distribution,
                expected_version=plan.source.package_version,
                receipt_root=Path(plan.layout.build_root),
                expected_provenance=(
                    _embedded_provenance_contents(plan.source)
                    if distribution in PROVENANCE_DISTRIBUTIONS
                    else None
                ),
            )
        )
    return tuple(sorted(artifacts, key=lambda item: item.distribution))


def _write_receipt(receipt: RuntimeBuildReceipt) -> None:
    receipt_path = Path(receipt.layout.receipt)
    temporary_path = receipt_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary_path, receipt_path)


def _run_logged(
    arguments: Sequence[str],
    index: int,
    logs: Path,
    environment: Mapping[str, str],
) -> None:
    stdout_path = logs / f"{index:02d}.stdout.log"
    stderr_path = logs / f"{index:02d}.stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            tuple(arguments),
            env=dict(environment),
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    if result.returncode == 0:
        return
    try:
        detail = stderr_path.read_text(errors="replace")[-16_384:].strip()
        if not detail:
            detail = stdout_path.read_text(errors="replace")[-16_384:].strip()
    except OSError:
        detail = "see retained command logs"
    raise RuntimeBuildError(
        f"build command {index} failed with exit {result.returncode}; "
        f"logs: {stdout_path}, {stderr_path}; {detail}"
    )


def execute_runtime_build(
    plan: RuntimeBuildPlan,
    *,
    build: bool = False,
) -> RuntimeBuildReceipt:
    if not build:
        return plan.preflight_receipt()

    current_source = observe_runtime_source(plan.source_path, plan.pins)
    selected_python, current_toolchain = observe_runtime_toolchain(
        plan.profile,
        plan.python_executable,
    )
    if current_source != plan.source or current_toolchain != plan.toolchain:
        raise RuntimeBuildError("source or toolchain changed after preflight")
    if selected_python != plan.python_executable:
        raise RuntimeBuildError("Python executable changed after preflight")

    build_root = Path(plan.layout.build_root)
    if build_root.exists():
        raise RuntimeBuildError(
            f"build root already exists; refusing to reuse incomplete state: {build_root}"
        )
    build_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        build_root.mkdir()
    except FileExistsError as error:
        raise RuntimeBuildError(
            f"another builder created the build root concurrently: {build_root}"
        ) from error
    state = Path(plan.layout.state)
    cache = Path(plan.layout.cache)
    logs = Path(plan.layout.logs)
    wheel_directory = Path(plan.layout.wheels)
    for directory in (
        state,
        cache / "pip",
        cache / "xdg",
        cache / "torch-extensions",
        state / "home",
        state / "tmp",
        state / "bootstrap-wheels",
        state / "wheel-output",
        logs,
        wheel_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    export_runtime_source_snapshot(plan)
    environment = dict(plan.environment)
    for index, command in enumerate(plan.commands[:2]):
        _run_logged(command, index, logs, environment)
    bootstrap_wheels = _collect_bootstrap_wheels(plan)
    for index, command in enumerate(plan.commands[2:], start=2):
        _run_logged(command, index, logs, environment)

    runtime_wheels = _collect_runtime_wheels(plan)
    if observe_runtime_source(plan.source_path, plan.pins) != plan.source:
        raise RuntimeBuildError(
            "prepared source changed while building archived sources"
        )

    receipt = RuntimeBuildReceipt(
        schema_version=SCHEMA_VERSION,
        status="wheel_build_complete",
        build_id=plan.build_id,
        builder_sha256=plan.builder_sha256,
        source=plan.source,
        toolchain=plan.toolchain,
        layout=plan.layout,
        build_environment=plan.environment,
        commands=plan.commands,
        bootstrap_wheels=bootstrap_wheels,
        runtime_wheels=runtime_wheels,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    _write_receipt(receipt)
    return receipt


class _CliArguments(argparse.Namespace):
    ktransformers_source: Path
    output_root: Path
    host_profile: HostProfileName | None
    python: Path | None
    build: bool


def _parse_arguments() -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ktransformers-source",
        required=True,
        type=Path,
        help="Prepared KTransformers checkout at Exo's exact integrated revision",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/var/lib/exo/runtimes/glm47-sglang-kt"),
        help="Root for host/build-id isolated state, caches, wheels, and receipt",
    )
    parser.add_argument(
        "--host-profile",
        choices=tuple(HOST_BUILD_PROFILES),
        help="Defaults to the local short hostname; must match the actual host",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="CPython 3.12 executable; defaults to python3.12 from PATH",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Perform the build. Without this flag, run read-only preflight only.",
    )
    arguments = _CliArguments()
    parser.parse_args(namespace=arguments)
    return arguments


def _selected_profile(name: HostProfileName | None) -> HostBuildProfile:
    if name is not None:
        return HOST_BUILD_PROFILES[name]
    hostname = socket.gethostname().split(".", maxsplit=1)[0].lower()
    if hostname == "dwagon":
        return HOST_BUILD_PROFILES["dwagon"]
    if hostname == "fwuff":
        return HOST_BUILD_PROFILES["fwuff"]
    raise RuntimeBuildError(
        f"no GLM-4.7 runtime build profile exists for host {hostname}"
    )


def main() -> int:
    arguments = _parse_arguments()
    try:
        profile = _selected_profile(arguments.host_profile)
        plan = plan_runtime_build(
            arguments.ktransformers_source,
            arguments.output_root,
            profile,
            arguments.python,
        )
        receipt = execute_runtime_build(plan, build=arguments.build)
    except (OSError, RuntimeBuildError) as error:
        raise SystemExit(f"GLM-4.7 runtime build failed: {error}") from error
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
