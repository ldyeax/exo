#!/usr/bin/env python3
"""Install Exo's receipt-bound GLM-4.7 wheels into an overlay runtime.

The default action is a read-only preflight. Pass ``--install`` to create a
fresh virtual environment that inherits a known-good runtime through one
absolute ``.pth`` entry and shadows only the three pinned runtime wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from zipfile import BadZipFile, ZipFile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)

InstallStatus = Literal["preflight", "install_complete"]
CommandExecutor = Callable[
    [Sequence[str], int, Path, Mapping[str, str]],
    None,
]
BaseRuntimeObserver = Callable[[Path, Path], "BaseRuntimeObservation"]

SCHEMA_VERSION = 1
BUILD_RECEIPT_SCHEMA_VERSION = 2
EXPECTED_PYTHON_VERSION = (3, 12)
EXPECTED_PACKAGE_VERSION = "0.6.3.post1"
EXPECTED_DISTRIBUTIONS = frozenset(("ktransformers", "kt-kernel", "sglang-kt"))
PROVENANCE_PACKAGES = {
    "kt-kernel": "kt_kernel",
    "sglang-kt": "sglang",
}
BASE_RUNTIME_PTH_NAME = "exo_glm47_base_runtime.pth"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


class RuntimeInstallError(RuntimeError):
    """Raised when an overlay runtime input or installation is not exact."""


@dataclass(frozen=True)
class RuntimeWheel:
    distribution: str
    version: str
    filename: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RuntimeBuildObservation:
    receipt_path: str
    receipt_sha256: str
    build_id: str
    builder_sha256: str
    host_profile: str
    ktransformers_revision: str
    sglang_revision: str
    package_version: str
    wheels: tuple[RuntimeWheel, ...]


@dataclass(frozen=True)
class BaseRuntimeObservation:
    python_path: str
    resolved_python_path: str
    python_sha256: str
    python_version: str
    python_implementation: str
    python_soabi: str
    prefix: str
    base_prefix: str
    site_packages: str
    pip_version: str
    pip_freeze: tuple[str, ...]
    pip_freeze_sha256: str


@dataclass(frozen=True)
class RuntimeInstallLayout:
    output_root: str
    install_root: str
    venv: str
    python: str
    site_packages: str
    base_runtime_pth: str
    base_runtime_pth_sha256: str
    logs: str
    receipt: str


@dataclass(frozen=True)
class InstalledDistribution:
    distribution: str
    version: str
    metadata_path: str
    record_sha256: str


@dataclass(frozen=True)
class RuntimeInstallReceipt:
    schema_version: int
    status: InstallStatus
    install_id: str
    installer_sha256: str
    build: RuntimeBuildObservation
    base_runtime: BaseRuntimeObservation
    layout: RuntimeInstallLayout
    environment: tuple[tuple[str, str], ...]
    commands: tuple[tuple[str, ...], ...]
    installed_distributions: tuple[InstalledDistribution, ...]
    completed_at_utc: str | None


@dataclass(frozen=True)
class RuntimeInstallPlan:
    install_id: str
    installer_sha256: str
    build: RuntimeBuildObservation
    base_runtime: BaseRuntimeObservation
    layout: RuntimeInstallLayout
    environment: tuple[tuple[str, str], ...]
    commands: tuple[tuple[str, ...], ...]

    def preflight_receipt(self) -> RuntimeInstallReceipt:
        return RuntimeInstallReceipt(
            schema_version=SCHEMA_VERSION,
            status="preflight",
            install_id=self.install_id,
            installer_sha256=self.installer_sha256,
            build=self.build,
            base_runtime=self.base_runtime,
            layout=self.layout,
            environment=self.environment,
            commands=self.commands,
            installed_distributions=(),
            completed_at_utc=None,
        )


def _run(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        tuple(arguments),
        check=False,
        capture_output=True,
        text=True,
        env=None if environment is None else dict(environment),
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeInstallError(f"{' '.join(arguments)} failed: {detail}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeInstallError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _json_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeInstallError(f"{description} is not a JSON object")
    return cast(dict[str, object], value)


def _required_object(
    values: Mapping[str, object], key: str, description: str
) -> dict[str, object]:
    return _json_object(values.get(key), f"{description}.{key}")


def _required_list(
    values: Mapping[str, object], key: str, description: str
) -> list[object]:
    value = values.get(key)
    if not isinstance(value, list):
        raise RuntimeInstallError(f"{description}.{key} is not a JSON list")
    return cast(list[object], value)


def _required_string(values: Mapping[str, object], key: str, description: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeInstallError(f"{description}.{key} is not a nonempty string")
    return value


def _required_integer(values: Mapping[str, object], key: str, description: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeInstallError(f"{description}.{key} is not an integer")
    return value


def _require_sha256(value: str, description: str) -> str:
    normalized = value.lower()
    if HEX_SHA256.fullmatch(normalized) is None:
        raise RuntimeInstallError(f"{description} is not a SHA-256 digest")
    return normalized


def _require_absolute_path(path: Path, description: str) -> Path:
    if not path.is_absolute():
        raise RuntimeInstallError(f"{description} must be an absolute path: {path}")
    return path


def _resolve_existing(path: Path, description: str) -> Path:
    _require_absolute_path(path, description)
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RuntimeInstallError(
            f"cannot resolve {description} {path}: {error}"
        ) from error


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _embedded_provenance_contents() -> bytes:
    return (
        '"""Generated by Exo\'s pinned SGLang-KTransformers builder."""\n'
        "\n"
        "SCHEMA_VERSION = 1\n"
        f'KTRANSFORMERS_REVISION = "{GLM_4_7_FLASH_KTRANSFORMERS_REVISION}"\n'
        f'SGLANG_REVISION = "{GLM_4_7_FLASH_SGLANG_REVISION}"\n'
    ).encode("ascii")


def _require_under(path: Path, root: Path, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeInstallError(f"{description} escapes {root}: {path}") from error


def _inspect_wheel_metadata(
    wheel_path: Path,
    expected_distribution: str,
    expected_version: str,
) -> None:
    try:
        with ZipFile(wheel_path) as wheel:
            names = tuple(wheel.namelist())
            if len(names) != len(set(names)):
                raise RuntimeInstallError(f"wheel has duplicate members: {wheel_path}")
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise RuntimeInstallError(f"wheel has an unsafe member: {wheel_path}")
            metadata_paths = tuple(
                name
                for name in names
                if len(PurePosixPath(name).parts) == 2
                and PurePosixPath(name).parent.name.endswith(".dist-info")
                and PurePosixPath(name).name == "METADATA"
            )
            if len(metadata_paths) != 1:
                raise RuntimeInstallError(
                    f"wheel has ambiguous distribution metadata: {wheel_path}"
                )
            metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
            provenance_package = PROVENANCE_PACKAGES.get(expected_distribution)
            if provenance_package is not None:
                provenance_path = f"{provenance_package}/_exo_build_provenance.py"
                if provenance_path not in names:
                    raise RuntimeInstallError(
                        f"{expected_distribution} wheel lacks embedded build provenance"
                    )
                if wheel.read(provenance_path) != _embedded_provenance_contents():
                    raise RuntimeInstallError(
                        f"{expected_distribution} wheel has unexpected build provenance"
                    )
    except (BadZipFile, OSError) as error:
        raise RuntimeInstallError(
            f"cannot inspect wheel {wheel_path}: {error}"
        ) from error
    distribution = str(metadata["Name"] or "")
    version = str(metadata["Version"] or "")
    if _normalize_distribution(distribution) != expected_distribution:
        raise RuntimeInstallError(
            f"wheel {wheel_path} is {distribution}, expected {expected_distribution}"
        )
    if version != expected_version:
        raise RuntimeInstallError(
            f"wheel {wheel_path} version is {version}, expected {expected_version}"
        )


def _observe_wheel(
    value: object,
    build_root: Path,
    expected_package_version: str,
) -> RuntimeWheel:
    artifact = _json_object(value, "runtime_wheels item")
    distribution = _normalize_distribution(
        _required_string(artifact, "distribution", "runtime wheel")
    )
    if distribution not in EXPECTED_DISTRIBUTIONS:
        raise RuntimeInstallError(
            f"unexpected runtime wheel distribution: {distribution}"
        )
    version = _required_string(artifact, "version", "runtime wheel")
    if version != expected_package_version:
        raise RuntimeInstallError(
            f"{distribution} wheel version is {version}, expected "
            f"{expected_package_version}"
        )
    filename = _required_string(artifact, "filename", "runtime wheel")
    relative_text = _required_string(artifact, "path", "runtime wheel")
    relative_path = PurePosixPath(relative_text)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or len(relative_path.parts) != 2
        or relative_path.parts[0] != "wheels"
        or relative_path.name != filename
        or not filename.endswith(".whl")
    ):
        raise RuntimeInstallError(f"unsafe runtime wheel receipt path: {relative_text}")
    wheel_path = _resolve_existing(
        build_root / Path(*relative_path.parts),
        f"{distribution} runtime wheel",
    )
    _require_under(wheel_path, build_root, f"{distribution} runtime wheel")
    if not wheel_path.is_file():
        raise RuntimeInstallError(f"runtime wheel is not a file: {wheel_path}")
    size_bytes = _required_integer(artifact, "size_bytes", "runtime wheel")
    if wheel_path.stat().st_size != size_bytes:
        raise RuntimeInstallError(
            f"{distribution} wheel size does not match build receipt"
        )
    expected_sha256 = _require_sha256(
        _required_string(artifact, "sha256", "runtime wheel"),
        f"{distribution} wheel SHA-256",
    )
    observed_sha256 = _sha256(wheel_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeInstallError(
            f"{distribution} wheel SHA-256 does not match build receipt"
        )
    _inspect_wheel_metadata(wheel_path, distribution, version)
    return RuntimeWheel(
        distribution=distribution,
        version=version,
        filename=filename,
        path=str(wheel_path),
        size_bytes=size_bytes,
        sha256=observed_sha256,
    )


def observe_runtime_build(build_receipt: Path) -> RuntimeBuildObservation:
    receipt_path = _resolve_existing(build_receipt, "build receipt")
    if not receipt_path.is_file():
        raise RuntimeInstallError(f"build receipt is not a file: {receipt_path}")
    try:
        raw_receipt = receipt_path.read_bytes()
        document = _json_object(cast(object, json.loads(raw_receipt)), "build receipt")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise RuntimeInstallError(
            f"cannot read build receipt {receipt_path}: {error}"
        ) from error
    if document.get("schema_version") != BUILD_RECEIPT_SCHEMA_VERSION:
        raise RuntimeInstallError(
            f"build receipt schema is {document.get('schema_version')}, expected "
            f"{BUILD_RECEIPT_SCHEMA_VERSION}"
        )
    if document.get("status") != "wheel_build_complete":
        raise RuntimeInstallError("build receipt is not a completed wheel build")

    build_id = _require_sha256(
        _required_string(document, "build_id", "build receipt"),
        "build ID",
    )
    builder_sha256 = _require_sha256(
        _required_string(document, "builder_sha256", "build receipt"),
        "builder SHA-256",
    )
    builder_path = (
        Path(__file__).with_name("build_sglang_kt_runtime.py").resolve(strict=True)
    )
    current_builder_sha256 = _sha256(builder_path)
    if builder_sha256 != current_builder_sha256:
        raise RuntimeInstallError(
            f"build receipt builder SHA-256 is {builder_sha256}, expected current "
            f"builder {current_builder_sha256}"
        )
    source = _required_object(document, "source", "build receipt")
    ktransformers_revision = _required_string(
        source, "ktransformers_revision", "build receipt.source"
    ).lower()
    sglang_revision = _required_string(
        source, "sglang_revision", "build receipt.source"
    ).lower()
    package_version = _required_string(
        source, "package_version", "build receipt.source"
    )
    if ktransformers_revision != GLM_4_7_FLASH_KTRANSFORMERS_REVISION:
        raise RuntimeInstallError(
            f"build receipt KTransformers revision is {ktransformers_revision}, "
            f"expected {GLM_4_7_FLASH_KTRANSFORMERS_REVISION}"
        )
    if sglang_revision != GLM_4_7_FLASH_SGLANG_REVISION:
        raise RuntimeInstallError(
            f"build receipt SGLang revision is {sglang_revision}, expected "
            f"{GLM_4_7_FLASH_SGLANG_REVISION}"
        )
    if package_version != EXPECTED_PACKAGE_VERSION:
        raise RuntimeInstallError(
            f"build receipt package version is {package_version}, expected "
            f"{EXPECTED_PACKAGE_VERSION}"
        )

    toolchain = _required_object(document, "toolchain", "build receipt")
    host_profile = _required_string(
        toolchain, "host_profile", "build receipt.toolchain"
    )
    if host_profile not in {"dwagon", "fwuff"}:
        raise RuntimeInstallError(f"unsupported build host profile: {host_profile}")
    layout = _required_object(document, "layout", "build receipt")
    build_root = _resolve_existing(
        Path(_required_string(layout, "build_root", "build receipt.layout")),
        "build root",
    )
    recorded_receipt = _resolve_existing(
        Path(_required_string(layout, "receipt", "build receipt.layout")),
        "recorded build receipt",
    )
    if recorded_receipt != receipt_path:
        raise RuntimeInstallError(
            f"build receipt records a different receipt path: {recorded_receipt}"
        )
    if receipt_path.parent != build_root or build_root.name != build_id:
        raise RuntimeInstallError("build receipt layout is not rooted at its build ID")

    wheel_values = _required_list(document, "runtime_wheels", "build receipt")
    wheels = tuple(
        sorted(
            (
                _observe_wheel(value, build_root, package_version)
                for value in wheel_values
            ),
            key=lambda wheel: wheel.distribution,
        )
    )
    distributions = tuple(wheel.distribution for wheel in wheels)
    if len(wheels) != len(EXPECTED_DISTRIBUTIONS) or set(distributions) != set(
        EXPECTED_DISTRIBUTIONS
    ):
        raise RuntimeInstallError(
            "build receipt runtime wheels must contain exactly one each of: "
            + ", ".join(sorted(EXPECTED_DISTRIBUTIONS))
        )
    if len({wheel.path for wheel in wheels}) != len(wheels):
        raise RuntimeInstallError("build receipt reuses a runtime wheel path")

    return RuntimeBuildObservation(
        receipt_path=str(receipt_path),
        receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),
        build_id=build_id,
        builder_sha256=builder_sha256,
        host_profile=host_profile,
        ktransformers_revision=ktransformers_revision,
        sglang_revision=sglang_revision,
        package_version=package_version,
        wheels=wheels,
    )


def _probe_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _canonical_freeze(stdout: str) -> tuple[tuple[str, ...], str]:
    lines = tuple(line.strip() for line in stdout.splitlines() if line.strip())
    if not lines:
        raise RuntimeInstallError("base runtime pip freeze is empty")
    if len(lines) != len(set(lines)):
        raise RuntimeInstallError("base runtime pip freeze contains duplicate lines")
    canonical_text = "\n".join(lines) + "\n"
    return lines, _text_sha256(canonical_text)


def observe_base_runtime(
    base_python: Path,
    base_site_packages: Path,
) -> BaseRuntimeObservation:
    _require_absolute_path(base_python, "base Python")
    if not base_python.is_file() or not os.access(base_python, os.X_OK):
        raise RuntimeInstallError(
            f"base Python is not an executable file: {base_python}"
        )
    resolved_python = _resolve_existing(base_python, "base Python")
    site_packages = _resolve_existing(base_site_packages, "base site-packages")
    if not site_packages.is_dir():
        raise RuntimeInstallError(
            f"base site-packages is not a directory: {site_packages}"
        )

    probe_code = (
        "import json,os,pip,platform,sys,sysconfig;"
        "print(json.dumps({"
        "'executable':sys.executable,'version':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'soabi':sysconfig.get_config_var('SOABI'),'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'purelib':sysconfig.get_path('purelib'),"
        "'pip_version':pip.__version__},sort_keys=True))"
    )
    probe = _run(
        (str(base_python), "-I", "-c", probe_code),
        environment=_probe_environment(),
    )
    try:
        facts = _json_object(
            cast(object, json.loads(probe.stdout)), "base Python probe"
        )
    except json.JSONDecodeError as error:
        raise RuntimeInstallError(
            f"base Python returned invalid JSON: {error}"
        ) from error
    executable = _resolve_existing(
        Path(_required_string(facts, "executable", "base Python probe")),
        "probed base Python",
    )
    if executable != resolved_python:
        raise RuntimeInstallError(
            f"base Python probe resolved {executable}, expected {resolved_python}"
        )
    version = _required_string(facts, "version", "base Python probe")
    implementation = _required_string(facts, "implementation", "base Python probe")
    soabi = _required_string(facts, "soabi", "base Python probe")
    try:
        version_components = tuple(int(item) for item in version.split("."))
    except ValueError as error:
        raise RuntimeInstallError(f"invalid base Python version: {version}") from error
    if version_components[:2] != EXPECTED_PYTHON_VERSION:
        raise RuntimeInstallError(
            f"overlay runtime requires Python 3.12, observed {version}"
        )
    if implementation != "CPython" or not soabi.startswith("cpython-312"):
        raise RuntimeInstallError(
            f"overlay runtime requires CPython 3.12 ABI, observed "
            f"{implementation}/{soabi}"
        )
    prefix = _resolve_existing(
        Path(_required_string(facts, "prefix", "base Python probe")),
        "base runtime prefix",
    )
    base_prefix = _resolve_existing(
        Path(_required_string(facts, "base_prefix", "base Python probe")),
        "base interpreter prefix",
    )
    if prefix == base_prefix:
        raise RuntimeInstallError("base Python is not inside an isolated runtime")
    observed_site_packages = _resolve_existing(
        Path(_required_string(facts, "purelib", "base Python probe")),
        "probed base site-packages",
    )
    if observed_site_packages != site_packages:
        raise RuntimeInstallError(
            f"base Python uses {observed_site_packages}, not {site_packages}"
        )
    pip_version = _required_string(facts, "pip_version", "base Python probe")

    freeze_result = _run(
        (str(base_python), "-I", "-m", "pip", "freeze", "--all"),
        environment=_probe_environment(),
    )
    pip_freeze, pip_freeze_sha256 = _canonical_freeze(freeze_result.stdout)
    if f"pip=={pip_version}" not in pip_freeze:
        raise RuntimeInstallError(
            "base runtime pip freeze --all does not bind the imported pip version"
        )
    return BaseRuntimeObservation(
        python_path=str(base_python),
        resolved_python_path=str(resolved_python),
        python_sha256=_sha256(resolved_python),
        python_version=version,
        python_implementation=implementation,
        python_soabi=soabi,
        prefix=str(prefix),
        base_prefix=str(base_prefix),
        site_packages=str(site_packages),
        pip_version=pip_version,
        pip_freeze=pip_freeze,
        pip_freeze_sha256=pip_freeze_sha256,
    )


def _require_base_observation(
    observation: BaseRuntimeObservation,
    base_python: Path,
    base_site_packages: Path,
) -> None:
    if observation.python_path != str(base_python):
        raise RuntimeInstallError(
            "base runtime observation has a different Python path"
        )
    if Path(observation.site_packages) != base_site_packages:
        raise RuntimeInstallError(
            "base runtime observation has a different site-packages path"
        )
    if (
        observation.python_implementation != "CPython"
        or not observation.python_soabi.startswith("cpython-312")
    ):
        raise RuntimeInstallError("base runtime observation is not CPython 3.12")
    try:
        version = tuple(int(item) for item in observation.python_version.split("."))
    except ValueError as error:
        raise RuntimeInstallError(
            f"invalid observed base Python version: {observation.python_version}"
        ) from error
    if version[:2] != EXPECTED_PYTHON_VERSION:
        raise RuntimeInstallError("base runtime observation is not Python 3.12")
    _require_sha256(observation.python_sha256, "base Python SHA-256")
    _require_sha256(observation.pip_freeze_sha256, "base pip-freeze SHA-256")
    canonical_freeze = "\n".join(observation.pip_freeze) + "\n"
    if _text_sha256(canonical_freeze) != observation.pip_freeze_sha256:
        raise RuntimeInstallError("base pip-freeze content does not match its SHA-256")


def _install_id_payload(
    build: RuntimeBuildObservation,
    base_runtime: BaseRuntimeObservation,
    installer_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "installer_sha256": installer_sha256,
        "build_receipt_sha256": build.receipt_sha256,
        "build_id": build.build_id,
        "base_python_path": base_runtime.python_path,
        "base_python_sha256": base_runtime.python_sha256,
        "base_site_packages": base_runtime.site_packages,
        "base_pip_freeze_sha256": base_runtime.pip_freeze_sha256,
        "wheels": [
            {
                "distribution": wheel.distribution,
                "version": wheel.version,
                "sha256": wheel.sha256,
            }
            for wheel in build.wheels
        ],
    }


def _make_layout(
    output_root: Path,
    host_profile: str,
    install_id: str,
    base_site_packages: Path,
) -> RuntimeInstallLayout:
    install_root = output_root / host_profile / install_id
    venv = install_root / "venv"
    site_packages = venv / "lib/python3.12/site-packages"
    pth_path = site_packages / BASE_RUNTIME_PTH_NAME
    pth_contents = f"{base_site_packages}\n"
    return RuntimeInstallLayout(
        output_root=str(output_root),
        install_root=str(install_root),
        venv=str(venv),
        python=str(venv / "bin/python"),
        site_packages=str(site_packages),
        base_runtime_pth=str(pth_path),
        base_runtime_pth_sha256=_text_sha256(pth_contents),
        logs=str(install_root / "logs"),
        receipt=str(install_root / "install-receipt.json"),
    )


def _make_commands(
    base_runtime: BaseRuntimeObservation,
    build: RuntimeBuildObservation,
    layout: RuntimeInstallLayout,
) -> tuple[tuple[str, ...], ...]:
    venv_command = (
        base_runtime.python_path,
        "-I",
        "-m",
        "venv",
        "--without-pip",
        layout.venv,
    )
    install_command = (
        layout.python,
        "-I",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--ignore-installed",
        "--no-compile",
        "--no-warn-script-location",
        "--target",
        layout.site_packages,
        *(wheel.path for wheel in build.wheels),
    )
    return (venv_command, install_command)


def _make_environment(layout: RuntimeInstallLayout) -> tuple[tuple[str, str], ...]:
    values = {
        "HOME": str(Path(layout.install_root) / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_CACHE_DIR": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    return tuple(sorted(values.items()))


def plan_runtime_install(
    build_receipt: Path,
    base_python: Path,
    base_site_packages: Path,
    output_root: Path,
    *,
    base_runtime: BaseRuntimeObservation | None = None,
) -> RuntimeInstallPlan:
    _require_absolute_path(build_receipt, "build receipt")
    _require_absolute_path(base_python, "base Python")
    _require_absolute_path(base_site_packages, "base site-packages")
    _require_absolute_path(output_root, "output root")
    build = observe_runtime_build(build_receipt)
    site_packages = _resolve_existing(base_site_packages, "base site-packages")
    selected_base = base_runtime or observe_base_runtime(base_python, site_packages)
    _require_base_observation(selected_base, base_python, site_packages)
    try:
        str(site_packages).encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeInstallError(
            "base site-packages must be ASCII for a literal .pth entry"
        ) from error
    if "\n" in str(site_packages) or "\r" in str(site_packages):
        raise RuntimeInstallError(
            "base site-packages cannot contain a newline in a .pth entry"
        )
    installer_path = Path(__file__).resolve(strict=True)
    installer_sha256 = _sha256(installer_path)
    install_id = hashlib.sha256(
        _canonical_json(_install_id_payload(build, selected_base, installer_sha256))
    ).hexdigest()
    resolved_output_root = output_root.resolve(strict=False)
    layout = _make_layout(
        resolved_output_root,
        build.host_profile,
        install_id,
        site_packages,
    )
    install_root = Path(layout.install_root)
    base_prefix = Path(selected_base.prefix)
    for protected_path, description in (
        (site_packages, "base site-packages"),
        (base_prefix, "base runtime prefix"),
        (Path(selected_base.python_path), "base Python"),
    ):
        if protected_path == install_root or protected_path.is_relative_to(
            install_root
        ):
            raise RuntimeInstallError(
                f"install root would contain {description}: {install_root}"
            )
        if install_root.is_relative_to(protected_path):
            raise RuntimeInstallError(
                f"install root would be inside {description}: {install_root}"
            )
    commands = _make_commands(selected_base, build, layout)
    return RuntimeInstallPlan(
        install_id=install_id,
        installer_sha256=installer_sha256,
        build=build,
        base_runtime=selected_base,
        layout=layout,
        environment=_make_environment(layout),
        commands=commands,
    )


def _run_logged(
    arguments: Sequence[str],
    index: int,
    logs: Path,
    environment: Mapping[str, str],
) -> None:
    stdout_path = logs / f"{index:02d}.stdout.log"
    stderr_path = logs / f"{index:02d}.stderr.log"
    with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
        result = subprocess.run(
            tuple(arguments),
            check=False,
            stdout=stdout,
            stderr=stderr,
            text=True,
            env={**os.environ, **environment},
        )
    if result.returncode != 0:
        raise RuntimeInstallError(
            f"install command {index} failed with exit {result.returncode}; "
            f"logs: {stdout_path}, {stderr_path}"
        )


def _write_base_runtime_pth(plan: RuntimeInstallPlan) -> None:
    site_packages = Path(plan.layout.site_packages)
    if not site_packages.is_dir():
        raise RuntimeInstallError(
            f"venv did not create expected site-packages: {site_packages}"
        )
    pth_path = Path(plan.layout.base_runtime_pth)
    contents = f"{plan.base_runtime.site_packages}\n"
    if _text_sha256(contents) != plan.layout.base_runtime_pth_sha256:
        raise RuntimeInstallError("planned base-runtime .pth digest is inconsistent")
    try:
        with pth_path.open("x", encoding="ascii", errors="strict") as destination:
            destination.write(contents)
    except OSError as error:
        raise RuntimeInstallError(f"cannot write base-runtime .pth: {error}") from error


def _observe_installed_distributions(
    plan: RuntimeInstallPlan,
) -> tuple[InstalledDistribution, ...]:
    site_packages = Path(plan.layout.site_packages)
    pth_path = Path(plan.layout.base_runtime_pth)
    expected_pth = f"{plan.base_runtime.site_packages}\n"
    try:
        observed_pth = pth_path.read_text(encoding="ascii", errors="strict")
    except (OSError, UnicodeError) as error:
        raise RuntimeInstallError(
            f"cannot verify base-runtime .pth: {error}"
        ) from error
    if observed_pth != expected_pth or _text_sha256(observed_pth) != (
        plan.layout.base_runtime_pth_sha256
    ):
        raise RuntimeInstallError("base-runtime .pth changed during installation")

    expected_versions = {
        wheel.distribution: wheel.version for wheel in plan.build.wheels
    }
    installed: list[InstalledDistribution] = []
    for metadata_path in sorted(site_packages.glob("*.dist-info/METADATA")):
        resolved_metadata = _resolve_existing(metadata_path, "installed metadata")
        _require_under(resolved_metadata, site_packages, "installed metadata")
        try:
            metadata = BytesParser().parsebytes(resolved_metadata.read_bytes())
        except OSError as error:
            raise RuntimeInstallError(
                f"cannot read installed metadata {resolved_metadata}: {error}"
            ) from error
        distribution = _normalize_distribution(str(metadata["Name"] or ""))
        version = str(metadata["Version"] or "")
        expected_version = expected_versions.get(distribution)
        if expected_version is None or version != expected_version:
            raise RuntimeInstallError(
                f"unexpected installed distribution: {distribution}=={version}"
            )
        record_path = resolved_metadata.parent / "RECORD"
        if not record_path.is_file():
            raise RuntimeInstallError(
                f"installed distribution has no RECORD: {distribution}"
            )
        installed.append(
            InstalledDistribution(
                distribution=distribution,
                version=version,
                metadata_path=str(resolved_metadata),
                record_sha256=_sha256(record_path),
            )
        )
    installed.sort(key=lambda item: item.distribution)
    distributions = tuple(item.distribution for item in installed)
    if len(installed) != len(EXPECTED_DISTRIBUTIONS) or set(distributions) != set(
        EXPECTED_DISTRIBUTIONS
    ):
        raise RuntimeInstallError(
            "overlay must contain exactly the three receipt-bound distributions"
        )
    return tuple(installed)


def _write_receipt(receipt: RuntimeInstallReceipt) -> None:
    destination = Path(receipt.layout.receipt)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
            encoding="ascii",
            errors="strict",
        )
        os.replace(temporary, destination)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeInstallError(f"cannot write install receipt: {error}") from error


def execute_runtime_install(
    plan: RuntimeInstallPlan,
    *,
    install: bool = False,
    base_observer: BaseRuntimeObserver = observe_base_runtime,
    command_executor: CommandExecutor = _run_logged,
) -> RuntimeInstallReceipt:
    if not install:
        return plan.preflight_receipt()

    current_build = observe_runtime_build(Path(plan.build.receipt_path))
    current_base = base_observer(
        Path(plan.base_runtime.python_path),
        Path(plan.base_runtime.site_packages),
    )
    if current_build != plan.build:
        raise RuntimeInstallError(
            "build receipt or runtime wheels changed after preflight"
        )
    if current_base != plan.base_runtime:
        raise RuntimeInstallError("base runtime changed after preflight")
    if _sha256(Path(__file__).resolve(strict=True)) != plan.installer_sha256:
        raise RuntimeInstallError("installer changed after preflight")

    install_root = Path(plan.layout.install_root)
    output_root = Path(plan.layout.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    install_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        install_root.mkdir()
    except FileExistsError as error:
        raise RuntimeInstallError(
            f"install root already exists; refusing to reuse it: {install_root}"
        ) from error
    logs = Path(plan.layout.logs)
    logs.mkdir()
    (install_root / "home").mkdir()
    environment = dict(plan.environment)

    command_executor(plan.commands[0], 0, logs, environment)
    _write_base_runtime_pth(plan)
    command_executor(plan.commands[1], 1, logs, environment)
    installed = _observe_installed_distributions(plan)

    if observe_runtime_build(Path(plan.build.receipt_path)) != plan.build:
        raise RuntimeInstallError("build inputs changed during installation")
    if (
        base_observer(
            Path(plan.base_runtime.python_path),
            Path(plan.base_runtime.site_packages),
        )
        != plan.base_runtime
    ):
        raise RuntimeInstallError("base runtime changed during installation")
    if _sha256(Path(__file__).resolve(strict=True)) != plan.installer_sha256:
        raise RuntimeInstallError("installer changed during installation")

    receipt = RuntimeInstallReceipt(
        schema_version=SCHEMA_VERSION,
        status="install_complete",
        install_id=plan.install_id,
        installer_sha256=plan.installer_sha256,
        build=plan.build,
        base_runtime=plan.base_runtime,
        layout=plan.layout,
        environment=plan.environment,
        commands=plan.commands,
        installed_distributions=installed,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    _write_receipt(receipt)
    return receipt


class _CliArguments(argparse.Namespace):
    build_receipt: Path
    base_python: Path
    base_site_packages: Path
    output_root: Path
    install: bool


def _parse_arguments() -> _CliArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-receipt",
        required=True,
        type=Path,
        help="Absolute path to a completed schema-2 runtime build receipt",
    )
    parser.add_argument(
        "--base-python",
        required=True,
        type=Path,
        help="Absolute path to the known-good CPython 3.12 runtime executable",
    )
    parser.add_argument(
        "--base-site-packages",
        required=True,
        type=Path,
        help="Absolute site-packages path reported by the base Python",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/var/lib/exo/runtimes/glm47-sglang-kt-overlay"),
        help="Absolute root for host/install-id isolated overlay runtimes",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Create the overlay. Without this flag, preflight is read-only.",
    )
    arguments = _CliArguments()
    parser.parse_args(namespace=arguments)
    return arguments


def main() -> int:
    arguments = _parse_arguments()
    try:
        plan = plan_runtime_install(
            arguments.build_receipt,
            arguments.base_python,
            arguments.base_site_packages,
            arguments.output_root,
        )
        receipt = execute_runtime_install(plan, install=arguments.install)
    except (OSError, RuntimeInstallError) as error:
        raise SystemExit(f"GLM-4.7 runtime install failed: {error}") from error
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
