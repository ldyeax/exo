from __future__ import annotations

import hashlib
import os
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.build_sglang_kt_runtime as builder
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from scripts.build_sglang_kt_runtime import (
    BOOTSTRAP_WHEEL_PINS,
    CUDA_ARCHITECTURES,
    EMBEDDED_PROVENANCE_MODULE,
    HOST_BUILD_PROFILES,
    HostProfileName,
    RuntimeBuildError,
    RuntimeSourcePins,
    RuntimeToolchainObservation,
    SubmoduleObservation,
    ToolObservation,
    _embedded_provenance_contents,
    execute_runtime_build,
    export_runtime_source_snapshot,
    inspect_wheel,
    observe_runtime_source,
    plan_runtime_build,
)

GIT_NAME = "Exo Runtime Build Test"
GIT_EMAIL = "exo-runtime-build-test@example.invalid"
TIMESTAMP = "2026-01-01T00:00:00+00:00"


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def initialize_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(
        ("git", "init", "--initial-branch=main", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    run_git(path, "config", "user.name", GIT_NAME)
    run_git(path, "config", "user.email", GIT_EMAIL)


def commit_all(repository: Path, message: str) -> str:
    run_git(repository, "add", "--all")
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": TIMESTAMP,
        "GIT_COMMITTER_DATE": TIMESTAMP,
    }
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-m", message),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return run_git(repository, "rev-parse", "HEAD")


def make_runtime_source(tmp_path: Path) -> tuple[Path, RuntimeSourcePins]:
    sglang = tmp_path / "sglang"
    initialize_repository(sglang)
    (sglang / "python").mkdir()
    (sglang / "python/sglang").mkdir()
    (sglang / "python/sglang/__init__.py").write_text("")
    (sglang / "python/pyproject.toml").write_text("[build-system]\n")
    (sglang / "python/setup.py").write_text("from setuptools import setup\n")
    (sglang / ".gitignore").write_text("python/build/\n")
    sglang_revision = commit_all(sglang, "test: SGLang source")

    build_submodules: dict[str, Path] = {}
    for name in ("llama.cpp", "pybind11"):
        dependency = tmp_path / name
        initialize_repository(dependency)
        (dependency / "dependency.txt").write_text(f"exact {name}\n")
        commit_all(dependency, f"test: {name} source")
        build_submodules[name] = dependency

    unused_dependency = tmp_path / "custom-flashinfer"
    initialize_repository(unused_dependency)
    (unused_dependency / "unused.txt").write_text("unused dependency\n")
    commit_all(unused_dependency, "test: unused dependency source")

    ktransformers = tmp_path / "ktransformers"
    initialize_repository(ktransformers)
    (ktransformers / "pyproject.toml").write_text("[build-system]\n")
    (ktransformers / "setup.py").write_text("from setuptools import setup\n")
    (ktransformers / "version.py").write_text('__version__ = "0.6.3.post1"\n')
    (ktransformers / "kt-kernel").mkdir()
    (ktransformers / "kt-kernel/python").mkdir()
    (ktransformers / "kt-kernel/python/__init__.py").write_text("")
    (ktransformers / "kt-kernel/pyproject.toml").write_text("[build-system]\n")
    (ktransformers / "kt-kernel/setup.py").write_text("from setuptools import setup\n")
    (ktransformers / "kt-kernel/CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
    )
    run_git(
        ktransformers,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(sglang),
        "third_party/sglang",
    )
    for name, dependency in build_submodules.items():
        run_git(
            ktransformers,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(dependency),
            f"third_party/{name}",
        )
    run_git(
        ktransformers,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(unused_dependency),
        "third_party/custom_flashinfer",
    )
    ktransformers_revision = commit_all(ktransformers, "test: runtime source")
    return ktransformers, RuntimeSourcePins(
        ktransformers_revision=ktransformers_revision,
        sglang_revision=sglang_revision,
        sglang_submodule_path=Path("third_party/sglang"),
    )


def fake_toolchain(
    tmp_path: Path, host: HostProfileName = "dwagon"
) -> RuntimeToolchainObservation:
    tools: list[ToolObservation] = []
    for name in (
        "python",
        "nvcc",
        "cmake",
        "ninja",
        "cc",
        "cxx",
        "cudart_static",
    ):
        path = tmp_path / "tools" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name)
        tools.append(
            ToolObservation(
                name=name,
                path=str(path),
                sha256=hashlib.sha256(name.encode()).hexdigest(),
                version=f"{name} test version",
            )
        )
    return RuntimeToolchainObservation(
        host_profile=host,
        hostname=host,
        operating_system="Linux",
        machine="x86_64",
        python_version="3.12.10",
        python_implementation="CPython",
        python_soabi="cpython-312-x86_64-linux-gnu",
        cuda_root=f"/test/{host}/cuda-13.1",
        cuda_version="13.1.80",
        cuda_version_manifest_sha256="a" * 64,
        cuda_architectures=CUDA_ARCHITECTURES,
        cpu_features=(
            "amx_bf16",
            "amx_int8",
            "amx_tile",
            "avx512_bf16",
            "avx512_vnni",
            "avx512f",
            "avx512vbmi",
        ),
        tools=tuple(tools),
    )


def test_admitted_pins_match_launch_spec() -> None:
    pins = RuntimeSourcePins.admitted()

    assert pins.ktransformers_revision == GLM_4_7_FLASH_KTRANSFORMERS_REVISION
    assert pins.sglang_revision == GLM_4_7_FLASH_SGLANG_REVISION


def test_observes_exact_clean_recursive_source(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)

    observation = observe_runtime_source(source, pins)

    assert observation.ktransformers_revision == pins.ktransformers_revision
    assert observation.sglang_revision == pins.sglang_revision
    assert observation.package_version == "0.6.3.post1"
    assert observation.submodules == (
        SubmoduleObservation(
            path="third_party/llama.cpp",
            revision=run_git(source / "third_party/llama.cpp", "rev-parse", "HEAD"),
        ),
        SubmoduleObservation(
            path="third_party/pybind11",
            revision=run_git(source / "third_party/pybind11", "rev-parse", "HEAD"),
        ),
        SubmoduleObservation(
            path="third_party/sglang",
            revision=pins.sglang_revision,
        ),
    )


def test_observes_independently_pinned_sglang_after_gitlink(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    sglang = source / pins.sglang_submodule_path
    (sglang / "tp_qk_norm.py").write_text("SHARDED = True\n")
    final_sglang_revision = commit_all(sglang, "fix: shard TP QK norm")

    observation = observe_runtime_source(
        source,
        replace(pins, sglang_revision=final_sglang_revision),
    )

    assert observation.sglang_revision == final_sglang_revision
    assert run_git(source, "status", "--porcelain=v1") == "M third_party/sglang"


def test_rejects_dirty_parent_source(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    (source / "untracked.txt").write_text("dirty\n")

    with pytest.raises(RuntimeBuildError, match="KTransformers source is not clean"):
        observe_runtime_source(source, pins)


def test_rejects_dirty_submodule_source(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    dependency = source / "third_party/pybind11"
    (dependency / "dependency.txt").write_text("dirty\n")

    with pytest.raises(RuntimeBuildError, match="not clean"):
        observe_runtime_source(source, pins)


def test_ignores_unmaterialized_non_build_submodule(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    unused_path = "third_party/custom_flashinfer"
    run_git(source, "submodule", "deinit", "--force", unused_path)

    observation = observe_runtime_source(source, pins)

    assert unused_path not in {submodule.path for submodule in observation.submodules}


def test_rejects_wrong_sglang_revision(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    sglang = source / "third_party/sglang"
    (sglang / "new.py").write_text("NEW = True\n")
    commit_all(sglang, "test: wrong SGLang revision")
    wrong_parent_revision = commit_all(source, "test: pin wrong SGLang revision")

    with pytest.raises(RuntimeBuildError, match="SGLang revision is"):
        observe_runtime_source(
            source,
            replace(pins, ktransformers_revision=wrong_parent_revision),
        )


def test_preflight_is_deterministic_and_does_not_create_output(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    output = tmp_path / "runtime-output"
    toolchain = fake_toolchain(tmp_path)

    first_plan = plan_runtime_build(
        source,
        output,
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )
    second_plan = plan_runtime_build(
        source,
        output,
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )
    receipt = execute_runtime_build(first_plan)

    assert first_plan.build_id == second_plan.build_id
    assert receipt.status == "preflight"
    assert receipt.builder_sha256 == first_plan.builder_sha256
    assert len(receipt.builder_sha256) == 64
    assert receipt.runtime_wheels == ()
    assert receipt.completed_at_utc is None
    assert not output.exists()
    assert [command[2:4] for command in receipt.commands[:3]] == [
        ("venv", receipt.layout.state + "/venv"),
        ("pip", "download"),
        ("pip", "install"),
    ]
    assert all(
        "install.sh" not in argument
        for command in receipt.commands
        for argument in command
    )


def test_toolchain_change_changes_build_id(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    toolchain = fake_toolchain(tmp_path)

    first = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )
    second = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=replace(toolchain, cuda_version_manifest_sha256="b" * 64),
    )

    assert first.build_id != second.build_id


def test_bootstrap_wheel_digest_change_changes_build_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, pins = make_runtime_source(tmp_path)
    toolchain = fake_toolchain(tmp_path)
    first = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )
    monkeypatch.setattr(
        builder,
        "BOOTSTRAP_WHEEL_PINS",
        (replace(BOOTSTRAP_WHEEL_PINS[0], sha256="f" * 64), *BOOTSTRAP_WHEEL_PINS[1:]),
    )

    second = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )

    assert first.build_id != second.build_id


def test_bootstrap_wheels_are_verified_before_install_or_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, pins = make_runtime_source(tmp_path)
    toolchain = fake_toolchain(tmp_path)
    plan = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=toolchain,
    )
    events: list[str] = []
    monkeypatch.setattr(
        builder,
        "observe_runtime_source",
        lambda _source_path, _pins: plan.source,
    )
    monkeypatch.setattr(
        builder,
        "observe_runtime_toolchain",
        lambda _profile, _python: (plan.python_executable, plan.toolchain),
    )
    monkeypatch.setattr(
        builder,
        "export_runtime_source_snapshot",
        lambda _plan: Path(plan.layout.state) / "source/ktransformers",
    )
    monkeypatch.setattr(
        builder,
        "_run_logged",
        lambda _command, index, _logs, _environment: events.append(f"run:{index}"),
    )
    monkeypatch.setattr(
        builder,
        "_collect_bootstrap_wheels",
        lambda _plan: events.append("verify-bootstrap") or (),
    )
    monkeypatch.setattr(
        builder,
        "_collect_runtime_wheels",
        lambda _plan: events.append("collect-runtime") or (),
    )

    receipt = execute_runtime_build(plan, build=True)

    assert receipt.status == "wheel_build_complete"
    assert events[:4] == ["run:0", "run:1", "verify-bootstrap", "run:2"]
    assert events[-1] == "collect-runtime"


def test_rejects_output_inside_git_worktree(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)

    with pytest.raises(RuntimeBuildError, match="inside Git worktree"):
        plan_runtime_build(
            source,
            source / "runtime-output",
            HOST_BUILD_PROFILES["dwagon"],
            pins=pins,
            toolchain=fake_toolchain(tmp_path),
        )


def test_source_export_excludes_ignored_build_output(tmp_path: Path) -> None:
    source, pins = make_runtime_source(tmp_path)
    contamination = (
        source / "third_party/sglang/python/build/lib/sglang/contaminated_runtime.py"
    )
    contamination.parent.mkdir(parents=True)
    contamination.write_text("CONTAMINATED = True\n")
    plan = plan_runtime_build(
        source,
        tmp_path / "output",
        HOST_BUILD_PROFILES["dwagon"],
        pins=pins,
        toolchain=fake_toolchain(tmp_path),
    )

    snapshot = export_runtime_source_snapshot(plan)

    assert not (
        snapshot / "third_party/sglang/python/build/lib/sglang/contaminated_runtime.py"
    ).exists()
    expected_provenance = _embedded_provenance_contents(plan.source)
    assert (
        snapshot / "third_party/sglang/python/sglang" / EMBEDDED_PROVENANCE_MODULE
    ).read_bytes() == expected_provenance
    assert (
        snapshot / "kt-kernel/python" / EMBEDDED_PROVENANCE_MODULE
    ).read_bytes() == expected_provenance


def write_test_wheel(
    path: Path,
    distribution: str,
    version: str,
    *,
    purelib: bool,
    provenance: bytes | None = None,
) -> None:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            f"Root-Is-Purelib: {str(purelib).lower()}\n"
            "Tag: cp312-cp312-linux_x86_64\n",
        )
        if distribution == "kt-kernel" and not purelib:
            wheel.writestr(
                "kt_kernel/kt_kernel_ext.cpython-312-x86_64-linux-gnu.so",
                b"native",
            )
        if provenance is not None:
            package_name = "sglang" if distribution == "sglang-kt" else "kt_kernel"
            wheel.writestr(
                f"{package_name}/{EMBEDDED_PROVENANCE_MODULE}",
                provenance,
            )


def test_inspect_wheel_hashes_and_validates_native_kernel(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    wheel_path = root / "kt_kernel-0.6.3.post1-cp312-cp312-linux_x86_64.whl"
    provenance = b"exact provenance\n"
    write_test_wheel(
        wheel_path,
        "kt-kernel",
        "0.6.3.post1",
        purelib=False,
        provenance=provenance,
    )

    artifact = inspect_wheel(
        wheel_path,
        "kt-kernel",
        "0.6.3.post1",
        root,
        expected_provenance=provenance,
    )

    assert artifact.path == wheel_path.name
    assert artifact.sha256 == hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    assert artifact.root_is_purelib is False


def test_inspect_wheel_ignores_vendored_dist_info_metadata(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    wheel_path = root / "setuptools-80.9.0-py3-none-any.whl"
    write_test_wheel(
        wheel_path,
        "setuptools",
        "80.9.0",
        purelib=True,
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr(
            "setuptools/_vendor/wheel-0.45.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: wheel\nVersion: 0.45.1\n",
        )
        wheel.writestr(
            "setuptools/_vendor/wheel-0.45.1.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )

    artifact = inspect_wheel(
        wheel_path,
        "setuptools",
        "80.9.0",
        root,
    )

    assert artifact.distribution == "setuptools"
    assert artifact.version == "80.9.0"


def test_inspect_wheel_rejects_pure_python_kt_kernel(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    wheel_path = root / "kt_kernel-0.6.3.post1-py3-none-any.whl"
    provenance = b"exact provenance\n"
    write_test_wheel(
        wheel_path,
        "kt-kernel",
        "0.6.3.post1",
        purelib=True,
        provenance=provenance,
    )

    with pytest.raises(RuntimeBuildError, match="platform-independent"):
        inspect_wheel(
            wheel_path,
            "kt-kernel",
            "0.6.3.post1",
            root,
            expected_provenance=provenance,
        )


def test_inspect_wheel_rejects_wrong_embedded_provenance(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    wheel_path = root / "kt_kernel-0.6.3.post1-cp312-cp312-linux_x86_64.whl"
    write_test_wheel(
        wheel_path,
        "kt-kernel",
        "0.6.3.post1",
        purelib=False,
        provenance=b"wrong provenance\n",
    )

    with pytest.raises(RuntimeBuildError, match="unexpected build provenance"):
        inspect_wheel(
            wheel_path,
            "kt-kernel",
            "0.6.3.post1",
            root,
            expected_provenance=b"exact provenance\n",
        )


def test_inspect_wheel_rejects_contaminated_sglang_archive(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    wheel_path = root / "sglang_kt-0.6.3.post1-py3-none-any.whl"
    provenance = b"exact provenance\n"
    write_test_wheel(
        wheel_path,
        "sglang-kt",
        "0.6.3.post1",
        purelib=True,
        provenance=provenance,
    )
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("build/lib/sglang/duplicate.py", "CONTAMINATED = True\n")

    with pytest.raises(RuntimeBuildError, match="entries, expected"):
        inspect_wheel(
            wheel_path,
            "sglang-kt",
            "0.6.3.post1",
            root,
            expected_provenance=provenance,
        )
