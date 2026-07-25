from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

import scripts.install_sglang_kt_runtime as installer
from exo.worker.sglang_kt.launch_spec import (
    GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    GLM_4_7_FLASH_SGLANG_REVISION,
)
from scripts.install_sglang_kt_runtime import (
    EXPECTED_DISTRIBUTIONS,
    EXPECTED_PACKAGE_VERSION,
    BaseRuntimeObservation,
    RuntimeInstallError,
    RuntimeInstallPlan,
    execute_runtime_install,
    observe_base_runtime,
    observe_runtime_build,
    plan_runtime_install,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_wheel(
    path: Path,
    distribution: str,
    *,
    provenance: bytes | None = None,
) -> None:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{EXPECTED_PACKAGE_VERSION}.dist-info"
    with ZipFile(path, "w") as wheel:
        wheel.writestr(f"{normalized}/__init__.py", "")
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {EXPECTED_PACKAGE_VERSION}\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        provenance_package = installer.PROVENANCE_PACKAGES.get(distribution)
        if provenance_package is not None:
            wheel.writestr(
                f"{provenance_package}/_exo_build_provenance.py",
                (
                    installer._embedded_provenance_contents()
                    if provenance is None
                    else provenance
                ),
            )


def write_build_receipt(
    tmp_path: Path,
    *,
    ktransformers_revision: str = GLM_4_7_FLASH_KTRANSFORMERS_REVISION,
    sglang_revision: str = GLM_4_7_FLASH_SGLANG_REVISION,
) -> Path:
    build_id = "b" * 64
    build_root = tmp_path / "builds" / "dwagon" / build_id
    wheel_directory = build_root / "wheels"
    wheel_directory.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    for distribution in sorted(EXPECTED_DISTRIBUTIONS):
        filename = (
            f"{distribution.replace('-', '_')}-{EXPECTED_PACKAGE_VERSION}"
            "-py3-none-any.whl"
        )
        wheel_path = wheel_directory / filename
        write_wheel(
            wheel_path,
            distribution,
            provenance=installer._embedded_provenance_contents(
                ktransformers_revision,
                sglang_revision,
            ),
        )
        artifacts.append(
            {
                "distribution": distribution,
                "version": EXPECTED_PACKAGE_VERSION,
                "filename": filename,
                "path": f"wheels/{filename}",
                "size_bytes": wheel_path.stat().st_size,
                "sha256": sha256_bytes(wheel_path.read_bytes()),
                "root_is_purelib": True,
                "tags": ["py3-none-any"],
            }
        )
    receipt_path = build_root / "build-receipt.json"
    document = {
        "schema_version": 2,
        "status": "wheel_build_complete",
        "build_id": build_id,
        "builder_sha256": sha256_bytes(
            Path(installer.__file__)
            .with_name("build_sglang_kt_runtime.py")
            .read_bytes()
        ),
        "source": {
            "ktransformers_revision": ktransformers_revision,
            "sglang_revision": sglang_revision,
            "package_version": EXPECTED_PACKAGE_VERSION,
        },
        "toolchain": {"host_profile": "dwagon"},
        "layout": {
            "build_root": str(build_root),
            "receipt": str(receipt_path),
        },
        "runtime_wheels": artifacts,
    }
    receipt_path.write_text(json.dumps(document))
    return receipt_path


def make_base_runtime(tmp_path: Path) -> BaseRuntimeObservation:
    prefix = tmp_path / "base-runtime"
    python = prefix / "bin/python"
    site_packages = prefix / "lib/python3.12/site-packages"
    base_prefix = tmp_path / "cpython-3.12"
    python.parent.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    base_prefix.mkdir()
    python.write_bytes(b"test CPython 3.12 executable")
    python.chmod(0o755)
    freeze = ("pip==25.2", "torch==2.9.1+cu128")
    freeze_sha256 = sha256_bytes(("\n".join(freeze) + "\n").encode())
    return BaseRuntimeObservation(
        python_path=str(python),
        resolved_python_path=str(python),
        python_sha256=sha256_bytes(python.read_bytes()),
        python_version="3.12.13",
        python_implementation="CPython",
        python_soabi="cpython-312-x86_64-linux-gnu",
        prefix=str(prefix),
        base_prefix=str(base_prefix),
        site_packages=str(site_packages),
        pip_version="25.2",
        pip_freeze=freeze,
        pip_freeze_sha256=freeze_sha256,
    )


def make_plan(
    tmp_path: Path,
) -> tuple[RuntimeInstallPlan, BaseRuntimeObservation, Path]:
    receipt_path = write_build_receipt(tmp_path)
    base = make_base_runtime(tmp_path)
    output_root = tmp_path / "overlays"
    plan = plan_runtime_install(
        receipt_path,
        Path(base.python_path),
        Path(base.site_packages),
        output_root,
        base_runtime=base,
    )
    return plan, base, output_root


def test_preflight_is_read_only_and_deterministic(tmp_path: Path) -> None:
    plan, base, output_root = make_plan(tmp_path)

    receipt = execute_runtime_install(plan)
    repeated = plan_runtime_install(
        Path(plan.build.receipt_path),
        Path(base.python_path),
        Path(base.site_packages),
        output_root,
        base_runtime=base,
    )

    assert receipt.status == "preflight"
    assert receipt.completed_at_utc is None
    assert plan == repeated
    assert not output_root.exists()
    assert len(plan.install_id) == 64
    assert plan.build.receipt_sha256 == sha256_bytes(
        Path(plan.build.receipt_path).read_bytes()
    )
    assert {wheel.distribution for wheel in plan.build.wheels} == set(
        EXPECTED_DISTRIBUTIONS
    )
    assert plan.commands[0][3:6] == ("venv", "--without-pip", plan.layout.venv)
    install_command = plan.commands[1]
    assert "--no-deps" in install_command
    assert "--no-index" in install_command
    assert "--ignore-installed" in install_command
    assert install_command[install_command.index("--target") + 1] == (
        plan.layout.site_packages
    )
    assert dict(plan.environment)["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.parametrize(
    "argument_index",
    range(4),
)
def test_plan_rejects_every_relative_path(
    tmp_path: Path,
    argument_index: int,
) -> None:
    receipt_path = write_build_receipt(tmp_path)
    base = make_base_runtime(tmp_path)
    arguments = [
        receipt_path,
        Path(base.python_path),
        Path(base.site_packages),
        tmp_path / "overlays",
    ]
    arguments[argument_index] = Path("relative")

    with pytest.raises(RuntimeInstallError, match="must be an absolute path"):
        plan_runtime_install(*arguments, base_runtime=base)


def test_build_observation_rejects_wheel_tampering(tmp_path: Path) -> None:
    receipt_path = write_build_receipt(tmp_path)
    document = json.loads(receipt_path.read_text())
    wheel_path = receipt_path.parent / document["runtime_wheels"][0]["path"]
    with wheel_path.open("ab") as wheel:
        wheel.write(b"tampered")

    with pytest.raises(RuntimeInstallError, match="size does not match"):
        observe_runtime_build(receipt_path)


def test_build_observation_rejects_non_glm_source(tmp_path: Path) -> None:
    receipt_path = write_build_receipt(tmp_path)
    document = json.loads(receipt_path.read_text())
    document["source"]["sglang_revision"] = "0" * 40
    receipt_path.write_text(json.dumps(document))

    with pytest.raises(RuntimeInstallError, match="SGLang revision"):
        observe_runtime_build(receipt_path)


def test_build_observation_accepts_explicit_experimental_revisions(
    tmp_path: Path,
) -> None:
    ktransformers_revision = "1" * 40
    sglang_revision = "2" * 40
    receipt_path = write_build_receipt(
        tmp_path,
        ktransformers_revision=ktransformers_revision,
        sglang_revision=sglang_revision,
    )

    observation = observe_runtime_build(
        receipt_path,
        expected_ktransformers_revision=ktransformers_revision,
        expected_sglang_revision=sglang_revision,
    )

    assert observation.ktransformers_revision == ktransformers_revision
    assert observation.sglang_revision == sglang_revision


def test_build_observation_rejects_different_builder(tmp_path: Path) -> None:
    receipt_path = write_build_receipt(tmp_path)
    document = json.loads(receipt_path.read_text())
    document["builder_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(document))

    with pytest.raises(RuntimeInstallError, match="expected current builder"):
        observe_runtime_build(receipt_path)


@pytest.mark.parametrize("distribution", ("kt-kernel", "sglang-kt"))
def test_build_observation_rejects_wrong_embedded_provenance(
    tmp_path: Path,
    distribution: str,
) -> None:
    receipt_path = write_build_receipt(tmp_path)
    document = json.loads(receipt_path.read_text())
    artifact = next(
        item
        for item in document["runtime_wheels"]
        if item["distribution"] == distribution
    )
    wheel_path = receipt_path.parent / artifact["path"]
    write_wheel(wheel_path, distribution, provenance=b"wrong provenance\n")
    artifact["size_bytes"] = wheel_path.stat().st_size
    artifact["sha256"] = sha256_bytes(wheel_path.read_bytes())
    receipt_path.write_text(json.dumps(document))

    with pytest.raises(RuntimeInstallError, match="unexpected build provenance"):
        observe_runtime_build(receipt_path)


def write_fake_installed_distributions(plan: RuntimeInstallPlan) -> None:
    site_packages = Path(plan.layout.site_packages)
    for wheel in plan.build.wheels:
        dist_info = site_packages / (
            f"{wheel.distribution.replace('-', '_')}-{wheel.version}.dist-info"
        )
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {wheel.distribution}\n"
            f"Version: {wheel.version}\n"
        )
        (dist_info / "RECORD").write_text("exact installed record\n")


def test_install_creates_isolated_overlay_and_bound_receipt(tmp_path: Path) -> None:
    plan, base, _ = make_plan(tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_executor(
        arguments: Sequence[str],
        index: int,
        _logs: Path,
        _environment: Mapping[str, str],
    ) -> None:
        calls.append(tuple(arguments))
        if index == 0:
            Path(plan.layout.site_packages).mkdir(parents=True)
            Path(plan.layout.python).parent.mkdir(exist_ok=True)
            Path(plan.layout.python).write_bytes(b"overlay Python")
        else:
            write_fake_installed_distributions(plan)

    base_before = tuple(sorted(Path(base.prefix).rglob("*")))
    receipt = execute_runtime_install(
        plan,
        install=True,
        base_observer=lambda _python, _site: base,
        command_executor=fake_executor,
    )

    assert calls == list(plan.commands)
    assert receipt.status == "install_complete"
    assert len(receipt.installed_distributions) == 3
    assert Path(plan.layout.base_runtime_pth).read_text() == (f"{base.site_packages}\n")
    document = json.loads(Path(plan.layout.receipt).read_text())
    assert document["build"]["receipt_sha256"] == plan.build.receipt_sha256
    assert document["base_runtime"]["pip_freeze_sha256"] == (base.pip_freeze_sha256)
    assert document["installer_sha256"] == plan.installer_sha256
    assert tuple(sorted(Path(base.prefix).rglob("*"))) == base_before


def test_install_refuses_to_reuse_existing_output(tmp_path: Path) -> None:
    plan, base, _ = make_plan(tmp_path)
    Path(plan.layout.install_root).mkdir(parents=True)

    with pytest.raises(RuntimeInstallError, match="already exists"):
        execute_runtime_install(
            plan,
            install=True,
            base_observer=lambda _python, _site: base,
            command_executor=lambda _arguments, _index, _logs, _environment: None,
        )


def test_install_rechecks_base_before_creating_output(tmp_path: Path) -> None:
    plan, base, output_root = make_plan(tmp_path)
    changed = replace(base, pip_freeze_sha256="f" * 64)

    with pytest.raises(RuntimeInstallError, match="changed after preflight"):
        execute_runtime_install(
            plan,
            install=True,
            base_observer=lambda _python, _site: changed,
            command_executor=lambda _arguments, _index, _logs, _environment: None,
        )

    assert not output_root.exists()


def test_observe_base_runtime_validates_python_and_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_base_runtime(tmp_path)
    responses = iter(
        (
            subprocess.CompletedProcess(
                (),
                0,
                json.dumps(
                    {
                        "executable": base.python_path,
                        "version": base.python_version,
                        "implementation": base.python_implementation,
                        "soabi": base.python_soabi,
                        "prefix": base.prefix,
                        "base_prefix": base.base_prefix,
                        "purelib": base.site_packages,
                        "pip_version": base.pip_version,
                    }
                ),
                "",
            ),
            subprocess.CompletedProcess(
                (),
                0,
                "\n".join(base.pip_freeze) + "\n",
                "",
            ),
        )
    )

    def fake_run(
        _arguments: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert environment is not None
        return next(responses)

    monkeypatch.setattr(installer, "_run", fake_run)

    observed = observe_base_runtime(Path(base.python_path), Path(base.site_packages))

    assert observed == base


def test_observe_base_runtime_rejects_wrong_python_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_base_runtime(tmp_path)
    probe = subprocess.CompletedProcess(
        (),
        0,
        json.dumps(
            {
                "executable": base.python_path,
                "version": "3.11.9",
                "implementation": "CPython",
                "soabi": "cpython-311-x86_64-linux-gnu",
                "prefix": base.prefix,
                "base_prefix": base.base_prefix,
                "purelib": base.site_packages,
                "pip_version": base.pip_version,
            }
        ),
        "",
    )
    monkeypatch.setattr(installer, "_run", lambda *_args, **_kwargs: probe)

    with pytest.raises(RuntimeInstallError, match="requires Python 3.12"):
        observe_base_runtime(Path(base.python_path), Path(base.site_packages))
