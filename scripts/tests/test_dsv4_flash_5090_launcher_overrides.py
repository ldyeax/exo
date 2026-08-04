from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_LAUNCHER = REPOSITORY_ROOT / "scripts" / "dsv4_flash_0731_tp2_dwagon.sh"


def launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("DSV4_")
        and name
        not in {
            "FLASHINFER_CUDA_ARCH_LIST",
            "NCCL_P2P_LEVEL",
            "TORCH_CUDA_ARCH_LIST",
        }
    }
    invocation_log = tmp_path / "python-invocations.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf 'flashinfer=%s|torch=%s|nccl=%s|args=%s\\n' "
        '"${FLASHINFER_CUDA_ARCH_LIST-<unset>}" '
        '"${TORCH_CUDA_ARCH_LIST-<unset>}" '
        '"${NCCL_P2P_LEVEL-<unset>}" "$*" '
        '>>"$DSV4_TEST_PYTHON_INVOCATIONS"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    binary_directory = tmp_path / "bin"
    binary_directory.mkdir()
    fake_ss = binary_directory / "ss"
    fake_ss.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_ss.chmod(0o755)
    fake_nvidia_smi = binary_directory / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\nprintf '24000\\n24000\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_numactl = binary_directory / "numactl"
    fake_numactl.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nshift 2\nexec "$@"\n',
        encoding="utf-8",
    )
    fake_numactl.chmod(0o755)

    environment.update(
        {
            "DSV4_CACHE_ROOT": str(tmp_path / "cache"),
            "DSV4_MODEL_PATH": str(tmp_path / "model"),
            "DSV4_PYTHON": str(fake_python),
            "DSV4_TEST_PYTHON_INVOCATIONS": str(invocation_log),
            "PATH": f"{binary_directory}:{environment['PATH']}",
        }
    )
    return environment, invocation_log


def run_launcher(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BASE_LAUNCHER), "--launch"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_3090_launch_defaults_remain_sm86_and_nvlink(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert launch.startswith("flashinfer=8.6|torch=8.6|nccl=NVL|")
    assert "-m sglang.launch_server" in launch


def test_mixed_arch_launch_can_request_sm86_sm120_and_nccl_auto(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment.update(
        {
            "DSV4_FLASHINFER_CUDA_ARCH_LIST": "8.6 12.0",
            "DSV4_TORCH_CUDA_ARCH_LIST": "8.6;12.0",
            "DSV4_NCCL_P2P_LEVEL": "",
        }
    )

    result = run_launcher(environment)

    assert result.returncode == 0, result.stderr
    launch = invocation_log.read_text(encoding="utf-8").splitlines()[-1]
    assert launch.startswith("flashinfer=8.6 12.0|torch=8.6;12.0|nccl=<unset>|")


def test_invalid_flashinfer_arch_list_fails_before_preparation(
    tmp_path: Path,
) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_FLASHINFER_CUDA_ARCH_LIST"] = "8.6 12.0 12.1"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_FLASHINFER_CUDA_ARCH_LIST must be" in result.stderr
    assert not invocation_log.exists()


def test_invalid_torch_arch_list_fails_before_preparation(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_TORCH_CUDA_ARCH_LIST"] = "8.6 12.0"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_TORCH_CUDA_ARCH_LIST must be" in result.stderr
    assert not invocation_log.exists()


def test_invalid_nccl_p2p_level_fails_before_preparation(tmp_path: Path) -> None:
    environment, invocation_log = launcher_environment(tmp_path)
    environment["DSV4_NCCL_P2P_LEVEL"] = "AUTO"

    result = run_launcher(environment)

    assert result.returncode == 2
    assert "DSV4_NCCL_P2P_LEVEL must be empty" in result.stderr
    assert not invocation_log.exists()
