from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPOSITORY_ROOT
    / "scripts"
    / "dsv4_flash_hybrid_ep2_dwagon_opencode_nvlink_diagnostic.sh"
)


def clean_environment() -> dict[str, str]:
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("DSV4_", "NCCL_"))
    }


def test_diagnostic_rejects_a_smaller_context_before_qualification() -> None:
    environment = clean_environment()
    environment["DSV4_CONTEXT_LENGTH"] = "131072"

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--launch"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "must remain 524288" in result.stderr
    assert "qualify_dsv4_tp2_interconnect.py" not in result.stdout


def test_python_shim_revalidates_evidence_and_forces_nccl_diagnostics() -> None:
    environment = clean_environment()
    environment.update(
        {
            "DSV4_TP2_INTERCONNECT_PYTHON_SHIM": "1",
            "DSV4_TP2_INTERCONNECT_REAL_PYTHON": "/bin/echo",
            "DSV4_TP2_INTERCONNECT_QUALIFIER_PATH": "/tmp/qualifier",
            "DSV4_TP2_INTERCONNECT_RECEIPT": "/tmp/receipt",
            "DSV4_TP2_INTERCONNECT_MAX_RECEIPT_AGE_SECONDS": "300",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "-u",
            "-m",
            "sglang.launch_server",
            "--sentinel",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == [
        "/tmp/qualifier --validate-receipt /tmp/receipt "
        "--maximum-receipt-age-seconds 300 --devices 0,1",
        "-u -m sglang.launch_server --sentinel --enable-p2p-check "
        "--pre-warm-nccl --disable-custom-all-reduce",
    ]
