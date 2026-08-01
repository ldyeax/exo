from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "run_kimi_k3_q2.sh"
RUNTIME_COMMIT = "d29a524eeaf39155825d6f0ef373075fe585cb12"


def clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("KIMI_Q2_")
    }


def fake_server(tmp_path: Path) -> Path:
    server = tmp_path / "build" / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    server.write_text(
        f"""#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "llama-server ({RUNTIME_COMMIT[:8]})"
  exit 0
fi
if [ "$1" = "--help" ]; then
  printf '%s\\n' --device --tensor-split --gpu-layers --split-mode --fit \\
    --ctx-checkpoints --cache-ram --no-cache-prompt --no-warmup \\
    --slot-save-path --reasoning-format --no-host --rpc --rpc-tensor-source \\
    --rpc-tensor-source-mode --spec-draft-device
  exit 0
fi
for argument in "$@"; do
  if [ "$argument" = "--help" ]; then
    exit 0
  fi
  if [ "$argument" = "--list-devices" ]; then
    printf '%s\\n' "$FAKE_DEVICE_LISTING"
    exit 0
  fi
done
if [ -n "${{FAKE_LAUNCH_MARKER:-}}" ]; then
  printf launched > "$FAKE_LAUNCH_MARKER"
fi
exit 2
"""
    )
    server.chmod(0o755)
    return server


def fake_rpc_environment(tmp_path: Path) -> dict[str, str]:
    runtime_root = tmp_path / "rpc-runtime"
    rpc_server = runtime_root / "build-kimik3-file-aware" / "bin" / "ggml-rpc-server"
    rpc_server.parent.mkdir(parents=True)
    rpc_server.write_text(
        """#!/bin/sh
if [ "$1" = "--help" ]; then
  printf '%s\\n' --tensor-source-root --tensor-source-max-files
fi
exit 0
""",
        encoding="utf-8",
    )
    rpc_server.chmod(0o755)
    subprocess.run(
        ["git", "init", "--quiet", str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    marker = runtime_root / "SOURCE_MARKER"
    marker.write_text("test runtime\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(runtime_root), "add", "SOURCE_MARKER"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(runtime_root),
            "-c",
            "user.name=Exo Test",
            "-c",
            "user.email=exo-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test runtime",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime_commit = subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "KIMI_Q2_EXPECTED_RUNTIME_COMMIT": runtime_commit,
        "KIMI_Q2_RPC_RUNTIME_ROOT": str(runtime_root),
        "KIMI_Q2_RPC_SERVER_BINARY": str(rpc_server),
    }


def fake_ssh_environment(
    tmp_path: Path,
    *,
    output: str = "fwuff 1048576000 1048576000 -1",
    exit_status: int = 0,
) -> dict[str, str]:
    binary_directory = tmp_path / "fake-bin"
    binary_directory.mkdir(exist_ok=True)
    ssh_binary = binary_directory / "ssh"
    ssh_binary.write_text(
        """#!/bin/sh
printf '%s\\n' "$FAKE_RPC_HEADROOM_OUTPUT"
exit "$FAKE_RPC_SSH_EXIT_STATUS"
""",
        encoding="utf-8",
    )
    ssh_binary.chmod(0o755)
    return {
        "FAKE_RPC_HEADROOM_OUTPUT": output,
        "FAKE_RPC_SSH_EXIT_STATUS": str(exit_status),
        "PATH": f"{binary_directory}{os.pathsep}{os.environ['PATH']}",
    }


def print_main_command(
    tmp_path: Path,
    *,
    preset: str,
    built_architectures: str,
    extra_arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": built_architectures,
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_DEVICE_CHECK": "1",
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
            "KIMI_Q2_TOPOLOGY_PRESET": preset,
        }
    )
    environment.update(fake_ssh_environment(tmp_path))
    command = [str(LAUNCHER), "--print-command"]
    if extra_arguments:
        command.extend(("--", *extra_arguments))
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def mixed_rpc_validation_environment(
    tmp_path: Path,
    *,
    draft_free_mib: int = 18_159,
    draft_endpoint: str = "10.44.0.2:50052",
) -> dict[str, str]:
    environment = clean_environment()
    environment.update(
        {
            "FAKE_DEVICE_LISTING": "\n".join(
                (
                    "Available devices:",
                    "  RPC0: 10.44.0.2:50052 (1048576 MiB, 1048576 MiB free)",
                    f"  RPC1: {draft_endpoint} (24576 MiB, {draft_free_mib} MiB free)",
                    "  CUDA0: local GPU 0 (24576 MiB, 24000 MiB free)",
                    "  CUDA1: local GPU 1 (24576 MiB, 24000 MiB free)",
                )
            ),
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_RPC_DRAFT_DEVICE_NAME": "RPC1",
            "KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_RPC_SERVER_DEVICE": "CPU,CUDA0",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
        }
    )
    environment.update(fake_ssh_environment(tmp_path))
    return environment


@pytest.mark.parametrize(
    ("preset", "architectures", "devices", "split", "gpu_layers", "uses_rpc"),
    [
        ("legacy", "86", "CUDA0,RPC0,CUDA1", "2,11,3", "16", True),
        (
            "future-4gpu-rpc2",
            "86;120",
            "RPC0,CUDA0,CUDA1,CUDA2,CUDA3",
            "2,2,2,2,4",
            "12",
            True,
        ),
        (
            "future-4gpu-local",
            "86;120",
            "CUDA0,CUDA1,CUDA2,CUDA3",
            "2,2,2,4",
            "10",
            False,
        ),
        (
            "future-5gpu-local-2080",
            "75;86;120",
            "CUDA0,CUDA1,CUDA2,CUDA3,CUDA4",
            "2,2,2,2,4",
            "12",
            False,
        ),
    ],
)
def test_topology_presets_assemble_expected_command(
    tmp_path: Path,
    preset: str,
    architectures: str,
    devices: str,
    split: str,
    gpu_layers: str,
    uses_rpc: bool,
) -> None:
    result = print_main_command(
        tmp_path,
        preset=preset,
        built_architectures=architectures,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:3] == ["/usr/bin/prlimit", "--core=0:0", "--"]
    assert option_value(command, "--device") == devices
    assert option_value(command, "--tensor-split") == split
    assert option_value(command, "--gpu-layers") == gpu_layers
    assert "--no-host" in command
    assert ("--rpc" in command) is uses_rpc
    assert ("--rpc-tensor-source" in command) is uses_rpc
    assert ("--rpc-tensor-source-mode" in command) is uses_rpc
    if uses_rpc:
        assert option_value(command, "--rpc-tensor-source-mode") == "same-backing-file"


@pytest.mark.parametrize(
    "extra_arguments",
    [
        ("--model", "/tmp/not-the-pinned-model.gguf"),
        ("--rpc", "127.0.0.1:1"),
        ("--rpc-tensor-source", "/tmp/not-sanic"),
        ("--rpc_tensor_source_mode=fallback",),
        ("--device", "CPU"),
    ],
)
def test_extra_arguments_cannot_override_frozen_topology(
    tmp_path: Path,
    extra_arguments: tuple[str, ...],
) -> None:
    result = print_main_command(
        tmp_path,
        preset="legacy",
        built_architectures="86",
        extra_arguments=extra_arguments,
    )

    assert result.returncode == 1
    assert "duplicates frozen model/RPC/topology option" in result.stderr


def test_benign_extra_argument_is_preserved(tmp_path: Path) -> None:
    result = print_main_command(
        tmp_path,
        preset="legacy",
        built_architectures="86",
        extra_arguments=("--verbose",),
    )

    assert result.returncode == 0, result.stderr
    assert "--verbose" in shlex.split(result.stdout)


def test_device_and_split_counts_must_align() -> None:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_DEVICE_LIST": "RPC0,CUDA0,CUDA1,CUDA2",
            "KIMI_Q2_TENSOR_SPLIT": "2,11,3",
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "has 4 devices" in result.stderr
    assert "has 3 weights" in result.stderr


def test_arbitrary_aligned_relative_split_is_accepted(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))
    environment.update(
        {
            "KIMI_Q2_DEVICE_LIST": "RPC0,CUDA0,CUDA1,CUDA2",
            "KIMI_Q2_MINIMUM_GPU_FREE_MIBS": "23000,23000,23000",
            "KIMI_Q2_TENSOR_SPLIT": "1,3,1,1",
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_per_device_gpu_minima_must_align() -> None:
    environment = clean_environment()
    environment["KIMI_Q2_MINIMUM_GPU_FREE_MIBS"] = "23000"

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "minima have 1 entries" in result.stderr
    assert "2 local devices" in result.stderr


def test_modified_remote_gpu_disables_cuda_graphs_by_default(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))
    environment.update(
        {
            "KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES": "75",
            "KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES": "75",
            "KIMI_Q2_RPC_SERVER_DEVICE": "CUDA0",
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:2] == ["/usr/bin/env", "GGML_CUDA_DISABLE_GRAPHS=1"]
    assert command[2:5] == ["/usr/bin/prlimit", "--core=0:0", "--"]
    assert option_value(command, "--device") == "CUDA0"


def test_cpu_rpc_command_disables_core_dumps_without_env_prefix(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout)[:3] == [
        "/usr/bin/prlimit",
        "--core=0:0",
        "--",
    ]


def test_mixed_rpc_command_selects_cpu_and_cuda_and_disables_graphs(
    tmp_path: Path,
) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))
    environment.update(
        {
            "KIMI_Q2_RPC_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_RPC_DRAFT_DEVICE_NAME": "RPC1",
            "KIMI_Q2_RPC_REQUIRED_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_RPC_SERVER_DEVICE": "CPU,CUDA0",
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:2] == ["/usr/bin/env", "GGML_CUDA_DISABLE_GRAPHS=1"]
    assert command[2:5] == ["/usr/bin/prlimit", "--core=0:0", "--"]
    assert option_value(command, "--device") == "CPU,CUDA0"


def test_mixed_rpc_binds_and_validates_draft_device(tmp_path: Path) -> None:
    environment = mixed_rpc_validation_environment(tmp_path)

    result = subprocess.run(
        [str(LAUNCHER), "--validate-only"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "draft_rpc_device=RPC1" in result.stderr

    printed_result = subprocess.run(
        [str(LAUNCHER), "--print-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert printed_result.returncode == 0, printed_result.stderr
    command = shlex.split(printed_result.stdout)
    assert command.count("--spec-draft-device") == 1
    assert option_value(command, "--spec-draft-device") == "RPC1"


def test_validate_only_does_not_execute_prlimited_main_command(
    tmp_path: Path,
) -> None:
    environment = mixed_rpc_validation_environment(tmp_path)
    launch_marker = tmp_path / "main-command-launched"
    environment["FAKE_LAUNCH_MARKER"] = str(launch_marker)

    result = subprocess.run(
        [str(LAUNCHER), "--validate-only"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "validation completed without loading the model" in result.stderr
    assert result.stdout == ""
    assert not launch_marker.exists()


def test_mixed_rpc_rejects_insufficient_draft_gpu_capacity(tmp_path: Path) -> None:
    environment = mixed_rpc_validation_environment(tmp_path, draft_free_mib=15_999)

    result = subprocess.run(
        [str(LAUNCHER), "--validate-only"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "RPC1 has only 15999 MiB" in result.stderr
    assert "require at least 16000 MiB for K3 DSpark" in result.stderr


def test_mixed_rpc_rejects_wrong_draft_endpoint(
    tmp_path: Path,
) -> None:
    environment = mixed_rpc_validation_environment(
        tmp_path,
        draft_endpoint="10.44.0.3:50052",
    )

    result = subprocess.run(
        [str(LAUNCHER), "--validate-only"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "does not describe expected draft RPC endpoint 10.44.0.2:50052" in result.stderr
    )


def test_rpc_device_pair_requires_explicit_draft_device_name() -> None:
    environment = clean_environment()
    environment["KIMI_Q2_RPC_SERVER_DEVICE"] = "CPU,CUDA0"

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "pair requires KIMI_Q2_RPC_DRAFT_DEVICE_NAME" in result.stderr


def test_rpc_draft_device_name_requires_server_device_pair() -> None:
    environment = clean_environment()
    environment["KIMI_Q2_RPC_DRAFT_DEVICE_NAME"] = "RPC1"

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "requires a target,draft KIMI_Q2_RPC_SERVER_DEVICE pair" in result.stderr


def test_fixed_rpc_draft_device_cannot_be_duplicated_after_separator(
    tmp_path: Path,
) -> None:
    environment = mixed_rpc_validation_environment(tmp_path)

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--print-command",
            "--",
            "--spec-draft-device",
            "RPC9",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "duplicates KIMI_Q2_RPC_DRAFT_DEVICE_NAME" in result.stderr


def test_local_only_preset_has_no_rpc_server_command() -> None:
    environment = clean_environment()
    environment["KIMI_Q2_TOPOLOGY_PRESET"] = "future-4gpu-local"

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "local-only and has no RPC command" in result.stderr


def test_no_host_can_be_disabled_for_control_run(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_NO_HOST": "0",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_DEVICE_CHECK": "1",
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
        }
    )
    environment.update(fake_ssh_environment(tmp_path))

    result = subprocess.run(
        [str(LAUNCHER), "--print-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--no-host" not in shlex.split(result.stdout)


def test_rpc_headroom_uses_cgroup_limit_and_fails_closed(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB": "128",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_DEVICE_CHECK": "1",
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
        }
    )
    environment.update(
        fake_ssh_environment(
            tmp_path,
            # MemAvailable is 1 GiB, but finite cgroup headroom is only 64 MiB.
            output="fwuff 1048576 65536 65536",
        )
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "effective available memory is 64 MiB" in result.stderr
    assert "MemAvailable 1024 MiB, cgroup headroom 64 MiB" in result.stderr
    assert "require at least 128 MiB" in result.stderr


def test_rpc_headroom_ssh_failure_is_fatal_even_when_device_check_is_skipped(
    tmp_path: Path,
) -> None:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_DEVICE_CHECK": "1",
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
        }
    )
    environment.update(fake_ssh_environment(tmp_path, exit_status=255))

    result = subprocess.run(
        [str(LAUNCHER), "--print-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "could not query fail-closed RPC host headroom" in result.stderr


def test_device_discovery_requires_exact_device_name_field(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(
        {
            "FAKE_DEVICE_LISTING": "\n".join(
                (
                    "Available devices:",
                    "  RPC0: CUDA0: 10.44.0.2:50052 (262144 MiB, 262144 MiB free)",
                    "  CUDA1: local GPU (24576 MiB, 24000 MiB free)",
                )
            ),
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_BUILT_CUDA_ARCHITECTURES": "86",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_MINIMUM_RPC_HOST_AVAILABLE_MIB": "128",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
        }
    )
    environment.update(fake_ssh_environment(tmp_path))

    result = subprocess.run(
        [str(LAUNCHER), "--validate-only"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "required device CUDA0 was not enumerated" in result.stderr


def test_future_blackwell_preset_rejects_sm86_only_build(tmp_path: Path) -> None:
    result = print_main_command(
        tmp_path,
        preset="future-4gpu-local",
        built_architectures="86",
    )

    assert result.returncode == 1
    assert "requires sm_120" in result.stderr


def test_future_blackwell_preset_accepts_architecture_specific_sm120a(
    tmp_path: Path,
) -> None:
    result = print_main_command(
        tmp_path,
        preset="future-4gpu-local",
        built_architectures="75;86;120a",
    )

    assert result.returncode == 0, result.stderr


def test_future_preset_rejects_missing_cuda_architecture_evidence(
    tmp_path: Path,
) -> None:
    environment = clean_environment()
    environment.update(
        {
            "KIMI_Q2_ALLOW_OTHER_HOST": "1",
            "KIMI_Q2_MINIMUM_HOST_AVAILABLE_MIB": "0",
            "KIMI_Q2_SERVER_BINARY": str(fake_server(tmp_path)),
            "KIMI_Q2_SKIP_DEVICE_CHECK": "1",
            "KIMI_Q2_SKIP_MODEL_CHECK": "1",
            "KIMI_Q2_SLOT_SAVE_PATH": str(tmp_path / "slot"),
            "KIMI_Q2_TOPOLOGY_PRESET": "future-4gpu-local",
        }
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "fail-closed CUDA architecture evidence" in result.stderr


def test_print_rpc_command_rejects_missing_local_binary(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))
    Path(environment["KIMI_Q2_RPC_SERVER_BINARY"]).unlink()

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "RPC server binary is not executable" in result.stderr


def test_print_rpc_command_rejects_wrong_local_source_commit(tmp_path: Path) -> None:
    environment = clean_environment()
    environment.update(fake_rpc_environment(tmp_path))
    environment["KIMI_Q2_EXPECTED_RUNTIME_COMMIT"] = RUNTIME_COMMIT

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "RPC runtime source is" in result.stderr
    assert f"expected {RUNTIME_COMMIT}" in result.stderr


def test_rpc_rejects_staged_client_copy_without_bypass() -> None:
    environment = clean_environment()
    environment["KIMI_Q2_MODEL_PATH"] = (
        "/mnt/llm-models/Kimi-K3-GGUF/UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL-00001-of-00019.gguf"
    )

    result = subprocess.run(
        [str(LAUNCHER), "--print-rpc-command"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 1
    assert "staged or alternate client copies require KIMI_Q2_RPC_ENABLED=0" in (
        result.stderr
    )
