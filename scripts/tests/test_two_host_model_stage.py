from __future__ import annotations

import copy
import errno
import hashlib
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import scripts.two_host_model_stage as model_stage
from scripts.benchmark_lease import BenchmarkLease, atomic_write_json
from scripts.two_host_model_stage import (
    AcquisitionConfig,
    AcquisitionMode,
    CpuBinding,
    GitIdentity,
    HcaBinding,
    LeaseMetadataInputs,
    ModelContractBinding,
    ModelSpec,
    OperationError,
    OwnedProcess,
    ProcessOperationResult,
    RemoteRequest,
    RemoteResponse,
    SignalLatch,
    SnapshotVerification,
    SourceDeployment,
    SshConfig,
    StageConfig,
    StageError,
    SystemEffects,
    TimeoutConfig,
    build_hf_download_argv,
    extract_tar_stream,
    load_config,
    minimum_cleanup_grace_seconds,
    owned_temporary_name,
    prepare_lease_metadata,
    reconcile_remote_cleanup,
    run_staging,
    source_identity,
    staging_metadata_contract,
    validate_active_lease,
    verify_snapshot,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "b" * 40
INDEXED_BYTES = 12
RUNTIME_METADATA = "runtime-metadata.json"
BENCHMARK_RESULT = "benchmark-result.json"
CURRENT_HCA_GIDS = {
    "dwagon": (
        "fe80::10:e000:166:3a19",
        "fe80::10:e000:166:3a1a",
    ),
    "fwuff": (
        "fe80::e41d:2d03:4d:32e1",
        "fe80::e41d:2d03:4d:32e2",
    ),
}


def read_json_object(path: Path) -> dict[str, object]:
    parsed = cast(object, json.loads(path.read_text(encoding="utf-8")))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def make_model() -> ModelSpec:
    return ModelSpec(
        model_id=MODEL_ID,
        revision=REVISION,
        expected_indexed_bytes=INDEXED_BYTES,
    )


def ssh_options(known_hosts: Path) -> tuple[str, ...]:
    return (
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=yes",
        f"-oUserKnownHostsFile={known_hosts}",
        "-oConnectTimeout=10",
        "-oServerAliveInterval=10",
        "-oServerAliveCountMax=3",
        "-oControlMaster=no",
        "-oControlPath=none",
        "-oControlPersist=no",
        "-oRequestTTY=no",
    )


def make_config(
    tmp_path: Path,
    *,
    source_snapshot: Path | None = None,
    acquisition_mode: AcquisitionMode | None = None,
    source_model_contract: ModelContractBinding | None = None,
    remote_destination: Path | None = None,
    result_directory: Path | None = None,
) -> StageConfig:
    model = make_model()
    commit = "a" * 40
    hosts = ("dwagon", "fwuff")
    return StageConfig(
        schema_version=1,
        run_id="smol-stage-test",
        namespace="exo-smol-stage-test",
        result_directory=str(result_directory or tmp_path / "results"),
        local_host_name="dwagon",
        remote_host_name="fwuff",
        model=model,
        local_destination=str(tmp_path / "local" / model.directory_name),
        remote_destination=str(
            remote_destination or Path("/mnt/models") / model.directory_name
        ),
        ssh=SshConfig(
            target="fwuff",
            executable="/usr/bin/ssh",
            remote_python_executable="/usr/bin/python3",
            options=ssh_options(tmp_path / "known_hosts"),
        ),
        acquisition=AcquisitionConfig(
            mode=(
                acquisition_mode
                if acquisition_mode is not None
                else (
                    "preverified_snapshot_copy"
                    if source_snapshot is not None
                    else "huggingface_cli"
                )
            ),
            hf_executable="/usr/local/bin/hf",
            source_snapshot=None if source_snapshot is None else str(source_snapshot),
            source_model_contract=source_model_contract,
            environment={"HOME": "/root", "HF_HUB_DISABLE_TELEMETRY": "1"},
        ),
        timeouts=TimeoutConfig(
            lease_bind_seconds=1.0,
            local_stage_seconds=30.0,
            remote_probe_seconds=5.0,
            remote_transfer_seconds=30.0,
            cleanup_seconds=2.0,
            poll_seconds=0.01,
        ),
        lease_metadata=LeaseMetadataInputs(
            reserved_ports=(53001,),
            git=GitIdentity(commit=commit, dirty=False, dirty_file_hashes={}),
            staging_script_sha256=hashlib.sha256(
                (Path(__file__).parents[1] / "two_host_model_stage.py").read_bytes()
            ).hexdigest(),
            gpu_bindings={host: () for host in hosts},
            cpu_bindings={
                host: CpuBinding(cpu_set="0", numa_nodes=(0,), memory_policy="bind:0")
                for host in hosts
            },
            hca_bindings={
                host: tuple(
                    HcaBinding(device="mlx4_0", port=port, gid=gid)
                    for port, gid in enumerate(CURRENT_HCA_GIDS[host], start=1)
                )
                for host in hosts
            },
            source_deployments={
                host: SourceDeployment(
                    path="/root/exo", commit=commit, dirty_file_hashes={}
                )
                for host in hosts
            },
            owner_pids={host: () for host in hosts},
        ),
    )


def make_system_effects(config: StageConfig) -> SystemEffects:
    result_directory = Path(config.result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    return SystemEffects(config, descriptor, observed_local_host_name="dwagon")


def remap_runtime_config(
    config: StageConfig,
    *,
    local_host_name: str,
    remote_host_name: str,
    remote_destination: Path,
    ssh_executable: Path,
    remote_python_executable: Path,
    hf_executable: Path | None = None,
) -> StageConfig:
    payload = config.model_dump(mode="json")
    old_hosts = (config.local_host_name, config.remote_host_name)
    new_hosts = (local_host_name, remote_host_name)
    payload["local_host_name"] = local_host_name
    payload["remote_host_name"] = remote_host_name
    payload["remote_destination"] = str(remote_destination)
    ssh = cast(dict[str, object], payload["ssh"])
    ssh["target"] = remote_host_name
    ssh["executable"] = str(ssh_executable)
    ssh["remote_python_executable"] = str(remote_python_executable)
    acquisition = cast(dict[str, object], payload["acquisition"])
    if config.acquisition.mode == "local_copy_from_existing_remote_source":
        acquisition["source_snapshot"] = str(remote_destination)
    if hf_executable is not None:
        acquisition["hf_executable"] = str(hf_executable)
    lease = cast(dict[str, object], payload["lease_metadata"])
    for section_name in (
        "gpu_bindings",
        "cpu_bindings",
        "hca_bindings",
        "source_deployments",
        "owner_pids",
    ):
        section = cast(dict[str, object], lease[section_name])
        lease[section_name] = {
            new_host: section[old_host]
            for old_host, new_host in zip(old_hosts, new_hosts, strict=True)
        }
    return StageConfig.model_validate_json(json.dumps(payload))


def bind_config_to_current_source(config: StageConfig) -> StageConfig:
    source = Path(__file__).parents[2]
    observed = source_identity(source)
    payload = config.model_dump(mode="json")
    lease_metadata = cast(dict[str, object], payload["lease_metadata"])
    lease_metadata["git"] = observed.model_dump(mode="json")
    deployments = cast(
        dict[str, dict[str, object]], lease_metadata["source_deployments"]
    )
    for deployment in deployments.values():
        deployment["path"] = str(source)
        deployment["commit"] = observed.commit
        deployment["dirty_file_hashes"] = observed.dirty_file_hashes
    return StageConfig.model_validate_json(json.dumps(payload))


def write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_fake_hf(path: Path, *, revision: str = REVISION) -> Path:
    return write_executable(
        path,
        f"""#!{sys.executable}
import json
import sys
import time
from pathlib import Path

revision = {revision!r}
destination = Path(sys.argv[sys.argv.index('--local-dir') + 1])
files = {{
    'config.json': b'{{}}\\n',
    'model-00001-of-00002.safetensors': b'first',
    'model-00002-of-00002.safetensors': b'second',
    'tokenizer.json': b'{{"tokenizer":true}}\\n',
    'model.safetensors.index.json': json.dumps({{
        'metadata': {{'total_size': {INDEXED_BYTES}}},
        'weight_map': {{
            'a': 'model-00001-of-00002.safetensors',
            'b': 'model-00002-of-00002.safetensors',
        }},
    }}).encode(),
}}
for relative, content in files.items():
    output = destination / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    metadata = destination / '.cache' / 'huggingface' / 'download' / (relative + '.metadata')
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(revision + '\\netag\\n', encoding='utf-8')
(destination / '.cache' / 'huggingface' / '.gitignore').write_text('*\\n', encoding='utf-8')
time.sleep(0.1)
""",
    )


def write_fake_transport(
    directory: Path, *, remote_host_name: str, remote_hook: str | None = None
) -> tuple[Path, Path]:
    remote_python = write_executable(
        directory / "remote-python",
        f"""#!{sys.executable}
import socket
import sys
socket.gethostname = lambda: {remote_host_name!r}
if len(sys.argv) != 3 or sys.argv[1] != '-c':
    raise SystemExit(64)
loader = sys.argv[2]
hook = {remote_hook!r}
if hook is not None:
    needle = 'raise SystemExit(module.remote_helper_main())'
    if needle not in loader:
        raise SystemExit(65)
    loader = loader.replace(
        needle,
        'exec(' + repr(hook) + ',module.__dict__);' + needle,
    )
exec(loader, {{'__name__': '__main__', '__file__': __file__}})
""",
    )
    ssh = write_executable(
        directory / "ssh",
        f"""#!{sys.executable}
import os
import shlex
import sys
command = shlex.split(sys.argv[-1])
os.execv(command[0], command)
""",
    )
    return ssh, remote_python


def write_snapshot(root: Path, *, model: ModelSpec | None = None) -> Path:
    selected_model = model or make_model()
    root.mkdir(parents=True)
    (root / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second")
    (root / "tokenizer.json").write_text('{"tokenizer":true}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    cache = root / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "metadata").write_text("cached\n", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": selected_model.expected_indexed_bytes},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / ".exo-huggingface-revision.json").write_text(
        json.dumps(
            {"repo_id": selected_model.model_id, "revision": selected_model.revision},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def freeze_test_tree(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        os.chown(path, 0, 0, follow_symlinks=False)
        path.chmod(0o555 if path.is_dir() else 0o444)
    os.chown(root, 0, 0, follow_symlinks=False)
    root.chmod(0o555)


def assert_test_tree_is_frozen(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        observed = path.lstat()
        assert observed.st_uid == 0
        assert observed.st_gid == 0
        assert observed.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0


def write_model_contract(
    path: Path, snapshot: Path, *, model: ModelSpec | None = None
) -> ModelContractBinding:
    selected_model = model or make_model()
    roles = {
        "chat_template.jinja": "chat_template",
        "config.json": "config",
        "generation_config.json": "generation_config",
        "model.safetensors.index.json": "safetensors_index",
        "tokenizer.json": "tokenizer",
        "tokenizer_config.json": "tokenizer_config",
        "model-00001-of-00002.safetensors": "weight_shard",
        "model-00002-of-00002.safetensors": "weight_shard",
    }
    files: list[dict[str, object]] = []
    for relative_path in sorted(roles):
        contents = (snapshot / relative_path).read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        is_weight = roles[relative_path] == "weight_shard"
        files.append(
            {
                "path": relative_path,
                "role": roles[relative_path],
                "size_bytes": len(contents),
                "sha256": digest,
                "huggingface_etag_algorithm": (
                    "sha256" if is_weight else "git_blob_sha1"
                ),
                "huggingface_etag_digest": digest if is_weight else "1" * 40,
            }
        )
    contract = {
        "schema_version": 1,
        "canonicalization": "exo-sglang-kt-model-contract-v1",
        "model_id": selected_model.model_id,
        "revision": selected_model.revision,
        "weight_format": "safetensors",
        "ktransformers_method": "BF16",
        "full_indexer_layer_starts": [0],
        "weight_map_entries": 2,
        "index_metadata_total_size": selected_model.expected_indexed_bytes,
        "physical_weight_bytes": sum(
            cast(int, file["size_bytes"])
            for file in files
            if file["role"] == "weight_shard"
        ),
        "files": files,
    }
    raw_contract = (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    path.write_bytes(raw_contract)
    return ModelContractBinding(
        path=str(path), sha256=hashlib.sha256(raw_contract).hexdigest()
    )


def make_verification(marker: str = "same") -> SnapshotVerification:
    return SnapshotVerification(
        model_id=MODEL_ID,
        revision=REVISION,
        indexed_bytes=INDEXED_BYTES,
        shard_count=2,
        manifest={"config.json": marker.ljust(64, "0")[:64]},
    )


class FakeEffects:
    def __init__(
        self,
        config: StageConfig,
        *,
        local_existing: SnapshotVerification | None = None,
        remote_existing: SnapshotVerification | None = None,
    ) -> None:
        self.config = config
        self.local_existing = local_existing
        self.remote_existing = remote_existing
        self.verification = make_verification("a")
        self.source_verification = self.verification
        self.remote_transfer_verification = self.verification
        self.remote_reverification = remote_existing
        self.fragments: list[tuple[str, dict[str, object]]] = []
        self.calls: list[str] = []
        self.local_temporary: Path | None = None
        self.download_cleanup = True
        self.remote_probe_cleanup = True
        self.remote_transfer_cleanup = True
        self.local_temporary_cleanup = True
        self.download_error: OperationError | None = None
        self.remote_probe_error: OperationError | None = None
        self.remote_transfer_error: OperationError | None = None
        self.next_pid = 100

    def _process(self, host_name: str, owner_token: str) -> OwnedProcess:
        self.next_pid += 1
        return OwnedProcess(
            host_name=host_name,
            pid=self.next_pid,
            process_group_id=self.next_pid,
            start_time_ticks=1000 + self.next_pid,
            owner_token=owner_token,
            namespace=self.config.namespace,
            transport_pid=2000 + self.next_pid,
            log_path=f"/tmp/{host_name}-{self.next_pid}.log",
        )

    def inspect_local_destination(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification | None:
        del path, model, latch
        self.calls.append("inspect-local")
        return self.local_existing

    def verify_source_snapshot(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification:
        del path, model, latch
        self.calls.append("verify-source")
        return self.source_verification

    def create_local_temporary(self, destination: Path, owned_name: str) -> Path:
        self.calls.append("create-local-temporary")
        self.local_temporary = destination.parent / owned_name
        return self.local_temporary

    def copy_preverified_snapshot(
        self,
        source: Path,
        destination: Path,
        model: ModelSpec,
        latch: SignalLatch,
    ) -> None:
        del source, destination, model, latch
        self.calls.append("copy-source")

    def download_snapshot(
        self,
        config: StageConfig,
        temporary_path: Path,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> bool:
        del config, temporary_path, latch
        self.calls.append("download")
        register_process(self._process(self.config.local_host_name, owner_token))
        if self.download_error is not None:
            raise self.download_error
        return self.download_cleanup

    def write_revision_receipt(self, path: Path, model: ModelSpec) -> None:
        del path, model
        self.calls.append("write-receipt")

    def verify_local_snapshot(
        self, path: Path, model: ModelSpec, latch: SignalLatch
    ) -> SnapshotVerification:
        del path, model, latch
        self.calls.append("verify-local")
        return self.verification

    def install_local_snapshot(
        self,
        temporary_path: Path,
        destination: Path,
        record_publication: Callable[[], None],
    ) -> None:
        del temporary_path, destination
        self.calls.append("install-local")
        self.local_temporary = None
        record_publication()

    def cleanup_local_temporary(self, path: Path) -> bool:
        del path
        self.calls.append("cleanup-local-temporary")
        self.local_temporary = None
        return self.local_temporary_cleanup

    def probe_remote_destination(
        self,
        config: StageConfig,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult:
        del config, latch
        self.calls.append("probe-remote")
        register_process(self._process(self.config.remote_host_name, owner_token))
        if self.remote_probe_error is not None:
            raise self.remote_probe_error
        return ProcessOperationResult(
            verification=self.remote_existing,
            cleanup_confirmed=self.remote_probe_cleanup,
            installed=False,
        )

    def reverify_remote_destination(
        self,
        config: StageConfig,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult:
        del config, latch
        self.calls.append("reverify-remote")
        register_process(self._process(self.config.remote_host_name, owner_token))
        return ProcessOperationResult(
            verification=self.remote_reverification,
            cleanup_confirmed=self.remote_probe_cleanup,
            installed=False,
        )

    def transfer_remote_snapshot(
        self,
        config: StageConfig,
        local_snapshot: Path,
        local_verification: SnapshotVerification,
        owner_token: str,
        register_process: Callable[[OwnedProcess], None],
        latch: SignalLatch,
    ) -> ProcessOperationResult:
        del config, local_snapshot, local_verification, latch
        self.calls.append("transfer-remote")
        register_process(self._process(self.config.remote_host_name, owner_token))
        if self.remote_transfer_error is not None:
            raise self.remote_transfer_error
        return ProcessOperationResult(
            verification=self.remote_transfer_verification,
            cleanup_confirmed=self.remote_transfer_cleanup,
            installed=True,
        )

    def write_result_json(self, filename: str, value: Mapping[str, object]) -> None:
        self.fragments.append((filename, copy.deepcopy(dict(value))))


def test_strict_config_accepts_exact_revision_pinned_paths(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.schema_version == 1
    assert Path(config.local_destination).name == config.model.directory_name
    assert set(option[2:].split("=", 1)[0] for option in config.ssh.options) == {
        "BatchMode",
        "StrictHostKeyChecking",
        "UserKnownHostsFile",
        "ConnectTimeout",
        "ServerAliveInterval",
        "ServerAliveCountMax",
        "ControlMaster",
        "ControlPath",
        "ControlPersist",
        "RequestTTY",
    }


def test_strict_json_config_round_trips_and_fully_binds_lease_metadata(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config_path = tmp_path / "stage-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    assert load_config(config_path) == config
    assert staging_metadata_contract(config) == config.model_dump(
        mode="json", exclude_none=True
    )


def test_config_forbids_extra_fields(tmp_path: Path) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    payload["surprise"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        StageConfig.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("source_snapshot", "expected_mode"),
    [(None, "huggingface_cli"), ("source", "preverified_snapshot_copy")],
)
def test_schema_v1_legacy_acquisition_mode_is_deterministically_inferred(
    tmp_path: Path,
    source_snapshot: str | None,
    expected_mode: AcquisitionMode,
) -> None:
    payload = make_config(
        tmp_path,
        source_snapshot=(
            None if source_snapshot is None else tmp_path / source_snapshot
        ),
    ).model_dump(mode="json")
    acquisition = cast(dict[str, object], payload["acquisition"])
    del acquisition["mode"]

    parsed = StageConfig.model_validate_json(json.dumps(payload))

    assert parsed.schema_version == 1
    assert parsed.acquisition.mode == expected_mode


def test_schema_v1_rejects_ambiguous_contract_binding_without_mode(
    tmp_path: Path,
) -> None:
    remote_source = tmp_path / "remote" / make_model().directory_name
    payload = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=ModelContractBinding(
            path=str(tmp_path / "contract.json"), sha256="a" * 64
        ),
        remote_destination=remote_source,
    ).model_dump(mode="json")
    acquisition = cast(dict[str, object], payload["acquisition"])
    del acquisition["mode"]

    with pytest.raises(ValidationError, match="mode is required"):
        StageConfig.model_validate_json(json.dumps(payload))


def test_config_binds_ssh_target_to_exact_remote_host(tmp_path: Path) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    cast(dict[str, object], payload["ssh"])["target"] = "other-host"
    with pytest.raises(ValidationError, match="exact remote_host_name"):
        StageConfig.model_validate_json(json.dumps(payload))


def test_config_emits_current_raw_verbs_port_gids(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert {
        host: tuple((binding.device, binding.port, binding.gid) for binding in bindings)
        for host, bindings in config.lease_metadata.hca_bindings.items()
    } == {
        host: tuple(
            ("mlx4_0", port, gid) for port, gid in enumerate(current_gids, start=1)
        )
        for host, current_gids in CURRENT_HCA_GIDS.items()
    }
    dumped = staging_metadata_contract(config)
    hca_bindings = cast(
        dict[str, list[dict[str, object]]],
        cast(dict[str, object], dumped["lease_metadata"])["hca_bindings"],
    )
    assert all(
        set(binding) == {"device", "port", "gid"} and isinstance(binding["gid"], str)
        for bindings in hca_bindings.values()
        for binding in bindings
    )


@pytest.mark.parametrize(
    ("ip_address", "gid"),
    [
        (None, None),
        ("10.0.0.1", "fe80::1"),
        (None, "not-a-gid"),
        (None, "10.0.0.1"),
        (None, "::"),
        (None, "fe80::"),
        (None, "fe80::1%mlx4_0"),
        (None, "fe80:0:0:0:10:e000:166:3a19"),
        ("not-an-address", None),
        ("0.0.0.0", None),
    ],
)
def test_hca_binding_rejects_ambiguous_or_invalid_identity(
    ip_address: str | None, gid: str | None
) -> None:
    with pytest.raises(ValidationError, match="ip_address|gid|IP address|canonical"):
        HcaBinding(
            device="mlx4_0",
            port=1,
            ip_address=ip_address,
            gid=gid,
        )


def test_hca_binding_rejects_unsafe_device_and_repeated_port(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="kernel device name"):
        HcaBinding(device="mlx4_0;bad", port=1, gid=CURRENT_HCA_GIDS["dwagon"][0])

    payload = make_config(tmp_path).model_dump(mode="json")
    metadata = cast(dict[str, object], payload["lease_metadata"])
    bindings = cast(dict[str, list[dict[str, object]]], metadata["hca_bindings"])
    bindings["dwagon"][1]["port"] = 1
    with pytest.raises(ValidationError, match="repeat a device port"):
        StageConfig.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39, "a" * 41])
def test_model_requires_exact_lowercase_commit(revision: str) -> None:
    with pytest.raises(ValidationError, match="40-hex"):
        ModelSpec(
            model_id=MODEL_ID,
            revision=revision,
            expected_indexed_bytes=INDEXED_BYTES,
        )


@pytest.mark.parametrize(
    "model_id", ["repo", "../repo", "owner/../repo", "owner/repo name", "-owner/repo"]
)
def test_model_id_rejects_path_and_command_syntax(model_id: str) -> None:
    with pytest.raises(ValidationError, match="owner/repository"):
        ModelSpec(
            model_id=model_id,
            revision=REVISION,
            expected_indexed_bytes=INDEXED_BYTES,
        )


def test_destination_requires_exact_exo_revision_suffix(tmp_path: Path) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    payload["local_destination"] = str(tmp_path / f"model-{REVISION}")
    with pytest.raises(ValidationError, match="exact revision-pinned Exo name"):
        StageConfig.model_validate_json(json.dumps(payload))


def test_config_rejects_source_and_result_path_overlap(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    payload = config.model_dump(mode="json")
    acquisition = cast(dict[str, object], payload["acquisition"])
    acquisition["mode"] = "preverified_snapshot_copy"
    acquisition["source_snapshot"] = str(Path(config.local_destination).parent)
    with pytest.raises(ValidationError, match="must not overlap"):
        StageConfig.model_validate_json(json.dumps(payload))

    payload = config.model_dump(mode="json")
    payload["result_directory"] = str(Path(config.local_destination).parent)
    with pytest.raises(ValidationError, match="must not overlap"):
        StageConfig.model_validate_json(json.dumps(payload))

    payload = config.model_dump(mode="json")
    acquisition = cast(dict[str, object], payload["acquisition"])
    acquisition["mode"] = "preverified_snapshot_copy"
    acquisition["source_snapshot"] = str(Path(config.result_directory))
    with pytest.raises(ValidationError, match="source snapshot must not overlap"):
        StageConfig.model_validate_json(json.dumps(payload))


def test_config_explicitly_admits_only_contract_bound_existing_remote_source(
    tmp_path: Path,
) -> None:
    remote_source = tmp_path / "remote" / make_model().directory_name
    binding = ModelContractBinding(
        path=str(tmp_path / "contract.json"), sha256="a" * 64
    )
    config = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=binding,
        remote_destination=remote_source,
    )
    assert config.acquisition.source_snapshot == config.remote_destination
    assert config.acquisition.source_model_contract == binding

    payload = config.model_dump(mode="json")
    cast(dict[str, object], payload["acquisition"])["source_model_contract"] = None
    with pytest.raises(ValidationError, match="requires a model contract"):
        StageConfig.model_validate_json(json.dumps(payload))

    payload = config.model_dump(mode="json")
    payload["remote_destination"] = str(
        tmp_path / "other" / make_model().directory_name
    )
    with pytest.raises(ValidationError, match="exactly equal remote_destination"):
        StageConfig.model_validate_json(json.dumps(payload))

    payload = config.model_dump(mode="json")
    cast(dict[str, object], payload["acquisition"])["mode"] = (
        "preverified_snapshot_copy"
    )
    with pytest.raises(ValidationError, match="only valid for an existing remote"):
        StageConfig.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("path", ["relative/path", "/tmp/a/../b", "/tmp/a/"])
def test_config_rejects_noncanonical_paths(tmp_path: Path, path: str) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    payload["result_directory"] = path
    with pytest.raises(ValidationError, match="absolute and lexically canonical"):
        StageConfig.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("option", "match"),
    [
        ("-oStrictHostKeyChecking=no", "must equal yes"),
        ("-oProxyCommand=evil", "unsupported"),
        ("-oConnectTimeout=0", "positive integer"),
        ("BatchMode=yes", "exact -oName=value"),
    ],
)
def test_ssh_options_are_exactly_whitelisted(
    tmp_path: Path, option: str, match: str
) -> None:
    options = list(ssh_options(tmp_path / "known_hosts"))
    key = option.removeprefix("-o").split("=", 1)[0]
    replacement = next(
        (
            index
            for index, candidate in enumerate(options)
            if candidate.removeprefix("-o").split("=", 1)[0] == key
        ),
        0,
    )
    options[replacement] = option
    with pytest.raises(ValidationError, match=match):
        SshConfig(
            target="fwuff",
            executable="/usr/bin/ssh",
            remote_python_executable="/usr/bin/python3",
            options=tuple(options),
        )


@pytest.mark.parametrize("target", ["-fwuff", "fwuff;touch", "fwuff host", "fwuff\n"])
def test_ssh_target_rejects_command_syntax(tmp_path: Path, target: str) -> None:
    with pytest.raises(ValidationError, match="unsafe"):
        SshConfig(
            target=target,
            executable="/usr/bin/ssh",
            remote_python_executable="/usr/bin/python3",
            options=ssh_options(tmp_path / "known_hosts"),
        )


def test_timeout_rejects_nonfinite_values(tmp_path: Path) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    cast(dict[str, object], payload["timeouts"])["cleanup_seconds"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        StageConfig.model_validate_json(json.dumps(payload))


def test_hf_argv_is_the_single_whitelisted_shape(tmp_path: Path) -> None:
    model = make_model()
    assert build_hf_download_argv(Path("/opt/hf/bin/hf"), model, tmp_path) == (
        "/opt/hf/bin/hf",
        "download",
        MODEL_ID,
        "--revision",
        REVISION,
        "--local-dir",
        str(tmp_path),
    )


def test_snapshot_verifier_hashes_every_regular_file(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    result = verify_snapshot(snapshot, make_model())
    assert result.indexed_bytes == INDEXED_BYTES
    assert result.shard_count == 2
    assert set(result.manifest) == {
        ".cache/huggingface/metadata",
        ".exo-huggingface-revision.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert all(len(digest) == 64 for digest in result.manifest.values())


@pytest.mark.parametrize("total_size", [True, INDEXED_BYTES + 1, "12"])
def test_snapshot_rejects_wrong_indexed_size(
    tmp_path: Path, total_size: object
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    index_path = snapshot / "model.safetensors.index.json"
    index = read_json_object(index_path)
    cast(dict[str, object], index["metadata"])["total_size"] = total_size
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(StageError, match="total_size"):
        verify_snapshot(snapshot, make_model())


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape.safetensors", "/absolute.safetensors", "a\\b.safetensors"],
)
def test_snapshot_rejects_index_path_traversal(
    tmp_path: Path, unsafe_name: str
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    index_path = snapshot / "model.safetensors.index.json"
    index = read_json_object(index_path)
    cast(dict[str, object], index["weight_map"])["a"] = unsafe_name
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(StageError, match="unsafe snapshot file path"):
        verify_snapshot(snapshot, make_model())


def test_snapshot_rejects_missing_and_extra_safetensors(tmp_path: Path) -> None:
    missing = write_snapshot(tmp_path / "missing")
    (missing / "model-00001-of-00002.safetensors").unlink()
    with pytest.raises(StageError, match="referenced safetensors shard is missing"):
        verify_snapshot(missing, make_model())

    extra = write_snapshot(tmp_path / "extra")
    (extra / "unindexed.safetensors").write_bytes(b"extra")
    with pytest.raises(StageError, match="unindexed safetensors"):
        verify_snapshot(extra, make_model())


def test_snapshot_requires_model_config(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / "config.json").unlink()
    with pytest.raises(StageError, match="config.json"):
        verify_snapshot(snapshot, make_model())


def test_snapshot_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    symlink_snapshot = write_snapshot(tmp_path / "symlink")
    (symlink_snapshot / "link").symlink_to("config.json")
    with pytest.raises(StageError, match="symlink"):
        verify_snapshot(symlink_snapshot, make_model())

    fifo_snapshot = write_snapshot(tmp_path / "fifo")
    os.mkfifo(fifo_snapshot / "pipe")
    with pytest.raises(StageError, match="non-regular"):
        verify_snapshot(fifo_snapshot, make_model())


@pytest.mark.parametrize(
    "receipt",
    [
        {"repo_id": MODEL_ID, "revision": "c" * 40},
        {"repo_id": MODEL_ID, "revision": REVISION, "extra": True},
    ],
)
def test_snapshot_requires_exact_exo_receipt(
    tmp_path: Path, receipt: dict[str, object]
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / ".exo-huggingface-revision.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(StageError, match="receipt"):
        verify_snapshot(snapshot, make_model())


def test_contract_bound_snapshot_allows_no_receipt_and_rejects_contract_tampering(
    tmp_path: Path,
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / ".exo-huggingface-revision.json").unlink()
    contract_path = tmp_path / "contract.json"
    binding = write_model_contract(contract_path, snapshot)
    verified = verify_snapshot(snapshot, make_model(), source_model_contract=binding)
    assert verified.model_contract_sha256 == binding.sha256
    with pytest.raises(StageError, match="revision receipt"):
        verify_snapshot(snapshot, make_model())

    with pytest.raises(StageError, match="configured SHA-256"):
        verify_snapshot(
            snapshot,
            make_model(),
            source_model_contract=binding.model_copy(update={"sha256": "0" * 64}),
        )

    original_contract = cast(dict[str, object], json.loads(contract_path.read_bytes()))
    wrong_revision = copy.deepcopy(original_contract)
    wrong_revision["revision"] = "c" * 40
    wrong_revision_bytes = (
        json.dumps(wrong_revision, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    contract_path.write_bytes(wrong_revision_bytes)
    wrong_revision_binding = ModelContractBinding(
        path=str(contract_path),
        sha256=hashlib.sha256(wrong_revision_bytes).hexdigest(),
    )
    with pytest.raises(StageError, match="identity differs"):
        verify_snapshot(
            snapshot,
            make_model(),
            source_model_contract=wrong_revision_binding,
        )

    wrong_file = copy.deepcopy(original_contract)
    contract_files = cast(list[dict[str, object]], wrong_file["files"])
    contract_files[0]["sha256"] = "f" * 64
    wrong_file_bytes = (
        json.dumps(wrong_file, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    contract_path.write_bytes(wrong_file_bytes)
    wrong_file_binding = ModelContractBinding(
        path=str(contract_path), sha256=hashlib.sha256(wrong_file_bytes).hexdigest()
    )
    with pytest.raises(StageError, match="differs from source model contract"):
        verify_snapshot(
            snapshot, make_model(), source_model_contract=wrong_file_binding
        )


@pytest.mark.parametrize("omitted_role", ["chat_template", "tokenizer"])
def test_contract_requires_every_authoritative_singleton_runtime_file(
    tmp_path: Path, omitted_role: str
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / ".exo-huggingface-revision.json").unlink()
    contract_path = tmp_path / "contract.json"
    write_model_contract(contract_path, snapshot)
    contract = cast(dict[str, object], json.loads(contract_path.read_bytes()))
    files = cast(list[dict[str, object]], contract["files"])
    contract["files"] = [file for file in files if file["role"] != omitted_role]
    raw_contract = (
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    contract_path.write_bytes(raw_contract)
    binding = ModelContractBinding(
        path=str(contract_path), sha256=hashlib.sha256(raw_contract).hexdigest()
    )

    with pytest.raises(StageError, match="invalid schema"):
        verify_snapshot(snapshot, make_model(), source_model_contract=binding)


@pytest.mark.parametrize("filename", ["remote_model.py", "native_kernel.so"])
def test_contract_bound_snapshot_rejects_uncontracted_executable_code(
    tmp_path: Path, filename: str
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / ".exo-huggingface-revision.json").unlink()
    binding = write_model_contract(tmp_path / "contract.json", snapshot)
    (snapshot / filename).write_bytes(b"uncontracted")

    with pytest.raises(StageError, match="uncontracted executable code"):
        verify_snapshot(snapshot, make_model(), source_model_contract=binding)


def test_contract_bound_snapshot_rejects_uncontracted_execute_mode(
    tmp_path: Path,
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / ".exo-huggingface-revision.json").unlink()
    binding = write_model_contract(tmp_path / "contract.json", snapshot)
    uncontracted = snapshot / "launcher"
    uncontracted.write_bytes(b"uncontracted")
    uncontracted.chmod(0o755)

    with pytest.raises(StageError, match="uncontracted executable code"):
        verify_snapshot(snapshot, make_model(), source_model_contract=binding)


def test_snapshot_rejects_nested_safetensors_shards(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    nested = snapshot / "nested"
    nested.mkdir()
    (snapshot / "model-00001-of-00002.safetensors").rename(
        nested / "model-00001-of-00002.safetensors"
    )
    index_path = snapshot / "model.safetensors.index.json"
    index = read_json_object(index_path)
    cast(dict[str, object], index["weight_map"])["a"] = (
        "nested/model-00001-of-00002.safetensors"
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(StageError, match="nested shard"):
        verify_snapshot(snapshot, make_model())


def test_model_contract_read_is_bounded_by_descriptor_size_before_parsing(
    tmp_path: Path,
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    contract_path = tmp_path / "oversized-contract.json"
    with contract_path.open("wb") as output:
        output.truncate(model_stage._MAX_MODEL_CONTRACT_BYTES + 1)
    binding = ModelContractBinding(path=str(contract_path), sha256="a" * 64)

    with pytest.raises(StageError, match="exceeds the size limit"):
        verify_snapshot(snapshot, make_model(), source_model_contract=binding)


def test_model_contract_requires_canonical_bytes_even_with_matching_digest(
    tmp_path: Path,
) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    contract_path = tmp_path / "contract.json"
    write_model_contract(contract_path, snapshot)
    parsed = cast(dict[str, object], json.loads(contract_path.read_bytes()))
    noncanonical = json.dumps(parsed, indent=2, sort_keys=True).encode("ascii")
    contract_path.write_bytes(noncanonical)
    binding = ModelContractBinding(
        path=str(contract_path), sha256=hashlib.sha256(noncanonical).hexdigest()
    )

    with pytest.raises(StageError, match="not canonical JSON"):
        verify_snapshot(snapshot, make_model(), source_model_contract=binding)


def test_system_effects_copy_only_accepts_preverified_source(tmp_path: Path) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    source = write_snapshot(tmp_path / "source")
    destination_parent = Path(config.local_destination).parent
    destination_parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(
        Path(config.local_destination), "owned.stage"
    )
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    copied = effects.verify_local_snapshot(temporary, config.model, SignalLatch())
    original = verify_snapshot(source, config.model)
    assert copied.model_id == original.model_id
    assert copied.revision == original.revision
    assert copied.indexed_bytes == original.indexed_bytes
    assert copied.shard_count == original.shard_count
    assert {
        path: digest
        for path, digest in copied.manifest.items()
        if path != ".exo-huggingface-revision.json"
    } == {
        path: digest
        for path, digest in original.manifest.items()
        if path != ".exo-huggingface-revision.json"
    }


def test_hf_metadata_is_validated_before_revision_receipt(tmp_path: Path) -> None:
    fake_hf = write_fake_hf(tmp_path / "hf")
    config = remap_runtime_config(
        make_config(tmp_path),
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=tmp_path / "remote" / make_model().directory_name,
        ssh_executable=Path("/usr/bin/ssh"),
        remote_python_executable=Path("/usr/bin/python3"),
        hf_executable=fake_hf,
    )
    effects = make_system_effects(config)
    destination = Path(config.local_destination)
    destination.parent.mkdir(parents=True)
    temporary = effects.create_local_temporary(destination, "owned-download.stage")
    processes: list[OwnedProcess] = []

    assert effects.download_snapshot(
        config,
        temporary,
        "owner-token",
        processes.append,
        SignalLatch(),
    )
    assert processes and processes[0].host_name == "dwagon"
    assert not (temporary / ".exo-huggingface-revision.json").exists()
    effects.write_revision_receipt(temporary, config.model)
    assert (
        effects.verify_local_snapshot(temporary, config.model, SignalLatch()).revision
        == REVISION
    )
    assert effects.cleanup_local_temporary(temporary) is False
    assert verify_snapshot(temporary, config.model).revision == REVISION


def test_hf_revision_metadata_mismatch_fails_before_receipt(tmp_path: Path) -> None:
    fake_hf = write_fake_hf(tmp_path / "hf", revision="c" * 40)
    config = remap_runtime_config(
        make_config(tmp_path),
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=tmp_path / "remote" / make_model().directory_name,
        ssh_executable=Path("/usr/bin/ssh"),
        remote_python_executable=Path("/usr/bin/python3"),
        hf_executable=fake_hf,
    )
    effects = make_system_effects(config)
    destination = Path(config.local_destination)
    destination.parent.mkdir(parents=True)
    temporary = effects.create_local_temporary(destination, "owned-download.stage")

    with pytest.raises(StageError, match="not pinned"):
        effects.download_snapshot(
            config,
            temporary,
            "owner-token",
            lambda _process: None,
            SignalLatch(),
        )
    assert not (temporary / ".exo-huggingface-revision.json").exists()
    assert effects.cleanup_local_temporary(temporary) is False
    assert (temporary / "config.json").is_file()


def test_system_effects_binds_exact_local_host_identity(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result_directory = Path(config.result_directory)
    result_directory.mkdir()
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(StageError, match="local host identity"):
            SystemEffects(config, descriptor, observed_local_host_name="not-dwagon")
    finally:
        os.close(descriptor)


def test_system_effects_rejects_a_result_descriptor_for_another_directory(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    Path(config.result_directory).mkdir()
    other_directory = tmp_path / "other-results"
    other_directory.mkdir()
    descriptor = os.open(
        other_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        with pytest.raises(StageError, match="path identity changed"):
            SystemEffects(config, descriptor, observed_local_host_name="dwagon")
    finally:
        os.close(descriptor)


def test_real_fake_ssh_transfer_drains_large_remote_receipt_concurrently(
    tmp_path: Path,
) -> None:
    ssh, remote_python = write_fake_transport(tmp_path, remote_host_name="fwuff")
    model = make_model()
    remote_destination = tmp_path / "remote" / model.directory_name
    config = remap_runtime_config(
        make_config(tmp_path),
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_destination,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    local_destination = write_snapshot(Path(config.local_destination))
    extras = local_destination / "extras"
    extras.mkdir()
    for index in range(900):
        (extras / f"file-{index:04d}.txt").write_text(
            f"payload-{index}\n", encoding="utf-8"
        )
    local_verification = verify_snapshot(local_destination, config.model)
    assert len(json.dumps(local_verification.model_dump(mode="json"))) > 65536
    remote_destination.parent.mkdir(parents=True)
    effects = make_system_effects(config)
    processes: list[OwnedProcess] = []

    probe = effects.probe_remote_destination(
        config, "owner-token", processes.append, SignalLatch()
    )
    assert probe.verification is None
    assert probe.cleanup_confirmed
    transfer = effects.transfer_remote_snapshot(
        config,
        local_destination,
        local_verification,
        "owner-token",
        processes.append,
        SignalLatch(),
    )
    assert transfer.cleanup_confirmed
    assert transfer.installed
    assert transfer.verification == local_verification
    assert verify_snapshot(remote_destination, config.model) == local_verification
    assert_test_tree_is_frozen(remote_destination)
    assert [process.host_name for process in processes] == ["fwuff", "fwuff"]


def test_real_fake_ssh_shared_remote_source_copies_only_to_local_destination(
    tmp_path: Path,
) -> None:
    ssh, remote_python = write_fake_transport(tmp_path, remote_host_name="fwuff")
    model = make_model()
    remote_source = write_snapshot(tmp_path / "remote" / model.directory_name)
    (remote_source / ".exo-huggingface-revision.json").unlink()
    binding = write_model_contract(tmp_path / "model-contract.json", remote_source)
    freeze_test_tree(remote_source)
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=binding,
        remote_destination=remote_source,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_source,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    Path(config.local_destination).parent.mkdir(parents=True)
    effects = make_system_effects(config)

    result = run_staging(config, effects, owner_token_factory=lambda: "token")

    assert result["status"] == "completed", result["error"]
    assert result["local_installed"] is True
    assert result["remote_preexisting"] is True
    assert result["remote_installed"] is False
    assert_test_tree_is_frozen(Path(config.local_destination))
    source_verification = verify_snapshot(
        remote_source, model, source_model_contract=binding
    )
    assert (
        verify_snapshot(
            Path(config.local_destination), model, source_model_contract=binding
        )
        == source_verification
    )
    assert (result_directory / "model-stage-remote-probe.log").is_file()
    assert (result_directory / "model-stage-remote-reverify.log").is_file()
    assert not (result_directory / "model-stage-remote-transfer.log").exists()


@pytest.mark.parametrize("mutable_entry", ["root", "directory", "file"])
@pytest.mark.parametrize("remote_operation", ["probe", "reverify"])
def test_fake_ssh_shared_source_and_remote_reject_writable_tree(
    tmp_path: Path, mutable_entry: str, remote_operation: str
) -> None:
    ssh, remote_python = write_fake_transport(tmp_path, remote_host_name="fwuff")
    model = make_model()
    remote_source = write_snapshot(tmp_path / "remote" / model.directory_name)
    (remote_source / ".exo-huggingface-revision.json").unlink()
    binding = write_model_contract(tmp_path / "model-contract.json", remote_source)
    freeze_test_tree(remote_source)
    mutable_path = {
        "root": remote_source,
        "directory": remote_source / ".cache",
        "file": remote_source / "config.json",
    }[mutable_entry]
    mutable_path.chmod(0o755 if mutable_path.is_dir() else 0o644)
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=binding,
        remote_destination=remote_source,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_source,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    effects = make_system_effects(config)

    with pytest.raises(StageError, match="shared source snapshot is writable"):
        effects.verify_source_snapshot(remote_source, model, SignalLatch())
    remote_verifier = (
        effects.probe_remote_destination
        if remote_operation == "probe"
        else effects.reverify_remote_destination
    )
    with pytest.raises(OperationError, match="existing remote destination is writable"):
        remote_verifier(config, "owner-token", lambda _process: None, SignalLatch())


@pytest.mark.parametrize("mutation", ["mode", "path_replacement"])
@pytest.mark.parametrize("remote_operation", ["probe", "reverify"])
def test_fake_ssh_remote_verification_rejects_mutation_after_retained_scan(
    tmp_path: Path, mutation: str, remote_operation: str
) -> None:
    if mutation == "mode":
        remote_hook = """
_original_read_retained_relative_file = _read_retained_relative_file
_injected_once = False
def _injected_read_retained_relative_file(tree, relative_path):
    global _injected_once
    contents = _original_read_retained_relative_file(tree, relative_path)
    if not _injected_once:
        _injected_once = True
        os.chmod(tree.directory.path, 0o755)
    return contents
_read_retained_relative_file = _injected_read_retained_relative_file
"""
    else:
        remote_hook = """
_original_read_retained_relative_file = _read_retained_relative_file
_injected_once = False
def _injected_read_retained_relative_file(tree, relative_path):
    global _injected_once
    contents = _original_read_retained_relative_file(tree, relative_path)
    if not _injected_once:
        _injected_once = True
        target = tree.directory.path / 'config.json'
        replacement_contents = target.read_bytes()
        target.rename(tree.directory.path / 'config.original')
        target.write_bytes(replacement_contents)
        target.chmod(0o444)
    return contents
_read_retained_relative_file = _injected_read_retained_relative_file
"""
    ssh, remote_python = write_fake_transport(
        tmp_path, remote_host_name="fwuff", remote_hook=remote_hook
    )
    model = make_model()
    remote_source = write_snapshot(tmp_path / "remote" / model.directory_name)
    (remote_source / ".exo-huggingface-revision.json").unlink()
    binding = write_model_contract(tmp_path / "model-contract.json", remote_source)
    freeze_test_tree(remote_source)
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=binding,
        remote_destination=remote_source,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_source,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )

    effects = make_system_effects(config)
    remote_verifier = (
        effects.probe_remote_destination
        if remote_operation == "probe"
        else effects.reverify_remote_destination
    )
    with pytest.raises(OperationError, match="existing remote destination"):
        remote_verifier(config, "owner-token", lambda _process: None, SignalLatch())


def test_remote_probe_uses_embedded_contract_bytes_not_a_local_contract_path(
    tmp_path: Path,
) -> None:
    ssh, remote_python = write_fake_transport(tmp_path, remote_host_name="fwuff")
    model = make_model()
    remote_source = write_snapshot(
        tmp_path / "simulated-fwuff-filesystem" / model.directory_name
    )
    (remote_source / ".exo-huggingface-revision.json").unlink()
    local_contract_directory = tmp_path / "dwagon-only-contract-filesystem"
    local_contract_directory.mkdir()
    contract_path = local_contract_directory / "model-contract.json"
    binding = write_model_contract(contract_path, remote_source)
    freeze_test_tree(remote_source)
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=binding,
        remote_destination=remote_source,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_source,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    effects = make_system_effects(config)
    request = effects._remote_request(config, "owner-token", "probe")
    serialized_request = request.model_dump_json()
    assert str(contract_path) not in serialized_request
    assert request.source_model_contract is not None
    contract_path.unlink()

    result = effects._run_remote(
        config,
        request,
        config.timeouts.remote_probe_seconds,
        lambda _process: None,
        SignalLatch(),
    )

    assert result.cleanup_confirmed is True
    assert result.verification is not None
    assert result.verification.model_contract_sha256 == binding.sha256


def test_remote_identity_partial_line_obeys_timeout(tmp_path: Path) -> None:
    effects = make_system_effects(make_config(tmp_path))
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time;sys.stdout.write('{');sys.stdout.flush();time.sleep(5)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        with pytest.raises(StageError, match="timed out"):
            effects._read_protocol_line(process, 0.05, 0.005, SignalLatch())
        assert time.monotonic() - started < 0.5
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_remote_final_receipt_is_recovered_when_registration_fails(
    tmp_path: Path,
) -> None:
    ssh, remote_python = write_fake_transport(tmp_path, remote_host_name="fwuff")
    model = make_model()
    remote_destination = tmp_path / "remote" / model.directory_name
    config = remap_runtime_config(
        make_config(tmp_path),
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_destination,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    remote_destination.parent.mkdir(parents=True)
    effects = make_system_effects(config)

    def reject_registration(_process: OwnedProcess) -> None:
        raise StageError("injected registration failure")

    with pytest.raises(OperationError, match="registration failure") as raised:
        effects.probe_remote_destination(
            config, "owner-token", reject_registration, SignalLatch()
        )
    assert raised.value.cleanup_confirmed is True


def test_existing_nonempty_destination_is_verified_without_overwrite(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    snapshot = write_snapshot(Path(config.local_destination))
    freeze_test_tree(snapshot)
    effects = make_system_effects(config)
    assert effects.inspect_local_destination(
        snapshot, config.model, SignalLatch()
    ) == verify_snapshot(snapshot, config.model)


def test_existing_empty_destination_is_never_replaced(tmp_path: Path) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = Path(config.local_destination)
    destination.mkdir(parents=True)
    freeze_test_tree(destination)
    effects = make_system_effects(config)
    with pytest.raises(StageError, match="pre-existing empty"):
        effects.inspect_local_destination(destination, config.model, SignalLatch())


def test_existing_writable_local_destination_is_rejected(tmp_path: Path) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = write_snapshot(Path(config.local_destination))
    freeze_test_tree(destination)
    (destination / "config.json").chmod(0o644)

    with pytest.raises(StageError, match="existing local destination is writable"):
        make_system_effects(config).inspect_local_destination(
            destination, config.model, SignalLatch()
        )


def test_owned_temp_creation_refuses_collision_and_cleanup_refuses_foreign_path(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    collision = destination.parent / "owned.stage"
    collision.mkdir()
    effects = make_system_effects(config)
    with pytest.raises(StageError, match="already exists"):
        effects.create_local_temporary(destination, collision.name)
    assert effects.cleanup_local_temporary(collision) is False
    assert collision.exists()

    owned = effects.create_local_temporary(destination, "retained.stage")
    (owned / "must-survive").write_text("retained", encoding="utf-8")
    moved = destination.parent / "moved.stage"
    owned.rename(moved)
    assert effects.cleanup_local_temporary(owned) is False
    assert (moved / "must-survive").read_text(encoding="utf-8") == "retained"


def test_owned_temp_open_failure_never_claims_failed_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    name = "open-failure.stage"
    real_open = os.open
    real_rmdir = os.rmdir

    def fail_retaining_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == name and flags & os.O_DIRECTORY and not flags & os.O_PATH:
            raise PermissionError("injected retained-open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def fail_cleanup_rmdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        if path == name:
            raise PermissionError("injected cleanup failure")
        real_rmdir(path, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(model_stage.os, "open", fail_retaining_open)
        patch.setattr(model_stage.os, "rmdir", fail_cleanup_rmdir)
        with pytest.raises(OperationError) as raised:
            effects.create_local_temporary(destination, name)
    assert raised.value.cleanup_confirmed is False
    assert (destination.parent / name).is_dir()
    (destination.parent / name).rmdir()


def test_atomic_rename_records_publication_before_pending_signal_is_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    publication_recorded = False
    handler_observed_publication = False
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGTERM,))

    def record_publication() -> None:
        nonlocal publication_recorded
        publication_recorded = True

    def handle_signal(signal_number: int, _frame: object) -> None:
        nonlocal handler_observed_publication
        handler_observed_publication = publication_recorded
        raise model_stage.ManagedSignalError(signal_number)

    def rename_and_signal(
        source_descriptor: int,
        source_name: bytes,
        destination_descriptor: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        assert flags == 1
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source_descriptor,
            dst_dir_fd=destination_descriptor,
        )
        os.kill(os.getpid(), signal.SIGTERM)
        return 0

    class FakeLibrary:
        def __init__(self) -> None:
            self.renameat2 = rename_and_signal

    try:
        signal.signal(signal.SIGTERM, handle_signal)
        with monkeypatch.context() as patch:
            patch.setattr(
                model_stage.ctypes,
                "CDLL",
                lambda *_arguments, **_keywords: FakeLibrary(),
            )
            with pytest.raises(model_stage.ManagedSignalError, match="managed signal"):
                model_stage._rename_entry_noreplace_at(
                    parent_descriptor,
                    source.name,
                    destination.name,
                    record_publication,
                )
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except model_stage.ManagedSignalError:
            pass
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            os.close(parent_descriptor)

    assert publication_recorded is True
    assert handler_observed_publication is True
    assert not source.exists()
    assert destination.is_dir()


def test_cleanup_quarantine_preserves_a_foreign_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "racy.stage")
    moved = destination.parent / "moved-racy.stage"
    real_rename = model_stage._rename_entry_noreplace_at
    raced = False

    def race_rename(
        parent_descriptor: int, source_name: str, destination_name: str
    ) -> None:
        nonlocal raced
        if source_name == temporary.name and not raced:
            raced = True
            os.rename(
                temporary.name,
                moved.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.mkdir(temporary.name, dir_fd=parent_descriptor)
            replacement = os.open(
                temporary.name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=parent_descriptor,
            )
            try:
                descriptor = os.open(
                    "foreign",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement,
                )
                os.write(descriptor, b"foreign")
                os.close(descriptor)
            finally:
                os.close(replacement)
        real_rename(parent_descriptor, source_name, destination_name)

    with monkeypatch.context() as patch:
        patch.setattr(model_stage, "_rename_entry_noreplace_at", race_rename)
        assert effects.cleanup_local_temporary(temporary) is False
    assert raced
    assert moved.is_dir()
    assert (temporary / "foreign").read_text(encoding="utf-8") == "foreign"
    assert not any(moved.iterdir())


def test_cleanup_preserves_a_complete_tree_and_requires_a_tombstone(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "journaled.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)

    assert effects.cleanup_local_temporary(temporary) is False
    assert verify_snapshot(temporary, config.model) == verify_snapshot(
        source, config.model
    )


def test_cleanup_fails_when_a_journaled_nested_directory_was_moved(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "nested-move.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    nested = temporary / ".cache"
    moved = destination.parent / "moved-cache"
    nested.rename(moved)
    nested.mkdir()
    (nested / "foreign").write_text("foreign", encoding="utf-8")

    assert effects.cleanup_local_temporary(temporary) is False
    assert (nested / "foreign").read_text(encoding="utf-8") == "foreign"
    assert (moved / "huggingface" / "metadata").read_text(encoding="utf-8") == (
        "cached\n"
    )


def test_cleanup_preserves_a_regular_file_replaced_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "nested-race.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    real_validate = model_stage._validate_owned_tree
    raced = False

    def validate_then_replace(directory: model_stage.OwnedDirectory) -> bool:
        nonlocal raced
        validated = real_validate(directory)
        if validated and not raced:
            raced = True
            os.rename(
                "config.json",
                "displaced-config.json",
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
            )
            descriptor = os.open(
                "config.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory.descriptor,
            )
            os.write(descriptor, b"foreign")
            os.close(descriptor)
        return validated

    with monkeypatch.context() as patch:
        patch.setattr(model_stage, "_validate_owned_tree", validate_then_replace)
        assert effects.cleanup_local_temporary(temporary) is False

    assert raced
    assert (temporary / "config.json").read_bytes() == b"foreign"
    assert (temporary / "displaced-config.json").read_text(encoding="utf-8") == ("{}\n")


def test_cleanup_preserves_a_nested_directory_replaced_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "nested-dir-race.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    real_validate = model_stage._validate_owned_tree
    raced = False

    def validate_then_replace(directory: model_stage.OwnedDirectory) -> bool:
        nonlocal raced
        validated = real_validate(directory)
        if validated and not raced:
            raced = True
            os.rename(
                ".cache",
                "displaced-cache",
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
            )
            os.mkdir(".cache", dir_fd=directory.descriptor)
            replacement = os.open(
                ".cache",
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=directory.descriptor,
            )
            try:
                descriptor = os.open(
                    "foreign",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement,
                )
                os.write(descriptor, b"foreign")
                os.close(descriptor)
            finally:
                os.close(replacement)
        return validated

    with monkeypatch.context() as patch:
        patch.setattr(model_stage, "_validate_owned_tree", validate_then_replace)
        assert effects.cleanup_local_temporary(temporary) is False

    assert raced
    assert (temporary / ".cache" / "foreign").read_bytes() == b"foreign"
    assert (temporary / "displaced-cache" / "huggingface" / "metadata").read_text(
        encoding="utf-8"
    ) == "cached\n"


def test_cleanup_preserves_a_root_quarantine_replaced_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "root-race.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    moved = destination.parent / "displaced-root-quarantine"
    real_validate = model_stage._validate_owned_tree
    raced = False

    def validate_then_replace(directory: model_stage.OwnedDirectory) -> bool:
        nonlocal raced
        validated = real_validate(directory)
        if validated and not raced:
            raced = True
            quarantine_name = directory.path.name
            os.rename(
                quarantine_name,
                moved.name,
                src_dir_fd=directory.parent_descriptor,
                dst_dir_fd=directory.parent_descriptor,
            )
            os.mkdir(quarantine_name, dir_fd=directory.parent_descriptor)
            replacement = os.open(
                quarantine_name,
                os.O_RDONLY | os.O_DIRECTORY,
                dir_fd=directory.parent_descriptor,
            )
            try:
                descriptor = os.open(
                    "foreign",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=replacement,
                )
                os.write(descriptor, b"foreign")
                os.close(descriptor)
            finally:
                os.close(replacement)
        return validated

    with monkeypatch.context() as patch:
        patch.setattr(model_stage, "_validate_owned_tree", validate_then_replace)
        assert effects.cleanup_local_temporary(temporary) is False

    assert raced
    assert (temporary / "foreign").read_bytes() == b"foreign"
    assert verify_snapshot(moved, config.model) == verify_snapshot(source, config.model)


def test_snapshot_hashing_observes_cooperative_interrupts(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x" * (32 << 20))
    checkpoints = 0

    def checkpoint() -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 20:
            raise StageError("injected cooperative interruption")

    with pytest.raises(StageError, match="cooperative interruption"):
        verify_snapshot(snapshot, make_model(), checkpoint)
    assert checkpoints == 20


def test_no_replace_install_refuses_destination_that_appears(tmp_path: Path) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "owned.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    destination.mkdir()
    (destination / "foreign").write_text("foreign", encoding="utf-8")
    with pytest.raises(StageError, match="destination appeared"):
        effects.install_local_snapshot(temporary, destination, lambda: None)
    assert (destination / "foreign").read_text(encoding="utf-8") == "foreign"
    assert (temporary / "config.json").read_text(encoding="utf-8") == "{}\n"


def test_install_validates_modes_through_retained_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "owned.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    real_fchmod = os.fchmod
    skipped_regular = False

    def skip_one_regular_mode(descriptor: int, mode: int) -> None:
        nonlocal skipped_regular
        if mode == 0o444 and not skipped_regular:
            skipped_regular = True
            return
        real_fchmod(descriptor, mode)

    with monkeypatch.context() as patch:
        patch.setattr(model_stage.os, "fchmod", skip_one_regular_mode)
        with pytest.raises(StageError, match="identity or mode validation failed"):
            effects.install_local_snapshot(temporary, destination, lambda: None)

    assert skipped_regular is True
    assert destination.exists() is False
    assert any(
        path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        for path in temporary.rglob("*")
        if path.is_file()
    )


def test_install_retains_nested_journal_until_post_rename_verification(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    temporary = effects.create_local_temporary(destination, "owned.stage")
    source = write_snapshot(tmp_path / "source")
    effects.copy_preverified_snapshot(source, temporary, config.model, SignalLatch())
    effects.write_revision_receipt(temporary, config.model)
    effects.install_local_snapshot(temporary, destination, lambda: None)
    assert_test_tree_is_frozen(destination)
    moved = destination.parent / "moved-installed-cache"
    (destination / ".cache").rename(moved)
    (destination / ".cache").mkdir()
    (destination / ".cache" / "foreign").write_text("foreign", encoding="utf-8")

    with pytest.raises(StageError, match="creation journal"):
        effects.verify_local_snapshot(destination, config.model, SignalLatch())

    assert (destination / ".cache" / "foreign").read_text(encoding="utf-8") == (
        "foreign"
    )
    assert (moved / "huggingface" / "metadata").read_text(encoding="utf-8") == (
        "cached\n"
    )


def test_run_staging_records_local_install_after_post_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_snapshot(tmp_path / "source")
    result_directory = tmp_path / "results"
    config = make_config(
        tmp_path,
        source_snapshot=source,
        result_directory=result_directory,
    )
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    validate_frozen_owned_tree = model_stage._validate_frozen_owned_tree

    def fail_only_after_publication(directory: model_stage.OwnedDirectory) -> bool:
        if directory.path == destination:
            return False
        return validate_frozen_owned_tree(directory)

    with monkeypatch.context() as patch:
        patch.setattr(
            model_stage,
            "_validate_frozen_owned_tree",
            fail_only_after_publication,
        )
        result = run_staging(
            config, effects, owner_token_factory=lambda: "post-rename-local"
        )

    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["local_installed"] is True
    assert result["remote_installed"] is False
    assert "durably frozen" in cast(str, result["error"])
    assert destination.is_dir()
    assert_test_tree_is_frozen(destination)
    assert not any(
        path.name.endswith(".stage") for path in destination.parent.iterdir()
    )


def test_run_staging_records_local_install_when_parent_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_snapshot(tmp_path / "source")
    result_directory = tmp_path / "results"
    config = make_config(
        tmp_path,
        source_snapshot=source,
        result_directory=result_directory,
    )
    destination = Path(config.local_destination)
    destination.parent.mkdir()
    effects = make_system_effects(config)
    parent_status = destination.parent.stat()
    real_fsync = os.fsync
    failed_parent_fsync = False

    def fail_first_post_publication_parent_fsync(descriptor: int) -> None:
        nonlocal failed_parent_fsync
        observed = os.fstat(descriptor)
        if (
            not failed_parent_fsync
            and destination.is_dir()
            and (observed.st_dev, observed.st_ino)
            == (parent_status.st_dev, parent_status.st_ino)
        ):
            failed_parent_fsync = True
            raise OSError(errno.EIO, "injected parent fsync failure")
        real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(model_stage.os, "fsync", fail_first_post_publication_parent_fsync)
        result = run_staging(
            config, effects, owner_token_factory=lambda: "parent-fsync-local"
        )

    assert failed_parent_fsync is True
    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["local_installed"] is True
    assert result["remote_installed"] is False
    assert "injected parent fsync failure" in cast(str, result["error"])
    assert destination.is_dir()
    assert_test_tree_is_frozen(destination)
    assert not any(
        path.name.endswith(".stage") for path in destination.parent.iterdir()
    )


def test_run_staging_records_remote_install_from_failed_post_rename_receipt(
    tmp_path: Path,
) -> None:
    remote_hook = """
_original_validate_frozen_owned_tree = _validate_frozen_owned_tree
def _fail_only_after_publication(directory):
    if not directory.path.name.endswith('.stage'):
        return False
    return _original_validate_frozen_owned_tree(directory)
_validate_frozen_owned_tree = _fail_only_after_publication
"""
    ssh, remote_python = write_fake_transport(
        tmp_path, remote_host_name="fwuff", remote_hook=remote_hook
    )
    source = write_snapshot(tmp_path / "source")
    remote_destination = tmp_path / "remote" / make_model().directory_name
    remote_destination.parent.mkdir()
    result_directory = tmp_path / "results"
    config = make_config(
        tmp_path,
        source_snapshot=source,
        remote_destination=remote_destination,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_destination,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    local_destination = Path(config.local_destination)
    local_destination.parent.mkdir()
    effects = make_system_effects(config)

    result = run_staging(
        config, effects, owner_token_factory=lambda: "post-rename-remote"
    )

    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["local_installed"] is True
    assert result["remote_installed"] is True
    assert "installed remote model tree is not durably frozen" in cast(
        str, result["error"]
    )
    assert remote_destination.is_dir()
    assert_test_tree_is_frozen(remote_destination)
    assert verify_snapshot(remote_destination, config.model) == verify_snapshot(
        local_destination, config.model
    )
    assert not any(
        path.name.endswith(".stage") for path in remote_destination.parent.iterdir()
    )


def test_run_staging_records_remote_install_when_parent_fsync_fails(
    tmp_path: Path,
) -> None:
    source = write_snapshot(tmp_path / "source")
    remote_destination = tmp_path / "remote" / make_model().directory_name
    remote_destination.parent.mkdir()
    remote_hook = f"""
_original_fsync = os.fsync
_destination = Path({str(remote_destination)!r})
_parent_fsync_failed = False
def _fail_first_post_publication_parent_fsync(descriptor):
    global _parent_fsync_failed
    observed = os.fstat(descriptor)
    parent = os.stat(_destination.parent)
    if (
        not _parent_fsync_failed
        and _destination.is_dir()
        and (observed.st_dev, observed.st_ino) == (parent.st_dev, parent.st_ino)
    ):
        _parent_fsync_failed = True
        raise OSError(errno.EIO, 'injected remote parent fsync failure')
    _original_fsync(descriptor)
os.fsync = _fail_first_post_publication_parent_fsync
"""
    ssh, remote_python = write_fake_transport(
        tmp_path, remote_host_name="fwuff", remote_hook=remote_hook
    )
    result_directory = tmp_path / "results"
    config = make_config(
        tmp_path,
        source_snapshot=source,
        remote_destination=remote_destination,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_destination,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    local_destination = Path(config.local_destination)
    local_destination.parent.mkdir()
    effects = make_system_effects(config)

    result = run_staging(
        config, effects, owner_token_factory=lambda: "parent-fsync-remote"
    )

    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["local_installed"] is True
    assert result["remote_installed"] is True
    assert "injected remote parent fsync failure" in cast(str, result["error"])
    assert remote_destination.is_dir()
    assert_test_tree_is_frozen(remote_destination)
    assert verify_snapshot(remote_destination, config.model) == verify_snapshot(
        local_destination, config.model
    )
    assert not any(
        path.name.endswith(".stage") for path in remote_destination.parent.iterdir()
    )


def test_run_staging_records_remote_install_when_signal_follows_rename(
    tmp_path: Path,
) -> None:
    source = write_snapshot(tmp_path / "source")
    remote_destination = tmp_path / "remote" / make_model().directory_name
    remote_destination.parent.mkdir()
    remote_hook = """
_original_cdll = ctypes.CDLL
class _SignalAfterRenameLibrary:
    def __init__(self, *arguments, **keywords):
        self._library = _original_cdll(*arguments, **keywords)
    def __getattr__(self, name):
        return getattr(self._library, name)
    def renameat2(self, *arguments):
        result = self._library.renameat2(*arguments)
        if result == 0:
            os.kill(os.getpid(), signal.SIGTERM)
        return result
ctypes.CDLL = _SignalAfterRenameLibrary
"""
    ssh, remote_python = write_fake_transport(
        tmp_path, remote_host_name="fwuff", remote_hook=remote_hook
    )
    result_directory = tmp_path / "results"
    config = make_config(
        tmp_path,
        source_snapshot=source,
        remote_destination=remote_destination,
        result_directory=result_directory,
    )
    config = remap_runtime_config(
        config,
        local_host_name="dwagon",
        remote_host_name="fwuff",
        remote_destination=remote_destination,
        ssh_executable=ssh,
        remote_python_executable=remote_python,
    )
    local_destination = Path(config.local_destination)
    local_destination.parent.mkdir()
    effects = make_system_effects(config)

    result = run_staging(
        config, effects, owner_token_factory=lambda: "rename-signal-remote"
    )

    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["local_installed"] is True
    assert result["remote_installed"] is True
    assert "received managed signal" in cast(str, result["error"])
    assert remote_destination.is_dir()
    assert_test_tree_is_frozen(remote_destination)
    assert verify_snapshot(remote_destination, config.model) == verify_snapshot(
        local_destination, config.model
    )
    assert not any(
        path.name.endswith(".stage") for path in remote_destination.parent.iterdir()
    )


def test_run_staging_downloads_transfers_and_reconciles_processes(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    result = run_staging(config, effects, owner_token_factory=lambda: "token")
    assert result["status"] == "completed"
    assert result["reportable"] is False
    assert result["performance_comparable"] is False
    assert result["performance_claim"] is None
    assert result["acquisition"] == "huggingface_cli"
    assert result["local_installed"] is True
    assert result["remote_installed"] is True
    assert effects.calls == [
        "inspect-local",
        "create-local-temporary",
        "download",
        "write-receipt",
        "verify-local",
        "install-local",
        "verify-local",
        "probe-remote",
        "transfer-remote",
    ]
    runtime_fragments = [
        value for filename, value in effects.fragments if filename == RUNTIME_METADATA
    ]
    result_fragment = next(
        value for filename, value in effects.fragments if filename == BENCHMARK_RESULT
    )
    assert [
        len(cast(list[object], fragment["owned_processes"]))
        for fragment in runtime_fragments
    ] == [0, 1, 2, 3]
    assert (
        result_fragment["owned_processes"] == runtime_fragments[-1]["owned_processes"]
    )
    assert all(
        process["owner_token"] == "smol-stage-test:token"
        for process in cast(list[dict[str, object]], result_fragment["owned_processes"])
    )


def test_run_staging_uses_preverified_source_instead_of_hf(tmp_path: Path) -> None:
    source = tmp_path / "source"
    config = make_config(tmp_path, source_snapshot=source)
    effects = FakeEffects(config)
    result = run_staging(config, effects)
    assert result["status"] == "completed"
    assert result["acquisition"] == "preverified_snapshot_copy"
    assert "copy-source" in effects.calls
    assert "download" not in effects.calls


def make_shared_remote_source_config(tmp_path: Path) -> StageConfig:
    remote_source = tmp_path / "remote" / make_model().directory_name
    return make_config(
        tmp_path,
        source_snapshot=remote_source,
        acquisition_mode="local_copy_from_existing_remote_source",
        source_model_contract=ModelContractBinding(
            path=str(tmp_path / "model-contract.json"), sha256="a" * 64
        ),
        remote_destination=remote_source,
    )


def test_shared_remote_source_is_probed_copied_and_reverified_without_transfer(
    tmp_path: Path,
) -> None:
    config = make_shared_remote_source_config(tmp_path)
    verification = make_verification("a")
    effects = FakeEffects(config, remote_existing=verification)
    effects.remote_reverification = verification
    result = run_staging(config, effects, owner_token_factory=lambda: "token")

    assert result["status"] == "completed"
    assert result["acquisition"] == "local_copy_from_existing_remote_source"
    assert result["local_installed"] is True
    assert result["remote_preexisting"] is True
    assert result["remote_installed"] is False
    assert result["source_verification"] == verification.model_dump(mode="json")
    assert effects.calls == [
        "verify-source",
        "probe-remote",
        "inspect-local",
        "create-local-temporary",
        "copy-source",
        "verify-local",
        "install-local",
        "verify-local",
        "reverify-remote",
    ]
    assert "transfer-remote" not in effects.calls


@pytest.mark.parametrize("failure", ["missing", "initial_mismatch", "changed"])
def test_shared_remote_source_fails_closed_without_remote_transfer(
    tmp_path: Path, failure: str
) -> None:
    config = make_shared_remote_source_config(tmp_path)
    initial = None if failure == "missing" else make_verification("a")
    if failure == "initial_mismatch":
        initial = make_verification("b")
    effects = FakeEffects(config, remote_existing=initial)
    effects.remote_reverification = (
        make_verification("c") if failure == "changed" else make_verification("a")
    )
    result = run_staging(config, effects)

    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert result["remote_installed"] is False
    assert result["remote_preexisting"] is (failure != "missing")
    assert "transfer-remote" not in effects.calls
    if failure != "changed":
        assert "create-local-temporary" not in effects.calls


def test_remote_preexisting_is_unknown_when_failure_precedes_remote_observation(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.download_error = OperationError(
        "injected acquisition failure", cleanup_confirmed=True
    )

    result = run_staging(config, effects)

    assert result["status"] == "staging_failed"
    assert result["remote_preexisting"] is None
    assert "probe-remote" not in effects.calls


def test_run_staging_preserves_matching_existing_destinations(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    verification = make_verification("a")
    effects = FakeEffects(
        config, local_existing=verification, remote_existing=verification
    )
    result = run_staging(config, effects)
    assert result["status"] == "completed"
    assert result["local_installed"] is False
    assert result["remote_installed"] is False
    assert "create-local-temporary" not in effects.calls
    assert "transfer-remote" not in effects.calls


def test_manifest_mismatch_fails_without_a_performance_result(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.remote_transfer_verification = make_verification("c")
    result = run_staging(config, effects)
    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["reportable"] is False
    assert "manifests differ" in cast(str, result["error"])


def test_process_cleanup_failure_fails_closed_and_reconciles_identity(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.download_error = OperationError(
        "download timed out", cleanup_confirmed=False
    )
    result = run_staging(config, effects)
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False
    runtime = [
        value for filename, value in effects.fragments if filename == RUNTIME_METADATA
    ][-1]
    assert result["owned_processes"] == runtime["owned_processes"]
    assert effects.calls[-1] == "cleanup-local-temporary"


def test_owned_temporary_cleanup_failure_fails_closed(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.download_error = OperationError("download failed", cleanup_confirmed=True)
    effects.local_temporary_cleanup = False
    result = run_staging(config, effects)
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False


def test_remote_cleanup_failure_fails_closed_with_remote_identity(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.remote_transfer_error = OperationError(
        "remote helper cleanup unconfirmed", cleanup_confirmed=False
    )
    result = run_staging(config, effects)
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False
    processes = cast(list[dict[str, object]], result["owned_processes"])
    assert [process["host_name"] for process in processes] == [
        "dwagon",
        "fwuff",
        "fwuff",
    ]


def test_probe_without_remote_final_receipt_fails_cleanup_closed(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    effects.remote_probe_error = OperationError(
        "probe transport ended without the remote final receipt",
        cleanup_confirmed=False,
    )
    result = run_staging(config, effects)
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False
    processes = cast(list[dict[str, object]], result["owned_processes"])
    assert [process["host_name"] for process in processes] == ["dwagon", "fwuff"]
    runtime = [
        value for filename, value in effects.fragments if filename == RUNTIME_METADATA
    ][-1]
    assert result["owned_processes"] == runtime["owned_processes"]


def test_remote_cleanup_reconciliation_requires_a_validated_final_receipt() -> None:
    assert reconcile_remote_cleanup(True, None) is False
    assert reconcile_remote_cleanup(False, None) is False
    failed_cleanup = RemoteResponse(
        schema_version=1,
        kind="result",
        status="failed",
        verification=None,
        installed=False,
        cleanup_succeeded=False,
        error="cleanup failed",
    )
    assert reconcile_remote_cleanup(True, failed_cleanup) is False
    confirmed_failure = failed_cleanup.model_copy(update={"cleanup_succeeded": True})
    assert reconcile_remote_cleanup(True, confirmed_failure) is True


def test_signal_before_staging_emits_cleanup_confirmed_result(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    latch = SignalLatch(signal_number=15)
    result = run_staging(config, effects, latch)
    assert result["status"] == "staging_failed"
    assert result["cleanup_succeeded"] is True
    assert result["interrupted_signal"] == 15
    assert cast(list[object], result["owned_processes"]) == []


def test_prepare_lease_emits_exact_validated_wrapper_contract(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    result_root.mkdir()
    config = make_config(tmp_path, result_directory=result_root / "smol-stage-test")
    config_path = tmp_path / "stage-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    metadata_path = tmp_path / "stage-metadata.json"
    staging_script = Path(__file__).parents[1] / "two_host_model_stage.py"
    wrapper_script = Path(__file__).parents[1] / "benchmark_lease.py"
    python = Path(sys.executable)

    prepared = prepare_lease_metadata(
        config_path=config_path,
        metadata_output=metadata_path,
        wrapper_python=python,
        child_python=python,
        benchmark_lease_script=wrapper_script,
        staging_script=staging_script,
        owner="codex:test-owner",
        purpose="stage exact model",
        expected_duration_seconds=60.0,
        heartbeat_seconds=1.0,
        cleanup_grace_seconds=300.0,
        lease_path=tmp_path / "lease.json",
        lock_path=tmp_path / "lease.lock",
        result_root=result_root,
        identity_reader=lambda _path: config.lease_metadata.git,
    )

    assert read_json_object(metadata_path) == prepared.metadata
    assert prepared.metadata["model_staging"] == staging_metadata_contract(config)
    assert prepared.metadata["gpu_bindings"] == {
        "dwagon": [],
        "fwuff": [],
    }
    assert prepared.metadata["hca_bindings"] == {
        host: [
            {"device": "mlx4_0", "port": port, "gid": gid}
            for port, gid in enumerate(current_gids, start=1)
        ]
        for host, current_gids in CURRENT_HCA_GIDS.items()
    }
    assert prepared.child_argv == tuple(cast(list[str], prepared.metadata["command"]))
    assert "--owner=codex:test-owner" in prepared.benchmark_lease_argv
    assert "--purpose=stage exact model" in prepared.benchmark_lease_argv
    assert prepared.minimum_cleanup_grace_seconds == minimum_cleanup_grace_seconds(
        config
    )
    with pytest.raises(StageError, match="already exists"):
        prepare_lease_metadata(
            config_path=config_path,
            metadata_output=metadata_path,
            wrapper_python=python,
            child_python=python,
            benchmark_lease_script=wrapper_script,
            staging_script=staging_script,
            owner="codex:test-owner",
            purpose="stage exact model",
            expected_duration_seconds=60.0,
            heartbeat_seconds=1.0,
            cleanup_grace_seconds=300.0,
            lease_path=tmp_path / "lease.json",
            lock_path=tmp_path / "lease.lock",
            result_root=result_root,
            identity_reader=lambda _path: config.lease_metadata.git,
        )


def make_lease_record(
    config: StageConfig,
    config_path: Path,
    lease_path: Path,
    lock_path: Path,
    *,
    child_pid: int = 101,
    parent_pid: int = 202,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    observed = timestamp or datetime.now(timezone.utc)
    lease_values = config.lease_metadata.model_dump(mode="json", exclude_none=True)
    command = [
        "/usr/bin/python3",
        str(Path(__file__).parents[1] / "two_host_model_stage.py"),
        "--config",
        str(config_path),
        "--lease-path",
        str(lease_path),
        "--lock-path",
        str(lock_path),
        "--result-dir",
        config.result_directory,
    ]
    metadata = {
        "schema_version": 1,
        "generated_at": observed.isoformat(),
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": [53001],
        "result_directory": config.result_directory,
        "command": command,
        "hosts": [config.local_host_name, config.remote_host_name],
        "models": [
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "paths": {
                    config.local_host_name: config.local_destination,
                    config.remote_host_name: config.remote_destination,
                },
            }
        ],
        "git": lease_values["git"],
        "gpu_bindings": lease_values["gpu_bindings"],
        "cpu_bindings": lease_values["cpu_bindings"],
        "hca_bindings": lease_values["hca_bindings"],
        "source_deployments": lease_values["source_deployments"],
        "owner_pids": lease_values["owner_pids"],
        "model_staging": staging_metadata_contract(config),
    }
    return {
        "lease_id": "lease-id",
        "run_id": config.run_id,
        "exo_namespace": config.namespace,
        "result_directory": config.result_directory,
        "wrapper_pid": parent_pid,
        "child_pid": child_pid,
        "command": command,
        "ports": [53001],
        "heartbeat": observed.isoformat(),
        "cleanup_grace_seconds": 300.0,
        "child_cleanup_confirmation_required": True,
        "metadata": metadata,
    }


def test_active_lease_binds_lock_parent_child_command_and_model_metadata(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config_path = tmp_path / "config.json"
    lease_path = tmp_path / "lease.json"
    lock_path = tmp_path / "lease.lock"
    lease_path.write_text(
        json.dumps(make_lease_record(config, config_path, lease_path, lock_path)),
        encoding="utf-8",
    )
    record = validate_active_lease(
        config,
        config_path=config_path,
        lease_path=lease_path,
        lock_path=lock_path,
        process_id=101,
        parent_process_id=202,
        lock_is_held=lambda path: path == lock_path,
    )
    assert record["child_cleanup_confirmation_required"] is True


def mutate_child(record: dict[str, object]) -> None:
    record["child_pid"] = 999


def mutate_parent(record: dict[str, object]) -> None:
    record["wrapper_pid"] = 999


def mutate_staging_contract(record: dict[str, object]) -> None:
    cast(dict[str, object], record["metadata"])["model_staging"] = {}


def mutate_cleanup_contract(record: dict[str, object]) -> None:
    record["child_cleanup_confirmation_required"] = False


def mutate_top_level_resource_contract(record: dict[str, object]) -> None:
    metadata = cast(dict[str, object], record["metadata"])
    gpu_bindings = cast(dict[str, object], metadata["gpu_bindings"])
    gpu_bindings["dwagon"] = [{"uuid": "unexpected", "pci_address": "0000:00:00.0"}]


LEASE_MUTATIONS: list[tuple[Callable[[dict[str, object]], None], str]] = [
    (mutate_child, "different child"),
    (mutate_parent, "wrapper_pid"),
    (mutate_staging_contract, "model_staging"),
    (mutate_cleanup_contract, "child_cleanup_confirmation_required"),
    (mutate_top_level_resource_contract, "gpu_bindings"),
]


@pytest.mark.parametrize(("mutation", "match"), LEASE_MUTATIONS)
def test_active_lease_rejects_mismatched_ownership_and_static_binding(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    config = make_config(tmp_path)
    config_path = tmp_path / "config.json"
    lease_path = tmp_path / "lease.json"
    lock_path = tmp_path / "lease.lock"
    record = make_lease_record(config, config_path, lease_path, lock_path)
    mutation(record)
    lease_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(StageError, match=match):
        validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
            process_id=101,
            parent_process_id=202,
            lock_is_held=lambda path: True,
        )


def test_active_lease_rejects_unheld_lock_and_stale_heartbeat(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config_path = tmp_path / "config.json"
    lease_path = tmp_path / "lease.json"
    lock_path = tmp_path / "lease.lock"
    now = datetime.now(timezone.utc)
    record = make_lease_record(
        config,
        config_path,
        lease_path,
        lock_path,
        timestamp=now - timedelta(minutes=3),
    )
    lease_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(StageError, match="lock is not held"):
        validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
            lock_is_held=lambda path: False,
        )
    with pytest.raises(StageError, match="heartbeat is stale"):
        validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
            process_id=101,
            parent_process_id=202,
            now=lambda: now,
            lock_is_held=lambda path: True,
        )


def test_active_lease_uses_the_canonical_cleanup_grace_bound(tmp_path: Path) -> None:
    payload = make_config(tmp_path).model_dump(mode="json")
    cast(dict[str, object], payload["timeouts"])["cleanup_seconds"] = 100.0
    config = StageConfig.model_validate_json(json.dumps(payload))
    config_path = tmp_path / "config.json"
    lease_path = tmp_path / "lease.json"
    lock_path = tmp_path / "lease.lock"
    record = make_lease_record(config, config_path, lease_path, lock_path)
    assert minimum_cleanup_grace_seconds(config) > 300.0
    lease_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(StageError, match="cleanup grace"):
        validate_active_lease(
            config,
            config_path=config_path,
            lease_path=lease_path,
            lock_path=lock_path,
            process_id=101,
            parent_process_id=202,
            lock_is_held=lambda _path: True,
        )


def test_remote_request_requires_owned_sibling_temporary(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ValidationError, match="distinct sibling"):
        RemoteRequest(
            schema_version=1,
            operation="receive",
            run_id=config.run_id,
            namespace=config.namespace,
            owner_token="token",
            expected_host_name=config.remote_host_name,
            destination=config.remote_destination,
            temporary_path="/tmp/foreign.stage",
            model=config.model,
        )
    valid = RemoteRequest(
        schema_version=1,
        operation="receive",
        run_id=config.run_id,
        namespace=config.namespace,
        owner_token="token",
        expected_host_name=config.remote_host_name,
        destination=config.remote_destination,
        temporary_path=str(
            Path(config.remote_destination).parent
            / owned_temporary_name(config, "token")
        ),
        model=config.model,
    )
    assert valid.operation == "receive"


def test_transported_model_contract_is_hash_bound() -> None:
    with pytest.raises(ValidationError, match="differs from its bytes"):
        model_stage.TransportedModelContract(
            canonical_bytes_base64="e30K",
            sha256="a" * 64,
        )


def test_remote_helper_reports_unconfirmed_early_temp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = tmp_path / "remote" / config.model.directory_name
    destination.parent.mkdir()
    request = RemoteRequest(
        schema_version=1,
        operation="receive",
        run_id=config.run_id,
        namespace=config.namespace,
        owner_token="owner-token",
        expected_host_name="fwuff",
        destination=str(destination),
        temporary_path=str(
            destination.parent / owned_temporary_name(config, "owner-token")
        ),
        model=config.model,
    )
    input_stream = io.TextIOWrapper(
        io.BytesIO(request.model_dump_json().encode() + b"\n")
    )
    output_stream = io.StringIO()

    def fail_creation(_path: Path) -> model_stage.OwnedDirectory:
        raise OperationError("injected early creation failure", cleanup_confirmed=False)

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdin", input_stream)
        patch.setattr(sys, "stdout", output_stream)
        patch.setattr(model_stage.socket, "gethostname", lambda: "fwuff")
        patch.setattr(model_stage, "_remote_process_identity", lambda: (101, 101, 1))
        return_code = model_stage.remote_helper_main(fail_creation)

    receipts = output_stream.getvalue().splitlines()
    assert return_code == 1
    assert len(receipts) == 2
    response = RemoteResponse.model_validate_json(receipts[-1])
    assert response.status == "failed"
    assert response.cleanup_succeeded is False


def test_remote_helper_checks_host_before_destination_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    destination = tmp_path / "missing-parent" / config.model.directory_name
    request = RemoteRequest(
        schema_version=1,
        operation="probe",
        run_id=config.run_id,
        namespace=config.namespace,
        owner_token="owner-token",
        expected_host_name="fwuff",
        destination=str(destination),
        temporary_path=str(
            destination.parent / owned_temporary_name(config, "owner-token")
        ),
        model=config.model,
    )
    input_stream = io.TextIOWrapper(
        io.BytesIO(request.model_dump_json().encode() + b"\n")
    )
    output_stream = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdin", input_stream)
        patch.setattr(sys, "stdout", output_stream)
        patch.setattr(model_stage.socket, "gethostname", lambda: "wrong-host")
        return_code = model_stage.remote_helper_main()
    response = RemoteResponse.model_validate_json(output_stream.getvalue())
    assert return_code == 1
    assert response.cleanup_succeeded is True
    assert "does not match" in cast(str, response.error)
    assert not destination.parent.exists()


def make_tar(member: tarfile.TarInfo, data: bytes = b"data") -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    stream.seek(0)
    return stream


def test_tar_extractor_creates_only_regular_in_tree_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    extract_tar_stream(root, make_tar(tarfile.TarInfo("nested/file")), lambda: None)
    assert (root / "nested" / "file").read_bytes() == b"data"


@pytest.mark.parametrize(
    "member",
    [
        tarfile.TarInfo("../escape"),
        tarfile.TarInfo("/absolute"),
        tarfile.TarInfo("safe-link"),
    ],
)
def test_tar_extractor_rejects_traversal_and_links(
    tmp_path: Path, member: tarfile.TarInfo
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    if member.name == "safe-link":
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
    with pytest.raises(StageError, match="unsafe|not a regular"):
        extract_tar_stream(root, make_tar(member), lambda: None)


def test_system_ssh_argv_is_explicit_and_model_data_is_not_shell_interpolated(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    effects = make_system_effects(config)
    command = effects.build_remote_command(config)
    assert command[: 1 + len(config.ssh.options)] == (
        config.ssh.executable,
        *config.ssh.options,
    )
    assert command[-3:-1] == ("--", "fwuff")
    assert config.model.model_id not in command[-1]
    assert config.remote_destination not in command[-1]
    assert command[-1].startswith("/usr/bin/python3 -c ")


def test_system_result_fragment_write_is_atomic_and_refuses_symlinks(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    effects = make_system_effects(config)
    effects.write_result_json(
        RUNTIME_METADATA,
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "namespace": config.namespace,
            "owner_token": "owner",
            "owned_processes": [],
        },
    )
    assert (
        read_json_object(result_directory / RUNTIME_METADATA)["run_id"] == config.run_id
    )
    (result_directory / BENCHMARK_RESULT).symlink_to(
        result_directory / RUNTIME_METADATA
    )
    with pytest.raises(StageError, match="symlink"):
        effects.write_result_json(
            BENCHMARK_RESULT,
            {
                "schema_version": 1,
                "run_id": config.run_id,
                "namespace": config.namespace,
                "cleanup_succeeded": True,
                "owned_processes": [],
            },
        )


def test_result_writes_remain_anchored_to_retained_wrapper_descriptor(
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    config = make_config(tmp_path, result_directory=result_directory)
    descriptor = os.open(
        result_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    effects = SystemEffects(config, descriptor, observed_local_host_name="dwagon")
    retained_directory = tmp_path / "retained-results"
    result_directory.rename(retained_directory)
    result_directory.mkdir()

    effects.write_result_json(
        RUNTIME_METADATA,
        {
            "schema_version": 1,
            "run_id": config.run_id,
            "namespace": config.namespace,
            "owner_token": "owner",
            "owned_processes": [],
        },
    )
    assert (retained_directory / RUNTIME_METADATA).is_file()
    assert not (result_directory / RUNTIME_METADATA).exists()
    os.close(descriptor)


def test_result_fragments_are_wrapper_v1_compatible(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    effects = FakeEffects(config)
    result = run_staging(config, effects)
    runtime = [
        value for filename, value in effects.fragments if filename == RUNTIME_METADATA
    ][-1]
    assert runtime["schema_version"] == 1
    assert runtime["run_id"] == config.run_id
    assert runtime["namespace"] == config.namespace
    assert isinstance(runtime["owner_token"], str)
    assert result["schema_version"] == 1
    assert isinstance(result["cleanup_succeeded"], bool)
    assert result["owned_processes"] == runtime["owned_processes"]
    for process in cast(list[dict[str, object]], result["owned_processes"]):
        assert set(
            (
                "host_name",
                "pid",
                "process_group_id",
                "start_time_ticks",
                "owner_token",
                "namespace",
                "transport_pid",
                "log_path",
            )
        ) <= set(process)


def test_wrapper_accepts_final_runtime_and_result_reconciliation(
    tmp_path: Path,
) -> None:
    run_id = make_config(tmp_path).run_id
    result_directory = tmp_path / run_id
    config = make_config(tmp_path, result_directory=result_directory)
    effects = FakeEffects(config)
    result = run_staging(config, effects, owner_token_factory=lambda: "wrapper-token")
    runtime = [
        value for filename, value in effects.fragments if filename == RUNTIME_METADATA
    ][-1]
    command = ("/usr/bin/python3", "-c", "raise SystemExit(0)")
    commit = "a" * 40
    hosts = [config.local_host_name, config.remote_host_name]
    metadata: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "namespace": config.namespace,
        "reserved_ports": [53001],
        "result_directory": config.result_directory,
        "command": list(command),
        "git": {"commit": commit, "dirty": False, "dirty_file_hashes": {}},
        "hosts": hosts,
        "models": [
            {
                "model_id": config.model.model_id,
                "revision": config.model.revision,
                "paths": {
                    config.local_host_name: config.local_destination,
                    config.remote_host_name: config.remote_destination,
                },
            }
        ],
        "gpu_bindings": {host: [] for host in hosts},
        "cpu_bindings": {
            host: {
                "cpu_set": "0",
                "numa_nodes": [0],
                "memory_policy": "bind:0",
            }
            for host in hosts
        },
        "hca_bindings": {
            host: [
                {"device": "mlx4_0", "port": port, "gid": gid}
                for port, gid in enumerate(CURRENT_HCA_GIDS[host], start=1)
            ]
            for host in hosts
        },
        "source_deployments": {
            host: {"path": "/root/exo", "commit": commit, "dirty_file_hashes": {}}
            for host in hosts
        },
        "owner_pids": {host: [] for host in hosts},
    }
    lease = BenchmarkLease(
        lock_path=tmp_path / "wrapper.lock",
        lease_path=tmp_path / "wrapper.json",
        result_directory=result_directory,
        owner="codex:test",
        purpose="stager-fragment-test",
        run_id=config.run_id,
        namespace=config.namespace,
        ports=(53001,),
        command=command,
        metadata=metadata,
        cleanup_grace_seconds=300.0,
    )
    lease.acquire()
    try:
        atomic_write_json(result_directory / RUNTIME_METADATA, runtime)
        atomic_write_json(result_directory / BENCHMARK_RESULT, result)
        assert lease.cleanup_succeeded(True) is True
        assert lease.cleanup_confirmation_error() is None
    finally:
        lease.release()


def test_prepared_wrapper_runs_real_child_hf_ssh_and_remote_lifecycle(
    tmp_path: Path,
) -> None:
    local_host = socket.gethostname()
    remote_host = "stage-remote"
    assert local_host != remote_host
    result_root = tmp_path / "results"
    result_root.mkdir()
    run_id = "smol-stage-test"
    fake_hf = write_fake_hf(tmp_path / "hf")
    fake_ssh, remote_python = write_fake_transport(
        tmp_path, remote_host_name=remote_host
    )
    model = make_model()
    remote_destination = tmp_path / "remote" / model.directory_name
    config = remap_runtime_config(
        make_config(tmp_path, result_directory=result_root / run_id),
        local_host_name=local_host,
        remote_host_name=remote_host,
        remote_destination=remote_destination,
        ssh_executable=fake_ssh,
        remote_python_executable=remote_python,
        hf_executable=fake_hf,
    )
    config = bind_config_to_current_source(config)
    Path(config.local_destination).parent.mkdir(parents=True)
    remote_destination.parent.mkdir(parents=True)
    config_path = tmp_path / "stage-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    metadata_path = tmp_path / "stage-metadata.json"
    staging_script = Path(__file__).parents[1] / "two_host_model_stage.py"
    wrapper_script = Path(__file__).parents[1] / "benchmark_lease.py"
    lease_path = tmp_path / "active-lease.json"
    lock_path = tmp_path / "benchmark.lock"
    python = Path(sys.executable)
    prepared = prepare_lease_metadata(
        config_path=config_path,
        metadata_output=metadata_path,
        wrapper_python=python,
        child_python=python,
        benchmark_lease_script=wrapper_script,
        staging_script=staging_script,
        owner="codex:test",
        purpose="full fake staging lifecycle",
        expected_duration_seconds=60.0,
        heartbeat_seconds=0.2,
        cleanup_grace_seconds=300.0,
        lease_path=lease_path,
        lock_path=lock_path,
        result_root=result_root,
        identity_reader=lambda _path: config.lease_metadata.git,
    )

    completed = subprocess.run(
        prepared.benchmark_lease_argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    result_directory = result_root / run_id
    result = read_json_object(result_directory / BENCHMARK_RESULT)
    manifest = read_json_object(result_directory / "manifest.json")
    assert result["status"] == "completed"
    assert result["cleanup_succeeded"] is True
    assert result["acquisition"] == "huggingface_cli"
    assert manifest["status"] == "completed"
    assert manifest["cleanup_succeeded"] is True
    assert not lease_path.exists()
    local_verification = verify_snapshot(Path(config.local_destination), config.model)
    assert verify_snapshot(remote_destination, config.model) == local_verification
    assert (result_directory / "model-stage-local-download.log").is_file()
    assert (result_directory / "model-stage-remote-transfer.log").is_file()


def test_wrapper_partial_remote_identity_preserves_cleanup_tombstone(
    tmp_path: Path,
) -> None:
    local_host = socket.gethostname()
    remote_host = "stage-remote"
    result_root = tmp_path / "results"
    result_root.mkdir()
    fake_hf = write_fake_hf(tmp_path / "hf")
    ssh_pid_path = tmp_path / "fake-ssh.pid"
    fake_ssh = write_executable(
        tmp_path / "partial-ssh",
        f"""#!{sys.executable}
import os
import time
from pathlib import Path
Path({str(ssh_pid_path)!r}).write_text(str(os.getpid()), encoding='ascii')
os.write(1, b'{{')
time.sleep(30)
""",
    )
    model = make_model()
    remote_destination = tmp_path / "remote" / model.directory_name
    config = remap_runtime_config(
        make_config(
            tmp_path,
            result_directory=result_root / "smol-stage-test",
        ),
        local_host_name=local_host,
        remote_host_name=remote_host,
        remote_destination=remote_destination,
        ssh_executable=fake_ssh,
        remote_python_executable=Path("/usr/bin/python3"),
        hf_executable=fake_hf,
    )
    config = bind_config_to_current_source(config)
    payload = config.model_dump(mode="json")
    timeouts = cast(dict[str, object], payload["timeouts"])
    timeouts["remote_probe_seconds"] = 0.1
    timeouts["cleanup_seconds"] = 0.5
    timeouts["poll_seconds"] = 0.005
    config = StageConfig.model_validate_json(json.dumps(payload))
    Path(config.local_destination).parent.mkdir(parents=True)
    remote_destination.parent.mkdir(parents=True)
    config_path = tmp_path / "stage-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    metadata_path = tmp_path / "stage-metadata.json"
    lease_path = tmp_path / "active-lease.json"
    lock_path = tmp_path / "benchmark.lock"
    python = Path(sys.executable)
    prepared = prepare_lease_metadata(
        config_path=config_path,
        metadata_output=metadata_path,
        wrapper_python=python,
        child_python=python,
        benchmark_lease_script=Path(__file__).parents[1] / "benchmark_lease.py",
        staging_script=Path(__file__).parents[1] / "two_host_model_stage.py",
        owner="codex:test",
        purpose="partial identity cleanup lifecycle",
        expected_duration_seconds=60.0,
        heartbeat_seconds=0.2,
        cleanup_grace_seconds=300.0,
        lease_path=lease_path,
        lock_path=lock_path,
        result_root=result_root,
        identity_reader=lambda _path: config.lease_metadata.git,
    )

    completed = subprocess.run(
        prepared.benchmark_lease_argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=20.0,
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )
    assert completed.returncode != 0
    ssh_pid = int(ssh_pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(ssh_pid, 0)
    result_directory = result_root / config.run_id
    result = read_json_object(result_directory / BENCHMARK_RESULT)
    manifest = read_json_object(result_directory / "manifest.json")
    tombstone = read_json_object(lease_path)
    assert result["status"] == "cleanup_failed"
    assert result["cleanup_succeeded"] is False
    assert manifest["cleanup_succeeded"] is False
    assert tombstone["manual_clearance_required"] is True
    assert tombstone["cleanup_succeeded"] is False
