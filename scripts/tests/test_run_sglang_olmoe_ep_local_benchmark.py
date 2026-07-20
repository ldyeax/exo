from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import run_sglang_olmoe_ep_local_benchmark as olmoe
from scripts.sglang_olmoe_serving_client import (
    EndpointObservation,
    GenerateObservation,
    OlmoeNativeGenerateRequest,
    OlmoeNativeServingClient,
    SanityResponseObservation,
    token_ids_sha256,
)

SHA = "a" * 64


def _base_config(tmp_path: Path) -> olmoe.OlmoeEpBenchmarkConfig:
    return olmoe.OlmoeEpBenchmarkConfig(
        run_id="olmoe-ep2-test",
        result_directory=tmp_path / "result",
        runtime_python="/runtime/python",
        runtime_install_receipt=tmp_path / "install-receipt.json",
        runtime_install_receipt_sha256="0" * 64,
        model_path=olmoe.OLMOE_MODEL_PATH,
        stage_contract=tmp_path / "stage-contract.json",
        stage_capture_output=None,
        expert_parallel_size=2,
        host="127.0.0.1",
        port=62_610,
        static_memory_fraction=0.9,
        readiness_timeout_seconds=60.0,
        request_timeout_seconds=90.0,
        cleanup_timeout_seconds=2.0,
        numactl_executable="/usr/bin/numactl",
        nvidia_smi_executable="/usr/bin/nvidia-smi",
    )


def _capture(ep_size: olmoe.ExpertParallelSize) -> olmoe.SanityCapture:
    input_ids = (101, 102, 103)
    output_ids = (201,)
    output_text = "42"
    config = replace(
        _base_config(Path("/tmp/olmoe-test-fixture")),
        expert_parallel_size=ep_size,
    )
    namespace = f"exo-olmoe-ep-test-{'a' * 32}"
    environment = olmoe.build_server_environment(config, "redacted", namespace, {})
    managed_launch = olmoe.ManagedLaunchEvidence(
        schema_version=1,
        status="owned_runtime_listener_verified",
        harness_sha256=hashlib.sha256(olmoe._HARNESS_PATH.read_bytes()).hexdigest(),
        runtime_admission_sha256="3" * 64,
        pid=123,
        process_group_id=123,
        start_time_ticks=456,
        owner_token_sha256="4" * 64,
        ownership_namespace=namespace,
        command=olmoe.build_server_command(config),
        launch_environment=tuple(
            sorted(
                (name, value)
                for name, value in environment.items()
                if name != olmoe._OWNER_TOKEN_ENVIRONMENT
            )
        ),
        listener_socket_inodes=(789,),
        listener_owner_pids=(123,),
        rank_local_numa_observation_sha256="5" * 64,
        verified_at_utc="2026-07-20T11:59:00+00:00",
    )
    return olmoe.SanityCapture(
        schema_version=1,
        status="captured",
        expert_parallel_size=ep_size,
        model_id=olmoe.OLMOE_MODEL_ID,
        model_revision=olmoe.OLMOE_MODEL_REVISION,
        model_path=olmoe.OLMOE_MODEL_PATH,
        sglang_revision=olmoe.OLMOE_SGLANG_REVISION,
        runtime_install_receipt_sha256="0" * 64,
        snapshot_canonical_sha256="1" * 64,
        prompt_text="17 + 25 =",
        tokenizer_class="OlmoeTokenizerFast",
        input_ids=input_ids,
        input_ids_sha256=token_ids_sha256(input_ids),
        output_ids=output_ids,
        output_ids_sha256=token_ids_sha256(output_ids),
        output_text=output_text,
        output_text_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
        max_new_tokens=1,
        sampling_seed=20_260_720,
        static_memory_fraction=0.9,
        server_host="127.0.0.1",
        server_port=62_610,
        server_info_sha256="2" * 64,
        managed_launch=managed_launch,
        producer_sha256=hashlib.sha256(
            olmoe._STAGE_CONTRACT_PRODUCER_PATH.read_bytes()
        ).hexdigest(),
        captured_at_utc="2026-07-20T12:00:00+00:00",
    )


def test_tiny_self_attested_runtime_receipt_is_rejected(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config.runtime_install_receipt.write_text(
        json.dumps({"schema_version": 1, "status": "install_complete"})
    )
    config = replace(
        config,
        runtime_install_receipt_sha256=hashlib.sha256(
            config.runtime_install_receipt.read_bytes()
        ).hexdigest(),
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="exact schema"):
        olmoe.verify_runtime_install(config)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_server_command_is_native_tp2_with_rank_local_numa(
    tmp_path: Path, ep_size: olmoe.ExpertParallelSize
) -> None:
    config = replace(_base_config(tmp_path), expert_parallel_size=ep_size)

    command = olmoe.build_server_command(config)

    assert command[:7] == (
        "/usr/bin/numactl",
        "--physcpubind",
        "0-111",
        "--interleave",
        "0,1",
        config.runtime_python,
        "-m",
    )
    assert command[command.index("--tp-size") + 1] == "2"
    assert command[command.index("--ep-size") + 1] == str(ep_size)
    assert command[command.index("--max-total-tokens") + 1] == "4096"
    assert "--disable-custom-all-reduce" in command
    numa_index = command.index("--numa-node")
    assert command[numa_index + 1 : numa_index + 3] == ("0", "1")
    assert not any("kt-" in value or "ktransformers" in value for value in command)


def test_server_environment_uses_all_cores_as_two_rank_local_pools(
    tmp_path: Path,
) -> None:
    environment = olmoe.build_server_environment(
        _base_config(tmp_path),
        "owner",
        "namespace",
        {
            "PATH": "/usr/bin",
            "KT_CONFIG": "bad",
            "PYTHONPATH": "/mutable",
            "CUDA_VISIBLE_DEVICES": "wrong",
            "LD_PRELOAD": "/profiler.so",
            "TORCH_LOGS": "all",
            "NCCL_ALGO": "Tree",
        },
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(olmoe.DWAGON_GPU_UUIDS)
    assert environment["OMP_NUM_THREADS"] == "56"
    assert environment["SGLANG_NUMA_BIND_V2"] == "1"
    assert environment["NCCL_P2P_LEVEL"] == "NVL"
    assert "KT_CONFIG" not in environment
    assert "PYTHONPATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert "TORCH_LOGS" not in environment
    assert "NCCL_ALGO" not in environment


def test_server_info_pins_observable_launch_fields(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    response = {
        "version": "0.0.0.dev0",
        "model_path": config.model_path,
        "host": config.host,
        "port": config.port,
        "tp_size": 2,
        "pp_size": 1,
        "ep_size": 2,
        "nnodes": 1,
        "node_rank": 0,
        "dtype": "bfloat16",
        "context_length": 4096,
        "max_total_tokens": 4096,
        "mem_fraction_static": 0.9,
        "max_running_requests": 1,
        "random_seed": olmoe.CANONICAL_SAMPLING_SEED,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "disable_radix_cache": True,
        "disable_custom_all_reduce": True,
        "numa_node": [0, 1],
    }

    evidence = olmoe._verify_server_info(response, config)

    assert evidence["version"] == "0.0.0.dev0"
    mutated = {**response, "max_running_requests": 2}
    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="server_info"):
        olmoe._verify_server_info(mutated, config)


def test_configuration_receipt_names_token_pool_and_nccl_fallback(
    tmp_path: Path,
) -> None:
    receipt = olmoe._configuration_receipt(_base_config(tmp_path))

    assert receipt["token_pool"] == {
        "context_length": 4096,
        "max_total_tokens": 4096,
        "max_running_requests": 1,
    }
    assert receipt["tensor_parallel_collective"] == {
        "custom_all_reduce": False,
        "fallback": "NCCL",
        "cuda_visible_devices_format": "GPU_UUID",
    }


def _valid_server_startup_log(ep_size: olmoe.ExpertParallelSize = 1) -> str:
    rank_labels = ("TP0", "TP1") if ep_size == 1 else ("TP0 EP0", "TP1 EP1")
    return "\n".join(
        (
            "server_args=ServerArgs(max_total_tokens=4096, "
            "disable_custom_all_reduce=True)",
            f"[2026-07-20 07:15:28 {rank_labels[0]}] KV Cache is allocated. "
            "#tokens: 4096, K size: 1 GB",
            f"[2026-07-20 07:15:28 {rank_labels[1]}] KV Cache is allocated. "
            "#tokens: 4096, K size: 1 GB",
        )
    )


@pytest.mark.parametrize("ep_size", [1, 2])
def test_server_log_contract_verifies_exact_rank_allocations(
    tmp_path: Path, ep_size: olmoe.ExpertParallelSize
) -> None:
    log_path = tmp_path / "native-sglang-server.log"
    contents = _valid_server_startup_log(ep_size).encode()
    log_path.write_bytes(contents)

    receipt = olmoe.verify_server_log_contract(log_path, ep_size)

    assert receipt["observed_sha256"] == hashlib.sha256(contents).hexdigest()
    assert receipt["kv_cache_allocation_line_count"] == 2
    expected_ep_ranks = [None, None] if ep_size == 1 else [0, 1]
    assert receipt["kv_cache_allocation_rank_bindings"] == [
        {"tp_rank": 0, "ep_rank": expected_ep_ranks[0]},
        {"tp_rank": 1, "ep_rank": expected_ep_ranks[1]},
    ]
    assert receipt["expert_parallel_size"] == ep_size
    assert receipt["max_total_tokens"] == 4096
    assert receipt["custom_all_reduce_disabled"] is True


@pytest.mark.parametrize(
    ("contents", "ep_size", "match"),
    (
        (
            _valid_server_startup_log().replace(
                "max_total_tokens=4096", "max_total_tokens=None"
            ),
            1,
            "server argument contract",
        ),
        (
            _valid_server_startup_log().replace("#tokens: 4096", "#tokens: 234355"),
            1,
            "rank-bound 4096-token KV allocations",
        ),
        (
            _valid_server_startup_log()
            + "\n[TP0] Setup Custom allreduce failed with invalid device ordinal",
            1,
            "custom all-reduce setup failure",
        ),
        (
            _valid_server_startup_log(2).replace(" EP0", "").replace(" EP1", ""),
            2,
            "rank-bound 4096-token KV allocations",
        ),
        (
            _valid_server_startup_log(2).replace("TP0 EP0", "TP0 EP1"),
            2,
            "rank-bound 4096-token KV allocations",
        ),
        (
            _valid_server_startup_log(2),
            1,
            "rank-bound 4096-token KV allocations",
        ),
    ),
)
def test_server_log_contract_rejects_ineffective_launch(
    tmp_path: Path,
    contents: str,
    ep_size: olmoe.ExpertParallelSize,
    match: str,
) -> None:
    log_path = tmp_path / "native-sglang-server.log"
    log_path.write_text(contents)

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match=match):
        olmoe.verify_server_log_contract(log_path, ep_size)


def test_nv4_admission_binds_uuid_order(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    def runner(
        arguments: tuple[str, ...], _timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if "--query-gpu=index,uuid,pci.bus_id" in arguments:
            stdout = (
                f"0, {olmoe.DWAGON_GPU_UUIDS[0]}, 00000000:27:00.0\n"
                f"1, {olmoe.DWAGON_GPU_UUIDS[1]}, 00000000:d8:00.0\n"
            )
        else:
            stdout = "GPU0 GPU1 CPU Affinity\nGPU0 X NV4 0-55\nGPU1 NV4 X 56-111\n"
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    evidence = olmoe.verify_dwagon_nv4_topology(config, runner)

    assert evidence["ordered_gpu_uuids"] == list(olmoe.DWAGON_GPU_UUIDS)
    assert evidence["observed_forward_link"] == "NV4"


def test_cpu_topology_selects_one_thread_per_core_across_both_nodes(
    tmp_path: Path,
) -> None:
    sysfs = tmp_path / "sys/devices/system"
    (sysfs / "cpu").mkdir(parents=True)
    (sysfs / "cpu/online").write_text("0-223\n")
    for node, cpulist in ((0, "0-55,112-167\n"), (1, "56-111,168-223\n")):
        path = sysfs / f"node/node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(cpulist)
    for cpu in range(112):
        path = sysfs / f"cpu/cpu{cpu}/topology"
        path.mkdir(parents=True)
        (path / "thread_siblings_list").write_text(f"{cpu},{cpu + 112}\n")

    evidence = olmoe.verify_dwagon_cpu_topology(sysfs)

    assert evidence["selected_cpu_count"] == 112


def _synthetic_process_stat(pid: int, process_group: int, start_ticks: int) -> str:
    fields = ["S", "1", str(process_group), *("0" for _ in range(16)), str(start_ticks)]
    return f"{pid} (scheduler) {' '.join(fields)}\n"


def _append_synthetic_vma(
    proc_root: Path,
    *,
    pid: int,
    address: str,
    policy: str | None,
    numa_details: str = "",
    permissions: str = "rw-p",
    offset: str = "00000000",
    device: str = "00:00",
    inode: int = 0,
    path: str | None = None,
    rss_kibibytes: int = 0,
    pss_kibibytes: int = 0,
    anonymous_kibibytes: int = 0,
    swap_kibibytes: int = 0,
    vm_flags: tuple[str, ...] = ("rd", "wr", "mr", "mw", "me", "ac", "sd"),
) -> None:
    process_root = proc_root / str(pid)
    end_address = f"{int(address, 16) + 0x1000:x}"
    path_field = "" if path is None else f" {path}"
    header = (
        f"{address}-{end_address} {permissions} {offset} {device} {inode}{path_field}\n"
    )
    with (process_root / "maps").open("a") as destination:
        destination.write(header)
    with (process_root / "smaps").open("a") as destination:
        destination.write(header)
        destination.write(f"Rss: {rss_kibibytes} kB\n")
        destination.write(f"Pss: {pss_kibibytes} kB\n")
        destination.write(f"Anonymous: {anonymous_kibibytes} kB\n")
        destination.write(f"Swap: {swap_kibibytes} kB\n")
        destination.write(f"VmFlags: {' '.join(vm_flags)}\n")
    if policy is not None:
        details_field = "" if not numa_details else f" {numa_details}"
        with (process_root / "numa_maps").open("a") as destination:
            destination.write(f"{address} {policy}{details_field}\n")


def _write_synthetic_scheduler(
    proc_root: Path,
    *,
    pid: int,
    process_group: int,
    title: str,
    cpus: str,
    memory_policy: str,
) -> None:
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True)
    (process_root / "stat").write_text(
        _synthetic_process_stat(pid, process_group, pid * 10)
    )
    (process_root / "cmdline").write_bytes(title.encode() + b"\0--worker\0")
    status = f"Name:\tscheduler\nCpus_allowed_list:\t{cpus}\n"
    (process_root / "status").write_text(status)
    _write_synthetic_task_affinity(proc_root, pid=pid, task_id=pid, cpus=cpus)
    remote_node = 1 if memory_policy == "bind:0" else 0
    expected_node = int(memory_policy.removeprefix("bind:"))
    for name in ("maps", "smaps", "numa_maps"):
        (process_root / name).write_bytes(b"")
    _append_synthetic_vma(
        proc_root,
        pid=pid,
        address="1000",
        policy=memory_policy,
        numa_details=(f"file=/mapped mapped=4 N{remote_node}=4 kernelpagesize_kB=4"),
        permissions="r--p",
        device="08:01",
        inode=1,
        path="/mapped",
        rss_kibibytes=16,
        pss_kibibytes=16,
        vm_flags=("rd", "mr", "mw", "me", "sd"),
    )
    _append_synthetic_vma(
        proc_root,
        pid=pid,
        address="2000",
        policy=memory_policy,
        numa_details=(f"heap anon=4 dirty=4 N{expected_node}=4 kernelpagesize_kB=4"),
        path="[heap]",
        rss_kibibytes=16,
        pss_kibibytes=16,
        anonymous_kibibytes=16,
    )
    _append_synthetic_vma(
        proc_root,
        pid=pid,
        address="3000",
        policy="local",
    )
    _append_synthetic_vma(
        proc_root,
        pid=pid,
        address="4000",
        policy=memory_policy,
        numa_details="file=/dev/nvidiactl",
        permissions="rw-s",
        device="00:05",
        inode=2,
        path="/dev/nvidiactl",
        vm_flags=("rd", "wr", "sh", "mr", "mw", "me", "ms", "sd"),
    )


def _write_synthetic_task_affinity(
    proc_root: Path, *, pid: int, task_id: int, cpus: str
) -> None:
    task_root = proc_root / str(pid) / "task" / str(task_id)
    task_root.mkdir(parents=True, exist_ok=True)
    task_root.joinpath("status").write_text(
        f"Name:\tscheduler\nCpus_allowed_list:\t{cpus}\n"
    )


def _synthetic_numa_tree(tmp_path: Path) -> tuple[Path, Path]:
    proc_root = tmp_path / "proc"
    sysfs_root = tmp_path / "sys/devices/system"
    for node, cpus in (
        (0, "0-55,112-167"),
        (1, "56-111,168-223"),
    ):
        node_root = sysfs_root / f"node/node{node}"
        node_root.mkdir(parents=True)
        (node_root / "cpulist").write_text(cpus + "\n")
        _write_synthetic_scheduler(
            proc_root,
            pid=100 + node,
            process_group=100,
            title=f"sglang::scheduler_TP{node}_EP{node}",
            cpus="0,112" if node == 0 else "56,168",
            memory_policy=f"bind:{node}",
        )
        _write_synthetic_task_affinity(
            proc_root,
            pid=100 + node,
            task_id=200 + node,
            cpus="1,113" if node == 0 else "57,169",
        )
    return proc_root, sysfs_root


def test_observed_rank_local_numa_binds_both_scheduler_ranks(tmp_path: Path) -> None:
    proc_root, sysfs_root = _synthetic_numa_tree(tmp_path)

    evidence = olmoe.observe_rank_local_numa(
        process_ids=(100, 101),
        process_group_id=100,
        expert_parallel_size=2,
        proc_root=proc_root,
        sysfs_root=sysfs_root,
    )

    ranks = cast(list[dict[str, object]], evidence["ranks"])
    assert [rank["expected_node"] for rank in ranks] == [0, 1]
    assert ranks[0]["task_affinity_sets"] == [
        {"task_id": 100, "cpus": [0, 112]},
        {"task_id": 200, "cpus": [1, 113]},
    ]
    assert ranks[0]["task_affinity_union"] == [0, 1, 112, 113]
    assert isinstance(ranks[0]["task_affinity_sha256"], str)
    memory = cast(dict[str, object], ranks[0]["numa_memory_placement"])
    assert memory["status"] == "verified"
    assert memory["policy_counts"] == {"bind:0": 3, "local": 1}
    assert memory["local_reservation_exception_count"] == 1
    assert memory["provable_sensitive_off_node_kibibytes"] == 0
    assert isinstance(memory["maps_sha256"], str)
    assert isinstance(memory["smaps_sha256"], str)
    assert isinstance(memory["numa_maps_sha256"], str)
    assert isinstance(memory["classified_rows_sha256"], str)
    classifications = cast(list[dict[str, object]], memory["classifications"])
    by_name = {item["classification"]: item for item in classifications}
    assert by_name["private_anonymous"]["policy_counts"] == {
        "bind:0": 1,
        "local": 1,
    }
    assert by_name["special_or_shared"]["policy_counts"] == {"bind:0": 1}


def test_observed_rank_local_numa_rejects_cross_node_task_affinity(
    tmp_path: Path,
) -> None:
    proc_root, sysfs_root = _synthetic_numa_tree(tmp_path)
    _write_synthetic_task_affinity(proc_root, pid=100, task_id=200, cpus="1,56")

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="thread outside NUMA"):
        olmoe.observe_rank_local_numa(
            process_ids=(100, 101),
            process_group_id=100,
            expert_parallel_size=2,
            proc_root=proc_root,
            sysfs_root=sysfs_root,
        )


def test_observed_rank_local_numa_retries_task_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root, sysfs_root = _synthetic_numa_tree(tmp_path)
    racing_status = proc_root / "100/task/200/status"
    real_reader = olmoe._task_status_cpu_affinity
    race_triggered = False

    def task_reader(path: Path) -> frozenset[int] | None:
        nonlocal race_triggered
        if path == racing_status and not race_triggered:
            race_triggered = True
            path.unlink()
            path.parent.rmdir()
        return real_reader(path)

    monkeypatch.setattr(olmoe, "_task_status_cpu_affinity", task_reader)

    evidence = olmoe.observe_rank_local_numa(
        process_ids=(100, 101),
        process_group_id=100,
        expert_parallel_size=2,
        proc_root=proc_root,
        sysfs_root=sysfs_root,
    )

    ranks = cast(list[dict[str, object]], evidence["ranks"])
    assert race_triggered is True
    assert ranks[0]["stable_observation_attempt"] == 2
    assert ranks[0]["task_count"] == 1


def test_task_affinity_reader_does_not_mask_non_exit_read_error(
    tmp_path: Path,
) -> None:
    invalid_status = tmp_path / "status"
    invalid_status.mkdir()

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="cannot read scheduler"):
        olmoe._task_status_cpu_affinity(invalid_status)


def test_observed_rank_local_numa_rejects_mixed_memory_policy(
    tmp_path: Path,
) -> None:
    proc_root, sysfs_root = _synthetic_numa_tree(tmp_path)
    _append_synthetic_vma(
        proc_root,
        pid=101,
        address="5000",
        policy="default",
        numa_details="anon=1 dirty=1 N1=1 kernelpagesize_kB=4",
        rss_kibibytes=4,
        pss_kibibytes=4,
        anonymous_kibibytes=4,
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="memory placement"):
        olmoe.observe_rank_local_numa(
            process_ids=(100, 101),
            process_group_id=100,
            expert_parallel_size=2,
            proc_root=proc_root,
            sysfs_root=sysfs_root,
        )


@pytest.mark.parametrize(
    ("policy", "numa_details", "permissions", "device", "inode", "path"),
    (
        ("default", "anon=1 N0=1 kernelpagesize_kB=4", "rw-p", "00:00", 0, None),
        (
            "local",
            "file=/dev/nvidiactl",
            "rw-s",
            "00:05",
            2,
            "/dev/nvidiactl",
        ),
        ("bind:1", "anon=1 N0=1 kernelpagesize_kB=4", "rw-p", "00:00", 0, None),
        (
            "interleave:0-1",
            "anon=1 N0=1 kernelpagesize_kB=4",
            "rw-p",
            "00:00",
            0,
            None,
        ),
        (
            "preferred:1",
            "anon=1 N0=1 kernelpagesize_kB=4",
            "rw-p",
            "00:00",
            0,
            None,
        ),
        ("default", "file=/runtime/lib.so", "r--p", "08:01", 3, "/runtime/lib.so"),
    ),
)
def test_numa_memory_placement_rejects_resident_or_unsupported_non_bind_policy(
    tmp_path: Path,
    policy: str,
    numa_details: str,
    permissions: str,
    device: str,
    inode: int,
    path: str | None,
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    is_resident = "N0=1" in numa_details
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="5000",
        policy=policy,
        numa_details=numa_details,
        permissions=permissions,
        device=device,
        inode=inode,
        path=path,
        rss_kibibytes=4 if is_resident else 0,
        pss_kibibytes=4 if is_resident else 0,
        anonymous_kibibytes=(4 if is_resident and path != "/dev/nvidiactl" else 0),
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="NUMA memory placement"):
        olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)


@pytest.mark.parametrize(
    ("path", "numa_details", "anonymous_kibibytes"),
    (
        (None, "anon=4 dirty=4 N1=4 kernelpagesize_kB=4", 16),
        ("[heap]", "heap anon=4 dirty=4 N1=4 kernelpagesize_kB=4", 16),
        (
            f"{olmoe.OLMOE_MODEL_PATH}/model-00001-of-00003.safetensors",
            (
                f"file={olmoe.OLMOE_MODEL_PATH}/model-00001-of-00003.safetensors "
                "mapped=4 N1=4 kernelpagesize_kB=4"
            ),
            0,
        ),
    ),
)
def test_numa_memory_placement_rejects_sensitive_memory_off_node(
    tmp_path: Path,
    path: str | None,
    numa_details: str,
    anonymous_kibibytes: int,
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    model_weight = path is not None and path.startswith(olmoe.OLMOE_MODEL_PATH)
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="5000",
        policy="bind:0",
        numa_details=numa_details,
        permissions="r--p" if model_weight else "rw-p",
        device="08:01" if model_weight else "00:00",
        inode=4 if model_weight else 0,
        path=path,
        rss_kibibytes=16,
        pss_kibibytes=16,
        anonymous_kibibytes=anonymous_kibibytes,
    )

    with pytest.raises(
        olmoe.OlmoeEpBenchmarkError, match="placement_sensitive_memory_off_node"
    ):
        olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)


def test_numa_memory_placement_bounds_mixed_file_private_residency(
    tmp_path: Path,
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="5000",
        policy="bind:0",
        numa_details=(
            "file=/runtime/lib.so anon=4 mapped=8 N0=4 N1=4 kernelpagesize_kB=4"
        ),
        device="08:01",
        inode=5,
        path="/runtime/lib.so",
        rss_kibibytes=32,
        pss_kibibytes=32,
        anonymous_kibibytes=16,
    )

    evidence = olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)

    assert evidence["provable_sensitive_off_node_kibibytes"] == 0
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="6000",
        policy="bind:0",
        numa_details=(
            "file=/runtime/other.so anon=4 mapped=8 N0=3 N1=5 kernelpagesize_kB=4"
        ),
        device="08:01",
        inode=6,
        path="/runtime/other.so",
        rss_kibibytes=32,
        pss_kibibytes=32,
        anonymous_kibibytes=16,
    )
    with pytest.raises(
        olmoe.OlmoeEpBenchmarkError, match="placement_sensitive_memory_off_node"
    ):
        olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)


@pytest.mark.parametrize(
    ("rss_kibibytes", "pss_kibibytes", "anonymous_kibibytes", "swap_kibibytes"),
    (
        (0, 0, 0, 4),
        (4, 4, 4, 0),
    ),
)
def test_numa_memory_placement_rejects_populated_local_reservation(
    tmp_path: Path,
    rss_kibibytes: int,
    pss_kibibytes: int,
    anonymous_kibibytes: int,
    swap_kibibytes: int,
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="5000",
        policy="local",
        rss_kibibytes=rss_kibibytes,
        pss_kibibytes=pss_kibibytes,
        anonymous_kibibytes=anonymous_kibibytes,
        swap_kibibytes=swap_kibibytes,
    )

    with pytest.raises(
        olmoe.OlmoeEpBenchmarkError,
        match="invalid_local_reservation_exception",
    ):
        olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)


def test_numa_memory_placement_rejects_mismatched_vma_join(tmp_path: Path) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    with (proc_root / "100/numa_maps").open("a") as destination:
        destination.write("5000 bind:0 anon=1 N0=1 kernelpagesize_kB=4\n")

    with pytest.raises(
        olmoe.OlmoeEpBenchmarkError, match="VMA evidence did not stabilize"
    ):
        olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)


def test_numa_memory_placement_accepts_structural_vsyscall_omission(
    tmp_path: Path,
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    _append_synthetic_vma(
        proc_root,
        pid=100,
        address="ffff0000",
        policy=None,
        permissions="--xp",
        path="[vsyscall]",
        vm_flags=("ex",),
    )

    evidence = olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)

    vma_join = cast(dict[str, object], evidence["vma_join"])
    assert vma_join["vsyscall_omissions"] == ["ffff0000"]


def test_numa_memory_placement_retries_vma_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root, _sysfs_root = _synthetic_numa_tree(tmp_path)
    real_reader = olmoe._read_bounded_proc_file
    maps_read_count = 0

    def changing_maps_reader(path: Path, maximum_bytes: int) -> bytes:
        nonlocal maps_read_count
        contents = real_reader(path, maximum_bytes)
        if path == proc_root / "100/maps":
            maps_read_count += 1
            if maps_read_count == 2:
                return contents + b"5000-6000 rw-p 00000000 00:00 0\n"
        return contents

    monkeypatch.setattr(olmoe, "_read_bounded_proc_file", changing_maps_reader)

    evidence = olmoe._observe_numa_memory_placement(100, 0, 1000, proc_root)

    vma_join = cast(dict[str, object], evidence["vma_join"])
    assert vma_join["stable_observation_attempt"] == 2


class _SanityClient:
    def generate_sanity(
        self, _request: OlmoeNativeGenerateRequest
    ) -> SanityResponseObservation:
        return SanityResponseObservation(
            text="42",
            output_ids=(201,),
            prompt_tokens=3,
            completion_tokens=1,
            cached_tokens=0,
            finish_reason={"type": "stop"},
            total_client_seconds=0.5,
        )

    def flush_cache(self) -> EndpointObservation:
        return EndpointObservation(200, SHA, 0.01)


def test_sanity_requires_exact_published_capture() -> None:
    evidence = olmoe._run_exact_sanity(
        cast(OlmoeNativeServingClient, cast(object, _SanityClient())), _capture(2)
    )

    assert evidence["status"] == "exact_match"
    assert evidence["expected_output_ids"] == [201]


class _WorkloadClient:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.flush_calls = 0

    def flush_cache(self) -> EndpointObservation:
        self.flush_calls += 1
        return EndpointObservation(200, SHA, 0.01)

    def generate(self, request: OlmoeNativeGenerateRequest) -> GenerateObservation:
        self.generate_calls += 1
        rate = float(self.generate_calls)
        output_ids = tuple(range(100, 228))
        return GenerateObservation(
            input_ids_sha256=token_ids_sha256(request.input_ids),
            output_ids=output_ids,
            prompt_tokens=len(request.input_ids),
            completion_tokens=128,
            cached_tokens=0,
            output_ids_sha256=token_ids_sha256(output_ids),
            finish_reason_sha256=SHA,
            stream_line_count=130,
            stream_event_count=128,
            output_bearing_event_count=128,
            maximum_stream_line_bytes=1024,
            first_stream_event_output_tokens=1,
            total_client_seconds=128.0 / rate,
            client_observed_ttft_seconds=0.1,
            client_observed_generation_window_seconds=127.0 / rate,
            client_observed_decode_tokens_per_second=rate,
        )


def test_canonical_workload_checks_liveness_after_every_phase() -> None:
    client = _WorkloadClient()
    phases: list[str] = []

    evidence = olmoe.run_canonical_workload(
        cast(OlmoeNativeServingClient, cast(object, client)),
        "decode",
        128,
        128,
        phases.append,
    )

    assert client.generate_calls == 5
    assert client.flush_calls == 5
    assert len(phases) == 10
    summary = cast(dict[str, object], evidence["summary"])
    assert summary["median_client_decode_tokens_per_second"] == 4.0


def test_ep_receipt_distinguishes_tp_control_from_true_ep() -> None:
    ep1 = olmoe.expert_parallel_semantics(1)
    ep2 = olmoe.expert_parallel_semantics(2)

    assert ep1["true_expert_parallel"] is False
    assert ep1["expert_ids_owned_per_ep_rank"] is None
    assert ep1["token_dispatch_semantics"] == (
        "standard_tensor_parallel_moe_no_expert_id_partition"
    )
    assert ep2["true_expert_parallel"] is True
    assert ep2["expert_ids_owned_per_ep_rank"] == 32


def _start_owned_sleeper(
    tmp_path: Path, owner: str, namespace: str
) -> olmoe.RunningServerProcess:
    environment = {
        **os.environ,
        olmoe._OWNER_TOKEN_ENVIRONMENT: owner,
        olmoe._OWNERSHIP_NAMESPACE_ENVIRONMENT: namespace,
    }
    log_path = tmp_path / "owned-sleeper.log"
    log_file = log_path.open("xb", buffering=0)
    code = "import time; time.sleep(60)"
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    process_group_id, start_ticks, _state = olmoe._read_process_stat(process.pid)
    running = olmoe.RunningServerProcess(
        owned=olmoe.OwnedServerProcess(
            pid=process.pid,
            process_group_id=process_group_id,
            start_time_ticks=start_ticks,
            owner_token=owner,
            ownership_namespace=namespace,
            command=(sys.executable, "-c", code),
            launch_environment=(),
            log_path=str(log_path),
        ),
        process=process,
        log_file=log_file,
    )
    return running


def _synthetic_proc_listener(root: Path, pid: int, port: int, inode: int) -> None:
    (root / "net").mkdir(parents=True)
    encoded_port = f"{port:04X}"
    row = (
        f"0: 0100007F:{encoded_port} 00000000:0000 0A "
        f"00000000:00000000 00:00000000 00000000 0 0 {inode} 1\n"
    )
    (root / "net/tcp").write_text("header\n" + row)
    (root / "net/tcp6").write_text("header\n")
    descriptor_root = root / str(pid) / "fd"
    descriptor_root.mkdir(parents=True)
    (descriptor_root / "3").symlink_to(f"socket:[{inode}]")


def test_listener_is_bound_to_owned_process_group_and_nonce(tmp_path: Path) -> None:
    port = 62_610
    running = _start_owned_sleeper(tmp_path, "owner", "namespace")
    proc_root = tmp_path / "proc"
    _synthetic_proc_listener(proc_root, running.owned.pid, port, 123_456)
    try:
        evidence = olmoe.verify_listener_owned(running, port, proc_root)
    finally:
        olmoe.stop_server(running, 2.0)

    assert evidence["status"] == "owned"
    assert evidence["owner_pids"] == [running.owned.pid]


def test_listener_ownership_rejects_stale_process(tmp_path: Path) -> None:
    port = 62_610
    running = _start_owned_sleeper(tmp_path, "owner", "namespace")
    proc_root = tmp_path / "proc"
    _synthetic_proc_listener(proc_root, running.owned.pid + 100_000, port, 123_456)
    try:
        with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="exclusively owned"):
            olmoe.verify_listener_owned(running, port, proc_root)
    finally:
        olmoe.stop_server(running, 2.0)


def test_owned_group_accepts_multiprocessing_descendant_with_scrubbed_environ(
    tmp_path: Path,
) -> None:
    script = tmp_path / "multiprocessing_tree.py"
    script.write_text(
        """
import multiprocessing
import os
import sys
import time


def scrubbed_worker():
    environment = dict(os.environ)
    environment.pop("EXO_OLMOE_EP_OWNER_TOKEN", None)
    environment.pop("EXO_OLMOE_EP_NAMESPACE", None)
    os.execve(
        sys.executable,
        [sys.executable, "-c", "import time; time.sleep(60)"],
        environment,
    )


if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=scrubbed_worker)
    child.start()
    time.sleep(60)
""".lstrip()
    )
    owner = "multiprocessing-owner"
    namespace = "multiprocessing-namespace"
    environment = {
        **os.environ,
        olmoe._OWNER_TOKEN_ENVIRONMENT: owner,
        olmoe._OWNERSHIP_NAMESPACE_ENVIRONMENT: namespace,
    }
    log_path = tmp_path / "multiprocessing-tree.log"
    log_file = log_path.open("xb", buffering=0)
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    process_group, start_ticks, _state = olmoe._read_process_stat(process.pid)
    running = olmoe.RunningServerProcess(
        owned=olmoe.OwnedServerProcess(
            pid=process.pid,
            process_group_id=process_group,
            start_time_ticks=start_ticks,
            owner_token=owner,
            ownership_namespace=namespace,
            command=(sys.executable, str(script)),
            launch_environment=(),
            log_path=str(log_path),
        ),
        process=process,
        log_file=log_file,
    )
    deadline = time.monotonic() + 5.0
    members: tuple[int, ...] = ()
    scrubbed: list[int] = []
    while time.monotonic() < deadline:
        members = olmoe._process_group_members(process_group)
        scrubbed = []
        for pid in members:
            if pid == process.pid:
                continue
            observed_environment = olmoe._process_environment(pid)
            if (
                olmoe._OWNER_TOKEN_ENVIRONMENT not in observed_environment
                and olmoe._OWNERSHIP_NAMESPACE_ENVIRONMENT not in observed_environment
            ):
                scrubbed.append(pid)
        if len(members) >= 3 and scrubbed:
            break
        time.sleep(0.05)
    try:
        assert len(members) >= 3
        assert scrubbed
        assert olmoe._owned_group_members(running.owned) == members
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
        remaining = olmoe._process_group_members(process_group)
        assert remaining
        assert process.pid not in remaining
    finally:
        cleanup = olmoe.stop_server(running, 2.0)

    assert cleanup["cleanup_complete"] is True


def test_owned_group_rejects_same_pgid_from_different_session() -> None:
    owned = olmoe.OwnedServerProcess(
        pid=100,
        process_group_id=100,
        start_time_ticks=1_000,
        owner_token="owner",
        ownership_namespace="namespace",
        command=("/runtime/python",),
        launch_environment=(),
        log_path="/tmp/server.log",
    )
    unrelated = olmoe.ProcessStatIdentity(
        parent_pid=99,
        process_group_id=100,
        session_id=99,
        start_time_ticks=1_001,
        state="S",
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="unowned"):
        olmoe._validate_owned_member_identity(owned, 101, unrelated)


def test_port_preflight_rejects_stale_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OccupiedSocket:
        def bind(self, _address: object) -> None:
            raise OSError("address already in use")

        def listen(self, _backlog: int) -> None:
            raise AssertionError("listen must not follow a failed bind")

        def close(self) -> None:
            pass

    monkeypatch.setattr(olmoe.socket, "socket", lambda *_args: _OccupiedSocket())

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="occupied"):
        olmoe.verify_port_vacant("127.0.0.1", 62_610)


def test_unexpected_child_exit_is_rejected() -> None:
    fake = cast(
        olmoe.RunningServerProcess,
        SimpleNamespace(process=SimpleNamespace(poll=lambda: 7)),
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="code 7"):
        olmoe.assert_server_alive(fake, "sample")


def test_cleanup_rejects_process_that_exited_before_managed_stop(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        olmoe._OWNER_TOKEN_ENVIRONMENT: "owner",
        olmoe._OWNERSHIP_NAMESPACE_ENVIRONMENT: "namespace",
    }
    log_path = tmp_path / "early-exit.log"
    log_file = log_path.open("xb", buffering=0)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.05)"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    process_group, start_ticks, _state = olmoe._read_process_stat(process.pid)
    owned = olmoe.OwnedServerProcess(
        pid=process.pid,
        process_group_id=process_group,
        start_time_ticks=start_ticks,
        owner_token="owner",
        ownership_namespace="namespace",
        command=(sys.executable,),
        launch_environment=(),
        log_path=str(log_path),
    )
    process.wait(timeout=2.0)

    cleanup = olmoe.stop_server(
        olmoe.RunningServerProcess(owned, process, log_file), 1.0
    )

    assert cleanup["cleanup_complete"] is False
    assert "before managed cleanup" in cast(str, cleanup["failure"])


def test_argument_parser_uses_trusted_stage_contract(tmp_path: Path) -> None:
    arguments = olmoe._parser().parse_args(
        [
            "--run-id",
            "ep2",
            "--result-directory",
            str(tmp_path / "result"),
            "--runtime-python",
            "/runtime/python",
            "--runtime-install-receipt",
            "/runtime/install.json",
            "--runtime-install-receipt-sha256",
            SHA,
            "--stage-contract",
            "/stage/contract.json",
            "--ep-size",
            "2",
        ]
    )

    config = olmoe.config_from_arguments(arguments)

    assert config.model_path == olmoe.OLMOE_MODEL_PATH
    assert config.stage_contract == Path("/stage/contract.json")
    assert config.stage_capture_output is None
    assert olmoe.CANONICAL_WARMUP_COUNT == 2
    assert olmoe.CANONICAL_SAMPLE_COUNT == 3


def test_argument_parser_selects_managed_stage_capture(tmp_path: Path) -> None:
    arguments = olmoe._parser().parse_args(
        [
            "--run-id",
            "ep1-capture",
            "--result-directory",
            str(tmp_path / "result"),
            "--runtime-python",
            "/runtime/python",
            "--runtime-install-receipt",
            "/runtime/install.json",
            "--runtime-install-receipt-sha256",
            SHA,
            "--stage-capture-output",
            "/stage/ep1.json",
            "--ep-size",
            "1",
        ]
    )

    config = olmoe.config_from_arguments(arguments)

    assert config.stage_contract is None
    assert config.stage_capture_output == Path("/stage/ep1.json")


@pytest.mark.parametrize(
    "capture_name",
    [
        olmoe._RESULT_FILENAME,
        olmoe._SERVER_LOG_FILENAME,
        olmoe._OWNERSHIP_JOURNAL_FILENAME,
        ".olmoe-ep-local-benchmark-result.json.pending.tmp",
        "capture.json",
    ],
)
def test_stage_capture_output_cannot_alias_run_result_tree(
    tmp_path: Path, capture_name: str
) -> None:
    result_directory = tmp_path / "result"
    arguments = olmoe._parser().parse_args(
        [
            "--run-id",
            "ep1-capture",
            "--result-directory",
            str(result_directory),
            "--runtime-python",
            "/runtime/python",
            "--runtime-install-receipt",
            "/runtime/install.json",
            "--runtime-install-receipt-sha256",
            SHA,
            "--stage-capture-output",
            str(result_directory / capture_name),
            "--ep-size",
            "1",
        ]
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="outside"):
        olmoe.config_from_arguments(arguments)


def test_stage_capture_output_rejects_resolved_parent_alias(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    arguments = olmoe._parser().parse_args(
        [
            "--run-id",
            "ep1-capture",
            "--result-directory",
            str(alias / "result"),
            "--runtime-python",
            "/runtime/python",
            "--runtime-install-receipt",
            "/runtime/install.json",
            "--runtime-install-receipt-sha256",
            SHA,
            "--stage-capture-output",
            str(tmp_path / "result/capture.json"),
            "--ep-size",
            "1",
        ]
    )

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="outside"):
        olmoe.config_from_arguments(arguments)


class _FakeCpuPolicySession:
    def __init__(self, *, restore_succeeds: bool) -> None:
        self.restore_succeeds = restore_succeeds
        self.evidence: olmoe.JsonObject = {
            "lifecycle": "new",
            "application_verified": False,
            "restoration_verified": False,
            "policies": [{"name": "policy0"}],
            "snapshots": {},
            "serialization": {
                "journal_published": True,
                "journal_removed": False,
                "lock_acquired": False,
                "lock_released": False,
            },
            "failures": [],
        }

    def __enter__(self) -> _FakeCpuPolicySession:
        self.evidence["lifecycle"] = "active"
        self.evidence["application_verified"] = True
        snapshots = cast(dict[str, object], self.evidence["snapshots"])
        snapshots["before"] = {}
        snapshots["active"] = {}
        return self

    def __exit__(self, *_arguments: object) -> bool:
        serialization = cast(dict[str, object], self.evidence["serialization"])
        snapshots = cast(dict[str, object], self.evidence["snapshots"])
        snapshots["performance_after"] = {}
        snapshots["restored"] = {}
        serialization["lock_released"] = True
        if not self.restore_succeeds:
            self.evidence["lifecycle"] = "failed"
            cast(list[object], self.evidence["failures"]).append(
                {"phase": "restoration"}
            )
            raise RuntimeError("injected CPU policy restore failure")
        self.evidence["lifecycle"] = "restored"
        self.evidence["restoration_verified"] = True
        serialization["journal_published"] = False
        serialization["journal_removed"] = True
        return False


def _fake_policy_factory(
    session: _FakeCpuPolicySession,
) -> olmoe.CpuPerformancePolicyFactory:
    return lambda: cast(olmoe.CpuPerformancePolicySession, cast(object, session))


def test_outer_cpu_policy_transaction_wraps_run_and_records_restore(
    tmp_path: Path,
) -> None:
    config = _base_config(tmp_path)
    session = _FakeCpuPolicySession(restore_succeeds=True)

    def managed_run(run_config: olmoe.OlmoeEpBenchmarkConfig) -> olmoe.JsonObject:
        assert session.evidence["lifecycle"] == "active"
        run_config.result_directory.mkdir(mode=0o700)
        return {
            "schema_version": 1,
            "status": "policy_restore_pending",
            "run_id": run_config.run_id,
        }

    payload = olmoe._run_with_cpu_performance_policy(
        config,
        managed_run,
        stage_capture=False,
        policy_factory=_fake_policy_factory(session),
    )

    policy = cast(dict[str, object], payload["cpu_performance_policy"])
    assert policy["status"] == "performance_policy_applied_and_restored"
    assert policy["transaction_scope"] == "one_outer_harness_transaction"
    assert policy["nested_or_rank_transactions"] is False
    evidence = cast(dict[str, object], policy["evidence"])
    assert evidence["lifecycle"] == "restored"
    serialization = cast(dict[str, object], evidence["serialization"])
    assert serialization["journal_published"] is False
    assert serialization["journal_removed"] is True
    assert (
        json.loads((config.result_directory / olmoe._RESULT_FILENAME).read_text())[
            "cpu_performance_policy"
        ]["evidence"]["restoration_verified"]
        is True
    )


def test_cpu_policy_receipt_rejects_journal_still_published_after_restore() -> None:
    session = _FakeCpuPolicySession(restore_succeeds=True)
    session.__enter__()
    session.__exit__(None, None, None)
    serialization = cast(dict[str, object], session.evidence["serialization"])
    serialization["journal_published"] = True

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="did not verify"):
        olmoe._cpu_performance_policy_receipt(
            cast(olmoe.CpuPerformancePolicySession, cast(object, session))
        )


def test_stage_capture_json_reloads_and_publishes_only_after_policy_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_output = tmp_path / "ep1-capture.json"
    config = replace(
        _base_config(tmp_path),
        stage_contract=None,
        stage_capture_output=capture_output,
        expert_parallel_size=1,
    )
    session = _FakeCpuPolicySession(restore_succeeds=True)
    capture = _capture(1)
    real_publish = olmoe.publish_sanity_capture
    publication_lifecycles: list[object] = []

    def publish_after_restore(*, capture: olmoe.SanityCapture, output: Path) -> str:
        publication_lifecycles.append(session.evidence["lifecycle"])
        return real_publish(capture=capture, output=output)

    monkeypatch.setattr(olmoe, "publish_sanity_capture", publish_after_restore)

    def managed_run(run_config: olmoe.OlmoeEpBenchmarkConfig) -> olmoe.JsonObject:
        assert session.evidence["lifecycle"] == "active"
        run_config.result_directory.mkdir(mode=0o700)
        return {
            "schema_version": 1,
            "status": "policy_restore_pending",
            "run_id": run_config.run_id,
            "capture": cast(olmoe.JsonObject, capture.model_dump(mode="json")),
        }

    payload = olmoe._run_with_cpu_performance_policy(
        config,
        managed_run,
        stage_capture=True,
        policy_factory=_fake_policy_factory(session),
    )

    assert publication_lifecycles == ["restored"]
    assert payload["status"] == "passed"
    assert isinstance(payload["capture_receipt_sha256"], str)
    assert capture_output.is_file()
    assert (
        olmoe.SanityCapture.model_validate_json(capture_output.read_bytes()) == capture
    )


def test_outer_cpu_policy_restore_failure_fails_completed_workload(
    tmp_path: Path,
) -> None:
    config = _base_config(tmp_path)
    session = _FakeCpuPolicySession(restore_succeeds=False)

    def managed_run(run_config: olmoe.OlmoeEpBenchmarkConfig) -> olmoe.JsonObject:
        run_config.result_directory.mkdir(mode=0o700)
        return {
            "schema_version": 1,
            "status": "policy_restore_pending",
            "run_id": run_config.run_id,
        }

    with pytest.raises(olmoe.OlmoeEpBenchmarkError, match="restore failure"):
        olmoe._run_with_cpu_performance_policy(
            config,
            managed_run,
            stage_capture=False,
            policy_factory=_fake_policy_factory(session),
        )

    receipt = json.loads((config.result_directory / olmoe._RESULT_FILENAME).read_text())
    assert receipt["status"] == "failed"
    assert receipt["cpu_performance_policy"]["status"] == "failed"
