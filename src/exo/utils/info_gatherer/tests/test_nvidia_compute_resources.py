from pathlib import Path

from exo.utils.info_gatherer.nvidia_compute_resources import (
    LinuxSysfsNvidiaGpuLocalityProbe,
    gather_nvidia_gpu_compute_resources_from_api,
    has_nvidia_gpu_from_api,
)


class FakeNvidiaManagementApi:
    def __init__(self) -> None:
        self.initialized = False
        self.was_shutdown = False

    def initialize(self) -> None:
        self.initialized = True

    def shutdown(self) -> None:
        self.was_shutdown = True

    def get_device_count(self) -> int:
        return 2

    def get_device_handle(self, index: int) -> object:
        return index

    def get_device_uuid(self, handle: object) -> str | bytes:
        return f"GPU-00000000-0000-0000-0000-00000000000{handle}"

    def get_device_name(self, handle: object) -> str | bytes:
        return b"NVIDIA GeForce RTX 3090"

    def get_device_memory_bytes(self, handle: object) -> int:
        return 24 * 1024**3

    def get_device_pci_bus_id(self, handle: object) -> str | bytes:
        return f"00000000:2{handle}:00.0"


def test_nvml_discovery_returns_every_gpu_and_releases_nvml(tmp_path: Path) -> None:
    management_api = FakeNvidiaManagementApi()
    locality_probe = LinuxSysfsNvidiaGpuLocalityProbe(tmp_path)

    resources = gather_nvidia_gpu_compute_resources_from_api(
        management_api, locality_probe=locality_probe
    )

    assert management_api.initialized
    assert management_api.was_shutdown
    assert [resource.device_uuid for resource in resources] == [
        "GPU-00000000-0000-0000-0000-000000000000",
        "GPU-00000000-0000-0000-0000-000000000001",
    ]
    assert [resource.pci_bus_id for resource in resources] == [
        "00000000:20:00.0",
        "00000000:21:00.0",
    ]
    assert all(resource.total_memory.in_gb == 24 for resource in resources)
    assert all(resource.numa_node is None for resource in resources)
    assert all(resource.cpu_affinity == () for resource in resources)


def test_sysfs_locality_normalizes_nvml_domain_and_parses_cpu_ranges(
    tmp_path: Path,
) -> None:
    device_path = tmp_path / "0000:20:00.0"
    device_path.mkdir()
    (device_path / "numa_node").write_text("2\n")
    (device_path / "local_cpulist").write_text("0-3,8,10-12\n")

    resources = gather_nvidia_gpu_compute_resources_from_api(
        FakeNvidiaManagementApi(),
        locality_probe=LinuxSysfsNvidiaGpuLocalityProbe(tmp_path),
    )

    assert resources[0].numa_node == 2
    assert resources[0].cpu_affinity == (0, 1, 2, 3, 8, 10, 11, 12)
    assert resources[1].numa_node is None
    assert resources[1].cpu_affinity == ()


def test_sysfs_minus_one_numa_node_and_malformed_cpu_list_are_unknown(
    tmp_path: Path,
) -> None:
    device_path = tmp_path / "0000:20:00.0"
    device_path.mkdir()
    (device_path / "numa_node").write_text("-1\n")
    (device_path / "local_cpulist").write_text("0-3,not-a-cpu\n")
    locality = LinuxSysfsNvidiaGpuLocalityProbe(tmp_path).get_locality(
        "00000000:20:00.0"
    )

    assert locality.numa_node is None
    assert locality.cpu_affinity == ()


def test_sysfs_unreadable_attributes_degrade_independently(tmp_path: Path) -> None:
    def read_sysfs_text(path: Path) -> str:
        if path.name == "numa_node":
            raise PermissionError(path)
        return "4-5"

    locality = LinuxSysfsNvidiaGpuLocalityProbe(
        tmp_path, text_reader=read_sysfs_text
    ).get_locality("0000:20:00.0")

    assert locality.numa_node is None
    assert locality.cpu_affinity == (4, 5)


def test_sysfs_invalid_pci_bus_id_does_not_read_files(tmp_path: Path) -> None:
    def fail_on_read(path: Path) -> str:
        raise AssertionError(f"unexpected sysfs read: {path}")

    locality = LinuxSysfsNvidiaGpuLocalityProbe(
        tmp_path, text_reader=fail_on_read
    ).get_locality("not-a-pci-address")

    assert locality.numa_node is None
    assert locality.cpu_affinity == ()


def test_nvml_availability_check_releases_nvml() -> None:
    management_api = FakeNvidiaManagementApi()

    assert has_nvidia_gpu_from_api(management_api)
    assert management_api.initialized
    assert management_api.was_shutdown
