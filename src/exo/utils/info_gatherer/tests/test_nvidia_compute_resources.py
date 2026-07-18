from exo.utils.info_gatherer.nvidia_compute_resources import (
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


def test_nvml_discovery_returns_every_gpu_and_releases_nvml() -> None:
    management_api = FakeNvidiaManagementApi()

    resources = gather_nvidia_gpu_compute_resources_from_api(management_api)

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


def test_nvml_availability_check_releases_nvml() -> None:
    management_api = FakeNvidiaManagementApi()

    assert has_nvidia_gpu_from_api(management_api)
    assert management_api.initialized
    assert management_api.was_shutdown
