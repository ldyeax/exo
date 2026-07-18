from importlib import import_module
from types import ModuleType
from typing import Callable, Protocol, cast

from exo.shared.types.compute_resources import NvidiaGpuComputeResource


class NvidiaManagementApi(Protocol):
    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def get_device_count(self) -> int: ...

    def get_device_handle(self, index: int) -> object: ...

    def get_device_uuid(self, handle: object) -> str | bytes: ...

    def get_device_name(self, handle: object) -> str | bytes: ...

    def get_device_memory_bytes(self, handle: object) -> int: ...

    def get_device_pci_bus_id(self, handle: object) -> str | bytes: ...


class PynvmlAdapter:
    def __init__(self, module: ModuleType) -> None:
        self._module = module

    def _function(self, name: str) -> Callable[..., object]:
        value = cast(object, getattr(self._module, name))
        if not callable(value):
            raise TypeError(f"pynvml attribute {name} is not callable")
        return value

    def initialize(self) -> None:
        self._function("nvmlInit")()

    def shutdown(self) -> None:
        self._function("nvmlShutdown")()

    def get_device_count(self) -> int:
        return cast(int, self._function("nvmlDeviceGetCount")())

    def get_device_handle(self, index: int) -> object:
        return self._function("nvmlDeviceGetHandleByIndex")(index)

    def get_device_uuid(self, handle: object) -> str | bytes:
        return cast(str | bytes, self._function("nvmlDeviceGetUUID")(handle))

    def get_device_name(self, handle: object) -> str | bytes:
        return cast(str | bytes, self._function("nvmlDeviceGetName")(handle))

    def get_device_memory_bytes(self, handle: object) -> int:
        memory_info = self._function("nvmlDeviceGetMemoryInfo")(handle)
        total_memory = _get_dynamic_attribute(memory_info, "total")
        if not isinstance(total_memory, int):
            raise TypeError("pynvml memory total must be an integer")
        return total_memory

    def get_device_pci_bus_id(self, handle: object) -> str | bytes:
        pci_info = self._function("nvmlDeviceGetPciInfo")(handle)
        return cast(str | bytes, _get_dynamic_attribute(pci_info, "busId"))


def _decode_nvml_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _get_dynamic_attribute(value: object, name: str) -> object:
    attribute = cast(object, getattr(value, name))
    return attribute


def gather_nvidia_gpu_compute_resources_from_api(
    management_api: NvidiaManagementApi,
) -> list[NvidiaGpuComputeResource]:
    management_api.initialize()
    try:
        resources: list[NvidiaGpuComputeResource] = []
        for device_index in range(management_api.get_device_count()):
            handle = management_api.get_device_handle(device_index)
            resources.append(
                NvidiaGpuComputeResource.from_device(
                    device_uuid=_decode_nvml_text(
                        management_api.get_device_uuid(handle)
                    ),
                    pci_bus_id=_decode_nvml_text(
                        management_api.get_device_pci_bus_id(handle)
                    ),
                    model_name=_decode_nvml_text(
                        management_api.get_device_name(handle)
                    ),
                    total_memory_bytes=management_api.get_device_memory_bytes(handle),
                )
            )
        return resources
    finally:
        management_api.shutdown()


def has_nvidia_gpu_from_api(management_api: NvidiaManagementApi) -> bool:
    management_api.initialize()
    try:
        return management_api.get_device_count() > 0
    finally:
        management_api.shutdown()


def gather_nvidia_gpu_compute_resources() -> list[NvidiaGpuComputeResource]:
    try:
        management_api = PynvmlAdapter(import_module("pynvml"))
        return gather_nvidia_gpu_compute_resources_from_api(management_api)
    except Exception:
        return []


def has_nvidia_gpu() -> bool:
    try:
        management_api = PynvmlAdapter(import_module("pynvml"))
        return has_nvidia_gpu_from_api(management_api)
    except Exception:
        return False
