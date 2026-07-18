import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, cast

from exo.shared.types.compute_resources import NvidiaGpuComputeResource

_LINUX_PCI_BUS_ID_PATTERN = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)
_DEFAULT_LINUX_PCI_DEVICES_PATH = Path("/sys/bus/pci/devices")


@dataclass(frozen=True, slots=True)
class NvidiaGpuLocality:
    numa_node: int | None = None
    cpu_affinity: tuple[int, ...] = ()


class NvidiaGpuLocalityProbe(Protocol):
    def get_locality(self, pci_bus_id: str) -> NvidiaGpuLocality: ...


type SysfsTextReader = Callable[[Path], str]


def _read_text(path: Path) -> str:
    return path.read_text()


class LinuxSysfsNvidiaGpuLocalityProbe:
    def __init__(
        self,
        pci_devices_path: Path = _DEFAULT_LINUX_PCI_DEVICES_PATH,
        text_reader: SysfsTextReader = _read_text,
    ) -> None:
        self._pci_devices_path = pci_devices_path
        self._text_reader = text_reader

    def get_locality(self, pci_bus_id: str) -> NvidiaGpuLocality:
        normalized_bus_id = _normalize_linux_pci_bus_id(pci_bus_id)
        if normalized_bus_id is None:
            return NvidiaGpuLocality()

        device_path = self._pci_devices_path / normalized_bus_id
        numa_node_text = self._read_optional_text(device_path / "numa_node")
        cpu_list_text = self._read_optional_text(device_path / "local_cpulist")
        return NvidiaGpuLocality(
            numa_node=_parse_numa_node(numa_node_text),
            cpu_affinity=_parse_linux_cpu_list(cpu_list_text),
        )

    def _read_optional_text(self, path: Path) -> str | None:
        try:
            return self._text_reader(path).strip()
        except (OSError, UnicodeError):
            return None


def _normalize_linux_pci_bus_id(pci_bus_id: str) -> str | None:
    match = _LINUX_PCI_BUS_ID_PATTERN.fullmatch(pci_bus_id.strip().rstrip("\0"))
    if match is None:
        return None

    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        return None
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def _parse_numa_node(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        numa_node = int(value)
    except ValueError:
        return None
    return numa_node if numa_node >= 0 else None


def _parse_linux_cpu_list(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()

    cpu_ids: set[int] = set()
    for item in value.strip().split(","):
        bounds = item.strip().split("-")
        if not 1 <= len(bounds) <= 2 or not all(bound.isdecimal() for bound in bounds):
            return ()
        first_cpu = int(bounds[0])
        last_cpu = int(bounds[-1])
        if first_cpu > last_cpu:
            return ()
        cpu_ids.update(range(first_cpu, last_cpu + 1))
    return tuple(sorted(cpu_ids))


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
    *,
    locality_probe: NvidiaGpuLocalityProbe | None = None,
) -> list[NvidiaGpuComputeResource]:
    resolved_locality_probe = (
        LinuxSysfsNvidiaGpuLocalityProbe() if locality_probe is None else locality_probe
    )
    management_api.initialize()
    try:
        resources: list[NvidiaGpuComputeResource] = []
        for device_index in range(management_api.get_device_count()):
            handle = management_api.get_device_handle(device_index)
            pci_bus_id = _decode_nvml_text(management_api.get_device_pci_bus_id(handle))
            locality = resolved_locality_probe.get_locality(pci_bus_id)
            resources.append(
                NvidiaGpuComputeResource.from_device(
                    device_uuid=_decode_nvml_text(
                        management_api.get_device_uuid(handle)
                    ),
                    pci_bus_id=pci_bus_id,
                    model_name=_decode_nvml_text(
                        management_api.get_device_name(handle)
                    ),
                    total_memory_bytes=management_api.get_device_memory_bytes(handle),
                    numa_node=locality.numa_node,
                    cpu_affinity=locality.cpu_affinity,
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
