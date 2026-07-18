from typing import ClassVar, Self, cast

from pydantic import field_validator, model_validator

from exo.shared.types.common import Id
from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import TaggedModel


class ComputeResourceId(Id):
    NVIDIA_GPU_PREFIX: ClassVar[str] = "nvidia-gpu:"

    @classmethod
    def from_nvidia_device_uuid(cls, device_uuid: str) -> Self:
        normalized_device_uuid = device_uuid.strip()
        if not normalized_device_uuid:
            raise ValueError("NVIDIA device UUID must not be empty")
        return cls(f"{cls.NVIDIA_GPU_PREFIX}{normalized_device_uuid}")

    def nvidia_device_uuid(self) -> str:
        if not self.startswith(self.NVIDIA_GPU_PREFIX):
            raise ValueError(f"Compute resource {self} is not an NVIDIA GPU")
        device_uuid = self.removeprefix(self.NVIDIA_GPU_PREFIX)
        if not device_uuid:
            raise ValueError("NVIDIA compute resource must include a device UUID")
        return device_uuid


class NvidiaGpuComputeResource(TaggedModel):
    resource_id: ComputeResourceId
    device_uuid: str
    pci_bus_id: str
    model_name: str
    total_memory: Memory
    numa_node: int | None = None
    cpu_affinity: tuple[int, ...] = ()

    @field_validator("cpu_affinity", mode="before")
    @classmethod
    def deserialize_cpu_affinity(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        return value

    @classmethod
    def from_device(
        cls,
        *,
        device_uuid: str,
        pci_bus_id: str,
        model_name: str,
        total_memory_bytes: int,
        numa_node: int | None = None,
        cpu_affinity: tuple[int, ...] = (),
    ) -> Self:
        return cls(
            resource_id=ComputeResourceId.from_nvidia_device_uuid(device_uuid),
            device_uuid=device_uuid.strip(),
            pci_bus_id=pci_bus_id.strip(),
            model_name=model_name.strip(),
            total_memory=Memory.from_bytes(total_memory_bytes),
            numa_node=numa_node,
            cpu_affinity=cpu_affinity,
        )

    @model_validator(mode="after")
    def validate_hardware_identity(self) -> "NvidiaGpuComputeResource":
        if not self.device_uuid:
            raise ValueError("NVIDIA device UUID must not be empty")
        if self.resource_id != ComputeResourceId.from_nvidia_device_uuid(
            self.device_uuid
        ):
            raise ValueError("Compute resource ID must be derived from the device UUID")
        if not self.pci_bus_id:
            raise ValueError("NVIDIA PCI bus ID must not be empty")
        if not self.model_name:
            raise ValueError("NVIDIA model name must not be empty")
        if self.total_memory.in_bytes <= 0:
            raise ValueError("NVIDIA total memory must be positive")
        if self.numa_node is not None and self.numa_node < 0:
            raise ValueError("NVIDIA NUMA node must not be negative")
        if any(cpu_id < 0 for cpu_id in self.cpu_affinity):
            raise ValueError("NVIDIA CPU affinity must not contain negative CPU IDs")
        if self.cpu_affinity != tuple(sorted(set(self.cpu_affinity))):
            raise ValueError("NVIDIA CPU affinity must be sorted and unique")
        return self


ComputeResource = NvidiaGpuComputeResource
