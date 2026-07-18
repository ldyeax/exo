from typing import ClassVar, Self

from pydantic import model_validator

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

    @classmethod
    def from_device(
        cls,
        *,
        device_uuid: str,
        pci_bus_id: str,
        model_name: str,
        total_memory_bytes: int,
    ) -> Self:
        return cls(
            resource_id=ComputeResourceId.from_nvidia_device_uuid(device_uuid),
            device_uuid=device_uuid.strip(),
            pci_bus_id=pci_bus_id.strip(),
            model_name=model_name.strip(),
            total_memory=Memory.from_bytes(total_memory_bytes),
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
        return self


ComputeResource = NvidiaGpuComputeResource
