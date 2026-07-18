import os
import resource
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxNcclInstance
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

logger: "loguru.Logger" = loguru.logger

DEFAULT_INFINIBAND_DEVICES_PATH = Path("/sys/class/infiniband")


@dataclass(frozen=True)
class RunnerTerminationError:
    exception_type: str
    exception_message: str
    exception_repr: str
    traceback: str

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(
            exception_type=type(e).__qualname__,
            exception_message=str(e),
            exception_repr=repr(e),
            traceback="".join(
                traceback.TracebackException.from_exception(e).format(chain=True)
            ),
        )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.exception_message}\n{self.traceback}"


def _infiniband_driver_modules(infiniband_devices_path: Path) -> dict[str, str]:
    try:
        infiniband_devices = infiniband_devices_path.iterdir()
    except OSError:
        return {}

    driver_modules: dict[str, str] = {}
    for infiniband_device in infiniband_devices:
        driver_module_path = infiniband_device / "device" / "driver" / "module"
        try:
            driver_module_target = os.readlink(driver_module_path)
        except OSError:
            continue
        driver_modules[infiniband_device.name] = Path(driver_module_target).name
    return driver_modules


def _nccl_hca_pattern_matches(
    device_name: str, pattern: str, *, exact_match: bool
) -> bool:
    exact_match = exact_match or pattern.startswith("=")
    device_pattern = pattern.removeprefix("=").partition(":")[0]
    if not device_pattern:
        return False
    return (
        device_name == device_pattern
        if exact_match
        else device_name.startswith(device_pattern)
    )


def _has_eligible_mlx4_infiniband_device(
    infiniband_devices_path: Path,
    hca_selector: str | None,
) -> bool:
    driver_modules = _infiniband_driver_modules(infiniband_devices_path)
    if not hca_selector:
        eligible_device_names = set(driver_modules)
    else:
        selector = hca_selector.strip()
        exclusion = selector.startswith("^")
        selector = selector.removeprefix("^")
        exact_match = selector.startswith("=")
        patterns = tuple(
            pattern.strip()
            for pattern in selector.removeprefix("=").split(",")
            if pattern.strip()
        )
        matching_device_names = {
            device_name
            for device_name in driver_modules
            if any(
                _nccl_hca_pattern_matches(
                    device_name,
                    pattern,
                    exact_match=exact_match,
                )
                for pattern in patterns
            )
        }
        eligible_device_names = (
            set(driver_modules) - matching_device_names
            if exclusion
            else matching_device_names
        )

    return any(
        driver_modules[device_name] == "mlx4_core"
        for device_name in eligible_device_names
    )


def configure_runner_environment(
    bound_instance: BoundInstance,
    infiniband_devices_path: Path = DEFAULT_INFINIBAND_DEVICES_PATH,
) -> None:
    if isinstance(bound_instance.instance, MlxNcclInstance):
        compute_resource_ids = bound_instance.bound_compute_resource_ids
        if compute_resource_ids:
            if len(compute_resource_ids) != 1:
                raise ValueError(
                    "MLX NCCL runners require exactly one compute resource"
                )
            device_uuid = compute_resource_ids[0].nvidia_device_uuid()
            os.environ["CUDA_VISIBLE_DEVICES"] = device_uuid
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        if _has_eligible_mlx4_infiniband_device(
            infiniband_devices_path,
            os.environ.get("NCCL_IB_HCA"),
        ):
            os.environ.setdefault("NCCL_GIN_ENABLE", "0")
            os.environ.setdefault("NCCL_GIN_TYPE", "0")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    configure_runner_environment(bound_instance)
    if isinstance(bound_instance.instance, MlxNcclInstance):
        logger.info(
            f"NCCL runner CUDA visibility: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    if sys.platform == "darwin":
        fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
        if fast_synch_override == "false":
            os.environ["MLX_METAL_FAST_SYNCH"] = "0"
        else:
            os.environ["MLX_METAL_FAST_SYNCH"] = "1"

        logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        event_sender_downcast: MpSender[Event] = cast(MpSender[Event], event_sender)

        from exo.worker.runner.runner import Runner

        builder: Builder
        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender_downcast, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender_downcast,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender_downcast, task_receiver)
        runner.main()
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(RunnerTerminationError.from_exception(e))
        raise SystemExit(1) from e
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
