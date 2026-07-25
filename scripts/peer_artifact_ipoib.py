#!/usr/bin/env python3
"""Provision pinned IPoIB rails and render fwuff/dwagon artifact configs.

The network operation is dry-run-only unless ``provision --apply`` is given.
It never starts, stops, reloads, or reconfigures OpenSM.  The topology is bound
to HCA and port GUIDs so a slot move or probe-order change fails closed instead
of assigning an address to the wrong fabric.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
TransportMode = Literal["connected", "datagram"]
ProfileName = Literal["all", "four-link"]

TOPOLOGY_SCHEMA_VERSION: Final = 1
NETWORK_RECEIPT_SCHEMA_VERSION: Final = 1
DEFAULT_TOPOLOGY_PATH: Final = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "peer_artifact"
    / "fwuffydwagon-five-link-topology.json"
)
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GUID = re.compile(r"^(?:[0-9a-f]{4}:){3}[0-9a-f]{4}$")
_IPOIB_HARDWARE_ADDRESS = re.compile(
    r"^(?:[0-9a-f]{2}:){19}[0-9a-f]{2}$"
)
_CONFLICTING_COMMAND_MARKERS = (
    "all_gather_perf",
    "all_reduce_perf",
    "ib_read_bw",
    "ib_read_lat",
    "ib_send_bw",
    "ib_send_lat",
    "ib_write_bw",
    "ib_write_lat",
    "nccl-tests",
    "run_sglang",
    "sglang.launch_server",
    "torchrun",
)


class ProvisioningError(RuntimeError):
    """Raised when a host cannot be changed without guessing."""


@dataclass(frozen=True)
class HostSpec:
    node_id: str
    hostname: str


@dataclass(frozen=True)
class InfiniBandHostSpec:
    node_guid: str
    port_guid: str
    address: ipaddress.IPv4Interface


@dataclass(frozen=True)
class InfiniBandRailSpec:
    link_id: str
    hca_name: str
    port: int
    minimum_rate_gbps: int
    transport_mode: TransportMode
    mtu: int
    estimated_bytes_per_second: int
    maximum_concurrent_chunks: int
    hosts: Mapping[str, InfiniBandHostSpec]


@dataclass(frozen=True)
class EthernetHostSpec:
    interface: str
    address: ipaddress.IPv4Address


@dataclass(frozen=True)
class EthernetRailSpec:
    link_id: str
    estimated_bytes_per_second: int
    maximum_concurrent_chunks: int
    hosts: Mapping[str, EthernetHostSpec]


@dataclass(frozen=True)
class ServedSnapshotSpec:
    model_id: str
    revision: str
    relative_directory: str


@dataclass(frozen=True)
class DeploymentSpec:
    source_host: str
    receiver_host: str
    served_root: Path
    manifest_cache_directory: Path
    receiver_disk_cache_directory: Path
    receiver_memory_cache_directory: Path
    receiver_materialization_root: Path
    served_snapshots: tuple[ServedSnapshotSpec, ...]


@dataclass(frozen=True)
class Topology:
    schema_version: int
    cluster_id: str
    api_port: int
    hosts: Mapping[str, HostSpec]
    infiniband_rails: tuple[InfiniBandRailSpec, ...]
    ethernet_rails: tuple[EthernetRailSpec, ...]
    deployment: DeploymentSpec
    sha256: str


@dataclass(frozen=True)
class HcaPortObservation:
    hca_name: str
    port: int
    node_guid: str
    port_guid: str
    state: str
    physical_state: str
    rate_gbps: int
    subnet_manager_lid: int
    local_identifier: int
    interface: str | None


@dataclass(frozen=True)
class InterfaceObservation:
    interface: str
    transport_mode: str
    mtu: int
    is_up: bool
    ipv4_addresses: tuple[str, ...]


def _object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ProvisioningError(f"{description} must be a JSON object")
    return cast(dict[str, object], value)


def _array(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise ProvisioningError(f"{description} must be a JSON array")
    return cast(list[object], value)


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProvisioningError(f"{description} must be a nonempty string")
    return value


def _integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProvisioningError(f"{description} must be an integer")
    return value


def _identifier(value: object, description: str) -> str:
    result = _string(value, description)
    if _SAFE_IDENTIFIER.fullmatch(result) is None:
        raise ProvisioningError(f"{description} is not a safe identifier")
    return result


def _guid(value: object, description: str) -> str:
    result = _string(value, description).lower()
    if _GUID.fullmatch(result) is None:
        raise ProvisioningError(f"{description} is not a normalized GUID")
    return result


def _absolute_path(value: object, description: str) -> Path:
    result = Path(_string(value, description))
    if not result.is_absolute():
        raise ProvisioningError(f"{description} must be absolute")
    return result


def _parse_infiniband_host(
    value: object, description: str
) -> InfiniBandHostSpec:
    raw = _object(value, description)
    try:
        address = ipaddress.IPv4Interface(
            _string(raw["address"], f"{description}.address")
        )
    except (KeyError, ValueError) as error:
        raise ProvisioningError(
            f"{description}.address must be a valid IPv4 interface"
        ) from error
    return InfiniBandHostSpec(
        node_guid=_guid(raw.get("node_guid"), f"{description}.node_guid"),
        port_guid=_guid(raw.get("port_guid"), f"{description}.port_guid"),
        address=address,
    )


def _parse_ethernet_host(value: object, description: str) -> EthernetHostSpec:
    raw = _object(value, description)
    try:
        address = ipaddress.IPv4Address(
            _string(raw["address"], f"{description}.address")
        )
    except (KeyError, ValueError) as error:
        raise ProvisioningError(
            f"{description}.address must be a valid IPv4 address"
        ) from error
    return EthernetHostSpec(
        interface=_identifier(
            raw.get("interface"), f"{description}.interface"
        ),
        address=address,
    )


def load_topology(path: Path = DEFAULT_TOPOLOGY_PATH) -> Topology:
    try:
        contents = path.read_bytes()
        root = _object(json.loads(contents), "topology")
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"cannot load topology {path}") from error
    schema_version = _integer(root.get("schema_version"), "schema_version")
    if schema_version != TOPOLOGY_SCHEMA_VERSION:
        raise ProvisioningError("unsupported topology schema version")
    raw_hosts = _object(root.get("hosts"), "hosts")
    hosts: dict[str, HostSpec] = {}
    for host_name, value in raw_hosts.items():
        raw_host = _object(value, f"hosts.{host_name}")
        hosts[host_name] = HostSpec(
            node_id=_identifier(
                raw_host.get("node_id"), f"hosts.{host_name}.node_id"
            ),
            hostname=_identifier(
                raw_host.get("hostname"), f"hosts.{host_name}.hostname"
            ),
        )
    if len(hosts) != 2:
        raise ProvisioningError("the point-to-point topology requires two hosts")

    infiniband_rails: list[InfiniBandRailSpec] = []
    for index, value in enumerate(
        _array(root.get("infiniband_rails"), "infiniband_rails")
    ):
        description = f"infiniband_rails[{index}]"
        raw = _object(value, description)
        raw_rail_hosts = _object(raw.get("hosts"), f"{description}.hosts")
        rail_hosts = {
            host_name: _parse_infiniband_host(
                host_value, f"{description}.hosts.{host_name}"
            )
            for host_name, host_value in raw_rail_hosts.items()
        }
        mode = _string(
            raw.get("transport_mode"), f"{description}.transport_mode"
        )
        if mode not in {"connected", "datagram"}:
            raise ProvisioningError(
                f"{description}.transport_mode is unsupported"
            )
        rail = InfiniBandRailSpec(
            link_id=_identifier(
                raw.get("link_id"), f"{description}.link_id"
            ),
            hca_name=_identifier(
                raw.get("hca_name"), f"{description}.hca_name"
            ),
            port=_integer(raw.get("port"), f"{description}.port"),
            minimum_rate_gbps=_integer(
                raw.get("minimum_rate_gbps"),
                f"{description}.minimum_rate_gbps",
            ),
            transport_mode=cast(TransportMode, mode),
            mtu=_integer(raw.get("mtu"), f"{description}.mtu"),
            estimated_bytes_per_second=_integer(
                raw.get("estimated_bytes_per_second"),
                f"{description}.estimated_bytes_per_second",
            ),
            maximum_concurrent_chunks=_integer(
                raw.get("maximum_concurrent_chunks"),
                f"{description}.maximum_concurrent_chunks",
            ),
            hosts=rail_hosts,
        )
        if set(rail.hosts) != set(hosts):
            raise ProvisioningError(
                f"{description} must specify every topology host"
            )
        if rail.port <= 0 or rail.minimum_rate_gbps <= 0:
            raise ProvisioningError(f"{description} has invalid port or rate")
        if rail.mtu < 2044 or rail.mtu > 65520:
            raise ProvisioningError(f"{description}.mtu is outside IPoIB bounds")
        if (
            rail.estimated_bytes_per_second <= 0
            or rail.maximum_concurrent_chunks <= 0
        ):
            raise ProvisioningError(f"{description} has invalid scheduler weights")
        networks = {host.address.network for host in rail.hosts.values()}
        addresses = {host.address.ip for host in rail.hosts.values()}
        if len(networks) != 1 or len(addresses) != len(hosts):
            raise ProvisioningError(
                f"{description} endpoints must be unique in one subnet"
            )
        infiniband_rails.append(rail)

    ethernet_rails: list[EthernetRailSpec] = []
    for index, value in enumerate(
        _array(root.get("ethernet_rails"), "ethernet_rails")
    ):
        description = f"ethernet_rails[{index}]"
        raw = _object(value, description)
        raw_rail_hosts = _object(raw.get("hosts"), f"{description}.hosts")
        rail = EthernetRailSpec(
            link_id=_identifier(
                raw.get("link_id"), f"{description}.link_id"
            ),
            estimated_bytes_per_second=_integer(
                raw.get("estimated_bytes_per_second"),
                f"{description}.estimated_bytes_per_second",
            ),
            maximum_concurrent_chunks=_integer(
                raw.get("maximum_concurrent_chunks"),
                f"{description}.maximum_concurrent_chunks",
            ),
            hosts={
                host_name: _parse_ethernet_host(
                    host_value, f"{description}.hosts.{host_name}"
                )
                for host_name, host_value in raw_rail_hosts.items()
            },
        )
        if set(rail.hosts) != set(hosts):
            raise ProvisioningError(
                f"{description} must specify every topology host"
            )
        if (
            rail.estimated_bytes_per_second <= 0
            or rail.maximum_concurrent_chunks <= 0
        ):
            raise ProvisioningError(f"{description} has invalid scheduler weights")
        ethernet_rails.append(rail)

    link_ids = tuple(
        rail.link_id for rail in (*infiniband_rails, *ethernet_rails)
    )
    if len(set(link_ids)) != len(link_ids):
        raise ProvisioningError("topology link IDs must be unique")
    networks = tuple(
        next(iter({endpoint.address.network for endpoint in rail.hosts.values()}))
        for rail in infiniband_rails
    )
    for index, network in enumerate(networks):
        if any(network.overlaps(other) for other in networks[index + 1 :]):
            raise ProvisioningError("IPoIB rail subnets must not overlap")

    raw_deployment = _object(root.get("deployment"), "deployment")
    served_snapshots: list[ServedSnapshotSpec] = []
    for index, value in enumerate(
        _array(
            raw_deployment.get("served_snapshots"),
            "deployment.served_snapshots",
        )
    ):
        raw = _object(value, f"deployment.served_snapshots[{index}]")
        relative_directory = _string(
            raw.get("relative_directory"),
            f"deployment.served_snapshots[{index}].relative_directory",
        )
        if (
            Path(relative_directory).is_absolute()
            or ".." in Path(relative_directory).parts
        ):
            raise ProvisioningError("snapshot directories must be relative")
        served_snapshots.append(
            ServedSnapshotSpec(
                model_id=_string(
                    raw.get("model_id"),
                    f"deployment.served_snapshots[{index}].model_id",
                ),
                revision=_string(
                    raw.get("revision"),
                    f"deployment.served_snapshots[{index}].revision",
                ),
                relative_directory=relative_directory,
            )
        )
    deployment = DeploymentSpec(
        source_host=_identifier(
            raw_deployment.get("source_host"), "deployment.source_host"
        ),
        receiver_host=_identifier(
            raw_deployment.get("receiver_host"), "deployment.receiver_host"
        ),
        served_root=_absolute_path(
            raw_deployment.get("served_root"), "deployment.served_root"
        ),
        manifest_cache_directory=_absolute_path(
            raw_deployment.get("manifest_cache_directory"),
            "deployment.manifest_cache_directory",
        ),
        receiver_disk_cache_directory=_absolute_path(
            raw_deployment.get("receiver_disk_cache_directory"),
            "deployment.receiver_disk_cache_directory",
        ),
        receiver_memory_cache_directory=_absolute_path(
            raw_deployment.get("receiver_memory_cache_directory"),
            "deployment.receiver_memory_cache_directory",
        ),
        receiver_materialization_root=_absolute_path(
            raw_deployment.get("receiver_materialization_root"),
            "deployment.receiver_materialization_root",
        ),
        served_snapshots=tuple(served_snapshots),
    )
    if (
        deployment.source_host not in hosts
        or deployment.receiver_host not in hosts
        or deployment.source_host == deployment.receiver_host
    ):
        raise ProvisioningError("deployment source/receiver hosts are invalid")
    if not deployment.served_snapshots:
        raise ProvisioningError("deployment must serve at least one snapshot")
    api_port = _integer(root.get("api_port"), "api_port")
    if not 1 <= api_port <= 65535:
        raise ProvisioningError("api_port is outside the TCP port range")
    return Topology(
        schema_version=schema_version,
        cluster_id=_identifier(root.get("cluster_id"), "cluster_id"),
        api_port=api_port,
        hosts=hosts,
        infiniband_rails=tuple(infiniband_rails),
        ethernet_rails=tuple(ethernet_rails),
        deployment=deployment,
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ProvisioningError(f"cannot read {description} at {path}") from error


def _sysfs_guid(path: Path, description: str) -> str:
    value = _read_text(path, description).lower().removeprefix("0x")
    compact = value.replace(":", "")
    if len(compact) != 16 or any(character not in "0123456789abcdef" for character in compact):
        raise ProvisioningError(f"{description} is not a 64-bit GUID")
    return ":".join(compact[index : index + 4] for index in range(0, 16, 4))


def _port_guid(port_root: Path) -> str:
    gid = _read_text(port_root / "gids" / "0", "port GID").lower()
    try:
        packed = ipaddress.IPv6Address(gid).packed[-8:].hex()
    except ValueError as error:
        raise ProvisioningError(f"invalid port GID {gid}") from error
    return ":".join(packed[index : index + 4] for index in range(0, 16, 4))


def _state_name(value: str) -> str:
    return value.partition(":")[2].strip().upper() or value.strip().upper()


def _rate_gbps(value: str) -> int:
    match = re.search(r"(\d+)\s+Gb/sec", value)
    if match is None:
        raise ProvisioningError(f"cannot parse InfiniBand rate {value!r}")
    return int(match.group(1))


def _resolve_parent_ipoib_interface(
    hca_root: Path, port: int, sysfs_root: Path
) -> str | None:
    network_root = hca_root / "device" / "net"
    if not network_root.is_dir():
        return None
    candidates: list[str] = []
    for candidate in network_root.iterdir():
        network_device = sysfs_root / "class" / "net" / candidate.name
        try:
            device_type = int(_read_text(network_device / "type", "netdev type"))
            device_port = int(
                _read_text(network_device / "dev_port", "netdev port")
            )
            interface_index = int(
                _read_text(network_device / "ifindex", "netdev ifindex")
            )
            link_index = int(
                _read_text(network_device / "iflink", "netdev iflink")
            )
        except (ProvisioningError, ValueError):
            continue
        if (
            device_type == 32
            and device_port == port - 1
            and interface_index == link_index
        ):
            candidates.append(candidate.name)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ProvisioningError(
            f"HCA port maps to multiple parent IPoIB interfaces: {candidates}"
        )
    return candidates[0]


def observe_hca_port(
    rail: InfiniBandRailSpec,
    host_name: str,
    *,
    sysfs_root: Path = Path("/sys"),
) -> HcaPortObservation:
    expected = rail.hosts[host_name]
    hca_root = sysfs_root / "class" / "infiniband" / rail.hca_name
    port_root = hca_root / "ports" / str(rail.port)
    if not port_root.is_dir():
        raise ProvisioningError(
            f"{rail.link_id}: missing {rail.hca_name} port {rail.port}"
        )
    observation = HcaPortObservation(
        hca_name=rail.hca_name,
        port=rail.port,
        node_guid=_sysfs_guid(hca_root / "node_guid", "HCA node GUID"),
        port_guid=_port_guid(port_root),
        state=_state_name(_read_text(port_root / "state", "HCA port state")),
        physical_state=_state_name(
            _read_text(port_root / "phys_state", "HCA physical state")
        ),
        rate_gbps=_rate_gbps(_read_text(port_root / "rate", "HCA port rate")),
        subnet_manager_lid=int(
            _read_text(port_root / "sm_lid", "subnet manager LID"), 0
        ),
        local_identifier=int(
            _read_text(port_root / "lid", "local identifier"), 0
        ),
        interface=_resolve_parent_ipoib_interface(
            hca_root, rail.port, sysfs_root
        ),
    )
    if observation.node_guid != expected.node_guid:
        raise ProvisioningError(
            f"{rail.link_id}: node GUID is {observation.node_guid}, "
            f"expected {expected.node_guid}"
        )
    if observation.port_guid != expected.port_guid:
        raise ProvisioningError(
            f"{rail.link_id}: port GUID is {observation.port_guid}, "
            f"expected {expected.port_guid}"
        )
    if observation.state != "ACTIVE" or observation.physical_state != "LINKUP":
        raise ProvisioningError(
            f"{rail.link_id}: port is {observation.state}/"
            f"{observation.physical_state}, not ACTIVE/LINKUP"
        )
    if observation.rate_gbps < rail.minimum_rate_gbps:
        raise ProvisioningError(
            f"{rail.link_id}: rate {observation.rate_gbps} Gb/s is below "
            f"{rail.minimum_rate_gbps} Gb/s"
        )
    if (
        observation.subnet_manager_lid <= 0
        or observation.local_identifier <= 0
    ):
        raise ProvisioningError(
            f"{rail.link_id}: no active subnet manager/LID assignment"
        )
    return observation


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            tuple(command),
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise ProvisioningError(
            f"command failed: {' '.join(command)}: {stderr}"
        ) from error


def _json_command(command: Sequence[str]) -> list[object]:
    result = _run(command)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProvisioningError(
            f"command did not return JSON: {' '.join(command)}"
        ) from error
    return _array(value, "command output")


def _interface_observation(
    interface: str, *, sysfs_root: Path = Path("/sys")
) -> InterfaceObservation:
    root = sysfs_root / "class" / "net" / interface
    if not root.is_dir():
        raise ProvisioningError(f"interface {interface} does not exist")
    addresses: list[str] = []
    for raw_device in _json_command(("ip", "-json", "address", "show", "dev", interface)):
        device = _object(raw_device, "ip address device")
        for raw_address in _array(device.get("addr_info", []), "addr_info"):
            address = _object(raw_address, "address")
            if address.get("family") == "inet":
                local = _string(address.get("local"), "IPv4 local address")
                prefix_length = _integer(
                    address.get("prefixlen"), "IPv4 prefix length"
                )
                addresses.append(f"{local}/{prefix_length}")
    flags = {
        _string(flag, "interface flag")
        for raw_device in _json_command(
            ("ip", "-json", "link", "show", "dev", interface)
        )
        for flag in _array(
            _object(raw_device, "ip link device").get("flags", []),
            "interface flags",
        )
    }
    return InterfaceObservation(
        interface=interface,
        transport_mode=_read_text(root / "mode", "IPoIB transport mode"),
        mtu=int(_read_text(root / "mtu", "IPoIB MTU")),
        is_up="UP" in flags,
        ipv4_addresses=tuple(sorted(addresses)),
    )


def _ipoib_hardware_address(
    interface: str,
    expected_port_guid: str,
    *,
    sysfs_root: Path = Path("/sys"),
) -> str:
    address = _read_text(
        sysfs_root / "class" / "net" / interface / "address",
        "IPoIB hardware address",
    ).lower()
    if _IPOIB_HARDWARE_ADDRESS.fullmatch(address) is None:
        raise ProvisioningError(
            f"{interface} has malformed IPoIB hardware address {address!r}"
        )
    compact_guid = expected_port_guid.replace(":", "")
    if address.replace(":", "")[-16:] != compact_guid:
        raise ProvisioningError(
            f"{interface} hardware address is not pinned to port GUID "
            f"{expected_port_guid}"
        )
    return address


def _all_ipv4_assignments() -> dict[ipaddress.IPv4Address, str]:
    assignments: dict[ipaddress.IPv4Address, str] = {}
    for raw_device in _json_command(("ip", "-json", "address", "show")):
        device = _object(raw_device, "ip address device")
        interface = _string(device.get("ifname"), "interface name")
        for raw_address in _array(device.get("addr_info", []), "addr_info"):
            address = _object(raw_address, "address")
            if address.get("family") != "inet":
                continue
            parsed = ipaddress.IPv4Address(
                _string(address.get("local"), "IPv4 local address")
            )
            assignments[parsed] = interface
    return assignments


def _validate_route_collisions(
    desired_routes: Mapping[ipaddress.IPv4Network, str | None],
) -> None:
    for raw_route in _json_command(
        ("ip", "-json", "route", "show", "table", "all")
    ):
        route = _object(raw_route, "route")
        destination = route.get("dst")
        if not isinstance(destination, str) or destination == "default":
            continue
        try:
            existing = ipaddress.ip_network(destination, strict=False)
        except ValueError:
            continue
        if not isinstance(existing, ipaddress.IPv4Network):
            continue
        if existing.prefixlen == 0:
            continue
        for desired, expected_interface in desired_routes.items():
            if not existing.overlaps(desired):
                continue
            observed_interface = route.get("dev")
            if (
                expected_interface is not None
                and observed_interface == expected_interface
                and existing.subnet_of(desired)
            ):
                continue
            raise ProvisioningError(
                f"desired IPoIB subnet {desired} collides with existing "
                f"route {existing} on {observed_interface!r}"
            )


def _assert_host(topology: Topology, host_name: str) -> HostSpec:
    try:
        host = topology.hosts[host_name]
    except KeyError as error:
        raise ProvisioningError(f"unknown topology host {host_name!r}") from error
    observed = socket.gethostname().split(".", maxsplit=1)[0]
    if observed != host.hostname:
        raise ProvisioningError(
            f"this command targets {host.hostname}, but hostname is {observed}"
        )
    return host


def _selected_infiniband_rails(
    topology: Topology, profile: ProfileName
) -> tuple[InfiniBandRailSpec, ...]:
    if profile == "all":
        return topology.infiniband_rails
    selected = tuple(
        rail
        for rail in topology.infiniband_rails
        if rail.link_id in {"ib-edr", "ib-qdr-a"}
    )
    if len(selected) != 2:
        raise ProvisioningError("four-link profile rails are absent")
    return selected


def _conflicting_processes() -> tuple[str, ...]:
    conflicts: list[str] = []
    own_process = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_process:
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except OSError:
            continue
        if any(marker in command_line for marker in _CONFLICTING_COMMAND_MARKERS):
            conflicts.append(f"{entry.name}:{command_line.strip()[:240]}")
    return tuple(sorted(conflicts))


def _opensm_process_snapshot() -> tuple[str, ...]:
    processes: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command_name = _read_text(entry / "comm", "process command")
            command_line = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
                .strip()
            )
            start_identity = (
                (entry / "stat").read_text(encoding="utf-8").rpartition(") ")[2].split()[19]
            )
        except (OSError, ProvisioningError, IndexError):
            continue
        if command_name == "opensm":
            processes.append(f"{entry.name}:{start_identity}:{command_line}")
    return tuple(sorted(processes))


def inspect_network(
    topology: Topology,
    host_name: str,
    profile: ProfileName,
    *,
    sysfs_root: Path = Path("/sys"),
) -> JsonObject:
    _assert_host(topology, host_name)
    rails: list[JsonValue] = []
    for rail in _selected_infiniband_rails(topology, profile):
        observation = observe_hca_port(
            rail, host_name, sysfs_root=sysfs_root
        )
        interface_observation = (
            _interface_observation(
                observation.interface, sysfs_root=sysfs_root
            )
            if observation.interface is not None
            else None
        )
        rails.append(
            {
                "link_id": rail.link_id,
                "desired_address": str(rail.hosts[host_name].address),
                "desired_transport_mode": rail.transport_mode,
                "desired_mtu": rail.mtu,
                "hca": cast(JsonObject, asdict(observation)),
                "interface": (
                    cast(JsonObject, asdict(interface_observation))
                    if interface_observation is not None
                    else None
                ),
            }
        )
    return {
        "schema_version": NETWORK_RECEIPT_SCHEMA_VERSION,
        "operation": "inspect",
        "topology_sha256": topology.sha256,
        "host": host_name,
        "profile": profile,
        "module_loaded": Path("/sys/module/ib_ipoib").is_dir(),
        "opensm_processes": list(_opensm_process_snapshot()),
        "rails": rails,
    }


def _apply_one_interface(
    rail: InfiniBandRailSpec,
    desired_address: ipaddress.IPv4Interface,
    before: InterfaceObservation,
    commands: list[list[str]],
) -> None:
    if before.ipv4_addresses and str(desired_address) not in before.ipv4_addresses:
        raise ProvisioningError(
            f"{before.interface} has unexpected IPv4 addresses "
            f"{before.ipv4_addresses}"
        )
    if before.transport_mode != rail.transport_mode:
        if before.is_up:
            command = ["ip", "link", "set", "dev", before.interface, "down"]
            _run(command)
            commands.append(command)
        command = [
            "ip",
            "link",
            "set",
            "dev",
            before.interface,
            "type",
            "ipoib",
            "mode",
            rail.transport_mode,
        ]
        _run(command)
        commands.append(command)
    if before.mtu != rail.mtu:
        command = [
            "ip",
            "link",
            "set",
            "dev",
            before.interface,
            "mtu",
            str(rail.mtu),
        ]
        _run(command)
        commands.append(command)
    command = ["ip", "link", "set", "dev", before.interface, "up"]
    _run(command)
    commands.append(command)
    if str(desired_address) not in before.ipv4_addresses:
        command = [
            "ip",
            "address",
            "add",
            str(desired_address),
            "dev",
            before.interface,
        ]
        _run(command)
        commands.append(command)


def _restore_interface(
    before: InterfaceObservation,
    desired_address: ipaddress.IPv4Interface,
) -> None:
    current = _interface_observation(before.interface)
    if (
        str(desired_address) in current.ipv4_addresses
        and str(desired_address) not in before.ipv4_addresses
    ):
        _run(
            (
                "ip",
                "address",
                "del",
                str(desired_address),
                "dev",
                before.interface,
            )
        )
    current = _interface_observation(before.interface)
    if current.is_up and (
        current.transport_mode != before.transport_mode
        or current.mtu != before.mtu
        or not before.is_up
    ):
        _run(("ip", "link", "set", "dev", before.interface, "down"))
    if current.transport_mode != before.transport_mode:
        _run(
            (
                "ip",
                "link",
                "set",
                "dev",
                before.interface,
                "type",
                "ipoib",
                "mode",
                before.transport_mode,
            )
        )
    current = _interface_observation(before.interface)
    if current.mtu != before.mtu:
        _run(
            (
                "ip",
                "link",
                "set",
                "dev",
                before.interface,
                "mtu",
                str(before.mtu),
            )
        )
    if before.is_up:
        _run(("ip", "link", "set", "dev", before.interface, "up"))


def provision_network(
    topology: Topology,
    host_name: str,
    profile: ProfileName,
    *,
    apply: bool,
) -> JsonObject:
    _assert_host(topology, host_name)
    rails = _selected_infiniband_rails(topology, profile)
    initial_hca = tuple(observe_hca_port(rail, host_name) for rail in rails)
    desired_routes = {
        rail.hosts[host_name].address.network: observation.interface
        for rail, observation in zip(rails, initial_hca, strict=True)
    }
    _validate_route_collisions(desired_routes)
    assignments = _all_ipv4_assignments()
    for rail in rails:
        desired_address = rail.hosts[host_name].address.ip
        assigned_interface = assignments.get(desired_address)
        expected_interface = next(
            (
                observation.interface
                for observation in initial_hca
                if observation.hca_name == rail.hca_name
                and observation.port == rail.port
            ),
            None,
        )
        if (
            assigned_interface is not None
            and assigned_interface != expected_interface
        ):
            raise ProvisioningError(
                f"{desired_address} is already assigned to {assigned_interface}"
            )
    if not apply:
        result = inspect_network(topology, host_name, profile)
        result["operation"] = "provision-dry-run"
        result["would_run"] = [
            ["modprobe", "ib_ipoib"],
            [
                "peer_artifact_ipoib.py",
                "provision",
                "--host",
                host_name,
                "--profile",
                profile,
                "--apply",
            ],
        ]
        return result
    if os.geteuid() != 0:
        raise ProvisioningError("network provisioning requires root")
    conflicts = _conflicting_processes()
    if conflicts:
        raise ProvisioningError(
            "refusing to change IPoIB while RDMA/model processes are active: "
            + "; ".join(conflicts)
        )
    opensm_before = _opensm_process_snapshot()
    commands: list[list[str]] = []
    _run(("modprobe", "ib_ipoib"))
    commands.append(["modprobe", "ib_ipoib"])
    deadline = time.monotonic() + 10
    hca_observations: tuple[HcaPortObservation, ...]
    while True:
        hca_observations = tuple(
            observe_hca_port(rail, host_name) for rail in rails
        )
        if all(observation.interface for observation in hca_observations):
            break
        if time.monotonic() >= deadline:
            missing = [
                rail.link_id
                for rail, observation in zip(
                    rails, hca_observations, strict=True
                )
                if observation.interface is None
            ]
            raise ProvisioningError(
                f"ib_ipoib did not create interfaces for {missing}"
            )
        time.sleep(0.1)
    before: list[InterfaceObservation] = []
    try:
        for rail, hca in zip(rails, hca_observations, strict=True):
            if hca.interface is None:
                raise AssertionError("IPoIB interface disappeared")
            observation = _interface_observation(hca.interface)
            before.append(observation)
            _apply_one_interface(
                rail, rail.hosts[host_name].address, observation, commands
            )
        after = tuple(
            _interface_observation(cast(str, hca.interface))
            for hca in hca_observations
        )
        for rail, observation in zip(rails, after, strict=True):
            desired_address = str(rail.hosts[host_name].address)
            if (
                observation.transport_mode != rail.transport_mode
                or observation.mtu != rail.mtu
                or not observation.is_up
                or desired_address not in observation.ipv4_addresses
            ):
                raise ProvisioningError(
                    f"{rail.link_id} did not reach its requested state"
                )
        final_hca = tuple(observe_hca_port(rail, host_name) for rail in rails)
        for initial, final in zip(initial_hca, final_hca, strict=True):
            if (
                initial.node_guid,
                initial.port_guid,
                initial.state,
                initial.physical_state,
                initial.subnet_manager_lid,
                initial.local_identifier,
            ) != (
                final.node_guid,
                final.port_guid,
                final.state,
                final.physical_state,
                final.subnet_manager_lid,
                final.local_identifier,
            ):
                raise ProvisioningError(
                    f"{initial.hca_name}/{initial.port} fabric state changed"
                )
        opensm_after = _opensm_process_snapshot()
        if opensm_after != opensm_before:
            raise ProvisioningError("OpenSM process ownership changed during apply")
    except BaseException:
        for index in reversed(range(len(before))):
            rail = rails[index]
            observation = before[index]
            _restore_interface(
                observation, rail.hosts[host_name].address
            )
        raise
    return {
        "schema_version": NETWORK_RECEIPT_SCHEMA_VERSION,
        "operation": "provision-apply",
        "topology_sha256": topology.sha256,
        "host": host_name,
        "profile": profile,
        "timestamp_unix_seconds": time.time(),
        "opensm_processes_before": list(opensm_before),
        "opensm_processes_after": list(opensm_after),
        "hca_ports": [
            cast(JsonObject, asdict(observation))
            for observation in final_hca
        ],
        "interfaces_before": [
            cast(JsonObject, asdict(observation)) for observation in before
        ],
        "interfaces_after": [
            cast(JsonObject, asdict(observation)) for observation in after
        ],
        "commands": commands,
        "rollback": [
            [
                "ip",
                "address",
                "del",
                str(rail.hosts[host_name].address),
                "dev",
                cast(str, hca.interface),
            ]
            for rail, hca, observation in zip(
                rails, hca_observations, before, strict=True
            )
            if str(rail.hosts[host_name].address)
            not in observation.ipv4_addresses
        ],
    }


def _secure_output_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ProvisioningError("output directories must be absolute")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    observed = path.lstat()
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_mode & 0o077
    ):
        raise ProvisioningError(
            f"output directory is not owner-only: {path}"
        )


def _atomic_owner_only_write(
    path: Path, contents: bytes, *, replace: bool
) -> None:
    _secure_output_directory(path.parent)
    if path.exists():
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise ProvisioningError(f"refusing unsafe output path {path}")
        if path.read_bytes() == contents:
            return
        if not replace:
            raise ProvisioningError(
                f"{path} already differs; pass --replace explicitly"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_secret(path: Path) -> str:
    try:
        observed = path.lstat()
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_mode & 0o077
            or observed.st_size > 4096
        ):
            raise ProvisioningError(
                "authentication secret file must be owner-only and regular"
            )
        secret = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ProvisioningError(f"cannot read secret file {path}") from error
    if len(secret.encode()) < 32 or "\n" in secret or "\r" in secret:
        raise ProvisioningError(
            "authentication secret must be one line of at least 32 bytes"
        )
    return secret


def _peer_links(
    topology: Topology,
    receiver_host: str,
    source_host: str,
    profile: ProfileName,
) -> list[JsonObject]:
    links: list[JsonObject] = []
    for rail in _selected_infiniband_rails(topology, profile):
        observation = observe_hca_port(rail, receiver_host)
        if observation.interface is None:
            raise ProvisioningError(
                f"{rail.link_id} has no IPoIB interface; provision it first"
            )
        interface = _interface_observation(observation.interface)
        expected_address = str(rail.hosts[receiver_host].address)
        if expected_address not in interface.ipv4_addresses:
            raise ProvisioningError(
                f"{rail.link_id} is not provisioned with {expected_address}"
            )
        links.append(
            {
                "link_id": rail.link_id,
                "peer_node_id": topology.hosts[source_host].node_id,
                "medium": "infiniband",
                "local_interface": observation.interface,
                "local_ip_address": str(
                    rail.hosts[receiver_host].address.ip
                ),
                "peer_endpoint": {
                    "ip": str(rail.hosts[source_host].address.ip),
                    "port": topology.api_port,
                },
                "estimated_bytes_per_second": (
                    rail.estimated_bytes_per_second
                ),
                "maximum_concurrent_chunks": rail.maximum_concurrent_chunks,
            }
        )
    assignments = _all_ipv4_assignments()
    for rail in topology.ethernet_rails:
        receiver = rail.hosts[receiver_host]
        source = rail.hosts[source_host]
        if assignments.get(receiver.address) != receiver.interface:
            raise ProvisioningError(
                f"{rail.link_id}: {receiver.address} is not assigned to "
                f"{receiver.interface}"
            )
        links.append(
            {
                "link_id": rail.link_id,
                "peer_node_id": topology.hosts[source_host].node_id,
                "medium": "ethernet",
                "local_interface": receiver.interface,
                "local_ip_address": str(receiver.address),
                "peer_endpoint": {
                    "ip": str(source.address),
                    "port": topology.api_port,
                },
                "estimated_bytes_per_second": (
                    rail.estimated_bytes_per_second
                ),
                "maximum_concurrent_chunks": rail.maximum_concurrent_chunks,
            }
        )
    return links


def render_deployment(
    topology: Topology,
    secret_file: Path,
    output_directory: Path,
    profile: ProfileName,
    *,
    replace: bool,
) -> tuple[Path, Path, Path]:
    receiver_host = topology.deployment.receiver_host
    source_host = topology.deployment.source_host
    _assert_host(topology, receiver_host)
    secret = _read_secret(secret_file)
    links = _peer_links(
        topology, receiver_host, source_host, profile
    )
    source_configuration: JsonObject = {
        "schema_version": 1,
        "authentication_secret": secret,
        "peers": [],
        "server": {
            "model_roots": [str(topology.deployment.served_root)],
            "manifest_cache_directory": str(
                topology.deployment.manifest_cache_directory
            ),
            "served_snapshots": [
                {
                    "model_id": snapshot.model_id,
                    "revision": snapshot.revision,
                    "model_root_index": 0,
                    "relative_directory": snapshot.relative_directory,
                    "allow_partial_snapshot": True,
                }
                for snapshot in topology.deployment.served_snapshots
            ],
            "chunk_size_bytes": 67108864,
            "maximum_range_bytes": 67108864,
            "maximum_snapshot_files": 10000,
        },
        "disk_cache_directory": None,
        "memory_cache_directory": None,
        "disk_reserve_bytes": 0,
        "memory_reserve_bytes": 0,
        "request_timeout_seconds": 1800,
        "authentication_window_seconds": 300,
        "fallback_to_origin": False,
    }
    receiver_configuration: JsonObject = {
        "schema_version": 1,
        "authentication_secret": secret,
        "peers": [
            {
                "peer_node_id": topology.hosts[source_host].node_id,
                "links": links,
            }
        ],
        "server": None,
        "disk_cache_directory": str(
            topology.deployment.receiver_disk_cache_directory
        ),
        "memory_cache_directory": str(
            topology.deployment.receiver_memory_cache_directory
        ),
        "disk_reserve_bytes": 10737418240,
        "memory_reserve_bytes": 8589934592,
        "request_timeout_seconds": 1800,
        "authentication_window_seconds": 300,
        "fallback_to_origin": False,
    }
    materialization_plan: JsonObject = {
        "schema_version": 1,
        "topology_sha256": topology.sha256,
        "profile": profile,
        "source_config": f"{source_host}-peer-artifacts.json",
        "receiver_config": f"{receiver_host}-peer-artifacts.json",
        "source_host": source_host,
        "receiver_host": receiver_host,
        "materializations": [
            {
                "peer_node_id": topology.hosts[source_host].node_id,
                "model_id": snapshot.model_id,
                "revision": snapshot.revision,
                "destination": str(
                    topology.deployment.receiver_materialization_root
                    / snapshot.relative_directory
                ),
            }
            for snapshot in topology.deployment.served_snapshots
        ],
    }
    source_path = output_directory / f"{source_host}-peer-artifacts.json"
    receiver_path = (
        output_directory / f"{receiver_host}-peer-artifacts.json"
    )
    plan_path = output_directory / "materialization-plan.json"
    for path, value in (
        (source_path, source_configuration),
        (receiver_path, receiver_configuration),
        (plan_path, materialization_plan),
    ):
        _atomic_owner_only_write(
            path,
            (
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            ).encode(),
            replace=replace,
        )
    return source_path, receiver_path, plan_path


def render_networkmanager_profiles(
    topology: Topology,
    host_name: str,
    output_directory: Path,
    profile: ProfileName,
    *,
    replace: bool,
) -> tuple[Path, ...]:
    _assert_host(topology, host_name)
    paths: list[Path] = []
    for rail in _selected_infiniband_rails(topology, profile):
        observation = observe_hca_port(rail, host_name)
        if observation.interface is None:
            raise ProvisioningError(
                f"{rail.link_id} has no IPoIB interface; load ib_ipoib first"
            )
        hardware_address = _ipoib_hardware_address(
            observation.interface,
            rail.hosts[host_name].port_guid,
        )
        profile_id = f"exo-peer-{rail.link_id}"
        profile_uuid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"exo:{topology.cluster_id}:{host_name}:{rail.link_id}",
        )
        contents = (
            "[connection]\n"
            f"id={profile_id}\n"
            f"uuid={profile_uuid}\n"
            "type=infiniband\n"
            f"interface-name={observation.interface}\n"
            "autoconnect=true\n"
            "autoconnect-priority=100\n"
            "\n"
            "[infiniband]\n"
            f"mac-address={hardware_address}\n"
            f"mtu={rail.mtu}\n"
            f"transport-mode={rail.transport_mode}\n"
            "\n"
            "[ipv4]\n"
            f"address1={rail.hosts[host_name].address}\n"
            "method=manual\n"
            "never-default=true\n"
            "may-fail=false\n"
            "\n"
            "[ipv6]\n"
            "method=disabled\n"
        ).encode()
        path = output_directory / f"{profile_id}.nmconnection"
        _atomic_owner_only_write(path, contents, replace=replace)
        paths.append(path)
    return tuple(paths)


def _write_receipt(
    path: Path, receipt: JsonObject, *, replace: bool
) -> None:
    _atomic_owner_only_write(
        path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        replace=replace,
    )


def _profile(value: str) -> ProfileName:
    if value not in {"all", "four-link"}:
        raise argparse.ArgumentTypeError("profile must be all or four-link")
    return cast(ProfileName, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology", type=Path, default=DEFAULT_TOPOLOGY_PATH
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--host", required=True)
    inspect_parser.add_argument("--profile", type=_profile, default="all")

    provision_parser = subparsers.add_parser("provision")
    provision_parser.add_argument("--host", required=True)
    provision_parser.add_argument("--profile", type=_profile, default="all")
    provision_parser.add_argument("--apply", action="store_true")
    provision_parser.add_argument("--receipt", type=Path)
    provision_parser.add_argument("--replace", action="store_true")

    deployment_parser = subparsers.add_parser("render-deployment")
    deployment_parser.add_argument("--secret-file", type=Path, required=True)
    deployment_parser.add_argument(
        "--output-directory", type=Path, required=True
    )
    deployment_parser.add_argument(
        "--profile", type=_profile, default="all"
    )
    deployment_parser.add_argument("--replace", action="store_true")

    networkmanager_parser = subparsers.add_parser(
        "render-networkmanager"
    )
    networkmanager_parser.add_argument("--host", required=True)
    networkmanager_parser.add_argument(
        "--output-directory", type=Path, required=True
    )
    networkmanager_parser.add_argument(
        "--profile", type=_profile, default="all"
    )
    networkmanager_parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        topology = load_topology(cast(Path, arguments.topology))
        operation = cast(str, arguments.operation)
        if operation == "inspect":
            result = inspect_network(
                topology,
                cast(str, arguments.host),
                cast(ProfileName, arguments.profile),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif operation == "provision":
            result = provision_network(
                topology,
                cast(str, arguments.host),
                cast(ProfileName, arguments.profile),
                apply=cast(bool, arguments.apply),
            )
            receipt_path = cast(Path | None, arguments.receipt)
            if receipt_path is not None:
                _write_receipt(
                    receipt_path,
                    result,
                    replace=cast(bool, arguments.replace),
                )
            print(json.dumps(result, indent=2, sort_keys=True))
        elif operation == "render-deployment":
            paths = render_deployment(
                topology,
                cast(Path, arguments.secret_file),
                cast(Path, arguments.output_directory),
                cast(ProfileName, arguments.profile),
                replace=cast(bool, arguments.replace),
            )
            print("\n".join(str(path) for path in paths))
        elif operation == "render-networkmanager":
            paths = render_networkmanager_profiles(
                topology,
                cast(str, arguments.host),
                cast(Path, arguments.output_directory),
                cast(ProfileName, arguments.profile),
                replace=cast(bool, arguments.replace),
            )
            print("\n".join(str(path) for path in paths))
        else:
            raise AssertionError(f"unsupported operation {operation}")
    except ProvisioningError as error:
        print(f"Peer artifact IPoIB operation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
