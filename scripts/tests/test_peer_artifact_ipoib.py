from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import cast

import pytest

from exo.download.peer_artifact_http import PeerArtifactDeploymentConfig
from scripts import peer_artifact_ipoib as ipoib


def _topology() -> ipoib.Topology:
    return ipoib.load_topology()


def _patch_configured_dwagon(
    monkeypatch: pytest.MonkeyPatch, topology: ipoib.Topology
) -> None:
    monkeypatch.setattr(ipoib.socket, "gethostname", lambda: "dwagon")
    interfaces = {
        ("mlx5_0", 1): "ibs5",
        ("mlx4_0", 1): "ibs4",
        ("mlx4_0", 2): "ibs4d1",
    }

    def observe(
        rail: ipoib.InfiniBandRailSpec,
        host_name: str,
        *,
        sysfs_root: Path = Path("/sys"),
    ) -> ipoib.HcaPortObservation:
        del sysfs_root
        endpoint = rail.hosts[host_name]
        return ipoib.HcaPortObservation(
            hca_name=rail.hca_name,
            port=rail.port,
            node_guid=endpoint.node_guid,
            port_guid=endpoint.port_guid,
            state="ACTIVE",
            physical_state="LINKUP",
            rate_gbps=rail.minimum_rate_gbps,
            subnet_manager_lid=1,
            local_identifier=2,
            interface=interfaces[(rail.hca_name, rail.port)],
        )

    def interface_observation(
        interface: str, *, sysfs_root: Path = Path("/sys")
    ) -> ipoib.InterfaceObservation:
        del sysfs_root
        rail = next(
            rail
            for rail in topology.infiniband_rails
            if interfaces[(rail.hca_name, rail.port)] == interface
        )
        return ipoib.InterfaceObservation(
            interface=interface,
            transport_mode=rail.transport_mode,
            mtu=rail.mtu,
            is_up=True,
            ipv4_addresses=(str(rail.hosts["dwagon"].address),),
        )

    assignments = {
        endpoint.address: endpoint.interface
        for rail in topology.ethernet_rails
        for endpoint in (rail.hosts["dwagon"],)
    }
    monkeypatch.setattr(ipoib, "observe_hca_port", observe)
    monkeypatch.setattr(ipoib, "_interface_observation", interface_observation)
    monkeypatch.setattr(ipoib, "_all_ipv4_assignments", lambda: assignments)


def test_exact_topology_has_three_disjoint_ib_rails_and_two_ethernet_rails() -> None:
    topology = _topology()

    assert topology.cluster_id == "fwuffydwagon-five-link-v1"
    assert [rail.link_id for rail in topology.infiniband_rails] == [
        "ib-edr",
        "ib-qdr-a",
        "ib-qdr-b",
    ]
    assert [rail.link_id for rail in topology.ethernet_rails] == [
        "ethernet-a",
        "ethernet-b",
    ]
    networks = [
        rail.hosts["dwagon"].address.network for rail in topology.infiniband_rails
    ]
    assert networks == [
        ipaddress.IPv4Network("10.44.0.0/30"),
        ipaddress.IPv4Network("10.44.1.0/30"),
        ipaddress.IPv4Network("10.44.2.0/30"),
    ]
    assert not any(
        left.overlaps(right)
        for index, left in enumerate(networks)
        for right in networks[index + 1 :]
    )


def test_guid_pinned_sysfs_observation_resolves_parent_port_netdev(
    tmp_path: Path,
) -> None:
    topology = _topology()
    rail = topology.infiniband_rails[1]
    endpoint = rail.hosts["dwagon"]
    sysfs = tmp_path / "sys"
    hca = sysfs / "class" / "infiniband" / rail.hca_name
    port = hca / "ports" / str(rail.port)
    network_device = sysfs / "class" / "net" / "ib-test"
    for directory in (
        hca / "device" / "net" / "ib-test",
        port / "gids",
        network_device,
    ):
        directory.mkdir(parents=True)
    (hca / "node_guid").write_text(endpoint.node_guid)
    compact_port_guid = endpoint.port_guid.replace(":", "")
    (port / "gids" / "0").write_text(
        f"fe80::{compact_port_guid[:4]}:{compact_port_guid[4:8]}:"
        f"{compact_port_guid[8:12]}:{compact_port_guid[12:]}"
    )
    (port / "state").write_text("4: ACTIVE")
    (port / "phys_state").write_text("5: LinkUp")
    (port / "rate").write_text("40 Gb/sec (4X QDR)")
    (port / "sm_lid").write_text("2")
    (port / "lid").write_text("1")
    (network_device / "type").write_text("32")
    (network_device / "dev_port").write_text("0")
    (network_device / "ifindex").write_text("12")
    (network_device / "iflink").write_text("12")

    observation = ipoib.observe_hca_port(rail, "dwagon", sysfs_root=sysfs)

    assert observation.interface == "ib-test"
    assert observation.node_guid == endpoint.node_guid
    assert observation.port_guid == endpoint.port_guid
    assert observation.state == "ACTIVE"
    assert observation.physical_state == "LINKUP"


def test_ipoib_hardware_address_must_end_with_pinned_port_guid(
    tmp_path: Path,
) -> None:
    interface_root = tmp_path / "sys" / "class" / "net" / "ib-test"
    interface_root.mkdir(parents=True)
    address_path = interface_root / "address"
    address_path.write_text(
        "00:00:00:00:fe:80:00:00:00:00:00:00:24:8a:07:03:00:95:af:d4"
    )

    assert ipoib._ipoib_hardware_address(
        "ib-test",
        "248a:0703:0095:afd4",
        sysfs_root=tmp_path / "sys",
    ).endswith("24:8a:07:03:00:95:af:d4")

    with pytest.raises(ipoib.ProvisioningError, match="not pinned"):
        ipoib._ipoib_hardware_address(
            "ib-test",
            "248a:0703:00a3:2154",
            sysfs_root=tmp_path / "sys",
        )


def test_rendered_deployment_is_schema_valid_and_uses_all_five_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _topology()
    _patch_configured_dwagon(monkeypatch, topology)
    secret_file = tmp_path / "secret"
    secret_file.write_text("s" * 64)
    os.chmod(secret_file, 0o600)
    output_directory = tmp_path / "output"

    source_path, receiver_path, plan_path = ipoib.render_deployment(
        topology,
        secret_file,
        output_directory,
        "all",
        replace=False,
    )

    source = PeerArtifactDeploymentConfig.model_validate_json(source_path.read_bytes())
    receiver = PeerArtifactDeploymentConfig.model_validate_json(
        receiver_path.read_bytes()
    )
    assert source.server is not None
    assert receiver.server is None
    assert len(receiver.peers) == 1
    assert [str(link.link_id) for link in receiver.peers[0].links] == [
        "ib-edr",
        "ib-qdr-a",
        "ib-qdr-b",
        "ethernet-a",
        "ethernet-b",
    ]
    assert [str(link.local_ip_address) for link in receiver.peers[0].links[:3]] == [
        "10.44.0.1",
        "10.44.1.1",
        "10.44.2.1",
    ]
    plan = cast(dict[str, object], json.loads(plan_path.read_bytes()))
    assert plan["profile"] == "all"
    assert len(cast(list[object], plan["materializations"])) == 2
    assert all(
        path.stat().st_mode & 0o077 == 0
        for path in (source_path, receiver_path, plan_path)
    )

    repeated = ipoib.render_deployment(
        topology,
        secret_file,
        output_directory,
        "all",
        replace=False,
    )
    assert repeated == (source_path, receiver_path, plan_path)


def test_four_link_profile_omits_only_second_qdr_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _topology()
    _patch_configured_dwagon(monkeypatch, topology)
    secret_file = tmp_path / "secret"
    secret_file.write_text("t" * 64)
    os.chmod(secret_file, 0o600)

    _, receiver_path, _ = ipoib.render_deployment(
        topology,
        secret_file,
        tmp_path / "output",
        "four-link",
        replace=False,
    )

    receiver = PeerArtifactDeploymentConfig.model_validate_json(
        receiver_path.read_bytes()
    )
    assert [str(link.link_id) for link in receiver.peers[0].links] == [
        "ib-edr",
        "ib-qdr-a",
        "ethernet-a",
        "ethernet-b",
    ]


def test_route_validation_allows_idempotent_local_routes_and_rejects_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes: list[object] = [
        {"dst": "10.44.0.0/30", "dev": "ibs5"},
        {"dst": "10.44.0.1", "dev": "ibs5", "table": "local"},
    ]
    monkeypatch.setattr(ipoib, "_json_command", lambda _command: routes)
    desired = {ipaddress.IPv4Network("10.44.0.0/30"): "ibs5"}

    ipoib._validate_route_collisions(desired)
    routes.append({"dst": "10.44.0.0/16", "dev": "wg0"})
    with pytest.raises(ipoib.ProvisioningError, match="collides"):
        ipoib._validate_route_collisions(desired)


def test_partial_apply_restores_only_interfaces_it_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = _topology()
    monkeypatch.setattr(ipoib.socket, "gethostname", lambda: "dwagon")
    interfaces = {
        ("mlx5_0", 1): "ibs5",
        ("mlx4_0", 1): "ibs4",
        ("mlx4_0", 2): "ibs4d1",
    }

    def observe(
        rail: ipoib.InfiniBandRailSpec,
        host_name: str,
        *,
        sysfs_root: Path = Path("/sys"),
    ) -> ipoib.HcaPortObservation:
        del sysfs_root
        endpoint = rail.hosts[host_name]
        return ipoib.HcaPortObservation(
            hca_name=rail.hca_name,
            port=rail.port,
            node_guid=endpoint.node_guid,
            port_guid=endpoint.port_guid,
            state="ACTIVE",
            physical_state="LINKUP",
            rate_gbps=rail.minimum_rate_gbps,
            subnet_manager_lid=1,
            local_identifier=2,
            interface=interfaces[(rail.hca_name, rail.port)],
        )

    def interface_observation(
        interface: str, *, sysfs_root: Path = Path("/sys")
    ) -> ipoib.InterfaceObservation:
        del sysfs_root
        return ipoib.InterfaceObservation(
            interface=interface,
            transport_mode="datagram",
            mtu=2044,
            is_up=True,
            ipv4_addresses=(),
        )

    apply_calls: list[str] = []

    def fail_second_apply(
        rail: ipoib.InfiniBandRailSpec,
        desired_address: ipaddress.IPv4Interface,
        before: ipoib.InterfaceObservation,
        commands: list[list[str]],
    ) -> None:
        del rail, desired_address, commands
        apply_calls.append(before.interface)
        if before.interface == "ibs4":
            raise ipoib.ProvisioningError("injected second-interface failure")

    restored: list[str] = []

    def record_restore(
        before: ipoib.InterfaceObservation,
        desired_address: ipaddress.IPv4Interface,
    ) -> None:
        del desired_address
        restored.append(before.interface)

    monkeypatch.setattr(ipoib, "observe_hca_port", observe)
    monkeypatch.setattr(ipoib, "_interface_observation", interface_observation)
    monkeypatch.setattr(ipoib, "_validate_route_collisions", lambda _routes: None)
    monkeypatch.setattr(ipoib, "_all_ipv4_assignments", dict)
    monkeypatch.setattr(ipoib.os, "geteuid", lambda: 0)
    monkeypatch.setattr(ipoib, "_conflicting_processes", tuple)
    monkeypatch.setattr(ipoib, "_opensm_process_snapshot", tuple)
    monkeypatch.setattr(ipoib, "_run", lambda _command: None)
    monkeypatch.setattr(ipoib, "_apply_one_interface", fail_second_apply)
    monkeypatch.setattr(ipoib, "_restore_interface", record_restore)

    with pytest.raises(
        ipoib.ProvisioningError, match="injected second-interface failure"
    ):
        ipoib.provision_network(
            topology,
            "dwagon",
            "all",
            apply=True,
        )

    assert apply_calls == ["ibs5", "ibs4"]
    assert restored == ["ibs4", "ibs5"]
