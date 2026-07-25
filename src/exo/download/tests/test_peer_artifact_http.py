from __future__ import annotations

import hashlib
import json
import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ParamSpec, TypeVar, cast

import httpx
import pytest
from aiohttp import web
from fastapi import FastAPI
from pydantic import SecretStr

from exo.download.peer_artifact_http import (
    PEER_ARTIFACT_MANIFEST_PATH,
    PEER_ARTIFACT_RANGE_PATH,
    PEER_ARTIFACT_SNAPSHOT_PATH,
    PeerArtifactConfigurationError,
    PeerArtifactDeploymentConfig,
    PeerArtifactHttpClient,
    PeerArtifactHttpSource,
    PeerArtifactLink,
    PeerArtifactLinkId,
    PeerArtifactPeerConfig,
    PeerArtifactServedSnapshot,
    PeerArtifactServerConfig,
    PeerArtifactSnapshotManifest,
    build_peer_artifact_authentication_headers,
    install_peer_artifact_http_routes,
    load_peer_artifact_deployment_config,
)
from exo.download.peer_artifact_transfer import (
    PeerArtifactManifest,
    PeerArtifactSnapshotId,
    Sha256Digest,
)
from exo.shared.types.common import Host, ModelId, NodeId

_SECRET = SecretStr("peer-artifact-test-secret-that-is-long-enough")
_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


async def _run_sync_inline(
    function: Callable[_Parameters, _Result],
    *args: _Parameters.args,
    **kwargs: _Parameters.kwargs,
) -> _Result:
    return function(*args, **kwargs)


def _link(port: int, *, local_ip_address: str = "127.0.0.1") -> PeerArtifactLink:
    return PeerArtifactLink(
        link_id=PeerArtifactLinkId("loopback"),
        peer_node_id=NodeId("peer"),
        medium="ethernet",
        local_interface="lo",
        local_ip_address=local_ip_address,
        peer_endpoint=Host(ip="127.0.0.1", port=port),
        estimated_bytes_per_second=1_000_000_000,
    )


def _deployment(
    model_root: Path,
    *,
    peers: tuple[PeerArtifactPeerConfig, ...] = (),
    disk_cache_directory: Path | None = None,
) -> PeerArtifactDeploymentConfig:
    return PeerArtifactDeploymentConfig(
        schema_version=1,
        authentication_secret=_SECRET,
        peers=peers,
        server=PeerArtifactServerConfig(
            model_roots=(model_root,),
            manifest_cache_directory=model_root.parent / "manifest-cache",
            served_snapshots=(
                PeerArtifactServedSnapshot(
                    model_id=ModelId("example/model"),
                    revision="main",
                    model_root_index=0,
                    relative_directory="snapshot",
                ),
            ),
            chunk_size_bytes=4,
            maximum_range_bytes=4,
        ),
        disk_cache_directory=disk_cache_directory,
    )


def _headers(
    path: str,
    parameters: Mapping[str, str | int],
    *,
    nonce: str | None = None,
) -> dict[str, str]:
    return build_peer_artifact_authentication_headers(
        _SECRET,
        method="GET",
        path=path,
        parameters=tuple(
            (name, str(value)) for name, value in parameters.items()
        ),
        nonce=nonce,
    )


async def test_authenticated_routes_serve_confined_manifests_and_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Codex test sandbox does not permit worker threads. The production
    # path uses AnyIO's standard thread offload so hashing and file reads do
    # not block Exo's event loop.
    monkeypatch.setattr(
        "exo.download.peer_artifact_http.to_thread.run_sync",
        _run_sync_inline,
    )
    model_root = tmp_path / "models"
    snapshot = model_root / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"abcdefgh")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (snapshot / "escaped.bin").symlink_to(outside)
    app = FastAPI()
    install_peer_artifact_http_routes(app, _deployment(model_root))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://peer",
    ) as client:
        unauthorized = await client.get(
            PEER_ARTIFACT_SNAPSHOT_PATH,
            params={"model_id": "example/model", "revision": "main"},
        )
        assert unauthorized.status_code == 401

        snapshot_parameters = {
            "model_id": "example/model",
            "revision": "main",
        }
        snapshot_response = await client.get(
            PEER_ARTIFACT_SNAPSHOT_PATH,
            params=snapshot_parameters,
            headers=_headers(
                PEER_ARTIFACT_SNAPSHOT_PATH,
                snapshot_parameters,
            ),
        )
        # The escaped symlink makes the configured snapshot invalid rather than
        # disclosing a file outside the configured roots.
        assert snapshot_response.status_code == 404

        escaped_parameters = {"artifact_path": "snapshot/escaped.bin"}
        escaped = await client.get(
            PEER_ARTIFACT_MANIFEST_PATH,
            params=escaped_parameters,
            headers=_headers(
                PEER_ARTIFACT_MANIFEST_PATH,
                escaped_parameters,
            ),
        )
        assert escaped.status_code == 422
        traversal_parameters = {"artifact_path": "../outside.bin"}
        traversal = await client.get(
            PEER_ARTIFACT_MANIFEST_PATH,
            params=traversal_parameters,
            headers=_headers(
                PEER_ARTIFACT_MANIFEST_PATH,
                traversal_parameters,
            ),
        )
        assert traversal.status_code == 422

        (snapshot / "escaped.bin").unlink()
        replay_headers = _headers(
            PEER_ARTIFACT_SNAPSHOT_PATH,
            snapshot_parameters,
            nonce="1" * 32,
        )
        snapshot_response = await client.get(
            PEER_ARTIFACT_SNAPSHOT_PATH,
            params=snapshot_parameters,
            headers=replay_headers,
        )
        assert snapshot_response.status_code == 200
        replay = await client.get(
            PEER_ARTIFACT_SNAPSHOT_PATH,
            params=snapshot_parameters,
            headers=replay_headers,
        )
        assert replay.status_code == 401
        snapshot_payload = PeerArtifactSnapshotManifest.model_validate_json(
            snapshot_response.content
        )
        assert len(snapshot_payload.files) == 1
        assert snapshot_payload.files[0].file_path == "config.json"
        assert snapshot_payload.files[0].artifact_path == "snapshot/config.json"
        assert snapshot_payload.files[0].size_bytes == 8

        pinned_file = snapshot_payload.files[0]
        manifest_parameters = {
            "snapshot_id": snapshot_payload.snapshot_id,
            "artifact_path": pinned_file.artifact_path,
            "manifest_sha256": pinned_file.manifest_sha256,
        }
        manifest_response = await client.get(
            PEER_ARTIFACT_MANIFEST_PATH,
            params=manifest_parameters,
            headers=_headers(
                PEER_ARTIFACT_MANIFEST_PATH,
                manifest_parameters,
            ),
        )
        assert manifest_response.status_code == 200
        assert manifest_response.json()["sha256"] == hashlib.sha256(
            b"abcdefgh"
        ).hexdigest()

        range_parameters = {
            "snapshot_id": snapshot_payload.snapshot_id,
            "artifact_sha256": pinned_file.sha256,
            "artifact_path": pinned_file.artifact_path,
            "offset_bytes": 2,
            "size_bytes": 4,
        }
        range_response = await client.get(
            PEER_ARTIFACT_RANGE_PATH,
            params=range_parameters,
            headers=_headers(PEER_ARTIFACT_RANGE_PATH, range_parameters),
        )
        assert range_response.status_code == 200
        assert range_response.content == b"cdef"
        oversized_parameters = {
            **range_parameters,
            "offset_bytes": 0,
            "size_bytes": 5,
        }
        oversized = await client.get(
            PEER_ARTIFACT_RANGE_PATH,
            params=oversized_parameters,
            headers=_headers(
                PEER_ARTIFACT_RANGE_PATH,
                oversized_parameters,
            ),
        )
        assert oversized.status_code == 416
        assert tuple(
            (model_root.parent / "manifest-cache" / "manifests").glob("*.json")
        )
        assert (
            model_root.parent
            / "manifest-cache"
            / "snapshots"
            / f"{snapshot_payload.snapshot_id}.json"
        ).is_file()
        deployment = _deployment(model_root)
        assert deployment.server is not None
        fresh_source = PeerArtifactHttpSource(deployment.server)

        def unexpected_manifest_rebuild(
            descriptor: int,
            artifact_path: str,
            chunk_size_bytes: int,
        ) -> PeerArtifactManifest:
            del descriptor, artifact_path, chunk_size_bytes
            raise AssertionError("persistent manifest cache was not reused")

        monkeypatch.setattr(
            "exo.download.peer_artifact_http._manifest_from_descriptor",
            unexpected_manifest_rebuild,
        )
        try:
            fresh_snapshot = await fresh_source.snapshot_manifest(
                ModelId("example/model"), "main"
            )
        finally:
            fresh_source.close()
        assert fresh_snapshot.snapshot_id == snapshot_payload.snapshot_id
        (snapshot / "config.json").write_bytes(b"ijklmnop")
        drifted = await client.get(
            PEER_ARTIFACT_RANGE_PATH,
            params=range_parameters,
            headers=_headers(PEER_ARTIFACT_RANGE_PATH, range_parameters),
        )
        assert drifted.status_code == 404


async def test_http_client_pins_outbound_connection_to_link_local_address() -> None:
    observed_source_addresses: list[str] = []

    async def range_handler(request: web.Request) -> web.Response:
        assert request.remote is not None
        observed_source_addresses.append(request.remote)
        return web.Response(body=b"bound")

    application = web.Application()
    application.router.add_get(PEER_ARTIFACT_RANGE_PATH, range_handler)
    runner = web.AppRunner(application)
    await runner.setup()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 0))
    server_address = cast(tuple[str, int], server_socket.getsockname())
    port = server_address[1]
    site = web.SockSite(runner, server_socket)
    await site.start()
    try:
        async with PeerArtifactHttpClient(_SECRET, 30) as client:
            contents = await client.read_artifact_range(
                link=_link(port, local_ip_address="127.0.0.1"),
                snapshot_id=PeerArtifactSnapshotId("a" * 64),
                artifact_sha256=cast(Sha256Digest, "b" * 64),
                artifact_path="snapshot/config.json",
                offset_bytes=0,
                size_bytes=5,
            )
        assert contents == b"bound"
        assert observed_source_addresses == ["127.0.0.1"]
    finally:
        await runner.cleanup()


def test_config_loader_is_disabled_when_absent_and_requires_private_file(
    tmp_path: Path,
) -> None:
    assert load_peer_artifact_deployment_config(None) is None
    model_root = tmp_path / "models"
    model_root.mkdir()
    config_path = tmp_path / "peer-artifacts.json"
    payload = _deployment(model_root).model_dump(mode="json")
    payload["authentication_secret"] = _SECRET.get_secret_value()
    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o644)
    with pytest.raises(PeerArtifactConfigurationError, match="owner-only"):
        load_peer_artifact_deployment_config(config_path)
    os.chmod(config_path, 0o600)
    loaded = load_peer_artifact_deployment_config(config_path)
    assert loaded is not None
    assert loaded.server is not None
    assert loaded.server.model_roots == (model_root,)
