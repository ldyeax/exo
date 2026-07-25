#!/usr/bin/env python3
"""Prepare, serve, and verify a tiny authenticated multi-link artifact."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import NewType, cast

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


def _install_python311_proof_type_stubs() -> None:
    """Avoid importing unrelated Python-3.13-only Exo types on fwuff's KT env."""
    if sys.version_info >= (3, 13):
        return
    common = types.ModuleType("exo.shared.types.common")
    model_cards = types.ModuleType("exo.shared.models.model_cards")

    class Host(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
        ip: str
        port: int

    common.Host = Host
    common.ModelId = NewType("ModelId", str)
    common.NodeId = NewType("NodeId", str)
    model_cards.HuggingFaceRevision = str
    model_cards.MODEL_REVISION_RECEIPT_FILENAME = (
        ".exo-huggingface-revision.json"
    )
    sys.modules[common.__name__] = common
    sys.modules[model_cards.__name__] = model_cards


_install_python311_proof_type_stubs()

from exo.download.peer_artifact_http import (  # noqa: E402
    PeerArtifactHttpClient,
    install_peer_artifact_http_routes,
    load_peer_artifact_deployment_config,
)
from exo.download.peer_artifact_transfer import (  # noqa: E402
    PeerArtifactStorage,
    execute_peer_artifact_transfer,
    observe_peer_artifact_storage_availability,
)
from exo.shared.models.model_cards import HuggingFaceRevision  # noqa: E402
from exo.shared.types.common import ModelId  # noqa: E402

PROOF_MODEL_ID = ModelId("local/peer-artifact-five-link-proof")
PROOF_REVISION = cast(HuggingFaceRevision, "main")
PROOF_FILE_PATH = "proof.bin"
PROOF_CHUNK_SIZE_BYTES = 1024 * 1024
PROOF_SIZE_BYTES = 16 * 1024 * 1024


class ProofError(RuntimeError):
    """Raised when a proof cannot establish payload traffic on every link."""


def _secure_directory(path: Path) -> None:
    if not path.is_absolute():
        raise ProofError("proof directories must be absolute")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    observed = path.lstat()
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_mode & 0o077
    ):
        raise ProofError(f"unsafe proof directory {path}")


def _atomic_write(path: Path, contents: bytes, mode: int = 0o600) -> None:
    _secure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
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


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ProofError(f"cannot load JSON {path}") from error
    if not isinstance(value, dict):
        raise ProofError(f"{path} is not a JSON object")
    return cast(dict[str, object], value)


def _write_proof_payload(path: Path) -> str:
    block = hashlib.sha256(b"exo-five-link-proof-v1").digest() * 32768
    if len(block) != PROOF_CHUNK_SIZE_BYTES:
        raise AssertionError("proof block size changed")
    hasher = hashlib.sha256()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for _ in range(PROOF_SIZE_BYTES // len(block)):
                output.write(block)
                hasher.update(block)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, 0o444)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return hasher.hexdigest()


def prepare_proof(
    source_config_path: Path,
    receiver_config_path: Path,
    output_directory: Path,
    remote_root: Path,
    port: int,
) -> tuple[Path, Path, Path, Path]:
    _secure_directory(output_directory)
    source_tree = output_directory / "source"
    snapshot_tree = source_tree / "snapshot"
    _secure_directory(source_tree)
    _secure_directory(snapshot_tree)
    payload_path = snapshot_tree / PROOF_FILE_PATH
    payload_sha256 = _write_proof_payload(payload_path)

    source = _load_json(source_config_path)
    receiver = _load_json(receiver_config_path)
    raw_server = source.get("server")
    if not isinstance(raw_server, dict):
        raise ProofError("source config has no server")
    server = cast(dict[str, object], raw_server)
    server["model_roots"] = [str(remote_root / "source")]
    server["manifest_cache_directory"] = str(
        remote_root / "manifest-cache"
    )
    server["served_snapshots"] = [
        {
            "model_id": str(PROOF_MODEL_ID),
            "revision": PROOF_REVISION,
            "model_root_index": 0,
            "relative_directory": "snapshot",
            "allow_partial_snapshot": True,
        }
    ]
    server["chunk_size_bytes"] = PROOF_CHUNK_SIZE_BYTES
    server["maximum_range_bytes"] = PROOF_CHUNK_SIZE_BYTES
    server["maximum_snapshot_files"] = 16
    raw_peers = receiver.get("peers")
    if not isinstance(raw_peers, list) or len(raw_peers) != 1:
        raise ProofError("receiver config must contain one peer")
    raw_peer = raw_peers[0]
    if not isinstance(raw_peer, dict):
        raise ProofError("receiver peer is invalid")
    raw_links = raw_peer.get("links")
    if not isinstance(raw_links, list) or len(raw_links) < 2:
        raise ProofError("receiver proof requires multiple links")
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            raise ProofError("receiver link is invalid")
        endpoint = raw_link.get("peer_endpoint")
        if not isinstance(endpoint, dict):
            raise ProofError("receiver peer endpoint is invalid")
        endpoint["port"] = port
    receiver["disk_cache_directory"] = str(
        output_directory / "receiver-cache"
    )
    receiver["memory_cache_directory"] = None
    receiver["disk_reserve_bytes"] = 0
    receiver["memory_reserve_bytes"] = 0
    source_proof_config = output_directory / "source-proof-config.json"
    receiver_proof_config = output_directory / "receiver-proof-config.json"
    metadata_path = output_directory / "prepared-proof.json"
    _atomic_write(
        source_proof_config,
        (json.dumps(source, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        receiver_proof_config,
        (json.dumps(receiver, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        metadata_path,
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": str(PROOF_MODEL_ID),
                    "revision": PROOF_REVISION,
                    "file_path": PROOF_FILE_PATH,
                    "size_bytes": PROOF_SIZE_BYTES,
                    "sha256": payload_sha256,
                    "chunk_size_bytes": PROOF_CHUNK_SIZE_BYTES,
                    "configured_link_count": len(raw_links),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    for config_path in (source_proof_config, receiver_proof_config):
        loaded = load_peer_artifact_deployment_config(config_path)
        if loaded is None:
            raise AssertionError("proof config was unexpectedly disabled")
    return (
        source_tree,
        source_proof_config,
        receiver_proof_config,
        metadata_path,
    )


def serve(config_path: Path, host: str, port: int) -> None:
    config = load_peer_artifact_deployment_config(config_path)
    if config is None or config.server is None:
        raise ProofError("proof server configuration is required")
    application = FastAPI()
    install_peer_artifact_http_routes(application, config)
    uvicorn.run(application, host=host, port=port, log_level="info")


async def transfer(config_path: Path, receipt_path: Path) -> None:
    config = load_peer_artifact_deployment_config(config_path)
    if config is None or len(config.peers) != 1:
        raise ProofError("proof receiver configuration is required")
    if config.disk_cache_directory is None:
        raise ProofError("proof receiver has no disk cache")
    peer = config.peers[0]
    first_link = peer.links[0]
    client = PeerArtifactHttpClient(
        config.authentication_secret,
        int(config.request_timeout_seconds),
    )
    started_at = time.monotonic()
    async with client:
        snapshot = await client.fetch_snapshot_manifest(
            link=first_link,
            model_id=PROOF_MODEL_ID,
            revision=PROOF_REVISION,
        )
        if len(snapshot.files) != 1:
            raise ProofError("proof snapshot must contain exactly one file")
        snapshot_file = snapshot.files[0]
        manifest = await client.fetch_artifact_manifest(
            link=first_link,
            snapshot_id=snapshot.snapshot_id,
            expected_manifest_sha256=snapshot_file.manifest_sha256,
            artifact_path=snapshot_file.artifact_path,
        )
        storage = PeerArtifactStorage(
            disk_cache_directory=config.disk_cache_directory,
            memory_cache_directory=config.memory_cache_directory,
            disk_reserve_bytes=config.disk_reserve_bytes,
            memory_reserve_bytes=config.memory_reserve_bytes,
        )
        availability = observe_peer_artifact_storage_availability(storage)
        published = await execute_peer_artifact_transfer(
            manifest,
            snapshot.snapshot_id,
            peer.links,
            client,
            storage,
            availability,
        )
        response_bytes = {
            str(link_id): count
            for link_id, count in client.received_bytes_by_link.items()
        }
    elapsed_seconds = time.monotonic() - started_at
    payload_bytes = {
        str(transfer.link_id): int(transfer.transferred_bytes)
        for transfer in published.link_transfers
    }
    expected_link_ids = {str(link.link_id) for link in peer.links}
    if set(payload_bytes) != expected_link_ids or any(
        count <= 0 for count in payload_bytes.values()
    ):
        raise ProofError(
            f"not every configured link transferred payload: {payload_bytes}"
        )
    if published.path.read_bytes() != (
        hashlib.sha256(b"exo-five-link-proof-v1").digest() * 32768
    ) * (PROOF_SIZE_BYTES // PROOF_CHUNK_SIZE_BYTES):
        raise ProofError("published proof bytes changed after verification")
    receipt = {
        "schema_version": 1,
        "proof": "exo-peer-artifact-five-link-v1",
        "timestamp_unix_seconds": time.time(),
        "snapshot_id": str(snapshot.snapshot_id),
        "artifact_sha256": str(published.sha256),
        "artifact_size_bytes": int(published.size_bytes),
        "cache_hit": published.cache_hit,
        "elapsed_seconds": elapsed_seconds,
        "payload_bytes_by_link": payload_bytes,
        "http_response_bytes_by_link": response_bytes,
        "links": [
            {
                "link_id": str(link.link_id),
                "medium": link.medium,
                "local_interface": link.local_interface,
                "local_ip_address": link.local_ip_address,
                "peer_ip_address": link.peer_endpoint.ip,
            }
            for link in peer.links
        ],
        "published_path": str(published.path),
    }
    _atomic_write(
        receipt_path,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-config", type=Path, required=True)
    prepare_parser.add_argument("--receiver-config", type=Path, required=True)
    prepare_parser.add_argument("--output-directory", type=Path, required=True)
    prepare_parser.add_argument("--remote-root", type=Path, required=True)
    prepare_parser.add_argument("--port", type=int, required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--config", type=Path, required=True)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, required=True)

    transfer_parser = subparsers.add_parser("transfer")
    transfer_parser.add_argument("--config", type=Path, required=True)
    transfer_parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.operation == "prepare":
            paths = prepare_proof(
                cast(Path, arguments.source_config),
                cast(Path, arguments.receiver_config),
                cast(Path, arguments.output_directory),
                cast(Path, arguments.remote_root),
                cast(int, arguments.port),
            )
            print("\n".join(str(path) for path in paths))
        elif arguments.operation == "serve":
            serve(
                cast(Path, arguments.config),
                cast(str, arguments.host),
                cast(int, arguments.port),
            )
        elif arguments.operation == "transfer":
            asyncio.run(
                transfer(
                    cast(Path, arguments.config),
                    cast(Path, arguments.receipt),
                )
            )
        else:
            raise AssertionError("unknown proof operation")
    except (OSError, ProofError, ValueError) as error:
        print(f"Peer artifact proof failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
