from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import socket
import stat
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, NoReturn, Protocol, Self, cast, final
from urllib.parse import urlencode

import aiohttp
import psutil
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)

from exo.download.peer_artifact_transfer import (
    DEFAULT_ARTIFACT_CHUNK_SIZE_BYTES,
    PeerArtifactChunk,
    PeerArtifactIntegrityError,
    PeerArtifactLink,
    PeerArtifactLinkId,
    PeerArtifactManifest,
    PeerArtifactProtocolError,
    PeerArtifactRangeReader,
    PeerArtifactSnapshotId,
    RelativeArtifactPath,
    Sha256Digest,
    peer_artifact_manifest_fingerprint,
)
from exo.shared.models.model_cards import (
    MODEL_REVISION_RECEIPT_FILENAME,
    HuggingFaceRevision,
)
from exo.shared.types.common import ModelId, NodeId

PEER_ARTIFACT_CONFIG_ENVIRONMENT_VARIABLE = "EXO_PEER_ARTIFACT_CONFIG"
PEER_ARTIFACT_SNAPSHOT_PATH = "/v1/peer-artifacts/snapshot"
PEER_ARTIFACT_MANIFEST_PATH = "/v1/peer-artifacts/manifest"
PEER_ARTIFACT_RANGE_PATH = "/v1/peer-artifacts/range"
PEER_ARTIFACT_SNAPSHOT_SCHEMA_VERSION = 1
_FILE_READ_SIZE_BYTES = 8 * 1024 * 1024
_MAXIMUM_JSON_RESPONSE_BYTES = 32 * 1024 * 1024
_AUTHENTICATION_CANONICALIZATION = "exo-peer-artifact-request-v1"
_AUTHENTICATION_TIMESTAMP_HEADER = "x-exo-peer-timestamp"
_AUTHENTICATION_NONCE_HEADER = "x-exo-peer-nonce"
_AUTHENTICATION_SIGNATURE_HEADER = "x-exo-peer-signature"
_MANIFEST_CACHE_SCHEMA_VERSION = 1
_MAXIMUM_AUTHENTICATION_NONCES = 100_000


class PeerArtifactHttpError(Exception):
    """Base error for the authenticated peer artifact HTTP protocol."""


class PeerArtifactUnavailableError(PeerArtifactHttpError):
    """Raised when a configured peer does not currently expose an artifact."""


class PeerArtifactAuthenticationError(PeerArtifactHttpError):
    """Raised when a peer rejects the configured bearer token."""


class PeerArtifactLinkUnavailableError(PeerArtifactUnavailableError):
    """Raised when one explicitly selected network link cannot serve a request."""

    def __init__(self, link_id: PeerArtifactLinkId, message: str) -> None:
        super().__init__(message)
        self._link_id = link_id

    @property
    def link_id(self) -> PeerArtifactLinkId:
        return self._link_id


class PeerArtifactConfigurationError(Exception):
    """Raised when explicit peer artifact deployment configuration is unsafe."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _validate_relative_path(value: str, *, description: str) -> str:
    path = PurePosixPath(value)
    if (
        "\\" in value
        or "\0" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{description} must be a normalized relative POSIX path")
    return value


@final
class PeerArtifactSnapshotFile(_StrictModel):
    file_path: RelativeArtifactPath
    artifact_path: RelativeArtifactPath
    size_bytes: NonNegativeInt
    sha256: Sha256Digest
    manifest_sha256: Sha256Digest

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        return _validate_relative_path(value, description="snapshot file paths")

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _validate_relative_path(value, description="artifact paths")


@final
class PeerArtifactSnapshotManifest(_StrictModel):
    schema_version: Literal[1]
    snapshot_id: PeerArtifactSnapshotId
    model_id: ModelId
    revision: HuggingFaceRevision
    allow_partial_snapshot: bool = False
    files: tuple[PeerArtifactSnapshotFile, ...]

    @model_validator(mode="after")
    def validate_files(self) -> PeerArtifactSnapshotManifest:
        file_paths = tuple(file.file_path for file in self.files)
        artifact_paths = tuple(file.artifact_path for file in self.files)
        if tuple(sorted(file_paths)) != file_paths:
            raise ValueError("snapshot files must be sorted by file path")
        if len(set(file_paths)) != len(file_paths):
            raise ValueError("snapshot file paths must be unique")
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ValueError("snapshot artifact paths must be unique")
        expected_snapshot_id = peer_artifact_snapshot_id(
            self.model_id,
            self.revision,
            self.files,
            allow_partial_snapshot=self.allow_partial_snapshot,
        )
        if self.snapshot_id != expected_snapshot_id:
            raise ValueError("snapshot ID does not match its pinned file manifests")
        return self


def peer_artifact_snapshot_id(
    model_id: ModelId,
    revision: HuggingFaceRevision,
    files: Sequence[PeerArtifactSnapshotFile],
    *,
    allow_partial_snapshot: bool = False,
) -> PeerArtifactSnapshotId:
    canonical = json.dumps(
        {
            "canonicalization": "exo-peer-artifact-snapshot-v1",
            "model_id": str(model_id),
            "revision": revision,
            "allow_partial_snapshot": allow_partial_snapshot,
            "files": [
                file.model_dump(mode="json", by_alias=True) for file in files
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return PeerArtifactSnapshotId(hashlib.sha256(canonical).hexdigest())


@final
class PeerArtifactServedSnapshot(_StrictModel):
    model_id: ModelId
    revision: HuggingFaceRevision
    model_root_index: NonNegativeInt
    relative_directory: RelativeArtifactPath
    allow_partial_snapshot: bool = False

    @field_validator("relative_directory")
    @classmethod
    def validate_relative_directory(cls, value: str) -> str:
        return _validate_relative_path(
            value, description="served snapshot directories"
        )


@final
class PeerArtifactServerConfig(_StrictModel):
    model_roots: tuple[Path, ...]
    served_snapshots: tuple[PeerArtifactServedSnapshot, ...]
    manifest_cache_directory: Path
    chunk_size_bytes: PositiveInt = DEFAULT_ARTIFACT_CHUNK_SIZE_BYTES
    maximum_range_bytes: PositiveInt = DEFAULT_ARTIFACT_CHUNK_SIZE_BYTES
    maximum_snapshot_files: PositiveInt = 10_000

    @model_validator(mode="after")
    def validate_server(self) -> PeerArtifactServerConfig:
        if not self.model_roots:
            raise ValueError("peer artifact servers require at least one model root")
        if not self.served_snapshots:
            raise ValueError(
                "peer artifact servers require at least one explicitly served snapshot"
            )
        if any(not root.is_absolute() for root in self.model_roots):
            raise ValueError("peer artifact model roots must be absolute paths")
        if not self.manifest_cache_directory.is_absolute():
            raise ValueError(
                "peer artifact manifest cache directory must be absolute"
            )
        snapshot_ids = tuple(
            (snapshot.model_id, snapshot.revision)
            for snapshot in self.served_snapshots
        )
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ValueError("served model snapshots must be unique")
        if any(
            snapshot.model_root_index >= len(self.model_roots)
            for snapshot in self.served_snapshots
        ):
            raise ValueError("a served snapshot references an unknown model root")
        if self.maximum_range_bytes < self.chunk_size_bytes:
            raise ValueError(
                "maximum peer artifact range must cover one manifest chunk"
            )
        return self


@final
class PeerArtifactPeerConfig(_StrictModel):
    peer_node_id: NodeId
    links: tuple[PeerArtifactLink, ...]

    @model_validator(mode="after")
    def validate_links(self) -> PeerArtifactPeerConfig:
        if not self.links:
            raise ValueError("configured artifact peers require at least one link")
        if any(link.peer_node_id != self.peer_node_id for link in self.links):
            raise ValueError("every peer link must name its containing peer")
        link_ids = tuple(link.link_id for link in self.links)
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("configured peer link IDs must be unique")
        return self


@final
class PeerArtifactDeploymentConfig(_StrictModel):
    schema_version: Literal[1]
    authentication_secret: SecretStr
    peers: tuple[PeerArtifactPeerConfig, ...] = ()
    server: PeerArtifactServerConfig | None = None
    disk_cache_directory: Path | None = None
    memory_cache_directory: Path | None = None
    disk_reserve_bytes: NonNegativeInt = 0
    memory_reserve_bytes: NonNegativeInt = 0
    request_timeout_seconds: PositiveInt = 1_800
    authentication_window_seconds: PositiveInt = 300
    fallback_to_origin: bool = True

    @field_validator("authentication_secret")
    @classmethod
    def validate_authentication_secret(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError(
                "peer artifact authentication secrets must contain at least 32 bytes"
            )
        return value

    @model_validator(mode="after")
    def validate_deployment(self) -> PeerArtifactDeploymentConfig:
        peer_ids = tuple(peer.peer_node_id for peer in self.peers)
        if len(set(peer_ids)) != len(peer_ids):
            raise ValueError("configured artifact peer node IDs must be unique")
        if self.peers and self.disk_cache_directory is None:
            raise ValueError(
                "peer artifact clients require an explicit disk cache directory"
            )
        if (
            self.disk_cache_directory is not None
            and not self.disk_cache_directory.is_absolute()
        ):
            raise ValueError("peer artifact disk cache paths must be absolute")
        if (
            self.memory_cache_directory is not None
            and not self.memory_cache_directory.is_absolute()
        ):
            raise ValueError("peer artifact memory cache paths must be absolute")
        if (
            self.memory_cache_directory is not None
            and self.memory_cache_directory == self.disk_cache_directory
        ):
            raise ValueError("peer artifact disk and memory caches must differ")
        return self


def load_peer_artifact_deployment_config(
    path: Path | None,
) -> PeerArtifactDeploymentConfig | None:
    """Load an explicitly named, owner-only JSON deployment configuration."""
    if path is None:
        return None
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | no_follow)
    except OSError as error:
        raise PeerArtifactConfigurationError(
            f"cannot open peer artifact configuration {path}"
        ) from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise PeerArtifactConfigurationError(
                "peer artifact configuration must be a regular file"
            )
        if file_stat.st_mode & 0o077:
            raise PeerArtifactConfigurationError(
                "peer artifact configuration contains an authentication secret and "
                "must have "
                "owner-only permissions"
            )
        if file_stat.st_size > 4 * 1024 * 1024:
            raise PeerArtifactConfigurationError(
                "peer artifact configuration exceeds the 4 MiB limit"
            )
        contents = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, file_stat.st_size + 1)):
            contents.extend(chunk)
            if len(contents) > 4 * 1024 * 1024:
                raise PeerArtifactConfigurationError(
                    "peer artifact configuration exceeds the 4 MiB limit"
                )
    finally:
        os.close(descriptor)
    try:
        return PeerArtifactDeploymentConfig.model_validate_json(contents)
    except ValueError as error:
        raise PeerArtifactConfigurationError(
            f"invalid peer artifact configuration {path}"
        ) from error


def _endpoint_url(link: PeerArtifactLink, path: str) -> str:
    host = (
        f"[{link.peer_endpoint.ip}]"
        if ":" in link.peer_endpoint.ip
        else link.peer_endpoint.ip
    )
    return f"http://{host}:{link.peer_endpoint.port}{path}"


def _canonical_query(parameters: Sequence[tuple[str, str]]) -> str:
    return urlencode(sorted(parameters), doseq=True)


def _request_signature(
    secret: SecretStr,
    *,
    method: str,
    path: str,
    parameters: Sequence[tuple[str, str]],
    timestamp: str,
    nonce: str,
) -> str:
    canonical_request = "\n".join(
        (
            _AUTHENTICATION_CANONICALIZATION,
            method.upper(),
            path,
            _canonical_query(parameters),
            timestamp,
            nonce,
        )
    ).encode()
    return hmac.new(
        secret.get_secret_value().encode(),
        canonical_request,
        hashlib.sha256,
    ).hexdigest()


def build_peer_artifact_authentication_headers(
    secret: SecretStr,
    *,
    method: str,
    path: str,
    parameters: Sequence[tuple[str, str]],
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    timestamp_value = str(int(time.time()) if timestamp is None else timestamp)
    nonce_value = secrets.token_hex(16) if nonce is None else nonce
    return {
        _AUTHENTICATION_TIMESTAMP_HEADER: timestamp_value,
        _AUTHENTICATION_NONCE_HEADER: nonce_value,
        _AUTHENTICATION_SIGNATURE_HEADER: _request_signature(
            secret,
            method=method,
            path=path,
            parameters=parameters,
            timestamp=timestamp_value,
            nonce=nonce_value,
        ),
    }


class PeerArtifactPeerClient(PeerArtifactRangeReader, Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None: ...

    async def fetch_snapshot_manifest(
        self,
        *,
        link: PeerArtifactLink,
        model_id: ModelId,
        revision: HuggingFaceRevision,
    ) -> PeerArtifactSnapshotManifest: ...

    async def fetch_artifact_manifest(
        self,
        *,
        link: PeerArtifactLink,
        snapshot_id: PeerArtifactSnapshotId,
        expected_manifest_sha256: Sha256Digest,
        artifact_path: RelativeArtifactPath,
    ) -> PeerArtifactManifest: ...


def _validate_link_interface(link: PeerArtifactLink) -> None:
    interface_addresses = psutil.net_if_addrs().get(link.local_interface)
    if interface_addresses is None:
        raise PeerArtifactConfigurationError(
            f"peer artifact interface {link.local_interface!r} does not exist"
        )
    configured_address = link.local_ip_address.split("%", maxsplit=1)[0]
    assigned_addresses = {
        address.address.split("%", maxsplit=1)[0]
        for address in interface_addresses
        if address.family in {socket.AF_INET, socket.AF_INET6}
    }
    if configured_address not in assigned_addresses:
        raise PeerArtifactConfigurationError(
            f"peer artifact source address {link.local_ip_address} is not assigned "
            f"to interface {link.local_interface}"
        )


def _bound_socket_factory(
    interface_name: str,
) -> Callable[[aiohttp.AddrInfoType], socket.socket]:
    bind_to_device = getattr(socket, "SO_BINDTODEVICE", None)
    if bind_to_device is None:
        raise PeerArtifactConfigurationError(
            "peer artifact rail pinning requires SO_BINDTODEVICE"
        )

    def create_socket(address_info: aiohttp.AddrInfoType) -> socket.socket:
        family, socket_type, protocol, _, _ = address_info
        result = socket.socket(family, socket_type, protocol)
        try:
            result.setsockopt(
                socket.SOL_SOCKET,
                bind_to_device,
                interface_name.encode() + b"\0",
            )
        except BaseException:
            result.close()
            raise
        return result

    return create_socket


@final
class PeerArtifactHttpClient(PeerArtifactPeerClient):
    """Authenticated HTTP client with one source-address-bound pool per link."""

    def __init__(self, authentication_secret: SecretStr, timeout_seconds: int) -> None:
        self._authentication_secret = authentication_secret
        self._timeout = aiohttp.ClientTimeout(
            total=timeout_seconds,
            connect=min(timeout_seconds, 30),
            sock_connect=min(timeout_seconds, 30),
            sock_read=min(timeout_seconds, 300),
        )
        self._sessions: dict[PeerArtifactLinkId, aiohttp.ClientSession] = {}
        self._received_bytes: dict[PeerArtifactLinkId, int] = {}

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback
        await self.close()

    async def close(self) -> None:
        sessions = tuple(self._sessions.values())
        self._sessions.clear()
        await asyncio.gather(
            *(session.close() for session in sessions),
            return_exceptions=False,
        )

    def _session(self, link: PeerArtifactLink) -> aiohttp.ClientSession:
        existing = self._sessions.get(link.link_id)
        if existing is not None:
            return existing
        _validate_link_interface(link)
        connector = aiohttp.TCPConnector(
            local_addr=(link.local_ip_address, 0),
            socket_factory=_bound_socket_factory(link.local_interface),
            limit_per_host=link.maximum_concurrent_chunks,
            ttl_dns_cache=0,
        )
        session = aiohttp.ClientSession(
            connector=connector,
            auto_decompress=False,
            timeout=self._timeout,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
        )
        self._sessions[link.link_id] = session
        self._received_bytes[link.link_id] = 0
        return session

    @property
    def received_bytes_by_link(self) -> dict[PeerArtifactLinkId, int]:
        return dict(self._received_bytes)

    async def _get_bytes(
        self,
        link: PeerArtifactLink,
        path: str,
        *,
        params: dict[str, str | int],
        maximum_response_bytes: int,
    ) -> bytes:
        authentication_parameters = tuple(
            (name, str(value)) for name, value in params.items()
        )
        headers = build_peer_artifact_authentication_headers(
            self._authentication_secret,
            method="GET",
            path=path,
            parameters=authentication_parameters,
        )
        try:
            async with self._session(link).get(
                _endpoint_url(link, path),
                params=params,
                headers=headers,
            ) as response:
                if response.status in {401, 403}:
                    raise PeerArtifactAuthenticationError(
                        f"peer {link.peer_node_id} rejected artifact credentials"
                    )
                if response.status == 404:
                    raise PeerArtifactUnavailableError(
                        f"peer {link.peer_node_id} does not expose the requested artifact"
                    )
                if response.status != 200:
                    raise PeerArtifactLinkUnavailableError(
                        link.link_id,
                        f"link {link.link_id} returned HTTP {response.status}",
                    )
                content_length = response.content_length
                if (
                    content_length is not None
                    and content_length > maximum_response_bytes
                ):
                    raise PeerArtifactProtocolError(
                        f"link {link.link_id} declared an oversized response"
                    )
                contents = bytearray()
                async for chunk in response.content.iter_chunked(
                    min(_FILE_READ_SIZE_BYTES, maximum_response_bytes + 1)
                ):
                    if len(contents) + len(chunk) > maximum_response_bytes:
                        raise PeerArtifactProtocolError(
                            f"link {link.link_id} returned an oversized response"
                        )
                    contents.extend(chunk)
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
            asyncio.TimeoutError,
            OSError,
        ) as error:
            raise PeerArtifactLinkUnavailableError(
                link.link_id,
                f"link {link.link_id} is unavailable",
            ) from error
        self._received_bytes[link.link_id] = (
            self._received_bytes.get(link.link_id, 0) + len(contents)
        )
        return bytes(contents)

    async def fetch_snapshot_manifest(
        self,
        *,
        link: PeerArtifactLink,
        model_id: ModelId,
        revision: HuggingFaceRevision,
    ) -> PeerArtifactSnapshotManifest:
        contents = await self._get_bytes(
            link,
            PEER_ARTIFACT_SNAPSHOT_PATH,
            params={"model_id": str(model_id), "revision": revision},
            maximum_response_bytes=_MAXIMUM_JSON_RESPONSE_BYTES,
        )
        try:
            return PeerArtifactSnapshotManifest.model_validate_json(contents)
        except ValueError as error:
            raise PeerArtifactProtocolError(
                f"link {link.link_id} returned an invalid snapshot manifest"
            ) from error

    async def fetch_artifact_manifest(
        self,
        *,
        link: PeerArtifactLink,
        snapshot_id: PeerArtifactSnapshotId,
        expected_manifest_sha256: Sha256Digest,
        artifact_path: RelativeArtifactPath,
    ) -> PeerArtifactManifest:
        contents = await self._get_bytes(
            link,
            PEER_ARTIFACT_MANIFEST_PATH,
            params={
                "snapshot_id": snapshot_id,
                "artifact_path": artifact_path,
                "manifest_sha256": expected_manifest_sha256,
            },
            maximum_response_bytes=_MAXIMUM_JSON_RESPONSE_BYTES,
        )
        try:
            manifest = PeerArtifactManifest.model_validate_json(contents)
        except ValueError as error:
            raise PeerArtifactProtocolError(
                f"link {link.link_id} returned an invalid artifact manifest"
            ) from error
        if manifest.artifact_path != artifact_path:
            raise PeerArtifactProtocolError(
                f"link {link.link_id} returned a manifest for another artifact"
            )
        if (
            peer_artifact_manifest_fingerprint(manifest)
            != expected_manifest_sha256
        ):
            raise PeerArtifactProtocolError(
                f"link {link.link_id} returned an unpinned artifact manifest"
            )
        return manifest

    async def read_artifact_range(
        self,
        *,
        link: PeerArtifactLink,
        snapshot_id: PeerArtifactSnapshotId,
        artifact_sha256: Sha256Digest,
        artifact_path: RelativeArtifactPath,
        offset_bytes: int,
        size_bytes: int,
    ) -> bytes:
        return await self._get_bytes(
            link,
            PEER_ARTIFACT_RANGE_PATH,
            params={
                "snapshot_id": snapshot_id,
                "artifact_sha256": artifact_sha256,
                "artifact_path": artifact_path,
                "offset_bytes": offset_bytes,
                "size_bytes": size_bytes,
            },
            maximum_response_bytes=size_bytes,
        )


@final
class _PinnedFileIdentity(_StrictModel):
    device: NonNegativeInt
    inode: NonNegativeInt
    size_bytes: NonNegativeInt
    modified_time_nanoseconds: NonNegativeInt
    changed_time_nanoseconds: NonNegativeInt


@final
class _PinnedSnapshotArtifact(_StrictModel):
    file: PeerArtifactSnapshotFile
    identity: _PinnedFileIdentity
    manifest: PeerArtifactManifest


@final
class _PersistedSnapshot(_StrictModel):
    schema_version: Literal[1]
    snapshot: PeerArtifactSnapshotManifest
    artifacts: tuple[_PinnedSnapshotArtifact, ...]


@dataclass(frozen=True, slots=True)
class _ServedSnapshotState:
    configuration: PeerArtifactServedSnapshot
    directory_descriptor: int


def _identity(file_stat: os.stat_result) -> _PinnedFileIdentity:
    return _PinnedFileIdentity(
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        size_bytes=file_stat.st_size,
        modified_time_nanoseconds=file_stat.st_mtime_ns,
        changed_time_nanoseconds=file_stat.st_ctime_ns,
    )


def _open_absolute_directory_without_symlinks(path: Path) -> int:
    if not path.is_absolute():
        raise PeerArtifactConfigurationError("directory path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_directory_beneath(
    root_descriptor: int, relative_directory: str
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in PurePosixPath(relative_directory).parts:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular_file_beneath(
    root_descriptor: int, relative_file_path: str
) -> int:
    normalized = _validate_relative_path(
        relative_file_path, description="snapshot file paths"
    )
    parts = PurePosixPath(normalized).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)
    file_stat = os.fstat(file_descriptor)
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(file_descriptor)
        raise PeerArtifactUnavailableError(
            "the requested artifact is not a regular file"
        )
    return file_descriptor


def _manifest_from_descriptor(
    descriptor: int,
    artifact_path: str,
    chunk_size_bytes: int,
) -> PeerArtifactManifest:
    before = os.fstat(descriptor)
    chunks: list[PeerArtifactChunk] = []
    artifact_hasher = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        contents = os.pread(
            descriptor,
            min(chunk_size_bytes, before.st_size - offset),
            offset,
        )
        if not contents:
            raise PeerArtifactIntegrityError(
                "artifact ended while its pinned manifest was built"
            )
        artifact_hasher.update(contents)
        chunks.append(
            PeerArtifactChunk(
                offset_bytes=offset,
                size_bytes=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
            )
        )
        offset += len(contents)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after):
        raise PeerArtifactIntegrityError(
            "artifact identity changed while its manifest was built"
        )
    return PeerArtifactManifest(
        schema_version=1,
        canonicalization="exo-peer-artifact-manifest-v1",
        artifact_path=artifact_path,
        size_bytes=offset,
        sha256=artifact_hasher.hexdigest(),
        chunks=tuple(chunks),
    )


@final
class PeerArtifactHttpSource:
    """Root-confined, immutable snapshot view for authenticated HTTP endpoints."""

    def __init__(self, config: PeerArtifactServerConfig) -> None:
        self._config = config
        self._model_root_descriptors: list[int] = []
        try:
            for root in config.model_roots:
                self._model_root_descriptors.append(
                    _open_absolute_directory_without_symlinks(root)
                )
        except OSError as error:
            for descriptor in self._model_root_descriptors:
                os.close(descriptor)
            raise PeerArtifactConfigurationError(
                "a model root or one of its components is unavailable or symlinked"
            ) from error
        self._snapshot_states: dict[
            tuple[ModelId, HuggingFaceRevision], _ServedSnapshotState
        ] = {}
        try:
            for snapshot in config.served_snapshots:
                descriptor = _open_directory_beneath(
                    self._model_root_descriptors[snapshot.model_root_index],
                    snapshot.relative_directory,
                )
                self._snapshot_states[(snapshot.model_id, snapshot.revision)] = (
                    _ServedSnapshotState(snapshot, descriptor)
                )
        except OSError as error:
            self.close()
            raise PeerArtifactConfigurationError(
                "a served snapshot contains a missing or symlinked directory"
            ) from error
        config.manifest_cache_directory.mkdir(
            mode=0o700, parents=True, exist_ok=True
        )
        cache_stat = config.manifest_cache_directory.lstat()
        if (
            stat.S_ISLNK(cache_stat.st_mode)
            or not stat.S_ISDIR(cache_stat.st_mode)
            or cache_stat.st_uid != os.geteuid()
            or cache_stat.st_mode & 0o022
        ):
            self.close()
            raise PeerArtifactConfigurationError(
                "manifest cache must be an owner-controlled non-symlink directory"
            )
        self._manifest_cache_directory = config.manifest_cache_directory
        self._pinned_snapshots: dict[
            PeerArtifactSnapshotId, _PersistedSnapshot
        ] = {}

    def close(self) -> None:
        for state in self._snapshot_states.values():
            os.close(state.directory_descriptor)
        self._snapshot_states.clear()
        for descriptor in self._model_root_descriptors:
            os.close(descriptor)
        self._model_root_descriptors.clear()

    def _cache_path(self, kind: str, identity_hash: str) -> Path:
        directory = self._manifest_cache_directory / kind
        directory.mkdir(mode=0o700, exist_ok=True)
        directory_stat = directory.lstat()
        if (
            stat.S_ISLNK(directory_stat.st_mode)
            or not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or directory_stat.st_mode & 0o022
        ):
            raise PeerArtifactConfigurationError(
                f"unsafe persistent manifest cache directory {directory}"
            )
        return directory / f"{identity_hash}.json"

    def _read_cache(self, path: Path, maximum_bytes: int) -> bytes | None:
        try:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except FileNotFoundError:
            return None
        try:
            contents = os.read(descriptor, maximum_bytes + 1)
        finally:
            os.close(descriptor)
        if len(contents) > maximum_bytes:
            raise PeerArtifactConfigurationError(
                f"persistent peer manifest cache entry is oversized: {path}"
            )
        return contents

    def _write_immutable_cache(self, path: Path, contents: bytes) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            existing = self._read_cache(path, len(contents))
            if existing != contents:
                raise PeerArtifactIntegrityError(
                    f"immutable peer manifest cache collision at {path}"
                ) from error
            return
        try:
            if os.write(descriptor, contents) != len(contents):
                raise OSError("short immutable peer manifest cache write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    async def _artifact_record(
        self, descriptor: int, artifact_path: str
    ) -> _PinnedSnapshotArtifact:
        file_identity = _identity(os.fstat(descriptor))
        identity_payload = json.dumps(
            {
                "artifact_path": artifact_path,
                "chunk_size_bytes": int(self._config.chunk_size_bytes),
                "identity": file_identity.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        identity_hash = hashlib.sha256(identity_payload).hexdigest()
        cache_path = self._cache_path("manifests", identity_hash)
        cached = self._read_cache(cache_path, _MAXIMUM_JSON_RESPONSE_BYTES)
        if cached is not None:
            record = _PinnedSnapshotArtifact.model_validate_json(cached)
            if (
                record.identity != file_identity
                or record.manifest.artifact_path != artifact_path
            ):
                raise PeerArtifactIntegrityError(
                    "persistent manifest cache identity mismatch"
                )
            return record
        manifest = await to_thread.run_sync(
            _manifest_from_descriptor,
            descriptor,
            artifact_path,
            int(self._config.chunk_size_bytes),
        )
        snapshot_file = PeerArtifactSnapshotFile(
            file_path="placeholder",
            artifact_path=artifact_path,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            manifest_sha256=peer_artifact_manifest_fingerprint(manifest),
        )
        record = _PinnedSnapshotArtifact(
            file=snapshot_file,
            identity=file_identity,
            manifest=manifest,
        )
        self._write_immutable_cache(
            cache_path, record.model_dump_json(by_alias=True).encode()
        )
        return record

    async def snapshot_manifest(
        self, model_id: ModelId, revision: HuggingFaceRevision
    ) -> PeerArtifactSnapshotManifest:
        state = self._snapshot_states.get((model_id, revision))
        if state is None:
            raise PeerArtifactUnavailableError(
                "the requested model snapshot is not configured for peer serving"
            )
        artifacts: list[_PinnedSnapshotArtifact] = []
        for directory, directory_names, file_names, directory_descriptor in os.fwalk(
            ".",
            topdown=True,
            follow_symlinks=False,
            dir_fd=state.directory_descriptor,
        ):
            directory_names[:] = [
                name for name in directory_names if name != ".cache"
            ]
            for directory_name in directory_names:
                directory_stat = os.stat(
                    directory_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(directory_stat.st_mode):
                    raise PeerArtifactUnavailableError(
                        "served snapshots cannot contain directory symlinks"
                    )
            relative_directory = PurePosixPath(directory)
            for file_name in file_names:
                if (
                    file_name == MODEL_REVISION_RECEIPT_FILENAME
                    or file_name.endswith(".partial")
                ):
                    continue
                file_path = (
                    relative_directory / file_name
                ).as_posix().removeprefix("./")
                artifact_path = (
                    PurePosixPath(state.configuration.relative_directory)
                    / file_path
                ).as_posix()
                try:
                    descriptor = _open_regular_file_beneath(
                        state.directory_descriptor, file_path
                    )
                except OSError as error:
                    raise PeerArtifactUnavailableError(
                        "served snapshots cannot contain symlinked artifacts"
                    ) from error
                try:
                    cached_record = await self._artifact_record(
                        descriptor, artifact_path
                    )
                finally:
                    os.close(descriptor)
                record = cached_record.model_copy(
                    update={
                        "file": cached_record.file.model_copy(
                            update={"file_path": file_path}
                        )
                    }
                )
                artifacts.append(record)
                if len(artifacts) > self._config.maximum_snapshot_files:
                    raise PeerArtifactConfigurationError(
                        "served snapshot exceeds its configured file-count limit"
                    )
        artifacts.sort(key=lambda artifact: artifact.file.file_path)
        files = tuple(artifact.file for artifact in artifacts)
        snapshot_id = peer_artifact_snapshot_id(
            model_id,
            revision,
            files,
            allow_partial_snapshot=state.configuration.allow_partial_snapshot,
        )
        snapshot = PeerArtifactSnapshotManifest(
            schema_version=PEER_ARTIFACT_SNAPSHOT_SCHEMA_VERSION,
            snapshot_id=snapshot_id,
            model_id=model_id,
            revision=revision,
            allow_partial_snapshot=state.configuration.allow_partial_snapshot,
            files=files,
        )
        persisted = _PersistedSnapshot(
            schema_version=_MANIFEST_CACHE_SCHEMA_VERSION,
            snapshot=snapshot,
            artifacts=tuple(artifacts),
        )
        self._write_immutable_cache(
            self._cache_path("snapshots", str(snapshot_id)),
            persisted.model_dump_json(by_alias=True).encode(),
        )
        self._pinned_snapshots[snapshot_id] = persisted
        return snapshot

    def _pinned_artifact(
        self,
        snapshot_id: PeerArtifactSnapshotId,
        artifact_path: str,
    ) -> tuple[int, _PinnedSnapshotArtifact]:
        persisted = self._pinned_snapshots.get(snapshot_id)
        if persisted is None:
            cached = self._read_cache(
                self._cache_path("snapshots", str(snapshot_id)),
                _MAXIMUM_JSON_RESPONSE_BYTES,
            )
            if cached is None:
                raise PeerArtifactUnavailableError(
                    "the requested pinned snapshot is unavailable"
                )
            persisted = _PersistedSnapshot.model_validate_json(cached)
            if persisted.snapshot.snapshot_id != snapshot_id:
                raise PeerArtifactIntegrityError(
                    "persistent snapshot cache identity mismatch"
                )
            self._pinned_snapshots[snapshot_id] = persisted
        state = self._snapshot_states.get(
            (persisted.snapshot.model_id, persisted.snapshot.revision)
        )
        if state is None:
            raise PeerArtifactUnavailableError(
                "the requested pinned snapshot is no longer configured"
            )
        record = next(
            (
                artifact
                for artifact in persisted.artifacts
                if artifact.file.artifact_path == artifact_path
            ),
            None,
        )
        if record is None:
            raise PeerArtifactUnavailableError(
                "artifact is not a member of the requested pinned snapshot"
            )
        prefix = f"{state.configuration.relative_directory}/"
        if not artifact_path.startswith(prefix):
            raise PeerArtifactIntegrityError(
                "pinned artifact escaped its selected snapshot root"
            )
        descriptor = _open_regular_file_beneath(
            state.directory_descriptor,
            artifact_path.removeprefix(prefix),
        )
        if _identity(os.fstat(descriptor)) != record.identity:
            os.close(descriptor)
            raise PeerArtifactUnavailableError(
                "artifact identity drifted after the snapshot was pinned"
            )
        return descriptor, record

    def artifact_manifest(
        self,
        snapshot_id: PeerArtifactSnapshotId,
        artifact_path: str,
        expected_manifest_sha256: Sha256Digest,
    ) -> PeerArtifactManifest:
        descriptor, record = self._pinned_artifact(snapshot_id, artifact_path)
        os.close(descriptor)
        if record.file.manifest_sha256 != expected_manifest_sha256:
            raise PeerArtifactUnavailableError(
                "requested manifest digest is not pinned by the snapshot"
            )
        return record.manifest

    def open_range(
        self,
        snapshot_id: PeerArtifactSnapshotId,
        artifact_path: str,
        expected_artifact_sha256: Sha256Digest,
        offset_bytes: int,
        size_bytes: int,
    ) -> int:
        if offset_bytes < 0 or size_bytes <= 0:
            raise ValueError(
                "artifact ranges require nonnegative offsets and positive sizes"
            )
        if size_bytes > self._config.maximum_range_bytes:
            raise ValueError(
                "requested artifact range exceeds the configured maximum"
            )
        descriptor, record = self._pinned_artifact(snapshot_id, artifact_path)
        if record.file.sha256 != expected_artifact_sha256:
            os.close(descriptor)
            raise PeerArtifactUnavailableError(
                "requested content digest is not pinned by the snapshot"
            )
        if offset_bytes + size_bytes > record.file.size_bytes:
            os.close(descriptor)
            raise ValueError("requested artifact range is outside the file")
        return descriptor


@final
class _RequestAuthenticator:
    def __init__(self, secret: SecretStr, window_seconds: int) -> None:
        self._secret = secret
        self._window_seconds = window_seconds
        self._nonces: dict[str, int] = {}

    def authorize(self, request: Request) -> None:
        timestamp_header = request.headers.get(_AUTHENTICATION_TIMESTAMP_HEADER, "")
        nonce = request.headers.get(_AUTHENTICATION_NONCE_HEADER, "")
        supplied_signature = request.headers.get(
            _AUTHENTICATION_SIGNATURE_HEADER, ""
        )
        try:
            timestamp = int(timestamp_header)
        except ValueError:
            self._reject()
        current_timestamp = int(time.time())
        if (
            abs(current_timestamp - timestamp) > self._window_seconds
            or len(nonce) != 32
            or any(character not in "0123456789abcdef" for character in nonce)
            or len(supplied_signature) != 64
        ):
            self._reject()
        expired_before = current_timestamp - self._window_seconds
        self._nonces = {
            existing_nonce: existing_timestamp
            for existing_nonce, existing_timestamp in self._nonces.items()
            if existing_timestamp >= expired_before
        }
        if nonce in self._nonces:
            self._reject()
        parameters = tuple(
            (name, value) for name, value in request.query_params.multi_items()
        )
        expected_signature = _request_signature(
            self._secret,
            method=request.method,
            path=request.url.path,
            parameters=parameters,
            timestamp=timestamp_header,
            nonce=nonce,
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            self._reject()
        if len(self._nonces) >= _MAXIMUM_AUTHENTICATION_NONCES:
            oldest_nonce = next(iter(self._nonces))
            del self._nonces[oldest_nonce]
        self._nonces[nonce] = timestamp

    @staticmethod
    def _reject() -> NoReturn:
        raise HTTPException(
            status_code=401,
            detail="valid peer artifact request authentication required",
        )


def install_peer_artifact_http_routes(
    app: FastAPI,
    config: PeerArtifactDeploymentConfig | None,
) -> None:
    """Install authenticated routes only when server configuration is explicit."""
    if config is None or config.server is None:
        return
    source = PeerArtifactHttpSource(config.server)
    authenticator = _RequestAuthenticator(
        config.authentication_secret,
        int(config.authentication_window_seconds),
    )

    async def snapshot_manifest(
        request: Request,
        model_id: Annotated[str, Query(min_length=1, max_length=512)],
        revision: Annotated[str, Query(pattern=r"^(?:main|[0-9a-f]{40})$")],
    ) -> PeerArtifactSnapshotManifest:
        authenticator.authorize(request)
        try:
            return await source.snapshot_manifest(
                ModelId(model_id),
                revision,
            )
        except PeerArtifactUnavailableError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (PeerArtifactConfigurationError, OSError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    async def artifact_manifest(
        request: Request,
        snapshot_id: Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")],
        artifact_path: Annotated[str, Query(min_length=1, max_length=4096)],
        manifest_sha256: Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")],
    ) -> PeerArtifactManifest:
        authenticator.authorize(request)
        try:
            return source.artifact_manifest(
                PeerArtifactSnapshotId(snapshot_id),
                artifact_path,
                cast(Sha256Digest, manifest_sha256),
            )
        except PeerArtifactUnavailableError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (PeerArtifactConfigurationError, OSError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    async def artifact_range(
        request: Request,
        snapshot_id: Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")],
        artifact_sha256: Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")],
        artifact_path: Annotated[str, Query(min_length=1, max_length=4096)],
        offset_bytes: Annotated[int, Query(ge=0)],
        size_bytes: Annotated[int, Query(gt=0)],
    ) -> StreamingResponse:
        authenticator.authorize(request)
        try:
            descriptor = source.open_range(
                PeerArtifactSnapshotId(snapshot_id),
                artifact_path,
                cast(Sha256Digest, artifact_sha256),
                offset_bytes,
                size_bytes,
            )
        except PeerArtifactUnavailableError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=416, detail=str(error)) from error

        async def contents() -> AsyncIterator[bytes]:
            consumed = 0
            try:
                while consumed < size_bytes:
                    chunk = await to_thread.run_sync(
                        os.pread,
                        descriptor,
                        min(_FILE_READ_SIZE_BYTES, size_bytes - consumed),
                        offset_bytes + consumed,
                    )
                    if not chunk:
                        raise OSError("artifact changed while its range was served")
                    consumed += len(chunk)
                    yield chunk
            finally:
                os.close(descriptor)

        return StreamingResponse(
            contents(),
            media_type="application/octet-stream",
            headers={
                "Content-Length": str(size_bytes),
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    app.add_api_route(
        PEER_ARTIFACT_SNAPSHOT_PATH,
        snapshot_manifest,
        methods=["GET"],
        response_model=PeerArtifactSnapshotManifest,
        include_in_schema=False,
    )
    app.add_api_route(
        PEER_ARTIFACT_MANIFEST_PATH,
        artifact_manifest,
        methods=["GET"],
        response_model=PeerArtifactManifest,
        include_in_schema=False,
    )
    app.add_api_route(
        PEER_ARTIFACT_RANGE_PATH,
        artifact_range,
        methods=["GET"],
        response_model=None,
        include_in_schema=False,
    )
    app.state.peer_artifact_http_source = source


def links_by_preference(
    links: Sequence[PeerArtifactLink],
) -> tuple[PeerArtifactLink, ...]:
    return tuple(
        sorted(
            links,
            key=lambda link: (
                -int(link.estimated_bytes_per_second),
                str(link.link_id),
            ),
        )
    )
