import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final


class SglangKtReceiptFileError(ValueError):
    """Raised when receipt evidence cannot be read without ambiguity."""


@final
@dataclass(frozen=True)
class SglangKtBoundFile:
    path: Path
    contents: bytes
    sha256: str


@final
@dataclass(frozen=True)
class SglangKtBoundFileHash:
    path: Path
    size_bytes: int
    sha256: str


def _open_absolute_regular_file(path: Path) -> int:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise SglangKtReceiptFileError(f"path must be absolute and normalized: {path}")
    if path == Path("/"):
        raise SglangKtReceiptFileError("receipt path must name a regular file")

    directory_descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY,
    )
    try:
        for component in path.parts[1:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise SglangKtReceiptFileError(
            f"cannot open regular file without following symlinks: {path}"
        ) from error
    finally:
        os.close(directory_descriptor)


def _stable_file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_nlink,
        file_stat.st_uid,
        file_stat.st_gid,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def read_sglang_kt_bound_file(
    path: Path,
    *,
    maximum_bytes: int,
) -> SglangKtBoundFile:
    """Read one stable regular file through a no-symlink descriptor walk."""

    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    descriptor = _open_absolute_regular_file(path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SglangKtReceiptFileError(
                f"receipt evidence must be a singly linked regular file: {path}"
            )
        if before.st_size > maximum_bytes:
            raise SglangKtReceiptFileError(
                f"receipt evidence exceeds {maximum_bytes} bytes: {path}"
            )

        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(
                descriptor, min(1024 * 1024, maximum_bytes + 1 - bytes_read)
            )
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > maximum_bytes:
                raise SglangKtReceiptFileError(
                    f"receipt evidence exceeds {maximum_bytes} bytes: {path}"
                )
        after = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(after):
            raise SglangKtReceiptFileError(
                f"receipt evidence changed while it was being read: {path}"
            )
        contents = b"".join(chunks)
        if len(contents) != before.st_size:
            raise SglangKtReceiptFileError(
                f"receipt evidence size changed while it was being read: {path}"
            )
        return SglangKtBoundFile(
            path=path,
            contents=contents,
            sha256=hashlib.sha256(contents).hexdigest(),
        )
    except OSError as error:
        raise SglangKtReceiptFileError(
            f"cannot read receipt evidence: {path}"
        ) from error
    finally:
        os.close(descriptor)


def hash_sglang_kt_bound_file(
    path: Path,
    *,
    expected_size_bytes: int | None = None,
) -> SglangKtBoundFileHash:
    """Hash a stable regular file without retaining its contents in memory."""

    if expected_size_bytes is not None and expected_size_bytes < 0:
        raise ValueError("expected_size_bytes must be nonnegative")
    descriptor = _open_absolute_regular_file(path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SglangKtReceiptFileError(
                f"receipt artifact must be a singly linked regular file: {path}"
            )
        if expected_size_bytes is not None and before.st_size != expected_size_bytes:
            raise SglangKtReceiptFileError(
                f"receipt artifact has size {before.st_size}, expected "
                f"{expected_size_bytes}: {path}"
            )

        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(after):
            raise SglangKtReceiptFileError(
                f"receipt artifact changed while it was being hashed: {path}"
            )
        if bytes_read != before.st_size:
            raise SglangKtReceiptFileError(
                f"receipt artifact size changed while it was being hashed: {path}"
            )
        return SglangKtBoundFileHash(
            path=path,
            size_bytes=bytes_read,
            sha256=digest.hexdigest(),
        )
    except OSError as error:
        raise SglangKtReceiptFileError(
            f"cannot hash receipt artifact: {path}"
        ) from error
    finally:
        os.close(descriptor)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in pairs:
        if key in values:
            raise SglangKtReceiptFileError(f"JSON object repeats key {key!r}")
        values[key] = value
    return values


def _reject_nonfinite_json(value: str) -> object:
    raise SglangKtReceiptFileError(f"JSON contains non-finite number {value}")


def parse_sglang_kt_strict_json(contents: bytes) -> object:
    try:
        text = contents.decode("utf-8")
        return cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SglangKtReceiptFileError("receipt is not strict UTF-8 JSON") from error


def canonical_sglang_kt_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SglangKtReceiptFileError(
            "value cannot be canonicalized as JSON"
        ) from error
