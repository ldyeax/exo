#!/usr/bin/env python3
"""Run one argv-only local command under the reversible CPU performance policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol, Self, cast

try:
    from scripts.cpu_performance_policy import cpu_performance_policy
    from scripts.run_dsv4_flash_cpu_policy_campaign import (
        validate_restored_policy_evidence,
    )
except ModuleNotFoundError:
    from cpu_performance_policy import cpu_performance_policy
    from run_dsv4_flash_cpu_policy_campaign import (
        validate_restored_policy_evidence,
    )


RECEIPT_FORMAT: Final = "cpu_performance_policy_command_v1"
_SCRIPT_PATH: Final = Path(__file__).resolve()
_POLICY_HELPER_PATH: Final = _SCRIPT_PATH.with_name("cpu_performance_policy.py")


class CompletedChild(Protocol):
    returncode: int


class PolicySession(Protocol):
    @property
    def evidence(self) -> object: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> object: ...


type PolicyFactory = Callable[[], PolicySession]
type ChildRunner = Callable[[Sequence[str]], CompletedChild]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    if path.is_symlink():
        raise ValueError("receipt path must not be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_child(command: Sequence[str]) -> CompletedChild:
    return subprocess.run(command, check=False, start_new_session=True)


def run_command(
    *,
    command: Sequence[str],
    receipt_path: Path,
    policy_factory: PolicyFactory = cpu_performance_policy,
    child_runner: ChildRunner = _run_child,
) -> int:
    if not command or not command[0]:
        raise ValueError("a non-empty command is required after --")
    normalized_command = tuple(str(argument) for argument in command)
    started = time.time()
    failures: list[dict[str, object]] = []
    child_returncode: int | None = None
    policy_evidence: object | None = None
    session = policy_factory()
    try:
        with session:
            completed = child_runner(normalized_command)
            child_returncode = int(completed.returncode)
        policy_evidence = validate_restored_policy_evidence(session.evidence)
        if child_returncode != 0:
            failures.append(
                {
                    "kind": "child_nonzero_exit",
                    "returncode": child_returncode,
                }
            )
    except Exception as error:  # noqa: BLE001 - this is the receipt boundary.
        policy_evidence = session.evidence
        failures.append(
            {
                "kind": "wrapper_or_policy_failure",
                "error_type": type(error).__name__,
                "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            }
        )
    completed_at = time.time()
    receipt = {
        "format": RECEIPT_FORMAT,
        "accepted": not failures,
        "started_unix_seconds": started,
        "completed_unix_seconds": completed_at,
        "command": list(normalized_command),
        "command_sha256": _canonical_sha256(list(normalized_command)),
        "child_returncode": child_returncode,
        "cpu_performance_policy": {
            "helper_path": str(_POLICY_HELPER_PATH),
            "helper_sha256": _sha256_file(_POLICY_HELPER_PATH),
            "evidence": policy_evidence,
            "evidence_sha256": (
                _canonical_sha256(policy_evidence)
                if policy_evidence is not None
                else None
            ),
        },
        "failures": failures,
        "wrapper": {
            "path": str(_SCRIPT_PATH),
            "sha256": _sha256_file(_SCRIPT_PATH),
        },
    }
    _write_json_atomic(receipt_path, receipt)
    if failures:
        if child_returncode is not None and 1 <= child_returncode <= 125:
            return child_returncode
        return 2
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> tuple[Path, tuple[str, ...]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command_values = cast(list[str], arguments.command)
    if command_values[:1] == ["--"]:
        command_values = command_values[1:]
    command = tuple(command_values)
    if not command:
        parser.error("a command is required after --")
    return cast(Path, arguments.receipt), command


def main(argv: Sequence[str] | None = None) -> int:
    receipt_path, command = parse_arguments(argv)
    return run_command(command=command, receipt_path=receipt_path)


if __name__ == "__main__":
    raise SystemExit(main())
