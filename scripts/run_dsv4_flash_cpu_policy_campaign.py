#!/usr/bin/env python3
"""Run one local DSV4 campaign stage under a reversible CPU policy.

The child campaign remains the authority for model, OSCAR, graph, topology,
coherency, and performance admission.  This wrapper owns exactly one
host-global CPU performance-policy transaction around that child.  It writes a
separate receipt only after the policy helper has restored and verified the
original governor/EPP state, including when the child fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol, Self, cast

try:
    from scripts.cpu_performance_policy import cpu_performance_policy
except ModuleNotFoundError:
    from cpu_performance_policy import cpu_performance_policy


RECEIPT_FORMAT: Final = "dsv4_cpu_performance_policy_campaign_v1"
_WRAPPER_PATH: Final = Path(__file__).resolve()
_CAMPAIGN_PATH: Final = _WRAPPER_PATH.with_name(
    "run_dsv4_flash_candidate_campaign.py"
)
_POLICY_HELPER_PATH: Final = _WRAPPER_PATH.with_name("cpu_performance_policy.py")
_DEFAULT_MANIFEST_PATH: Final = (
    _WRAPPER_PATH.parent / "data" / "dsv4_flash_candidate_campaign_v1.json"
)


class CpuPolicyCampaignError(RuntimeError):
    """Raised when the wrapper cannot prove the complete transaction."""


class CompletedChild(Protocol):
    returncode: int


class PolicySession(Protocol):
    @property
    def evidence(self) -> object: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> object: ...


class _ParsedArguments(Protocol):
    python: Path
    manifest: Path
    work_dir: Path
    receipt: Path
    execute_next: bool
    execute_candidate: str | None
    stage: str | None


type PolicyFactory = Callable[[], PolicySession]
type ChildRunner = Callable[[Sequence[str]], CompletedChild]


@dataclass(frozen=True)
class CampaignInvocation:
    python: Path
    manifest: Path
    work_directory: Path
    execute_next: bool
    candidate: str | None
    stage: str | None

    def command(self) -> tuple[str, ...]:
        if self.execute_next == (self.candidate is not None):
            raise CpuPolicyCampaignError(
                "select exactly one of execute-next or execute-candidate"
            )
        command = [
            str(self.python),
            str(_CAMPAIGN_PATH),
            "--manifest",
            str(self.manifest),
            "--work-dir",
            str(self.work_directory),
        ]
        if self.execute_next:
            if self.stage is not None:
                raise CpuPolicyCampaignError(
                    "stage is valid only with execute-candidate"
                )
            command.append("--execute-next")
        else:
            if self.stage not in {"screen", "confirm"}:
                raise CpuPolicyCampaignError(
                    "execute-candidate requires screen or confirm stage"
                )
            assert self.candidate is not None
            command.extend(("--execute-candidate", self.candidate, "--stage", self.stage))
        return tuple(command)


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


def _json_copy(value: object) -> object:
    return cast(
        object, json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CpuPolicyCampaignError(f"{label} must be an object")
    untyped = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in untyped):
        raise CpuPolicyCampaignError(f"{label} keys must be strings")
    return cast(Mapping[str, object], untyped)


def validate_restored_policy_evidence(value: object) -> dict[str, object]:
    """Validate apply and exact restoration evidence from the shared helper."""

    evidence = _mapping(value, "CPU performance-policy evidence")
    serialization = _mapping(
        evidence.get("serialization"), "CPU policy serialization evidence"
    )
    snapshots = _mapping(evidence.get("snapshots"), "CPU policy snapshots")
    failures = evidence.get("failures")
    policies = evidence.get("policies")
    if (
        evidence.get("lifecycle") != "restored"
        or evidence.get("application_verified") is not True
        or evidence.get("restoration_verified") is not True
        or not isinstance(failures, list)
        or failures
        or not isinstance(policies, list)
        or not policies
        or serialization.get("journal_published") is not False
        or serialization.get("journal_removed") is not True
        or serialization.get("lock_released") is not True
        or serialization.get("lock_acquired") is not False
        or not {
            "before",
            "active",
            "performance_after",
            "restored",
        }.issubset(snapshots)
    ):
        raise CpuPolicyCampaignError(
            "CPU performance-policy transaction did not prove apply and restore"
        )
    return dict(evidence)


def _default_child_runner(command: Sequence[str]) -> CompletedChild:
    return subprocess.run(tuple(command), check=False)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise CpuPolicyCampaignError(f"refusing to overwrite receipt {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def run_policy_campaign(
    invocation: CampaignInvocation,
    receipt_path: Path,
    *,
    policy_factory: PolicyFactory = cpu_performance_policy,
    child_runner: ChildRunner = _default_child_runner,
) -> dict[str, object]:
    """Execute one stage and publish post-restoration policy evidence."""

    if receipt_path.exists():
        raise CpuPolicyCampaignError(f"refusing to overwrite receipt {receipt_path}")
    command = invocation.command()
    if not invocation.python.is_file() or not os.access(invocation.python, os.X_OK):
        raise CpuPolicyCampaignError(
            f"Python must resolve to an executable file: {invocation.python}"
        )
    if not invocation.python.resolve(strict=True).is_file():
        raise CpuPolicyCampaignError(
            f"Python target is not a regular file: {invocation.python}"
        )
    for required_path in (invocation.manifest, _CAMPAIGN_PATH, _POLICY_HELPER_PATH):
        if not required_path.is_file() or required_path.is_symlink():
            raise CpuPolicyCampaignError(
                f"required path must be a regular non-symlink file: {required_path}"
            )

    started_unix_seconds = time.time()
    session: PolicySession | None = None
    completed: CompletedChild | None = None
    failure: BaseException | None = None
    try:
        session = policy_factory()
        with session:
            active_evidence = _mapping(
                session.evidence, "active CPU performance-policy evidence"
            )
            if (
                active_evidence.get("lifecycle") != "active"
                or active_evidence.get("application_verified") is not True
            ):
                raise CpuPolicyCampaignError(
                    "CPU performance policy was not verified before launch"
                )
            completed = child_runner(command)
            if completed.returncode != 0:
                raise CpuPolicyCampaignError(
                    f"candidate campaign exited with status {completed.returncode}"
                )
    except (CpuPolicyCampaignError, OSError, RuntimeError, TypeError, ValueError) as error:
        failure = error

    restored_evidence: dict[str, object] | None = None
    restoration_error: BaseException | None = None
    if session is not None:
        try:
            restored_evidence = validate_restored_policy_evidence(session.evidence)
        except (CpuPolicyCampaignError, TypeError, ValueError) as error:
            restoration_error = error
    else:
        restoration_error = CpuPolicyCampaignError(
            "CPU performance-policy session was not created"
        )

    failures = tuple(
        error for error in (failure, restoration_error) if error is not None
    )
    evidence_for_receipt = None if session is None else _json_copy(session.evidence)
    payload: dict[str, object] = {
        "format": RECEIPT_FORMAT,
        "accepted": not failures,
        "started_unix_seconds": started_unix_seconds,
        "completed_unix_seconds": time.time(),
        "transaction_scope": "one_outer_local_campaign_stage",
        "nested_or_rank_transactions": False,
        "command": list(command),
        "command_sha256": _canonical_sha256(list(command)),
        "child_returncode": None if completed is None else completed.returncode,
        "wrapper": {
            "path": str(_WRAPPER_PATH),
            "sha256": _sha256_file(_WRAPPER_PATH),
        },
        "campaign": {
            "path": str(_CAMPAIGN_PATH),
            "sha256": _sha256_file(_CAMPAIGN_PATH),
        },
        "manifest": {
            "path": str(invocation.manifest),
            "sha256": _sha256_file(invocation.manifest),
        },
        "cpu_performance_policy": {
            "helper_path": str(_POLICY_HELPER_PATH),
            "helper_sha256": _sha256_file(_POLICY_HELPER_PATH),
            "evidence": evidence_for_receipt,
            "evidence_sha256": (
                None
                if restored_evidence is None
                else _canonical_sha256(restored_evidence)
            ),
        },
        "failures": [
            {
                "error_type": type(error).__name__,
                "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            }
            for error in failures
        ],
    }
    _write_json_atomic(receipt_path, payload)
    if failures:
        raise CpuPolicyCampaignError(
            "CPU-policy campaign failed; restoration evidence is in "
            f"{receipt_path}"
        ) from failures[0]
    return payload


def _arguments() -> _ParsedArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/dsv4-candidate-campaign")
    )
    parser.add_argument("--receipt", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--execute-next", action="store_true")
    actions.add_argument("--execute-candidate")
    parser.add_argument("--stage", choices=("screen", "confirm"))
    return cast(_ParsedArguments, cast(object, parser.parse_args()))


def main() -> int:
    arguments = _arguments()
    invocation = CampaignInvocation(
        # Preserve a virtual environment's interpreter path. Resolving its
        # ``bin/python`` symlink would bypass pyvenv.cfg and run the host
        # interpreter instead.
        python=arguments.python.absolute(),
        manifest=arguments.manifest.resolve(),
        work_directory=arguments.work_dir.resolve(),
        execute_next=arguments.execute_next,
        candidate=arguments.execute_candidate,
        stage=arguments.stage,
    )
    try:
        payload = run_policy_campaign(
            invocation=invocation,
            receipt_path=arguments.receipt.resolve(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (CpuPolicyCampaignError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "accepted": False,
                    "error_type": type(error).__name__,
                    "error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
