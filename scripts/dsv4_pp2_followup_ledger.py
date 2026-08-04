#!/usr/bin/env python3
"""Authorize exactly two receipt-bound DeepSeek-V4 PP2 model launches.

The first launch is the direct EP-winner transfer.  The second is one PP-only
optimization informed by the first benchmark receipt.  Authorization happens
in the launch-server shim immediately before the model process is exec'd, so
preparation failures do not consume one of the two allowed model launches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Protocol, cast

if __package__:
    from scripts.transfer_dsv4_ep2_plan_to_pp2 import (
        OSCAR_SPLIT_HISTORY_EXECUTION,
        OSCAR_SPLIT_HISTORY_SPLIT_MAP,
        OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES,
        QUALIFIED_LUT_HASH,
        QUALIFIED_N_BLOCK,
        QUALIFIED_NATIVE_ARTIFACT,
        QUALIFIED_NATIVE_ARTIFACT_SHA256,
        EPConfirmation,
        PlanTransferError,
        load_ep_confirmation,
        sha256_file,
    )
else:
    from transfer_dsv4_ep2_plan_to_pp2 import (
        OSCAR_SPLIT_HISTORY_EXECUTION,
        OSCAR_SPLIT_HISTORY_SPLIT_MAP,
        OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES,
        QUALIFIED_LUT_HASH,
        QUALIFIED_N_BLOCK,
        QUALIFIED_NATIVE_ARTIFACT,
        QUALIFIED_NATIVE_ARTIFACT_SHA256,
        EPConfirmation,
        PlanTransferError,
        load_ep_confirmation,
        sha256_file,
    )

LEDGER_FORMAT: Final = "dsv4_pp2_two_launch_ledger_v1"
AUTHORIZATION_FORMAT: Final = "dsv4_pp2_model_launch_authorization_v1"
MAXIMUM_MODEL_LAUNCHES: Final = 2
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}", re.ASCII)


class LaunchLedgerError(RuntimeError):
    """The PP2 launch request failed its immutable two-run contract."""


class _ParsedArguments(Protocol):
    ledger: Path
    output: Path
    ep_confirmation_receipt: Path
    ep_coherency_receipt: Path
    source_ep2_plan: Path
    transferred_plan: Path
    run_role: str
    pipeline_layer_partition: str
    pp_async_batch_depth: int
    chunked_prefill_size: int
    first_benchmark_receipt: Path | None
    native_artifact: Path
    native_artifact_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _resolve_regular_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise LaunchLedgerError(f"{label} path must be absolute: {path}")
    if path.is_symlink():
        raise LaunchLedgerError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LaunchLedgerError(f"cannot resolve {label} {path}: {error}") from error
    if not resolved.is_file():
        raise LaunchLedgerError(f"{label} is not a regular file: {resolved}")
    return resolved


def _prepare_output_parent(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise LaunchLedgerError(f"{label} path must be absolute: {path}")
    if path.is_symlink():
        raise LaunchLedgerError(f"{label} must not be a symlink: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise LaunchLedgerError(f"cannot prepare {label} parent: {error}") from error
    if not parent.is_dir():
        raise LaunchLedgerError(f"{label} parent is not a directory: {parent}")
    return parent / path.name


def _read_json_file(path: Path, *, label: str) -> tuple[Path, str, dict[str, object]]:
    resolved = _resolve_regular_file(path, label=label)
    digest = sha256_file(resolved)
    try:
        raw = cast(object, json.loads(resolved.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchLedgerError(f"{label} is not readable JSON") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise LaunchLedgerError(f"{label} must contain a JSON object")
    if sha256_file(resolved) != digest:
        raise LaunchLedgerError(f"{label} changed while validating")
    return resolved, digest, cast(dict[str, object], raw)


def _atomic_write_json(path: Path, document: dict[str, object]) -> str:
    encoded = _canonical_json_bytes(document) + b"\n"
    digest = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if path.exists() or path.is_symlink():
            raise LaunchLedgerError(f"refusing to overwrite {path}")
        os.rename(temporary, path)
        published = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise LaunchLedgerError(f"cannot publish {path}: {error}") from error
    finally:
        if not published and temporary.exists():
            temporary.unlink()
    return digest


def _replace_json(path: Path, document: dict[str, object]) -> None:
    encoded = _canonical_json_bytes(document) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if path.is_symlink():
            raise LaunchLedgerError(f"launch ledger must not be a symlink: {path}")
        os.replace(temporary, path)
        replaced = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise LaunchLedgerError(f"cannot publish launch ledger {path}: {error}") from error
    finally:
        if not replaced and temporary.exists():
            temporary.unlink()


@contextmanager
def _ledger_lock(ledger_path: Path) -> Iterator[None]:
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        raise LaunchLedgerError(f"cannot open launch-ledger lock: {error}") from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except OSError as error:
        raise LaunchLedgerError(f"cannot lock launch ledger: {error}") from error
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _ep_binding(confirmation: EPConfirmation) -> dict[str, object]:
    if (
        confirmation.coherency_receipt_path is None
        or confirmation.coherency_receipt_sha256 is None
    ):
        raise LaunchLedgerError(
            "PP2 launch requires a direct EP hotspot receipt bound to coherency"
        )
    return {
        "receipt_path": str(confirmation.receipt_path),
        "receipt_sha256": confirmation.receipt_sha256,
        "coherency_receipt_path": str(confirmation.coherency_receipt_path),
        "coherency_receipt_sha256": confirmation.coherency_receipt_sha256,
        "source_ep2_plan_path": str(confirmation.source_plan_path),
        "source_ep2_plan_sha256": confirmation.source_plan_sha256,
        "source_placement_semantics_sha256": (
            confirmation.source_placement_semantics_sha256
        ),
    }


def _new_ledger(confirmation: EPConfirmation) -> dict[str, object]:
    return {
        "format": LEDGER_FORMAT,
        "maximum_model_launches": MAXIMUM_MODEL_LAUNCHES,
        "ep_winner": _ep_binding(confirmation),
        "launches": [],
    }


def _load_ledger(path: Path, confirmation: EPConfirmation) -> dict[str, object]:
    if not path.exists():
        return _new_ledger(confirmation)
    _, _, ledger = _read_json_file(path, label="PP2 launch ledger")
    launches = ledger.get("launches")
    if (
        ledger.get("format") != LEDGER_FORMAT
        or ledger.get("maximum_model_launches") != MAXIMUM_MODEL_LAUNCHES
        or ledger.get("ep_winner") != _ep_binding(confirmation)
        or not isinstance(launches, list)
        or len(launches) > MAXIMUM_MODEL_LAUNCHES
        or any(not isinstance(launch, dict) for launch in launches)
    ):
        raise LaunchLedgerError("existing PP2 launch ledger violates its exact schema")
    expected_roles = ["transfer", "optimized"][: len(launches)]
    if [launch.get("run_role") for launch in launches] != expected_roles:
        raise LaunchLedgerError("existing PP2 launch ledger has an invalid run order")
    if [launch.get("ordinal") for launch in launches] != list(
        range(1, len(launches) + 1)
    ):
        raise LaunchLedgerError("existing PP2 launch ledger has invalid ordinals")
    return ledger


def _validate_first_benchmark(
    path: Path,
    *,
    first_launch: dict[str, object],
) -> dict[str, object]:
    resolved, digest, receipt = _read_json_file(
        path, label="first PP2 benchmark receipt"
    )
    configuration = receipt.get("configuration")
    if not isinstance(configuration, dict):
        raise LaunchLedgerError("first PP2 benchmark configuration is malformed")
    provenance = configuration.get("benchmark_provenance")
    launch_authorization = configuration.get("launch_authorization")
    server_contract = configuration.get("server_contract")
    native_generate = receipt.get("native_generate")
    openai_tool_calls = receipt.get("openai_tool_calls")
    nvlink_traffic = receipt.get("nvlink_traffic")
    if not isinstance(provenance, dict):
        raise LaunchLedgerError("first PP2 benchmark provenance is missing")
    if not isinstance(launch_authorization, dict):
        raise LaunchLedgerError(
            "first PP2 benchmark launch authorization is missing"
        )
    if (
        not isinstance(server_contract, dict)
        or not isinstance(native_generate, dict)
        or not isinstance(openai_tool_calls, dict)
        or not isinstance(nvlink_traffic, dict)
    ):
        raise LaunchLedgerError("first PP2 benchmark evidence is incomplete")
    counter_deltas = nvlink_traffic.get("counter_deltas")
    complete_concurrency_evidence = (
        native_generate.get("request_count") == 2
        and native_generate.get("all_streams_complete") is True
        and native_generate.get("all_decode_timing_valid") is True
        and native_generate.get("all_naturally_terminated") is True
        and native_generate.get("concurrent_start_confirmed") is True
        and native_generate.get("concurrent_overlap_observed") is True
        and openai_tool_calls.get("request_count") == 2
        and openai_tool_calls.get("all_streams_complete") is True
        and openai_tool_calls.get("concurrent_start_confirmed") is True
        and openai_tool_calls.get("concurrent_overlap_observed") is True
        and isinstance(counter_deltas, dict)
        and len(counter_deltas) == 16
        and all(type(value) is int and value > 0 for value in counter_deltas.values())
    )
    complete_server_contract = (
        server_contract.get("tp_size") == 1
        and server_contract.get("pp_size") == 2
        and server_contract.get("ep_size") == 1
        and server_contract.get("context_length") == 524_288
        and server_contract.get("max_total_tokens") == 524_288
        and server_contract.get("dsv4_oscar_int2_kv_storage") is True
        and server_contract.get("dsv4_oscar_int2_split_history") is True
        and server_contract.get("cuda_graph_backend_decode") == "full"
        and server_contract.get("cuda_graph_bs_decode") == [1, 2]
    )
    benchmark_ok = receipt.get("ok")
    performance_claim_eligible = receipt.get("performance_claim_eligible")
    first_configuration = first_launch.get("configuration")
    if not isinstance(first_configuration, dict):
        raise LaunchLedgerError("first PP2 launch configuration is malformed")
    if (
        receipt.get("schema_version") != 4
        or type(benchmark_ok) is not bool
        or type(performance_claim_eligible) is not bool
        or benchmark_ok is not performance_claim_eligible
        or not complete_concurrency_evidence
        or not complete_server_contract
        or provenance.get("run_label") != "pp2-transfer"
        or provenance.get("expert_plan_path")
        != first_configuration.get("transferred_plan_path")
        or provenance.get("expert_plan_sha256")
        != first_configuration.get("transferred_plan_sha256")
        or launch_authorization.get("authorization_receipt_path")
        != first_launch.get("authorization_receipt_path")
        or launch_authorization.get("authorization_receipt_sha256")
        != first_launch.get("authorization_receipt_sha256")
        or launch_authorization.get("ordinal") != 1
        or launch_authorization.get("run_role") != "transfer"
        or configuration.get("expected_chunked_prefill_size") != 1024
        or configuration.get("expected_pp_async_batch_depth") != 0
        or configuration.get("expected_cpuinfer_threads") != 56
    ):
        raise LaunchLedgerError(
            "first PP2 benchmark is not a complete receipt for the transfer launch"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "ok": benchmark_ok,
        "performance_claim_eligible": performance_claim_eligible,
    }


def authorize_launch(
    *,
    ledger_path: Path,
    output_path: Path,
    ep_confirmation_receipt: Path,
    ep_coherency_receipt: Path,
    source_ep2_plan: Path,
    transferred_plan: Path,
    run_role: str,
    pipeline_layer_partition: str,
    pp_async_batch_depth: int,
    chunked_prefill_size: int,
    first_benchmark_receipt: Path | None,
    native_artifact: Path,
    native_artifact_sha256: str,
) -> Path:
    if run_role not in ("transfer", "optimized"):
        raise LaunchLedgerError("run role must be transfer or optimized")
    if pipeline_layer_partition not in ("21,22", "22,21"):
        raise LaunchLedgerError("pipeline layer partition must be 21,22 or 22,21")
    if pp_async_batch_depth not in (0, 1):
        raise LaunchLedgerError("PP async batch depth must be zero or one")
    if chunked_prefill_size != 1024:
        raise LaunchLedgerError("both receipt-bound PP2 launches must use chunk size 1024")
    if native_artifact != QUALIFIED_NATIVE_ARTIFACT:
        raise LaunchLedgerError("PP2 launch does not name the qualified native artifact")
    if native_artifact_sha256 != QUALIFIED_NATIVE_ARTIFACT_SHA256:
        raise LaunchLedgerError("PP2 launch does not use the qualified native digest")

    try:
        confirmation = load_ep_confirmation(
            ep_confirmation_receipt,
            coherency_receipt=ep_coherency_receipt,
            expected_source_plan=source_ep2_plan,
        )
    except PlanTransferError as error:
        raise LaunchLedgerError(str(error)) from error
    resolved_plan = _resolve_regular_file(transferred_plan, label="transferred PP2 plan")
    transferred_plan_sha256 = sha256_file(resolved_plan)
    prepared_ledger = _prepare_output_parent(ledger_path, label="PP2 launch ledger")
    prepared_output = _prepare_output_parent(
        output_path, label="PP2 launch authorization receipt"
    )
    if prepared_output == prepared_ledger:
        raise LaunchLedgerError("authorization receipt must not overwrite the ledger")

    with _ledger_lock(prepared_ledger):
        ledger = _load_ledger(prepared_ledger, confirmation)
        launches = cast(list[dict[str, object]], ledger["launches"])
        ordinal = len(launches) + 1
        if ordinal > MAXIMUM_MODEL_LAUNCHES:
            raise LaunchLedgerError("the two permitted PP2 model launches are exhausted")
        expected_role = "transfer" if ordinal == 1 else "optimized"
        if run_role != expected_role:
            raise LaunchLedgerError(
                f"PP2 launch {ordinal} must use run role {expected_role}"
            )

        first_benchmark: dict[str, object] | None = None
        if ordinal == 1:
            if first_benchmark_receipt is not None:
                raise LaunchLedgerError(
                    "the transfer launch cannot consume a prior benchmark receipt"
                )
            if pipeline_layer_partition != "21,22" or pp_async_batch_depth != 0:
                raise LaunchLedgerError(
                    "the transfer launch must use partition 21,22 and async depth 0"
                )
        else:
            if first_benchmark_receipt is None:
                raise LaunchLedgerError(
                    "the optimized launch requires the first benchmark receipt"
                )
            first_benchmark = _validate_first_benchmark(
                first_benchmark_receipt,
                first_launch=launches[0],
            )
            first_configuration = cast(dict[str, object], launches[0]["configuration"])
            changed_pp_knobs = sum(
                (
                    pipeline_layer_partition
                    != first_configuration["pipeline_layer_partition"],
                    pp_async_batch_depth
                    != first_configuration["pp_async_batch_depth"],
                )
            )
            if changed_pp_knobs != 1:
                raise LaunchLedgerError(
                    "the optimized launch must change exactly one PP-only knob"
                )

        configuration: dict[str, object] = {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 2,
            "expert_parallel_size": 1,
            "pipeline_layer_partition": pipeline_layer_partition,
            "pp_async_batch_depth": pp_async_batch_depth,
            "pp_max_micro_batch_size": 1,
            "max_running_requests": 2,
            "chunked_prefill_size": chunked_prefill_size,
            "context_length": 524_288,
            "max_total_tokens": 524_288,
            "decode_cuda_graph_backend": "full",
            "decode_cuda_graph_batch_sizes": [1, 2],
            "prefill_cuda_graph_backend": "disabled",
            "kv_cache_public_carrier": "fp8_e4m3",
            "physical_kv_cache_storage": "oscar-int2-asymmetric",
            "oscar_split_history": True,
            "oscar_split_history_execution": OSCAR_SPLIT_HISTORY_EXECUTION,
            "oscar_split_history_split_map": OSCAR_SPLIT_HISTORY_SPLIT_MAP,
            "oscar_split_history_workspace_bytes_per_worker": (
                OSCAR_SPLIT_HISTORY_WORKSPACE_BYTES
            ),
            "oscar_split_history_worker_identities": [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
            ],
            "transferred_plan_path": str(resolved_plan),
            "transferred_plan_sha256": transferred_plan_sha256,
            "native_artifact_path": str(native_artifact),
            "native_artifact_sha256": native_artifact_sha256,
            "cpuinfer_threads": 56,
            "worker_spin_us": 1000,
            "task_queue_pin_first_core": True,
            "single_numa_inline_dispatch": True,
            "scale_fold_mode": "lut-v1",
            "scale_fold_n_block": QUALIFIED_N_BLOCK,
            "scale_fold_lut_hash": QUALIFIED_LUT_HASH,
        }
        authorization: dict[str, object] = {
            "format": AUTHORIZATION_FORMAT,
            "ordinal": ordinal,
            "run_role": run_role,
            "authorized_unix_seconds": time.time(),
            "ep_winner": _ep_binding(confirmation),
            "configuration": configuration,
            "configuration_sha256": _canonical_sha256(configuration),
            "first_benchmark_receipt": first_benchmark,
        }
        authorization_sha256 = _atomic_write_json(prepared_output, authorization)
        launches.append(
            {
                **authorization,
                "authorization_receipt_path": str(prepared_output),
                "authorization_receipt_sha256": authorization_sha256,
            }
        )
        _replace_json(prepared_ledger, ledger)
    return prepared_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-confirmation-receipt", type=Path, required=True)
    parser.add_argument("--ep-coherency-receipt", type=Path, required=True)
    parser.add_argument("--source-ep2-plan", type=Path, required=True)
    parser.add_argument("--transferred-plan", type=Path, required=True)
    parser.add_argument("--run-role", choices=("transfer", "optimized"), required=True)
    parser.add_argument(
        "--pipeline-layer-partition", choices=("21,22", "22,21"), required=True
    )
    parser.add_argument("--pp-async-batch-depth", type=int, choices=(0, 1), required=True)
    parser.add_argument("--chunked-prefill-size", type=int, choices=(1024,), required=True)
    parser.add_argument("--first-benchmark-receipt", type=Path)
    parser.add_argument(
        "--native-artifact", type=Path, default=QUALIFIED_NATIVE_ARTIFACT
    )
    parser.add_argument(
        "--native-artifact-sha256",
        default=QUALIFIED_NATIVE_ARTIFACT_SHA256,
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = cast(_ParsedArguments, cast(object, build_parser().parse_args(arguments)))
    try:
        result = authorize_launch(
            ledger_path=parsed.ledger,
            output_path=parsed.output,
            ep_confirmation_receipt=parsed.ep_confirmation_receipt,
            ep_coherency_receipt=parsed.ep_coherency_receipt,
            source_ep2_plan=parsed.source_ep2_plan,
            transferred_plan=parsed.transferred_plan,
            run_role=parsed.run_role,
            pipeline_layer_partition=parsed.pipeline_layer_partition,
            pp_async_batch_depth=parsed.pp_async_batch_depth,
            chunked_prefill_size=parsed.chunked_prefill_size,
            first_benchmark_receipt=parsed.first_benchmark_receipt,
            native_artifact=parsed.native_artifact,
            native_artifact_sha256=parsed.native_artifact_sha256,
        )
    except LaunchLedgerError as error:
        print(f"dsv4_pp2_followup_ledger: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
