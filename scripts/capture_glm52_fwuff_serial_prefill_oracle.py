#!/usr/bin/env python3
"""Capture deterministic serial-prefill oracles from a resident fwuff drafter.

The intended use is immediately before replacing the service's per-row decode
prefill with an ordinary batched EXTEND.  Each case opens a fresh sequence with
N committed rows, advances it by one more deterministic row, and finishes it.
The resulting candidate chains and logical state are a compact black-box
equivalence oracle for the batched implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

from scripts.glm52_fwuff_remote_draft import (  # noqa: E402
    DEFAULT_PORT,
    DEFAULT_REMOTE_ADDRESS,
    HIDDEN_ROW_BYTES,
    HIDDEN_SIZE,
    SCHEMA_VERSION,
    DraftRequest,
    _receive_response,
    send_frame,
)

ORACLE_SCHEMA: Final = "glm52-fwuff-serial-prefill-oracle-v1"
GENERATOR_SCHEMA: Final = "positive-finite-bf16-pattern-v1"
CASE_ROWS: Final = (2, 5, 64, 65)
DRAFT_DEPTH: Final = 4
ADVANCE_ROWS: Final = 1
BATCH_MINIMUM_ROWS: Final = 64
ExecutionPath = Literal[
    "serial_per_row_decode_prefill",
    "batched_extend_prefill",
    "boundary_canonicalized_prefill",
]


class OracleError(RuntimeError):
    """The resident service failed an oracle invariant."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_token_ids(
    case_rows: int,
    start_row: int,
    rows: int,
) -> tuple[int, ...]:
    return tuple(
        100 + ((case_rows * 37 + (start_row + row) * 101) % 10_000)
        for row in range(rows)
    )


def deterministic_hidden_payload(
    case_rows: int,
    start_row: int,
    rows: int,
) -> bytes:
    if sys.byteorder != "little":
        raise OracleError("the deterministic BF16 generator requires little endian")
    # 0x3f00..0x3f7f are finite positive BF16 values in [0.5, 0.99609375].
    words = array(
        "H",
        (
            0x3F00 + (case_rows * 29 + (start_row + row) * 17 + column * 13) % 0x80
            for row in range(rows)
            for column in range(HIDDEN_SIZE)
        ),
    )
    payload = words.tobytes()
    expected_bytes = rows * HIDDEN_ROW_BYTES
    if len(payload) != expected_bytes:
        raise OracleError(f"generated {len(payload)} bytes, expected {expected_bytes}")
    return payload


def exchange(
    connection: socket.socket,
    request: DraftRequest,
    payload: bytes = b"",
) -> dict[str, object]:
    send_frame(connection, request.to_header(), payload)
    response = _receive_response(connection)
    if response.get("status") != "ok":
        raise OracleError(f"fwuff rejected {request.action}: {response.get('error')}")
    if (
        response.get("request_id") != request.request_id
        or response.get("round_id") != request.round_id
    ):
        raise OracleError("response identity does not match the request")
    proposals = response.get("proposal_ids")
    expected_proposals = 0 if request.action in {"FINISH", "ABORT"} else DRAFT_DEPTH
    if not isinstance(proposals, list) or len(proposals) != expected_proposals:
        raise OracleError(
            f"{request.action} returned {proposals!r}, expected "
            f"{expected_proposals} proposals"
        )
    return response


def response_committed_total_milliseconds(response: dict[str, object]) -> float:
    raw_timings = response.get("timings")
    committed_forward_count = response.get("committed_forward_count")
    if (
        isinstance(raw_timings, list)
        and isinstance(committed_forward_count, int)
        and committed_forward_count > 0
        and len(raw_timings) >= committed_forward_count
    ):
        values: list[float] = []
        for timing in raw_timings[:committed_forward_count]:
            if not isinstance(timing, dict):
                raise OracleError("raw timing is not an object")
            total = timing.get("total_milliseconds")
            if not isinstance(total, int | float):
                raise OracleError("raw timing lacks total_milliseconds")
            values.append(float(total))
        return sum(values)

    summary = response.get("timing_summary")
    if not isinstance(summary, dict):
        raise OracleError("response has neither raw nor summarized timings")
    committed = summary.get("committed")
    if not isinstance(committed, dict):
        raise OracleError("timing summary lacks committed timings")
    total_summary = committed.get("total_milliseconds")
    iterations = committed.get("iterations")
    if not isinstance(total_summary, dict) or not isinstance(iterations, int):
        raise OracleError("committed timing summary is malformed")
    mean = total_summary.get("mean")
    if not isinstance(mean, int | float):
        raise OracleError("committed timing summary lacks a mean")
    return float(mean) * iterations


def policy_open_row_counts(
    response: dict[str, object],
    case_rows: int,
    execution_path: ExecutionPath,
) -> None:
    timings = response.get("timings")
    if case_rows < BATCH_MINIMUM_ROWS:
        expected = [1] * (case_rows + DRAFT_DEPTH - 1)
    elif execution_path == "boundary_canonicalized_prefill":
        expected = [case_rows - 1, *(1 for _ in range(DRAFT_DEPTH))]
    else:
        expected = [case_rows, *(1 for _ in range(DRAFT_DEPTH - 1))]
    if not isinstance(timings, list) or len(timings) != len(expected):
        raise OracleError(
            "OPEN timing count does not match the serial-small/batched-large policy"
        )
    observed: list[int] = []
    for timing in timings:
        if not isinstance(timing, dict):
            raise OracleError("OPEN timing is not an object")
        row_count = timing.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int):
            raise OracleError("OPEN timing lacks row_count")
        observed.append(row_count)
    if observed != expected:
        raise OracleError(f"OPEN timing row counts {observed} != {expected}")


def capture_case(
    *,
    address: str,
    port: int,
    case_rows: int,
    execution_path: ExecutionPath,
) -> dict[str, object]:
    request_id = f"serial-prefill-oracle-n{case_rows}"
    open_tokens = deterministic_token_ids(case_rows, 0, case_rows)
    open_payload = deterministic_hidden_payload(case_rows, 0, case_rows)
    advance_tokens = deterministic_token_ids(case_rows, case_rows, ADVANCE_ROWS)
    advance_payload = deterministic_hidden_payload(
        case_rows,
        case_rows,
        ADVANCE_ROWS,
    )

    connection = socket.create_connection((address, port), timeout=30.0)
    connection.settimeout(300.0)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    started = datetime.now(timezone.utc)
    try:
        open_response = exchange(
            connection,
            DraftRequest(
                request_id=request_id,
                round_id=0,
                action="OPEN",
                token_ids=open_tokens,
                hidden_rows=case_rows,
                payload_bytes=len(open_payload),
                draft_depth=DRAFT_DEPTH,
            ),
            open_payload,
        )
        advance_response = exchange(
            connection,
            DraftRequest(
                request_id=request_id,
                round_id=1,
                action="ADVANCE",
                token_ids=advance_tokens,
                hidden_rows=ADVANCE_ROWS,
                payload_bytes=len(advance_payload),
                draft_depth=DRAFT_DEPTH,
            ),
            advance_payload,
        )
        finish_response = exchange(
            connection,
            DraftRequest(
                request_id=request_id,
                round_id=2,
                action="FINISH",
                token_ids=(),
                hidden_rows=0,
                payload_bytes=0,
                draft_depth=DRAFT_DEPTH,
            ),
        )
    finally:
        connection.close()
    completed = datetime.now(timezone.utc)

    open_forward_count = case_rows
    if case_rows >= BATCH_MINIMUM_ROWS:
        if execution_path == "batched_extend_prefill":
            open_forward_count = 1
        elif execution_path == "boundary_canonicalized_prefill":
            open_forward_count = 2
    expected_counts = (
        (open_response, open_forward_count, DRAFT_DEPTH - 1, case_rows),
        (advance_response, ADVANCE_ROWS, DRAFT_DEPTH - 1, case_rows + ADVANCE_ROWS),
        (finish_response, 0, 0, case_rows + ADVANCE_ROWS),
    )
    for response, committed, tentative, sequence_length in expected_counts:
        if response.get("committed_forward_count") != committed:
            raise OracleError("committed forward count changed")
        if response.get("tentative_forward_count") != tentative:
            raise OracleError("tentative forward count changed")
        if response.get("sequence_length") != sequence_length:
            raise OracleError("logical sequence length changed")
    if execution_path != "serial_per_row_decode_prefill":
        policy_open_row_counts(open_response, case_rows, execution_path)

    inputs = {
        "open_token_ids": list(open_tokens),
        "open_token_ids_sha256": canonical_sha256(list(open_tokens)),
        "open_hidden_payload_sha256": hashlib.sha256(open_payload).hexdigest(),
        "advance_token_ids": list(advance_tokens),
        "advance_token_ids_sha256": canonical_sha256(list(advance_tokens)),
        "advance_hidden_payload_sha256": hashlib.sha256(advance_payload).hexdigest(),
        "combined_input_sha256": canonical_sha256(
            {
                "open_token_ids": list(open_tokens),
                "open_hidden_payload_sha256": hashlib.sha256(open_payload).hexdigest(),
                "advance_token_ids": list(advance_tokens),
                "advance_hidden_payload_sha256": hashlib.sha256(
                    advance_payload
                ).hexdigest(),
            }
        ),
    }
    return {
        "case_rows": case_rows,
        "advance_rows": ADVANCE_ROWS,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "inputs": inputs,
        "responses": {
            "open": open_response,
            "advance": advance_response,
            "finish": finish_response,
        },
        "committed_open_total_milliseconds": (
            response_committed_total_milliseconds(open_response)
        ),
    }


def load_serial_oracle(path: Path) -> tuple[dict[str, object], str]:
    try:
        oracle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OracleError(f"cannot load serial oracle {path}") from error
    if not isinstance(oracle, dict) or oracle.get("schema") != ORACLE_SCHEMA:
        raise OracleError("serial oracle schema is invalid")
    recorded_content_sha256 = oracle.get("receipt_content_sha256")
    unsigned_oracle = dict(oracle)
    unsigned_oracle.pop("receipt_content_sha256", None)
    if recorded_content_sha256 != canonical_sha256(unsigned_oracle):
        raise OracleError("serial oracle self-canonicalized hash is invalid")
    return cast(dict[str, object], oracle), file_sha256(path)


def equivalence_against_serial_oracle(
    *,
    serial_oracle: dict[str, object],
    batch_cases: list[dict[str, object]],
    execution_path: ExecutionPath,
) -> dict[str, object]:
    raw_serial_cases = serial_oracle.get("cases")
    if not isinstance(raw_serial_cases, list):
        raise OracleError("serial oracle cases are missing")
    serial_cases: dict[int, dict[str, object]] = {}
    for raw_case in raw_serial_cases:
        if not isinstance(raw_case, dict):
            raise OracleError("serial oracle case is malformed")
        case_rows = raw_case.get("case_rows")
        if isinstance(case_rows, bool) or not isinstance(case_rows, int):
            raise OracleError("serial oracle case row count is malformed")
        serial_cases[case_rows] = cast(dict[str, object], raw_case)

    comparisons: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for batch_case in batch_cases:
        case_rows = batch_case["case_rows"]
        if not isinstance(case_rows, int) or case_rows not in serial_cases:
            raise OracleError("batched case has no serial oracle")
        serial_case = serial_cases[case_rows]
        inputs_exact = batch_case["inputs"] == serial_case.get("inputs")
        if not inputs_exact:
            failures.append(
                {
                    "case_rows": case_rows,
                    "action": "inputs",
                    "detail": "deterministic inputs changed",
                }
            )
        batch_responses = batch_case.get("responses")
        serial_responses = serial_case.get("responses")
        if not isinstance(batch_responses, dict) or not isinstance(
            serial_responses,
            dict,
        ):
            raise OracleError("case responses are malformed")

        action_results: dict[str, object] = {}
        for action in ("open", "advance", "finish"):
            batch_response = batch_responses.get(action)
            serial_response = serial_responses.get(action)
            if not isinstance(batch_response, dict) or not isinstance(
                serial_response,
                dict,
            ):
                raise OracleError(f"N={case_rows} {action} response is malformed")
            compared_fields = (
                "proposal_ids",
                "sequence_length",
                "status",
                "tentative_forward_count",
            )
            mismatches = {
                field: {
                    "serial": serial_response.get(field),
                    "batched": batch_response.get(field),
                }
                for field in compared_fields
                if serial_response.get(field) != batch_response.get(field)
            }
            if mismatches:
                failures.append(
                    {
                        "case_rows": case_rows,
                        "action": action,
                        "mismatches": mismatches,
                    }
                )
            action_results[action] = {
                "result": "exact" if not mismatches else "mismatch",
                "compared_fields": list(compared_fields),
                "mismatches": mismatches,
            }

        serial_total = serial_case.get("committed_open_total_milliseconds")
        if serial_total is None:
            serial_open_response = serial_responses.get("open")
            if not isinstance(serial_open_response, dict):
                raise OracleError("serial OPEN response is malformed")
            serial_total = response_committed_total_milliseconds(
                cast(dict[str, object], serial_open_response)
            )
        batch_total = batch_case.get("committed_open_total_milliseconds")
        if not isinstance(serial_total, int | float) or not isinstance(
            batch_total,
            int | float,
        ):
            raise OracleError("case lacks committed OPEN timing")
        comparisons.append(
            {
                "case_rows": case_rows,
                "inputs": "exact" if inputs_exact else "mismatch",
                "responses": action_results,
                "serial_committed_open_total_milliseconds": float(serial_total),
                "batched_committed_open_total_milliseconds": float(batch_total),
                "committed_open_speedup": float(serial_total) / float(batch_total),
                "page_boundary_role": (
                    "ends_at_page_64_then_ADVANCE_crosses_to_65"
                    if case_rows == 64
                    else (
                        "OPEN_crosses_page_64_then_ADVANCE_reuses_post_boundary_state"
                        if case_rows == 65
                        else "small_shape"
                    )
                ),
            }
        )
    return {
        "result": "pass" if not failures else "fail",
        "requirements": {
            "deterministic_inputs": "exact",
            "open_candidate_chain": "exact",
            "post_open_advance_candidate_chain": "exact",
            "logical_sequence_lengths": "exact",
            "finish_cleanup": "exact",
            "timing_policy": (
                (
                    f"N<{BATCH_MINIMUM_ROWS} serial row_count=1 per forward; "
                    f"N>={BATCH_MINIMUM_ROWS} committed row_counts=N-1,1 "
                    "then tentative row_counts=1,1,1"
                )
                if execution_path == "boundary_canonicalized_prefill"
                else (
                    f"N<{BATCH_MINIMUM_ROWS} serial row_count=1 per forward; "
                    f"N>={BATCH_MINIMUM_ROWS} batched row_counts=N,1,1,1"
                )
            ),
            "microtiming": "reported_not_a_hard_gate",
        },
        "cases": comparisons,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=DEFAULT_REMOTE_ADDRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--service-pid", type=int, required=True)
    parser.add_argument("--server-source-sha256", required=True)
    parser.add_argument(
        "--execution-path",
        choices=(
            "serial_per_row_decode_prefill",
            "batched_extend_prefill",
            "boundary_canonicalized_prefill",
        ),
        required=True,
    )
    parser.add_argument(
        "--serial-oracle",
        type=Path,
        help="Required for non-serial capture; must be the immutable serial receipt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        raise OracleError("port is invalid")
    if args.service_pid <= 0:
        raise OracleError("service PID is invalid")
    if len(args.server_source_sha256) != 64:
        raise OracleError("server source SHA-256 is invalid")
    execution_path = cast(ExecutionPath, args.execution_path)
    if execution_path != "serial_per_row_decode_prefill" and args.serial_oracle is None:
        raise OracleError("--serial-oracle is required for batched capture")
    if (
        execution_path == "serial_per_row_decode_prefill"
        and args.serial_oracle is not None
    ):
        raise OracleError("--serial-oracle only applies to batched capture")

    harness_path = Path(__file__).resolve()
    cases = [
        capture_case(
            address=args.address,
            port=args.port,
            case_rows=rows,
            execution_path=execution_path,
        )
        for rows in CASE_ROWS
    ]
    receipt: dict[str, object] = {
        "schema": ORACLE_SCHEMA,
        "wire_schema_version": SCHEMA_VERSION,
        "status": "success",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": {"address": args.address, "port": args.port},
        "runtime": {
            "host": "fwuff",
            "service_pid": args.service_pid,
            "server_source_sha256": args.server_source_sha256,
            "draft_depth": DRAFT_DEPTH,
            "execution_path": execution_path,
        },
        "generator": {
            "schema": GENERATOR_SCHEMA,
            "case_rows": list(CASE_ROWS),
            "advance_rows": ADVANCE_ROWS,
            "batch_minimum_rows": BATCH_MINIMUM_ROWS,
            "hidden_size": HIDDEN_SIZE,
            "hidden_dtype": "BF16",
            "hidden_value_bits": "0x3f00 + ((N*29 + row*17 + column*13) % 0x80)",
            "token_formula": "100 + ((N*37 + row*101) % 10000)",
        },
        "harness": {
            "path": str(harness_path),
            "sha256": file_sha256(harness_path),
        },
        "cases": cases,
    }
    if args.serial_oracle is not None:
        serial_oracle, serial_oracle_file_sha256 = load_serial_oracle(
            args.serial_oracle
        )
        receipt["serial_oracle"] = {
            "path": str(args.serial_oracle),
            "file_sha256": serial_oracle_file_sha256,
            "receipt_content_sha256": serial_oracle["receipt_content_sha256"],
            "server_source_sha256": cast(dict[str, object], serial_oracle["runtime"])[
                "server_source_sha256"
            ],
        }
        equivalence = equivalence_against_serial_oracle(
            serial_oracle=serial_oracle,
            batch_cases=cases,
            execution_path=execution_path,
        )
        receipt["equivalence"] = equivalence
        receipt["status"] = "success" if equivalence["result"] == "pass" else "failed"
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(args.output, 0o444)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
