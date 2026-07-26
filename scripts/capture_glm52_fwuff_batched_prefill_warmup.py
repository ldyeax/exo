#!/usr/bin/env python3
"""Run exactly one deterministic 128+112 fwuff prefill shape warmup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))

from scripts.capture_glm52_fwuff_serial_prefill_oracle import (  # noqa: E402
    BATCH_MINIMUM_ROWS,
    DRAFT_DEPTH,
    canonical_sha256,
    deterministic_hidden_payload,
    deterministic_token_ids,
    exchange,
    file_sha256,
)
from scripts.glm52_fwuff_remote_draft import (  # noqa: E402
    DEFAULT_PORT,
    DEFAULT_REMOTE_ADDRESS,
    SCHEMA_VERSION,
    DraftRequest,
)

OPEN_ROWS = 128
ADVANCE_ROWS = 112
TOTAL_ROWS = OPEN_ROWS + ADVANCE_ROWS
RECEIPT_SCHEMA = "glm52-fwuff-boundary-canonicalized-prefill-warmup-v1"


class WarmupError(RuntimeError):
    """The single shape warmup violated its lifecycle or timing contract."""


def compact_response(
    response: dict[str, object],
    *,
    expected_rows: int,
    expected_sequence_length: int,
) -> dict[str, object]:
    if (
        response.get("sequence_length") != expected_sequence_length
        or response.get("committed_forward_count") != 2
        or response.get("tentative_forward_count") != DRAFT_DEPTH - 1
    ):
        raise WarmupError("canonicalized response state/count contract changed")
    raw_timings = response.get("timings")
    if not isinstance(raw_timings, list) or len(raw_timings) != DRAFT_DEPTH + 1:
        raise WarmupError("canonicalized response timing count changed")
    expected_row_counts = [
        expected_rows - 1,
        *(1 for _ in range(DRAFT_DEPTH)),
    ]
    timings: list[dict[str, object]] = []
    for index, raw_timing in enumerate(raw_timings):
        if not isinstance(raw_timing, dict):
            raise WarmupError("canonicalized response timing is not an object")
        if raw_timing.get("row_count") != expected_row_counts[index]:
            raise WarmupError("canonicalized response timing row coverage changed")
        timings.append(
            {
                name: raw_timing.get(name)
                for name in (
                    "sequence_length",
                    "row_count",
                    "h2d_milliseconds",
                    "allocator_wall_milliseconds",
                    "allocator_cuda_milliseconds",
                    "metadata_wall_milliseconds",
                    "metadata_cuda_milliseconds",
                    "model_milliseconds",
                    "proposal_milliseconds",
                    "total_milliseconds",
                )
            }
        )
    return {
        "proposal_ids": response["proposal_ids"],
        "sequence_length": response["sequence_length"],
        "committed_forward_count": response["committed_forward_count"],
        "tentative_forward_count": response["tentative_forward_count"],
        "timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=DEFAULT_REMOTE_ADDRESS)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--service-pid", type=int, required=True)
    parser.add_argument("--server-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        raise WarmupError("port is invalid")
    if args.service_pid <= 0:
        raise WarmupError("service PID is invalid")
    if len(args.server_source_sha256) != 64:
        raise WarmupError("server source SHA-256 is invalid")

    open_tokens = deterministic_token_ids(TOTAL_ROWS, 0, OPEN_ROWS)
    open_payload = deterministic_hidden_payload(TOTAL_ROWS, 0, OPEN_ROWS)
    advance_tokens = deterministic_token_ids(TOTAL_ROWS, OPEN_ROWS, ADVANCE_ROWS)
    advance_payload = deterministic_hidden_payload(
        TOTAL_ROWS,
        OPEN_ROWS,
        ADVANCE_ROWS,
    )
    request_id = "batched-prefill-shape-warmup-128-112"
    connection = socket.create_connection((args.address, args.port), timeout=30.0)
    connection.settimeout(300.0)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    try:
        open_response = exchange(
            connection,
            DraftRequest(
                request_id=request_id,
                round_id=0,
                action="OPEN",
                token_ids=open_tokens,
                hidden_rows=OPEN_ROWS,
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
    elapsed_seconds = time.monotonic() - started_monotonic
    if (
        finish_response.get("sequence_length") != TOTAL_ROWS
        or finish_response.get("committed_forward_count") != 0
        or finish_response.get("tentative_forward_count") != 0
        or finish_response.get("proposal_ids") != []
    ):
        raise WarmupError("FINISH did not preserve the clean final sequence state")

    harness_path = Path(__file__).resolve()
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "wire_schema_version": SCHEMA_VERSION,
        "status": "success",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "endpoint": {"address": args.address, "port": args.port},
        "runtime": {
            "host": "fwuff",
            "service_pid": args.service_pid,
            "server_source_sha256": args.server_source_sha256,
            "draft_depth": DRAFT_DEPTH,
            "batch_minimum_rows": BATCH_MINIMUM_ROWS,
            "committed_transition": "EXTEND_N_minus_1_then_final_DECODE",
        },
        "harness": {
            "path": str(harness_path),
            "sha256": file_sha256(harness_path),
        },
        "input": {
            "generator_case_rows": TOTAL_ROWS,
            "open_rows": OPEN_ROWS,
            "advance_rows": ADVANCE_ROWS,
            "open_token_ids_sha256": canonical_sha256(list(open_tokens)),
            "open_hidden_payload_sha256": hashlib.sha256(open_payload).hexdigest(),
            "advance_token_ids_sha256": canonical_sha256(list(advance_tokens)),
            "advance_hidden_payload_sha256": hashlib.sha256(
                advance_payload
            ).hexdigest(),
        },
        "responses": {
            "open": compact_response(
                cast(dict[str, object], open_response),
                expected_rows=OPEN_ROWS,
                expected_sequence_length=OPEN_ROWS,
            ),
            "advance": compact_response(
                cast(dict[str, object], advance_response),
                expected_rows=ADVANCE_ROWS,
                expected_sequence_length=TOTAL_ROWS,
            ),
            "finish": {
                "proposal_ids": finish_response["proposal_ids"],
                "sequence_length": finish_response["sequence_length"],
                "committed_forward_count": finish_response["committed_forward_count"],
                "tentative_forward_count": finish_response["tentative_forward_count"],
            },
        },
        "lifecycle": {
            "exactly_one_open": True,
            "exactly_one_advance": True,
            "finish_observed": True,
            "final_sequence_length": TOTAL_ROWS,
        },
    }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(args.output, 0o444)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
