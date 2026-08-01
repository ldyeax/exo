#!/usr/bin/env python3
"""Materialize one pinned peer snapshot for direct SGLang/KTransformers launch."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import cast

from exo.download.peer_artifact_downloader import (
    materialize_peer_artifact_snapshot,
)
from exo.download.peer_artifact_http import (
    load_peer_artifact_deployment_config,
)
from exo.shared.models.model_cards import HuggingFaceRevision
from exo.shared.types.common import ModelId, NodeId


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--peer-node-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, required=True)
    return parser


async def _run(arguments: argparse.Namespace) -> Path:
    config_path = cast(Path, arguments.config)
    config = load_peer_artifact_deployment_config(config_path)
    if config is None:
        raise ValueError("peer artifact configuration is required")
    destination = cast(Path, arguments.destination)
    snapshot = await materialize_peer_artifact_snapshot(
        config,
        peer_node_id=NodeId(cast(str, arguments.peer_node_id)),
        model_id=ModelId(cast(str, arguments.model_id)),
        revision=cast(HuggingFaceRevision, arguments.revision),
        destination=destination,
    )
    print(
        f"{destination} snapshot_id={snapshot.snapshot_id} files={len(snapshot.files)}"
    )
    return destination


def main() -> int:
    try:
        asyncio.run(_run(_parser().parse_args()))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Peer snapshot materialization failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
