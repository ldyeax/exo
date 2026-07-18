#!/usr/bin/env python3
"""Download a model to every node in an exo cluster, bypassing placement.

Usage:
    uv run python scripts/download_model_to_cluster.py \
        mlx-community/SmolLM2-135M-Instruct-8bit \
        --revision 0f0d9b8218915bc34d401e1a340b8c049d300d5e \
        --host dwagon

This resolves an exact or curated ModelCard locally, constructs a full-model
PipelineShardMetadata (world_size=1, one shard covering every layer), and POSTs
/download/start to the target exo API for each node currently in the topology.
It then polls /state/downloads until every node reports DownloadCompleted for
the same model revision.

No placement is required. Works with a cluster of any size, including 1.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
from loguru import logger
from pydantic import TypeAdapter

from exo.api.types.api import ModelList, ModelListModel
from exo.shared.models.model_cards import (
    HuggingFaceRevision,
    ModelCard,
    ModelId,
    validate_hugging_face_revision,
)
from exo.shared.topology import TopologySnapshot
from exo.shared.types.common import NodeId
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.shards import PipelineShardMetadata

_DOWNLOADS_STATE_ADAPTER = TypeAdapter(dict[NodeId, list[DownloadProgress]])


async def fetch_topology_nodes(client: httpx.AsyncClient, base: str) -> list[NodeId]:
    response = await client.get(f"{base}/state/topology")
    response.raise_for_status()
    topology = TopologySnapshot.model_validate(response.json())
    return list(topology.nodes)


def build_shard_payload(card: ModelCard) -> dict[str, object]:
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=card.n_layers,
        n_layers=card.n_layers,
    )
    return shard.model_dump(mode="json", by_alias=True)


async def ensure_model_card_registered(
    client: httpx.AsyncClient, base: str, card: ModelCard
) -> None:
    response = await client.get(f"{base}/models")
    response.raise_for_status()
    models = ModelList.model_validate(response.json())
    for model in models.data:
        registered_model_id = model.hugging_face_id or model.id
        if registered_model_id == card.model_id and model.revision == card.revision:
            logger.info(
                f"Model already registered on cluster: {card.model_id}@{card.revision}"
            )
            return

    logger.info(
        f"Registering model on cluster via /models/add: {card.model_id}@{card.revision}"
    )
    response = await client.post(
        f"{base}/models/add",
        json={"model_id": card.model_id, "revision": card.revision},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"/models/add failed ({response.status_code}): {response.text}"
        )
    registered = ModelListModel.model_validate(response.json())
    if registered.revision != card.revision:
        raise RuntimeError(
            "Cluster did not register the requested exact model revision: "
            f"requested {card.revision}, received {registered.revision}"
        )


def node_model_status(
    downloads_state: dict[NodeId, list[DownloadProgress]],
    node_id: NodeId,
    model_id: ModelId,
    revision: HuggingFaceRevision,
) -> str:
    entries = downloads_state.get(node_id) or []
    best_status = "not_present"
    for entry in entries:
        card = entry.shard_metadata.model_card
        if card.model_id != model_id or card.revision != revision:
            continue
        if isinstance(entry, DownloadCompleted):
            return "completed"
        if isinstance(entry, DownloadOngoing):
            best_status = "ongoing"
        elif isinstance(entry, DownloadFailed) and best_status == "not_present":
            best_status = "failed"
    return best_status


async def poll_until_complete(
    client: httpx.AsyncClient,
    base: str,
    node_ids: list[NodeId],
    model_id: ModelId,
    revision: HuggingFaceRevision,
    timeout_s: float,
) -> None:
    start = time.monotonic()
    while True:
        response = await client.get(f"{base}/state/downloads")
        response.raise_for_status()
        downloads_state = _DOWNLOADS_STATE_ADAPTER.validate_python(response.json())

        statuses = {
            nid: node_model_status(downloads_state, nid, model_id, revision)
            for nid in node_ids
        }

        for nid, status in statuses.items():
            entries = downloads_state.get(nid) or []
            if status == "ongoing":
                for entry in entries:
                    if not isinstance(entry, DownloadOngoing):
                        continue
                    entry_card = entry.shard_metadata.model_card
                    if (
                        entry_card.model_id != model_id
                        or entry_card.revision != revision
                    ):
                        continue
                    progress = entry.download_progress
                    downloaded_bytes = progress.downloaded.in_bytes
                    total_bytes = progress.total.in_bytes
                    percentage = (
                        downloaded_bytes / total_bytes * 100 if total_bytes else 0.0
                    )
                    speed_megabytes = progress.speed / (1024 * 1024)
                    logger.info(
                        f"{nid}: {percentage:.1f}% @ {speed_megabytes:.1f} MB/s"
                    )
                    break

        if all(s == "completed" for s in statuses.values()):
            logger.info(f"Download complete on all nodes: {list(statuses.keys())}")
            return

        failed = [nid for nid, s in statuses.items() if s == "failed"]
        if failed:
            raise RuntimeError(f"Download failed on nodes: {failed}")

        if time.monotonic() - start > timeout_s:
            pending = [nid for nid, s in statuses.items() if s != "completed"]
            raise TimeoutError(
                f"Downloads did not complete within {timeout_s}s; pending: {pending}"
            )

        await asyncio.sleep(2)


async def run(args: argparse.Namespace) -> int:
    base = f"http://{args.host}:{args.port}"
    model_id = args.model
    revision = (
        validate_hugging_face_revision(args.revision)
        if args.revision is not None
        else None
    )

    requested_model = (
        f"{model_id}@{revision}"
        if revision is not None
        else f"{model_id} (curated pin)"
    )
    logger.info(f"Resolving ModelCard for {requested_model}...")
    card = await ModelCard.load(ModelId(model_id), revision)
    logger.info(
        f"Card: revision={card.revision}, n_layers={card.n_layers}, "
        f"storage={card.storage_size.in_gb:.1f}GB, "
        f"quant={card.quantization or '-'}"
    )

    shard_payload = build_shard_payload(card)

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        await ensure_model_card_registered(client, base, card)

        node_ids = await fetch_topology_nodes(client, base)
        if not node_ids:
            logger.error("No nodes in topology on {}", base)
            return 1
        logger.info(f"Topology has {len(node_ids)} node(s): {node_ids}")

        for node_id in node_ids:
            payload = {"targetNodeId": node_id, "shardMetadata": shard_payload}
            logger.info(f"POST /download/start -> {node_id}")
            response = await client.post(f"{base}/download/start", json=payload)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"/download/start for {node_id} failed "
                    f"({response.status_code}): {response.text}"
                )

        logger.info("Polling for completion...")
        await poll_until_complete(
            client,
            base,
            node_ids,
            ModelId(model_id),
            card.revision,
            timeout_s=args.timeout,
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HuggingFace model id, e.g. zai-org/GLM-5.1")
    parser.add_argument(
        "--revision",
        help=(
            "Exact lowercase 40-hex Hugging Face commit. When omitted, Exo's "
            "unique curated pin is used; an unregistered model falls back to main."
        ),
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=52415)
    parser.add_argument(
        "--timeout",
        type=float,
        default=14400.0,
        help="HTTP + overall wait timeout (seconds). Default 4h.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
