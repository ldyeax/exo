# Peer artifact deployment for fwuff and dwagon

The deployment path transfers immutable model artifacts once, verifies every
chunk and completed file, and publishes them into a persistent content-addressed
cache. The deployment code itself never quantizes weights. A completed artifact
is reused by SHA-256 on later launches; interrupted transfers resume from
verified chunks. This avoids repeated transfer, but it only avoids repeated
quantization when the served artifact is already the saved, quantized form.

Peer transfer is disabled unless `--peer-artifact-config` or
`EXO_PEER_ARTIFACT_CONFIG` names an owner-only JSON file. Requests use an
HMAC-SHA256 timestamp and one-use nonce. The transport authenticates and
integrity-checks traffic but does not encrypt it, so use it only on the trusted
private fabric.

## Live topology

The GUID-pinned source of truth is
`resources/peer_artifact/fwuffydwagon-five-link-topology.json`. Interface names
are observations, not identities: `scripts/peer_artifact_ipoib.py` resolves
each interface from its HCA, port number, node GUID, and port GUID and fails
closed if the hardware no longer matches.

| Link | dwagon | fwuff | Nominal rate |
| --- | --- | --- | --- |
| `ib-edr` | `ibs5`, `10.44.0.1/30` | `ibs2`, `10.44.0.2/30` | 100 Gb/s |
| `ib-qdr-a` | `ibs4`, `10.44.1.1/30` | `ibs4`, `10.44.1.2/30` | 40 Gb/s |
| `ib-qdr-b` | `ibs4d1`, `10.44.2.1/30` | `ibs4d1`, `10.44.2.2/30` | 40 Gb/s |
| `ethernet-a` | `ens13f0np0`, `192.168.40.24` | `ens17f0`, `192.168.40.93` | 10 Gb/s |
| `ethernet-b` | `ens13f1np1`, `192.168.40.250` | `ens17f1`, `192.168.40.228` | 10 Gb/s |

All three IPoIB rails use datagram mode and MTU 2044. Connected mode was
rejected with `EINVAL` by the mlx5 IPoIB device, so the deployment does not
assume enhanced/connected IPoIB. The default `all` profile uses all five
physical paths. The secondary `four-link` profile omits `ib-qdr-b` and retains
EDR, one QDR port, and both Ethernet links.

## Provision IPoIB safely

Run from the corresponding host before starting NCCL, perftest, or model
servers. The provisioner refuses an apply while one of those processes is
active. A dry run validates the GUIDs, ACTIVE/LINKUP state, link rate, subnet
manager LID, address ownership, and route collisions without changing state:

```bash
python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  provision --host dwagon

python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  provision --host fwuff
```

Apply only after both dry runs pass:

```bash
python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  provision --host dwagon --apply \
  --receipt /var/lib/exo/peer-artifact-deployment/ipoib-dwagon.json

python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  provision --host fwuff --apply \
  --receipt /var/lib/exo/peer-artifact-deployment/ipoib-fwuff.json
```

Apply is idempotent. It snapshots OpenSM process identities and fabric state,
never starts or stops OpenSM, and restores every interface it began changing if
a later validation fails.

The 2026-07-25 live applies are recorded at:

- dwagon:
  `/var/lib/exo/peer-artifact-deployment/ipoib-dwagon-20260725-v1.json`
- fwuff:
  `/var/lib/exo/peer-artifact-deployment/ipoib-fwuff-20260725-v1.json`

They show unchanged OpenSM ownership on fwuff and no local OpenSM on dwagon.
All six bound IPoIB pings passed in both directions.

### Persistence

The live addresses above were applied with `ip` and are not persistent across
a reboot. Owner-only NetworkManager profiles have been generated but not
installed. Each profile matches both the observed interface name and the full
20-byte IPoIB hardware address, whose final eight bytes are validated against
the pinned port GUID:

- dwagon:
  `/var/lib/exo/peer-artifact-deployment/networkmanager-dwagon/`
- fwuff:
  `/var/lib/exo/peer-artifact-deployment/networkmanager-fwuff/`

Regenerate them after any topology edit:

```bash
python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  render-networkmanager --host "$HOSTNAME" \
  --output-directory "/var/lib/exo/peer-artifact-deployment/networkmanager-$HOSTNAME"
```

At a maintenance window, install the three files on their corresponding host:

```bash
install -m 600 \
  "/var/lib/exo/peer-artifact-deployment/networkmanager-$HOSTNAME/"*.nmconnection \
  /etc/NetworkManager/system-connections/
nmcli connection reload
```

Do not activate or reload those connections during an NCCL/model run. Validate
with the provisioner dry run after the profiles activate.

### Rollback

Each apply receipt contains its exact `rollback` command list. The 2026-07-25
state had all three interfaces already up in datagram mode at MTU 2044, so its
complete rollback is only address removal:

```bash
# dwagon
ip address del 10.44.0.1/30 dev ibs5
ip address del 10.44.1.1/30 dev ibs4
ip address del 10.44.2.1/30 dev ibs4d1

# fwuff
ip address del 10.44.0.2/30 dev ibs2
ip address del 10.44.1.2/30 dev ibs4
ip address del 10.44.2.2/30 dev ibs4d1
```

If persistent profiles were installed, first remove only the three
`exo-peer-*.nmconnection` files from
`/etc/NetworkManager/system-connections/`, then reload NetworkManager. Do not
unload `ib_ipoib`, stop OpenSM, or change RDMA device state as part of this
rollback.

## Stage the 26/28/24 views

Run the stager on the writable host that owns the checkpoint. Hardlinks must be
created on the same filesystem as their source shards; create the views on
fwuff when `/mnt/sanic` is read-only on dwagon.

```bash
uv run python scripts/stage_glm52_pp_checkpoint_views.py \
  --model-source /mnt/sanic/glm52 \
  --ktransformers-source /mnt/sanic/glm52-AMXINT4 \
  --destination-root /mnt/sanic/glm52-pp-26-28-24 \
  --partition 26,28,24 \
  --link-mode hardlink
```

This produces `rank-0`, `rank-1`, and `rank-2`, with layer ranges `[0,26)`,
`[26,54)`, and `[54,78)`. Each directory contains:

```text
rank-N/
  stage-view-manifest.json
  model/
  ktransformers/
```

The stage manifest and generated index files are members of the pinned peer
snapshot. The snapshot ID therefore binds the partition, layer range, source
weight maps, and staged files. The suite was atomically materialized on
2026-07-25 at `/mnt/sanic/glm52-pp-26-28-24`; its plan SHA-256 is
`7d74e0272ae5b443f56ebb297b09bf534f8b5d93ba0f747b6c863834018d2a3d`.
An inode check confirmed that the views are hardlinks rather than tensor
copies.

## Render and run the exact deployment

Generate one random secret, copy the same owner-only file to fwuff, and render
the configs on dwagon after all five addresses are active:

```bash
umask 077
openssl rand -hex 32 \
  > /var/lib/exo/peer-artifact-deployment/authentication-secret

python scripts/peer_artifact_ipoib.py \
  --topology resources/peer_artifact/fwuffydwagon-five-link-topology.json \
  render-deployment \
  --secret-file /var/lib/exo/peer-artifact-deployment/authentication-secret \
  --output-directory /var/lib/exo/peer-artifact-deployment/configs
```

The current generated files are:

- fwuff source:
  `/var/lib/exo/peer-artifact-deployment/configs/fwuff-peer-artifacts.json`
- dwagon receiver:
  `/var/lib/exo/peer-artifact-deployment/configs/dwagon-peer-artifacts.json`
- materialization plan:
  `/var/lib/exo/peer-artifact-deployment/configs/materialization-plan.json`

The source config and authentication secret have also been copied owner-only to
the same paths on fwuff. Start the fwuff Exo API with its source config:

```bash
uv run exo \
  --peer-artifact-config \
  /var/lib/exo/peer-artifact-deployment/configs/fwuff-peer-artifacts.json
```

Then materialize the two dwagon ranks. A completed file is atomically published
to the persistent SHA-256 cache, so later runs reuse it without conversion or
network transfer:

```bash
uv run python scripts/materialize_peer_artifact_snapshot.py \
  --config /var/lib/exo/peer-artifact-deployment/configs/dwagon-peer-artifacts.json \
  --peer-node-id fwuff \
  --model-id local/glm-5.2-pp-rank-0 \
  --revision main \
  --destination /var/lib/exo/glm52-pp/rank-0

uv run python scripts/materialize_peer_artifact_snapshot.py \
  --config /var/lib/exo/peer-artifact-deployment/configs/dwagon-peer-artifacts.json \
  --peer-node-id fwuff \
  --model-id local/glm-5.2-pp-rank-1 \
  --revision main \
  --destination /var/lib/exo/glm52-pp/rank-1
```

Launch paths are `<rank-directory>/model` for SGLang and
`<rank-directory>/ktransformers` for KTransformers. Rank 2 remains local on
fwuff. The materializer verifies and publishes snapshots; it does not itself
launch SGLang.

## Five-link proof

`scripts/prove_peer_artifact_multilink.py` performs a small authenticated
end-to-end proof through the real manifest, scheduler, range, integrity, and
cache code. On 2026-07-25, a cold 16 MiB payload transferred and hash-verified
in 0.181 seconds with payload on every configured path:

| Link | Payload bytes |
| --- | ---: |
| `ib-edr` | 8,388,608 |
| `ib-qdr-a` | 3,145,728 |
| `ib-qdr-b` | 3,145,728 |
| `ethernet-a` | 1,048,576 |
| `ethernet-b` | 1,048,576 |

The owner-only receipt is
`/var/lib/exo/peer-artifact-deployment/proofs/five-link-20260725-v1/five-link-transfer-receipt.json`.
This is a routing and integrity proof, not a fabric bandwidth benchmark; its
small size deliberately favors a fast operational check.
