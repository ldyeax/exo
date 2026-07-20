# InfiniBand cards

## Overview

`dwagon` and `fwuff` are directly connected using both ports of dual-port Mellanox ConnectX-3 VPI adapters. Both hosts negotiate PCIe 3.0 x8 with a 512-byte maximum PCIe payload size. Each InfiniBand link currently trains as 4x QDR (40 Gb/s signaling, approximately 32 Gb/s payload bandwidth).

The `fwuff` card originally had an OEM QDR-only firmware personality which imposed an additional card-wide dual-port throughput limit. Cross-flashing it to the generic Mellanox FDR personality removed that limit. The two ports now reach about 52.9 Gb/s aggregate in one direction, which is the practical unidirectional ceiling of the ConnectX-3/PCIe 3.0 x8 data path.

## Adapter details

### `dwagon`

- Device: Mellanox ConnectX-3, MT4099 / MT27500 family
- Firmware: 2.35.6312
- PSID: `ORC1090120019`
- Firmware personality: Oracle OEM FDR/QDR
- PCIe: Gen3 x8

### `fwuff`

- Device: Mellanox ConnectX-3, MT4099 / MT27500 family
- PCI address: `16:00.0`
- Original firmware: 2.40.5030
- Original PSID: `ISL1090110018`
- Original profile: `cx3-1_MCX354A_qdr.prs`
- Current firmware: 2.42.5000
- Current PSID: `MT_1090120019`
- Current profile: `cx3-1_MCX354A_fdr_09v.prs`
- PCIe: Gen3 x8

## Changes made on `fwuff`

1. Installed the official NVIDIA MFT 4.35.0-159 kernel DKMS package so MFT could access the ConnectX-3 flash through `/dev/mst/mt4099_pci_cr0`.
2. Corrected the DKMS build to use `/usr/bin/gcc-15`, matching the compiler used for the running `6.17.0-1013-aws` kernel. The DKMS configuration now builds the MFT modules with that compiler.
3. Saved a complete firmware, configuration, VPD, PCI, and query backup before changing the card.
4. Stopped OpenSM and cross-flashed the card from the OEM QDR PSID to the generic Mellanox FDR PSID using `flint --allow_psid_change`.
5. Performed a complete post-burn verification. The original GUIDs and MAC addresses were preserved.
6. Rebooted `fwuff`, loaded `ib_umad`, restarted OpenSM for both direct links, and verified both ports and RDMA traffic.

The flashed image was:

```text
fw-ConnectX3-rel-2_42_5000-MCX354A-FCB_A2-A5-FlexBoot-3.4.752.bin
```

The burn and full verification completed successfully.

## Firmware backups

The pre-flash backup exists on both machines:

```text
fwuff:  /root/mft-backup-fwuff-20260719
dwagon: /root/infiniband_fiberchannel/mft-backup-fwuff-20260719
```

The original firmware image is:

```text
fwuff-ISL1090110018-2.40.5030.bin
SHA256: 85b114722988d6697be80ad27b3b25173a4e573342f519245c76959228f1f107
```

Keep this backup if reverting the PSID or recovering the card is ever necessary.

## Benchmark results

Tests used RDMA write traffic over the two direct InfiniBand links.

| Test | Before cross-flash | After cross-flash |
|---|---:|---:|
| One port | 31.53 Gb/s | 31.53 Gb/s |
| Both ports, `dwagon` to `fwuff` | 32.07 Gb/s total | 52.90 Gb/s total |
| Per port during that dual test | about 16.03 Gb/s | about 26.45 Gb/s |
| Both ports, `fwuff` to `dwagon` | 33.02 Gb/s total | 52.74 Gb/s total |

Additional baseline results:

- One port bidirectional: 61.42 Gb/s combined.
- Two ports carrying traffic in opposite directions: 62.54 Gb/s combined.
- Error and discard counters did not increase during the post-flash load tests.

The cross-flash improved same-direction dual-port throughput by approximately 65 percent without changing single-port performance. This isolates the original deficiency to the OEM QDR firmware personality rather than the cables, PCIe link width, RDMA stack, or OpenSM.

The approximately 52.9 Gb/s post-flash aggregate is consistent with the practical unidirectional limit of PCIe 3.0 x8. PCIe 3.0 x8 provides 63.02 Gb/s after 128b/130b encoding; the measured RDMA payload is about 84 percent of that after PCIe transaction, flow-control, DMA, and adapter overhead. The opposite-direction test performs better because PCIe is full-duplex.

## Remaining cable/FDR behavior

Both links remain at QDR rather than training at FDR. All four port endpoints report:

```text
LinkSpeedExtSupported: 14.0625 Gbps
LinkSpeedExtEnabled:   0
LinkSpeedExtActive:    No Extended Speed
LinkSpeedActive:       10.0 Gbps
LinkWidthActive:       4X
```

After the generic FDR firmware was activated, `fwuff` logged one:

```text
Unsupported cable detected
```

`dwagon` logged no cable warning. The Linux `mlx4` driver does not include a port number in this particular warning, so the event cannot identify whether one cable or both cables fail Mellanox's FDR cable qualification. Both cable paths are currently held to QDR. MFT could not enumerate the cable EEPROMs through this generation of ConnectX-3 hardware.

Replacing the cables with known FDR-qualified QSFP cables is the sensible next step if actual 56 Gb/s FDR link training is desired. It will not materially improve aggregate same-direction throughput beyond the present approximately 53 Gb/s because PCIe 3.0 x8 is already the bottleneck, but it could improve single-port speed and traffic distribution.

To isolate the warning to a physical cable, connect one cable at a time and fully power-cycle the `fwuff` adapter between tests. A real PCIe hot-swap slot power cycle is sufficient if the platform removes slot and auxiliary power; a PCI function reset or sysfs remove/rescan alone may not force ConnectX-3 to reload newly flashed firmware.

## Recovery notes

- The original flash is recoverable from the backup above with MFT/`flint` and an allowed PSID change.
- If the adapter cannot boot normally, ConnectX-3 Livefish/recovery mode can be used with MFT.
- `fwuff`'s BMC was intentionally unavailable during this work and was not needed.
- Run `mst start --with_unknown` before using `/dev/mst/mt4099_pci_cr0` for future flash operations.
