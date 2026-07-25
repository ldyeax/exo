• The dual-port “deficiency” is caused by fwuff’s HCA personality, not the links or PCIe slot.

  fwuff contains an OEM Isilon MCX354A-QCBT with PSID ISL1090110018. Its firmware dump identifies the source profile as:

  cx3-1_MCX354A_qdr.prs
  dual-port QSFP; QDR IB (40Gb/s)

  That QDR personality shares one 40 Gb/s full-duplex data engine across both ports. QDR’s 8b/10b encoding leaves about 32 Gb/s of usable bandwidth per direction, exactly
  matching the measurement. NVIDIA documents this model as the dual-port 40 Gb/s QDR MCX354A-QCBT. ConnectX-3 manual
  (https://network.nvidia.com/sites/default/files/pdf/user_manuals/ConnectX-3_VPI_Single_and_Dual_QSFP_Port_Adapter_Card_User_Manual.pdf), firmware release notes
  (https://network.nvidia.com/pdf/firmware/ConnectX3-FW-2_42_5000-release_notes.pdf).

  Evidence:

   Traffic pattern                                   Result
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━
   One port, one direction                       31.53 Gb/s
  ────────────────────────────────  ────────────────────────
   Two ports, same direction         32.07–33.02 Gb/s total
  ────────────────────────────────  ────────────────────────
   Two ports, opposite directions          62.54 Gb/s total
  ────────────────────────────────  ────────────────────────
   One port, bidirectional                 61.42 Gb/s total
  ────────────────────────────────  ────────────────────────
   Dual-port RDMA read                     33.06 Gb/s total

  This rules out:

  - PCIe: both cards negotiate PCIe 3.0 ×8; the driver reports 63.008 Gb/s available per direction.
  - CPU scheduling: separately pinned workers still split the same 32 Gb/s.
  - perftest skew: matching 6.24 binaries still gave 32.07 Gb/s.
  - RDMA-write behavior: RDMA read has the same ceiling.
  - Cabling/link faults: both links are active at QDR with zero errors or discards.

  dwagon has an Oracle FDR/QDR profile (ORC1090120019), but every test traverses fwuff’s QDR-profile HCA, making fwuff the limiting endpoint.

  The practical way to obtain additive same-direction bandwidth is to replace fwuff’s QDR-profile card with an FDR/newer HCA or use two separate HCAs. Cross-flashing this OEM QDR
  card to an FDR profile would be unsupported and is not a dependable fix.

  OpenSM remains active, no benchmark processes remain, and I left mft plus mstflint installed on fwuff for firmware diagnostics.
