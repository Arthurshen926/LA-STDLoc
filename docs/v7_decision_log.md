# V7 decision log

## 2026-08-26 — mainline reset at `9982d2b`

- Created `codex/v7-safe-closed-loop-mainline` from the correctness-fixed base.
- Enabled only P0 identity no-op reproduction.
- Added a recursive source allowlist and legacy import denylist.
- Registered the twelve immutable method invariants and fail-closed artifact checks.
- Deferred novel views, render certification, selection, feedback learning, and
  acquisition until their preceding phase gates pass.

This log records formal-method decisions only. Historical V6 experiment choices
must not be copied here as V7 defaults.

## 2026-08-26 — P0 passed on StMarysChurch

- The no-op output preserved the frozen 200,255-Anchor map and identity metric
  byte-for-byte and tensor-for-tensor.
- All non-timing fields of all 530 test queries matched the frozen baseline.
- The fixed online contract matched: 2,048-keypoint native SuperPoint, exact
  global Top-1, and one standard PoseLib solve.
- The formal recursive import graph contained only the V7 runner and contract;
  no forbidden historical module was reachable.

This pass unlocks P1 only. Test results were used solely for P0 parity and did
not update, tune, or select a map.

## 2026-08-26 — P1 role and planning contract passed

- Registered five disjoint roles with exact allowed and forbidden operations.
- Added a deterministic planner driven by within-sequence baseline and rotation
  statistics, not fixed cross-scene metre thresholds.
- StMarysChurch validation used 1,487 mapping cameras and generated 64 clean
  feedback plus 64 clean confirmation poses. Neither batch duplicated a mapping
  pose, and the two batches were disjoint.
- Feedback and confirmation artifacts explicitly prohibit Track, Anchor CSR,
  and descriptor-bank membership.
- P2 remains disabled until these plans are rendered and certified.

## 2026-08-26 — P2 clean-render certificate passed

- Rendered the two preregistered, mutually disjoint P1 batches from the frozen
  StMarysChurch 2DGS prior: 64 feedback poses on GPU 1 and 64 confirmation poses
  on GPU 2. Mapping RGB was never loaded and the map was not an input.
- Applied native frozen SuperPoint to each complete, unmasked rendered RGB.
  Alpha, depth, black-hole, and distortion evidence was sampled only after
  detection at keypoint rows; distortion remained a local row mask and was not
  converted into a global render-reliability score.
- Feedback decisions were 60 ACCEPT, 2 UNCERTAIN, and 2 REJECT. Confirmation
  decisions were 63 ACCEPT, 1 UNCERTAIN, and 0 REJECT. Every non-ACCEPT record
  is excluded from map updates by schema and contract.
- Visual audit confirmed that the three marginal-keypoint cases contain large
  near-foreground curtain artifacts. The two depth-curtain rejections look
  plausible in RGB but disagree strongly with camera-support expected depth,
  so they remain conservatively rejected.
- Thresholds were not adjusted after observing these outcomes. An earlier
  renderer-environment failure and a development render lacking all supported
  diagnostic channels are retained as non-formal attempts; only the complete
  v3 manifests are formal evidence.
- All 128 record hashes were verified. The frozen Anchor map hash remained
  unchanged, and a complete P0 rerun again matched all 530 test-query records
  with zero non-timing mismatches and zero forbidden imports.

This pass unlocks P3 only. It does not authorize feedback learning or any use of
UNCERTAIN/REJECT observations.
