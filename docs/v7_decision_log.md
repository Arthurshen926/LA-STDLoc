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
