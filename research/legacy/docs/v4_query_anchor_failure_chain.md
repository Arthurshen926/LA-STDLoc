# V4 per-query Anchor failure-chain audit

`scripts/audit_query_anchor_failure_chain.py` is a diagnostic-only command for
separating map coverage, detector access, descriptor recall, candidate
selection, and solver failures. It does not change the materialized map, shared
metric, default localizer, or PoseLib parameters.

The audit consumes the existing materialized Anchor map, identity metric,
rendered Track payload, positive teacher, query cache, and scene calibration.
Every input is specified together with its expected SHA-256. The command also
requires a clean producer worktree, verifies the map/metric/payload/teacher
lineage, writes atomically, and records source-file hashes. A test-query cache is
rejected unless both the artifact declares `frozen_after_mapping=True` and the
operator explicitly supplies `--allow-test-diagnostic`. Test diagnostics are
read-only and may not update the map or any hyperparameter.

## Failure layers

1. **L1 map coverage or geometry.** Project all selected Track and surface
   completion Anchors at the GT pose. Report in-frame and alpha-supported
   visibility, Track/surface counts, 4x4 image coverage, 3D extent and axis
   ratios, task-scaled Fisher/D-opt information, and non-degenerate AP3P sets.
2. **L2 detector access.** Match detected keypoints to GT-visible Anchors within
   the frozen teacher radius using maximum-cardinality, minimum-distance
   bipartite matching. Report detector recall, spatial coverage, and rank.
3. **L3 descriptor recall.** Replay the mapping-query leave-one-out descriptor
   bank and exact global Top-32. Report correct-Anchor recall at
   1/2/4/8/16/32 and best-positive-minus-best-wrong margins for all, Track, and
   surface completion Anchors separately.
4. **L4 candidate selection or structure.** Report the maximum GT-positive
   matching and its spatial/3D diagnostics, then run the unchanged standard
   PoseLib solver directly on that oracle correspondence set.
5. **L5 solver gap.** Compare the direct oracle solve with the unchanged
   deployed Top-1 solve. This layer records a solver gap only; it never changes
   solver settings.

Gaussian alpha is usable as visibility evidence. Gaussian native depth is
reported but is `audit_only_never_hard_reject` for the current teacher: Track
geometry was obtained by ray triangulation and is not required to lie on the
native Gaussian depth surface. Hard depth rejection reduced a known-correct
ShopFacade query from 5,412 in-frame Anchors to 6, so treating it as an occlusion
oracle would create a false L1 diagnosis.

## Initial mapping-only probe

The frozen probe used producer commit `e29536b5f2d39490152b6382fde4ea08b0543fb1`.
The compact evidence manifest is
`docs/evidence/v4_query_anchor_failure_chain_probe.json`; full reports and
tensor sidecars remain under the recorded `/mnt/pool/sqy` paths and are
SHA-bound in that manifest.

- ShopFacade guard `seq2/frame00001.png` passes (deployed 0.895 cm / 0.025 deg).
- ShopFacade tails `seq2/frame00036.png` and `seq2/frame00043.png` are L1: zero
  selected Anchors project in-frame at GT.
- Stairs guard `seq-02/frame-000000.color.png` passes (1.490 cm / 0.371 deg).
- Stairs `seq-02/frame-000464.color.png` is L3: 14 detector-accessible
  GT-positive correspondences and a 0.128 cm oracle pose exist, but only one
  distinct correct match remains in Top-32.
- Stairs `seq-02/frame-000458.color.png`, `seq-05/frame-000234.color.png`, and
  `seq-05/frame-000484.color.png` are L4. Their GT-positive oracle poses are
  0.119, 0.113, and 0.190 cm, while Top-32 contains 22, 6, and 21 distinct
  correct correspondences respectively. This is bounded headroom for a better
  candidate-structure method, not evidence for changing PoseLib.

The earlier co-visibility/group hypothesis prototype remains rejected: this
audit shows that correct candidates exist for three Stairs tails, but not that
the existing group metadata organizes them into a correct pose hypothesis.

