# V6 execution checklist

V6 starts from tagged commit `v4-render-only-frozen` and does not inherit any
historical V5 adapter or target-materialization branch.  The only formal
deployment protocol is native SuperPoint, global Top-1 per query row, and one
standard PoseLib solve.

The required dependency order is:

1. render-valid observations;
2. one evidence-gated multiview association graph;
3. ray-triangulated Projective Anchors, including completion proposals;
4. query-local LOO L1--L4 feedback;
5. independent descriptor, reconstruction, and selection proposals;
6. exact mapping replay and lexicographic guarded acceptance;
7. repeat until no proposal is accepted, with a maximum of three rounds.

The following are compatibility-only and forbidden from the V6 default
runner: appearance ensemble, post-hoc support repair, parent/child Track
semantics, child caps, direct-depth deployable Surface Anchors, source RGB,
render-to-real adapters, retrieval, stronger online features, group-aware
PoseLib variants, and pose refinement.

Every item in the external fourteen-item implementation plan is tracked by a
test or a real-scene evidence artifact before the V6 mainline can be frozen.

| # | Plan item | V6 implementation status |
|---|---|---|
| 1 | Immutable V4 baseline | Done: `v4-render-only-frozen` |
| 2 | Clean V6 branch and schemas | Done: `codex/v6-closed-loop-projective-anchor` |
| 3 | Surface duplicate/pixel/covariance/self-certification bugs | Implemented; compatibility regression pending full suite |
| 4 | Alpha validity before SuperPoint NMS/Top-K | Implemented and unit-tested |
| 5 | Rebuild hard/protection observations, Tracks, geometry | Real-scene run pending clean producer commit |
| 6 | Carry support into one unified association | Implemented in `projective_association_graph_v2` |
| 7 | Remove repair/parent-child/child-cap from formal path | Implemented; all are absent from V6 materializer |
| 8 | Replace direct Surface rows with projective completion | Implemented: depth proposal + reciprocal/epipolar + pure-ray xyz |
| 9 | Formal query-local descriptor and geometry LOO feedback | Implemented with L1–L4 serialized feedback |
| 10 | Descriptor-only and selection-only arms | Implemented as offline map proposals |
| 11 | Reconstruction arm and complete round-one panel | Implemented runner pieces; real round-one pending |
| 12 | Lexicographic guarded acceptance and round two | Implemented; real round two depends on round-one acceptance |
| 13 | Full 24-scene panel | Intentionally pending hard/protection gate |
| 14 | Method/config/runner alignment | Method and config updated; final evidence table pending runs |
