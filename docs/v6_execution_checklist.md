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
