# V4 Capacity-Feasible Correspondence Assignment

## Motivation

The frozen Projective Anchor map is constructed under a matching-feasibility
constraint, but the original deployment rule independently selected one global
Top-1 Anchor for every query row.  That mismatch has two consequences:

1. several query rows may vote for the same 3D Anchor; and
2. a row whose correct Anchor is ranked second or fourth cannot use it even when
   another row has the stronger claim on the shared Top-1 Anchor.

This revision changes correspondence extraction only.  It does not alter the
Gaussian prior, rendered observations, Tracks, Anchor geometry, descriptor
bank, selector, camera model, or robust-pose solver.

## Deployment objective

For query row \(q\), retrieve its exact global cosine Top-K Anchor candidates
\(C_q\).  With cosine score \(s_{qa}\) and a shared dustbin score \(\tau\), solve

\[
\max_x \sum_{q,a\in C_q} x_{qa}(s_{qa}-\tau)
\]

subject to

\[
\sum_a x_{qa}\leq 1,\qquad \sum_q x_{qa}\leq 1,\qquad x_{qa}\in\{0,1\}.
\]

Every row has its own unmatched/dustbin edge.  A real edge must be strictly
above \(\tau\).  The implementation constructs only the sparse Top-K graph and
uses an exact maximum-weight bipartite solver; it never materializes a dense
query-by-map assignment matrix.  Candidates are returned in query-row order
and are passed to exactly one standard PoseLib call.

The resulting method has a single coherent set-level interpretation:

- map construction preserves a matching-feasible Anchor set;
- deployment extracts a capacity-feasible correspondence set; and
- existing D-optimal/observability selection preserves pose-support diversity.

## Frozen experiment

The first experiment compares three globally shared protocols:

- `top1`: the frozen independent global Top-1 baseline;
- `assignment_k4`: exact global Top-4 plus capacity assignment; and
- `assignment_k8`: exact global Top-8 plus capacity assignment.

Both assignment variants initially use \(\tau=-1\), so the first comparison
isolates capacity and fallback rank rather than threshold rejection.  No
scene-specific K, threshold, group rule, or prototype count is allowed.

All 24 scenes are evaluated first with full-mapping leave-one-query-out (LOO)
descriptor replay.  The current mapping camera remains part of Track identity
and geometry, but its descriptor observations are removed from every affected
Anchor before retrieval.  This is not a held-out fold and it never opens the
test split.

The automation entrypoint is
`scripts/run_v4_assignment_mapping_matrix.py`; its paired summarizer is
`scripts/summarize_v4_assignment_mapping_matrix.py`.  A candidate proceeds to
one frozen official-test run per scene only if mean translation error, P90,
CVaR95, and 5cm/5deg recall are all non-regressive separately on Cambridge,
7Scenes, and 12Scenes.  There is no pooled compensation across datasets.
If both K values pass, selection minimizes the worst-dataset CVaR95 ratio,
then the worst-dataset mean-error ratio, then prefers the smaller K.  This rule
is frozen before results are read.  Only that one selected variant is permitted
one official-test run per scene through
`scripts/run_v4_assignment_test_matrix.py`.

## Scope boundary

Multi-prototype Anchors remain deferred until the all-24 assignment audit shows
residual positive-in-Top-K headroom after capacity matching.  If needed, a
future K=2 prototype experiment must collapse prototypes to one Anchor identity
before assignment, retain a shared xyz, and still produce only one PnP vote.
It cannot be mixed into the present assignment factor.
