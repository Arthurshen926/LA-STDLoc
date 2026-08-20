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

## Frozen experiment and result

The first experiment compares three globally shared protocols:

- `top1`: the frozen independent global Top-1 baseline;
- `assignment_k4`: exact global Top-4 plus capacity assignment; and
- `assignment_k8`: exact global Top-8 plus capacity assignment.

Both assignment variants initially use \(\tau=-1\), so the first comparison
isolates capacity and fallback rank rather than threshold rejection.  No
scene-specific K, threshold, group rule, or prototype count is allowed.

Stairs rejected both forced variants. Relative to Top-1, K4/K8 roughly doubled
mean translation error (3.087 cm to 6.153/6.281 cm) and CVaR95 (54.20 cm to
113.81/116.64 cm), while recall fell from 98.45% to 97.95%. With dustbin -1,
collision resolution forced 1.96M/2.23M rows away from Top-1 and raised mean
PoseLib hypotheses from 3479 to 6150/7278. Forced K4/K8 are permanently
stopped.

The exact reusable Top-8 sidecar is retained: it reproduces historical Top-1
queries and summaries exactly. A no-Pose audit measures positive
R@1/2/4/8=79.92/87.49/92.01/94.69% and maximum correct matching rank 94.42%,
so real candidate headroom remains. The next bounded experiment therefore uses
a true Top-1/Top-2 margin dustbin plus fallback regret, only on the eight fixed
hard scenes.

The hard scenes are evaluated with full-mapping leave-one-query-out (LOO)
descriptor replay.  The current mapping camera remains part of Track identity
and geometry, but its descriptor observations are removed from every affected
Anchor before retrieval.  This is not a held-out fold and it never opens the
test split.

The historical all-scene automation entrypoint is
`scripts/run_v4_assignment_mapping_matrix.py`; its paired summarizer is
`scripts/summarize_v4_assignment_mapping_matrix.py`.  A candidate proceeds to
one frozen official-test run per scene only if mean translation error, P90,
CVaR95, and 5cm/5deg recall are all non-regressive separately on Cambridge,
7Scenes, and 12Scenes.  There is no pooled compensation across datasets.
The current hard-scene panel uses pooled tail/mean/recall safeguards and a
deterministic catastrophic-count, CVaR95, mean, recall, median tie break. It may
select one globally shared candidate, but does not itself authorize official
test access or an all-scene expansion.

## Scope boundary

Multi-prototype Anchors remain deferred until the all-24 assignment audit shows
residual positive-in-Top-K headroom after capacity matching.  If needed, a
future K=2 prototype experiment must collapse prototypes to one Anchor identity
before assignment, retain a shared xyz, and still produce only one PnP vote.
It cannot be mixed into the present assignment factor.
