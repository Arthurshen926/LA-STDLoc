# Method

## V6 formal mainline: Closed-Loop Projective Anchor Distillation

The formal V6 method begins at the immutable `v4-render-only-frozen` tag and
does not inherit any V5 adapter experiment.  A mapping observation exists only
when Gaussian-rendered alpha support is valid **before** native SuperPoint NMS
and Top-K.  All such observations enter one reciprocal descriptor,
known-pose-epipolar, cycle/chain-confidence association graph.  V6 has no
post-hoc support repair, Track parent/child split, child cap, or direct
depth-surface deployable row.

Every deployable coordinate is reconstructed by robust multi-camera rays with
the shared physical pixel-center convention.  Gaussian alpha/depth may define
the rendered domain, propose a local completion neighborhood, and audit
visibility; neither Gaussian primitive centers nor rendered depth become the
final Anchor coordinate.  Completion observations must independently pass
reciprocal descriptor and known-pose epipolar support before the same ray
triangulator is called.

The current map self-localizes every ordered mapping query with its own
descriptor observations and affected geometry removed.  The online operation
inside that replay is exactly the deployment operation: native SuperPoint,
one global cosine Top-1 Anchor for every query row, and one standard PoseLib
solve.  The offline-only replay emits explicit L1 visibility, L2
detectability, L3 matching, and L4 pose-information failures plus clean,
harmful, and confusion evidence.

Each round evaluates independent descriptor-only, selection-only, and
L1-targeted reconstruction proposals.  Descriptor updates are bounded tangent
residuals on stored map vectors, not an online adapter.  Selection enforces the
ordered hierarchy visibility → detectability → one-to-one matching → pose
information, rather than a weighted heuristic sum.  Candidate acceptance
first enforces catastrophe, R5, map-size, and latency guards and then compares
the frozen lexicographic risk tuple.  Repeated states are rejected and the
loop stops after at most three rounds or immediately when no proposal is
accepted.

The only formal online protocol remains native SuperPoint + global Top-1 + one
PoseLib call.  Retrieval, stronger online features, a learned query adapter,
group-aware RANSAC, pose refinement, and render-time refinement are outside
the method.

## V4 compatibility architecture (non-formal in V6)

The remaining sections document the retained V4 artifact/API compatibility
surface. They do not override the V6 contracts above and are not called by the
V6 runner.

## Evidence-Grounded Anchor Registry

LaFGS reconstructs one sparse localization representation, the
**Evidence-Grounded Anchor Registry**. An Anchor is not a Track row plus a
Gaussian row. It is one deployable observation identity with orthogonal
geometry, descriptor, evidence, and selection fields:

\[
a_i = (\iota_i,\mathcal O_i,\mathbf x_i,\Sigma_i,\mathbf d_i,
E_i^{\mathrm{surf}},E_i^{\mathrm{vis}},L_i,r_i).
\]

- `identity` \(\iota_i\) is the stable Anchor identity and its evidence mode.
- `observations` \(\mathcal O_i=\{(q,k)\}\) are real mapping-image/keypoint
  references, stored as CSR.
- `geometry` \((\mathbf x_i,\Sigma_i)\) is the single 3D point and uncertainty
  consumed by PnP.
- `descriptor` \(\mathbf d_i\) is the single vector used by global retrieval.
- `surface` and `visibility` evidence record whether the frozen RGB Gaussian
  prior supports the point and where it is raster-visible.
- `lineage` \(L_i\) records the contributing Gaussian primitive sources; it
  does not become the Anchor identity.
- `selection reason` \(r_i\) is `precision`, `matching_completion`, or
  `observability_completion`. Old artifacts without exact row provenance are
  explicitly `legacy_unresolved`.

Reliability, matchability, alias-risk, and dependency-group annotations can be
attached without changing any of these identities. This separation gives four
important invariants: one Gaussian may support multiple Anchors; one Anchor may
retain multiple Gaussian sources; surface evidence does not imply that the
deployed geometry depends on the surface; and geometry provenance does not
define a second landmark class.

The serialized `anchor_type` field is retained because historical artifacts
and deployment code consume it. It is a compatibility tag for the row's
construction provenance, not the method's semantic representation.

## Mapping evidence acquisition and initialization

Mapping images are processed by native SuperPoint. The processed resolution,
keypoint budget, NMS radius, and cross-view pair graph form an evidence-
acquisition contract: they determine which observation identities can exist,
but do not create a new deployed Anchor type. Any detector-density or pair-
policy change must be frozen from mapping data and must rebuild all downstream
tracks, geometry, graphs, teachers, topology, and metric state. Alternative
density or pair policies are interfaces for controlled factors; no accuracy
gain is assumed by the representation.

The frozen RGB Gaussian prior supplies the initial surface scaffold. KCS keeps
primitives repeatedly supported by native keypoints across mapping views and
trajectory bins. GWFF initializes their descriptors by geometry-weighted,
robustly trimmed fusion. The 48K value is only a safety cap: adaptive runs stop
at the mapping-supported consensus count rather than filling the scaffold with
ineligible primitives.

This wide scaffold localizes the mapping views with the same global retrieval
rule used online. Keep, swap, miss, false-attractor, bounded-trust, and local-
peak outcomes reconstruct the Stage-A descriptors while the prior geometry and
RGB appearance remain frozen. Its duration is calibrated in mapping-query
epochs,

\[
S_A=\left\lceil E_A\,|\mathcal Q_m|\right\rceil,
\]

not fixed to one cross-scene step count. The resulting A0 state is the baseline
and descriptor source for observation evidence.

## Observation evidence and unified geometry

Reciprocal, cycle-consistent associations are built from real mapping
observations before Gaussian lineage is assigned. Robust multi-view
triangulation estimates image geometry and rejects insufficient parallax,
large reprojection residuals, and excessive uncertainty. Raster provenance
then contributes surface support, visibility, composition mass, and primitive
lineage without replacing the observation identity.

All candidate geometry crosses one boundary:

```text
image triangulation + accepted surface evidence + surface fallback
                              |
                              v
              materialize_geometry(anchor_evidence)
                              |
                              v
       xyz + covariance + geometry_mode + surface_dependence
```

The compatibility policy has three geometry outcomes:

1. an image-stable observation Track keeps its image-only triangulation;
2. a weak Track may use a surface-regularized estimate only if an upstream
   mapping gate accepted that estimate;
3. a surface-initialized fallback row retains its materialized surface point.

`surface_evidence` says that surface support exists;
`surface_dependence` says the deployed coordinate actually uses it. They must
not be conflated. In the audited Heads, Stairs, and ShopFacade deployment
artifacts, accepted surface-regularized Track rows are zero: the validated
behavior is image-triangulated rows plus surface-initialized fallback rows.
The regularization interface is therefore not claimed as an accuracy
contribution.

## One sufficiency selector

Candidate construction may begin with observation-Track evidence or a
surface-initialized fallback, but every candidate enters one registry and one
selected state. The selector has two conceptual phases and three row-level
reasons:

```text
eligible Anchor candidates
          |
          v
   Precision Core                     reason: precision
          |
          v
   Sufficiency Completion
      |-- matching constraint          reason: matching_completion
      `-- pose-observability constraint reason: observability_completion
          |
          v
   one compact Anchor Registry
```

The Precision Core accepts quality-ranked, cross-view observation candidates
until the mapping-query p10 matching-rank target is reached or an eligibility
safety cap is exhausted. Sufficiency Completion operates on the same selected
set and the unified leftover candidate registry. Its matching step uses an
incremental query-row/Anchor bipartite state with unit query-row capacity. Its
observability step adds candidates by full-SE(3) D-optimal gain, reliability,
and image/depth/spatial diversity while enforcing source and voxel capacities.
After every addition, matching or information state is updated and the row is
given exactly one primary reason.

The intended design goal is a smallest sufficient registry,

\[
\min_{S\subseteq\mathcal A}|S|
\quad\text{s.t.}\quad
\operatorname{matchingRank}_q(S)\ge m_q,\qquad
\operatorname{poseInfo}_q(S)\ge h_q,
\]

but the current compatibility implementation does not solve or certify these
per-query constraints exactly. It uses the mapping-query p10 matching target,
then feasible per-query matching completion, followed by a budgeted greedy
full-SE(3) objective with marginal-gain stopping. This distinction matters:
the equation states the organizing objective, while the recorded selector
trace states what was actually enforced. The standard monotone-submodular
guarantee applies only to the cardinality-constrained D-optimal term, not to the
full practical selector.
Historical code and reports may retain `Track core`, `coverage reserve`, or
`pose reserve` names for artifact parity; these are selection stages or reasons,
not separate deployed maps.

## Bounded shared-metric refresh

Once topology is fixed, the compact A1 stage learns one bounded low-rank metric
shared by mapping/query descriptors and Anchor descriptors. Complete-positive
retrieval, current-map hard outcomes, protected correspondences, and
trajectory-group DRO train this metric. The duration is again calibrated only
from mapping-query epochs,

\[
S_C=\left\lceil E_C\,|\mathcal Q_m|\right\rceil,
\]

so a checkpoint suffix is not a universal protocol (for example, a 2,000-view
mapping split resolves to 1,520 steps under the current policy).
The resolved value is written to `scene_calibration.json` and used in the
artifact filenames. Geometry, RGB appearance, candidate identities, and
topology remain frozen during this refresh. Deployment still stores one
descriptor per Anchor; there is no prototype family or test-time context
branch in the main method.

## Sparse deployment

The query image is processed once by native SuperPoint. The shared metric is
applied to query and map descriptors, every query descriptor takes its global
cosine top-1 Anchor, and all resulting 2D-3D correspondences enter one standard
PoseLib absolute-pose RANSAC solve. There is no landmark-side match cap, learned
test-time selector, rendering refinement, or second pose pass. Grid-index
keypoints receive the fixed `+0.5` pixel-center offset before PnP.

Detector density follows processed image area. Reprojection, epipolar, teacher,
and PnP thresholds follow focal/angular scale; metric-space geometry thresholds
follow the mapping track-graph baseline. The PnP threshold is the larger of the
focal-normalized angular gate and the mapping-only stable-track residual term;
the latter is capped at 12 pixels so a heavy tail cannot silently relax the
solver. Test queries never enter calibration, evidence construction,
descriptor learning, selection, or threshold resolution.
