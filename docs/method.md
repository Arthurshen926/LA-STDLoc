# Method

## V6 formal mainline: independent feedback-core convergence

The formal V6 method begins at the immutable `v4-render-only-frozen` baseline
and does not inherit any V5 adapter experiment. Its mapping images are renders
from the Gaussian model, not the original mapping RGB images. A mapping
observation exists only when rendered alpha support is valid **before** native
SuperPoint NMS and Top-K. All surviving observations enter one reciprocal-
descriptor, known-pose-epipolar, cycle/chain-confidence association graph.
V6 has no post-hoc support repair, Track parent/child split, child cap, or
direct depth-surface deployable row.

Every deployable coordinate is reconstructed by robust multi-camera rays with
the shared physical pixel-center convention. Gaussian alpha/depth may define
the rendered domain, propose a local completion neighborhood, and audit
visibility; neither Gaussian primitive centers nor rendered depth become the
final Anchor coordinate. Completion observations must independently pass
reciprocal descriptor and known-pose epipolar support before the same pure-ray
triangulator is called. Thus the formal construction order is:

```text
Gaussian render -> alpha-before-NMS observations -> unified association
-> pure-ray Projective Anchor map -> fixed-plant v7 feedback
-> bounded representation/structure/redundancy actions -> compact deployment export
```

The baseline and every proposal receive fresh v7 feedback. The main observer
is **F0 fixed-map feedback**: every ordered mapping query localizes against the
same immutable geometry, descriptor bank, topology, and Anchor subset used by
the deployment plant. **F1 descriptor-leave-self-out** holds geometry and
topology fixed and removes only the current query's direct contribution to
affected Anchor descriptors. Historical pose-neighborhood geometry rebuild is
retained only as the **F2 stress test**; it is not the main control signal.
Strong descriptor positives require exact projective identity. A non-identity
2D neighbor remains diagnostic ambiguity unless aligned surface depth certifies
it as a pose-valid alternative; missing depth fails closed. The operation
measured by every observer remains native SuperPoint, one global cosine Top-1
Anchor for every query row, and one standard PoseLib solve. The evaluator emits separate L1
image-cell visibility, L2 detectability, L3 one-to-one matching, and L4
task-scaled pose-information failures, together with clean, harmful, and
confusion evidence.

The map controller has three bounded actuator families:

- **Representation** changes bounded map-side Anchor descriptors using exact
  identity and, only when available, certified pose-valid alternatives.
- **Structure** changes observation association or pure-ray geometry using
  certified multi-view evidence; 2D proximity alone cannot merge Anchors.
- **Redundancy** removes Anchors only by reverse pruning after representation
  and structure have converged.

Each proposal is evaluated with fresh F0 feedback, audited with F1, and paired
against the same baseline. F2 is a low-frequency fragility audit. The formal runner records diagnostics and artifacts but
does not apply an automatic hard gate, accept a candidate, choose a winner, or
start another round. Acceptance and method selection are an explicit external
manual review of the preregistered panel.

The whole-sequence `seq2` split is a **feedback-action holdout**. D2/D3 do not
use it for descriptor gradients, S1 does not use it to select Anchors, and R1
does not use it as a target or reconstruction-support query. However, the
initial immutable map was constructed from all mapping sequences, including
`seq2`; consequently `seq2` is not an independent validation set for initial
map construction. Paired results on it must be described only as validation
of the feedback-driven action.

The 4-pixel RANSAC setting is retained only for the strict diagnostic replay.
Formal feedback uses the threshold in the SHA-bound Gaussian-render
mapping-only `scene_calibration.json`. A second SHA-bound calibration-binding
artifact attests that exact calibration to the map, observation cache, and
ordered query registry. Baseline and candidate feedback must carry the same
two calibration SHAs.

Training checkpoints retain dense evidence required for F1/F2 audits. Compact
export bakes the final per-Anchor descriptor and removes this
training-only state. The only formal online protocol is therefore the compact
Anchor map plus native SuperPoint, global Top-1, and one PoseLib call.
Retrieval, stronger online features, a learned query adapter, group-aware
RANSAC, pose refinement, and render-time refinement are outside the method.

The sole formal orchestrator is
`scripts/run_v6_feedback_core_pipeline.py`. A one-arm invocation performs the
proposal, full-checkpoint fresh feedback, compact export, and paired
diagnostics. Four one-arm invocations are used because D2 and D3 have different
descriptor weights. `scripts/run_closed_loop_projective_distillation.py` is a
legacy diagnostic/reproduction entry point and cannot define a formal result.

Mapping feedback and manual arm review occur without access to test queries.
After the method, candidate, configuration, and deployment artifact have been
frozen, the test set is evaluated once. Test results cannot be used to revise
the arm choice, thresholds, or map. No formal D2/D3/S1/R1 or test result is
claimed here until its SHA-bound artifact has actually been produced.

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
