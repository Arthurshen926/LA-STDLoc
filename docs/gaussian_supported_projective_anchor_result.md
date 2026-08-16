# Gaussian-supported Projective Anchor architecture and result

## Outcome

The source-image-free branch now implements the intended two-part method:

1. **Gaussian-supported Projective Anchor Construction**; and
2. **Matching-and-Pose Sufficiency Distillation**.

Real and rendered evidence share the same code-level observation contract,
Track construction inputs, descriptor fusion, Anchor candidate registry,
selector state, training materializer, and sparse localizer.  Gaussian
primitives are evidence providers, not a second PnP map.  Track identity and
geometry remain observation/ray defined.  The optional pose-feedback revision
is retained only as historical diagnostics and is not part of this method.

The bounded non-Track completion is useful on Stairs and neutral on
ShopFacade.  It is therefore retained as a unified candidate provider with a
scene-level mapping-only enable decision, not forced into every scene.  The
structured-outlier oracle exposes real hypothesis-generation headroom, but the
bounded practical wrapper does not improve tail failures enough to replace
standard PoseLib.  Its default remains off.

Machine-readable values and hashes are in
[`docs/evidence/gaussian_supported_projective_anchor_result.json`](evidence/gaussian_supported_projective_anchor_result.json).
The compact raw record archive is in
[`docs/evidence/projective_anchor_runs`](evidence/projective_anchor_runs/README.md).

## Shared observation and Anchor construction

`ObservationView` is the common boundary for both providers:

```text
ObservationProvider
  +-- RealRGBObservationProvider
  +-- GaussianRenderObservationProvider
          |
          v
Track / surface evidence -> UnifiedAnchorConstructor
          |
          v
single candidate registry -> Precision Core -> Sufficiency Completion
          |
          v
one descriptor bank -> global Top-1 -> one robust-pose wrapper
```

The contract includes ordered image identity, pose/intrinsics, keypoints,
descriptors, detector scores, validity, alpha/depth support, source kind,
sequence and pose-view bins.  Compatibility replay reproduced the historical
render-only inputs and all 13 legacy map tensors exactly on ShopFacade and
Stairs.  With completion disabled, the new compact maps contain exactly 5,788
and 5,702 Track Anchors respectively; only the new semantic/support fields are
added.

KCS/GWFF no longer describe a parallel Gaussian landmark pipeline:

- KCS/raster evidence supplies visibility, alpha/depth, support-component and
  Gaussian lineage to a candidate provider;
- GWFF-style detector, view, visibility and sequence weighting is the shared
  observation fusion used by Track and completion Anchors;
- historical KCS/GWFF canonical maps enter through the same
  `SurfaceCompletionProvider` interface;
- current render-only completion uses a multi-view rendered-depth surface
  component as identity and weighted depth unprojection as geometry.  It does
  **not** copy a Gaussian primitive center into PnP geometry.

This is the intended representation-level absorption.  It does not claim that
the old primitive-selection implementation was rerun as a new accuracy factor.

## Bounded non-Track Gaussian completion

The completion materializer uses only source-pose Gaussian RGB/depth/alpha and
mapping poses.  It takes at most 256 valid sparse rows per view, requires at
least three observations, two views and two pose bins, clusters observations
in a calibrated surface voxel, and fuses descriptors with the same projective
GWFF routine.  Original mapping RGB and test queries are absent.

| Scene | Legal observations | Eligible components | Candidate cap | Selected by unified selector | Final Track / surface |
|---|---:|---:|---:|---:|---:|
| ShopFacade | 59,136 | 2,096 | 512 | 9 | 5,785 / 9 |
| Stairs | 512,000 | 11,854 | 512 | 136 | 5,636 / 136 |

The completion rows participate in the same matching teacher, observability
evidence, selector trace, compact map and identity metric as Track rows.  There
is no second map, no route-by-anchor-type online logic, and no second pose solve.

### Mapping-only LOO result

| Scene | Arm | Median TE cm | Mean TE cm | P90 TE cm | CVaR95 TE cm | 5 cm recall | Catastrophic >=1 m | Raw precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | Track-only | 0.3416 | 45.2717 | 0.7414 | 864.8672 | 99.134% | 2 | 24.9677% |
| ShopFacade | unified completion | **0.3369** | **45.2700** | 0.7661 | 864.8688 | 99.134% | 2 | 24.9611% |
| Stairs | Track-only | 0.3637 | 3.6779 | 0.9280 | 65.7437 | 98.550% | 13 | 6.3264% |
| Stairs | unified completion | **0.3627** | **3.4452** | **0.8829** | **61.2390** | **98.750%** | **12** | **6.6390%** |

ShopFacade is mapping-neutral with a small P90/precision exchange.  Stairs
improves mean, P90, CVaR, recall, catastrophic count and raw precision in the
same mapping-only replay.

### Frozen real-test result

Values below are means over PoseLib seeds 2026/2027/2028.  Test images were
used only after construction, selection, mapping LOO and the identity metric
were frozen.

| Scene | Arm | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2 cm recall | 5 cm recall | Catastrophic |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | Track-only | 2.0436 | **4.7327** | 8.0572 | 0.22535 | 49.515% | 82.201% | 0.0 |
| ShopFacade | unified completion | **2.0391** | 4.7378 | 8.0572 | **0.22532** | 49.515% | 82.201% | 0.0 |
| Stairs | Track-only | 2.1444 | 7.1927 | 6.5601 | 1.6217 | 46.967% | 85.100% | 9.67 |
| Stairs | unified completion | **2.0553** | **6.5574** | **5.9945** | **1.3714** | **49.067%** | **86.967%** | **9.00** |

Stairs improves every listed central, tail, recall and precision statistic;
mean catastrophic count also drops by 0.67.  ShopFacade remains effectively
neutral.  The evidence supports enabling this completion in the Stairs
render-only configuration, while keeping the ShopFacade/default cross-scene
capacity at zero pending a positive mapping benefit.  It does not authorize a
shared mixed-pipeline default change.

## Structured-outlier-aware pose

The mapping-only oracle compares 32 standard random AP3P samples with 32
distinct-group samples.  Repaired-parent groups are substantially better than
coarse spatial groups:

| Scene | Standard winner correct | Combined standard-score winner correct | Combined group-score winner correct |
|---|---:|---:|---:|
| ShopFacade | 32.03% | **53.25%** | 52.38% |
| Stairs | 39.45% | **66.41%** | 65.23% |

This isolates hypothesis generation, not group-capped scoring, as the useful
upper bound.  A bounded wrapper was therefore implemented: standard PoseLib
remains an explicit candidate, 32 distinct-parent AP3P samples are added, the
standard inlier score selects the winner, and a candidate that does not beat
the original score falls back exactly.

The realized mapping gain is too small.  On ShopFacade, 28/231 group candidates
are selected; median improves 0.0257 cm but P90, CVaR, recall and catastrophic
count are unchanged.  On Stairs, 104/2,000 are selected; median/mean/P90 improve
only 0.0056/0.0067/0.0014 cm and CVaR, recall and catastrophic count are exactly
unchanged.  ShopFacade frozen test is likewise nearly neutral and adds about
64.7 ms mean RANSAC time: median improves 0.0390 cm and 2 cm recall 0.324 pp,
but mean worsens 0.0038 cm and P90/5 cm recall remain identical.

The wrapper remains an explicit research option with tests and exact fallback,
but it is not enabled by default and Stairs test was not run.  A future robust
pose contribution needs a sampler integrated into the full PoseLib hypothesis
process or a genuinely effective structured scorer; this bounded supplement
does not close the tail.

## Final method boundary

The source-image-free method is now structurally complete enough to freeze the
architecture:

- shared Real/Render observation interface;
- observation-defined Track Anchors with ray-triangulated geometry;
- Gaussian support/visibility/lineage and GWFF fusion inside one constructor;
- optional rendered-surface completion inside the same registry and selector;
- hierarchical Precision Core then Matching/Observability Sufficiency
  Completion;
- one compact descriptor bank, global Top-1 and one robust-pose wrapper;
- mapping-only method choice and frozen real-test evaluation.

It still does not claim a joint global pose-error optimum, iterative
feedback/descriptor/topology convergence, or a solved correlated-outlier
estimator.  Optional pose feedback and the bounded group wrapper remain
diagnostics.  The next useful work is broader validation of this frozen
architecture, not another local threshold or revision loop.
