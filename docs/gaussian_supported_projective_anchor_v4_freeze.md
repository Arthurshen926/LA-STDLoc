# Gaussian-supported Projective Anchor V4 freeze

## Decision

The render-only method is frozen around the simplest shared deployment path:

```text
RealRGBProvider or GaussianRenderProvider
  -> projective Track and surface-completion candidates
  -> one Unified Anchor Registry
  -> Hierarchical Sufficiency Selector
  -> one identity descriptor bank
  -> global Top-1
  -> one standard PoseLib solve
```

There is no learned descriptor transform in the formal render-only path. The
metric artifact is an identity contract and the evaluator rejects a learned
shared transform or residual. Historical metric-learning modules remain only
for archive reproduction and other experimental pipelines. GWFF-style
observation fusion still materializes one descriptor per Anchor; it is not a
learned deployment transform.

Surface completion is always an available candidate provider. A positive
candidate cap is required, and Track and surface candidates enter the same
registry and selected state. There is no scene-level mechanism switch or
online route by Anchor type: a surface candidate is deployed only when the
shared selector assigns it a strict matching or observability contribution.
The previous `CompatibilitySufficiencySelector` name remains as an archive
alias; the formal name is `HierarchicalSufficiencySelector` and its numerical
policy is unchanged.

Map-to-query brightness/color affine adaptation was neither implemented nor
run. Optional pose feedback and the bounded structured-outlier wrapper also
remain outside the formal method.

Machine-readable results and hashes are in
[`docs/evidence/gaussian_supported_projective_anchor_v4_freeze.json`](evidence/gaussian_supported_projective_anchor_v4_freeze.json).
The 18 small source records are archived in
[`docs/evidence/projective_anchor_v4_simplification_runs`](evidence/projective_anchor_v4_simplification_runs/README.md).

## Cycle-core / chain-reserve ablation

This ablation allowed only cycle-seeded Tracks into the initial Precision Core;
chain-only Tracks could still be recovered by the ordinary sufficiency stages.
It fixed the same candidates, completion input, identity metric, global Top-1,
mapping queries and PoseLib seed as V4.

ShopFacade selected exactly the same 5,794 Anchors and reproduced every reported
mapping metric. Stairs collapsed from 5,772 to 2,055 Anchors: the selected
Track count fell from 5,636 to 1,872, while 183 surface candidates were selected.

| Stairs mapping LOO | V4 baseline | Cycle-core | Delta |
|---|---:|---:|---:|
| Median TE cm | 0.3627 | 0.4542 | +0.0915 |
| Mean TE cm | 3.4452 | 5.1853 | +1.7401 |
| P90 TE cm | 0.8829 | 1.0204 | +0.1375 |
| CVaR95 TE cm | 61.2390 | 94.4642 | +33.2252 |
| 5 cm recall | 98.750% | 99.000% | +0.250 pp |
| Catastrophic >=1 m | 12 | 14 | +2 |
| Raw precision | 6.6390% | 7.2854% | +0.6465 pp |
| Inlier precision | 33.7872% | 31.0394% | -2.7478 pp |
| Solver inlier ratio | 42.3288% | 28.5581% | -13.7707 pp |

The higher raw precision and recall do not compensate for worse central error,
tail risk, catastrophic count and geometric consensus. The ablation is a
scientific **Stop**. The formal selector therefore keeps stable cycle and
chain Tracks in its Precision Core. Because G1 failed, chain-edge LGCV was not
started as a post-hoc rescue.

## Fixed-camera nonlinear point refinement

The second bounded factor keeps every mapping camera, Track component,
observation row, descriptor and candidate budget fixed. It re-triangulates the
existing Track points and performs five robust nonlinear reprojection
iterations on 3D point coordinates only. Camera poses are never optimized.

All 15,210 Stairs and 14,753 ShopFacade Tracks were retained. The robust
objective did not increase for any refined Track. Geometry proxies mostly
improved:

| Scene | Median point shift | Reprojection median px | Reprojection P90 px | High-confidence Tracks |
|---|---:|---:|---:|---:|
| Stairs | 0.01195 m | 0.92534 -> 0.91797 | 2.06040 -> 2.04325 | 12,733 -> 12,827 |
| ShopFacade | 0.00532 m | 0.65170 -> 0.65599 | 1.28330 -> 1.27342 | 14,096 -> 14,144 |

Those proxy gains did not transfer consistently to localization:

| Mapping LOO delta, refined - V4 | Stairs | ShopFacade |
|---|---:|---:|
| Median TE cm | -0.0150 | +0.0174 |
| Mean TE cm | +0.4678 | -0.0745 |
| P90 TE cm | -0.0141 | -0.0395 |
| CVaR95 TE cm | +9.5460 | -1.6215 |
| 5 cm recall | -0.300 pp | 0.000 pp |
| Catastrophic >=1 m | +1 | 0 |
| Raw precision | -0.0330 pp | -0.1556 pp |
| Inlier precision | -3.9361 pp | +0.0295 pp |

Stairs tail error and consensus quality regress materially, while ShopFacade
shows a mixed central/tail exchange. The factor is therefore **not** promoted
and no test queries were read.

## Subpixel boundary

A row-stable 3x3 quadratic SuperPoint peak refinement is implemented and unit
tested. It preserves detector row count, NMS and Top-K identity while moving
only continuous coordinates. It is default-off and was not claimed as a
scientific result: the frozen sparse caches do not contain dense score maps, so
a valid evaluation requires a fresh render, rematch and Track build rather than
an unverifiable edit of the existing rows.

## Final V4 boundary

The default remains the already frozen unified-completion map with identity
descriptors, the ordinary cycle+chain Track policy, projective ray geometry and
standard PoseLib. V4 does not include learned descriptor transforms,
photometric affine adaptation, cycle-core routing, point-only nonlinear
refinement, optional feedback or the structured pose wrapper. The architecture
is ready for broader scene validation; local factors should be reopened only
with a new, clearly different hypothesis.
