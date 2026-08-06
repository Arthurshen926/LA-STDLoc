# Experiments

Formal reports must include median, mean, and P90 translation error; median and
mean rotation error; 2 cm/2 degree and 5 cm/5 degree recall; raw and RANSAC
inlier GT precision; final anchor count; and runtime/RANSAC iterations.

Prior robustness compares off-the-shelf vanilla 3DGS, vanilla 2DGS, AnySplat
feed-forward Gaussians, and an enhanced 2DGS reference. Only the prior changes.
The mapping images, mapping poses, feature frontend, candidate budget,
topology/A1 schedule, and one-shot PoseLib protocol remain fixed.

Historical ablations and no-go branches are not shipped in this branch. Their
complete source remains in Git tag `archive-full-research-20260803`.

## Adaptive V2 smoke validation

The mapping-only adaptive topology was isolated on ShopFacade by reusing the
same frozen V1 scaffold, function graph, Track-First payload, and native query
cache. Compact-map labels and the metric were rebuilt, then all 103 test images
were evaluated with seed 2026 and the unchanged one-shot sparse PoseLib path.

| Topology | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen V1 | 7,145 | 1.808 cm | 4.647 cm | 7.312 cm | 10.074% | 40.313% | 81.553% |
| Adaptive V2, aggressive rank target | 4,159 | 1.792 cm | 4.744 cm | 7.784 cm | 9.004% | 44.782% | 81.553% |
| Adaptive V2, retained policy | 6,481 | 1.938 cm | 4.692 cm | 8.132 cm | 10.001% | 42.426% | 82.524% |

The retained policy reduces map capacity by 9.3% while preserving mean error,
raw precision, and recall within a small range, but does not improve median or
P90. The aggressive variant demonstrates additional redundancy but loses raw
cleanliness and RANSAC efficiency. This is a compatibility smoke validation,
not a multi-seed replacement for the registered frozen V1 result.

## Cross-scale adaptive validation

The completed adaptive pipeline was then checked on an outdoor Cambridge scene
and a clean, from-prior indoor rebuild. All reported localization runs use
native full-resolution inputs, uncapped global top-1 matching, one PoseLib
solve, and the complete test split. Pose metrics below are means over seeds
2026/2027/2028 unless noted otherwise.

| Scene / method | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade frozen V1 | 7,145 | 1.808 cm | 4.647 cm | 7.312 cm | 10.074% | 40.313% | 81.553% |
| ShopFacade adaptive | 6,357 | 1.993 cm | 4.74 cm | 8.318 cm | 9.979% | 42.538% | 81.553% |
| Heads frozen V1 | 8,264 | 0.465 cm | 0.575 cm | 1.062 cm | 9.186% | 18.346% | 100.0% |
| Heads adaptive, angular-only PnP | 8,119 | 0.447 cm | 0.721 cm | 1.435 cm | 11.136% | 48.717% | 99.2% |
| Heads adaptive, track-residual PnP | 8,119 | 0.498 cm | 0.619 cm | 1.145 cm | 11.142% | 24.277% | 99.93% |

The indoor angular-only threshold (`3.788 px`) was below the residual scale of
stable mapping tracks. The final mapping-only rule uses their per-track p90
residual at quantile 0.975 (`10.419 px` on Heads), capped at 12 pixels. It
reduces Heads P90 by 20%, restores 5 cm recall, and cuts mean RANSAC hypotheses
from 11,790 to 3,713 relative to the angular-only run. Against frozen V1 it is
a mixed but useful efficiency/cleanliness Pareto point: fewer anchors and half
the detector keypoints, higher raw and inlier GT precision, similar P90 and
recall, but a slightly higher median and mean. ShopFacade automatically remains
at 12 pixels and reproduces the prior adaptive result.

These two scenes establish scale adaptation and catch the indoor solver-gate
failure, but they are not sufficient evidence for universal non-inferiority.
The full Cambridge, 7Scenes, and 12Scenes evaluation remains required.

## Deployment-aware topology revision gate

A one-pass mapping-only revision was tested after compact reconstruction on
ShopFacade. It replayed exact global top-1 plus standard PoseLib on all 231
mapping views, scored counterfactual clean replacements, clean/harmful solver
inliers, full-SE(3) Fisher deletion loss, and matching-rank criticality, then
protected anchors with non-improving replacements in the worst 10% mapping
queries. The revision removed 7 of 6,357 anchors, all from the Gaussian reserve;
the matching-rank p10 remained 159 against a target of 119.

| Test method, three-seed mean | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 2 cm recall | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive V2 | 6,357 | 1.993 cm | 4.7449 cm | 8.318 cm | 9.979% | 42.538% | 50.49% | 81.55% |
| Revision only | 6,350 | 1.981 cm | 4.7439 cm | 8.350 cm | 9.977% | 42.547% | 51.46% | 81.55% |
| Revision + 44-step full-shard refresh | 6,350 | 1.981 cm | 4.7496 cm | 8.350 cm | 9.982% | 42.552% | 51.46% | 82.52% |

The original trainer replayed only one of seven rotating mapping shards when
`refresh_interval=0`. Enabling all shards exposed an uncapped entropic Group-DRO
collapse: one trajectory group reached 98.6% weight after two refreshes. A
capped update (at most three times uniform weight) fixes that failure mode and
kept an effective 5.1--5.3 groups during the short refresh.

The result is mixed rather than Pareto improving: median, clean precision, and
recall improve slightly, while mean, P90, and RANSAC hypotheses do not. The
mapping-only meaningful-improvement gate therefore rejects the revision as a
default method. The frozen adaptive mainline and its one-shot deployment remain
unchanged; the revision is retained as an opt-in, reproducible experiment.
