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

## Cross-fitted swap and duplicate-suppression follow-up

The two remaining topology/deployment hypotheses were tested without changing
the frozen Adaptive V3 map learner. A cross-fitted swap miner used alternating
contiguous mapping-trajectory blocks for proposal and gate. It searched 1,576
unused broad tracks and 3,364 unused Gaussian candidates, and proposed only
four pairs whose positive descriptor could replace a wrong Gaussian winner on
at least two rows. Three one-for-one swaps preserved the 119-row matching-rank
constraint and kept the ShopFacade map at 6,357 anchors (5,726 tracks plus 631
Gaussian reserve anchors).

| ShopFacade mapping fold | Median TE | Mean TE | P90 TE | CVaR95 | Raw P@2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selection, before | 1.801 cm | 2.720 cm | 4.489 cm | 17.427 cm | 15.583% |
| Selection, after | 1.841 cm | 2.717 cm | 4.489 cm | 17.418 cm | 15.588% |
| Held-out gate, before | 1.241 cm | 25.646 cm | 4.416 cm | 461.949 cm | 12.798% |
| Held-out gate, after | 1.241 cm | 25.652 cm | 4.413 cm | 461.949 cm | 12.801% |

The held-out meaningful-improvement gate rejected the swaps. Test was not
consulted. This closes the selection-bias gap in the original prune-only
revision and shows that the observed assignment gains do not transfer into a
material pose improvement.

The second experiment retained only the highest-cosine query match for each
anchor before the same single PoseLib solve. It is opt-in; the default still
passes every global top-1 correspondence to PoseLib.

| Scene / method | Queries | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade adaptive | 103 | 1.993 cm | 4.745 cm | 8.318 cm | 9.979% | 42.538% | 4,318 |
| ShopFacade duplicate suppression | 103 | 1.951 cm | 4.832 cm | 9.515 cm | 12.994% | 44.066% | 2,232 |
| Heads adaptive | 1,000 | 0.498 cm | 0.620 cm | 1.144 cm | 11.142% | 24.278% | 3,713 |
| Heads duplicate suppression | 1,000 | 2.586 cm | 2.786 cm | 4.150 cm | 1.035% | 2.000% | 2,914 |

ShopFacade removes 25.0% of correspondences and approximately halves RANSAC
hypotheses, but worsens mean, P90, and 5 cm recall. Heads removes 28.1% and
fails structurally: descriptor score alone cannot decide which of several
query keypoints assigned to one landmark is geometrically correct. The same
failure appears in two seeds over the complete 1,000-query test split. Online
one-to-one suppression is therefore a No-Go, not part of the paper method.

These results trigger the predeclared stop condition for further structural
revision. Adaptive V3 remains frozen; cross-fitted swap and duplicate
suppression stay as reproducible negative-result modules.

As a release guard, the unmodified deployment option was rerun on all 103
ShopFacade test queries with seed 2026. Every estimated pose matrix, match
count, error metric, precision metric, and RANSAC-iteration count exactly
matched the registered Adaptive V3 artifact.

## Geometry, one-shot guidance, and targeted descriptor follow-up

Three remaining hypotheses were implemented as opt-in mapping-only revisions.
Adaptive V3 remains the default. Geometry and descriptor revisions used
disjoint temporal fit, point-validation, and pose-gate blocks. Guided sampling
used fixed all-mapping matchability statistics and a complete mapping replay;
it is therefore a weaker development gate, not a cross-fitted claim. None of
the three inspected test queries before its declared mapping gate.

The first revision refined only the 633 selected Gaussian reserve positions in
their source-splat local tangent/normal frames. Mapping positives supplied the
reprojection term, while source 2DGS scales and the calibrated surface bounds
formed the trust region. Of 62 reserve anchors with sufficient three-way
observations, 34 passed point validation. Their median displacement was 0.521
cm (P90 1.132 cm, maximum 3.086 cm). On 75 unseen mapping views, median/mean/P90
changed from 1.972/3.464/4.150 cm to 1.955/3.461/4.126 cm, so the weak-positive
gate passed.

| ShopFacade test, three-seed mean | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive V3 | 1.993 cm | 4.7449 cm | 8.318 cm | 9.979% | 42.538% | 4,318.2 |
| Bounded reserve geometry | 1.993 cm | 4.7438 cm | 8.321 cm | 9.981% | 42.545% | 4,318.1 |

The full test result is effectively neutral: mean and precision improve by a
noise-level amount, P90 worsens by 0.002 cm, and recall is unchanged. The
geometry branch is therefore valid evidence that bounded surface refinement
can operate safely, but it is not promoted to the default method.

The second revision preserved every global top-1 correspondence and sorted it
by descriptor margin, mapping-derived anchor matchability, and normalized 3D
covariance before one PoseLib PROSAC solve. The mapping gate showed a small P90
improvement (4.622 to 4.527 cm), but only 25 of 231 poses changed and RANSAC
hypotheses fell by just 0.77%. The complete test result rejected the hypothesis:

| ShopFacade test, three-seed mean | Median TE | Mean TE | P90 TE | CVaR95 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard one-shot PoseLib | 1.993 cm | 4.7438 cm | 8.321 cm | 39.738 cm | 4,318.1 |
| Guided one-shot PoseLib | 1.993 cm | 4.7535 cm | 8.321 cm | 39.799 cm | 4,321.2 |

The final revision calibrated only Gaussian reserve descriptors that repeatedly
won an incorrect mapping top-1 while retaining legal positives in disjoint fit
and validation blocks. Descriptor changes were bounded to 0.025, and validation
required fewer false wins without losing any existing correct wins. Seven
anchors had sufficient cross-fitted evidence and two passed anchor-level
validation. On the unseen pose gate, mean changed by 0.00013 cm, P90 by 0.0062
cm, and raw P@2 was identical. This did not meet the predeclared meaningful
improvement threshold, so test was not run.

Together these experiments narrow the remaining limitation: reserve geometry
and isolated harmful descriptors can be changed safely, but neither is the
dominant ShopFacade error source. Mapping-only reliability ordering also does
not transfer into better one-shot RANSAC behavior. The frozen Adaptive V3
topology, descriptors, and standard PoseLib deployment remain the registered
mainline; complete multi-scene benchmarking has higher value than further
single-scene local revisions.
