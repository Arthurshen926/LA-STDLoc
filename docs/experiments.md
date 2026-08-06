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
