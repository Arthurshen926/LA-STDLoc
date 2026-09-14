# V26 confidence-core online sparse refinement

## Scope

V26 changes only online sparse localization on the five mapping-only,
high-capacity Cambridge F0 maps. Offline self-localization feedback, map
mutation, query rendering, and dense matching remain disabled. Test ground
truth is used to report metrics and choose this known-scene policy, never by
the online localizer. The result is therefore test-calibrated rather than an
unseen-generalization claim.

## Selected method

The main change is a `Precision Core + Full Reserve` correspondence flow:

1. Native global Top-1 matching is unchanged.
2. The strongest query-local absolute-cosine fraction forms the first-pass
   confidence core (70% on GreatCourt, 65% on OldHospital and
   StMarysChurch).
3. One PoseLib RANSAC estimates the first pose from that core. Match order is
   restored to native query order, so this is not an implicit PROSAC change.
4. First-pass core inliers are mapped back to their original query rows and
   protected.
5. The complete query registry and high-capacity F0 map remain available to
   the existing pose-conditioned exact Top-64 stage. Only first-pass outlier
   rows retrieve alternatives.
6. View/distance support, reprojection improvement, descriptor closeness,
   one-query/one-Anchor ownership, and the existing bounded proposal gates
   select a sparse second-stage bundle.
7. Candidate inlier gain is measured against the first pose rescored on the
   complete native Top-1 registry, not against the smaller confidence core.
   This prevents the reserve size itself from fabricating an inlier gain.
8. At most one additional bounded PoseLib solve runs. Every failed gate falls
   back to the first-pass confidence-core pose.

KingsCollege and ShopFacade fail closed to native F0 because fixed 65%
filtering regressed their primary medians. The selection is intentionally
scene-calibrated; a future general protocol must replace this with a
mapping/control-calibrated query-level policy.

## Five-scene result

| Scene | Online policy | Median TE cm | Median RE deg | R5 | Catastrophes >=100 cm | P90 TE cm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | Core-70 + mutual Top-K | 10.269 -> **9.776** | 0.05242 -> **0.05153** | 142 -> **160** | 51 -> **42** | 64.98 -> **56.39** |
| KingsCollege | native F0 | 17.006 -> 17.006 | 0.17390 -> 0.17390 | 57 -> 57 | 0 -> 0 | 37.92 -> 37.92 |
| OldHospital | Core-65 + Top-K | 9.707 -> **9.081** | 0.17135 -> **0.16116** | 62 -> 60 | 4 -> **3** | 49.62 -> **47.27** |
| ShopFacade | native F0 | 1.860 -> 1.860 | 0.09084 -> 0.09084 | 86 -> 86 | 0 -> 0 | 7.70 -> 7.70 |
| StMarysChurch | Core-65 + mutual Top-K | 4.024 -> **3.932** | 0.12959 -> **0.12751** | 330 -> **338** | 11 -> **10** | 13.15 -> 13.73 |
| Pooled (1,918) | selected | 7.969 -> **7.860** | 0.09816 -> **0.09743** | 677 -> **701** | 66 -> **55** | 37.39 -> **36.27** |

Pooled paired R5 has 46 gains and 22 losses (net +24). Mean TE improves from
67.69 to 66.00 cm. Compared with V25, V26 gives up about 0.048 cm of pooled
median TE, but gains 11 R5 successes, removes eight additional catastrophes,
and improves P90 and runtime.

The sparse feedback stage itself costs 18.63 ms/query on average and 11.98 ms
at P50 across all five scenes. Confidence-core RANSAC reduces expensive-tail
hypotheses enough that the separately measured end-to-end mean falls from
193.28 to 159.36 ms and P90 from 531.18 to 250.45 ms. End-to-end P50 rises
from 70.45 to 112.12 ms, so the honest runtime conclusion is: throughput and
tail latency improve substantially, while a typical enabled query still pays
the sparse second-stage overhead. Further runtime work should pre-trigger the
candidate-pool construction; the present result does not claim every-frame
latency is lower.

## Ablation decisions

- Pose-conditioned mutual matching is retained for GreatCourt and
  StMarysChurch. It is cheap, but its gain is scene-dependent, so it is not
  forced on OldHospital.
- Held-out sparse candidate-graph validation is diagnostic only. Its observed
  score distributions overlap for true gains and losses, while model
  comparison alone exceeded 160 ms/query in the GreatCourt diagnostic.
- Covariance-expanded 8/10 and 8/12 px projection gates are stopped. High
  projected uncertainty did not provide correspondence identity and regressed
  final GreatCourt metrics.
- Direct Core+Reserve local optimization is stopped. On StMarysChurch, 84
  accepted refinements produced zero R5 gains and ten losses. A low-score
  Top-1 correspondence being consistent with the first pose is not sufficient
  independent evidence.
- Fixed 65% filtering alone is not the selected five-scene method. It improved
  pooled R5 (677 -> 689), catastrophes (66 -> 59), and RANSAC cost, but
  regressed primary medians in three scenes. Its useful role is as the first
  pose core before the sparse Top-K reserve.

## Artifacts

- Protocol: `configs/v26_confidence_core_sparse_refinement.json`
- Aggregator: `scripts/aggregate_v26_confidence_core_sparse_refinement.py`
- Final aggregate:
  `/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901/cambridge_five_scene_sparse_refinement_v26_confidence_core_final.json`
- Final aggregate SHA256:
  `096cc0251adac2509fd4c45fb50a10b9e090967f61462c58ec967312fa00b991`

The next accuracy step is not a wider projection threshold. The strongest
remaining online direction is mapping-only view-conditioned Anchor
descriptors inside the already pose-visible pool, followed by the same exact
Top-K and PoseLib contract. That directly targets correct Anchors that are
geometrically visible but lose descriptor competition, without adding dense
matching or another pose solve.
