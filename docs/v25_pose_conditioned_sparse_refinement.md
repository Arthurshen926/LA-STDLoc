# V25 pose-conditioned sparse correspondence refinement

## Scope

V25 evaluates an online-only sparse refinement on the five mapping-only,
uncapped Cambridge F0 maps. Offline self-localization feedback, query
rendering, dense matching, descriptor training, and map mutation are disabled.
Test ground truth is used for metrics and the explicitly test-calibrated scene
policy, never as an online input. The result is therefore a known-scene,
test-calibrated deployment result rather than unseen-generalization evidence.

## Selected online path

For one query, the runtime performs native Top-1 matching and one PoseLib solve,
then optionally:

1. projects F0 Anchors with the first pose, without rendering an image;
2. applies mapping-only viewing-direction and distance support before Top-K;
3. retrieves exact Top-64 only for first-pass outlier rows;
4. selects descriptor-close, reprojection-improving alternatives while keeping
   first-pass inliers hard and reserving their Anchors;
5. uses an 8 px projection gate, widened to 12 px only for GreatCourt queries
   with at most 254 first-pass inliers;
6. skips weak proposal sets, otherwise runs one additional robust PoseLib solve;
7. applies the frozen protection/acceptance gate and falls back exactly to F0
   if any requirement fails.

The adaptive gate chooses one radius before correspondence selection. It does
not evaluate both radii online and never adds a third pose solve.

## Five-scene result

| Scene | Policy | Median TE cm | Median RE deg | R5 | Catastrophes >=100 cm |
| --- | --- | ---: | ---: | ---: | ---: |
| GreatCourt | adaptive 8/12 px | 10.269 -> **9.830** | 0.05242 -> **0.05160** | 142 -> 154 | 51 -> 49 |
| KingsCollege | fail closed to F0 | 17.006 -> 17.006 | 0.17390 -> 0.17390 | 57 -> 57 | 0 -> 0 |
| OldHospital | fixed 8 px | 9.707 -> **9.133** | 0.17135 -> **0.16948** | 62 -> 62 | 4 -> 3 |
| ShopFacade | fail closed to F0 | 1.860 -> 1.860 | 0.09084 -> 0.09084 | 86 -> 86 | 0 -> 0 |
| StMarysChurch | fixed 8 px | 4.024 -> **3.965** | 0.12959 -> **0.12854** | 330 -> 331 | 11 -> 11 |
| Pooled, 1,918 queries | selected | 7.969 -> **7.812** | 0.09816 -> **0.09691** | 677 -> 690 | 66 -> 63 |

Pooled median TE improves 1.97%, median RE 1.28%, p90 TE improves from
37.394 to 37.057 cm, and mean TE improves from 67.693 to 67.528 cm. The paired
R5 result is 27 gains and 14 losses. The sparse online stage adds 17.07 ms per
pooled query on average (median 10.70 ms), 8.83% of the measured 193.28 ms F0
mean. This is a 36.5% reduction from the previous 26.9 ms online-stage result.

## Funnel and what limits the gain

GreatCourt illustrates the remaining bottleneck: 743 queries enter the online
band, 494 receive a second solve, and the candidate pose improves both TE and
RE for 212 queries. However, observable self-consistency cannot reliably
separate all good candidates from bad ones: the candidate pool contains 26 R5
gains and 14 losses. The final policy changes 484 outputs and remains
conservative on aggregate metrics.

The complete candidate funnel is stored in the final aggregate. It deliberately
marks correspondence-identity oracle availability as false: no ground-truth
Anchor identity is inferred from pose improvement alone.

## Ablation decisions

- Active-row retrieval plus pre-Top-K view support is selected. It preserves
  every candidate pose exactly while reducing the pose-conditioned geometry
  stage by 45.9% on GreatCourt, 48.8% on OldHospital, and 66.6% on
  StMarysChurch.
- Bounded soft-inlier replacement is rejected. Two GreatCourt settings
  regressed final pose metrics; residual and descriptor margin did not provide
  reliable identity evidence.
- Common-candidate-grid pose energy is diagnostic only. It costs about
  1.60 ms/query but its score distributions for true gains and losses overlap;
  using it as a hard gate reduced GreatCourt R5.
- Simple PROSAC ordering is rejected. OldHospital produced 23/23 identical
  candidate poses and identical iteration counts while adding CPU ordering
  overhead.
- A shallow shared query policy is rejected. Its best same-test result and its
  leave-one-scene-out result both underperformed the frozen scene policy.
- Fixed 12/16 px gates are rejected as global replacements. They improve the
  candidate oracle but add bad candidates and second solves. Only the bounded
  low-inlier 8/12 px route is selected.

## Artifacts

- Protocol: `configs/v25_cambridge_pose_conditioned_sparse_refinement.json`
- Final aggregate:
  `/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901/cambridge_five_scene_sparse_refinement_v25_final_v2.json`
- Aggregate SHA256:
  `7788ceef8e8f944d9a2137f28790483981685eba91a1f3a8b1880aaaf0c5f4f6`
- GreatCourt adaptive results SHA256:
  `2d9cafa89fb6cdf3e15678821cfafd848db565608eeb5e789d0dba7fae2b5a31`

The next accuracy step should not be another acceptance threshold. The real
open problem is a no-GT signal that predicts whether the second pose is truly
better rather than merely more self-consistent. A calibrated pose uncertainty
or independently supported correspondence identity is needed before safely
opening more first-pass inliers or a substantially wider basin.
