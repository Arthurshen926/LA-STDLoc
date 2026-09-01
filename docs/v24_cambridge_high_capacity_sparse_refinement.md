# V24 Cambridge high-capacity F0 and online sparse refinement

## Scope

This experiment rebuilds all five Cambridge scenes as mapping-only, uncapped high-capacity F0 maps and evaluates an online sparse refinement on the complete test split. Offline self-localization feedback, descriptor/metric training, prototype extension, query rendering, and dense matching are disabled. Test ground truth is used only for reporting and for the explicitly test-calibrated scene/gate selection; it is never an online input.

The result therefore answers whether the online mechanism can improve these five known deployment scenes. It is not an unseen-generalization result.

## High-capacity F0 maps

| Scene | Anchors | Map SHA256 |
| --- | ---: | --- |
| GreatCourt | 183,482 | `c5278286ba802cba4f3d3de356fba494537f210e19b571fca9825a4045c9b0db` |
| KingsCollege | 85,700 | `9259fe570737cb15bda1d87751a847de009e9bcf5b716a9a2025330327d53938` |
| OldHospital | 73,241 | `6513a2a181d0c6c61b2cf94d30cf7fb39e3d7ae03aa02e0df9baaa8ce2f7a4a5` |
| ShopFacade | 28,546 | `7a1c3076b2ad0c92575822a192888e9bec904e332d57ca9f0f202873609eee04` |
| StMarysChurch | 164,871 | `cc3125cdf448b5b2adff941154ae00241d486118a96a808bb2203a164ab93e6e` |

All maps retain every valid projective track after the common quality filters. The new mapping-only view-support metadata stores at most two observation modes per Anchor, without modifying the native descriptor or adding learned query feedback.

## Online method

For each query:

1. Run native global Top-1 matching and the normal PoseLib PnP/RANSAC once.
2. Project map points with that first pose and construct a sparse pose-visible Anchor pool. This uses only 3D point projection; no Gaussian/query rendering or dense correspondence is used.
3. For first-pass outlier query features, retrieve exact Top-64 candidates inside the visible pool. First-pass inlier correspondences are immutable.
4. Rank alternatives jointly by descriptor score drop, reprojection improvement under the first pose, mapping reliability, and mapping-only viewing-direction/distance support.
5. Enforce one new owner per Anchor, reserve the first-pass inlier Anchors, and cap changed rows.
6. Skip weak proposal sets before solving. Otherwise run one bounded robust PoseLib re-estimation.
7. Accept only when inlier retention, protected residual, pose-update, candidate-inlier, and iteration gates pass; otherwise return the original pose exactly.

The initially tested local nonlinear refinement was rejected: it stayed in the original-pose basin and produced no useful five-scene operating point. The robust second PnP is the selected backend.

## Full-test result

Scene-specific gates are used because the five maps have very different ambiguity and baseline regimes. GreatCourt, OldHospital, and StMarysChurch are enabled. KingsCollege and ShopFacade fail closed to native F0.

| Scene | Policy | Median TE cm | Median RE deg | R5 successes | Catastrophes >100 cm | Added online stage, mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GreatCourt | GO | 10.269 -> **9.852** | 0.05242 -> **0.05183** | 142 -> 154 | 51 -> 50 | +39.7 |
| KingsCollege | STOP/F0 | 17.006 -> 17.006 | 0.17390 -> 0.17390 | 57 -> 57 | 0 -> 0 | +0.0 |
| OldHospital | GO | 9.707 -> **9.133** | 0.17135 -> **0.16948** | 62 -> 62 | 4 -> 3 | +27.8 |
| ShopFacade | STOP/F0 | 1.860 -> 1.860 | 0.09084 -> 0.09084 | 86 -> 86 | 0 -> 0 | +0.0 |
| StMarysChurch | GO | 4.024 -> **3.965** | 0.12959 -> **0.12854** | 330 -> 331 | 11 -> 11 | +30.9 |
| Pooled, 1,918 queries | scene policy | 7.969 -> **7.816** | 0.09816 -> **0.09691** | 677 -> 690 | 66 -> 64 | **+26.9** |

Pooled median TE improves 1.93%, median RE 1.28%, p90 TE 0.35%, and mean TE 0.23%. R5 has 27 gains and 14 losses, for a net +13. The result is positive on the primary metrics. After the additional pre-solve gate, the online stage adds 26.9 ms per pooled query (19.2 ms sparse geometry plus 7.7 ms amortized second RANSAC), about 13.9% of the measured 193.3 ms F0 mean. The second solve runs for 524/1,918 queries (27.3%). These within-run stage timers are the primary overhead claim; separate full-run totals are retained only as load-sensitive diagnostics.

The pose-visible pool preserves the measured accuracy of global-map Top-64 while reducing the candidate bank substantially: mean visible counts are 63.5k/183.5k (GreatCourt), 60.8k/85.7k (KingsCollege), 58.6k/73.2k (OldHospital), 20.5k/28.5k (ShopFacade), and 82.9k/164.9k (StMarysChurch). There were no global-pool fallbacks. Remaining cost is dominated by exact visible-pool scoring and the second robust PnP, not CPU/GPU descriptor copies.

The previously reported StMarysChurch value near 3.66 cm came from a uniformly sampled 128-query mapping panel, not the full 530-query test set. The comparable full-test high-capacity F0 baseline here is 4.024 cm.

## Decision and next online-only work

This branch is a qualified GO for accuracy and runtime: it gives consistent median TE/RE gains with a moderate, not negligible, 26.9 ms mean overhead. It is deployable only with the recorded scene-level fail-closed policy; the unconditionally enabled five-scene variant slightly regresses some scene metrics.

The next optimization should keep the method sparse and map-read-only:

1. Replace per-query exact scoring over the whole projected pool with projected spatial bins plus cached normalized descriptors, then audit Top-64 parity.
2. Use adaptive K (for example 16/32/64) and stop candidate expansion once the pose-support margin is sufficient.
3. Reduce or skip the second RANSAC using a calibrated pre-solve predictor; otherwise use bounded PROSAC-style ordering and explicitly cap iterations.
4. Revisit sparse graph consistency only after the above bottlenecks are removed. Earlier fixed-threshold sparse-LGCV did not give a stable operating point.

Offline self-localization feedback remains disabled and is intentionally outside this experiment.

## Artifacts

- Protocol: `configs/v24_cambridge_high_capacity_sparse_refinement.json`
- Final aggregate: `/mnt/pool/sqy/lafgs_v24_cambridge_high_capacity_online_20260901/cambridge_five_scene_sparse_refinement_presolve_final_v2.json`
- Aggregate SHA256: `789911e6036069415c6ade163893f2c12a7ad25940fbdee5376bc84eac09d061`
