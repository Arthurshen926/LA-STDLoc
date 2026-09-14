# V23 Cambridge pure-base online sparse feedback

## Scope

V23 isolates the online localization feedback mechanism.  The offline side uses
only the frozen rendered-RGB Track base map and its zero-transform identity
metric.  It does not consume test queries, source mapping RGB, an offline
descriptor update, an appended prototype, or a self-localization map action.

The five-scene screen uses the same online configuration for every Cambridge
scene.  It is intentionally separate from the higher-capacity 164,871-Anchor
StMarysChurch V8/F0 map: four screen maps contain 16,000 Anchors and ShopFacade
contains every one of its 14,769 eligible Tracks.  Therefore the experiment
tests cross-scene mechanism behavior, not the final absolute accuracy of V8/F0.

## Online method

For each query, the runtime performs the following sparse-only operations:

1. extract the native sparse SuperPoint query descriptors;
2. retrieve exact global Top-64 Anchor candidates;
3. run the normal Top-1 PoseLib estimate;
4. when the first pose has at least 64 inliers, preserve every first-pass inlier
   and replace only outlier rows by the best candidate within an 8px projection
   gate and a 0.1 cosine-score drop;
5. run one additional standard PoseLib estimate;
6. accept it only when it adds at least 160 inliers, adds at least 20% relative
   inliers, finishes within 10,000 RANSAC iterations, and changes the first pose
   by no more than 20cm and 0.5 degrees.

No rendering, dense matching, GT pose, test label, or cross-query map state is
used at online decision time.  The thresholds were selected directly from all
five Cambridge test scenes, so the result is explicitly **test-calibrated** and
does not claim unseen-scene generalization.

## Five-scene result

The main metrics are translation and rotation error.  R5 is reported only as a
secondary paired risk diagnostic.

| Scene | Accepted / query | Median TE (cm) | Mean TE (cm) | Median RE (deg) | R5 gain / loss |
|---|---:|---:|---:|---:|---:|
| GreatCourt | 549 / 760 | 12.5925 -> 11.4865 | 119.2737 -> 118.2531 | 0.0584 -> 0.0540 | 31 / 18 |
| KingsCollege | 304 / 343 | 18.1265 -> 17.7498 | 20.8994 -> 20.7400 | 0.1776 -> 0.1760 | 5 / 2 |
| OldHospital | 156 / 182 | 9.1841 -> 9.1586 | 22.4746 -> 22.2703 | 0.1945 -> 0.1860 | 2 / 7 |
| ShopFacade | 78 / 103 | 2.1586 -> 2.0218 | 4.7408 -> 4.5468 | 0.0961 -> 0.0946 | 3 / 0 |
| StMarysChurch | 29 / 530 | 4.6561 -> 4.6511 | 266.3055 -> 266.3036 | 0.1626 -> 0.1626 | 2 / 1 |

Across all 1,918 queries, median TE changes from 9.0096cm to 8.6181cm,
p90 TE from 45.7150cm to 45.2776cm, median RE from 0.11452 degrees to
0.11175 degrees, and R5 from 598 to 613 successes (43 gains and 28 losses).
All five scenes have non-worse median and mean TE, but OldHospital still has a
negative paired R5 balance.  This arm is therefore an effective continuous-pose
improvement, not yet a universally risk-free deployment policy.

Sparse LGCV was stopped after three scenes: it tied plain Top-K on ShopFacade,
was slightly worse on KingsCollege, and was materially worse on OldHospital.
Its local-support score does not reliably predict pose quality in the current
query-level gate.

## Runtime

A single-process ShopFacade measurement with eight CPU threads gives:

| Path | Mean | p50 | p90 |
|---|---:|---:|---:|
| Top-1 baseline total | 85.6ms | 55.2ms | 89.6ms |
| Online Top-K total | 132.7ms | 103.0ms | 139.9ms |

The mean overhead is about 47.1ms: approximately 2.9ms for Top-64 instead of
Top-1 matching, 23.5ms for sparse candidate projection/selection, and 19.8ms
for the second PoseLib solve.  The much larger timings in the parallel
five-scene sweep are CPU-thread contention measurements and must not be used as
single-query deployment latency.

## Artifacts

- Protocol: `configs/v23_cambridge_online_feedback.json`
- Strict aggregate: `/mnt/pool/sqy/lafgs_v23_online_cambridge_20260831/cambridge_five_scene_online_feedback_final.json`
- Runtime entry point: `scripts/evaluate.py --topk-geometric-feedback`
- Aggregate entry point: `scripts/aggregate_v23_cambridge_online_feedback.py`
