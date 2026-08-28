# V7 Full-map continuous-feedback mainline result

## Outcome

The decisive P0 diagnostic classifies the failure as
`SITUATION_B_RENDER_REAL_GAP`. The prescribed action is
`STOP_RENDERED_FEEDBACK_FOR_REAL_TEST_IMPROVEMENT`.

This is a completed gated experiment, not an unfinished Planner-v3 run. The
conditional stages were valid only under Situation A and were therefore not
activated.

## Frozen setup and protocol isolation

- Initial/deployed map: Full projective Anchor map, 200,255 Anchors, SHA-256
  `78e408cce366af8efea724ade5c35bebbf8f6edb02d62ecf40c2a4531364baed`.
- Initialization selector: disabled.
- Full Teacher to Large Student path: disabled.
- P0 input from the real-test split: camera poses, intrinsics, sizes, names, and
  indices only. Test RGB was not loaded.
- P0 status: non-formal transductive pose-distribution oracle, report-only,
  update-ineligible, and map-mutation count zero.
- Renderer records were split across GPUs 1 and 2. No v2 certificate threshold
  was lowered to fill a quota.
- The real-RGB values below reuse the already frozen once-only 530-query test
  result. No second real-test evaluation was performed.

## Render certification

| Decision | Count |
|---|---:|
| ACCEPT | 464 |
| UNCERTAIN | 64 |
| REJECT | 2 |

The 64 uncertain records comprise 61 weak mapping-camera support cases, three
marginal-valid-keypoint cases, and one depth-discontinuity-row case (one record
has multiple uncertainty reasons). The two rejects are an expected-depth
curtain mismatch and a camera outside the support envelope.

## Exact-pose comparison

| Input and subset | N | Median TE | P90 TE | Median AE | P90 AE | R5 |
|---|---:|---:|---:|---:|---:|---:|
| Render, all poses | 530 | 0.476cm | 1.177cm | 0.0000deg | 0.0382deg | 96.792% |
| Render, v2 ACCEPT | 464 | 0.465cm | 1.133cm | 0.0000deg | 0.0389deg | 96.767% |
| Real RGB, all poses | 530 | 4.021cm | 12.620cm | 0.1307deg | 0.4244deg | 61.887% |
| Real RGB, same ACCEPT poses | 464 | 4.179cm | 13.278cm | 0.1311deg | 0.4500deg | 60.345% |

On the same 464 poses, the median translation error is 8.98 times larger for
real RGB, the median absolute translation gap is 3.714cm, and R5 falls by 36.42
percentage points. The render and real R5-failure sets intersect in only 15
queries. These paired results establish an observational render--real gap but
do not by themselves distinguish content, detector, descriptor, or geometric
causes. The follow-up P0.5 causal diagnostic performs that decomposition.

The viewpoint distribution is not completely harmless: all 530 renders contain
17 R5 failures and 11 catastrophic cases. It contributes a sparse tail, but
does not explain the real-RGB median or the 202 real-RGB R5 failures.

## Preregistered decision

Situation B required at least 256 ACCEPT queries, render median TE no greater
than 1cm, and render R5 at least 95%. P0 has 464 ACCEPT queries, 0.465cm median
TE, and 96.767% R5, so every conjunct passes.

Consequently:

- Planner v3's 60/25/15 pose allocation remains locked.
- The continuous observer/headroom controller remains locked.
- Descriptor updates and reverse pruning remain locked.
- Fresh confirmation and round two are not run.
- The Full map remains unchanged.

The evidence does not justify another rendered-query closed-loop iteration for
the goal of improving real imagery. The follow-up P0.5 result identifies both
content-correspondence contamination and a remaining shared-content
detector/descriptor gap, while rejecting fixed geometry and the existing mask
as primary explanations. A future branch must address these mechanisms and
define a new leakage-safe validation protocol before using any real
target-domain RGB.

## Reproducibility artifacts

- Preregistration: `configs/v7_full_continuous_mainline.yaml`
- Machine-readable report:
  `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827/StMarysChurch/p0_test_pose_render_diagnostic_report.json`
- Render shard manifests:
  `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827/StMarysChurch/p0_test_pose_render_shard0_gpu1/manifest.json`
  and
  `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827/StMarysChurch/p0_test_pose_render_shard1_gpu2/manifest.json`.
- Localization shard manifests:
  `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827/StMarysChurch/p0_test_pose_localize_fast_shard0_gpu1/manifest.json`
  and
  `/mnt/pool/sqy/lafgs_v7_full_continuous_mainline_20260827/StMarysChurch/p0_test_pose_localize_fast_shard1_gpu2/manifest.json`.
