# XFeat Arm-A detector-repeatability preregistration

## Status and question

This protocol was frozen before any real Stairs Arm-A probe, detector audit,
or gate result was produced. Implementation and tests are CPU-only and
synthetic. No metric in this document was selected after observing an Arm-A
result.

The single question is whether the locked single-image XFeat detector reaches
more frozen, GT-projected, depth/alpha/mask-legal map anchors than the frozen
SuperPoint detector under the same requested keypoint budget. Descriptors,
pair matching, map rebuilding, and pose evaluation are outside this gate.

## Fixed data and implementation lineage

The Stairs inputs are fixed by path and SHA256:

- compact state:
  `/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/anchor_map_step_1520.pt`,
  `5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620`;
- fresh K1024/NMS4 query cache:
  `/mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt`,
  `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9`;
- V3 complete-positive teacher:
  `/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt`,
  `3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81`;
- XFeat checkpoint:
  `/mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt`,
  `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`.

The candidate is exactly XFeat tree
`4f804566cb1cf72469b7d7174fba9308885c5c5a`, model SHA256
`d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964`,
interpolator SHA256
`d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da`,
and wrapper SHA256
`f1b0f73c77e34381a46578866bb1531b98180e8d870c0fc61fbfdbd29ac64f31`.
The expected implementation ID is
`xfeat_tree_4f804566cb1c__model_d9a665f18fce__arm_a_v1`.

Both detectors request K=1024 before the frozen native-mask filter. The
reference cache uses its registered SuperPoint NMS radius 4. The candidate
uses XFeat-native single-image semantics: 65-way softmax with dustbin removed,
8x8 unpacking, one 5x5 NMS pass (radius 2), strict probability `>0.05`,
nearest-sampled probability times bilinear-sampled reliability, `(0,0)`
sentinel removal, stable row-major score-tie order, and top-K before native
masking. SuperPoint NMS is not reapplied to XFeat.

Every successful probe must attest detector-only capability, no materialized
descriptor rows, no pair matcher, mapping-only queries, no test queries, exact
cache/teacher paths and hashes, and per-query identity XFeat resize. For the
Stairs native 480x640 inputs, integer XFeat coordinates make the cache's
`round()` mask lookup and the consumer's `floor()` lookup identical. The probe
stores and the gate checks equal index and mask-decision hashes for every
query; any non-identity resize fails before a probe is written.

## Frozen evaluator and target universe

The evaluator must run from a clean Git worktree. Its report stores
`lafgs_frontend_detector_evaluation_code` with the exact Git commit and hashes
of:

- `map_learning/frontend_upper_bound.py`;
- `scripts/audit_frontend_upper_bound.py`; and
- `scripts/compare_frontend_detector_arm_a.py`.

The gate recomputes that identity from a clean worktree and requires exact
equality. A detector report has an exact top-level registry and is rejected if
it contains `descriptor_identity` or any other audit arm.

The target universe is unchanged for both detectors:
`frozen_map_gt_projection_depth_alpha_mask_legal`, with absolute depth
tolerance 0.05 m, relative depth tolerance 0.02, and minimum alpha 0.01. The
only reachability radii are 2, 4, and 8 pixels. Baseline and candidate target
counts must be equal per query and for `all`, `track_core`, and
`gaussian_reserve`; Track-Core plus Gaussian-Reserve must exactly reconstruct
`all`. Per-query counts and hits must add exactly to pooled counts. Reported
fractions and deltas are rederived from integer counts with absolute checking
tolerance `1e-15`.

## Frozen mechanism gate

The primary scale is 4 px: it matches the current reference detector's NMS
radius and asks whether candidate coverage improves at the localization scale
that directly limits correspondence availability. The gate is:

1. pooled `all@4px`: candidate integer hit count must be strictly greater;
2. pooled `all@2px`: candidate integer hit count must not decrease;
3. pooled `all@8px`: candidate integer hit count must not decrease;
4. pooled `track_core@4px`: candidate integer hit count must not decrease; and
5. pooled `gaussian_reserve@4px`: candidate integer hit count must not
   decrease.

All five conditions must hold. The serialized protocol also records a
non-regression absolute tolerance of `1e-12`, but integer hit-count
non-regression is additionally required and is decisive. This prevents a
floating-point tolerance from hiding the loss of even one legal target.

`GO` authorizes only a mapping-only detector rebuild followed by the existing
mapping-pose gate. It is not a pose or test-set claim. A valid `STOP` writes its
JSON decision and exits 2. Any lineage, schema, count, code, capability, NMS,
descriptor-contamination, or source-hash failure is an invalid experiment: it
raises before a gate decision is written.

The evaluator's pooled target counts, report SHA256, probe SHA256, and total
validated post-mask detector rows are filled into the gate command after
materialization. They are exact lineage checks, not result-dependent
thresholds; changing them cannot change the five frozen comparisons.

## Commands after review and merge

No command below was executed while preregistering the gate. First materialize
the reviewed detector probe:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python -m scripts.materialize_xfeat_arm_a \
  --dataset /mnt/pool/sqy/datasets/7Scenes_pgt_lafgs_v1/stairs \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --expected-query-cache-sha256 6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9 \
  --teacher /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt \
  --expected-teacher-sha256 3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81 \
  --xfeat-worktree /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean \
  --weights /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt \
  --expected-weights-sha256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b \
  --expected-parent-commit b28d53258ab4461ba1a02eaa60ef504e9b82b9ab \
  --expected-xfeat-tree 4f804566cb1cf72469b7d7174fba9308885c5c5a \
  --expected-model-sha256 d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964 \
  --expected-interpolator-sha256 d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da \
  --expected-wrapper-sha256 f1b0f73c77e34381a46578866bb1531b98180e8d870c0fc61fbfdbd29ac64f31 \
  --device cpu \
  --output /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector.pt
```

Then run the detector-only evaluator:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python -m scripts.audit_frontend_upper_bound evaluate \
  --state /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/anchor_map_step_1520.pt \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --teacher /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt \
  --probe-cache /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector.pt \
  --arm detector \
  --reachability-radii-px 2 4 8 \
  --depth-abs-tolerance-m 0.05 \
  --depth-rel-tolerance 0.02 \
  --alpha-minimum 0.01 \
  --output /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector_report.json
```

These three target-universe values must be passed explicitly. Omitting any of
them is invalid and fails before evaluation; the evaluator must not inherit a
different scene-teacher tolerance for this preregistered Arm-A report.
The earlier report that inherited the Stairs teacher value
`depth_abs_tolerance_m=0.0020965913212974` is therefore invalid for this gate
and must be replaced, not accepted by changing the preregistered threshold.

Freeze the printed/artifact SHA256 values and copy only the exact lineage
counts from the valid report into the following command:

```bash
PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python -m scripts.compare_frontend_detector_arm_a \
  --report /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector_report.json \
  --expected-report-sha256 REPORT_SHA256 \
  --state /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/anchor_map_step_1520.pt \
  --expected-state-sha256 5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620 \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --expected-query-cache-sha256 6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9 \
  --teacher /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt \
  --expected-teacher-sha256 3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81 \
  --probe-cache /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector.pt \
  --expected-probe-cache-sha256 PROBE_SHA256 \
  --candidate-weights /mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt \
  --expected-candidate-weights-sha256 0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b \
  --expected-query-count 2000 \
  --expected-validated-detector-keypoints VALIDATED_POST_MASK_ROWS \
  --expected-all-target-count ALL_TARGET_COUNT \
  --expected-track-target-count TRACK_TARGET_COUNT \
  --expected-reserve-target-count RESERVE_TARGET_COUNT \
  --output /mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/detector_arm_a_gate.json
```
