# Stairs XFeat Arm A detector-repeatability result

## Decision

The locked same-K XFeat detector is a scientifically valid **STOP before
mapping-only detector rebuild** on Stairs. Relative to the frozen SuperPoint
detector, it reaches fewer frozen, GT-projected, depth/alpha/mask-legal map
identities at every preregistered gate measure. All five fail-closed checks
fail, so this arm does not authorize a detector map rebuild, function-graph or
metric refresh, mapping pose, or formal test run.

This is a detector-only, mapping-only result. It uses 2,000 mapping images,
requests exactly 1,024 keypoints from each detector per image, validates
2,048,000 post-mask XFeat keypoints, materializes no candidate descriptor, and
uses no test query.

## Frozen inputs and artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| Frozen V3 state | `/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/anchor_map_step_1520.pt` | `5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620` |
| Fresh K1024/NMS4 query cache | `/mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt` | `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9` |
| Complete-positive teacher | `/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/complete_positive_teacher.pt` | `3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81` |
| XFeat weights | `/mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt` | `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b` |
| XFeat Arm A probe | `/mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector.pt` | `a30bfe4974c2583bb66820b3b8394bfc83203767a575917f4a1dcdf94692601e` |
| Detector report | `/mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/xfeat64_detector_report.json` | `0a52382f11a9de547ade5d67bf574c14ad36afab6b5104678e13bf9e317b2f37` |
| Fail-closed gate | `/mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/detector_arm_a_gate.json` | `5790a1229938ac62a23beac9e705677b2d93ebb952f8ff93b23b225d2aaabc55` |

The report and gate bind clean evaluator commit
`e4e5125dabe72ecd0ce3140254fc2f7069e04094` and the exact hashes of the
evaluator, runner, and comparator entrypoints. The target universe is fixed as
`frozen_map_gt_projection_depth_alpha_mask_legal`, with the three geometry
values passed explicitly: absolute depth tolerance `0.05 m`, relative depth
tolerance `0.02`, and minimum alpha `0.01`.

The pooled target registry is identical for both detectors: 3,896,031 total
targets, split exactly into 931,876 Track Core and 2,964,155 Gaussian Reserve
targets. The reference uses its registered SuperPoint NMS radius 4. The
candidate uses the preregistered XFeat single-image detector semantics, 5x5
NMS, and strict probability threshold `>0.05`; descriptors, pair matching,
topology, and pose are absent.

## Repeatability result

| Mapping-only reachability | SuperPoint | XFeat | Delta | Hit-count delta |
|---|---:|---:|---:|---:|
| all @ 2 px | 22.75534% | 11.24213% | **-11.51320 pp** | -448,558 |
| all @ 4 px | 51.30537% | 34.45296% | **-16.85241 pp** | -656,575 |
| all @ 8 px | 77.78508% | 62.77607% | **-15.00902 pp** | -584,756 |
| Track Core @ 4 px | 64.00594% | 36.13539% | **-27.87055 pp** | -259,719 |
| Gaussian Reserve @ 4 px | 47.31254% | 33.92404% | **-13.38850 pp** | -396,856 |

The primary gate requires a strict positive integer hit-count change for
`all@4px`. The other four measures require integer non-regression. Every
condition is false, with the largest fractional loss on Track Core at 4 px.
The persisted gate is therefore `valid=true`, `decision=STOP`,
`mechanism_gate_passed=false`, and
`advance_to_mapping_only_detector_rebuild=false`.

## Execution-lineage correction

An earlier evaluator invocation inherited the Stairs teacher's absolute depth
tolerance `0.0020965913212974 m` instead of the preregistered `0.05 m`. Its
report was moved to
`/mnt/pool/sqy/lafgs_xfeat_arm_a_20260813/stairs/invalid_default_teacher_tolerance/xfeat64_detector_report_depth_abs_0.0020965913212974.json`
(SHA-256
`8574cba2db8c0606a6daaf398157b1a621b6624ef1d15400e3b04fa349a5c8a9`).
It is retained only as invalid execution evidence: it was not admitted to the
gate and is not used as a scientific result. The formal report above was
regenerated with all three preregistered geometry values explicit before the
gate was evaluated.

## Interpretation

The result rejects this locked same-budget XFeat detector substitution for the
current frozen-map identity geometry. It does not reject XFeat descriptors,
all possible detector redesigns, or the broader method. In fact, the separate
equal-energy SuperPoint+XFeat descriptor arm passes its identity gate at the
unchanged SuperPoint rows. Together, the two independent arms localize the
useful signal: XFeat contributes complementary identity representation, while
its locked detector does not preserve the repeatable sampling coverage needed
by this Stairs map.

The correct next action is therefore to keep the frozen SuperPoint detector
and advance only the already-authorized single-vector 320D descriptor factor
to its mapping-pose gate. Combining the failed detector with that descriptor,
tuning detector thresholds after this result, or running pose to rescue the
failed mechanism gate would break the single-factor protocol.
