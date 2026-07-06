# LA-STDLoc Clean Default Mainline Validation

Date: 2026-06-30

## Purpose

This update records the first validation after the training-mainline boundary cleanup in
`LA_update12_training_mainline_boundaries.md`.

The goal was not to add another artifact or teacher-cache heuristic. The goal was to verify that
the default pseudo-query training path is now clean:

- no teacher hard gate,
- no teacher-cache reliability weighting,
- no pseudo-query selector ranking in the default path,
- all accepted `train_rgb + synthetic_rgb` records are available to training,
- no-reference region/support weighting remains the only artifact-aware default signal for synthetic RGB.

## Code State Verified

The current default path has the following boundaries:

1. Teacher-cache reliability code is isolated in `la_artifacts/pseudo_query_training.py`.
2. `train_locaware.py` defaults `pseudo_query_reliability_mode=none`.
3. Missing pseudo teacher cache is optional unless cache-dependent options are explicitly enabled.
4. `scripts/run_la_pseudo_query_pipeline.sh` defaults `PSEUDO_QUERY_RELIABILITY_MODE=none`.
5. Cache diagnostics can still be produced and inspected, but they do not silently control student training.

## Tests

The LA-related regression suite passed before launching the validation runs:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_la_artifacts tests.test_train_locaware_masks tests.test_full_script_args \
  tests.test_no_reference_valid_mask tests.test_support_sparse_pnp \
  tests.test_pseudo_query_ab tests.test_stdloc_config_paths
```

Result: 157 tests passed.

## Clean-Default 100-Step Runs

Experiment root:

```text
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630
```

Both runs used:

- `RUN_PSEUDO_QUERY_MANIFEST=0`
- `RUN_TEACHER_CACHE=0`
- `RUN_PSEUDO_QUERY_GATE=0`
- `RUN_PSEUDO_QUERY_SELECT=0`
- `RUN_TRAIN=1`
- `RUN_LA_FRONTEND_REFRESH=1`
- `RUN_EVAL=1`
- `PSEUDO_QUERY_RELIABILITY_MODE=none`
- `PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none`

### Results

| Scene | Run | Median TE cm | Median AE deg | R5 cm/5deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: |
| ShopFacade | clean default 100 | 3.1756 | 0.1551 | 72.82 | 403.97 |
| OldHospital | clean default 100 | 18.7609 | 0.3472 | 6.04 | 162.20 |

Saved summaries:

```text
/root/STDLoc/results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_mainline_clean_100_20260630_ShopFacade_student_100step_seed0-20260630_065804/summary.json
/root/STDLoc/results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_mainline_clean_100_20260630_OldHospital_student_100step_seed0-20260630_070340/summary.json
```

## Comparison With Recent Ablations

| Scene | Config | Median TE cm | Median AE deg | R5 cm/5deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: |
| ShopFacade | previous no-reliability 100 | 3.2167 | 0.1553 | 72.82 | 403.88 |
| ShopFacade | clean default 100 | 3.1756 | 0.1551 | 72.82 | 403.97 |
| ShopFacade | reliability soft-loss 100 | 3.5284 | 0.1673 | 72.82 | 392.90 |
| ShopFacade | reliability stats-only 100 | 3.4736 | 0.1847 | 67.96 | 392.16 |
| OldHospital | previous no-reliability 100 | 19.6867 | 0.3460 | 6.04 | 162.32 |
| OldHospital | clean default 100 | 18.7609 | 0.3472 | 6.04 | 162.20 |
| OldHospital | reliability soft-loss 100 | 31.9810 | 0.5490 | 3.85 | 111.29 |
| OldHospital | reliability stats-only 100 | 31.9715 | 0.5232 | 2.75 | 112.46 |

## Interpretation

The clean-default refactor did what it was supposed to do:

- ShopFacade remains positive and essentially reproduces the previous no-reliability mainline.
- OldHospital no longer has the large degradation introduced by reliability weighting, but it is still not solved.
- Teacher-cache reliability and selector-style hard decisions should stay outside the default mainline.

The remaining OldHospital issue is not explained by missing teacher gating. The official sparse-only evaluation shows many frames with low inlier counts and large translation errors, especially in difficult subsequences. This points to the next main bottleneck:

- sparse matching quality,
- landmark detector refresh quality,
- scene-specific landmark distribution,
- how synthetic/no-reference support masks affect feature selection and dense render use.

## Current Method Flow After Refactor

1. Build pseudo-query manifest from all real train RGB plus accepted synthetic RGB.
2. Synthetic RGB currently comes from the RGB teacher backend, with MAtCha preferred after the WildGaussians quality issues.
3. No-reference support/region masks are computed for synthetic renders.
4. Training consumes the manifest directly; samples are not excluded by teacher-cache success by default.
5. `train_locaware.py` trains the localization-aware student map/front-end from RGB-derived features.
6. LA frontend refresh is run after LA-map training.
7. Official sparse-only STDLoc evaluation is run on the formal test split.

Teacher cache remains useful as diagnostics and as an explicit ablation input, but not as default supervision or hard filtering.

## Next Work

Do not promote OldHospital to long 500-step as a success case yet. The next useful work is a smaller, controlled sparse-stage diagnosis:

1. Compare baseline vs clean-default OldHospital per-frame inliers and pose errors.
2. Trace whether bad frames come from detector refresh, descriptor matching, landmark visibility, or PnP/RANSAC robustness.
3. Verify no-reference support masks are improving keypoint/match quality instead of merely reducing usable evidence.
4. Run a ShopFacade 500-step clean-default rerun only as a stability check; it already has positive 100-step and previous 500-step support.
5. For OldHospital, test targeted sparse-stage fixes before spending more GPU time on long runs.

## Sparse-Stage Delta Diagnosis

To make the next diagnosis repeatable, this update also added:

- `localization_training.eval_analysis.paired_sparse_stage_summary`
- `localization_training.eval_analysis.paired_sparse_stage_rows`
- `scripts/diagnose_sparse_stage_delta.py`

The script compares two `results.json` files by `image_name` and writes:

- paired TE/AE/inlier deltas,
- recall gain/loss counts,
- sequence-level groups,
- top TE degradations,
- top inlier drops,
- optional per-frame CSV for visual inspection.

Important implementation note: if one result file lacks optional sparse-stage fields such as `matches` or
`detected_keypoints`, the metric is reported as missing rather than as zero. This avoids fake deltas such as
`delta_matches=2048` when comparing old baseline results with newer LA runs.

### Diagnosis Verification

Commands passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_eval_analysis

/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  localization_training/eval_analysis.py scripts/diagnose_sparse_stage_delta.py
```

### Baseline vs Clean-Default Sparse Diagnosis

Outputs:

```text
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/ShopFacade/diagnostics/sparse_stage_delta_vs_baseline.json
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/ShopFacade/diagnostics/sparse_stage_delta_vs_baseline.csv
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/OldHospital/diagnostics/sparse_stage_delta_vs_baseline.json
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/OldHospital/diagnostics/sparse_stage_delta_vs_baseline.csv
```

ShopFacade diagnosis:

| Metric | Value |
| --- | ---: |
| median TE delta | -0.0263 cm |
| mean TE delta | -0.3854 cm |
| median inlier delta | +31.0 |
| mean inlier delta | +15.86 |
| 5cm recall gain/loss | 10 / 10 |
| inlier-drop frames, threshold 50 | 18 |

OldHospital diagnosis:

| Metric | Value |
| --- | ---: |
| median TE delta | -1.2989 cm |
| mean TE delta | +38.8380 cm |
| median inlier delta | -109.5 |
| mean inlier delta | -112.61 |
| 5cm recall gain/loss | 11 / 6 |
| inlier-drop frames, threshold 50 | 166 |
| pose-degraded plus inlier-drop frames | 76 |
| pose-improved despite inlier-drop frames | 90 |

OldHospital sequence split:

| Sequence | Count | Baseline median TE | Candidate median TE | Delta median TE | Delta mean inliers | Candidate avg inliers | Recall gain/loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seq4 | 56 | 14.24 | 18.33 | +3.13 | -102.05 | 126.89 | 4 / 6 |
| seq8 | 126 | 20.84 | 19.76 | -2.77 | -117.30 | 177.89 | 7 / 0 |

Worst OldHospital TE degradations:

| Image | Baseline TE | Candidate TE | Delta TE | Baseline inliers | Candidate inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seq4/frame00054.png` | 60.85 | 5225.55 | +5164.70 | 139 | 36 |
| `seq8/frame00010.png` | 57.58 | 570.67 | +513.09 | 106 | 47 |
| `seq4/frame00038.png` | 33.71 | 516.12 | +482.41 | 129 | 55 |
| `seq8/frame00009.png` | 78.84 | 553.86 | +475.03 | 127 | 41 |
| `seq8/frame00011.png` | 132.98 | 481.22 | +348.24 | 104 | 43 |

## Updated Interpretation

The current clean-default mainline has positive support on ShopFacade and no longer suffers from the
teacher-cache reliability regression. However, OldHospital exposes a real sparse-stage problem:

- the median TE delta is slightly favorable, so the method is not uniformly worse;
- the average TE is dominated by a few severe pose failures;
- the candidate inlier count is much lower on almost every OldHospital frame;
- seq4 is the most problematic split because it loses recall and has a positive median TE delta;
- seq8 gains recall but still loses many inliers, so the frontend is trading evidence quantity for some pose-quality gains.

This suggests the next fix should not be another global pseudo-query filter. The likely bottleneck is the interaction between
LA-map/front-end refresh and sparse matching in scenes with weaker texture or broader viewpoint/appearance changes.

The next controlled ablations should be:

1. Refresh-off ablation: train LA-map but evaluate with the original detector/front-end.
2. Detector-refresh-only ablation: keep the map unchanged and refresh the detector to isolate frontend damage.
3. Keypoint budget ablation on OldHospital: test whether fixed 2048 candidate keypoints is too restrictive after refresh.
4. Support-mask ablation on sparse matching and dense render separately, rather than a single shared switch.
5. Per-sequence OldHospital diagnosis for synthetic/train pseudo-query composition to see whether seq4-specific geometry is underrepresented.
