# LA_update18: Source/Stage Diagnostics and 500-Step Stage-Aware Validation

## Why this update was needed

LA_update17 made pseudo-query supervision stage-aware, but the training logs still did not show
which pseudo-query source and teacher-stage records were actually sampled during student training.
That made it too easy to misread a result as a method failure when the real issue might be a bad
episode distribution.

This update adds explicit source/stage diagnostics and runs 500-step official sparse-only
validation for the current stage-aware direct mainline.

## Code changes

- `train_locaware.py`
  - Added `_pseudo_query_stage_source_diagnostics(...)`.
  - Training diagnostics now log one-hot source marginals:
    - `pseudo_query_source_train_rgb`
    - `pseudo_query_source_synthetic_rgb`
    - `pseudo_query_source_other`
  - Training diagnostics now log one-hot teacher-stage marginals:
    - `pseudo_query_stage_teacher_ok`
    - `pseudo_query_stage_dense_improves_sparse`
    - `pseudo_query_stage_mixed_or_uncertain`
    - `pseudo_query_stage_dense_rescues_sparse`
    - `pseudo_query_stage_sparse_failure`
    - `pseudo_query_stage_dense_regression_after_good_sparse`
    - `pseudo_query_stage_unknown`

- Tests
  - Added coverage that source and stage diagnostics are one-hot.
  - Added coverage that missing teacher-stage metadata defaults to `unknown`.

## Verification

TDD check:

- RED: new tests failed with `ImportError` before implementation.
- GREEN: both new tests passed after implementation.

Targeted suite:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_empty_detector_landmark_sample_fails_before_training \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_nonempty_detector_landmark_sample_is_returned_as_long_tensor \
  tests.test_episode_sampler.EpisodeSamplerTest.test_sparse_mode_uses_stage_failed_cache_by_default \
  tests.test_episode_sampler.EpisodeSamplerTest.test_sparse_mode_can_opt_in_to_reject_cache_stage_sparse_failure \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_defaults_to_no_pseudo_query_stage_gate \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_pseudo_teacher_cache_is_optional_for_default_mainline \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_required_pseudo_teacher_cache_raises_clear_error \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_sparse_init_requires_teacher_cache \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_soft_pseudo_query_reliability_scales_loc_loss \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_disabled_pseudo_query_reliability_keeps_loc_loss_unscaled \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_softly_downweights_bad_teacher_cache \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_none_keeps_mainline_unweighted \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_stage_aware_direct_policy_treats_failure_modes_differently \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_stage_aware_direct_policy_scales_loss_components_before_reliability \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_objective_requires_teacher_cache \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_are_one_hot \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_defaults_missing_stage_to_unknown \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector
```

Result: `Ran 18 tests ... OK`.

Static checks:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py train_detector.py localization_training/episode_sampler.py \
  scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py

bash -n scripts/run_la_pseudo_query_pipeline.sh
git diff --check -- train_locaware.py tests/test_train_locaware_masks.py \
  tests/test_full_script_args.py scripts/run_la_pseudo_query_pipeline.sh \
  LA_update17_stage_aware_direct_objective.md
```

Result: all exited with code 0.

## 500-step official sparse-only validation

Command pattern:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630 \
SCENES=<scene> GPU=<0-or-1> LA_ADAPT_STEPS=500 TRAIN_SEED=118 FORCE_TRAIN_COPY=1 \
RUN_PSEUDO_QUERY_MANIFEST=0 RUN_TEACHER_CACHE=0 RUN_PSEUDO_QUERY_GATE=0 \
RUN_PSEUDO_QUERY_SELECT=0 RUN_LA_FRONTEND_REFRESH=0 RUN_EVAL=1 \
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct \
scripts/run_la_pseudo_query_pipeline.sh
```

Result paths:

```text
results/pseudo-query-30500-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_ShopFacade_student_500step_seed118-20260630_094758
results/pseudo-query-30500-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_OldHospital_student_500step_seed118-20260630_095020
```

Official sparse-only metrics:

| Scene | Run | Median TE cm | Median AE deg | R5cm/5deg % | R2cm/2deg % | Avg inliers |
|---|---|---:|---:|---:|---:|---:|
| ShopFacade | stage-aware direct 500, seed118 | 2.9907 | 0.1581 | 72.82 | 30.10 | 478.06 |
| OldHospital | stage-aware direct 500, seed118 | 19.0208 | 0.3565 | 4.95 | 0.00 | 270.25 |

Comparison with earlier references:

| Scene | Run | Median TE cm | Median AE deg | R5cm/5deg % | R2cm/2deg % | Avg inliers |
|---|---|---:|---:|---:|---:|---:|
| ShopFacade | equal-weight 500, seed0, LA_update15 | 3.2226 | 0.1560 | 76.70 | 26.21 | 477.18 |
| ShopFacade | stage-aware direct 100, seed118, LA_update17 | 3.1019 | 0.1582 | 72.82 | 23.30 | 415.83 |
| ShopFacade | stage-aware direct 500, seed118, LA_update18 | 2.9907 | 0.1581 | 72.82 | 30.10 | 478.06 |
| OldHospital | equal-weight 500, seed0, LA_update15 | 19.8546 | 0.3519 | 4.40 | 0.00 | 269.37 |
| OldHospital | stage-aware direct 100, seed118, LA_update17 | 18.4817 | 0.3683 | 2.20 | 1.10 | 276.54 |
| OldHospital | stage-aware direct 500, seed118, LA_update18 | 19.0208 | 0.3565 | 4.95 | 0.00 | 270.25 |

## Training source/stage diagnostics

500-step source marginals:

| Scene | train_rgb | synthetic_rgb | other |
|---|---:|---:|---:|
| ShopFacade | 330 | 170 | 0 |
| OldHospital | 323 | 177 | 0 |

500-step teacher-stage marginals:

| Scene | teacher_ok | dense_improves_sparse | mixed_or_uncertain | dense_rescues_sparse | sparse_failure | dense_regression_after_good_sparse | unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | 330 | 17 | 116 | 9 | 28 | 0 | 0 |
| OldHospital | 54 | 46 | 127 | 120 | 147 | 6 | 0 |

500-step stage-objective policy histograms:

| Scene | desc weights | full-bank weights | memory/stat updates |
|---|---|---|---|
| ShopFacade | `{0.25: 28, 0.55: 9, 0.7: 116, 1.0: 347}` | `{0.5: 116, 0.75: 28, 0.85: 9, 1.0: 347}` | `{0.0: 209, 1.0: 291}` |
| OldHospital | `{0.25: 147, 0.35: 6, 0.55: 120, 0.7: 127, 1.0: 100}` | `{0.5: 133, 0.75: 147, 0.85: 120, 1.0: 100}` | `{0.0: 424, 1.0: 76}` |

These are marginal counts, not source-by-stage cross-tabs. A source-by-stage cross-tab is still
needed to prove whether synthetic records are disproportionately responsible for bad stages.

## Interpretation

Implementation-level conclusions:

- The current pseudo-query mainline is connected: 500-step training samples both `train_rgb` and
  `synthetic_rgb`, logs stage labels, and applies stage-aware direct-objective weights.
- The no-hard-gate policy is in effect. Bad-stage records are not rejected from the pool; instead
  they receive different objective composition.
- The direct-objective policy behaves as designed: sparse-failure records suppress descriptor and
  multiview memory pressure, while dense-rescue records retain more full-bank/multiview signal.

Method-level conclusions:

- ShopFacade has a real but mixed positive signal. Against LA_update15 equal-weight 500, median TE
  improves from `3.2226cm` to `2.9907cm`, R2 improves from `26.21%` to `30.10%`, and inliers stay
  essentially unchanged. R5 drops from `76.70%` to `72.82%`, so this is not a clean win across all
  metrics.
- OldHospital remains the main unresolved case. Against LA_update15 equal-weight 500, median TE
  improves from `19.8546cm` to `19.0208cm` and R5 improves from `4.40%` to `4.95%`, but R2 stays
  at `0%` and AE is slightly worse.
- The OldHospital bottleneck is now clearer: only `100/500` sampled records are strong
  `teacher_ok` or `dense_improves_sparse`, and only `76/500` iterations update memory/stats. The
  training stream is dominated by `sparse_failure`, `dense_rescues_sparse`, and
  `mixed_or_uncertain` records.

## What this means for the training-mainline refactor

The previous target could not be completed cleanly because the method was treating several
different things as one problem:

1. RGB/synthetic view generation quality.
2. No-reference valid/support mask quality.
3. Sparse teacher stability.
4. Dense teacher rescue/regression behavior.
5. Student direct-map descriptor learning.
6. Student detector/frontend learning.

The latest refactor separates one more axis: source/stage diagnostics now expose what the student
is actually seeing. The result is useful but also shows the current design is still not the final
form. OldHospital is not failing because the sample pool is empty or because synthetic records are
hard-gated out; it is failing because the available teacher episodes are weak or ambiguous, and
the current student objective does not yet learn a robust recovery policy from those episodes.

## Next steps

1. Add source-by-stage cross-tab diagnostics.
2. Split official evaluation by per-image failure cluster and compare against teacher-cache stage.
3. Train source/stage-specific ablations:
   - `train_rgb` only.
   - `synthetic_rgb` only.
   - strong-stage only.
   - all-stage current policy.
4. Rework OldHospital training objective so `dense_rescues_sparse` is not merely downweighted but
   becomes an explicit sparse-recovery supervision signal.
5. Audit whether student detector/frontend is actually being trained from pseudo-query episodes;
   if not, promote it from an auxiliary/offline component into the main objective.

