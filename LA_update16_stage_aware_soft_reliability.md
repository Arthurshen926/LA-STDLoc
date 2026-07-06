# LA-STDLoc Stage-Aware Soft Reliability Update

Date: 2026-06-30

## Purpose

`LA_update15_training_mainline_refactor_results.md` showed that the refactored mainline was valid
but still weak: ShopFacade had modest positive support, while OldHospital remained unresolved.

The main diagnosis was that the 500-step refactor run used all pseudo-query records almost equally.
The teacher cache contained stage/error/inlier/support information, but the pipeline default still
left pseudo-query reliability disabled:

```text
PSEUDO_QUERY_RELIABILITY_MODE=none
PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none
```

This update makes stage-aware soft reliability the default training-mainline behavior.

## Implementation

### Default Pipeline Behavior

`scripts/run_la_pseudo_query_pipeline.sh` now defaults to:

```text
PSEUDO_QUERY_RELIABILITY_MODE=soft
PSEUDO_QUERY_RELIABILITY_LOSS_MODE=soft
```

The old equal-weight behavior remains available as an explicit ablation:

```text
PSEUDO_QUERY_RELIABILITY_MODE=none
PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none
```

### Loss Scaling Helper

`train_locaware.py` now routes pseudo-query loss scaling through:

```python
_scale_loc_loss_by_pseudo_reliability(loc_loss, pseudo_query_reliability, args)
```

This keeps the behavior testable instead of burying it inside the training loop.

The helper only scales the localization loss when:

- `pseudo_query_reliability_loss_mode == "soft"`,
- the computed reliability weight is below 1.0.

## Current Data-Flow Audit

The current pseudo-query pipeline still trains with:

```text
--loc_teacher direct
--query_mode mixed
--mixed_sparse_probability 1.0
```

Important consequence:

- The direct teacher uses the pseudo-query RGB-derived feature map and GT pose.
- It does not use the cached sparse or dense pose as a geometric initialization target.
- Therefore the teacher cache currently contributes only reliability signals:
  stage weight, pose-error weight, inlier weight, support weight, and memory/stat update gating.

This explains why the previous 500-step "no gate, all records" run did not benefit from the rich
OldHospital teacher-stage distribution. The cache was present, but its stage information was not
active in the default loss.

## Reliability Weight Distribution On Existing Full Cache

The existing full cache was reused:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/*/pseudo_query/pseudo_teacher_cache.pt
```

### ShopFacade

| Group | Count | Mean weight | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| all | 247 | 0.814 | 0.250 | 1.000 |
| train_rgb | 231 | 0.829 | 0.500 | 1.000 |
| synthetic_rgb | 16 | 0.589 | 0.250 | 0.981 |

Stage means:

| Stage | Count | Mean weight |
| --- | ---: | ---: |
| teacher_ok | 191 | 0.900 |
| mixed_or_uncertain | 48 | 0.543 |
| sparse_failure | 6 | 0.375 |
| dense_rescues_sparse | 1 | 0.250 |
| dense_improves_sparse | 1 | 0.470 |

### OldHospital

| Group | Count | Mean weight | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| all | 911 | 0.650 | 0.250 | 1.000 |
| train_rgb | 895 | 0.656 | 0.500 | 1.000 |
| synthetic_rgb | 16 | 0.334 | 0.250 | 0.607 |

Stage means:

| Stage | Count | Mean weight |
| --- | ---: | ---: |
| teacher_ok | 104 | 0.920 |
| dense_improves_sparse | 136 | 0.838 |
| mixed_or_uncertain | 331 | 0.642 |
| dense_rescues_sparse | 239 | 0.512 |
| sparse_failure | 89 | 0.473 |
| dense_regression_after_good_sparse | 12 | 0.482 |

Interpretation:

- The soft weights do not silently delete difficult records.
- They reduce update pressure from noisy stages, especially synthetic and sparse-failure records.
- OldHospital remains dominated by non-clean stages, so soft weighting alone is a partial fix.

## 100-Step Smoke

To isolate this change, the run reused the existing pseudo manifest/cache and did not rebuild
synthetic renders or teacher cache:

```text
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630
LA_ADAPT_STEPS=100
TRAIN_SEED=101
RUN_PSEUDO_QUERY_MANIFEST=0
RUN_TEACHER_CACHE=0
RUN_PSEUDO_QUERY_GATE=0
RUN_PSEUDO_QUERY_SELECT=0
RUN_LA_FRONTEND_REFRESH=0
RUN_EVAL=1
PSEUDO_QUERY_RELIABILITY_MODE=soft
PSEUDO_QUERY_RELIABILITY_LOSS_MODE=soft
```

### Results

| Scene | Config | Median TE cm | Median AE deg | R5 cm/5deg | R2 cm/2deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | clean default 100 | 3.1756 | 0.1551 | 72.82 | n/a | 403.97 |
| ShopFacade | soft reliability 100, seed101 | 3.2660 | 0.1594 | 73.79 | 26.21 | 418.40 |
| ShopFacade | refactor 500, equal-weight | 3.2226 | 0.1560 | 76.70 | 26.21 | 477.18 |
| OldHospital | baseline | 18.3941 | 0.3380 | 3.30 | n/a | 274.81 |
| OldHospital | clean default 100 + frontend refresh | 18.7609 | 0.3472 | 6.04 | n/a | 162.20 |
| OldHospital | refactor 500, equal-weight | 19.8546 | 0.3519 | 4.40 | 0.00 | 269.37 |
| OldHospital | soft reliability 100, seed101 | 18.3385 | 0.3468 | 2.75 | 0.55 | 275.16 |

Artifacts:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/ShopFacade/student_100step_seed101
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/OldHospital/student_100step_seed101
results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_ShopFacade_student_100step_seed101-20260630_085750
results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_OldHospital_student_100step_seed101-20260630_090026
```

## Conclusion

Soft reliability is the correct default mainline behavior because it uses teacher-cache evidence
without reverting to a hard teacher-ok gate. The 100-step smoke confirms it is wired correctly and
does not destabilize training.

However, it is not sufficient:

- ShopFacade gets a small 100-step recall/inlier improvement over the earlier clean-default 100-step
  reference, but not enough to outperform the 500-step equal-weight run.
- OldHospital improves median TE and inliers relative to several references, but high-precision
  5cm recall is worse than the previous clean-default frontend-refresh run and worse than the
  500-step equal-weight run.

The updated conclusion is sharper:

- The previous failure was partly an implementation/default issue because stage-aware cache signals
  were disabled.
- After enabling them, the remaining problem is not simply sample quality or hard gating.
- The main missing piece is the student objective: direct teacher training still treats hard
  teacher-stage records mostly as weaker direct-supervision examples, not as distinct supervision
  types.

## Next Required Refactor

The next step should split pseudo-query supervision by stage instead of only scaling it:

1. `teacher_ok` and `dense_improves_sparse`
   - Keep as positive direct map adaptation.

2. `dense_rescues_sparse`
   - Do not only downweight.
   - Use it as a correction/stability signal: sparse pose failed but dense teacher found a better pose.

3. `sparse_failure`
   - Avoid strong direct map updates.
   - Use for uncertainty/negative mining or skip memory/stat updates.

4. `mixed_or_uncertain`
   - Use as weak supervision with conservative memory update.

This requires adding a stage-aware training objective, not just another scalar reliability weight.

## Verification

Passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
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
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py train_detector.py localization_training/episode_sampler.py \
  scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py

bash -n scripts/run_la_pseudo_query_pipeline.sh

git diff --check -- \
  train_locaware.py scripts/run_la_pseudo_query_pipeline.sh \
  tests/test_train_locaware_masks.py tests/test_full_script_args.py \
  LA_update15_training_mainline_refactor_results.md
```
