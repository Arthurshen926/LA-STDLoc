# LA_update19: Pseudo-Query Source-by-Stage Cross Diagnostics

## Why this update was needed

LA_update18 added source and teacher-stage marginals, but marginals are not enough. They show that
training samples both `train_rgb` and `synthetic_rgb`, and they show which teacher stages are
common, but they do not answer whether bad stages are concentrated in synthetic views, real train
views, or both.

This update adds source-by-stage cross diagnostics so the next 100/500-step runs can directly
measure where weak teacher episodes come from.

## Code changes

- `train_locaware.py`
  - `_pseudo_query_stage_source_diagnostics(...)` now also emits cross one-hot TensorBoard tags:

```text
train_diagnostics/pseudo_query_source_stage_<source>_<stage>
```

  - Supported sources:
    - `train_rgb`
    - `synthetic_rgb`
    - `other`
  - Supported stages:
    - `teacher_ok`
    - `dense_improves_sparse`
    - `mixed_or_uncertain`
    - `dense_rescues_sparse`
    - `sparse_failure`
    - `dense_regression_after_good_sparse`
    - `unknown`

- `tests/test_train_locaware_masks.py`
  - Added `test_pseudo_query_stage_source_diagnostics_include_cross_tab_terms`.

## Verification

TDD check:

- RED: the new cross-tab test failed before implementation because no
  `pseudo_query_source_stage_*` keys existed.
- GREEN: the new test passed after implementation.

Targeted test:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_are_one_hot \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_defaults_missing_stage_to_unknown \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_include_cross_tab_terms
```

Result: `Ran 3 tests ... OK`.

Related suite:

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
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_stage_source_diagnostics_include_cross_tab_terms \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector
```

Result: `Ran 19 tests ... OK`.

Static checks:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py train_detector.py localization_training/episode_sampler.py \
  scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py

bash -n scripts/run_la_pseudo_query_pipeline.sh
git diff --check -- train_locaware.py tests/test_train_locaware_masks.py \
  LA_update18_stage_source_diagnostics_and_500step.md
```

Result: all exited with code 0.

## 10-step smoke

Command pattern:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630 \
SCENES=<scene> GPU=<0-or-1> LA_ADAPT_STEPS=10 TRAIN_SEED=119 FORCE_TRAIN_COPY=1 \
RUN_PSEUDO_QUERY_MANIFEST=0 RUN_TEACHER_CACHE=0 RUN_PSEUDO_QUERY_GATE=0 \
RUN_PSEUDO_QUERY_SELECT=0 RUN_LA_FRONTEND_REFRESH=0 RUN_EVAL=0 \
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct \
scripts/run_la_pseudo_query_pipeline.sh
```

Outputs:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/ShopFacade/student_10step_seed119
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/OldHospital/student_10step_seed119
```

Both runs completed and saved `iteration_30010`.

Source-by-stage cross table from TensorBoard:

| Scene | Source | teacher_ok | dense_improves_sparse | mixed_or_uncertain | dense_rescues_sparse | sparse_failure | dense_regression_after_good_sparse | unknown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | train_rgb | 6 | 0 | 2 | 0 | 0 | 0 | 0 |
| ShopFacade | synthetic_rgb | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| OldHospital | train_rgb | 2 | 0 | 6 | 0 | 0 | 0 | 0 |
| OldHospital | synthetic_rgb | 0 | 0 | 0 | 0 | 2 | 0 | 0 |

All 21 cross terms were present for both scenes; `missing_cross_terms=0`.

## Interpretation

This is only a 10-step smoke, so it is not a distributional conclusion. It is still valuable:

- The training path, not only the helper, writes source-by-stage diagnostics.
- The first smoke already shows a plausible failure mode: both sampled OldHospital synthetic
  records were `sparse_failure`.
- The next 100/500-step runs can now answer whether synthetic records are consistently weaker, or
  whether OldHospital real train records also dominate the bad-stage mass.

## Next steps

1. Run 100-step source/stage cross-tab runs for ShopFacade and OldHospital.
2. If OldHospital synthetic is strongly concentrated in `sparse_failure`, isolate a
   `train_rgb`-only 100/500-step ablation before changing the objective again.
3. If real train also dominates `mixed_or_uncertain`, the issue is not synthetic quality alone and
   should be treated as a teacher/sparse-stage supervision problem.
4. Use the cross-tab to decide whether the next mainline refactor should focus on source-balanced
   sampling, stage-specific objective design, or synthetic QA.

