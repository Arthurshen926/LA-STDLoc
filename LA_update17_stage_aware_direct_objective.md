# LA_update17: Stage-Aware Direct Objective Mainline

## Why this update was needed

LA_update16 made pseudo-query reliability soft instead of hard-gated, but the teacher cache still
only acted as a scalar multiplier on the final localization loss. That was too coarse:

- `teacher_ok` and `dense_improves_sparse` should remain strong positive direct-map supervision.
- `dense_rescues_sparse` should not be discarded; it should keep contrastive and multiview signal
  while reducing direct descriptor pressure.
- `sparse_failure` should not strongly update direct descriptors or multiview memory, but can still
  provide hard-negative/full-bank and anchor regularization.
- `mixed_or_uncertain` should be weak positive supervision instead of an all-or-nothing sample.

The implementation now uses the teacher stage to change the internal direct-objective composition,
not only the outer sample weight.

## Code changes

- `train_locaware.py`
  - Added `_pseudo_query_stage_direct_loss_policy(...)`.
  - Added `_compose_direct_loc_loss(...)`.
  - Added `--pseudo_query_stage_objective_mode {none,direct}`.
  - A non-`none` stage objective now requires a teacher cache, because stage labels come from cache.
  - Direct teacher multiview-memory updates now use the stage-aware policy.
  - Direct loss now applies per-component stage weights before the outer reliability loss scale.
  - TensorBoard diagnostics now include:
    - `pseudo_query_stage_objective_enabled`
    - `pseudo_query_stage_objective_desc_weight`
    - `pseudo_query_stage_objective_multiview_weight`
    - `pseudo_query_stage_objective_full_bank_weight`
    - `pseudo_query_stage_objective_anchor_weight`
    - `pseudo_query_stage_objective_update_memory`
    - `pseudo_query_stage_objective_update_stats`

- `scripts/run_la_pseudo_query_pipeline.sh`
  - Default mainline now sets `PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct`.
  - The script passes `--pseudo_query_stage_objective_mode`.

- Tests
  - Added coverage for stage policy behavior.
  - Added coverage for stage-aware direct loss composition.
  - Added coverage that stage objective requires teacher cache.
  - Updated pseudo-query pipeline script assertions.

## Stage policy v1

| Teacher cache stage | desc | multiview | full-bank | anchor | memory/stats update |
|---|---:|---:|---:|---:|---|
| `teacher_ok` | 1.00 | 1.00 | 1.00 | 1.00 | reliability-gated |
| `dense_improves_sparse` | 1.00 | 1.00 | 1.00 | 1.00 | reliability-gated |
| `mixed_or_uncertain` | 0.70 | 0.70 | 0.50 | 1.00 | reliability-gated |
| `dense_rescues_sparse` | 0.55 | 1.00 | 0.85 | 1.00 | reliability-gated |
| `sparse_failure` | 0.25 | 0.00 | 0.75 | 1.00 | disabled |
| `dense_regression_after_good_sparse` | 0.35 | 0.25 | 0.50 | 1.00 | disabled |
| `unknown` | 0.60 | 0.50 | 0.50 | 1.00 | reliability-gated |

The outer soft reliability multiplier still applies after this per-component composition.

## Verification

Targeted tests:

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
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector
```

Result: `Ran 16 tests ... OK`.

Static checks:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py train_detector.py localization_training/episode_sampler.py \
  scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py

bash -n scripts/run_la_pseudo_query_pipeline.sh
git diff --check -- train_locaware.py tests/test_train_locaware_masks.py \
  tests/test_full_script_args.py scripts/run_la_pseudo_query_pipeline.sh
```

Result: all exited with code 0.

## Smoke runs

Both smoke runs reused the existing pseudo-query manifest and teacher cache under:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/{ShopFacade,OldHospital}/pseudo_query
```

Command pattern:

```bash
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630 \
SCENES=<scene> GPU=<0-or-1> LA_ADAPT_STEPS=10 TRAIN_SEED=117 FORCE_TRAIN_COPY=1 \
RUN_PSEUDO_QUERY_MANIFEST=0 RUN_TEACHER_CACHE=0 RUN_PSEUDO_QUERY_GATE=0 \
RUN_PSEUDO_QUERY_SELECT=0 RUN_LA_FRONTEND_REFRESH=0 RUN_EVAL=0 \
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct \
scripts/run_la_pseudo_query_pipeline.sh
```

Outputs:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/ShopFacade/student_10step_seed117
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/OldHospital/student_10step_seed117
```

Both completed and wrote `iteration_30010` point clouds and `chkpnt_locaware_30010.pth`.

10-step stage-policy diagnostic histograms:

| Scene | desc weights | full-bank weights | update memory |
|---|---|---|---|
| ShopFacade | `{0.25: 1, 0.7: 1, 1.0: 8}` | `{0.5: 1, 0.75: 1, 1.0: 8}` | `{0.0: 3, 1.0: 7}` |
| OldHospital | `{0.25: 4, 0.35: 1, 0.55: 3, 0.7: 2}` | `{0.5: 3, 0.75: 4, 0.85: 3}` | `{0.0: 10}` |

## 100-step official sparse-only validation

Command pattern:

```bash
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630 \
SCENES=<scene> GPU=<0-or-1> LA_ADAPT_STEPS=100 TRAIN_SEED=118 FORCE_TRAIN_COPY=1 \
RUN_PSEUDO_QUERY_MANIFEST=0 RUN_TEACHER_CACHE=0 RUN_PSEUDO_QUERY_GATE=0 \
RUN_PSEUDO_QUERY_SELECT=0 RUN_LA_FRONTEND_REFRESH=0 RUN_EVAL=1 \
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct \
scripts/run_la_pseudo_query_pipeline.sh
```

Results:

| Scene | Run | Median TE cm | Median AE deg | R5cm/5deg % | R2cm/2deg % | Avg inliers |
|---|---|---:|---:|---:|---:|---:|
| ShopFacade | stage-aware direct 100, seed118 | 3.1019 | 0.1582 | 72.82 | 23.30 | 415.83 |
| OldHospital | stage-aware direct 100, seed118 | 18.4817 | 0.3683 | 2.20 | 1.10 | 276.54 |

Result paths:

```text
results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_ShopFacade_student_100step_seed118-20260630_092835
results/pseudo-query-30100-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_OldHospital_student_100step_seed118-20260630_093108
```

100-step stage-policy diagnostic histograms:

| Scene | desc weights | full-bank weights | update memory |
|---|---|---|---|
| ShopFacade | `{0.25: 5, 0.55: 1, 0.7: 23, 1.0: 71}` | `{0.5: 23, 0.75: 5, 0.85: 1, 1.0: 71}` | `{0.0: 42, 1.0: 58}` |
| OldHospital | `{0.25: 34, 0.55: 21, 0.7: 26, 1.0: 19}` | `{0.5: 26, 0.75: 34, 0.85: 21, 1.0: 19}` | `{0.0: 86, 1.0: 14}` |

## Interpretation

Compared with LA_update16's 100-step soft-reliability run, this is not a perfect seed-matched
comparison, but it is useful as a sanity check:

| Scene | Previous 100-step reference | Median TE cm | R5cm/5deg % | Avg inliers |
|---|---|---:|---:|---:|
| ShopFacade | soft reliability 100, seed101 | 3.2660 | 73.79 | 418.40 |
| ShopFacade | stage-aware direct 100, seed118 | 3.1019 | 72.82 | 415.83 |
| OldHospital | soft reliability 100, seed101 | 18.3385 | 2.75 | 275.16 |
| OldHospital | stage-aware direct 100, seed118 | 18.4817 | 2.20 | 276.54 |

Conclusions:

- The new mainline is wired correctly and no longer treats teacher stages as a single scalar.
- ShopFacade gets a better median TE than the previous soft-only 100-step reference, while recall
  and inliers are roughly similar.
- OldHospital remains unresolved. The stage histogram shows why: most sampled episodes are
  sparse-failure, dense-rescue, or mixed, so only 14/100 iterations update direct memory/stats.
- This supports the architectural suspicion: the main bottleneck on OldHospital is not simply the
  student loss scalar. The pseudo-query teacher-stage distribution and map/query geometry remain
  limiting factors.

## Remaining open items

1. Run seed-matched `seed101` stage-aware direct 100-step if we need a stricter LA_update16 A/B.
2. Run 500-step stage-aware direct on ShopFacade and OldHospital.
3. Split student training diagnostics by pseudo-query source and teacher stage, not just by scalar
   weights.
4. Revisit OldHospital pseudo-query generation/cache quality, because the current distribution
   heavily suppresses memory/stat updates.
5. Consider a second-stage objective for dense-rescue records that explicitly learns sparse-stage
   recovery instead of only downweighting descriptor pressure.
