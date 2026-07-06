# LA_update22: Cache-Aligned Pseudo-Query Training Mainline

Date: 2026-06-30

## Motivation

LA_update21 cleaned the default data source policy to train-RGB-only, but one high-impact implementation confound remained:

- the pseudo-query manifest could contain all train RGB records;
- the teacher cache could be only a small subset, especially in smoke or interrupted runs;
- `train_locaware.py` still sampled from the full manifest by default;
- samples without cache entries fell back to unknown/noisy teacher metadata.

That made "all train RGB" training less controlled than intended. It also mixed two different concepts:

- cache coverage alignment: a record is usable by this teacher-cache training path only if its cache entry exists;
- teacher quality gating: filtering by teacher-ok, stage, sparse/dense error, or failure status.

The refactor keeps these separate. Coverage alignment is default-on; quality gating remains default-off.

## Code Changes

1. `train_locaware.py`
   - Added `_pseudo_teacher_cache_get_for_record`.
   - Added `_align_pseudo_manifest_to_teacher_cache`.
   - Added `--pseudo_query_require_teacher_cache` / `--no-pseudo_query_require_teacher_cache`, default `True`.
   - If a pseudo-query manifest is provided and cache is required, missing teacher cache now raises immediately.
   - Manifest records are aligned to teacher-cache coverage before sampling.
   - Alignment drops only missing cache records. It does not filter failed, mixed, sparse-failure, or non-teacher-ok records.

2. `scripts/run_la_pseudo_query_pipeline.sh`
   - Added `PSEUDO_QUERY_REQUIRE_TEACHER_CACHE`, default `1`.
   - Passes the corresponding train flag explicitly.
   - This preserves strict default behavior while still allowing explicit exploratory runs without cache.
   - Added explicit `LA_TRAIN_MODE=adapt|scratch`.
   - `adapt` keeps the previous behavior: copy/load the 30k STDLoc baseline map, then run LA adaptation for `LA_ADAPT_STEPS`.
   - `scratch` no longer silently reuses baseline detector landmarks. It requires `LA_LANDMARK_PATH` and, for official eval, `RUN_LA_FRONTEND_REFRESH=1`.
   - Added `LA_LOC_START_ITER`, passed through to `train_locaware.py`, so the localization loss schedule is explicit instead of hidden in the shell wrapper.

3. Tests
   - Added coverage-alignment tests that keep cached failed records and drop only missing records.
   - Added default parser tests for required cache behavior and opt-out.
   - Added shell-script tests for the new pipeline flag.
   - Added shell-script tests that guard the new train-mode behavior and the scratch-mode safety check.

## Verification

Passed:

- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_train_locaware_masks`
- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_full_script_args`
- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_la_artifacts.PseudoQueryManifestTest`
- `bash -n scripts/run_la_pseudo_query_pipeline.sh`

## Smoke Result

Reused a ShopFacade smoke manifest with 231 train RGB records and a 3-record cache.

Before this update, training sampled from all 231 records and many episodes had unknown teacher stage metadata.

After this update:

- teacher cache loaded: 3 items;
- alignment: `before=231 after=3 dropped_missing=228`;
- pseudo-query source: `{'train_rgb:accepted': 3}`;
- 10-step TensorBoard diagnostics:
  - `pseudo_query_source_train_rgb`: mean `1.0`;
  - `pseudo_query_source_synthetic_rgb`: mean `0.0`;
  - `pseudo_query_stage_unknown`: mean `0.0`.

This confirms that smoke/interrupted cache runs no longer contaminate training with uncovered pseudo-query records.

## ShopFacade 100-Step Clean Mainline

Command class:

- `LA_ENABLE_SYNTHETIC=0`
- `PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=1`
- full train-RGB manifest/cache
- 100 LA adaptation steps
- official test sparse-only eval
- no LA frontend refresh, so this is map-only evaluation with the baseline detector/landmark frontend

Teacher cache:

- manifest records: 231 train RGB;
- cache records: 231;
- stage counts: `teacher_ok=185`, `mixed_or_uncertain=43`, `sparse_failure=3`;
- sparse valid mask: disabled, as expected for train-RGB-only.

Training diagnostics:

- `pseudo_query_source_train_rgb`: mean `1.0`;
- `pseudo_query_source_synthetic_rgb`: mean `0.0`;
- `pseudo_query_stage_unknown`: mean `0.0`;
- sampled stage mix over 100 steps:
  - `teacher_ok`: mean `0.80`;
  - `mixed_or_uncertain`: mean `0.20`;
  - `sparse_failure`: mean `0.0`.

Official sparse-only result:

| Model | Median AE | Median TE | 5cm/5deg | 2cm/2deg | Avg Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| STDLoc baseline | 0.1665 deg | 3.3500 cm | 0.7282 | 0.2621 | 388.1 |
| LA train-RGB 100-step | 0.1630 deg | 3.1760 cm | 0.7282 | 0.2233 | 419.4 |
| LA train-RGB 500-step | 0.1622 deg | 3.1599 cm | 0.7379 | 0.2913 | 486.2 |

Interpretation:

- cache-aligned training is now clean and reproducible;
- 100-step improves median pose and inliers but hurts 2cm/2deg recall;
- 500-step is clearly more positive on ShopFacade: median TE, 5cm/5deg, 2cm/2deg, and inliers all improve over baseline;
- because LA frontend refresh is disabled, this is not the final full LA-STDLoc system.

## OldHospital 100-Step Clean Mainline

Command class:

- `LA_ENABLE_SYNTHETIC=0`
- `PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=1`
- full train-RGB manifest/cache
- 100 LA adaptation steps
- official test sparse-only eval
- no LA frontend refresh, so this is map-only evaluation with the baseline detector/landmark frontend

Teacher cache:

- manifest records: 895 train RGB;
- cache records: 895;
- stage counts: `teacher_ok=103`, `dense_improves_sparse=136`, `dense_rescues_sparse=237`, `dense_regression_after_good_sparse=11`, `mixed_or_uncertain=330`, `sparse_failure=78`;
- sparse valid mask: disabled, as expected for train-RGB-only.

Official sparse-only result:

| Model | Median AE | Median TE | 5cm/5deg | 2cm/2deg | Avg Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| STDLoc baseline | 0.3380 deg | 18.3941 cm | 0.0330 | 0.0055 | 274.8 |
| LA train-RGB 100-step | 0.3517 deg | 19.0959 cm | 0.0385 | 0.0055 | 274.6 |

Interpretation:

- 5cm/5deg recall improves slightly, but median AE/TE and 2m/5deg recall regress;
- this scene does not yet provide a clean positive accuracy claim;
- the stage distribution shows many mixed/sparse-failure records, so the current soft reliability design may still be too weak for OldHospital.

## Train-Mode Correction

The current positive ShopFacade result is an adaptation result, not a full new-data retraining result.

Previously, the wrapper always did this implicitly:

- copied the 30k STDLoc baseline model;
- loaded iteration 30000;
- used baseline `sampled_idx.pkl`;
- trained only `BASELINE_ITERS + LA_ADAPT_STEPS`.

That made it too easy to describe 100/500-step runs as if they were full retraining. The script now makes this explicit:

- `LA_TRAIN_MODE=adapt`: valid for current short verification, but still a map-adaptation experiment;
- `LA_TRAIN_MODE=scratch`: blocked unless a compatible landmark path and frontend-refresh path are provided.

This is a guardrail, not a completed scratch training implementation. A real full LA-STDLoc retraining path still needs landmark/bootstrap refresh and longer training.

## Current Status

The immediate implementation bug is closed: pseudo-query training no longer samples records that lack teacher cache coverage by default.

The method-level state is still staged:

- default mainline: all train RGB, no synthetic;
- synthetic: opt-in only, pending MAtCha-backed strict QA and integration;
- teacher cache: full sparse+dense diagnostics, used as soft reliability/stage signal, not hard teacher-ok gate;
- current 100/500-step runs: adaptation from the 30k STDLoc map, not full from-scratch LA retraining;
- current official eval without frontend refresh: map-only sparse eval, not final detector/frontend-coupled LA.

Accuracy evidence is currently mixed:

- ShopFacade 500-step gives positive sparse-only evidence under the clean cache-aligned mainline.
- OldHospital 100-step does not; it suggests the current reliability/stage weighting and map-only adaptation are not enough for harder scenes.

## Next Checks

Required before claiming method closure:

- run OldHospital 500-step under the same clean policy;
- implement a real full-training path instead of only adaptation from the 30k STDLoc baseline;
- run at least one frontend-refresh validation, because map-only sparse eval may understate or distort LA detector/landmark effects;
- reintroduce synthetic only as a separate MAtCha-backed ablation;
- keep synthetic valid/support mask as a no-reference synthetic-only module, not as a default train-RGB dependency.

Note: two long-running shell jobs printed `unexpected EOF while looking for matching '"'` after saving their result summaries. This is consistent with editing the same shell script while those old bash processes were still reading it. The current script passes `bash -n`, and the recorded result summaries were written before that post-run read conflict.
