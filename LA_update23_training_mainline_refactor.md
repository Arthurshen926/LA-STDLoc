# LA_update23: Training Mainline Refactor and Scratch Bootstrap

Date: 2026-06-30

## Why This Update

The previous clean mainline was still too easy to misread:

- most reported 100/500-step runs were `adapt` experiments copied from the 30k STDLoc baseline map;
- the baseline detector landmarks could still leak into experiments described as new LA training;
- a scratch student map had no valid `sampled_idx.pkl`, so direct localization supervision could not start cleanly;
- first-run scratch model directories had no `cfg_args`, which broke tools that assumed an existing trained model.

This update separates short adaptation validation from a real scratch/full-training path.

## Code Changes

1. `arguments/__init__.py`
   - `get_combined_args` now tolerates a missing `cfg_args` file and falls back to command-line/default parser values.
   - This is required for first-run scratch model directories.

2. `train_detector.py`
   - Added `--landmark_only`.
   - In landmark-only mode, the script samples/writes `sampled_idx.pkl` and exits before detector training.
   - Added default filling for sentinel-created model args when no `cfg_args` exists, so `train_detector.py` can bootstrap a fresh model folder.
   - Existing detector training remains unchanged when `--landmark_only` is not set.

3. `scripts/run_la_pseudo_query_pipeline.sh`
   - `LA_TRAIN_MODE=adapt|scratch` is now explicit.
   - `adapt` keeps the previous short-check behavior: copy/load the 30k STDLoc baseline map, then run LA adaptation.
   - `scratch` creates a fresh student model directory.
   - If scratch mode has no explicit `LA_LANDMARK_PATH`, it now bootstraps landmarks from the initialized scratch map:
     - detector folder: `LA_BOOTSTRAP_DETECTOR_FOLDER`, default `detector_bootstrap`;
     - sampling mode: `LA_BOOTSTRAP_SAMPLING_MODE`, default `baseline`;
     - output: `$train_model/$LA_BOOTSTRAP_DETECTOR_FOLDER/sampled_idx.pkl`.
   - Scratch official eval still requires `RUN_LA_FRONTEND_REFRESH=1`, because baseline frontend indices are not valid for a scratch map.

4. Tests
   - Added tests for missing `cfg_args` fallback.
   - Added tests for `--landmark_only` parser/default behavior.
   - Added shell-script tests that guard scratch bootstrap and the scratch-eval frontend-refresh safety check.

## Verification

Passed:

- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_arguments tests.test_detector_soft_targets tests.test_full_script_args tests.test_train_locaware_masks`
  - 123 tests OK.
- `bash -n scripts/run_la_pseudo_query_pipeline.sh`
- `git diff --check -- arguments/__init__.py train_detector.py tests/test_arguments.py tests/test_detector_soft_targets.py tests/test_full_script_args.py scripts/run_la_pseudo_query_pipeline.sh LA_update22_cache_aligned_mainline.md`

## Scratch Smoke

Direct bootstrap smoke:

- scene: ShopFacade;
- mode: `train_detector.py --iteration 0 --landmark_only`;
- output: `/mnt/pool/sqy/stdloc_la_scratch_smoke_20260630/ShopFacade/detector_bootstrap/sampled_idx.pkl`;
- result: 64 landmarks sampled successfully from the initialized map.

Pipeline-level scratch smoke:

- command class: `LA_TRAIN_MODE=scratch`, `LA_ADAPT_STEPS=1`, `RUN_EVAL=0`, `LA_BOOTSTRAP_LANDMARK_NUM=64`;
- output model: `/mnt/pool/sqy/stdloc_la_mainline_clean_20260630/ShopFacade/student_scratch_1step_seed131`;
- bootstrap artifact: `detector_bootstrap/sampled_idx.pkl`;
- trained artifact: `point_cloud/iteration_1/point_cloud.ply`;
- pseudo cache alignment: `before=231 after=231 dropped_missing=0`;
- loaded direct landmarks: 64;
- train log: `[ITER 1] base 0.349042 loc 0.000000 psnr 12.346`;
- result: the scratch pipeline now starts from an empty student model directory, bootstraps landmarks, loads pseudo-query cache, and saves a trained point cloud.

Important caveat:

- this is a plumbing smoke only;
- `loc 0.000000` at one step means it is not evidence that the scratch objective is effective;
- full training still needs longer schedules, normal landmark counts, and frontend refresh before official accuracy claims.

## OldHospital 500-Step Clean Adapt Result

Command class:

- `LA_TRAIN_MODE=adapt`;
- `LA_ENABLE_SYNTHETIC=0`;
- `PSEUDO_QUERY_REQUIRE_TEACHER_CACHE=1`;
- full train-RGB manifest/cache;
- 500 LA adaptation steps;
- official test sparse-only eval;
- no LA frontend refresh.

| Model | Median AE | Median TE | 5cm/5deg | 2cm/2deg | Avg Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| STDLoc baseline | 0.3380 deg | 18.3941 cm | 0.0330 | 0.0055 | 274.8 |
| LA train-RGB 500-step adapt | 0.3466 deg | 20.2754 cm | 0.0385 | 0.0000 | 271.4 |

Interpretation:

- OldHospital 500-step adapt is mixed/negative;
- 5cm/5deg recall improves slightly, but median TE/AE, 2cm/2deg recall, 2m/5deg recall, and inliers regress;
- this reinforces that map-only short adaptation is not a sufficient final LA-STDLoc design for harder scenes.

## Current Method Boundary

What is cleanly implemented now:

- all-train-RGB pseudo-query cache can be used with coverage alignment;
- `adapt` experiments are explicitly labeled as adaptation from the STDLoc baseline;
- `scratch` experiments no longer silently reuse baseline detector landmarks;
- scratch can bootstrap legal initial landmarks from the current initialized student map;
- frontend refresh is available as a separate step after LA map training.

What is not solved yet:

- no full long scratch LA training has been validated;
- no official scratch eval with refreshed LA detector/frontend has been completed;
- synthetic/MAtCha data is still not part of the default mainline;
- the current positive ShopFacade 500-step result remains an adaptation result;
- OldHospital still does not support a broad positive accuracy claim.

## Next Required Checks

Before claiming LA-STDLoc method closure:

- run scratch 100/500-step smoke with normal landmark counts;
- run `RUN_LA_FRONTEND_REFRESH=1` after scratch/adapt training and evaluate with the refreshed detector/frontend;
- compare `adapt` vs `scratch` vs `scratch+frontend-refresh` on ShopFacade and OldHospital;
- only after this, reintroduce MAtCha synthetic as a separate ablation;
- keep synthetic no-reference valid/support masks as synthetic-specific QA unless they prove useful in the full teacher-cache path.

## Follow-up: Mainline Refactor v2

The 1-step scratch smoke above exposed a real implementation problem rather than a method conclusion: the training path could run and save a point cloud while localization-aware supervision/stats stayed effectively unused. I audited the path before changing defaults.

Root-cause checks:

- direct landmark supervision itself was not dead: OldHospital train-RGB records produced nonzero visible anchors and nonzero descriptor/full-bank losses in a direct teacher sanity check;
- the bootstrap landmark pools were geometrically visible from real train cameras: ShopFacade and OldHospital both had hundreds of projected visible landmarks per camera in projection-only audits;
- the previous failures were therefore caused by mainline control flow and filtering, not by a proof that the LA objective is invalid.

Implementation changes after the audit:

- `scripts/run_la_pseudo_query_pipeline.sh`
  - default `PSEUDO_QUERY_RELIABILITY_MODE=none`;
  - default `PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none`;
  - default `PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=none`;
  - default `LA_DIRECT_DEPTH_CHECK=0`, with `--direct_depth_check` passed only when explicitly enabled.
- `train_locaware.py`
  - added `loc_training_summary.json`;
  - logs direct teacher episode counts, visible anchor counts, nonzero loss counts, localization-stat update counts, and observed-point coverage;
  - decouples localization-stat updates from teacher-cache reliability/stage gates unless those gates are explicitly requested.

Rationale:

- the default training pool should not silently enforce a hard or soft teacher-ok policy;
- sparse/dense teacher diagnostics remain useful for analysis, but they should not decide whether a pseudo-query can affect student learning unless an ablation explicitly asks for that;
- direct depth checking is too fragile as a default because it depends on render/JIT/toolchain behavior and can suppress otherwise valid geometric supervision.

## Refactor v2 Verification

Passed code checks:

- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_full_script_args tests.test_train_locaware_masks tests.test_detector_soft_targets tests.test_arguments`
  - 123 tests OK.
- `bash -n scripts/run_la_pseudo_query_pipeline.sh`
- `PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile train_locaware.py train_detector.py`

ShopFacade scratch smoke:

- scene: ShopFacade;
- mode: `LA_TRAIN_MODE=scratch`;
- synthetic: disabled;
- teacher cache: train-RGB only, capped to 32 for smoke speed;
- training: 5 steps;
- output model: `/mnt/pool/sqy/stdloc_la_mainline_refactor_smoke_20260630/ShopFacade/student_scratch_5step_seed140`.

The new summary shows the mainline now really updates localization-aware state:

| Metric | Value |
| --- | ---: |
| episodes | 5 |
| pseudo_query_episodes | 5 |
| direct_episodes | 5 |
| direct_visible_episodes | 5 |
| direct_visible_total | 575 |
| direct_visible_max | 120 |
| direct_nonzero_loss_episodes | 5 |
| stats_candidate_episodes | 5 |
| stats_update_episodes | 5 |
| stats_update_points_total | 575 |
| stats_skip_reliability_episodes | 0 |
| stats_skip_stage_episodes | 0 |
| stats_skip_no_visible_episodes | 0 |
| observed_points | 120 |
| observed_points_ge_4 | 118 |
| observed_points_max | 5 |

Frontend refresh smoke:

- command class: same scratch 5-step model, `RUN_LA_FRONTEND_REFRESH=1`, `LA_DETECTOR_ITERS=1`, `LA_DETECTOR_LANDMARK_NUM=128`;
- output detector: `/mnt/pool/sqy/stdloc_la_mainline_refactor_smoke_20260630/ShopFacade/student_scratch_5step_seed140/detector_la_smoke/1_detector.pth`;
- output landmarks: `/mnt/pool/sqy/stdloc_la_mainline_refactor_smoke_20260630/ShopFacade/student_scratch_5step_seed140/detector_la_smoke/sampled_idx.pkl`;
- result: localization-aware sampling and detector saving completed.

Important scope:

- this is still not an accuracy claim;
- it closes the immediate engineering bug where the nominal LA training path could execute without meaningful LA observation/state updates;
- full validation still requires uncapped all-train cache, normal landmark counts, 100/500-step runs, frontend refresh, and official sparse-only test evaluation.
