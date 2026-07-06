# LA-STDLoc Training Mainline Refactor

Date: 2026-06-30

## Purpose

This update records the training-mainline refactor after the clean-default validation in
`LA_update13_clean_default_validation.md`.

The key correction is architectural: the default LA-STDLoc path should only include modules that
have a clear role in the sparse-only relocalization objective. Risky or inconclusive modules stay
available as explicit ablations, but they should not silently control the default student training
or official sparse-only evaluation.

## Evidence Driving The Refactor

The clean-default 100-step validation showed:

| Scene | Config | Median TE cm | Median AE deg | R5 cm/5deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: |
| ShopFacade | clean default 100 | 3.1756 | 0.1551 | 72.82 | 403.97 |
| OldHospital | clean default 100 with frontend refresh | 18.7609 | 0.3472 | 6.04 | 162.20 |

The OldHospital sparse-stage diagnosis showed a large inlier drop with frontend refresh:

| Comparison | Median TE delta | Mean TE delta | Median inlier delta | Mean inlier delta |
| --- | ---: | ---: | ---: | ---: |
| clean default vs baseline | -1.2989 cm | +38.8380 cm | -109.5 | -112.61 |

The refresh-off ablation isolated part of the cause:

| OldHospital config | Median TE cm | Median AE deg | R5 cm/5deg | Avg inliers |
| --- | ---: | ---: | ---: | ---: |
| baseline | 18.3941 | 0.3380 | 3.30 | 274.81 |
| LA-map only, frontend refresh off | 20.4676 | 0.3562 | 3.85 | 274.65 |
| LA-map plus refreshed frontend | 18.7609 | 0.3472 | 6.04 | 162.20 |

Interpretation:

- The severe OldHospital inlier collapse is tied to the refreshed detector/landmark frontend, not
  to the LA-map alone.
- LA-map alone is neutral to slightly worse on OldHospital, so it is not yet a positive support case.
- Refreshed frontend can improve some pose medians/recall, but it also creates severe outlier risk.
- Therefore frontend refresh should not remain in the default mainline until detector-only and
  interaction ablations explain the tradeoff.

## Default Mainline Changes

1. `scripts/run_la_pseudo_query_pipeline.sh` now defaults `RUN_LA_FRONTEND_REFRESH=0`.
   - LA detector/landmark refresh remains available with `RUN_LA_FRONTEND_REFRESH=1`.
   - This makes the default official eval measure the LA-map change with the baseline STDLoc sparse frontend.

2. `scripts/run_la_pseudo_query_pipeline.sh` now uses `scripts/make_stdloc_eval_cfg.py` for eval config generation.
   - This removes duplicated inline YAML editing from the bash script.
   - It supports optional sparse eval overrides:
     - `EVAL_SPARSE_DETECT_NUM`
     - `EVAL_SPARSE_REPROJECTION_ERROR`

3. `scripts/run_la_pseudo_query_pipeline.sh` now checks CUDA toolchain early.
   - It verifies `$CUDA_HOME/bin/nvcc`.
   - It verifies `cuda_runtime.h` under `$CUDA_HOME`.
   - This prevents late `gsplat` JIT failures caused by accidentally using the `iclpose` conda `nvcc`.

4. No-reference valid/support guidance now reaches dense teacher refinement.
   - `STDLoc.localize(...)` passes the same sparse guidance into `loc_dense(...)`.
   - `loc_dense(...)` applies the valid mask to dense coarse query-cell correlation before MNN matching.
   - This means synthetic RGB invalid regions are no longer only filtered in sparse keypoint selection; they
     are also excluded from dense coarse matching.

5. Teacher cache records dense mask diagnostics.
   - `dense_valid_mask_enabled`
   - `dense_valid_mask_valid_cells`
   - `dense_valid_mask_valid_frac`

6. Pseudo-query sparse pose initialization no longer hides a teacher-stage gate.
   - `EpisodeSampler` previously rejected cache items with `failure_stage in {sparse_failure, dense_rescues_sparse}`
     even when the script-level teacher gate was disabled.
   - The default now uses the cached sparse pose whenever the cache item exists and is not explicitly marked
     `failed`.
   - The old behavior is still available as an explicit ablation through
     `--pseudo_query_exclude_sparse_failure_stages` / `PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=1`.

7. Pseudo-query mixed/sparse training now requires the teacher cache.
   - If `pseudo_query_manifest` is set and `query_mode` is `sparse`, or `mixed` with nonzero
     `mixed_sparse_probability`, a missing `pseudo_teacher_cache` now fails early.
   - This prevents silent fallback to noise initialization in the intended teacher-cache training path.

8. `TRAIN_STEPS` is now treated as a backwards-compatible alias for `LA_ADAPT_STEPS`.
   - `LA_ADAPT_STEPS` means extra LA adaptation steps after loading the STDLoc baseline checkpoint.
   - `end_iter = BASELINE_ITERS + LA_ADAPT_STEPS`.
   - This clarifies that the current mainline is checkpoint-based LA adaptation, not RGB/3DGS training
     from scratch.

## Current Default Flow

The default training path is now:

1. Build pseudo-query manifest from all real train RGB plus synthetic RGB.
2. Use MAtCha synthetic RGB by default.
3. Do not use synthetic quality gate, pseudo-query selector, or teacher hard gate by default.
4. Build teacher cache as full STDLoc sparse plus dense diagnostics when `RUN_TEACHER_CACHE=1`.
5. Use no-reference valid/support guidance for synthetic RGB.
6. Train the LA student map from pseudo-query RGB-derived features using cached sparse teacher poses by default.
7. Run official sparse-only eval with the baseline sparse frontend unless frontend refresh is explicitly enabled.

## Detector-Only Ablation

A controlled OldHospital detector-only run was attempted:

```text
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/OldHospital/detector_only_baseline_map_30000det
```

The command used:

- baseline map at iteration 30000,
- `detector_la_envfix`,
- localization-aware landmark sampling,
- soft detector target,
- fixed CUDA toolchain `/usr/local/cuda-11.8`,
- official sparse-only evaluation after detector training.

Result:

- The run reached 30000/30000 detector iterations but exited with a floating point exception before
  saving `30000_detector.pth`.
- Root cause was not CUDA: the copied baseline map has `loc_observation_count == 0` for every Gaussian.
- With `--sampling_mode localization_aware --min_loc_observations 4`, no landmark is eligible, so
  `sampled_idx.pkl` was an empty tensor and `landmark_meta.pt` contained only empty tensors.
- Therefore this detector-only run is invalid and cannot be used as evidence about the frontend.

Code guard added:

- `train_detector.validate_detector_sampled_indices(...)` now fails before the detector training loop
  when sampling returns zero landmarks.
- This prevents another 30k-iteration run from silently training against an empty landmark set.

Valid detector-only ablation still needs one of:

- baseline sampling on the baseline map, or
- localization-aware sampling with `min_loc_observations=0`, or
- a map whose localization state has nonzero observation counts.

## Verification

Passed targeted tests:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_support_sparse_pnp tests.test_no_reference_valid_mask

CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_make_stdloc_eval_cfg tests.test_eval_analysis

CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_shared_eval_cfg_builder \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_checks_cuda_toolchain_before_training \
  tests.test_la_artifacts.ArtifactDetectorRepairTest.test_teacher_cache_helper_can_build_no_reference_support_mask \
  tests.test_la_artifacts.ArtifactDetectorRepairTest.test_teacher_cache_records_dense_valid_mask_diagnostics

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_empty_detector_landmark_sample_fails_before_training \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_nonempty_detector_landmark_sample_is_returned_as_long_tensor \
  tests.test_episode_sampler.EpisodeSamplerTest.test_sparse_mode_uses_stage_failed_cache_by_default \
  tests.test_episode_sampler.EpisodeSamplerTest.test_sparse_mode_can_opt_in_to_reject_cache_stage_sparse_failure \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_defaults_to_no_pseudo_query_stage_gate \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_pseudo_teacher_cache_is_optional_for_default_mainline \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_required_pseudo_teacher_cache_raises_clear_error \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_sparse_init_requires_teacher_cache
```

Passed syntax and whitespace checks:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  stdloc.py scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py \
  scripts/diagnose_sparse_stage_delta.py localization_training/eval_analysis.py

bash -n scripts/run_la_pseudo_query_pipeline.sh

git diff --check -- \
  stdloc.py scripts/build_pseudo_teacher_cache.py scripts/make_stdloc_eval_cfg.py \
  scripts/run_la_pseudo_query_pipeline.sh tests/test_support_sparse_pnp.py \
  tests/test_la_artifacts.py tests/test_full_script_args.py tests/test_make_stdloc_eval_cfg.py \
  tests/test_eval_analysis.py localization_training/eval_analysis.py \
  scripts/diagnose_sparse_stage_delta.py LA_update13_clean_default_validation.md
```

## Dense Mask Cache Smoke

A 4-query ShopFacade synthetic teacher-cache smoke was run on GPU2 to verify that no-reference
guidance is actually recorded and consumed by dense teacher refinement:

```text
/mnt/pool/sqy/stdloc_la_mainline_clean_100_20260630/ShopFacade/pseudo_query/densemask_smoke
```

Command properties:

- `--sources synthetic_rgb`
- `--max_queries 4`
- `--sparse_valid_mask`
- `--sparse_valid_mask_mode no_reference`
- dense refinement enabled through the normal `STDLoc.localize(...)` path.

Smoke result:

| Query | Sparse TE cm | Dense TE cm | Dense mask enabled | Dense valid frac | Sparse valid frac |
| --- | ---: | ---: | --- | ---: | ---: |
| `synthetic/000000.png` | 3.1027 | 4.9112 | true | 1.0000 | 1.0000 |
| `synthetic/000001.png` | 1248.0236 | 1355.7311 | true | 0.4739 | 0.4752 |
| `synthetic/000002.png` | 33.0204 | 32.1649 | true | 1.0000 | 1.0000 |
| `synthetic/000003.png` | 6.3993 | 5.7076 | true | 1.0000 | 1.0000 |

This is only a data-flow smoke, not a performance claim. It verifies that the dense stage now sees
the no-reference valid mask and records dense mask diagnostics in the teacher cache. The `000001`
case is also a useful future diagnostic sample because the no-reference mask marks less than half
of the coarse query cells as valid, while both sparse and dense teacher poses are poor.

## Next Decisions

1. Re-run detector-only OldHospital with a valid non-empty sampling configuration before drawing any frontend conclusion.
2. Re-run the default no-refresh mainline with dense valid-mask support and no hidden pseudo-query stage gate.
3. Treat 100-step runs as smoke/adaptation checks; use larger `LA_ADAPT_STEPS` for method evidence.
4. Only after the default path is stable should long 500-step or multi-seed runs be treated as evidence for the core claim.
