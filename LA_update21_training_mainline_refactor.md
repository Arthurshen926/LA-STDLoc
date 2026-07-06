# LA_update21: Training Mainline Refactor

Date: 2026-06-30

## Motivation

LA_update19/20 showed that the previous default training path mixed `train_rgb` and `synthetic_rgb` too aggressively:

- OldHospital all-source 100-step had 28/34 synthetic samples in sparse-failure or dense-regression stages.
- Removing synthetic improved OldHospital median TE directionally, but did not solve the scene, so synthetic is a confound rather than the only failure cause.
- ShopFacade synthetic was not obviously harmful, so synthetic should remain an explicit ablation path, not be deleted.

The mainline now defaults to a conservative train-RGB-only path. Synthetic data is opt-in.

## Code Changes

1. `scripts/run_la_pseudo_query_pipeline.sh`
   - Added `LA_ENABLE_SYNTHETIC`, default `0`.
   - If `LA_ENABLE_SYNTHETIC=0` and `SYNTHETIC_COUNT` is unset, `SYNTHETIC_COUNT=0`.
   - If `LA_ENABLE_SYNTHETIC=1`, default synthetic count remains 16.
   - Default `PSEUDO_QUERY_SOURCES` is now `train_rgb`.
   - Default `TEACHER_CACHE_SOURCES` is now `train_rgb`.
   - Default `PSEUDO_QUERY_SAMPLING_MODE` is now `record_proportional`.
   - No-reference synthetic region weighting and sparse valid-mask teacher-cache guidance default to `LA_ENABLE_SYNTHETIC`.
   - MAtCha/WildGaussians backend checks only run when `SYNTHETIC_RENDER_COUNT > 0`.

2. `train_locaware.py`
   - Parser default `--pseudo_query_sources` changed from `train_rgb,synthetic_rgb` to `train_rgb`.
   - Parser default `--pseudo_query_sampling_mode` changed from `source_balanced` to `record_proportional`.

3. `la_artifacts/pseudo_query.py`
   - `PseudoQuerySampler` API default changed to `record_proportional`.
   - Explicit `source_balanced` is still supported for ablations.

## Smoke Results

Default train-only pipeline smoke:

```bash
OUT_ROOT=/mnt/pool/sqy/stdloc_la_mainline_smoke_20260630
SCENES=ShopFacade
LA_ENABLE_SYNTHETIC=0
RUN_TRAIN=0
RUN_EVAL=0
TEACHER_CACHE_MAX=3
```

Results:

- Manifest: `{'train_rgb:accepted': 231}`
- Teacher cache sources: `['train_rgb']`
- Teacher cache count: `3`
- Teacher stage counts: `{'teacher_ok': 3}`
- Dense teacher fields present in all cached entries: `True`
- Sparse valid mask default: disabled

10-step training smoke:

```bash
OUT_ROOT=/mnt/pool/sqy/stdloc_la_mainline_smoke_20260630
SCENES=ShopFacade
LA_ENABLE_SYNTHETIC=0
LA_ADAPT_STEPS=10
TRAIN_SEED=122
RUN_PSEUDO_QUERY_MANIFEST=0
RUN_TEACHER_CACHE=0
RUN_TRAIN=1
RUN_EVAL=0
```

Training log:

- Pseudo-query manifest counts: `{'train_rgb:accepted': 231}`
- Sampling mode: `record_proportional`
- Saved iteration: `30010`

TensorBoard diagnostics:

- `pseudo_query_source_train_rgb`: `10.0`
- `pseudo_query_source_synthetic_rgb`: `0.0`
- `pseudo_query_is_synthetic`: `0.0`

## Verification

Passed:

- Target TDD tests for parser, shell defaults, and sampler defaults.
- `tests.test_full_script_args`: 34 tests.
- `tests.test_la_artifacts.PseudoQueryManifestTest`.
- `tests.test_train_locaware_masks`: 70 tests.
- `bash -n scripts/run_la_pseudo_query_pipeline.sh`.
- `py_compile` for affected Python entry points.
- `git diff --check` for affected files.

## Current Interpretation

This refactor closes a real implementation issue: the default path no longer silently over-samples synthetic pseudo-query data or depends on synthetic render backends.

It does not yet prove the final LA-STDLoc method. The current verified state is:

- The train-RGB mainline is clean and reproducible.
- Synthetic remains available but must be explicitly enabled.
- Teacher cache remains a full sparse+dense diagnostic/soft signal path, not a hard default gate.
- The previous 100/500-step experiments are still adaptation experiments from the 30k STDLoc map, not full 30k LA retraining from scratch.

The next experimental step should be a longer clean-mainline run:

- ShopFacade and OldHospital train-RGB-only 500-step with the new defaults.
- Then `LA_ENABLE_SYNTHETIC=1` with record-proportional synthetic on ShopFacade only first.
- Only after the clean path is stable should OldHospital synthetic be reintroduced with scene-specific QA/source policy.

