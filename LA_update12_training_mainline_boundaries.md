# LA-STDLoc Training Mainline Boundary Refactor

Date: 2026-06-30

## Purpose

This update continues the training-mainline cleanup after `LA_update11_training_mainline_refactor.md`.
The goal is to make the default pseudo-query training path easier to reason about and less dependent on experimental side channels.

## Mainline Boundary Decisions

Default pseudo-query training is now treated as:

- `train_rgb + synthetic_rgb` records are allowed to train the student.
- Teacher-cache stage/error information is diagnostic by default.
- `pseudo_query_reliability` remains available only as an explicit ablation.
- Missing teacher cache must not break the default mainline unless a cache-dependent option is explicitly enabled.
- No-reference support/region weighting is still the default artifact-aware signal for synthetic RGB.

## Code Changes

1. Extracted teacher-cache reliability logic into a dedicated module:

   - `la_artifacts/pseudo_query_training.py`
   - Provides:
     - `pseudo_teacher_cache_reliability_stats`
     - `pseudo_query_reliability_decision`
     - support helpers for final TE, support fraction, and stage weights

   `train_locaware.py` now imports this module instead of owning the reliability implementation.

2. Added optional teacher-cache loading boundary:

   - New helper: `_load_training_pose_cache(args)`
   - Missing `--pseudo_teacher_cache` is skipped when both are true:
     - `pseudo_query_filter_teacher_cache=False`
     - `pseudo_query_reliability_mode=none`
   - Missing cache raises a clear `FileNotFoundError` when cache-dependent behavior is enabled.

3. Added regression coverage:

   - Reliability tests now import the independent module directly.
   - Default mainline test confirms missing pseudo teacher cache is optional.
   - Required-cache test confirms explicit cache-dependent options fail clearly.

## Why This Matters

Earlier iterations allowed teacher-cache diagnostics to control training dynamics too easily. That produced negative results in both ShopFacade and OldHospital reliability-gated runs. This refactor makes that boundary explicit:

- Diagnostics can be computed.
- Ablations can enable them.
- The default training path does not silently depend on them.

This also makes future full validation cleaner: a default run without teacher-cache gating is now less likely to fail simply because the cache was not generated or copied.

## Verification

Commands passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_softly_downweights_bad_teacher_cache \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_none_keeps_mainline_unweighted

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_pseudo_teacher_cache_is_optional_for_default_mainline \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_missing_required_pseudo_teacher_cache_raises_clear_error

/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py la_artifacts/pseudo_query_training.py

bash -n scripts/run_la_pseudo_query_pipeline.sh

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_la_artifacts tests.test_train_locaware_masks tests.test_full_script_args \
  tests.test_no_reference_valid_mask tests.test_support_sparse_pnp \
  tests.test_pseudo_query_ab tests.test_stdloc_config_paths
```

The broader LA-related suite passed with 157 tests.

## Next Validation Step

The next meaningful experiment should use the cleaned default mainline:

- `PSEUDO_QUERY_RELIABILITY_MODE=none`
- `RUN_PSEUDO_QUERY_SELECT=0`
- `RUN_PSEUDO_QUERY_GATE=0`
- `SYNTHETIC_QUALITY_GATE=0` unless testing QA explicitly
- MAtCha synthetic RGB backend
- No-reference region/support weighting enabled

Run ShopFacade and OldHospital 100-step first, then only promote to 500-step if the 100-step result is not obviously negative. OldHospital should also be analyzed by sparse-matching stage, because current evidence suggests the main failure is not solved by teacher-cache gating.
