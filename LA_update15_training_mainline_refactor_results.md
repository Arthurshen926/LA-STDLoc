# LA-STDLoc Training Mainline Refactor Results

Date: 2026-06-30

## Scope

This update records the first end-to-end validation after the training-mainline refactor in
`LA_update14_training_mainline_refactor.md`.

The goal was not to add another artifact-specific trick. The goal was to make the default training
path internally consistent and falsifiable:

- no hidden teacher-stage gate,
- no silent fallback to random/noisy sparse initialization when pseudo-query sparse training is intended,
- no default frontend refresh unless explicitly requested,
- MAtCha synthetic RGB as the default synthetic backend,
- no hard synthetic teacher-ok filter,
- no selector that ranks pseudo queries before training,
- no-reference support masks applied to synthetic teacher sparse/dense localization,
- official test kept separate from train/synthetic pseudo-query construction.

## Implemented Corrections

### Pseudo-Query Teacher Cache Is Required When It Is Semantically Required

`train_locaware.py` now treats the pseudo teacher cache as required when:

- `pseudo_query_filter_teacher_cache` is enabled,
- `pseudo_query_reliability_mode != none`,
- a pseudo-query manifest is used with `query_mode=sparse`,
- a pseudo-query manifest is used with `query_mode=mixed` and nonzero
  `mixed_sparse_probability`.

This fixes a high-impact implementation risk: previously the intended pseudo-query sparse/mixed
training path could silently fall back to noise initialization if the cache was missing.

### Hidden Teacher Stage Gate Removed From Default Sampling

`EpisodeSampler` previously rejected teacher-cache records with failure stages such as
`sparse_failure` and `dense_rescues_sparse`, even when the script-level teacher gate was disabled.

The default now uses cached sparse poses for all non-failed cache records. The old behavior remains
an explicit ablation:

```text
--pseudo_query_exclude_sparse_failure_stages
PSEUDO_QUERY_EXCLUDE_SPARSE_FAILURE_STAGES=1
```

This matters because OldHospital's cache distribution is broad. A hard teacher-ok gate would remove
most of the available training data and would entangle dataset difficulty with a manually selected
threshold.

### LA Adaptation Steps Are Explicit

`scripts/run_la_pseudo_query_pipeline.sh` now uses `LA_ADAPT_STEPS` as the primary variable.
`TRAIN_STEPS` remains a backwards-compatible alias.

The training endpoint is:

```text
end_iter = BASELINE_ITERS + LA_ADAPT_STEPS
```

This makes it explicit that the current pipeline adapts from the STDLoc baseline checkpoint. It is
not a full RGB/3DGS retraining run from iteration zero.

### Detector-Only Empty-Landmark Failure Is Guarded

`train_detector.validate_detector_sampled_indices(...)` now fails before training if detector
landmark sampling returns zero landmarks.

This fixed the invalid OldHospital detector-only ablation where `loc_observation_count` was all zero
and `--sampling_mode localization_aware --min_loc_observations 4` produced an empty `sampled_idx`.

## Current Mainline Flow

The current default pipeline is:

1. Build a pseudo-query manifest from all real train RGB plus synthetic RGB.
2. Render synthetic RGB with MAtCha.
3. Do not apply pseudo-query gate or selector by default.
4. Build a full STDLoc teacher cache for real train and synthetic query records.
5. Apply no-reference support/valid masks to synthetic teacher localization.
6. Train the LA student map using source-balanced sampling from real train and synthetic records.
7. Evaluate official test with sparse-only STDLoc localization and the baseline sparse frontend.

The current v1 synthetic scale in the full validation was intentionally small:

| Scene | Real train records | Synthetic records | Total teacher cache |
| --- | ---: | ---: | ---: |
| ShopFacade | 231 | 16 | 247 |
| OldHospital | 895 | 16 | 911 |

## Smoke Validation

A 10-step ShopFacade smoke was run to verify the refactored end-to-end path:

```text
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_smoke_20260630
SCENES=ShopFacade
LA_ADAPT_STEPS=10
SYNTHETIC_COUNT=2
TEACHER_CACHE_SOURCES=synthetic_rgb
PSEUDO_QUERY_SOURCES=synthetic_rgb
RUN_PSEUDO_QUERY_GATE=0
RUN_PSEUDO_QUERY_SELECT=0
RUN_LA_FRONTEND_REFRESH=0
RUN_EVAL=1
```

Result:

| Scene | Steps | Median TE cm | Median AE deg | R5 cm/5deg | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 10 | 3.3170 | 0.1584 | 72.82 | 387.79 |

This is only a data-flow smoke. It proves the refactored path runs through synthetic rendering,
teacher cache, student training, and official sparse-only evaluation.

## 500-Step Validation

Both scenes used:

```text
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630
LA_ADAPT_STEPS=500
TRAIN_SEED=0
SYNTHETIC_COUNT=16
SYNTHETIC_CANDIDATE_MULTIPLIER=1
TEACHER_CACHE_SOURCES=train_rgb,synthetic_rgb
PSEUDO_QUERY_SOURCES=train_rgb,synthetic_rgb
PSEUDO_QUERY_REAL_WEIGHT=2.0
PSEUDO_QUERY_SYNTHETIC_WEIGHT=1.0
PSEUDO_QUERY_SAMPLING_MODE=source_balanced
RUN_PSEUDO_QUERY_GATE=0
RUN_PSEUDO_QUERY_SELECT=0
RUN_LA_FRONTEND_REFRESH=0
RUN_EVAL=1
```

### Teacher Cache Stage Counts

| Scene | Count | Teacher ok | Sparse failure | Dense rescues sparse | Dense improves sparse | Dense regression | Mixed/uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 247 | 191 | 6 | 1 | 1 | 0 | 48 |
| OldHospital | 911 | 104 | 89 | 239 | 136 | 12 | 331 |

Interpretation:

- ShopFacade teacher cache is mostly stable.
- OldHospital has many non-clean teacher stages. A hard teacher-ok gate would discard most records,
  so removing the hidden default gate was the right architectural correction.
- At the same time, OldHospital now exposes the real method problem: the current student loss does
  not yet use this stage diversity well enough.

### Official Sparse-Only Test Metrics

| Scene | Config | Median TE cm | Median AE deg | R5 cm/5deg | R2 cm/2deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | clean default 100 | 3.1756 | 0.1551 | 72.82 | n/a | 403.97 |
| ShopFacade | refactor 500 | 3.2226 | 0.1560 | 76.70 | 26.21 | 477.18 |
| OldHospital | baseline | 18.3941 | 0.3380 | 3.30 | n/a | 274.81 |
| OldHospital | clean default 100 + frontend refresh | 18.7609 | 0.3472 | 6.04 | n/a | 162.20 |
| OldHospital | LA-map only, refresh off | 20.4676 | 0.3562 | 3.85 | n/a | 274.65 |
| OldHospital | refactor 500 | 19.8546 | 0.3519 | 4.40 | 0.00 | 269.37 |

Result artifacts:

```text
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/ShopFacade/student_500step_seed0
/mnt/pool/sqy/stdloc_la_refactor_full_20260630/OldHospital/student_500step_seed0
results/pseudo-query-30500-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_ShopFacade_student_500step_seed0-20260630_082918
results/pseudo-query-30500-_mnt_pool_sqy_stdloc_la_refactor_full_20260630_OldHospital_student_500step_seed0-20260630_084456
```

## What This Supports

ShopFacade gives weak positive support:

- R5 improves from 72.82 to 76.70.
- Average sparse inliers improve from 403.97 to 477.18.
- Median TE/AE remain effectively unchanged.

OldHospital does not give positive support yet:

- It improves over the refresh-off LA-map-only R5 baseline, but remains worse than the previous
  clean-default 100-step with frontend refresh.
- It is close to baseline on median error and inliers, but still does not solve the high-precision
  5cm recall target.

Therefore the refactor improves the experimental validity of the pipeline, but it does not by
itself complete the LA-STDLoc method objective.

## Current First-Principles Diagnosis

The current student mostly adapts map-localized appearance/features from cached pseudo-query
episodes. That is a weak learning signal if the sparse pose target is noisy, ambiguous, or already
close to the original map behavior.

The intended student should learn more than "fit teacher-rendered features from a cached pose":

- which rendered regions are trustworthy for localization,
- which 3D points/landmarks are stable under real and synthetic query appearance,
- how to reduce harmful contributors for localization without destroying useful geometry,
- how to exploit hard or partially failed teacher episodes instead of treating every cached pose as
  equally useful supervision.

The refactor exposes that these are not solved yet. OldHospital is the evidence: many records are
not clean teacher-ok cases, but simply keeping them without stage-aware supervision does not produce
the desired gain.

## Remaining Risks

1. Synthetic count is still small.
   - The validation used only 16 synthetic frames per scene.
   - This is enough to validate the pipeline, not enough to prove synthetic augmentation value.

2. No-reference support is currently a masking/guidance signal, not a learned artifact repair model.
   - It helps avoid obvious invalid synthetic regions during teacher localization.
   - It does not yet repair the RGB or feature map contributors.

3. The student objective is not stage-aware enough.
   - `teacher_ok`, `dense_rescues_sparse`, `sparse_failure`, and `mixed_or_uncertain` records likely
     should not contribute identically.
   - This should be learned/weighted, not handled by a hard default gate.

4. The frontend refresh remains unresolved.
   - It can improve some OldHospital recall but causes inlier collapse.
   - A valid detector-only ablation still needs to be run with non-empty landmark sampling.

5. Online pseudo-query generation is not implemented.
   - The current pipeline uses offline manifests/cache for reproducibility and diagnostics.
   - Online generation may be useful later, but only after the training objective is clarified.

## Next Refactor Direction

The next change should not be another manual sample selector. The more defensible direction is:

1. Keep all real train and valid synthetic records in the pool.
2. Replace hard teacher gates with per-episode reliability weights and stage-aware losses.
3. Separate supervision targets:
   - stable real-train teacher episodes for conservative map adaptation,
   - synthetic episodes for appearance/viewpoint robustness,
   - sparse-failure or dense-rescue episodes for contrastive/uncertainty learning rather than direct
     pose imitation.
4. Add explicit localization stability heads/metrics for landmarks and rendered regions.
5. Revisit frontend refresh only after detector-only ablation is valid.

The current refactored mainline is a better experimental base. It is not yet the final method.
