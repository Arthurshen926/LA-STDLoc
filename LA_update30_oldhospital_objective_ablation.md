# LA_update30: OldHospital clean student-objective ablation

Date: 2026-06-30

## Purpose

This update pauses the broad LA-STDLoc integration goal and closes a smaller
testable loop:

> On a clean OldHospital real-train baseline, check whether one
> student-objective change improves sparse-only high-precision recall and
> stability.

The run intentionally keeps the LA_update29 clean data boundary:

- OldHospital only;
- all real train RGB pseudo queries only;
- no synthetic RGB;
- no artifact detector, valid/support mask, or artifact repair;
- no pseudo-query selector or teacher gate;
- no direct-depth check;
- same reused 895-record `train_rgb` teacher cache for the paired 2000-step A/B;
- scratch LA student training;
- refreshed scene-specific detector;
- official sparse-only test evaluation.

The only intended change is the student objective:

- `PSEUDO_QUERY_RELIABILITY_MODE=soft`
- `PSEUDO_QUERY_RELIABILITY_LOSS_MODE=soft`
- `PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct`

This is a deliberately narrow ablation. It does not validate synthetic RGB,
MAtCha integration, no-reference masks, teacher gating, artifact repair, or
Gaussian pruning.

## Implementation

Added wrapper:

```bash
scripts/run_la_oldhospital_objective_ablation.sh
```

Important fixed controls:

```bash
export SCENES=OldHospital
export LA_ENABLE_SYNTHETIC=0
export SYNTHETIC_COUNT=0
export TEACHER_CACHE_SOURCES=train_rgb
export TEACHER_CACHE_SPARSE_VALID_MASK=0
export RUN_PSEUDO_QUERY_GATE=0
export RUN_PSEUDO_QUERY_SELECT=0
export PSEUDO_QUERY_SOURCES=train_rgb
export PSEUDO_QUERY_MAX_SYNTHETIC=0
export PSEUDO_QUERY_FILTER_TEACHER_CACHE=0
export PSEUDO_QUERY_ENABLE_TEACHER_GATE=0
export PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=0
export LA_DIRECT_DEPTH_CHECK=0
```

Objective settings:

```bash
export PSEUDO_QUERY_RELIABILITY_MODE=soft
export PSEUDO_QUERY_RELIABILITY_LOSS_MODE=soft
export PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct
export PSEUDO_QUERY_RELIABILITY_MIN_WEIGHT=0.20
export PSEUDO_QUERY_RELIABILITY_REAL_MIN_WEIGHT=0.45
export PSEUDO_QUERY_RELIABILITY_MEMORY_MIN_WEIGHT=0.80
export PSEUDO_QUERY_RELIABILITY_ERROR_SCALE=1.5
export PSEUDO_QUERY_RELIABILITY_INLIER_POWER=0.75
export PSEUDO_QUERY_RELIABILITY_TEACHER_OK_WEIGHT=1.0
export PSEUDO_QUERY_RELIABILITY_DENSE_IMPROVES_WEIGHT=0.95
export PSEUDO_QUERY_RELIABILITY_MIXED_WEIGHT=0.60
export PSEUDO_QUERY_RELIABILITY_DENSE_RESCUES_WEIGHT=0.70
export PSEUDO_QUERY_RELIABILITY_SPARSE_FAILURE_WEIGHT=0.25
export PSEUDO_QUERY_RELIABILITY_DENSE_REGRESSION_WEIGHT=0.25
export PSEUDO_QUERY_RELIABILITY_UNKNOWN_WEIGHT=0.50
```

Added script-argument coverage in:

```bash
tests/test_full_script_args.py
```

The test checks that the objective wrapper keeps the clean data boundary and
only turns on the intended soft reliability/direct objective branch.

## Teacher Cache

Both 2000-step runs reused the same OldHospital pseudo-query teacher cache:

```text
/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630/OldHospital/pseudo_query
```

Cache summary:

| Field | Value |
| --- | ---: |
| Records | 895 |
| Source | `train_rgb` only |
| Missing cache records | 0 |
| Extra cache records | 0 |
| Sparse valid mask | disabled |

Stage counts:

| Stage | Count |
| --- | ---: |
| `teacher_ok` | 103 |
| `sparse_failure` | 78 |
| `dense_rescues_sparse` | 237 |
| `dense_improves_sparse` | 136 |
| `dense_regression_after_good_sparse` | 11 |
| `mixed_or_uncertain` | 330 |

This keeps the earlier LA_update29 conclusion: OldHospital teacher episodes are
noisy, so objective design matters.

## Results

Metrics are official sparse-only `summary.json` values.

### 500-Step Smoke

| Run | Median TE cm | Median AE deg | 5cm recall | 2cm recall | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean seed411 | 19.445 | 0.372 | 0.0769 | 0.0000 | 197.1 |
| Objective seed411 | 18.111 | 0.371 | 0.0879 | 0.0055 | 200.4 |
| Delta objective-clean | -1.334 | -0.001 | +0.0110 | +0.0055 | +3.3 |

Result paths:

- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_clean500_seed411_20260630_OldHospital_student_scratch_500step_seed411-20260630_233628/summary.json`
- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective500_seed411_20260630_OldHospital_student_scratch_500step_seed411-20260630_233209/summary.json`

### 2000-Step Paired A/B

| Run | Median TE cm | Median AE deg | 5cm recall | 2cm recall | 2m recall | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean seed411 | 17.904 | 0.354 | 0.1044 | 0.0000 | 0.9890 | 258.4 |
| Objective seed411 | 17.550 | 0.336 | 0.1099 | 0.0165 | 1.0000 | 263.3 |
| Delta objective-clean | -0.355 | -0.017 | +0.0055 | +0.0165 | +0.0110 | +4.8 |

Result paths:

- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_oldhospital_clean2000_seed411_20260630_OldHospital_student_scratch_2000step_seed411-20260701_004048/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_oldhospital_objective2000_seed411_20260630_OldHospital_student_scratch_2000step_seed411-20260701_002602/summary.json`

Prior clean references:

| Run | Median TE cm | Median AE deg | 5cm recall | 2cm recall | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| LA_update29 clean 2000 seed302 | 16.593 | 0.330 | 0.1264 | 0.0110 | 251.8 |
| LA_update28 train-only 500 seed271 | 19.069 | 0.363 | 0.0714 | 0.0110 | 199.1 |

The paired seed411 result is the only direct objective-control comparison in
this update. The seed302/seed271 rows are context, not paired baselines.

## Conclusion

The objective change has weak but consistent positive support on this clean
OldHospital A/B:

- 500-step smoke improves median TE, 5cm recall, 2cm recall, and inliers.
- 2000-step paired A/B also improves median TE, median AE, 5cm recall, 2cm
  recall, 2m recall, and inliers.
- The 2000-step high-precision gain is small but directionally useful:
  2cm recall moves from 0.0 to 1.65%, and 5cm recall improves by 0.55pp.

This is not enough to claim the full LA-STDLoc goal is solved. It is enough to
keep stage-aware soft reliability/direct objective as a candidate for the next
clean baseline.

## Why This Helps Close The Previous Problem

Earlier rounds failed to reach a stable conclusion because synthetic rendering,
artifact detection, valid/support masks, teacher gates, pseudo-query selection,
student training, and detector refresh were changed together. This run removes
those high-impact confounds and shows that, under a clean real-train-only
boundary, changing the student objective alone can produce a small positive
official sparse-only improvement.

The remaining issue is still first-principles: the student is learning from
noisy teacher episodes, and OldHospital has many mixed or rescued dense-stage
cases. Therefore the next useful question is not simply adding more data. It is
how to make the student objective robust to teacher-stage quality while still
preserving hard or partially useful examples.

## Next Step

Recommended next closed loop:

1. Keep this objective as candidate, but do not promote it as default yet.
2. Run a small multi-seed OldHospital check on the same clean boundary.
3. Add a diagnostic split by teacher stage to see which stages drive the 2cm
   and 5cm changes.
4. Only after this objective is stable, reintroduce one additional module at a
   time, starting with MAtCha synthetic RGB or no-reference support masks, not
   both together.

