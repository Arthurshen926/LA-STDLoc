# LA_update31: OldHospital objective multi-seed audit

Date: 2026-07-01

## Purpose

This update follows LA_update30 and keeps the narrower clean-loop goal:

> On the clean OldHospital real-train-only baseline, verify whether one
> student-objective change gives stable sparse-only improvement.

The update also fixes one high-impact implementation issue found during the
multi-seed audit: the previous "soft" objective still hard-gated the landmark
statistics updates.

The data boundary is unchanged:

- OldHospital only;
- all real train RGB pseudo queries only;
- no synthetic RGB;
- no artifact detector, no valid/support mask, no artifact repair;
- no teacher gate or pseudo-query selector;
- no direct-depth check;
- reused 895-record `train_rgb` teacher cache;
- scratch LA student training;
- refreshed scene-specific detector;
- official sparse-only test evaluation.

## Implementation Fix

The previous objective runs used soft loss weights, but their statistics update
path still shared the memory-update threshold and stage gate. That made most
episodes unable to update localization-aware landmark statistics.

Observed hard-stats coverage:

| Run | Stats updates | Reliability skips | Stage skips | Observed points |
| --- | ---: | ---: | ---: | ---: |
| hard seed411 | 102 / 500 | 398 | 398 | 7813 |
| hard seed412 | 110 / 500 | 390 | 390 | 7766 |
| hard seed413 | 95 / 500 | 405 | 405 | 7801 |

Changes made:

- `la_artifacts/pseudo_query_training.py`
  - added independent `pseudo_query_reliability_stats_min_weight`;
  - separated `update_memory` from `update_stats`;
  - kept legacy behavior when the new stats threshold is unset.
- `train_locaware.py`
  - added `--pseudo_query_stage_stats_policy {hard,soft}`;
  - `hard` preserves the old stage-gated stats path;
  - `soft` lets low-quality stages update stats while memory remains gated.
- `scripts/run_la_pseudo_query_pipeline.sh`
  - forwards the new soft-stats controls.
- `scripts/run_la_oldhospital_objective_ablation.sh`
  - defaults objective ablations to:
    - `PSEUDO_QUERY_STAGE_STATS_POLICY=soft`
    - `PSEUDO_QUERY_RELIABILITY_STATS_MIN_WEIGHT=0.0`
- Tests added in:
  - `tests/test_pseudo_query_ab.py`
  - `tests/test_full_script_args.py`

The fix restored full stats coverage:

| Run | Stats updates | Reliability skips | Stage skips | Observed points |
| --- | ---: | ---: | ---: | ---: |
| soft seed412 | 500 / 500 | 0 | 0 | 8165 |
| soft seed413 | 500 / 500 | 0 | 0 | 8168 |

This confirms that the prior objective result was underpowered by an
implementation issue.

## Teacher Cache

All runs reused:

```text
/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630/OldHospital/pseudo_query
```

Cache summary:

| Field | Value |
| --- | ---: |
| Records | 895 |
| Source | `train_rgb` only |
| Sparse valid mask | disabled |
| Missing cache records | 0 |

Stage counts:

| Stage | Count |
| --- | ---: |
| `teacher_ok` | 103 |
| `sparse_failure` | 78 |
| `dense_rescues_sparse` | 237 |
| `dense_improves_sparse` | 136 |
| `dense_regression_after_good_sparse` | 11 |
| `mixed_or_uncertain` | 330 |

This remains a noisy teacher setting, especially for stage-mixed episodes.

## Official Sparse-Only Results

All values are from official `summary.json` files.

| Seed | Run | Median TE cm | Median AE deg | 5cm recall | 2cm recall | Avg inliers |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 411 | clean | 19.445 | 0.372 | 0.0769 | 0.0000 | 197.1 |
| 411 | hard objective | 18.111 | 0.371 | 0.0879 | 0.0055 | 200.4 |
| 412 | clean | 18.534 | 0.357 | 0.0659 | 0.0000 | 190.6 |
| 412 | hard objective | 17.930 | 0.347 | 0.0549 | 0.0110 | 190.9 |
| 412 | soft-stats objective | 15.830 | 0.336 | 0.0275 | 0.0000 | 190.3 |
| 413 | clean | 19.046 | 0.352 | 0.0549 | 0.0000 | 194.9 |
| 413 | hard objective | 19.770 | 0.390 | 0.0495 | 0.0110 | 189.5 |
| 413 | soft-stats objective | 19.286 | 0.386 | 0.0549 | 0.0055 | 189.1 |

Result paths:

```text
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_clean500_seed411_20260630_OldHospital_student_scratch_500step_seed411-20260630_233628/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective500_seed411_20260630_OldHospital_student_scratch_500step_seed411-20260630_233209/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_clean500_seed412_20260701_OldHospital_student_scratch_500step_seed412-20260701_105253/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective500_seed412_20260701_OldHospital_student_scratch_500step_seed412-20260701_104646/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective_softstats500_seed412_20260701_OldHospital_student_scratch_500step_seed412-20260701_113630/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_clean500_seed413_20260701_OldHospital_student_scratch_500step_seed413-20260701_105257/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective500_seed413_20260701_OldHospital_student_scratch_500step_seed413-20260701_111202/summary.json
/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_oldhospital_objective_softstats500_seed413_20260701_OldHospital_student_scratch_500step_seed413-20260701_113609/summary.json
```

## Paired Delta Diagnostics

Delta is candidate minus clean, so negative TE is better.

| Run | Mean TE delta cm | Median TE delta cm | Improved fraction | 5cm gain/loss |
| --- | ---: | ---: | ---: | ---: |
| hard seed412 | +4.565 | -0.278 | 0.522 | 7 / 9 |
| hard seed413 | +0.116 | +0.007 | 0.500 | 6 / 7 |
| soft seed412 | +7.263 | +0.173 | 0.489 | 4 / 11 |
| soft seed413 | +0.392 | -0.113 | 0.505 | 5 / 5 |

Diagnostic paths:

```text
/mnt/pool/sqy/stdloc_la_oldhospital_objective500_seed412_20260701/oldhospital_objective_vs_clean_500_stage_delta.json
/mnt/pool/sqy/stdloc_la_oldhospital_objective500_seed413_20260701/oldhospital_objective_vs_clean_500_stage_delta.json
/mnt/pool/sqy/stdloc_la_oldhospital_objective_softstats500_seed412_20260701/oldhospital_objective_softstats_vs_clean_500_stage_delta.json
/mnt/pool/sqy/stdloc_la_oldhospital_objective_softstats500_seed413_20260701/oldhospital_objective_softstats_vs_clean_500_stage_delta.json
```

Qualitative pattern from the paired diagnostics:

- `seq8` remains the dominant difficult region.
- Soft-stats seed412 improves median TE overall but introduces one severe
  outlier: `seq8/frame00012.png`, from 72.6cm to 1259.0cm.
- Soft-stats seed412 loses more 5cm successes than it gains: 4 gain vs 11 loss.
- Soft-stats seed413 is essentially neutral at 5cm: 5 gain vs 5 loss.
- Inliers generally decrease under soft-stats, especially in `seq8`
  (`-7.47` average inliers for seed413), even when some poses improve.

## Conclusion

The implementation issue is closed: the objective can now update stats for all
episodes while keeping memory updates reliability-gated.

The method evidence is not yet positive enough:

- Hard-stats seed411 looked positive, but multi-seed results show instability.
- Soft-stats fixes the training coverage bug, but does not produce stable
  high-precision recall gains.
- Median TE can improve, especially seed412, but 5cm recall and outlier
  behavior can regress.
- The current objective tends to reduce inlier counts, which suggests the
  learned detector/landmark utility is not reliably aligned with sparse PnP
  robustness.

Therefore the current direct soft-reliability objective should not be promoted
as the default LA-STDLoc mainline. The old broad goal failed because too many
modules were coupled, but even after isolating a clean objective and fixing its
stats path, this specific objective is still weak.

## Next Step

The next closed loop should stay small:

1. Keep the clean OldHospital real-train-only boundary.
2. Stop adding synthetic RGB, masks, or artifact repair until the student
   objective is stable.
3. Design an objective that directly optimizes sparse PnP robustness:
   landmark ranking, covisibility diversity, and match/inlier stability, not
   only per-landmark direct feature similarity.
4. Add diagnostics that compare detector-selected landmarks before and after
   training on the same query, especially `seq8/frame00008-00016` and
   `seq8/frame00029-00052`.
5. Treat the current soft-stats path as an implementation tool, not as a
   validated method contribution.
