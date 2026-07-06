# LA_update28: Refactored Mainline 500-step Results and Self-audit

Date: 2026-06-30

## Scope

This update records the first full 500-step validation of the refactored
training mainline from `LA_update27`.

The goal was to test the new default path with:

1. all train RGB pseudo-queries;
2. optional MAtCha synthetic RGB pseudo-queries;
3. spatial-offset synthetic pose sampling;
4. no teacher hard gate;
5. no selector ranking;
6. no teacher-stage reliability or stage-objective loss;
7. complete sparse+dense STDLoc teacher cache;
8. synthetic-only no-reference support mask/score in teacher localization;
9. scratch LA student training plus frontend refresh;
10. official sparse-only test evaluation.

This update is deliberately not a success claim. It is a checkpoint to separate
which parts of the refactor are now executable from which parts are actually
supported by pose metrics.

## Commands

Train + synthetic:

```bash
SCENES=ShopFacade SYNTHETIC_COUNT=128 LA_ADAPT_STEPS=500 TRAIN_SEED=271 GPU=0 \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactored_mainline_500_20260630 \
bash scripts/run_la_refactored_mainline.sh

SCENES=OldHospital SYNTHETIC_COUNT=128 LA_ADAPT_STEPS=500 TRAIN_SEED=271 GPU=1 \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactored_mainline_500_20260630 \
bash scripts/run_la_refactored_mainline.sh
```

Train-only causal controls:

```bash
SCENES=ShopFacade SYNTHETIC_COUNT=0 LA_ADAPT_STEPS=500 TRAIN_SEED=271 GPU=2 \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactored_trainonly_500_20260630 \
bash scripts/run_la_refactored_mainline.sh

SCENES=OldHospital SYNTHETIC_COUNT=0 LA_ADAPT_STEPS=500 TRAIN_SEED=271 GPU=2 \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactored_trainonly_500_20260630 \
bash scripts/run_la_refactored_mainline.sh
```

## Official Sparse-only Results

| Scene | Run | Median TE cm | Median AE deg | 5cm/5deg | 2cm/2deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | train + 128 synthetic | 3.488 | 0.195 | 0.689 | 0.223 | 335.4 |
| ShopFacade | train-only | 3.261 | 0.177 | 0.738 | 0.194 | 338.1 |
| OldHospital | train + 128 synthetic | 17.604 | 0.361 | 0.038 | 0.000 | 199.2 |
| OldHospital | train-only | 19.069 | 0.363 | 0.071 | 0.011 | 199.1 |

Reference from `LA_update26`:

| Scene | Reference | Median TE cm | Median AE deg | 5cm/5deg | 2cm/2deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | STDLoc baseline 30000 | 3.350 | 0.167 | 0.728 | 0.262 | 388.1 |
| ShopFacade | clean 8192, 2000-step | 3.438 | 0.165 | 0.777 | 0.184 | 384.4 |
| OldHospital | STDLoc baseline 30000 | 18.394 | 0.338 | 0.033 | 0.005 | 274.8 |
| OldHospital | clean 8192, 2000-step | 16.773 | 0.332 | 0.093 | 0.038 | 256.1 |
| OldHospital | clean 16384, 2000-step | 16.066 | 0.324 | 0.099 | 0.005 | 252.3 |

Result paths:

- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_refactored_mainline_500_20260630_ShopFacade_student_scratch_500step_seed271-20260630_180554/summary.json`
- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_refactored_trainonly_500_20260630_ShopFacade_student_scratch_500step_seed271-20260630_183036/summary.json`
- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_refactored_mainline_500_20260630_OldHospital_student_scratch_500step_seed271-20260630_184433/summary.json`
- `/root/STDLoc/results/pseudo-query-500-_mnt_pool_sqy_stdloc_la_refactored_trainonly_500_20260630_OldHospital_student_scratch_500step_seed271-20260630_191540/summary.json`

## Teacher-cache Source Breakdown

| Scene | Run | Source | Count | teacher_ok | sparse_failure | dense_rescues | dense_improves | dense_regress | mixed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | train + synthetic | train_rgb | 231 | 185 | 3 | 0 | 0 | 0 | 43 |
| ShopFacade | train + synthetic | synthetic_rgb | 128 | 73 | 3 | 4 | 3 | 0 | 45 |
| OldHospital | train + synthetic | train_rgb | 895 | 103 | 78 | 237 | 136 | 11 | 330 |
| OldHospital | train + synthetic | synthetic_rgb | 128 | 2 | 62 | 30 | 0 | 3 | 31 |

All four cache runs had complete manifest/cache coverage. The support-mask
configuration was enabled but applied only to `synthetic_rgb`:

- mode: `support_mask_score`
- sources: `synthetic_rgb`
- hard mask minimum fraction: `0.5`
- candidate multiplier: `2.0`
- support score weight: `0.5`

## Training-entry Evidence

| Scene | Run | Episodes | train_rgb episodes | synthetic episodes | Visible episodes | Nonzero loss episodes | Observed points | Observed >=4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | train + synthetic | 500 | 390 | 110 | 488 | 488 | 8132 | 7972 |
| ShopFacade | train-only | 500 | 500 | 0 | 497 | 497 | 8080 | 7980 |
| OldHospital | train + synthetic | 500 | 461 | 39 | 500 | 500 | 8182 | 8016 |
| OldHospital | train-only | 500 | 500 | 0 | 500 | 500 | 8169 | 8018 |

This rules out an empty-training or missing-cache failure. The runs consumed the
pseudo-query cache and updated a comparable number of localization landmarks.

## Interpretation

The refactored wrapper is executable end to end, but the new mainline is not yet
a reliable accuracy improvement.

ShopFacade:

- Train-only is the best of these 500-step refactored runs.
- Adding 128 MAtCha synthetic samples hurts median TE, 5cm recall, and inliers.
- Train-only slightly beats the STDLoc baseline on median TE, but is worse on
  median AE, 2cm recall, and inliers. This is not a clean positive result.
- Compared with the clean 8192/2000-step result from `LA_update26`, the 500-step
  refactored train-only result has lower 5cm recall and fewer inliers.

OldHospital:

- Train + synthetic has better median TE than train-only, but worse 5cm and 2cm
  recall.
- Both 500-step refactored runs are worse than the clean 8192/16384 2000-step
  capacity runs from `LA_update26`.
- OldHospital train RGB teacher cache is already weak: only 103/895 teacher_ok,
  with 78 sparse failures and many dense rescue/mixed cases.
- OldHospital synthetic is much worse: only 2/128 teacher_ok and 62/128 sparse
  failures. This is a real synthetic/teacher-chain bottleneck, not just a
  selector or teacher-gate artifact.

## Self-audit

The current refactored path fixes several old implementation confounders:

- no test data is used for training;
- teacher hard gate and selector ranking are off;
- synthetic uses MAtCha and spatial-offset pose sampling;
- train-only controls are now available;
- support mask/score is synthetic-only and no-reference;
- training is scratch, not a tiny fine-tune;
- official sparse-only evaluation runs after frontend refresh.

But the path still mixes too many responsibilities for a default method:

- `run_la_refactored_mainline.sh` always couples synthetic generation, teacher
  cache, LA feature training, detector refresh, and official eval;
- 500-step results cannot be compared directly to the 2000-step clean capacity
  runs, but the inlier and recall degradation is already visible;
- OldHospital's teacher cache quality is weak even for real train RGB, so adding
  synthetic data on top of it does not solve the core problem;
- current synthetic samples are used with fixed sampling weights, while the
  teacher diagnostics show scene-dependent reliability differences;
- the training objective still mostly optimizes visible teacher feature/pose
  consistency, not final PnP leverage, ambiguity rejection, or sequence-level
  failure modes.

## Current Conclusion

The user's concern that previous results could be distorted by implementation
confounding remains valid. This refactor removes several confounders, but the
new evidence does not support making synthetic RGB part of the default training
pool yet.

The best-supported default should remain a clean all-train RGB path with larger
capacity and frontend refresh, while synthetic/MAtCha should be an isolated
ablation until its teacher-cache behavior is improved.

## Next Refactor Direction

Recommended next mainline split:

1. `clean-real-train-mainline`: all train RGB, no synthetic, no valid/support
   masks, no artifact modules, no teacher gate, no selector, 2000-step default,
   capacity 8192/16384 per scene.
2. `synthetic-ablation-mainline`: add MAtCha synthetic only after the clean path
   is reproduced, with separate per-source sampling and per-source loss logging.
3. `teacher-cache-diagnostics`: standalone cache generation and source/stage
   reporting, not an implicit training-pool selector.
4. `frontend-refresh`: standalone, force-refreshable step with explicit capacity
   and min-observation controls.
5. `geometry-aware-landmark-selection`: next method work should target final PnP
   conditioning, view diversity, and ambiguous-landmark suppression instead of
   only adding more RGB views.

Immediate next experiment should be a 2000-step rerun of the clean real-train
path using this refactored wrapper style but with synthetic disabled and with
the same capacity settings as `LA_update26`. Only after that reproduces the
clean baseline should synthetic be reintroduced.
