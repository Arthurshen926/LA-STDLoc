# LA_update29: clean mainline closure and progress summary

Date: 2026-06-30

## Purpose

This update closes the long-running goal mode round by separating the reliable
part of the current LA-STDLoc implementation from the still-experimental
synthetic/artifact branches.

The main question was whether the current implementation failure was caused by
the LA idea itself, or by accumulated implementation and experiment
confounds. This round intentionally removed those confounds and reran a clean
control:

- all Cambridge train RGB as pseudo queries;
- no synthetic RGB;
- no teacher gate;
- no valid/support mask;
- no artifact detector or repair;
- no reliability weighting;
- full STDLoc teacher cache kept only for diagnostics;
- scratch LA student training;
- refreshed scene-specific detector;
- official sparse-only test evaluation.

## Commands And Outputs

Clean wrapper:

```bash
scripts/run_la_clean_real_train_mainline.sh
```

Important wrapper defaults:

- `LA_ENABLE_SYNTHETIC=0`
- `SYNTHETIC_COUNT=0`
- `TEACHER_CACHE_SOURCES=train_rgb`
- `TEACHER_CACHE_SPARSE_VALID_MASK=0`
- `RUN_PSEUDO_QUERY_GATE=0`
- `RUN_PSEUDO_QUERY_SELECT=0`
- `PSEUDO_QUERY_RELIABILITY_MODE=none`
- `PSEUDO_QUERY_RELIABILITY_LOSS_MODE=none`
- `PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=none`
- `PSEUDO_QUERY_NO_REFERENCE_REGION_WEIGHT=0`
- `LA_DIRECT_DEPTH_CHECK=0`
- `LA_TRAIN_MODE=scratch`

Logged runs:

| Scene | Capacity | Seed | GPU | Output root | Log |
| --- | ---: | ---: | ---: | --- | --- |
| ShopFacade | 8192 | 301 | 0 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` | `/mnt/pool/sqy/stdloc_la_clean_mainline_logs_20260630/shop8192.log` |
| OldHospital | 8192 | 302 | 1 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` | `/mnt/pool/sqy/stdloc_la_clean_mainline_logs_20260630/old8192.log` |
| OldHospital | 16384 | 303 | 2 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_16384_2000_20260630` | `/mnt/pool/sqy/stdloc_la_clean_mainline_logs_20260630/old16384_reusecache.log` |

The first unlogged launch lost process visibility after interruption. The
logged rerun fixed this process issue and all three runs completed.

## Teacher Cache Diagnostics

The clean pseudo-query manifests contain only real train RGB:

| Scene | Records | Source counts | Valid/support mask | Cache coverage |
| --- | ---: | --- | --- | --- |
| ShopFacade | 231 | `train_rgb: 231` | disabled | 231/231 |
| OldHospital | 895 | `train_rgb: 895` | disabled | 895/895 |

Teacher stage counts:

| Scene | teacher_ok | sparse_failure | dense_rescues_sparse | dense_improves_sparse | dense_regression_after_good_sparse | mixed_or_uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 185 | 3 | 0 | 0 | 0 | 43 |
| OldHospital | 103 | 78 | 237 | 136 | 11 | 330 |

This is a key result. Even without synthetic or artifact modules, OldHospital's
teacher diagnostics are noisy: only 103/895 train queries are classified as
`teacher_ok`, with many dense rescues and mixed cases. Therefore dense teacher
outputs should not be treated as an oracle for student learning.

## Official Sparse-Only Results

Metrics are from `summary.json` sparse results.

| Run | Scene | Capacity | Steps | Median TE cm | Median AE deg | 5cm recall | 2cm recall | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STDLoc baseline from LA_update26 | ShopFacade | - | - | 3.350 | 0.167 | 0.728 | 0.262 | 388.1 |
| LA_update26 clean-ish | ShopFacade | 8192 | 2000 | 3.438 | 0.165 | 0.777 | 0.184 | 384.4 |
| LA_update28 refactor train-only | ShopFacade | 8192 | 500 | 3.261 | 0.177 | 0.738 | 0.194 | 338.1 |
| LA_update29 clean logged | ShopFacade | 8192 | 2000 | 3.220 | 0.159 | 0.816 | 0.233 | 371.3 |
| STDLoc baseline from LA_update26 | OldHospital | - | - | 18.394 | 0.338 | 0.033 | 0.005 | 274.8 |
| LA_update26 clean-ish | OldHospital | 8192 | 2000 | 16.773 | 0.332 | 0.093 | 0.038 | 256.1 |
| LA_update28 refactor train-only | OldHospital | 8192 | 500 | 19.069 | 0.363 | 0.071 | 0.011 | 199.1 |
| LA_update29 clean logged | OldHospital | 8192 | 2000 | 16.593 | 0.330 | 0.126 | 0.011 | 251.8 |
| LA_update26 clean-ish | OldHospital | 16384 | 2000 | 16.066 | 0.324 | 0.099 | 0.005 | 252.3 |
| LA_update29 clean logged | OldHospital | 16384 | 2000 | 14.760 | 0.326 | 0.099 | 0.000 | 249.0 |

New result paths:

- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_clean_mainline_logged_8192_2000_20260630_ShopFacade_student_scratch_2000step_seed301-20260630_203846/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_clean_mainline_logged_8192_2000_20260630_OldHospital_student_scratch_2000step_seed302-20260630_210125/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_clean_mainline_logged_16384_2000_20260630_OldHospital_student_scratch_2000step_seed303-20260630_210448/summary.json`

## Conclusions

The clean all-train RGB mainline has positive support, but not a complete
solution.

Positive evidence:

- ShopFacade improves over the carried STDLoc baseline on median TE, median AE,
  and 5cm recall.
- ShopFacade also improves over the LA_update26 8192 result on median TE,
  median AE, 5cm recall, and 2cm recall.
- OldHospital 8192 improves over the carried STDLoc baseline on median TE,
  median AE, and 5cm recall.
- OldHospital 16384 gives the best OldHospital median TE so far in this round:
  14.760cm, compared with 18.394cm baseline and 16.066cm in LA_update26.

Limitations:

- 2cm recall is not consistently improved. OldHospital 16384 reaches the best
  median translation but 2cm recall is 0.
- Avg inliers are lower than the original baseline in both scenes.
- This result uses only real train RGB. It does not validate synthetic RGB,
  MAtCha integration, no-reference support masks, artifact repair, or physical
  Gaussian pruning.
- OldHospital teacher diagnostics show that teacher stage labels are noisy.
  Training against all cached teacher signals can help median pose, but it does
  not guarantee high-precision recall.

Therefore the current defensible claim is narrower than the original broad
goal:

> A clean LA student trained from all real train RGB pseudo-query episodes can
> improve sparse-only median relocalization and 5cm recall on ShopFacade and
> OldHospital, but the current pipeline does not yet robustly improve
> high-precision recall or prove that synthetic/artifact modules help.

## Why The Long Goal Did Not Reach The Original Expectation

1. Too many modules were mixed at once.
   RGB teacher training, synthetic view generation, artifact detection/repair,
   teacher gates, valid/support masks, teacher cache sampling, and student
   training were all changing together. This made negative results ambiguous.

2. The dense teacher is not an oracle.
   OldHospital's cache has only 103/895 `teacher_ok` records, with 78 sparse
   failures, 237 dense rescues, and 330 mixed cases. This means teacher cache
   quality is a first-order variable, not a passive label source.

3. Synthetic RGB was not reliable enough.
   WildGaussians produced blurred ShopFacade renders and severe OldHospital
   failures in earlier rounds. MAtCha appears stronger visually and by heldout
   PSNR/SSIM, but it has not yet been integrated into the clean default
   training loop.

4. Several artifact/mask ideas were not yet methodologically stable.
   The earlier `ArtifactDetector`, valid selector, teacher gate, and support
   mask branches mixed no-reference and reference assumptions. The current
   clean run intentionally disables them until they can be tested as isolated
   ablations.

5. The student learning target is still limited.
   The current student mainly learns a localization-aware feature Gaussian map
   and refreshed scene-specific detector. It is not yet a fully online
   synthetic-query training framework, and it does not directly learn to repair
   rendering artifacts.

6. Process control was initially weak.
   Early long GPU jobs were launched without robust logs. After an interruption,
   process state was unclear and only partial artifacts remained. This was fixed
   by the logged rerun in this update.

## Current Method Flow

The current clean default flow is:

1. Build a pseudo-query manifest from all Cambridge train RGB.
2. Run full STDLoc teacher inference for those train RGB records and write a
   teacher cache with sparse pose, dense pose, errors, and stage diagnostics.
3. Train the LA student map from scratch using `train_locaware.py` with only
   `train_rgb` pseudo queries and no reliability gates or masks.
4. Refresh the scene-specific detector with localization-aware landmark
   sampling.
5. Run official sparse-only test evaluation.

Experimental branches kept out of the default flow:

- MAtCha/WildGaussians synthetic RGB generation;
- spatial synthetic pose sampling;
- no-reference support/valid masks;
- artifact detector and repair;
- teacher gate and pseudo-query selector;
- physical pruning.

## Next Decisions

The clean real-train path should now be treated as the control baseline for
future ablations.

Recommended next steps:

1. Do not reintroduce synthetic RGB and artifact modules together. Add one
   ablation at a time against this clean control.
2. If synthetic RGB is revisited, prefer MAtCha over WildGaussians, but use it
   first as a training-data ablation with no teacher gate.
3. Redesign no-reference support masks as a separate module and measure PnP
   inliers, sparse failures, and official pose metrics before using them in
   student training.
4. Re-examine the student objective. The biggest remaining question is not data
   volume alone, but what the student should learn from noisy teacher episodes
   so that high-precision recall improves rather than only median TE.
