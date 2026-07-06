# LA_update26: Full-chain Frontend Capacity Ablation

Date: 2026-06-30

## Scope

This update continues the clean training-mainline refactor from LA_update25.

The goal was to test whether the remaining gap was partly caused by an under-capacity localization frontend. The ablation reruns the full chain with larger landmark banks, rather than only changing evaluation-time knobs:

1. reuse the audited all-train RGB pseudo-query manifest and STDLoc teacher cache;
2. bootstrap a larger landmark set;
3. train LA student from scratch for 2000 pseudo-query episodes;
4. refresh the scene-specific detector/frontend with the same landmark capacity;
5. run official sparse-only test evaluation.

Synthetic RGB, teacher-ok gate, selector ranking, ArtifactDetector/Repair, NoReferenceValidMask, valid/support masks, soft reliability, stage objectives, and direct depth check remain disabled.

## Code Changes

Added:

- `/root/STDLoc/scripts/run_la_capacity_fullchain_ablation.sh`

The wrapper runs one scene and one capacity setting while reusing a previously audited pseudo-query directory. Important defaults:

- `PSEUDO_QUERY_SOURCE_ROOT=/mnt/pool/sqy/stdloc_la_mainline_refactor_2000_20260630`
- `LA_ADAPT_STEPS=2000`
- `LA_BOOTSTRAP_LANDMARK_NUM=$LANDMARK_NUM`
- `LA_DETECTOR_LANDMARK_NUM=$LANDMARK_NUM`
- `RUN_PSEUDO_QUERY_MANIFEST=0`
- `RUN_TEACHER_CACHE=0`
- `RUN_TEACHER_CACHE_AUDIT=1`
- `FORCE_LA_FRONTEND_REFRESH=1`

This avoids regenerating teacher cache while still forcing the landmark bootstrap, LA training, detector refresh, and official evaluation to use the ablated capacity.

Test added:

- `test_capacity_fullchain_ablation_reuses_cache_and_runs_clean_mainline`

## Verification

Commands run:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_full_script_args.FullRunScriptArgsTest.test_capacity_fullchain_ablation_reuses_cache_and_runs_clean_mainline \
  tests.test_full_script_args.FullRunScriptArgsTest.test_clean_real_train_mainline_hard_disables_experimental_branches \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector

bash -n \
  scripts/run_la_capacity_fullchain_ablation.sh \
  scripts/run_la_clean_real_train_mainline.sh \
  scripts/run_la_pseudo_query_pipeline.sh
```

Both passed before the long runs. The default `pytest` environment is still unsuitable on this machine because it routes through the broken `iclpose` PyPy/NumPy stack, so targeted verification used `ulfloc_repro` and `unittest`.

## Run Roots

4096 clean mainline:

- `/mnt/pool/sqy/stdloc_la_mainline_refactor_2000_20260630`

8192 full-chain capacity:

- `/mnt/pool/sqy/stdloc_la_capacity_fullchain_8192_2000_20260630`

16384 full-chain capacity:

- `/mnt/pool/sqy/stdloc_la_capacity_fullchain_16384_2000_20260630`

## Training-entry Evidence

| Scene | Capacity | Observed points | Observed >=2 | Observed >=4 | Visible episodes | No-visible skips | Visible projections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 4096 | 4091 | 4091 | 4091 | 1977 | 23 | 3983577 |
| ShopFacade | 8192 | 8082 | 8082 | 8082 | 1979 | 21 | 4048196 |
| ShopFacade | 16384 | 13544 | 13544 | 13521 | 1982 | 18 | 4059136 |
| OldHospital | 4096 | 4096 | 4096 | 4096 | 2000 | 0 | 3918422 |
| OldHospital | 8192 | 8184 | 8179 | 8162 | 2000 | 0 | 4083379 |
| OldHospital | 16384 | 14446 | 14283 | 13906 | 2000 | 0 | 4095833 |

This closes one important implementation concern: the larger capacity was not only passed to evaluation. It changed bootstrap, training coverage, detector refresh, and final sparse-only evaluation.

## Official Sparse-only Results

| Scene | Model | Median TE cm | Median AE deg | 5cm/5deg | 2cm/2deg | 2m/5deg | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | STDLoc baseline 30000 | 3.350 | 0.167 | 0.728 | 0.262 | 1.000 | 388.1 |
| ShopFacade | clean 4096, seed170 | 3.622 | 0.181 | 0.689 | 0.126 | 1.000 | 351.1 |
| ShopFacade | clean 8192, seed180 | 3.438 | 0.165 | 0.777 | 0.184 | 1.000 | 384.4 |
| ShopFacade | clean 16384, seed182 | 3.319 | 0.164 | 0.680 | 0.175 | 1.000 | 374.9 |
| OldHospital | STDLoc baseline 30000 | 18.394 | 0.338 | 0.033 | 0.005 | 1.000 | 274.8 |
| OldHospital | clean 4096, seed171 | 16.285 | 0.327 | 0.082 | 0.005 | 1.000 | 234.6 |
| OldHospital | clean 8192, seed181 | 16.773 | 0.332 | 0.093 | 0.038 | 1.000 | 256.1 |
| OldHospital | clean 16384, seed183 | 16.066 | 0.324 | 0.099 | 0.005 | 0.995 | 252.3 |

Result paths:

- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_capacity_fullchain_8192_2000_20260630_ShopFacade_student_scratch_2000step_seed180-20260630_160236/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_capacity_fullchain_8192_2000_20260630_OldHospital_student_scratch_2000step_seed181-20260630_161228/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_capacity_fullchain_16384_2000_20260630_ShopFacade_student_scratch_2000step_seed182-20260630_165418/summary.json`
- `/root/STDLoc/results/pseudo-query-2000-_mnt_pool_sqy_stdloc_la_capacity_fullchain_16384_2000_20260630_OldHospital_student_scratch_2000step_seed183-20260630_171345/summary.json`

## Sequence-level Diagnostics

ShopFacade:

| Model | seq1 median TE / AE | seq1 5cm | seq1 inliers | seq3 median TE / AE | seq3 5cm | seq3 inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3.951 / 0.187 | 0.611 | 322.8 | 3.050 / 0.143 | 0.791 | 423.2 |
| 4096 | 4.846 / 0.199 | 0.528 | 294.2 | 3.298 / 0.174 | 0.776 | 381.7 |
| 8192 | 4.016 / 0.161 | 0.639 | 328.2 | 3.019 / 0.167 | 0.851 | 414.6 |
| 16384 | 4.452 / 0.175 | 0.528 | 311.4 | 3.006 / 0.164 | 0.761 | 409.0 |

OldHospital:

| Model | seq4 median TE / AE | seq4 5cm | seq4 inliers | seq8 median TE / AE | seq8 5cm | seq8 inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 14.240 / 0.343 | 0.107 | 228.9 | 20.836 / 0.336 | 0.000 | 295.2 |
| 4096 | 13.043 / 0.313 | 0.125 | 226.2 | 20.997 / 0.339 | 0.063 | 238.3 |
| 8192 | 12.923 / 0.307 | 0.107 | 233.6 | 19.438 / 0.341 | 0.087 | 266.1 |
| 16384 | 11.680 / 0.285 | 0.161 | 225.3 | 21.226 / 0.337 | 0.071 | 264.3 |

## Interpretation

Capacity is a real factor, but not the only bottleneck.

ShopFacade:

- 8192 is the best balanced setting: it beats STDLoc baseline on 5cm recall, median angle, and nearly matches inliers.
- 16384 gives the best median TE/AE, even slightly better than the STDLoc baseline, but it hurts 5cm recall. The tail gets worse, especially on `seq1`.
- This indicates a capacity sweet spot, not a monotonic benefit from more landmarks.

OldHospital:

- 16384 gives the best median TE/AE and best 5cm recall among the clean LA runs.
- 8192 gives the best 2cm recall.
- `seq4` consistently improves as capacity increases.
- `seq8` remains the hard regime: inliers recover from 4096 to 8192/16384, but median translation does not improve over baseline. This looks like a biased-pose/high-inlier regime rather than a complete matching failure.

Overall:

- The clean all-train RGB mainline now has positive support on both scenes in at least one primary metric.
- The result is still not a closed final claim, because improvements are not uniform across recall thresholds and sequences.
- The old suspicion that previous negative results were caused only by implementation confusion is partly resolved: capacity and clean frontend refresh matter, but method design still has a real unresolved gap.

## What Is Now Closed

- The training pool is all train RGB, not a small heldout subset.
- The current clean path does not use synthetic RGB, test data, teacher gate, selector ranking, valid/support masks, or artifact modules.
- The teacher cache is used as pose/diagnostic supervision metadata; query features are recomputed from RGB through the frozen feature extractor.
- The larger landmark capacity is actually consumed during bootstrap, LA training, frontend refresh, and official sparse-only evaluation.
- The 4096 bottleneck was real for ShopFacade, and partially real for OldHospital.

## What Remains Open

The remaining issue is not just "more data" or "more landmarks." The method needs a more principled link between the training objective and final PnP success:

- Landmark sampling should optimize geometric diversity and final PnP conditioning, not only visibility and detector match score.
- Training should distinguish helpful positives, ambiguous positives, and hard negatives instead of only regressing visible teacher features.
- The frontend should avoid over-capacity tail degradation, probably with per-scene capacity or learned landmark reliability.
- OldHospital `seq8` needs targeted diagnostics for high-inlier biased poses.
- Synthetic RGB should stay outside the default path until the real-train mainline is stable and the synthetic QA pipeline is rebuilt around no-reference render quality.

## Next Refactor Direction

Recommended next step:

1. make landmark selection geometry-aware: coverage, view diversity, triangulation angle, reprojection stability, and PnP leverage;
2. add a hard-negative/ambiguity objective over train RGB episodes, using final pose/inlier diagnostics as labels but not as a hard gate;
3. keep 8192 as the default ShopFacade capacity and test 8192/16384 per-scene for OldHospital;
4. add sequence-level and frame-level diagnostics for `ShopFacade/seq1` and `OldHospital/seq8`;
5. only after this, reintroduce synthetic RGB as a separate ablation with MAtCha/WildGaussians QA, not as part of the clean default.

