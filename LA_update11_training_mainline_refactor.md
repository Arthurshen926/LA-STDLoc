# LA-STDLoc Training Mainline Refactor Update

Date: 2026-06-30

## Goal

This update refactors the pseudo-query student training mainline after reviewing the all-train + synthetic RGB pipeline. The main goal is to keep the default method simple and evidence-backed:

- Use all real train RGB and accepted synthetic RGB as pseudo-query episodes.
- Do not use official test images for training.
- Do not hard-gate synthetic samples by teacher-cache success.
- Use no-reference support/valid masks as artifact-aware regional evidence.
- Keep teacher-cache reliability only as an explicit diagnostic/ablation, not as a default training controller.

## Current Default Training Flow

1. Build pseudo-query data from `train_rgb` and `synthetic_rgb`.
   - `train_rgb`: Cambridge train images with GT poses from preprocessing.
   - `synthetic_rgb`: RGB renders from the external RGB teacher backend, now favoring MAtCha over WildGaussians because current MAtCha renders are more stable.

2. Generate pseudo-query metadata.
   - Each query has source, pose, intrinsics, image path, support/mask statistics, and artifact diagnostics.
   - Teacher gating is disabled by default.
   - Pool selection is not intended to hard-rank normal samples into or out of the training set; it remains a utility for ablation and capping.

3. Run STDLoc teacher cache for diagnostics.
   - Full sparse/dense teacher outputs are cached when requested.
   - The cache records sparse pose, dense pose, errors, inliers, and failure stage.
   - The cache is useful for analysis, but should not be a default hard criterion for whether a pseudo-query is allowed to train the student.

4. Train the LA student.
   - Query features are extracted from RGB through the frozen SuperPoint feature extractor.
   - Real/synthetic pseudo-query sampling uses source-balanced sampling by default.
   - No-reference support masks can affect synthetic matching/region weighting.
   - Localization-aware map updates, detector refresh, and landmark sampling are trained from pseudo-query episodes.

5. Evaluate official sparse-only relocalization.
   - Official test images are only used for final sparse-only evaluation.
   - Dense refinement is not used at student inference in the sparse-only result.

## Refactor Decision

`pseudo_query_reliability_mode` was tested as a possible way to prevent bad teacher-cache episodes from corrupting multiview memory and localization stats. Two variants were tested:

- `soft_loss_100`: reliability also scales the localization loss.
- `stats_only_100`: reliability only gates multiview memory/localization stats, with no loss scaling.

Both variants were negative. Therefore:

- `--pseudo_query_reliability_mode` default is now `none`.
- `PSEUDO_QUERY_RELIABILITY_MODE` default in `scripts/run_la_pseudo_query_pipeline.sh` is now `none`.
- The reliability implementation remains available with `PSEUDO_QUERY_RELIABILITY_MODE=soft` for explicit ablations only.

## 100-Step Result Summary

| Setting | Scene | Median TE cm | Median AE deg | 5cm/5deg | Avg inliers | Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | ShopFacade | 3.3500 | 0.1665 | 72.80% | 388.10 | reference |
| no_reliability_100 | ShopFacade | 3.2167 | 0.1553 | 72.82% | 403.88 | positive vs baseline |
| no_reliability_500 | ShopFacade | 2.8919 | 0.1453 | 76.70% | 483.10 | strongest current support |
| soft_loss_100 | ShopFacade | 3.5284 | 0.1673 | 72.82% | 392.90 | worse |
| stats_only_100 | ShopFacade | 3.4736 | 0.1847 | 67.96% | 392.16 | worse |
| baseline | OldHospital | 18.3941 | 0.3380 | 3.30% | 274.80 | reference |
| no_reliability_100 | OldHospital | 19.6867 | 0.3460 | 6.04% | 162.32 | mixed: recall up, TE/inliers down |
| soft_loss_100 | OldHospital | 31.9810 | 0.5490 | 3.85% | 111.29 | much worse |
| stats_only_100 | OldHospital | 31.9715 | 0.5232 | 2.75% | 112.46 | much worse |

## Verified Conclusions

- The current all-train + synthetic mainline has positive evidence on ShopFacade, especially at 500 steps.
- OldHospital remains unresolved: the method does not yet provide positive precision support there.
- Teacher-cache reliability gating is a high-impact negative confounder in the current implementation.
- Loss downweighting by teacher-cache reliability is harmful in the tested form.
- Memory/stats gating by teacher-cache reliability is also harmful in the tested form.
- This failure does not disprove the LA-STDLoc idea; it mainly shows that teacher-cache-derived reliability should not control default training dynamics.

## Remaining Main Issues

- The student objective is still indirect: it learns localization-aware features, landmark sampling, and detector state through teacher episodes, but the final sparse-only metric depends on sparse matching/PnP behavior.
- Synthetic RGB quality and cross-map consistency still matter, but obvious black/blur render bugs are now treated as data-generation issues rather than training objectives.
- No-reference support masks are the current artifact-aware signal; previous reference-residual artifact detectors and hard selectors should remain out of the default synthetic path.
- OldHospital likely needs a scene-specific failure analysis focused on sparse matching geometry, inlier distribution, and detector/landmark refresh behavior rather than more teacher-cache gating.

## Verification

Commands passed:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_softly_downweights_bad_teacher_cache \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pseudo_query_reliability_none_keeps_mainline_unweighted

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_uses_candidate_multiplier_and_pool_selector

/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile train_locaware.py
bash -n scripts/run_la_pseudo_query_pipeline.sh

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_la_artifacts tests.test_train_locaware_masks tests.test_full_script_args \
  tests.test_no_reference_valid_mask tests.test_support_sparse_pnp \
  tests.test_pseudo_query_ab tests.test_stdloc_config_paths
```

The broader LA-related suite passed with 155 tests.
