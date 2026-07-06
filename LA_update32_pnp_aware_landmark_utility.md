# LA_update32: PnP-Aware Landmark Utility Small Loop

Date: 2026-07-01

## Scope

This update closes a small, clean OldHospital loop after pausing the broader LA-STDLoc goal.

Boundary:
- Scene: OldHospital only.
- Training: scratch student, 100 steps, train_rgb pseudo-query cache only.
- No synthetic RGB.
- No artifact detector, valid/support mask, teacher gate, or pseudo-query selector changes.
- Evaluation: official test sparse-only.

The tested idea is `localization_aware_pnp`: keep localization-aware utility scoring, but add a PnP-oriented spatial balancing step so selected landmarks are less collapsed into a few dense areas.

## Code Changes

Implemented:
- `localization_training/landmark_distill.py`
  - Added `voxel_balanced_score(...)`.
  - Added `pnp_balance`, `pnp_voxel_size`, `pnp_max_per_voxel`, and `pnp_preserve_ratio` to `localization_aware_sample(...)`.
  - Fixed the PnP branch to preserve the spatial baseline candidate count instead of selecting every eligible landmark.
- `train_detector.py`
  - Added CLI mode `--sampling_mode localization_aware_pnp`.
  - Added PnP controls:
    - `--pnp_voxel_size`
    - `--pnp_max_per_voxel`
    - `--pnp_preserve_ratio`
- `scripts/run_la_pseudo_query_pipeline.sh`
  - Forwards PnP sampling controls into LA frontend refresh.
- Tests:
  - PnP voxel balancing behavior.
  - Hybrid preservation of top utility landmarks.
  - Regression test for the candidate-count expansion bug.
  - CLI/script argument coverage.

## Important Bug/Confound Found

The first PnP implementation was not a fair comparison.

Old behavior:
- Control selected 6406 landmarks.
- Initial PnP selected 6835 landmarks.
- The initial PnP set was a strict superset of control:
  - intersection: 6406
  - pnp-only: 429
  - control-only: 0

This meant the negative result was partly confounded by adding a low-score tail, not just by spatial balancing.

Fix:
- PnP now first computes the normal spatial localization-aware candidate count, then applies voxel balancing within the same target count.
- Fixed PnP selected 6406 landmarks.
- Compared with control:
  - intersection: 6250
  - control-only: 156
  - pnpfixed-only: 156
  - Jaccard: 0.9525

## OldHospital 100-Step Sparse-Only Results

All runs use seed 414 and the same 100-step student checkpoint boundary.

| Variant | Median TE cm | Median AE deg | Recall 5m/10deg | Recall 2m/5deg | Avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| control `localization_aware` | 64.18 | 1.1468 | 75.27% | 69.23% | 61.36 |
| old PnP / hybrid, confounded | 65.09 | 1.2490 | 71.98% | 65.38% | 61.86 |
| fixed PnP | 63.45 | 1.1719 | 74.18% | 68.13% | 61.68 |

Delta of fixed PnP vs control:
- Median TE: -0.72 cm.
- Median AE: +0.025 deg.
- Recall 5m/10deg: -1.10 pp.
- Recall 2m/5deg: -1.10 pp.
- Avg inliers: +0.32.

## Interpretation

The implementation confound is closed: the initial negative PnP result was partly caused by unintended candidate-pool expansion.

However, the fixed version is still not a clear positive result:
- It improves median translation slightly and inliers slightly.
- It still loses recall at both 5m/10deg and 2m/5deg.
- It does not solve the major OldHospital failure cluster, especially many `seq8` frames with large sparse PnP failures.

Decision:
- Keep `localization_aware_pnp` as an experimental mode.
- Do not promote it to the default training/evaluation mainline.
- The next small-loop objective should focus on OldHospital high-precision recall and sequence-level sparse failure stability, not on broad synthetic/artifact pipeline changes.

## Verification

Passed:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_landmark_distill \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_detector_parser_accepts_pnp_aware_sampling_mode \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_forwards_pnp_sampling_controls
```

Result: `Ran 8 tests ... OK`.

Passed:

```bash
bash -n scripts/run_la_pseudo_query_pipeline.sh
bash -n scripts/run_la_oldhospital_objective_ablation.sh
```

Runtime note:
- The default shell environment had `CUDA_HOME` unset and `PATH` pointed to `/root/miniconda3/envs/iclpose/bin/nvcc`.
- `gsplat` JIT compilation failed with missing `cuda_runtime.h`.
- Running with `CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH` fixed the environment issue.
