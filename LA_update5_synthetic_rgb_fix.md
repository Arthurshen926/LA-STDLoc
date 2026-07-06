# LA_update5 Synthetic RGB Fix

## Scope

This update focuses on the current synthetic RGB blocker:

- ShopFacade WildGaussians renders are mostly plausible but blurry.
- OldHospital WildGaussians synthetic renders were all black.

The goal was to separate implementation bugs from RGB teacher fidelity limits before using synthetic RGB pseudo-queries for student training.

## Findings

### OldHospital all-black was a bad checkpoint/config issue

The original OldHospital WildGaussians checkpoint was numerically unhealthy. Critical tensors contained NaNs, including `xyz`, `features_rest`, appearance embeddings, and appearance MLP weights. Rendering from this checkpoint produced black images.

A second "safe" attempt with sky and uncertainty disabled still diverged near 1100 iterations, which pointed to the SH/appearance path rather than only sky or densification.

The stable OldHospital config is:

```bash
--set sh_degree=0
--set appearance_enabled=false
--set num_sky_gaussians=0
--set uncertainty_mode=disabled
--set densify_from_iter=9000
```

The 8000-step checkpoint at:

```text
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/OldHospital_wg_sh0_noapp_8000/checkpoint-8000
```

passes the RGB teacher checkpoint health gate and renders non-black synthetic images.

### ShopFacade blur is an RGB teacher fidelity issue

ShopFacade default WildGaussians and no-appearance/no-sky WildGaussians both render non-black images, but the outputs remain strongly low-pass. The no-appearance/no-sky 8000-step model improves metrics slightly but does not solve the blur.

The current checkpoint is:

```text
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/ShopFacade_wg_noapp_nosky_8000/checkpoint-8000
```

Its evaluation metrics are:

```text
PSNR 13.59521
SSIM 0.39811
MAE  0.15590
MSE  0.04692
LPIPS 0.67927
```

This is not yet high-fidelity enough to treat synthetic RGB as a strong teacher source.

### Synthetic render resolution mismatch was fixed

The pseudo-query builder previously passed trajectory camera width/height through to NerfBaselines, which could render OldHospital synthetic views at 1920x1080 even when the RGB teacher map was trained/evaluated at a lower resolution. This made synthetic query generation inconsistent.

The pipeline now supports an explicit WildGaussians render resolution:

- `scripts/build_pseudo_query_manifest.py`: `--wildgaussians_render_resolution`
- `scripts/prepare_rgb_teacher_manifest.py`: `--render_resolution`
- `scripts/run_la_pseudo_query_pipeline.sh`: `WILDGAUSSIANS_RENDER_RESOLUTION`

The command template records and forwards `--resolution WIDTHxHEIGHT`.

## Smoke Results

OldHospital synthetic smoke with the repaired 8000-step checkpoint and explicit `960x540` rendering:

```text
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/OldHospital_synth_smoke_sh0_8000_960
```

Result:

```text
synthetic_rgb accepted: 3
synthetic_rgb rejected: 1
train_rgb accepted: 895
```

The four synthetic images have shape `540x960x3`, non-black means in the range `116.75-134.22`, and standard deviations in the range `58.67-67.84`.

The rejected sample had the highest artifact score:

```text
000000 accepted score 0.401
000001 accepted score 0.540
000002 rejected score 0.764
000003 accepted score 0.297
```

Visual checks:

```text
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/OldHospital_synth_smoke_sh0_8000_960/synthetic_visual_check.png
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/ShopFacade_wg_noapp_nosky_8000/visual_check/gt_vs_render_grid.png
/mnt/pool/sqy/stdloc_la_rgb_teacher_fix_v1/OldHospital_wg_sh0_noapp_8000/visual_check/gt_vs_render_grid.png
```

## Code Changes

- Added render resolution validation and command support in `la_artifacts/rgb_teacher.py`.
- Added explicit WildGaussians render resolution support in `scripts/build_pseudo_query_manifest.py`.
- Added explicit render resolution support in `scripts/prepare_rgb_teacher_manifest.py`.
- Added `WILDGAUSSIANS_RENDER_RESOLUTION` to `scripts/run_la_pseudo_query_pipeline.sh`.
- Added tests for resolution command generation and validation in `tests/test_la_artifacts.py`.

## Verification

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_la_artifacts.RgbTeacherManifestTest
Ran 7 tests - OK

PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile la_artifacts/rgb_teacher.py scripts/build_pseudo_query_manifest.py scripts/prepare_rgb_teacher_manifest.py
OK

bash -n scripts/run_la_pseudo_query_pipeline.sh
OK

git diff --check
OK
```

## Remaining Blocker

The all-black OldHospital bug is fixed, but WildGaussians RGB quality is still too blurry for reliable large-scale synthetic pseudo-query training. The next decision should be whether to:

1. keep WildGaussians only for low-risk synthetic interpolation after strict artifact filtering, or
2. replace the RGB teacher backend with a higher-fidelity RGB-only 3DGS method and keep the same manifest/cache interface.

GPU2 still has a stale driver-level `Not Found` context using about 18GB. The PID no longer exists in the process table, and `nvidia-smi --gpu-reset -i 2` failed with an unknown driver error. Current runs should avoid GPU2 until the host is rebooted or the driver is reset externally.
