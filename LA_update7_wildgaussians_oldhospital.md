# LA_update7: WildGaussians RGB Teacher Controlled Fix

Date: 2026-06-27

## Scope

This update closes the immediate WildGaussians RGB-teacher issue for OldHospital synthetic RGB generation. The goal was to explain why the previous appearance/uncertainty choices produced unusable renders, run controlled alternatives, and land a reproducible fallback preset.

## Root Cause

WildGaussians appearance rendering is not safe by default for synthetic/no-target views on OldHospital. For synthetic RGB query generation there is no target image to optimize an appearance embedding against, so the render path effectively uses the no-target embedding path. In controlled OldHospital runs this produced near-black/broken renders and extremely poor official-test quality, even before the later 3D-filter crash.

Uncertainty was kept disabled in this round to isolate the appearance failure first. The failure reproduced with `appearance_enabled=true` and `uncertainty_mode=disabled`, so uncertainty was not the primary cause of the black OldHospital renders.

## Controlled Results

| Run | Key settings | Status | Official test result |
| --- | --- | --- | --- |
| `app_nounc_scale10xdown_nodensify_3p5k_smoke` | appearance on, uncertainty off, sky50k, no densify, scale lr down | failed at final 3D filter | eval-few PSNR 4.7968, SSIM 0.0048 |
| `app_nounc_kernel001_nodensify_3p5k_smoke` | appearance on, uncertainty off, smaller kernel, no densify | failed at final 3D filter | eval-few PSNR 4.7968, SSIM 0.0048 |
| `noapp_nounc_sky_stopdens7k_15k_960` | appearance off, uncertainty off, no explicit sky override in result, stop densify at 7k | completed | PSNR 13.7338, SSIM 0.3945, LPIPS 0.5828 |
| `noapp_nounc_sky50k_stopdens7k_15k_960` | appearance off, uncertainty off, sky50k, stop densify at 7k | completed | PSNR 13.6778, SSIM 0.3944, LPIPS 0.5812 |
| `noapp_nounc_nosky_stopdens7k_30k_960` | appearance off, uncertainty off, no explicit sky override, stop densify at 7k | completed | PSNR 14.1108, SSIM 0.4106, LPIPS 0.5519 |

Conclusion: for OldHospital, more iterations help only after removing the appearance path. The checked config still reports WildGaussians' default `num_sky_gaussians: 50000`, so the practical distinction in these runs is explicit override vs default rather than proven physical removal of sky gaussians.

## Landed Changes

- Added `oldhospital_noapp_nosky_30k_v1` in `scripts/prepare_rgb_teacher_manifest.py`.
- Added explicit `--train_output_path` support so a manifest can reproduce non-default controlled experiment directories.
- Fixed `scripts/run_la_pseudo_query_pipeline.sh` so `RGB_TEACHER_WILDGAUSSIANS_PRESET` and semicolon-separated `RGB_TEACHER_WILDGAUSSIANS_SET` are applied to both manifest generation and actual WildGaussians training.
- Added `la_artifacts/nerfbaselines_visuals.py` and `scripts/visualize_nerfbaselines_predictions.py` for GT/render grid checks from NerfBaselines prediction archives.

## Current Ready Artifact

- Manifest: `/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/OldHospital_wg_noapp_nounc_nosky_stopdens7k_30k_960/rgb_teacher_manifest.json`
- Checkpoint: `/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/OldHospital_wg_noapp_nounc_nosky_stopdens7k_30k_960/checkpoint-30000`
- Visual check: `/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/OldHospital_wg_noapp_nounc_nosky_stopdens7k_30k_960/visual_check/gt_vs_render_grid.png`

## Recommendation

Use `oldhospital_noapp_nosky_30k_v1` as the current OldHospital RGB-teacher fallback for pseudo-query generation. Treat `nosky` in the experiment name as "no explicit sky override"; the checkpoint config still carries the WildGaussians default sky setting. Do not use `appearance_enabled=true` for OldHospital synthetic RGB in v1 unless the render path is changed to provide a reliable target-conditioned or nearest-train appearance embedding.

The remaining quality issue is blur and thin-structure/occluder mismatch, not black-frame failure. The next optimization should therefore move to appearance-conditioned rendering for real train images or nearest-neighbor appearance interpolation for synthetic views, with explicit render QA before admitting synthetic queries.
