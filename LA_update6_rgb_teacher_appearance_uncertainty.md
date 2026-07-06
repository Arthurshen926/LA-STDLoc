# LA_update6: RGB Teacher appearance / uncertainty control closure

## Question

Why did we disable WildGaussians `appearance` and `uncertainty`, and should that remain the default?

## Short Answer

No. For ShopFacade, the controlled evidence now shows that disabling `appearance` and `uncertainty` was a temporary confound-removal step, not the correct final default.

The high-impact root cause of the dark WildGaussians renders is the default densification schedule continuing past 7k (`densify_until_iter=35000`). Freezing densification at 7k keeps RGB brightness and quality stable through 15k.

## Evidence

Fixed 3-train-view render audit:

| Config | Step | PSNR | MAE | GT mean | Render mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| default app+dino, default densify | 7k | 13.008 | 0.153 | 0.487 | 0.490 |
| default app+dino, default densify | 15k | 6.463 | 0.418 | 0.487 | 0.073 |
| default app+dino, default densify | 30k | 6.263 | 0.428 | 0.487 | 0.062 |
| app+nounc, stopdens7k | 7k | 12.987 | 0.153 | 0.487 | 0.493 |
| app+nounc, stopdens7k | 15k | 12.905 | 0.155 | 0.487 | 0.501 |
| app+dino, stopdens7k | 7k | 13.009 | 0.153 | 0.487 | 0.490 |
| app+dino, stopdens7k | 15k | 12.827 | 0.157 | 0.487 | 0.493 |

Official 103-image ShopFacade eval at 15k:

| Config | PSNR | SSIM | MAE | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| noapp+nounc, stopdens7k | 13.959 | 0.420 | 0.149 | 0.628 |
| app+nounc, stopdens7k | 13.961 | 0.415 | 0.151 | 0.625 |
| app+dino, stopdens7k | 13.664 | 0.409 | 0.155 | 0.634 |

Visual audit:

`/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/appearance_embedding_smoke/appearance_uncertainty_stopdens_visual_grid.png`

Metric JSON:

`/mnt/pool/sqy/stdloc_la_rgb_teacher_control_v1/appearance_embedding_smoke/appearance_uncertainty_stopdens_metrics.json`

## Decision

- `cambridge_stable_v1` now means `appearance_enabled=true`, `uncertainty_mode=disabled`, `num_sky_gaussians=50000`, `densify_until_iter=7000`.
- The old no-appearance/no-uncertainty recipe is preserved as `cambridge_legacy_noapp_nounc_v1`.
- `cambridge_app_nounc_v1` is retained as an explicit alias for the current stable recipe.
- `cambridge_app_dino_v1` is retained as an uncertainty ablation. DINO uncertainty is not the dark-render root cause, but it was slower and slightly worse on ShopFacade, so it is not the current default.

## Caveat

This closes the ShopFacade confound. OldHospital still needs the same controlled rerun, because earlier appearance-enabled OldHospital runs hit a separate `compute_3D_filter` valid-points crash near 3k. Do not claim the appearance/uncertainty decision is globally closed for all Cambridge scenes until OldHospital is rerun under the stopdens7k preset.
