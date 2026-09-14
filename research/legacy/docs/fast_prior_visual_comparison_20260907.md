# Measured prior-render comparison and selection

## Recommendation

For the specific claim that AnyGSLoc accepts an externally reconstructed,
feed-forward Gaussian prior, use **Long-LRM raw** as the current primary
feed-forward input. Freeze the exported world-coordinate PLY, build Anchors,
and localize without reconstruction gradients or localization-driven changes
to that prior. Treat **Long-LRM + 1000 mapping-only optimization steps** as a
separately named hybrid prior, not a pure feed-forward result.

This selects a usable existing implementation, not a demonstrated universally
best reconstruction model. Only ShopFacade has the completed Long-LRM
rendering/localization comparison. Its 32-image posed input does not establish
coverage of arbitrarily large, multi-sequence scenes. It uses existing mapping
COLMAP cameras; it is not a pose-free reconstruction pipeline. The checkout is
the authors' published self-reimplementation, pinned in `fast_long_lrm_prior.md`.

i3DGS is an incremental optimization-based reconstruction method, not a
feed-forward Gaussian predictor. It remains a valid *fast reconstruction*
reference, but must not represent the feed-forward category.

## Visual protocol

Artifacts: `/mnt/pool/sqy/anygsloc_fast_prior_20260904/visual_comparison_20260907/`.
Reproduce with `python -m scripts.compose_fast_prior_comparison` in `g4splat`.
The JSON manifest records every source report and displayed sample index.

- ShopFacade compares exactly the same 12 uniformly sampled test cameras.
  The overview displays indices 0, 4, 7, 11 independent of per-view scores;
  the full sheet includes all 12. Crops use the same central 40% rectangle.
- Stored reference/render JPEGs are composed without color or contrast edits.
  PSNR/SSIM values come from the original float-render reports, not JPEGs.
- Raw Long-LRM uses white background; both optimized methods use black.
  Therefore raw-to-optimized appearance changes include a background change.
- AnySplat has no existing ShopFacade PLY in the searched experiment roots.
  KingsCollege and OldHospital are shown separately, with both the original
  window-aligned prior and the complete-mapping-coverage variant. Neither
  uses photometric post-optimization according to its saved protocol.
  These different scenes cannot establish a head-to-head model ranking.
- This visualization is reporting only. Test RGB and poses are used to render
  evaluation views, not as reconstruction inputs or parameter updates.

## ShopFacade evidence

| Prior | Median PSNR / SSIM, 12 test views | Median TE / RE, all 103 queries |
| --- | --- | --- |
| Long-LRM raw | 13.8707 / 0.41722 | 8.4492 cm / 0.24684 deg |
| Long-LRM + 1k | 14.1954 / 0.45983 | 7.0750 cm / 0.19617 deg |
| i3DGS + 1k | 13.5121 / 0.43636 | 6.9438 cm / 0.16495 deg |

Long-LRM raw retains recognizable storefront structure but has blurry edges
and duplicated thin structures. Optimization reduces this in the displayed
views. i3DGS shows sharper texture along with fragmented foreground/boundary
artifacts. Rendering scores alone do not predict localization: Long-LRM + 1k
has higher PSNR/SSIM but slightly worse median pose errors than i3DGS + 1k.
See `evidence/long_lrm_shopfacade_20260907.json` for exact localization sources.

Raw Long-LRM's measured mean localization was 408 ms versus about 96 ms after
refinement, largely from increased RANSAC work on the raw prior. The measured
initial construction took 861.73 s including setup, preprocessing and export;
it must not be presented as a measured one-second end-to-end pipeline.

## Method sources

The additional AnySplat audits completed on the same 12 cameras per scene:

| Scene | Original window prior PSNR / SSIM | Complete mapping coverage PSNR / SSIM |
| --- | --- | --- |
| KingsCollege | 10.6453 / 0.18872 | 9.5304 / 0.15650 |
| OldHospital | 9.8335 / 0.17701 | 9.5104 / 0.13817 |

The complete-coverage variant does not improve these sampled medians. Both
variants retain severe streaks and overlapping building contours. This is
evidence against simply increasing window coverage in this fusion pipeline,
not evidence that more input images intrinsically harm AnySplat.

- Long-LRM: https://arthurhero.github.io/projects/llrm/index.html
- i3DGS: https://repo-sam.inria.fr/nerphys/i3dgs/

Do not infer that the current AnySplat window-fusion artifacts demonstrate a
fundamental failure of the pretrained AnySplat model. They also depend on our
windowing, alignment, fusion and pruning pipeline. Keeping this extra
reconstruction engineering out of AnyGSLoc's contribution is a reason to
prefer the already usable posed Long-LRM input, not a model-quality proof.
