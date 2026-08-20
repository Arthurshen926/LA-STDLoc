# 7Scenes Gaussian-prior audit

## Outcome

The indoor prior is usable, but the seven scenes are not on an equal input
contract.  The prepared official sparse models intentionally contain no 3D
points.  The pipeline therefore builds a new sparse model from **mapping RGB at
the published mapping poses**, then trains a vanilla 2DGS for 30,000 steps.  It
does not use test images.  The later rendered-Track localization route can be
source-image-free, but construction of the Gaussian prior is not.

This distinction matters: the prior is not an arbitrary geometry blob supplied
by 7Scenes, nor does it inherit official SfM points.  Its geometry comes from
known-pose RGB matching and ray triangulation; its appearance comes from the
same mapping RGB.  The empty reference file is explicit in
`data/preparation.py` and has the text `Reference points deliberately excluded`.

## Measured quality

The following values were recomputed directly from the COLMAP reconstructions
and the final 2DGS logs.  Reprojection statistics are per triangulated point.

| scene | views | points | observations | reprojection median / p90 px | mean track length | train PSNR / L1 | Gaussians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chess | 4,000 | 186,682 | 4,360,515 | 0.868 / 1.858 | 23.36 | 27.896 / 0.02706 | 511,062 |
| Fire | 2,000 | 199,969 | 3,701,384 | 0.516 / 1.406 | 18.51 | 27.589 / 0.02572 | 468,022 |
| Heads | 1,000 | 43,926 | 578,382 | 0.704 / 1.744 | 13.17 | 29.596 / 0.01963 | 530,183 |
| Office | 6,000 | 207,192 | 3,771,426 | 0.864 / 1.787 | 18.20 | 27.366 / 0.03067 | 626,534 |
| Pumpkin | 4,000 | 90,933 | 2,491,177 | 0.921 / 1.918 | 27.40 | 28.269 / 0.02404 | 315,948 |
| RedKitchen | 7,000 | 371,485 | 6,875,699 | 0.784 / 1.726 | 18.51 | **21.896 / 0.04769** | 687,211 |
| Stairs | 2,000 | 59,070 | 1,002,818 | **1.063 / 2.103** | 16.98 | **29.104 / 0.02393** | 350,765 |

The table exposes two different disadvantages.  RedKitchen has the weakest
appearance fit.  Stairs has a strong appearance fit but the weakest sparse
reprojection statistics and repeated geometry.  Thus mapping-view PSNR cannot
stand in for geometric reliability.

## Camera-model disadvantage

The original v1 import normalized camera parameters but did not rectify the
pixels with the calibrated distortion model.  The current v4 preparation
undistorts images to a same-resolution `PINHOLE` camera and publishes a shared
valid mask.  Only Fire and Stairs have been rebuilt under that contract:

| scene | version | points | reprojection median / p90 px | train PSNR / L1 | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: |
| Fire | v4 rectified | 212,129 | 0.453 / 1.305 | 28.276 / 0.02427 | 417,249 |
| Stairs | v4 rectified | 55,660 | 1.046 / 2.128 | 29.774 / 0.02254 | 319,972 |

The formal Stairs rendered-Track route already uses the v4 prior.  Chess,
Heads, Office, Pumpkin, and RedKitchen still carry the older camera-preparation
disadvantage and should not be used for a fair all-indoor comparison until they
are rebuilt.

## Remaining quality gaps

- No held-out mapping-trajectory render/depth audit currently separates
  memorized training-view appearance from usable view synthesis.
- Neither prior version includes semantic masks.
- The current artifact proxy depends on 2DGS distortion, so it is not yet a
  generic contract for 3DGS or AnySplat.
- Repeated structure can yield coherent but wrong Tracks even when RGB render
  quality is high; Stairs demonstrates this directly.

The next prior-level improvement is therefore mechanical, not another
localization threshold sweep: rebuild the remaining five scenes with v4
rectification and add a mapping-only held-out-trajectory audit covering RGB,
depth consistency, reprojection, alpha/valid-mask coverage, and pose bins.
Test images remain evaluation-only.

The exact machine-readable measurements and source hashes are stored in
`docs/evidence/7scenes_gaussian_prior_audit.json`.
