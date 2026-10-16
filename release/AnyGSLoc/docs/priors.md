# Gaussian priors and image quality

AnyGSLoc consumes a Gaussian prior; it does not train a scene-specific localization feature field. The exact accepted PLY files are bound by SHA-256 in `reproduction/assets.json`, with geometry type, SH degree, and localization-render background in the corresponding configuration. This is the reproducible starting point for all downstream paper experiments.

| Configuration | Prior used in the paper | Localization rendering |
|---|---|---|
| `cambridge.json` | vanilla 3DGS, 30,000 RGB-only optimization steps | black background |
| `prior_matcha.json` | accepted MAtCha 2DGS export | white background |
| `prior_vanilla_2dgs.json` | vanilla 2DGS, 30,000 optimization steps | black background |
| `prior_i3dgs_raw.json` | i3DGS exported prior | per-config native background |
| `prior_i3dgs_1k.json` | i3DGS followed by 1,000 RGB optimization steps | per-config native background |
| `prior_long_lrm_raw.json` | Long-LRM exported prior | white background |
| `prior_long_lrm_1k.json` | Long-LRM followed by 1,000 RGB optimization steps | per-config native background |
| `7scenes.json`, `12scenes.json` | retained indoor 2DGS priors in the SfM pseudo-GT frame | black background |

Consult the JSON rather than inferring a renderer from a PLY filename. Priors differ in initialization, input preparation, and native training recipe; this is a downstream sensitivity experiment. No per-prior best-of-Full/Compact selection is performed. The repaired St. Mary's Church priors and all 530 test queries are included.

## Downstream reconstruction and localization

```bash
python -m reproduction.run main --config configs/prior_vanilla_2dgs.json --execute
python -m anygsloc reproduce report --config configs/prior_vanilla_2dgs.json
```

Repeat for the seven configurations listed in [reproduction.md](reproduction.md). All Cambridge priors use unmasked rendered SuperPoint top-2048 followed by V2 filtering and the same construction/selector/backend. The raw/1k prior pair changes the upstream Gaussian scene, not an AnyGSLoc fine-tuning phase.

## Reference-view PSNR

```bash
python -m reproduction.run psnr --config configs/cambridge.json --execute
```

For each of 5,364 mapping/reference images, render at the native registered image size and pose, compare against decoded RGB uint8 values divided by 255, and compute `-10 log10(mean squared RGB error)`. The prediction is float32 RGB clamped to [0,1]; the error is computed in float64. There is no semantic mask, crop, exposure fit, or quantization of the predicted image. Average image PSNR within each scene, then average the five scene means equally. Do not compute PSNR from a pooled MSE and do not mix reference and test views.

Quality rendering uses a **black background for every prior**, independently of localization-render background. This includes Long-LRM raw. Use gsplat 1.5.3, native 2DGS rasterization, and antialiased 3DGS rasterization. Tile size is 16; 2DGS retries at 32 only after an out-of-memory exception. Each per-image result records its tile size.

The accepted scene-equal reference PSNR is 16.71590 dB for vanilla 3DGS, 15.45854 for MAtCha, 15.69779 for vanilla 2DGS, 13.65977/15.13806 for i3DGS raw/+1k, and 13.55941/15.34247 for Long-LRM raw/+1k. These values describe training/reference appearance, not held-out novel-view generalization.

## Rebuilding upstream priors

The primary experiment recipe starts from the supplied PLY assets. Rebuilding the producers is a separate task and requires their original source/checkpoints and prepared camera inputs. The local maintainer inventory retains the original producer manifests and training argv, where available. They are not replaced with a generic trainer invocation that would change the initialization or camera convention.

For vanilla 3DGS the accepted Cambridge runs use 30,000 iterations, SH degree 3, resolution 1, CPU data storage, depth-loss weights zero, black background, and mapping-only prepared images. The prepared sparse initialization retains only points whose **complete original track** contains mapping images only and has at least two observations. Camera poses/intrinsics must agree with the prepared rectified mapping-image domain. A training run on the original unrectified COLMAP cameras is not equivalent, especially for St. Mary's Church.

MAtCha, i3DGS, and Long-LRM require their native preprocessing/export and coordinate transforms; their exported PLY hashes are the paper's interface. The source candidate does not claim that these upstream neural producers were retrained during release validation. The exact downstream experiments are executable from their retained priors; publishing those prior bundles is part of the remaining release work.
