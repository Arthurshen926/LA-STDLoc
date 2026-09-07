# Fast generative Gaussian priors

AnyGSLoc treats reconstruction as an external, frozen RGB Gaussian prior. The
prior builder may be feed-forward or optimization-based, but it must consume
mapping images only. Test queries, localization results, descriptor losses,
and map feedback are excluded.

## Recommended presets

Two presets are retained because they serve different latency/quality points:

- **Quality preset:** MAtCha/MASt3R initialization followed by an early-stopped
  2DGS refinement at iteration 7000. Use
  [`matcha_7k_fast.yaml`](../configs/priors/matcha_7k_fast.yaml).
- **Speed preset:** i3DGS fixed-pose reconstruction followed by a 1000-step
  full-resolution GraphDeco pass. Use
  [`i3dgs.yaml`](../configs/priors/i3dgs.yaml) and the detailed recipe in
  [`fast_i3dgs_prior.md`](fast_i3dgs_prior.md).

The i3DGS preset is useful when construction must finish in a few minutes. It
is not currently the quality default: on ShopFacade its downstream
localization remained substantially worse than the early-stopped MAtCha prior.

## Frozen ShopFacade validation

All rows below use the same AnyGSLoc map construction, 2048 mapping/query
keypoints, identity metric, PoseLib configuration, and official 103-query test
split. No offline localization feedback is enabled.

| RGB prior | Gaussian optimization | Test PSNR / SSIM | Median TE / RE | R5 | Mean localization |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw i3DGS | about 111 s including 3 short fine-tune epochs | 12.65 / 0.353 | 12.24 cm / 0.363 deg | 1.94% | 128 ms |
| i3DGS + full-resolution global 1k | about 67 s after initialization | 13.51 / 0.436 | 6.94 cm / 0.165 deg | 24.27% | 95 ms |
| **MAtCha 7k early stop** | about 20.5 min for the Gaussian stage | **14.12 / 0.529** | **2.17 cm / 0.109 deg** | **83.50%** | 158 ms |
| historical MAtCha 30k | about 75 min for the Gaussian stage | 13.93 / 0.534 | 1.86 cm / 0.091 deg | 83.50% | 83 ms |

The MAtCha 7k result is the current fast-quality choice. The 7k map contains
29,724 anchors and 331,882 mapping observations, with 85.31% of rendered
SuperPoint rows retained. The 30k row is a historical reference, not an input
to preset selection.

The reported 20.5 minutes covers the final Gaussian optimization on the
observed RTX 3090 run. The reusable MASt3R/chart preparation took roughly an
additional 12 minutes on that run; dataset staging and checkpoint I/O are
machine-dependent. Therefore a fresh end-to-end reconstruction should be
described as roughly half an hour, not as a 20-minute pipeline.

A 12-view test-render screen found no scene-specific collapse in the four
available 7k checkpoints:

| Scene | Median PSNR | Median SSIM | Median MAE |
| --- | ---: | ---: | ---: |
| GreatCourt | 15.29 | 0.698 | 0.128 |
| KingsCollege | 14.22 | 0.557 | 0.138 |
| OldHospital | 14.24 | 0.522 | 0.134 |
| ShopFacade | 14.12 | 0.529 | 0.135 |

This four-scene screen establishes reconstruction stability, but only
ShopFacade has completed the unchanged end-to-end localization evaluation in
this round. StMarysChurch had no retained 7k checkpoint and is deliberately
not back-filled with its 30k output.
The exact artifact hashes and metrics are frozen in
[`fast_generative_prior_shopfacade.json`](evidence/fast_generative_prior_shopfacade.json).

## MAtCha 7k materialization

Starting from a mapping-only MAtCha scene and its posed dense mapping set, run
the same 30k-schedule refinement but terminate immediately after saving
iteration 7000. Apply
[`matcha_schedule_preserving_early_stop.patch`](../third_party_patches/matcha_schedule_preserving_early_stop.patch)
to expose this behavior. Keeping `--iterations 30000` is important: changing
the optimization horizon to 7000 also changes learning-rate schedules and is
not the experiment validated here. The validated ShopFacade command was
equivalent to:

```bash
cd /root/MAtCha
python 2d-gaussian-splatting/train_with_charts.py \
  -s /path/to/mapping_only/mast3r_sfm \
  -m /path/to/output/free_gaussians \
  --iterations 30000 --early_stop_iteration 7000 \
  --save_iterations 7000 --test_iterations 7000 \
  --densify_until_iter 15000 --opacity_reset_interval 3000 \
  --depth_ratio 0.5 --use_mip_filter \
  --dense_data_path /path/to/posed_mapping_train_opt \
  --normal_consistency_from 7000 --distortion_from 3000 \
  --depthanythingv2_checkpoint_dir /path/to/Depth-Anything-V2/checkpoints \
  --depthanything_encoder vitl --dense_regul default \
  --cambridge_mask_pickle /path/to/mapping/masks.pkl \
  --cambridge_mask_dataset_path /path/to/posed_mapping_train_opt \
  --cambridge_mask_indices 0 2 \
  --rgb_loss_type l1 --rgb_charbonnier_eps 0.001
```

Normalize the output into the immutable AnyGSLoc prior contract:

```bash
cd /root/STDLoc
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.import_prior \
  --input_ply /path/to/free_gaussians/point_cloud/iteration_7000/point_cloud.ply \
  --output_model /path/to/normalized_matcha7k \
  --gaussian_type 2dgs --sh_degree 3 --iteration 7000 \
  --source_path /path/to/cambridge_scene --images processed \
  --longest_edge 1600 --prior_kind rgb_only --black_background \
  --source_method matcha --source_commit <pinned-commit>
```

Then validate both the held-out rendering and the actual localization task.
The localization check is mandatory because PSNR/SSIM alone do not measure
cross-view sparse-feature consistency:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.run_anygsloc_scene \
  --scene ShopFacade --dataset /path/to/ShopFacade --images processed \
  --prior-manifest /path/to/normalized_matcha7k/rgb_prior_manifest.json \
  --output /path/to/anygsloc_matcha7k \
  --config /root/STDLoc/configs/anygsloc_mainline.yaml \
  --gpu 0 --seed 2026 --audit-shards 1 \
  --mapping-raw-keypoints 2048 --mapping-retained-keypoints 2048
```

## Rejected variants

- Appearance-only optimization improved the raw i3DGS prior but did not repair
  its geometry sufficiently.
- Extending the joint pass from 1k to 3k steps did not improve the downstream
  result consistently.
- Re-enabling densification after the 1k i3DGS pass increased primitive count
  but reduced novel-view quality.
