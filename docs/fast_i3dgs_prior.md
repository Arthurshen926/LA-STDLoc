# Fast i3DGS prior for AnyGSLoc

This route treats the Gaussian reconstruction as an external, RGB-only prior.
Only Cambridge mapping images are consumed; test images and localization
outcomes never enter reconstruction. The adapter converts repeated sequence
paths into collision-free flat names, undistorts the mapping images, and writes
an exact COLMAP text model. The exporter removes i3DGS hierarchy fields and
transforms positions, scales, rotations, and degree-3 spherical harmonics back
to the original mapping world frame.

## Pinned sources

- i3DGS: `cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42`
- MegaLoc: `5fe0dd6` (runtime dependency only)
- Local runtime patches:
  [`i3dgs_local_hub.patch`](../third_party_patches/i3dgs_local_hub.patch) and
  [`megaloc_fp16_sinkhorn.patch`](../third_party_patches/megaloc_fp16_sinkhorn.patch)

The first patch makes pretrained XFeat/MegaLoc loading independent of GitHub
Hub rate limits. The second preserves the input dtype in MegaLoc's Sinkhorn
normalization and prevents a float32/float16 runtime mismatch.

## ShopFacade quality-validation recipe

Prepare the exact mapping split at half resolution:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.prepare_i3dgs_cambridge \
  --dataset /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  --images processed --downscale 2 \
  --output /mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/i3dgs_input_d2
```

Run i3DGS with fixed mapping poses. `--max_active_keyframes 250` retains all
ShopFacade keyframes for the short global appearance fine-tuning phase:

```bash
cd /root/third_party_lafgs_priors/i3dgs
CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/usr/local/cuda-11.8 \
HF_HOME=/mnt/pool/sqy/huggingface_cache TORCH_HOME=/mnt/pool/sqy/torch_cache \
I3DGS_XFEAT_ROOT=/root/third_party_lafgs_priors/i3dgs/submodules/xfeat \
I3DGS_MEGALOC_ROOT=/root/third_party_lafgs_priors/MegaLoc \
PATH=/usr/local/cuda-11.8/bin:/root/miniconda3/envs/i3dgs/bin:/usr/bin:/bin \
/root/miniconda3/envs/i3dgs/bin/python -u train.py \
  -s /mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/i3dgs_input_d2 \
  -m /mnt/pool/sqy/anygsloc_fast_prior_20260904/ShopFacade/i3dgs_sh3_globalft3_d2 \
  --use_colmap_poses --fix_focal --lr_poses 0 --downsampling 1 \
  --num_loader_threads 4 --max_active_keyframes 250 --no-hierarchy \
  --no-use_rotated_descriptors --sh_degree 3 --num_iterations 30 \
  --save_at_finetune_epoch 1 3 --display_runtimes
```

Export a standard 3DGS PLY in mapping-world coordinates:

```bash
cd /root/STDLoc
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.export_i3dgs_prior \
  --hierarchy-ply /path/to/i3dgs/output/3/hierarchy.ply \
  --prepared-manifest /path/to/i3dgs/input/anygsloc_i3dgs_input.json \
  --output-ply /path/to/export/gaussians.ply \
  --output-manifest /path/to/export/i3dgs_export.json
```

The final prior must then pass `scripts.import_prior` and
`scripts.audit_gaussian_prior_views` before AnyGSLoc map construction. A low
training-time pose error is not accepted as a substitute for rendered-view or
localization validation.
