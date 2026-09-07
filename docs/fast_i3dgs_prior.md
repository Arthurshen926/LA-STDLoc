# Fast i3DGS prior for AnyGSLoc

This is the few-minute speed preset. For the currently recommended
fast-quality preset and measured downstream comparison, see
[`fast_generative_gaussian_prior.md`](fast_generative_gaussian_prior.md).

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
- GraphDeco 3DGS: `54c035f7834b564019656c3e3fcc3646292f727d`
- Local runtime patches:
  [`i3dgs_local_hub.patch`](../third_party_patches/i3dgs_local_hub.patch) and
  [`megaloc_fp16_sinkhorn.patch`](../third_party_patches/megaloc_fp16_sinkhorn.patch)
- Warm-start refinement patch:
  [`graphdeco_warmstart.patch`](../third_party_patches/graphdeco_warmstart.patch)

The first patch makes pretrained XFeat/MegaLoc loading independent of GitHub
Hub rate limits. The second preserves the input dtype in MegaLoc's Sinkhorn
normalization and prevents a float32/float16 runtime mismatch.
The GraphDeco patch adds a complete Gaussian-Ply warm start and avoids the
upstream million-tuple PLY serialization path.

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

## Short mapping-only global consistency pass

The raw online reconstruction is fast, but on ShopFacade its test-view
structure was not strong enough for localization. Keep the i3DGS geometry as
the warm start and run a short, known-pose, full-resolution global pass. This
stage still consumes mapping images only. It does not optimize per-camera
exposure because that state is not exported into the AnyGSLoc prior, and it
does not densify because the warm start already contains roughly one million
primitives.

Prepare the mapping split at native resolution:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.prepare_i3dgs_cambridge \
  --dataset /mnt/pool/sqy/Cambridge_stdloc/ShopFacade \
  --images processed --downscale 1 \
  --output /path/to/i3dgs_input_d1
```

Apply the pinned patch to a clean GraphDeco checkout, install its official
rasterizer into an isolated environment, and refine for 1000 steps:

```bash
cd /root/third_party_lafgs_priors/gaussian-splatting
git apply /root/STDLoc/third_party_patches/graphdeco_warmstart.patch

CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/root/miniconda3/envs/g4splat/lib:/root/miniconda3/envs/g4splat/lib/python3.9/site-packages/torch/lib:/usr/local/cuda/lib64 \
/root/miniconda3/envs/g4splat/bin/python train.py \
  -s /path/to/i3dgs_input_d1 -m /path/to/global_joint_d1_ft1000 \
  --init_ply /path/to/export/gaussians.ply \
  --images images --resolution 1 --sh_degree 3 --iterations 1000 \
  --position_lr_init 0.00008 --position_lr_final 0.0000008 \
  --position_lr_max_steps 1000 --scaling_lr 0.0025 \
  --rotation_lr 0.0005 --opacity_lr 0.0125 --feature_lr 0.0025 \
  --exposure_lr_init 0 --exposure_lr_final 0 \
  --depth_l1_weight_init 0 --depth_l1_weight_final 0 \
  --densify_until_iter 0 --lambda_dssim 0.2 \
  --test_iterations 1001 --save_iterations 1000 \
  --disable_viewer --quiet
```

On ShopFacade this pass took about 67 seconds of optimization on an RTX 3090.
The output PLY is then normalized with `scripts.import_prior` and evaluated by
the unchanged AnyGSLoc mapping and localization pipeline.
