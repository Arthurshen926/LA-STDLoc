# Data Preparation

Each scene uses a COLMAP-compatible layout:

```text
scene/
  processed/                 # RGB images used by SuperPoint
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin      # optional to LaFGS after camera loading
  dataset_test.txt           # Cambridge test image names and poses, or
  sparse/0/list_test.txt     # indoor pGT test image names
```

Text-form COLMAP files are accepted when binary files are absent. Camera poses
and RGB images used to construct a prior or localization map must come only from
the mapping split. `dataset_test.txt` (Cambridge) or `sparse/0/list_test.txt`
(indoor pGT) defines held-out evaluation queries.

LaFGS consumes the image dimensions and camera intrinsics recorded by COLMAP.
The canonical configuration uses full processed resolution (`resolution=1`,
`longest_edge=0`) and a `+0.5` pixel-center conversion at PnP.

An optional `processed/masks.pkl` may contain object, sky, and distortion-valid
masks. Masks are deployment inputs, not semantic reconstruction dependencies;
the method also supports RGB-only priors and no-mask experiments.

## Indoor datasets

Prepare one official 7Scenes or 12Scenes scene with:

```bash
python -m data.preparation \
  --dataset 7scenes \
  --source /data/7Scenes/chess \
  --output /data/lafgs/7Scenes/chess

python -m data.preparation \
  --dataset 12scenes \
  --source /data/12Scenes/apt1/kitchen \
  --output /data/lafgs/12Scenes/apt1_kitchen
```

For direct comparison with localization methods that use the published SfM
pseudo-ground-truth (pGT) camera registries, pass the corresponding reference
model explicitly:

```bash
python -m data.preparation \
  --dataset 7scenes \
  --source /data/7Scenes/chess \
  --reference-model /data/7scenes_reference_models/chess/sfm_gt \
  --output /data/lafgs_pgt/7Scenes/chess
```

This mode imports registered camera poses, intrinsics, the published test list,
and the complete `points3D` model. The reference point cloud initializes the
Gaussian prior, matching the STDLoc/ULF-Loc indoor contract. Per-image SfM
feature observations are discarded and `prior_input/images` contains mapping
RGB only, so test RGB never supervises Gaussian training. The manifest
explicitly records that the published point cloud may contain reconstruction
evidence from the full reference registry. Pass `--discard-reference-points`
only to reproduce the older pose-only/RGB-retriangulation protocol.

The camera model is calibrated and rectified to a same-resolution pinhole
domain matching the Gaussian renderer; source distortion coefficients and
remap statistics are recorded. Raw D-SLAM-pose, pose-only pGT, and full-SfM-pGT
results are separate protocols and must not be combined in one table.

The complete published pGT registries used by the paper contain 26,000 mapping
and 17,000 test images across seven 7Scenes scenes, and 16,989 mapping and 5,782
test images across twelve 12Scenes scenes. These are registry counts after
checking image availability and finite poses, not frame estimates from the raw
archives. Every scene manifest records the source camera model and parameters,
the mapping/test counts, and hashes of the reference camera registry and test
list.

Without `--reference-model`, 7Scenes RGB poses are calibrated from the original
depth-sensor poses with the published Kinect extrinsic and use the standard
525-pixel RGB focal length. For 12Scenes, each room's official `info.txt`
calibration and `split.txt` are used; sequence 0 is the official test capture
and all later sequences are used for mapping. Non-finite poses are rejected.
The generated `points3D.txt` is deliberately empty. `prior_input/` is a second
COLMAP tree containing only mapping RGB and poses. Run external RGB-only SfM
triangulation there before training an off-the-shelf vanilla 2DGS or 3DGS
prior. The test images are not present in that tree.

The canonical local full-reference layouts are recorded at
`/mnt/pool/sqy/7scenes/DATASET_LAYOUT.json` and
`/mnt/pool/sqy/12scenes/DATASET_LAYOUT.json`. They are non-destructive symlink
views over the downloaded raw datasets and published reference models. The
first end-to-end Stairs revalidation and the representative `office2/5b` prior
rebuild are documented in `docs/indoor_full_reference_prior_revalidation.md`.
