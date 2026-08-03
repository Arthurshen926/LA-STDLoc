# Data Preparation

Each scene uses a COLMAP-compatible layout:

```text
scene/
  processed/                 # RGB images used by SuperPoint
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin      # optional to LaFGS after camera loading
  dataset_test.txt           # Cambridge test image names and poses
```

Text-form COLMAP files are accepted when binary files are absent. Camera poses
and RGB images used to construct a prior or localization map must come only from
the mapping split. `dataset_test.txt` defines held-out evaluation queries.

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

This mode imports only registered camera poses, intrinsics, and the published
test list. Reference 3D points and feature observations are discarded. The
camera model is normalized to its pinhole focal length and principal point,
matching the undistorted-camera assumption used by the Gaussian renderer;
source distortion coefficients are recorded but not applied. The
generated `prior_input/` contains only mapping images and cameras, so the RGB
Gaussian prior must be triangulated and reconstructed again without test
images. Raw D-SLAM-pose and SfM-pGT results are separate protocols and must not
be combined in one table.

Without `--reference-model`, 7Scenes RGB poses are calibrated from the original
depth-sensor poses with the published Kinect extrinsic and use the standard
525-pixel RGB focal length. For 12Scenes, each room's official `info.txt`
calibration and `split.txt` are used; sequence 0 is the official test capture
and all later sequences are used for mapping. Non-finite poses are rejected.
The generated `points3D.txt` is deliberately empty. `prior_input/` is a second
COLMAP tree containing only mapping RGB and poses. Run external RGB-only SfM
triangulation there before training an off-the-shelf vanilla 2DGS or 3DGS
prior. The test images are not present in that tree.
