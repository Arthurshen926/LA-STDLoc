# Third-Party Dependencies

- **PoseLib 2.0.5:** installed from PyPI and used through its Python absolute
  pose API. LaFGS carries no PoseLib source modifications or submodule.
- **gsplat 1.5.3:** installed as a Python package and used only for frozen RGB
  Gaussian rasterization/provenance.
- **SuperPoint:** the minimal compatible model implementation is included, but
  Magic Leap's `superpoint_v1.pth` is not redistributed. Its upstream license
  limits use and redistribution. Users must obtain the weight from the
  [official repository](https://github.com/magicleap/SuperPointPretrainedNetwork),
  accept its terms, and set `LAFGS_SUPERPOINT_WEIGHTS` or place it at
  `~/.cache/lafgs/superpoint_v1.pth`. LaFGS requires SHA256
  `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`.
- **FeatureBooster:** an inference-only, parameter-name-compatible adaptation of
  the official Apache-2.0 implementation is included for mapping-only
  diagnostics. The official `SuperPoint+Boost-F.pth` is not redistributed.
  Obtain it from the
  [FeatureBooster repository](https://github.com/SJTU-ViSYS/FeatureBooster),
  set `LAFGS_FEATUREBOOSTER_WEIGHTS` or place it at
  `~/.cache/lafgs/SuperPoint+Boost-F.pth`. LaFGS requires SHA256
  `5334d9aa861e877a2b99baff0d682e1ac8a749cdd65eb1d4b8bd0a8bb8bf0359`.
- **PyTorch/OpenCV/Pillow:** tensor, image, and geometry runtime dependencies.

External reconstruction systems (GraphDeco 3DGS, official 2DGS, AnySplat, and
MAtCHA) are not dependencies of the LaFGS code. They only produce an input PLY
that is normalized by `scripts/import_prior.py`.
