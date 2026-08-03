# Third-Party Dependencies

- **PoseLib 2.0.5:** installed from PyPI and used through its Python absolute
  pose API. LaFGS carries no PoseLib source modifications or submodule.
- **gsplat 1.5.3:** installed as a Python package and used only for frozen RGB
  Gaussian rasterization/provenance.
- **SuperPoint:** the minimal model implementation and pretrained
  `superpoint_v1.pth` are bundled for deterministic frontend parity. Credit:
  Magic Leap's SuperPoint reference release.
- **PyTorch/OpenCV/Pillow:** tensor, image, and geometry runtime dependencies.

External reconstruction systems (GraphDeco 3DGS, official 2DGS, AnySplat, and
MAtCHA) are not dependencies of the LaFGS code. They only produce an input PLY
that is normalized by `scripts/import_prior.py`.
