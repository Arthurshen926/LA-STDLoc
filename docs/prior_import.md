# Prior Import

RGB Gaussian reconstruction is external to LaFGS. This repository intentionally
does not vendor or wrap the training code for GraphDeco 3DGS, official 2DGS,
AnySplat, or MAtCHA.

`scripts/import_prior.py` validates and converts a producer PLY into:

```text
normalized_prior/
  point_cloud/iteration_30000/point_cloud.ply
  prior_manifest.json
  rgb_prior_manifest.json
  cfg_args
```

The adapter strips localization features and detector state, validates 2D/3D
scale cardinality and SH degree, removes non-finite primitives, and records
geometry/appearance hashes. The RGB geometry and appearance are frozen during
all LaFGS stages.

Supported producer labels are `vanilla_3dgs`, `vanilla_2dgs`, `anysplat`, and
`matcha`. AnySplat canonical-frame outputs must be mapping-only Sim(3)-aligned
before import; query/test cameras must not participate in that alignment.

Do not equalize primitive counts across 2DGS and 3DGS. Fair prior comparisons
hold mapping RGB, mapping poses, localization budget, SuperPoint, topology, A1,
and PoseLib settings fixed.
