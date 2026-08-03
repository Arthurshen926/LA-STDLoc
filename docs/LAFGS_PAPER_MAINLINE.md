# LaFGS Paper Mainline

The release method is **Rendering-to-Localization Topology Distillation**.
`configs/lafgs_paper_mainline.yaml` is the source of truth and is validated by
`lafgs.protocol.MainlineProtocol` before release tooling runs.

## Offline pipeline

1. Load a frozen RGB Gaussian scaffold.
2. Extract native SuperPoint observations from mapping images.
3. Build KCS/GWFF initialization and Track-First cross-view tracks.
4. Robustly triangulate localization geometry and attach raster lineage.
5. Distill a track core plus Gaussian-supported coverage reserve.
6. Reconstruct one descriptor per final anchor through self-localization.

## Deployment pipeline

Deployment is deliberately fixed to native SuperPoint, global cosine top-1,
all correspondences, and one standard PoseLib PnP/RANSAC solve. It performs no
rendering, dense refinement, learned selection, or custom consensus.

## Package boundaries

- `lafgs.protocol`: frozen method and deployment contract.
- `lafgs.priors`: off-the-shelf prior adapters and held-out RGB evaluation.
- `lafgs.visualization`: manifest-driven paper figures with correspondence
  parity checks.
- `localization_training`: validated training primitives retained during the
  incremental migration.
- `scripts`: thin orchestration and compatibility entrypoints.

Historical feature-learning branches remain available for reproducing
ablations, but components listed in `method.excluded_by_default` are not part
of the release method.

## Release checks

```bash
python scripts/validate_lafgs_paper_mainline.py \
  --config configs/lafgs_paper_mainline.yaml \
  --output resolved_mainline_protocol.json
```

Paper figures must be generated from formal `frozen_results.json`, map states,
prior-quality reports, and their SHA256 hashes. The A0/A1 visualizer repeats
the native sparse correspondence graph and fails if match count or raw P@2
does not exactly reproduce the recorded evaluation.
