# LaFGS

LaFGS reconstructs a compact sparse localization map from an off-the-shelf RGB
Gaussian scene prior and posed mapping images. Rendering primitives are treated
as surface support, not as localization identities. The identities and PnP
geometry are reconstructed from cross-view SuperPoint tracks.

The frozen paper pipeline is:

```text
RGB Gaussian prior (frozen)
  -> native SuperPoint mapping observations
  -> KCS/GWFF initialization
  -> Track-First matching and robust triangulation
  -> Gaussian raster provenance
  -> Track core + Gaussian coverage reserve
  -> self-localization-guided descriptor reconstruction
  -> compact single-descriptor localization map
```

Online localization deliberately remains minimal:

```text
native SuperPoint -> global cosine top-1 -> one standard PoseLib PnP/RANSAC
```

There is no scene-specific detector, dense refinement, test-time rendering, or
custom pose solver in the release path.

## Install

Python 3.9 and CUDA 11.8 are the reference environment.

```bash
pip install -e '.[test]'
```

`poselib==2.0.5` and `gsplat==1.5.3` are ordinary Python dependencies. No
third-party repository or Git submodule is required. See
[`docs/third_party.md`](docs/third_party.md).

## Inputs

LaFGS does not train an RGB Gaussian model. Reconstruct one with an external
3DGS, 2DGS, AnySplat, or MAtCHA checkout, then normalize its PLY:

```bash
python scripts/import_prior.py \
  --input_ply /path/to/external/point_cloud.ply \
  --output_model /path/to/normalized_prior \
  --gaussian_type 2dgs --sh_degree 3 \
  --source_path /data/Cambridge/OldHospital \
  --prior_kind rgb_only --source_method vanilla_2dgs
```

Mapping and test cameras follow the COLMAP layout documented in
[`docs/data_preparation.md`](docs/data_preparation.md). Test images never enter
prior reconstruction, evidence construction, or map learning.

## Reconstruct

The canonical command runs the complete frozen method and, by default,
evaluates the official test split:

```bash
python scripts/run_pipeline.py \
  --dataset /data/Cambridge/OldHospital \
  --prior /data/priors/OldHospital/vanilla_2dgs \
  --gaussian-type 2dgs \
  --output /data/runs/OldHospital
```

For large mapping sets, `--function-graph-shards N` evaluates independent
query shards concurrently and deterministically merges them. It is an exact
execution backend; it does not change the frozen method or per-query seeds.

The seven stable CLIs under `scripts/` expose the same stages individually.
All defaults are resolved through [`configs/paper_mainline.yaml`](configs/paper_mainline.yaml).

## Evaluate

```bash
python scripts/evaluate.py \
  --dataset /data/Cambridge/OldHospital \
  --map /data/runs/OldHospital/map_learning/anchor_map_step_0175.pt \
  --metric-state /data/runs/OldHospital/map_learning/metric_state_step_0175.pt \
  --output /data/runs/OldHospital/evaluation
```

The release parity fixture and frozen ShopFacade/OldHospital metrics live in
`paper_baseline/`; large maps and checkpoints are intentionally excluded.

## Documentation

- [`docs/method.md`](docs/method.md): method and objective semantics
- [`docs/architecture.md`](docs/architecture.md): package ownership
- [`docs/reproduction.md`](docs/reproduction.md): full commands and parity gates
- [`docs/artifact_contract.md`](docs/artifact_contract.md): cross-stage schemas
- [`docs/prior_import.md`](docs/prior_import.md): external prior adapters
- [`docs/limitations.md`](docs/limitations.md): scope and known limitations

The full research history remains available at Git tag
`archive-full-research-20260803`; it is intentionally absent from this release
branch.

## License

LaFGS code is MIT licensed. SuperPoint weights are not redistributed and retain
their upstream noncommercial terms; see
[`docs/third_party.md`](docs/third_party.md).
