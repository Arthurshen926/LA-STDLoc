# LaFGS

LaFGS reconstructs a compact **Evidence-Grounded Anchor Registry** from an
off-the-shelf RGB Gaussian scene prior and posed mapping images. A deployed
Anchor has one observation identity, one materialized 3D position and
uncertainty, one descriptor, orthogonal surface/visibility/lineage evidence,
and one selection reason. Feature tracks propose observation-grounded
identities and rendering primitives contribute surface evidence; neither is a
separate deployment map.

The adaptive paper pipeline is:

```text
RGB Gaussian prior (frozen)
  -> native SuperPoint mapping observations
  -> KCS/GWFF initialization
  -> wide-scaffold self-localization (A0)
  -> cross-view associations + robust triangulation
  -> Gaussian surface, visibility, and raster-lineage evidence
  -> unified geometry materialization + Anchor candidate registry
  -> one sufficiency selector
       [Precision Core -> matching/observability completion]
  -> bounded shared-metric refresh (A1)
  -> compact single-descriptor Anchor Registry
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

The canonical command runs the complete method and, by default,
evaluates the official test split:

```bash
python scripts/run_pipeline.py \
  --dataset /data/Cambridge/OldHospital \
  --prior /data/priors/OldHospital/vanilla_2dgs \
  --gaussian-type 2dgs \
  --output /data/runs/OldHospital
```

For large mapping sets, use `--function-graph-shards N --provenance-shards N
--observation-shards N --pose-scoring-shards N`. These options evaluate
independent query shards concurrently and deterministically merge them before
the single global topology selection. They are exact execution backends and do
not change the method or per-query seeds.

The seven stable CLIs under `scripts/` expose the same stages individually.
All adaptive defaults are resolved through
[`configs/paper_mainline.yaml`](configs/paper_mainline.yaml). The historical
fixed protocol remains available as `configs/paper_mainline_frozen_v1.yaml`.

## Evaluate

```bash
METRIC_STEPS=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["parameters"]["metric_steps"])' \
  /data/runs/OldHospital/map_learning/scene_calibration.json)
METRIC_TAG=$(printf '%04d' "$METRIC_STEPS")
python scripts/evaluate.py \
  --dataset /data/Cambridge/OldHospital \
  --map /data/runs/OldHospital/map_learning/anchor_map_step_${METRIC_TAG}.pt \
  --metric-state /data/runs/OldHospital/map_learning/metric_state_step_${METRIC_TAG}.pt \
  --output /data/runs/OldHospital/evaluation
```

The release parity fixture and frozen ShopFacade/OldHospital metrics live in
`paper_baseline/`; large maps and checkpoints are intentionally excluded.

Adaptive runs name the final map and metric with the mapping-only calibrated
`metric_steps` recorded in `scene_calibration.json`; the suffix is not a fixed
cross-scene constant.

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
