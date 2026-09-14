# Architecture

The repository is organized by artifact ownership rather than experiment date.

| Package | Responsibility |
| --- | --- |
| `common` | strict config, hashes, schemas, registries, contracts |
| `data` | COLMAP cameras, splits, resize/intrinsics, optional masks |
| `priors` | normalized 2DGS/3DGS import, frozen model, raster provenance |
| `features` | native SuperPoint, sparse/dense sampling, KCS/GWFF |
| `evidence` | function graph, tracks, visibility, triangulation |
| `topology` | candidates, Track core, coverage reserve, map materialization |
| `map_learning` | Stage-A objective, final A1 metric, canonical pipeline |
| `localization` | native frontend, global top-1, standard PoseLib |
| `evaluation` | metrics, reports, frozen parity fixture |
| `visualization` | deterministic map and evidence figures |

Method logic belongs in these packages. Files under `scripts/` are thin CLIs and
must not define alternative algorithms or override config through environment
variables.

The only public orchestration entry is `scripts/run_pipeline.py`. Individual
stage CLIs exist for debugging and distributed execution, but consume and emit
the same artifact schemas.
