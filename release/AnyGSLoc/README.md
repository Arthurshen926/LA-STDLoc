# AnyGSLoc

[中文入口](README.zh-CN.md)

Convert a Gaussian scene and calibrated reference cameras into a sparse landmark map for visual localization. The online pipeline extracts query features, matches them to the map, and solves one camera pose; it does not load or render the Gaussian scene.

This source candidate follows the **October 13, 2026 paper experiments and October 14 figures**. Cambridge uses vanilla 3DGS by default. Indoor results use 2DGS. `Full` keeps all valid landmarks; `Compact` applies quality pruning, coverage completion, pose-aware completion, and support-preserving cleanup.

## Start here

- [Installation and tested environment](docs/installation.md)
- [Environment profiles, checks, and launches](docs/environments.md)
- [Data, checkpoints, and portable paths](docs/data.md)
- [Reproduce every paper experiment](docs/reproduction.md)
- [Method and implementation map](docs/method.md)
- [Native STDLoc and ULF-Loc comparisons](docs/baselines.md)
- [Prior construction and PSNR](docs/priors.md)
- [Figures and table provenance](docs/figures.md)
- [Validation and remaining release work](docs/validation.md)

The included compressed records reproduce the reported statistics **without a GPU or datasets**:

```bash
python -m pip install numpy==1.26.4
python -m reproduction.aggregate --check --output runs/reference_report
```

To localize with a supplied paper map, install the runtime, prepare the assets, and run:

```bash
export ANYGSLOC_ASSETS="$PWD/assets"
export ANYGSLOC_SUPERPOINT_WEIGHTS="$ANYGSLOC_ASSETS/weights/superpoint_v1.pth"
python -m anygsloc reproduce plan --config configs/cambridge.json --scene ShopFacade
python -m anygsloc reproduce evaluate --config configs/cambridge.json \
  --scene ShopFacade --map compact --output runs/shop_eval --device cuda:0
```

To reconstruct and evaluate a scene from its Gaussian prior:

```bash
python -m reproduction.run main --config configs/cambridge.json \
  --scene ShopFacade --output runs/shop_rebuild --device cuda:0 --execute
```

Omit `--execute` to inspect commands. All paths are relative to `ANYGSLOC_ASSETS`; outputs are separate. Completed stages carry file hashes. An incomplete stage stops the run instead of being silently treated as complete.

## Repository layout

```text
anygsloc/             method, rendering interface, localization, evaluation
  pipeline/           paper stages and experimental controls
configs/              dataset/prior profiles and native baseline settings
reproduction/         experiment plans, asset catalogue, records, aggregation
  paper_tables/       accepted TeX tables for comparison
  figures/            editable paper figure sources and plot entry points
docs/                 reproduction instructions
tests/               numerical and protocol regression checks
```

Large datasets, Gaussian priors, maps, and baseline models are described in `reproduction/assets.json`; they are **not embedded in the source archive or uploaded by this preparation task**. The current server can populate the layout using local asset bindings. Publishing the data/model bundles is still required before an external user can run all GPU experiments. The included reference records can already be checked independently.

## License and attribution

The retained project license is in [LICENSE](LICENSE). Third-party models, datasets, native baseline repositories, and pretrained weights retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). This candidate does not change their licenses or redistribute their weights inside the source tree.
