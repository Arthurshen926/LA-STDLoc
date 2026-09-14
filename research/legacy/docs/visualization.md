# Visualization

`scripts/visualize.py` produces a compact map overview. The publication figure
pipeline is `scripts/build_paper_figures.py`. It is read-only with respect to
training artifacts and emits PDF, SVG, PNG, and a provenance manifest.

```bash
python scripts/visualize.py --help
python scripts/build_paper_figures.py --help
```

Visualizations distinguish RGB Gaussian primitives, triangulated Track anchors,
Gaussian-supported reserve anchors, and final sparse correspondences. They
should display provenance and failures without implying that primitive identity
equals localization identity.

The default Cambridge build writes five figures to
`/mnt/pool/sqy/lafgs_paper_figures_20260806`:

1. the complete offline-to-online method overview;
2. one rendering primitive supporting multiple track-anchor identities;
3. topology distillation from splats and candidate evidence to the compact map;
4. frozen A0/A1 correspondences on the same test query;
5. prior flexibility across vanilla 3DGS, vanilla 2DGS, and AnySplat.

Figure 4 replays one frozen query to recover correspondence coordinates. It
asserts the same map, native SuperPoint frontend, full-resolution coordinate
convention, global top-1 matcher, and PoseLib configuration used by evaluation.
No figure command updates a descriptor, geometry, topology, or result file.
