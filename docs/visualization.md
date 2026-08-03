# Visualization

`scripts/visualize.py` produces paper-facing qualitative views from existing
artifacts. It does not train a model or alter evaluation state.

```bash
python scripts/visualize.py --help
```

Visualizations distinguish RGB Gaussian primitives, triangulated Track anchors,
Gaussian-supported reserve anchors, and final sparse correspondences. They
should display provenance and failures without implying that primitive identity
equals localization identity.
