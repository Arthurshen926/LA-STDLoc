# Paper Draft

This directory is the minimal manuscript implementation for the frozen LaFGS
paper mainline. It intentionally contains no qualitative figures yet and does
not claim unregistered 7Scenes or 12Scenes results.

Build with:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

The method text must stay aligned with `configs/paper_mainline.yaml` and
`map_learning/pipeline.py`. Experiment values should be generated from the
registered JSON summaries rather than entered manually.
