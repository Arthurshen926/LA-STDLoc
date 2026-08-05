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

Generate registered pooled results and LaTeX rows with:

```bash
python scripts/aggregate_results.py \
  --run-root /path/to/registered/runs \
  --dataset-label 7Scenes \
  --scenes heads chess fire office pumpkin redkitchen stairs \
  --json-output paper/results/7scenes.json \
  --tex-output paper/tables/7scenes_rows.tex
```

The command rejects incomplete scenes, mismatched protocol hashes, nonstandard
seed sets, priors that are not mapping-only, and runs that do not certify zero
test-image leakage.
