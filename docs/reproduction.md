# Reproduction

## Environment

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PYTHONPATH=$PWD
pip install -e '.[test]'
```

## Full pipeline

```bash
python scripts/run_pipeline.py \
  --dataset /data/Cambridge/ShopFacade \
  --prior /data/priors/ShopFacade/matcha_2dgs \
  --gaussian-type 2dgs \
  --config configs/paper_mainline.yaml \
  --output /data/runs/ShopFacade
```

The mapping split is used in full; the test split is read only by the final
evaluation stage. Re-running the command reuses only artifacts whose contracts
match.

Use `--function-graph-shards 4` when host memory permits parallel function
graph construction. Shards preserve the global query indices and PoseLib seeds;
the merged artifact is semantically identical to the default single-shard run.

## Stage commands

The stable CLI sequence is `import_prior.py`, `build_evidence.py`,
`distill_map.py`, `train_map.py`, `evaluate.py`, and `visualize.py`.
`run_pipeline.py` invokes the same package APIs end to end.

Evaluate the A0 bootstrap through the same sparse runtime by passing the frozen
Stage-A state. The CLI materializes the 48K map with an exact identity metric:

```bash
python scripts/evaluate.py \
  --dataset /data/scene \
  --stage-state /data/run/bootstrap/stage_a/1000_lafgs_map_state.pt \
  --output /data/run/evaluation_a0_seed2026 \
  --seed 2026
```

For A1, pass `--map` and `--metric-state` as before. These modes are mutually
exclusive and share the same frontend, matching, PoseLib, and reporting code.

## Parity gates

Run unit and golden tests:

```bash
pytest -q
LAFGS_RUN_CUDA_SMOKE=1 pytest -q tests/test_renderer_smoke.py
python -m evaluation.parity \
  --fixture paper_baseline/golden_fixture \
  --dataset-root /data/Cambridge/ShopFacade \
  --map /data/maps/ShopFacade/anchor_map_step_0175.pt \
  --metric-state /data/maps/ShopFacade/metric_state_step_0175.pt
```

The CUDA smoke must run from an activated environment so the `ninja`
executable installed with the Python package is present on `PATH`. Importing
`gsplat` alone does not compile or execute its rasterizer and is not a
sufficient release check.

The frozen full-evaluation summaries are in `paper_baseline/expected_metrics.json`.
Refactoring is accepted only when keypoints, top-1 IDs, inliers, poses, and
summary metrics meet the registered exact/tolerance gates.
