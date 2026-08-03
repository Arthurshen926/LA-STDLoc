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

## Stage commands

The stable CLI sequence is `import_prior.py`, `build_evidence.py`,
`distill_map.py`, `train_map.py`, `evaluate.py`, and `visualize.py`.
`run_pipeline.py` invokes the same package APIs end to end.

## Parity gates

Run unit and golden tests:

```bash
pytest -q
python -m evaluation.parity \
  --fixture paper_baseline/golden_fixture \
  --dataset-root /data/Cambridge/ShopFacade \
  --map /data/maps/ShopFacade/anchor_map_step_0175.pt \
  --metric-state /data/maps/ShopFacade/metric_state_step_0175.pt
```

The frozen full-evaluation summaries are in `paper_baseline/expected_metrics.json`.
Refactoring is accepted only when keypoints, top-1 IDs, inliers, poses, and
summary metrics meet the registered exact/tolerance gates.
