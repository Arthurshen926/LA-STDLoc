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

The default V2 configuration derives angular pixel thresholds, metric thresholds,
query exposure, pose/view bins, matching-rank targets, and final topology size
from mapping-only statistics. Every resolved value is stored in
`scene_calibration.json`; stale mixed-threshold artifacts are rejected. Use
`configs/paper_mainline_frozen_v1.yaml` only to
reproduce the historical fixed 48K/8K/96/256/1000/175 protocol.

In adaptive V2, `scaffold_safety_cap` is an upper bound. A scaffold with fewer
KCS-eligible primitives remains smaller; fixed-budget non-consensus fallback is
reserved for explicit V1 compatibility. Raster provenance persists sampled
depth and alpha from the same frozen-prior render used for primitive lineage.
Legacy query caches without `native_alpha` therefore require current raster
provenance and cannot silently treat every pixel as valid surface support.

The YAML keeps globally shared policy constants and resource caps, while
`scene_calibration.json` records scene-derived values. Do not copy a metric
reference constant from an artifact produced by a different baseline
statistic: the current scale definition is the median first-to-last camera span
of real co-visible mapping tracks.

Use `--function-graph-shards 4 --provenance-shards 4 --observation-shards 4
--pose-scoring-shards 4` when host memory permits parallel evidence
construction. Shards preserve global query indices and PoseLib seeds; pose
scores are merged before one global greedy selection. The results are
semantically identical to the default single-shard run.

## Stage commands

The stable CLI sequence is `import_prior.py`, `build_evidence.py`,
`distill_map.py`, `train_map.py`, `evaluate.py`, and `visualize.py`.
`run_pipeline.py` invokes the same package APIs end to end.

Evaluate the A0 bootstrap through the same sparse runtime by passing its
Stage-A state. The CLI materializes the wide map with an exact identity metric:

```bash
python scripts/evaluate.py \
  --dataset /data/scene \
  --stage-state /data/run/bootstrap/stage_a/STEPS_lafgs_map_state.pt \
  --output /data/run/evaluation_a0_seed2026 \
  --seed 2026
```

Adaptive maps store `scene_calibration.json` beside the trained map. The
evaluator discovers it automatically so the final PoseLib solve uses the same
mapping-derived RANSAC gate as self-localization training. Use
`--scene-calibration` only when evaluating a relocated map artifact; a
calibration that does not explicitly declare `uses_test_queries: false` is
rejected.

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
