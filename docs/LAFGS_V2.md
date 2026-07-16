# LaFGS V2: Progressive Localization Coreset

## Architecture contract

LaFGS V2 keeps four objects separate:

1. `G_R`: the complete external RGB 2DGS. Geometry, appearance, primitive IDs,
   rendering, depth, and visibility are frozen.
2. `P`: observable surface-patch localization atoms represented by stable 2DGS
   medoid IDs. Identity, coverage, and redundancy use separate groupings.
3. `A_t`: an exact-budget active subset of `P`. Descriptors are optimized while
   membership is updated by alternating discrete selection.
4. `C_q`: the query-specific top-k query-to-active-set correspondence graph.
5. `I_q`: the correspondences retained by deployment-time RANSAC/PnP.

The core path intentionally excludes raw-XYZ updates, topology operations, Pair,
FIM weighting, and Diff-PnP. Those modules are not required to produce or deploy
the V2 coreset.

## Training flow

1. Load the external 30k MAtCha 2DGS and freeze it.
2. Build stable surface groups from primitive position and normal.
3. Aggregate frozen image-encoder features into every primitive (full-map MVInit),
   requiring repeated observations and optionally blending a strong prior field.
4. Discover identity patches exercised by real detector queries, then materialize
   a coverage-preserving atom pool with real 2DGS medoids.
5. Warm up atom descriptors with query-induced matching supervision.
6. Reduce the exact active budget by at most 25% per stage down to the requested
   final budget, exporting every stage for a budget-curve audit.
7. Recover top-K primitive composition weights from the 2DGS raster metadata and
   aggregate them into soft surface-group provenance targets.
8. Train with deployment cosine scores only. Coverage misses still reject current
   false candidates and update inactive shadow positives so they can re-enter.
9. Alternate descriptor optimization with exact-budget discrete drop/add/swap
   selection based on match utility, negative risk, coverage, and redundancy.
10. Optionally add rendered RGB episodes to real episodes. Synthetic episodes do
   not replace real updates and update only localization descriptors/statistics.
11. Materialize the final coreset, freeze it, retrain the scene detector, then run
   standard sparse matching and RANSAC/PnP.

## Outputs

Each coreset run writes:

- `coreset_state.pt`: complete V2 state and provenance.
- `final_candidate_teacher_state.pt`: evaluation-compatible feature override.
- `sampled_idx.pkl`: stable source primitive IDs.
- `localization_features.pt`: compact normalized descriptors.
- `coverage_cell_ids.pt`: compact coverage IDs; identity and redundancy mappings
  remain in `coreset_state.pt`.
- `landmark_meta.pt`: gates and selection metadata.
- `training_log.jsonl` and `training_summary.json`: stage diagnostics and config.

## ShopFacade experiment protocol

The entry point is `scripts/run_lafgs_v2_shopfacade.sh`. It pins all work to
physical GPU 2 and uses the external MAtCha 2DGS at iteration 30000.

```bash
scripts/run_lafgs_v2_shopfacade.sh c0       # 30k real-only V2.1 coreset
scripts/run_lafgs_v2_shopfacade.sh eval_c0  # 10k detector + test localization
scripts/run_lafgs_v2_shopfacade.sh c1       # 30k real + 15% rendered episodes
scripts/run_lafgs_v2_shopfacade.sh eval_c1  # 10k detector + test localization
```

The final evaluation has no Pair, FIM selection, Diff-PnP, topology, or geometry
updates. This keeps changes in pose quality attributable to the learned coreset.
