# LaFGS V2: Progressive Localization Coreset

## Architecture contract

LaFGS V2 keeps four objects separate:

1. `G_R`: the complete external RGB 2DGS. Geometry, appearance, primitive IDs,
   rendering, depth, and visibility are frozen.
2. `A_t`: a trainable localization active set over the stable primitive IDs.
   Only descriptors and activation gates are optimized.
3. `C_q`: the query-specific top-k query-to-active-set correspondence graph.
4. `I_q`: the correspondences retained by deployment-time RANSAC/PnP.

The core path intentionally excludes raw-XYZ updates, topology operations, Pair,
FIM weighting, and Diff-PnP. Those modules are not required to produce or deploy
the V2 coreset.

## Training flow

1. Load the external 30k MAtCha 2DGS and freeze it.
2. Build stable surface groups from primitive position and normal.
3. Aggregate frozen image-encoder features into every primitive (full-map MVInit).
4. Warm up all primitive descriptors with query-induced matching supervision.
5. Progressively project the active budget through
   `N -> N/4 -> N/16 -> 65536 -> 32768 -> 16384`.
6. Recover top-K primitive composition weights from the 2DGS raster metadata and
   aggregate them into soft surface-group provenance targets.
7. Use exact group coverage probability, budget, redundancy, and MVInit trust
   losses. Covered queries train the current active representative of their
   surface group; coverage misses train gates but are never relabeled dustbin.
8. Optionally add rendered RGB episodes to real episodes. Synthetic episodes do
   not replace real updates and update only localization descriptors and gates.
9. Materialize the final coreset, freeze it, retrain the scene detector, then run
   standard sparse matching and RANSAC/PnP.

## Outputs

Each coreset run writes:

- `coreset_state.pt`: complete V2 state and provenance.
- `final_candidate_teacher_state.pt`: evaluation-compatible feature override.
- `sampled_idx.pkl`: stable source primitive IDs.
- `localization_features.pt`: compact normalized descriptors.
- `surface_group_ids.pt`: compact surface group IDs.
- `landmark_meta.pt`: gates and selection metadata.
- `training_log.jsonl` and `training_summary.json`: stage diagnostics and config.

## ShopFacade experiment protocol

The entry point is `scripts/run_lafgs_v2_shopfacade.sh`. It pins all work to
physical GPU 2 and uses the external MAtCha 2DGS at iteration 30000.

```bash
scripts/run_lafgs_v2_shopfacade.sh c0       # 30k real-only coreset
scripts/run_lafgs_v2_shopfacade.sh eval_c0  # 10k detector + test localization
scripts/run_lafgs_v2_shopfacade.sh c1       # 30k real + 15% rendered episodes
scripts/run_lafgs_v2_shopfacade.sh eval_c1  # 10k detector + test localization
```

The final evaluation has no Pair, FIM selection, Diff-PnP, topology, or geometry
updates. This keeps changes in pose quality attributable to the learned coreset.
