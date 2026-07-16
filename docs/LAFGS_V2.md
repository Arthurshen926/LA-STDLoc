# LaFGS V2: Progressive Localization Coreset

## Architecture contract

LaFGS V2 keeps four objects separate:

1. `G_R`: the complete external RGB 2DGS. Geometry, appearance, primitive IDs,
   rendering, depth, and visibility are frozen.
2. `S`: the exact strong localization bank used by the reproducible baseline.
   Its primitive IDs, 3D anchors, and descriptors are preserved exactly.
3. `P`: the full observed surface-patch shadow pool, represented by weighted
   geometric medoids of stable 2DGS IDs. It has no fixed atom-count cap.
4. `A_t`: an exact-budget deployment set initialized from `S`. A shadow atom can
   enter only through a fixed held-out query probe; rejected swaps leave `S`
   unchanged.
5. `C_q`: the query-specific top-k query-to-active-set correspondence graph.
6. `I_q`: the correspondences retained by deployment-time RANSAC/PnP.

The core path intentionally excludes raw-XYZ updates, topology operations, Pair,
FIM weighting, and Diff-PnP. Those modules are not required to produce or deploy
the V2 coreset.

## Training flow

1. Load the external 30k MAtCha 2DGS and freeze it.
2. Build stable surface groups from primitive position and normal.
3. Aggregate frozen image-encoder features into every primitive (full-map MVInit),
   requiring repeated observations and optionally blending a strong prior field.
4. Discover every surface patch exercised by real detector queries. Keep
   provenance mass on active, shadow, and missing atoms separate; never
   renormalize a partially observed target back to probability one.
5. Insert the strong bank as exact atoms rather than merging it into 2 cm patches.
   Keep its descriptors frozen while shadow descriptors learn from query episodes.
7. Recover top-K primitive composition weights from the 2DGS raster metadata and
   aggregate them into soft surface-group provenance targets.
8. Add GT-reprojected strong landmarks as a fixed extra positive channel, guarded
   by both renderer visibility and full-map rendered depth. This channel remains
   present even when a view has no visible strong landmark.
9. Train with deployment cosine scores only. Coverage misses reject false active
   candidates and update inactive shadow positives so they can become proposals.
10. Propose small exact-budget shadow swaps after warm-up. Accept a proposal only
    when a fixed held-out real-query probe improves top-1 precision plus weighted
    top-16 recall; otherwise restore the prior active set exactly.
11. Optionally add rendered RGB episodes to real episodes. Synthetic RGB first
    passes through `valid_support_mask`; invalid pixels are neutralized before the
    frozen encoder and invalid feature cells are excluded from detector sampling,
    matching, provenance labels, and self-localization. Synthetic episodes update
    only localization descriptors/statistics.
12. Materialize the final coreset, freeze it, retrain the scene detector, then run
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
scripts/run_lafgs_v2_shopfacade.sh c0       # 30k real-only V2.2 shadow-swap run
scripts/run_lafgs_v2_shopfacade.sh eval_c0  # 10k detector + test localization
scripts/run_lafgs_v2_shopfacade.sh c1       # real + 15% masked rendered episodes
scripts/run_lafgs_v2_shopfacade.sh eval_c1  # 10k detector + test localization
```

The final evaluation has no Pair, FIM selection, Diff-PnP, topology, or geometry
updates. This keeps changes in pose quality attributable to the learned coreset.

## Acceptance criteria

A run is an improvement only if the fixed probe accepts at least one swap and the
held-out localization evaluation improves pose error without reducing raw GT
precision, RANSAC-inlier GT precision, or pose information. If all proposals are
rejected, the output is deliberately the exact strong bank; this is a valid safe
fallback, not evidence that the shadow-pool objective improved the map.

## ShopFacade V2.2 result (2026-07-17)

The corrected C1 run used the external MAtCha 30k 2DGS, 16,384 exact seed
landmarks, the full 516,147-patch shadow pool, 10k optimization steps, and 15%
masked online-render episodes.

- GT seed reprojection now requires renderer visibility and rendered-depth
  consistency. Before this fix, occluded landmarks were incorrectly assigned
  0.75 positive mass. Clean real reprojection positives are roughly 10-18% of
  detected queries; rendered novel views contain substantially fewer.
- The online artifact mask retained 99.65% of feature cells on average and
  96.33% in the most strongly masked logging window. Invalid cells cannot enter
  detector sampling or matching.
- The fixed clean seed probe objective was 0.07385. Seventeen independent shadow
  proposals ranged from 0.06750 to 0.07324, so all were rejected.
- The exported C1 set is the exact strong bank: all IDs match; descriptor cosine
  similarity has mean 1.0 and minimum 0.99999976.
- A fresh held-out test with the co-adapted seed detector produced 3.7571 cm
  median translation error, 0.18365 degree median rotation error, 13.97% raw GT
  precision at 4 px, 77.63% RANSAC-inlier GT precision at 4 px, and 549.36 mean
  inliers.

Therefore V2.2 is a safe, normally localizing fallback, but the current shadow
utility and masked render branch do not yet improve the strong bank. More steps
or scalar reweighting are not supported by this run; the next method change must
improve query-specific visible-surface positives and final-set solvability before
allowing deployment-set churn.
