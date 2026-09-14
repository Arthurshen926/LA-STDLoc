# AnyGSLoc paper experiment protocol

## Method freeze

`AnyGSLoc-Base` is the formal method. It uses a frozen RGB Gaussian prior, rendered mapping observations, V2 pre-association filtering, projective tracks, robust ray triangulation, robust track descriptor fusion, an uncapped Projective Anchor map, exact global cosine Top-1, and one PoseLib solve.

Exact duplicate mapping-camera geometries are registered once, keeping the lexicographically first image name. This prevents repeated poses in 7Scenes/12Scenes from receiving duplicate association votes while retaining the complete source-name registry for provenance.

`AnyGSLoc-R` is optional. It may use the first pose to select a sparse confidence Core and Reserve, but it must remain query-specific, sparse, map-read-only, and bounded to one additional pose solve. It is never folded silently into Base.

The following are outside this branch's formal method: offline self-localization feedback, map/descriptor/metric updates from queries, query rendering, dense refinement, learned result selection, and source mapping RGB.

## Experiment order

1. **Input audit.** Verify every dataset and prior path. Normalized prior manifests must be RGB-only, localization-state-free, and content-hash valid.
2. **Base construction.** Build all missing 24-scene Base maps with the same configuration and code commit. Reuse the five Cambridge high-capacity F0 maps only as explicitly hash-bound existing artifacts.
3. **Base evaluation.** Run seeds 2026/2027/2028. Report per-scene and pooled metrics; never pool query errors without also reporting each scene.
4. **Prior robustness.** Rebuild KingsCollege and OldHospital with vanilla 2DGS, vanilla 3DGS, and AnySplat. The renderer type and SH degree come from each manifest.
5. **Core ablations.** Remove V2, completion, multi-view robust fusion, and reduce map/camera capacity one factor at a time. Add the real-mapping-RGB result only as a clearly labeled upper bound.
6. **Optional online arm.** Evaluate the frozen AnyGSLoc-R policy separately. If a policy was selected on the evaluation split, label it test-calibrated. Do not use its queries to change the map.
7. **Resources.** Measure end-to-end mean/p50/p90, frontend/matching/PnP/refinement timing, second-solve rate, peak CUDA allocation, peak RSS, map bytes, Anchor count, and build wall time.

## Primary tables

- 24-scene comparison: Base, optional `-R`, and external baselines.
- Gaussian-prior robustness: identical AnyGSLoc construction over each prior producer.
- Construction ablation: V2, track association, completion, descriptor aggregation.
- Efficiency: accuracy against map size, build time, memory, and query latency.

Median TE/RE are primary. p90 TE, mean TE, R5, and catastrophic failures are required robustness diagnostics. Runtime claims use end-to-end measurements after warm-up with batch size one; isolated CUDA-event stage timers are supplementary.

## Fail-closed rules

- A feature-trained or localization-aware Gaussian manifest is rejected.
- A partial map or evaluation directory is not resumed implicitly.
- Mapping artifacts must state `uses_source_mapping_rgb=false`, `uses_test_queries=false`, and `feedback_used=false`.
- AnySplat is rendered according to its manifest (`3dgs`, SH degree 0 for current normalized assets), not a hard-coded 2DGS path.
- Historical feedback artifacts cannot be passed to either formal runner.
- Existing outputs are reusable only with their exact content hashes recorded in the final aggregate.

## Current reusable assets

Five high-capacity Cambridge Base maps already satisfy the mapping-only V2 contract and remain valid existing-artifact rows. Vanilla 2DGS, vanilla 3DGS, and AnySplat normalized RGB-only manifests exist for KingsCollege and OldHospital. The 7Scenes and 12Scenes cells require fresh V2 high-capacity Base construction under this branch.
