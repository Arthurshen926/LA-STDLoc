# Feed-forward prior matrix and matched-view canopy comparison

## Verified completion

The fixed Cambridge matrix is complete: 5 scenes x 6 conditions x 2 seeds =
60 full-test evaluations. Its 20 feed-forward cells (Long-LRM/i3DGS, raw/+1k,
5 scenes) account for 40 evaluations. The merged ledger validates exact test
query-name sets and map/result hashes:
`/mnt/pool/sqy/anygsloc_invalid_primitive_recovery_20260910/cambridge_matrix.json`.

Median translation cm / rotation degrees, seed 2026 (not seed averaging):

| Scene | Long-LRM raw | Long-LRM +1k | i3DGS raw | i3DGS +1k |
|---|---:|---:|---:|---:|
| GreatCourt | 379.49 / 2.442 | 50.75 / 0.315 | 29.42 / 0.162 | 14.00 / 0.072 |
| KingsCollege | 24.65 / 0.299 | 16.99 / 0.156 | 22.63 / 0.232 | 16.85 / 0.162 |
| OldHospital | 24.46 / 0.472 | 10.46 / 0.205 | 37.50 / 0.908 | 12.31 / 0.268 |
| ShopFacade | 4.71 / 0.200 | 2.13 / 0.110 | 4.77 / 0.176 | 2.02 / 0.111 |
| StMarysChurch | 718.14 / 17.392 | 185.77 / 5.547 | 19.12 / 0.527 | 8.08 / 0.284 |

## Supplemental budget control is separate

StMarysChurch +2974 iterations completed both seeds: median TE 13.9516 cm;
RE 0.4138 / 0.4041 degrees. Both seeds have 134/530 queries above 1 m.
Seed 2026 mean TE is 14.70 m and p90 TE 60.59 m: the improved median is not
proof of reliable localization. Both optimizer budget and LR horizon changed,
so this does not isolate unseen-mapping-view coverage as the cause.

Update: GreatCourt +3062 final tests have now completed for both seeds:
median TE 25.9649 / 26.2878 cm, RE 0.15222 / 0.15207 degrees, and 128 / 127
queries at or above 1 m. Thus all fixed-budget and both supplemental two-epoch
experiments are complete. The verified paper-table bundle is
`/mnt/pool/sqy/anygsloc_paper_tables_20260910`. Completion state:
`/mnt/pool/sqy/anygsloc_long_lrm_two_epoch_gpu2_control_20260910/gpu2/state.json`.

## Added render comparison

Output root: `/mnt/pool/sqy/anygsloc_stmary_feedforward_canopy_20260910`.
The same G4Splat registry views 632/633/634/408 are rendered for Long-LRM raw,
Long-LRM +1k, i3DGS raw, i3DGS +1k, and the separate Long-LRM +2974 control.
Each has actual PLY hashes and camera checks in its per-method report.

`00633_comparison.png` uses a 2x3 layout: reference, Long-LRM raw, Long-LRM +1k;
i3DGS raw, i3DGS +1k, Long-LRM +2974. `00633_all_methods.png` includes the
previous MAtCHA/Vanilla 2DGS/Vanilla 3DGS and historical G4Splat panels as well.
Other three camera indices have the same two layouts and matched crops.

Camera, 640x360 resolution, black background, and 3DGS antialiased settings
are fixed across feed-forward variants. This is a harmonized AnyGSLoc-render
comparison, not the native Long-LRM white/classic rendering protocol. Different
input-view budgets, possible training-view overlap, and historical G4 settings
preclude a controlled reconstruction-method ranking from these images alone.

Visual inspection of frames 633 and 408: Long-LRM raw/+1k has broad smoothing
and smeared architectural contours; the longer optimization recovers clearer
window boundaries in frame 633 but does not recover fine canopy structure.
i3DGS retains substantially more high-frequency tree texture in frame 408,
but frame 633 has severe colored streaks and foreground-like occluding artifacts
in both raw/+1k. These observations should not be summarized as "i3DGS is
uniformly visually superior", nor used to infer the exact geometric or renderer
cause without a separate audit. Global median localization can improve even
when these selected render views remain visually poor.
