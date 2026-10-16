# Reproduce the paper experiments

All commands below run from the repository root after [installation](installation.md) and [asset preparation](data.md). The table identifiers refer to the main manuscript T1–T9 and the thirteen named supplementary tables. `reproduction/catalog.json` binds the accepted per-scene records to experiment groups. `expected_metrics.json` stores their full-precision statistics; accepted TeX is in `reproduction/paper_tables/`.

## Three entry levels

1. **Recompute statistics** from the included raw records: CPU only, no datasets or checkpoints.
2. **Relocalize** using the exact supplied deployment maps: query images, calibration, masks, and SuperPoint are required.
3. **Rebuild maps** from the supplied Gaussian priors: rendering and construction are rerun before localization.

Upstream Gaussian reconstruction and native baseline training are separate producers; see [priors.md](priors.md) and [baselines.md](baselines.md). Literature rows are explicitly cited published results, not local reproductions of every listed method.

```bash
python -m reproduction.aggregate --check --output runs/reference_report
python -m reproduction.verify_tables --metrics runs/reference_report/metrics.json
```

This checks the compressed and uncompressed record hashes, complete query sets, full scene coverage, and all stored metric values. It writes JSON and CSV. It does not run localization or measure new latency.

## Coverage index

| Paper item | Experiment family / records | Command or section |
|---|---|---|
| T1, our Cambridge row | vanilla 3DGS, Full | `main`, Cambridge |
| T2, T3, our indoor rows | 2DGS, Compact v2 | `main`, indoor |
| T4 | Native STDLoc/ULF-Loc mapping-mask-on initial/final; Any Full/Compact | native accuracy + timing |
| T5 | center/depth/ray with identical common observations, vanilla 3DGS and MAtCha | `geometry`, two configs |
| T6 | observation filtering × descriptor fusion, Compact | `filter-fusion` |
| T7 | Full, natural selector stages, and equal-budget controls | `selection` |
| T8 | seven Gaussian priors, Compact + reference-view PSNR | `main` and `psnr`, seven configs |
| T9 | rendered / real / hybrid appearance, Full and Compact | `inputs` |
| S_baselines, S_baseline_tails | native mapping semantics on/off, both stages; Any Full/Compact | native accuracy; per-scene tails |
| S_prior_full, S_prior_tails | same seven priors, Full and error tails | prior `main` |
| S_t6_full, S_filter_fusion | four construction conditions, Full and tails | `filter-fusion` |
| S_timing, S_resources | 90 fresh processes, full-frame/core timing and memory | timing section |
| S_native_stages | native initial segment, cumulative coarse prefix and dense suffix | formal timing records |
| S_geometry | two-prior coordinate control error tails | `geometry` |
| S_selection | selector-stage and equal-budget error tails | `selection` |
| S_input_tails | real/rendered/hybrid error tails | `inputs` |
| S_gs_assisted | real descriptors with auxiliary GS depth completion | `assisted` |
| Figures 1–9 | method diagrams, qualitative maps, geometry and deployment plots | [figures.md](figures.md) |

## Main benchmarks: T1–T3

Relocalization uses configured released maps when a generated map is absent:

```bash
python -m anygsloc reproduce evaluate --config configs/cambridge.json --map full
python -m anygsloc reproduce evaluate --config configs/cambridge.json --map compact
python -m anygsloc reproduce evaluate --config configs/7scenes.json --map compact
python -m anygsloc reproduce evaluate --config configs/12scenes.json --map compact
```

Rebuild and evaluate all scenes (remove `--execute` to print the plan):

```bash
python -m reproduction.run main --config configs/cambridge.json --execute
python -m reproduction.run main --config configs/7scenes.json --execute
python -m reproduction.run main --config configs/12scenes.json --execute
```

Start with `--scene ShopFacade --output runs/shop_smoke` for Cambridge. Indoor conversion uses the combined `build` stage because its alpha-gated observation/audit workflow differs from Cambridge. Indoor old observation caches have been removed from the experiment server; conversion rerenders from the retained 2DGS priors. Supplied deployment maps and complete query results remain available.

The current paper's indoor benchmark rows are **Compact**, despite the existence of separate Full exports and a partially executed historical Full evaluation protocol. That incomplete protocol is not treated as the source of T2/T3.

## Coordinate controls: T5 and S_geometry

```bash
python -m reproduction.run geometry --config configs/cambridge.json --execute
python -m reproduction.run geometry --config configs/prior_matcha.json --execute
```

Within each prior, the three estimators share landmark identities, fused descriptors, and the same valid reference observations. Center means the coordinatewise lower median of the associated projected Gaussian centers; depth means the same aggregation of backprojected rendered-depth proposals. Ray uses all common observations and three Huber refinement iterations. At least three valid observations are required. Invalid ray solutions remove that landmark from all three arms. The common pools are constructed separately for the two priors, not forced to the same point count. These common-support ray maps are distinct from the unmodified production Full map.

The stage also writes `common_observations.npz` and per-observation `mapping_residuals.npy`, which generate the geometry CDF in Fig. 6. Failed projections retain their place in its denominator.

## Construction ablation: T6, S_t6_full, S_filter_fusion

```bash
python -m reproduction.run filter-fusion --config configs/cambridge.json --execute
```

The four combinations are filtering on/off × track fusion/single descriptor. The single descriptor is the first observation in lexicographic camera-name order; it is not a mean descriptor or a selected best query. Filtering off removes the V2 admission rule but preserves physical alpha/depth validity for depth proposals. Completion stays enabled. Within each filtering condition, fused and single-descriptor maps share coordinates and selected rows. T6 reports Compact; S_t6_full reports Full. Do not substitute the old Full-only ablation for main T6.

## Selection: T7 and S_selection

```bash
python -m reproduction.run selection --config configs/cambridge.json --execute
```

The seven outputs are Full, quality stage, coverage stage, final Compact, spatial uniform, quality Top-K, and track-length Top-K. Stage maps use their natural sizes before terminal cleanup. The three alternative selectors use exactly the final Compact count of that scene. Top-K ties break by source-row index. Selection never reads query poses or query accuracy. Full and all subsets retain the original landmark coordinates and descriptors.

## Prior sensitivity: T8 and supplementary Full/tails

```bash
for config in cambridge prior_matcha prior_vanilla_2dgs prior_i3dgs_raw prior_i3dgs_1k prior_long_lrm_raw prior_long_lrm_1k; do
  python -m reproduction.run main --config "configs/$config.json" --execute
  python -m reproduction.run psnr --config "configs/$config.json" --execute
done
```

Use the same map mode across all priors. Natural map sizes differ. Main T8 is Compact; the supplement is Full. PSNR covers all **5,364 mapping/reference images**, averaged first per image, then per scene, then equally over the five scenes. All quality renders use black background; the localization-render backgrounds remain those configured for each prior. A held-out novel-view PSNR or a pooled-MSE PSNR is a different experiment.

## Appearance inputs: T9, S_input_tails, S_gs_assisted

```bash
python -m reproduction.run inputs --config configs/cambridge_inputs.json --execute
python -m reproduction.run assisted --config configs/cambridge_inputs.json --execute
```

`pure_gs` uses rendered RGB descriptors only. `raw_only` detects on real mapping RGB and disables V2 rendered-observation filtering and depth completion. `hybrid` uses the strongest 1,024 real and 1,024 rendered observations before filtering; rejected observations are not refilled. All use the same cameras, query evaluation, and current selector. The auxiliary experiment adds Gaussian-depth proposals to the real-only tracks while keeping **all descriptors real**. It must not be relabeled as hybrid appearance.

## Accuracy reports and metric definitions

```bash
python -m anygsloc reproduce report --config configs/cambridge.json
```

For a custom run use its same `--output` and optional `--scene`. A partial-scene report is marked as such. The reporter rejects missing scenes, duplicate/reordered query identities, and wrong query counts. TE is measured in cm between camera centers; RE is in degrees. Compute each median, mean, percentile, and recall **within each scene**, then take the unweighted scene mean. Counts and map bytes are summed. R5 and R15 use TE < 5/15 cm and RE < 5 degrees in these accepted records. Failed queries remain in the denominator and contribute infinite error; JSON represents nonfinite aggregate errors as null. Zero-inlier native outputs and extreme errors are retained.

A release-map file's metadata may differ after rebuilding; compare the tensors for numerical parity and report its actual serialized bytes. Published decimal MB are based on accepted inventory bytes divided by 1,000,000.

## Summarize newly executed experiments

`reproduction.aggregate` always reaggregates the accepted bundled records. For **new runs**, use the reporters below. They do not substitute reference values for missing results:

```bash
# All completed AnyGSLoc ablations under the configured run root:
python -m anygsloc reproduce report --config configs/cambridge.json
# Repeat with each input/prior/indoor config after its experiment plan.

# New reference-view renders, one prior at a time:
python -m reproduction.report psnr --config configs/cambridge.json \
  --root runs/cambridge --output runs/cambridge/psnr_report.json

# Both native stages, one method and mapping condition at a time:
python -m reproduction.report native --root runs/native/STDLoc/on \
  --output runs/native/STDLoc/on/accuracy_report.json

# Recursively collect all round.json files under this campaign directory:
python -m reproduction.report timing --root runs/native_timing \
  --output runs/native_timing/report.json
python -m reproduction.report timing --root runs/timing \
  --output runs/timing/report.json
```

For AnyGSLoc timing, use `--output runs/timing` consistently across all scene/map/round invocations. Native accuracy expects `<root>/<scene>/` directories from the example above. PSNR and native reports require every configured scene unless `--scene` explicitly requests a partial check. Timing requires all five scenes and three distinct rounds per configuration; `--partial` labels a smaller smoke check. Keep incomplete/failed runs outside the report root. Write each report to a new filename.

These reports retain per-scene accuracy/PSNR, scene-equal means, actual storage, and both cumulative native stages. Timing percentiles pool the measured queries and exclude warm-ups. Match their configurations to the coverage index above when populating new table rows; the accepted TeX is retained separately and is not overwritten by a new experiment.

## Formal runtime: T4, S_timing, S_resources

Use an otherwise idle RTX 3090, four CPU threads, and one benchmark process at a time. Do not derive paper latency from the accuracy run's telemetry. For each scene, the fixed subset contains 100 queries ordered by a seeded name hash and restored to test-list order. The first 50 also run as separate warm-ups. Each of three rounds starts a fresh process and recomputes features; no query feature cache is used.

```bash
for map in full compact; do
  for round in 0 1 2; do
    python -m anygsloc reproduce benchmark --config configs/cambridge.json \
      --scene ShopFacade --map "$map" --round "$round" --output runs/timing_shop
  done
done
```

**Always specify one scene per timing invocation** so each scene/round gets a fresh process; repeat for all five scenes. Run baseline timing using [baselines.md](baselines.md) for both mapping-semantic conditions. This produces 6 configurations × 5 scenes × 3 rounds = 90 processes and 1,500 measured queries per configuration. Core/frame means are scene-equal; timing percentiles pool the 1,500 measured observations. Memory is the maximum measurement-phase PyTorch allocated/reserved high-water counter, in decimal GB. Initialization is recorded separately.

Native initial and final core times are cumulative checkpoints in the **same complete localization call**. The refinement increment is final minus initial. STDLoc shared feature preparation and ULF-Loc earlier dense features remain included. ULF-Loc's native GT-projection/inlier diagnostic is also retained in core time; it does not determine the estimated pose. Frame includes decoding, materialization and preprocessing. Precomputed-mask generation by a segmentation network is excluded.
