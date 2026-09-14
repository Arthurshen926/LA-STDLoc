# Indoor full-reference Gaussian prior revalidation

## Decision

The indoor data contract is now aligned with the public STDLoc/ULF-Loc
protocol: a published full pseudo-ground-truth SfM reconstruction initializes
the Gaussian prior, while only mapping RGB images supervise Gaussian training.
The older pose-only preparation path remains available only through the
explicit `--discard-reference-points` replay flag.

The first end-to-end revalidation on 7Scenes/Stairs is positive. Replacing the
retriangulated 55,660-point initialization with the published 131,766-point
reference model materially improves the source-image-free V1.4 route on real
test queries. This does not make every prior-dependent conclusion invalid, but
it proves that the previous Stairs operating point was partly limited by its
Gaussian initialization.

## Canonical indoor data roots

The downloaded archives passed `unzip -tq` and are exposed through
non-destructive canonical roots:

- `/mnt/pool/sqy/7scenes`
- `/mnt/pool/sqy/12scenes`
- `/mnt/pool/sqy/Cambridge_stdloc` remains the existing Cambridge source root.

`DATASET_LAYOUT.json` in each indoor root records the raw-data root, reference
archive and SHA, extracted full reference models, all scene names, and the
mapping-only RGB/full-reference-point contract. Symbolic links avoid a second
physical copy of the raw datasets and reference reconstructions.

All seven 7Scenes and twelve 12Scenes full reference models are present. Their
point counts are recorded in the machine evidence. The data preparation code
supports every scene; this round physically prepared and trained representative
priors for Stairs and `office2/5b`. The remaining 17 scene priors are therefore
mechanical materialization work, not yet completed scientific results.

## Exact preparation semantics

With `--reference-model`, preparation now:

1. imports published poses, intrinsics, test list, and the complete
   `points3D.bin`/`points3D.txt`;
2. rectifies the source camera model into the recorded pinhole domain;
3. places only mapping RGB in `prior_input/images`;
4. links the full reference point cloud into both COLMAP trees;
5. discards per-image SfM feature observations; and
6. records point path, format, count, SHA, and role in `dataset_manifest.json`.

No test RGB is used as Gaussian supervision. The public point cloud may contain
reconstruction evidence from the full registered image set; this is disclosed
as `may_include_test_view_reconstruction_evidence=true` rather than being
misrepresented as a mapping-only geometric reconstruction.

## Representative 30k Gaussian rebuilds

Both priors use the same vanilla 2DGS trainer and 30,000 iterations.

| Scene | Initialization | Initial points | Final Gaussians | train PSNR | train L1 |
|---|---|---:|---:|---:|---:|
| Stairs old | mapping-RGB known-pose retriangulation | 55,660 | 319,972 | 29.7741 | 0.022541 |
| Stairs full reference | published `sfm_gt` | 131,766 | 321,622 | 29.7005 | 0.022959 |
| office2/5b old | mapping-RGB known-pose retriangulation | 257,087 | 773,020 | 21.1751 | 0.069131 |
| office2/5b full reference | published `sfm_gt` | 580,447 | 767,796 | 19.7990 | 0.070450 |

Training-view PSNR is not a reliable proxy for localization. The full-reference
prior is slightly worse on Stairs and substantially worse on `office2/5b` by
this metric, so the new initialization is not accepted merely because its
point cloud is larger.

## Stairs render-only structural replay

The V1.4 method was replayed without source mapping RGB, parameter tuning, or
test-driven map construction. Only the Gaussian prior changed.

| Quantity | old retriangulated prior | full-reference prior | delta |
|---|---:|---:|---:|
| trajectory-balanced source Tracks | 72,269 | 72,900 | +631 |
| support hard rejects | 6,389 | 3,045 | -3,344 |
| support-repaired broad Tracks | 15,072 | 15,210 | +138 |
| support-repaired observations | 729,393 | 715,475 | -13,918 |
| compact selector anchors | 5,811 | 5,702 | -109 |
| final matching-rank P10 | 51 | 53 | +2 |
| strong teacher positives | 185,887 | 161,460 | -24,427 |

The result is not a monotone capacity increase. The new prior rejects fewer
support edges and improves matching coverage with a smaller deployed map, but
also yields fewer observations and fewer strong teacher positives.

## Mapping-only LOO result

All 2,000 mapping cameras participate in construction. For each mapping query,
its observations are removed only from affected Track descriptor fusion before
the same one-global-Top-1/one-PoseLib evaluation.

| Metric | old V1.4 | full-reference V1.4 | delta |
|---|---:|---:|---:|
| median TE (cm) | 0.3631 | 0.3637 | +0.0006 |
| mean TE (cm) | 6.1029 | 3.6779 | -2.4250 |
| P90 TE (cm) | 0.9598 | 0.9280 | -0.0318 |
| P99 TE (cm) | 168.1795 | 30.3503 | -137.8292 |
| CVaR95 TE (cm) | 114.2736 | 65.7437 | -48.5298 |
| 5 cm / 5 deg recall | 96.60% | 98.55% | +1.95 pp |
| catastrophic queries (>=1 m) | 22 | 13 | -9 |
| raw GT precision | 7.2787% | 6.3264% | -0.9523 pp |

This is a tail improvement rather than a universal correspondence improvement.

## Frozen real-test result

After the map, metric, calibration, and implementation were frozen, test RGB
was used only for final evaluation with PoseLib seeds 2026/2027/2028. Values
below are seed means over the same 1,000 Stairs test queries.

| Metric | old V1.4 | full-reference V1.4 | delta |
|---|---:|---:|---:|
| median TE (cm) | 2.2730 | **2.1444** | -0.1286 |
| mean TE (cm) | 10.9263 | **7.1927** | -3.7336 |
| P90 TE (cm) | 9.3347 | **6.5601** | -2.7746 |
| mean AE (deg) | 3.1415 | **1.6217** | -1.5198 |
| 2 cm / 2 deg recall | 42.80% | **46.97%** | +4.17 pp |
| 5 cm / 5 deg recall | 82.97% | **85.10%** | +2.13 pp |
| raw GT precision | 2.9535% | **2.9868%** | +0.0333 pp |
| inlier GT precision | 20.9877% | **21.9854%** | +0.9977 pp |
| catastrophic queries (>=1 m) | 15.67 | **9.67** | -6.00 |

All three seeds improve mean/P90 translation, both recalls, and catastrophic
count. The full-reference prior therefore replaces the retriangulated prior as
the indoor initialization contract for future render-only work.

It still does not eliminate coherent false consensus. Relative to the previous
mixed Gaussian+Track mainline, typical Stairs accuracy is now close, but mean
error and catastrophic count remain worse. The next useful work is therefore
tail/consensus repair on top of this corrected prior, not a return to the old
initialization.

## Scope of earlier Stop conclusions

- **Must be re-scoped or rerun:** render-only V1.4/R1/conditional-fusion results
  whose feature cache, support evidence, Track components, or selector map was
  rendered from the old Gaussian prior. Their old artifacts remain valid for
  that prior, but are not evidence against the corrected prior.
- **Unchanged:** P8 pair-selection V1/V2, XFeat detector/descriptor gates, and
  other experiments built from frozen original mapping-RGB caches. They do not
  consume Gaussian renders, so changing Gaussian initialization cannot repair
  their measured failures.
- **Not authorized:** test-driven map selection, test RGB in Gaussian training,
  or claiming all 19 scene priors have already been trained.

Exact paths, hashes, scene inventories, and per-seed summaries are recorded in
[`docs/evidence/indoor_full_reference_prior_revalidation.json`](evidence/indoor_full_reference_prior_revalidation.json).
