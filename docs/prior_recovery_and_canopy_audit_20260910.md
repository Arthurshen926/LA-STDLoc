# Remaining-prior recovery and matched-camera canopy audit

## Completion semantics

The preceding execution completed the mainline 24 scenes and 25/30 Cambridge
prior cells. Five cells were missing: GreatCourt/OldHospital i3DGS raw/+1k,
and ShopFacade Vanilla 2DGS. The new queues below recover these cells without
retraining their successful reconstructions or overwriting old experiments.
Running a queue is not equivalent to a completed full-test evaluation.

## Repairs

1. i3DGS final `native/3/hierarchy.ply` contains 11 NaN-quaternion leaves among
   2,661,300 on GreatCourt and one among 822,641 on OldHospital. Their scales
   are also NaN. The earlier root-level PLY has valid rotations; it is not the
   final optimized output and is not substituted. The exporter previously
   called scipy's rotation conversion before its non-finite filter, aborting
   the entire scene. It now masks invalid rotations through conversion and
   excludes those rows explicitly. Temporary identity rotations are never
   exported. Both repaired exports completed, retaining 2,661,289 / 822,640
   primitives. Reports count invalid rotations. NaN/zero rotation tests pass.
2. ShopFacade Vanilla 2DGS full-resolution rendering exhausts intersection-list
   memory at mapping frame `seq2/frame00026.png`. Inference now retries only
   this OOM with 32px instead of 16px tiles, after releasing the failed call's
   intermediates. No resolution reduction, opacity threshold, Gaussian pruning,
   camera change, or feature-budget change. Training and repeated OOMs still
   fail explicitly. The failing view renders at approximately 10.15 GB peak
   allocated memory on the 32px path. All 231 observation views subsequently
   completed in the actual recovery queue.
3. Direct loading of Long-LRM's byte-normal PLY exposed a structured-array
   stride incompatibility in the diagnostic frozen model loader. Opacity is
   now made contiguous before conversion to Torch. This is a loader robustness
   fix, **not an explanation of the previously completed localization errors**:
   localization uses the exported all-float PLY, which already loaded normally.

57 focused regression tests pass across adapters, OOM handling, byte-normal
loading, image/pose contracts, render validity, pipeline construction and queue
handling. The full repository suite has not been rerun for these changes.

### Tile parity qualification

Probe: `/mnt/pool/sqy/anygsloc_shop_tile_recovery_20260910/tile_probe.json`.
Across seven views that fit both tile sizes, mean RGB absolute differences are
about 1.2e-6–5.5e-6. Results are **not bit-identical**: RGB maxima reach ~0.036;
depth discrepancies can be much larger at individual pixels (up to ~35.6 m).
Do not describe this as exact numerical equivalence. For index 25 the recorded
"16" attempt automatically fell back to 32, hence its zero difference is NOT
an independent 16-versus-32 parity measurement. Shared-host probe timing is
not a deployment benchmark.

## Long-LRM large-scene audit

Artifacts:

- `/mnt/pool/sqy/anygsloc_long_lrm_large_scene_audit_20260910_v2/report.json`
- `/mnt/pool/sqy/anygsloc_long_lrm_render_settings_20260910/report.json`

Independent Torch reproduction of upstream normalization and actual input
camera checks were performed; selected inputs all have the same 1920x1080
dimensions. Canonical-versus-exported-world rendering PSNR is about 76–95 dB
across 18 mapping views (including ShopFacade as a control), with RGB MAE
around 1e-6. This argues strongly against a large export Sim(3)/quaternion/SH
frame error. It does not establish correctness of the learned geometry.

The exported and canonical renders both exhibit smeared building edges and
overlaid structure. For sampled input views StMarysChurch render/reference
PSNR is 14.3–16.5 dB; some mapping views farther from selected inputs are
10.9–13.6 dB. These are small diagnostic samples, not full-scene image-quality
scores or localization results.

Upstream uses white background, classic rasterization, near=.01 and far=1e6
in normalized units. The current mapping pipeline uses black/antialiased and
world-space clipping. A controlled render-settings diagnostic separates
white-only from the combined upstream settings. StMarysChurch sampled PSNR
improvements with native settings are +0.30/+0.31/+0.58 dB; GreatCourt
+0.40–0.44 dB. Visible reconstruction weakness persists. No claim is made that
this modest photometric improvement fixes final pose accuracy. Existing
matrix results have not silently been replaced by this settings variant.

### Supplemental budget control

`+1k` means 1,000 sampled-view optimizer iterations, not a full mapping pass.
The optimizer pops each view from a shuffled stack before replenishing it.
StMarysChurch has 1,487 views and GreatCourt 1,531, so 487/531 mapping views
are not sampled at all during that 1k optimization. This is a budget limitation,
not proof that it explains all accuracy differences.

A supplemental control uses 2,974 / 3,062 iterations respectively, preserving
the input raw Long-LRM prior, black background, all-mapping data, no
densification, and other training settings; the learning-rate horizon follows
the longer budget. It is explicitly **not** the fixed +1k comparison and does
not isolate iteration count from learning-rate schedule. No test queries are
used for reconstruction or parameter selection. Final tests use both seeds.

## Matched-camera canopy comparison

`/mnt/pool/sqy/anygsloc_stmary_canopy_comparison_20260910_v3/comparison_grid.png`

Four G4Splat camera registry indices: 632/633/634/408, corresponding to
`seq2/frame00075.png`, `seq2/frame00076.png`, `seq2/frame00077.png`, and
`seq1/frame00075.png`. All three Gaussian priors use identical 640x360 camera
poses/intrinsics, black background, and the same crops. Source PLY/image
hashes are in the per-method reports. Original MAtCHA views are reused from
the previously verified matched-camera render, not reconstructed again.

Columns: reference, MAtCHA, Vanilla 2DGS, Vanilla 3DGS, historical G4Splat.
For the three seq2 views, G4 is v310 source-SH-slow step800; seq1 uses v113
mixed-foliage 2k because a corresponding v310 panel was unavailable.

At frame00076 all three priors visibly smooth the foreground canopy into
sheet-like regions despite relatively recognizable background architecture.
G4 v310 retains more leaf texture in this sample but also noise. This is a
qualitative appearance observation, not a recovered hidden-background ground
truth or a proven geometric cause. MAtCHA used 512 optimization views, Vanilla
used full mapping, and G4 settings differ; these panels are not a controlled
training-budget ranking. Views may belong to reconstruction training.

## Active execution artifacts

- i3DGS recovery: `/mnt/pool/sqy/anygsloc_invalid_primitive_recovery_20260910`
  GPU 0 GreatCourt, GPU 1 OldHospital; eight stages each.
- Shop recovery: `/mnt/pool/sqy/anygsloc_shop_tile_recovery_20260910`
  GPU 2; map and two full-test seeds.
- Supplemental Long-LRM control:
  `/mnt/pool/sqy/anygsloc_long_lrm_two_epoch_gpu2_control_20260910`.
  GPU 2 runs StMarysChurch then GreatCourt after its ShopFacade matrix branch
  completed. The preceding `anygsloc_long_lrm_two_epoch_control_20260910`
  wait-only queues were stopped before any training began; their immutable
  code is reused and the handoff is recorded. No training output was discarded.

Each experiment has its own frozen code/hash manifest, outputs, lane state and
logs. Collectors publish only completed evaluations after checking map/result
provenance. Check live state and log progress for completion; this document
is an implementation/audit record, not a dynamic completion ledger.

At the subsequent handoff, OldHospital i3DGS raw and ShopFacade Vanilla 2DGS
both completed full test for both seeds (27/30 matrix cells, three still
outstanding). The merged ledger
`anygsloc_invalid_primitive_recovery_20260910/cambridge_matrix.json` additionally
checks each evaluation's query names exactly equal the scene's official test
list; it does not certify completion from process exit alone.
