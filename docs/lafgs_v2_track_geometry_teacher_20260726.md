# LaFGS V2 Track-First Geometry Teacher

## Locked protocol

- Scene: Cambridge OldHospital.
- Training images: all 895 mapping frames; no validation split.
- Frozen map: canonical 48K A4 RGB-only 2DGS.
- Query frontend: native full-resolution SuperPoint, 2048 keypoints.
- Deployment reference: uncapped cosine top-1 and one RANSAC/PnP.
- Geometry-teacher experiments do not modify descriptors or use the 182-image
  development set.

The exact canonical 2DGS and dirty 3DGS artifacts are recorded in
`configs/locaware/lafgs_v2_oldhospital_canonical_48k_a4.json`.

## Implemented changes

- G0 map-top1 and G1 GT-clean-map-top1 controls.
- G2 image-only reciprocal SuperPoint matching, known-pose epipolar gating,
  cycle-supported tracks, and robust triangulation.
- G3 frozen 2DGS raster provenance assignment.
- Center-and-view-axis camera bins and robust p75 ray parallax.
- Projection-Jacobian covariance and rendered-depth confidence gates.
- Multi-positive source responsibility in landmark statistics.
- Surface-aware 2DGS evidence: tangent support and normal displacement are
  measured in the surfel frame. Natural track-to-center tangent distance is no
  longer treated as geometric corruption.
- Sparse G3 track-to-Gaussian groups with bounded provenance responsibility.
- Optional map-independent LGCV on 2D-2D track edges. Both a hard gate and a
  soft triangulation-confidence mode are implemented; the feature is disabled
  by default because the controlled result below is negative.

## Controlled C1 result

C1 moves localization anchors by 0.1 m while leaving the frozen RGB renderer
unchanged.

| Teacher | Evidence coverage | Conditional AUROC | Conditional AUPRC | All-bank AUROC |
| --- | ---: | ---: | ---: | ---: |
| G0 map top-1 | 1.175% | 0.743 | 0.155 | 0.498 |
| G1 GT-clean top-1 | 1.190% | 0.938 | 0.436 | 0.498 |
| G2 track-first nearest | 2.523% | 0.853 | 0.360 | 0.500 |
| G3 hard provenance, center metric | 1.781% | 0.685 | 0.155 | 0.499 |
| G3 hard provenance, surface metric | 1.781% | 0.953 | 0.942 | 0.504 |
| G3 group-4 provenance, surface metric | 3.762% | 0.938 | 0.922 | 0.506 |

For hard G3, a surface mismatch threshold of 0.25 selected 73 of 84
evidence-covered corruptions with 100% precision and 86.9% recall. Group-4
assignment increased the number of covered primitives from 855 to 1806 and
query-support edges from 14,978 to 34,050, with a small loss in conditional
ranking quality.

### LGCV track-edge ablation

All three rows use the same frozen 48K field and the same 599,896 reciprocal
epipolar edges. The LGCV cue uses only local triangles in the two images and
does not consume map identity, map geometry, or development queries.

| G2 track mode | Evidence coverage | Conditional AUROC | Conditional AUPRC | FPR@95 |
| --- | ---: | ---: | ---: | ---: |
| cycle only | 2.523% | 0.853 | 0.360 | 0.415 |
| hard LGCV | 1.256% | 0.846 | 0.337 | 0.419 |
| soft LGCV confidence | 2.510% | 0.847 | 0.362 | 0.461 |

Hard LGCV retained 265,859 edges, reduced high-confidence tracks from 1,579
to 754, and halved evidence coverage. Soft LGCV retained all cycle components
and only downweighted low-support measurements; it preserved coverage but
worsened AUROC and FPR@95. Local 2D similarity is therefore not a reliable
sparse-track identity prior under the perspective/depth variation in this
scene. LGCV remains appropriate as a protected semidense measurement loss, not
as a gate in the track-first teacher.

## Decision

The track-first and raster-provenance teacher is now independent of the
localization map identity, and the 2DGS surface-center semantic bug is fixed.
It has strong conditional corruption evidence, but only 3.76% full-bank
coverage and an all-bank AUROC of 0.506. This is not enough to activate global
C2/C3 sanitization, automatic repair, or replace the canonical 48K map.

The next geometry experiment should create coverage anchors from stable tracks
for repeatedly unmatchable query regions. Relaxing provenance responsibility
further would inflate nominal coverage with weak neighboring primitives rather
than establish reliable sanitization.

The stage gate deliberately prevents C2/C3 global sanitization, physical anchor
repair, topology changes, and coverage-anchor insertion in the canonical map.
Those operations would turn 3.76% conditional evidence into unsupported labels
for the other 96.24% of the bank. This is an unresolved method limitation, not
an omitted deployment feature.
