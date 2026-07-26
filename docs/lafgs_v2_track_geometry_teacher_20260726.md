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
- Canonical runners now fail closed: 2DGS defaults to G3
  `track_first_provenance`, 3DGS defaults to G2 `track_first`, and G0
  `map_top1` requires an explicit legacy-control opt-in.
- The sanitization CLI and runners share one mode contract. Continuous
  `loc_geo` ranking is legacy-only; exact high-confidence reject-only is a
  canonical mode.
- Epipolar-first top-K track matching is implemented. It searches descriptor
  top-K, applies the known-pose epipolar gate, and then applies reciprocal and
  gated-margin tests.
- Graded track construction is implemented. Cycle-supported edges are merged
  first, reciprocal epipolar chain edges are then added in confidence order,
  and a merge is rejected if it would create two observations from one query
  in the same component. Level A denotes a cycle-seeded component and Level B
  a pure chain component; Level A may contain chain extensions.
- G3 now retains a landmark-side CSR set of supporting tracks, normalized
  responsibilities, effective support, and triangulated-point RMS/max
  disagreement. These fields are diagnostic only: conflicting tracks are not
  averaged into the repair target.
- Selective localization-only surface repair is executable. The triangulated
  target is projected into the frozen Gaussian's allowed tangent/normal or
  covariance support, and a consistent reversible `raw_anchor_offset` is
  saved. RGB geometry and descriptors remain frozen.

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

### Epipolar-first top-K ablation

All rows use the same 895-image native cache and frozen 48K field.

| Track matcher | Tracks | High-confidence tracks | G2 assigned landmarks | C1 conditional AUROC/AUPRC |
| --- | ---: | ---: | ---: | ---: |
| T0 global top-1 | 40,348 | 1,579 | 1,211 | 0.853/0.360 |
| T1 epipolar-first top-4 | 46,688 | 1,995 | 1,504 | 0.847/0.320 |
| T2 epipolar-first top-8 | 47,666 | 2,013 | 1,517 | 0.847/0.315 |

T1 recovers substantially more independent tracks, while T2 adds little over
T1. The extra G2 nearest assignments are less pure, so top-4 is the useful
coverage operating point and top-8 is not justified.

With G3 group-4 provenance, T1 increases assigned landmarks from 1,806 to
2,197 and nominal coverage from 3.762% to 4.577%. Conditional C1 AUROC/AUPRC
changes from 0.938/0.922 to 0.925/0.904. At provenance consensus rate 0.35,
T1 retains 2.283% coverage with 0.945/0.935 AUROC/AUPRC. This is a real
precision-coverage frontier expansion, not a uniformly dominant teacher.

### Graded cycle/chain track result

T3 uses the T1 epipolar-first top-4 matcher and adds conflict-aware graded
cycle/chain components.

| Teacher | Tracks | High-conf. tracks | Assigned landmarks | Coverage | Conditional AUROC/AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 G2 strict cycle | 46,688 | 1,995 | 1,504 | 3.133% | 0.847/0.320 |
| T3 G2 graded | 69,772 | 3,097 | 2,240 | 4.667% | 0.830/0.309 |
| T1 G3 group-4 | 46,688 | 1,995 | 2,197 | 4.577% | 0.925/0.904 |
| T3 G3 group-4 | 69,772 | 3,097 | 3,193 | 6.652% | 0.953/0.940 |

The graded builder accepted 490,737 cycle edges and 277,568 chain edges while
rejecting 10,276 query-collision merges. G2 nearest assignment again loses
purity, so chain tracks are not a safe geometry teacher by themselves. Frozen
2DGS provenance reverses that result: the T3 G3 precision-coverage frontier
strictly improves over T1 G3. The cycle-seeded subset covers 6.302% of the bank
at 0.957/0.943 AUROC/AUPRC. Pure Level-B contributes only another 0.350%, at
0.881/0.887, and should remain weak/abstaining evidence.

The real G3 multi-track audit finds 972 of 3,193 assigned landmarks (30.44%)
have more than one supporting track, with mean effective support 1.55. Of
these, 831 have a maximum triangulated-point disagreement above 3 cm. This is
not evidence that all 831 tracks are wrong: a large surfel/group can explain
multiple valid surface locations. It does show that one Gaussian primitive
cannot safely be collapsed to one averaged localization anchor. The result
supports track-derived one-to-many coverage anchors and keeping multi-track
dispersion as an abstention cue.

### Selective reject and repair

The downstream test uses all 182 development images, native full-resolution
SuperPoint 2048, uncapped cosine top-1, and one RANSAC/PnP. It remains a sparse
single-stage evaluation.

| C1 map operation | Median TE | Mean TE | P90 TE | R5 | Raw P@2 | Inlier P@2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 cm corruption | 12.378 cm | 22.793 cm | 48.355 cm | 12.09% | 3.577% | 21.546% |
| reject 47 | 12.364 cm | 22.742 cm | 48.021 cm | 12.09% | 3.577% | 21.564% |
| reject 47 + repair 72 | 12.009 cm | 22.615 cm | 47.816 cm | 12.64% | 3.621% | 21.822% |
| T3 repair 112 only | 12.106 cm | 22.588 cm | 48.147 cm | 13.19% | 3.636% | 21.887% |
| T3 reject 72 + repair 112 | 12.175 cm | 22.568 cm | 47.816 cm | 13.19% | 3.637% | 21.910% |

All 47 rejected and all 72 repaired anchors are controlled corruptions; no
clean anchor is modified. The same operation on the clean map selects zero
rejects and zero repairs, and preserves IDs, descriptors, positions, and raw
offsets bit-for-bit. The strict three-view-bin repair gate selects only six
anchors; the reported causal row uses the teacher's established two-bin
high-confidence gate. This exposes camera-bin coverage, rather than repair
precision, as the immediate bottleneck.

T3 also makes no change to the clean map, while its C1 operation acts on 184
controlled corruptions with zero clean false actions. It improves mean error
and 5 cm recall over T1, but not the median. Repair-only is better than
reject-and-repair on the median, showing that deleting even correctly
identified corrupted landmarks can perturb the uncapped top-1/RANSAC candidate
set unfavorably. The old T1 operation remains the best median result, so no
further C1 threshold sweep is justified.

The result supports a narrow claim: a map-independent track teacher can
abstain and selectively repair evidence-covered localization anchors. It does
not support global sanitization: only 119 of 4,800 controlled corruptions are
acted on.

## Decision

The track-first and raster-provenance teacher is now independent of the
localization map identity, and the 2DGS surface-center semantic bug is fixed.
It has strong conditional corruption evidence and selective repair now gives a
small but internally consistent pose improvement. T1 raises nominal evidence
coverage to 4.577%, while full-bank ranking remains close to chance because
unknown anchors are intentionally not assigned a label. This is not enough to
activate global C2/C3 sanitization, physical pruning, or replace the canonical
48K map.

The next geometry experiment should create coverage anchors from stable tracks
for repeatedly unmatchable query regions. Relaxing provenance responsibility
further would inflate nominal coverage with weak neighboring primitives rather
than establish reliable sanitization.

The stage gate deliberately prevents C2/C3 global sanitization, physical
Gaussian pruning, topology changes, and coverage-anchor insertion in the
canonical map. Selective localization-anchor repair is now allowed only for
certified evidence; unknown anchors remain unchanged. True coverage-anchor
insertion still requires separating localization-anchor identity from source
primitive identity, otherwise duplicate primitive IDs would silently violate
training/statistics assumptions.

The G3 provenance group loop previously recomputed each track's observation
count by scanning all observations, giving quadratic-like runtime in the
number of tracks. It now computes those counts once with `bincount`; this is a
runtime fix and does not alter teacher or localization semantics. The same
formal T3 G3 replay fell from about 8m37s to 5m27s. Existing geometry fields
were unchanged apart from NaN equality, and the maximum finite statistics
difference from nondeterministic accumulation was 6.9e-5.
