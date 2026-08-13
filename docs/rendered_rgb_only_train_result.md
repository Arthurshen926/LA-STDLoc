# Gaussian-rendered RGB-only Track map: train-only result

Status: **geometry feasibility established; descriptor/localization route not yet
ready to replace the main map**.

This experiment deliberately used only the four Stairs mapping trajectories.
It did not read source mapping RGB, test images, test descriptors, or test pose
summaries.  RGB was rendered at the original mapping poses from the supplied
Gaussian prior.  SuperPoint features, camera-pair matching, Track construction,
ray triangulation, map selection, descriptor training, and every localization
diagnostic below consumed only those rendered mapping views.

## Constructed map

The trajectory-balanced graph used the same 7,450-pair budget as the nearest
graph: 5,022 within-trajectory pairs and 2,428 balanced cross-trajectory pairs.
Reciprocal epipolar matching and cycle/chain assembly produced 72,269 Tracks;
15,533 triangulated and 14,855 passed the broad Track contract.  Triangulation
used camera rays only, with median reprojection error 0.959 px, P90 2.209 px,
and median parallax 2.802 degrees.  No rendered depth or Gaussian primitive
position entered triangulation.

Unlike the nearest graph, the balanced graph produced 1,876 selected Tracks
supported by more than one mapping trajectory (12.63%).  The positive teacher
contains 2,000 mapping records, 1,155,567 positive rows, 1,885,996 strong
pairs, and 721,040 exact Track-observation positives.

## Train-only localization evidence

All localization numbers are leave-one-mapping-sequence-out: the held sequence
does not contribute to its map descriptors or training objective.

| Variant | Scope | Recall @ 5 cm/5 deg | Median TE | CVaR95 TE | Catastrophic >=1 m |
|---|---:|---:|---:|---:|---:|
| One fused Track descriptor | 4-fold combined | 77.85% | 1.861 cm | 467.89 cm | 222 |
| + harmful-Track capacity selection | 4-fold combined | 79.60% | 1.694 cm | 478.93 cm | 212 |
| + two view-bin prototypes | 4-fold combined | 79.70% | 1.765 cm | 470.92 cm | 206 |
| Capacity map | held seq-05 | 42.00% | 58.68 cm | 534.87 cm | 168 |
| Two prototypes | held seq-05 | 42.60% | 57.86 cm | 536.67 cm | 164 |
| Top-4 candidates, one PnP | held seq-05 | 42.60% | 60.67 cm | 643.28 cm | 158 |
| Eight-view pooled observation retrieval | held seq-05 | 35.00% | 87.46 cm | 317.43 cm | 209 |
| Eight-view reciprocal pair retrieval | held seq-05 | 37.20% | 86.80 cm | 517.38 cm | 211 |
| Anchor-specific descriptor residual | held seq-05 | 42.20% | 58.64 cm | 519.21 cm | 172 |
| Shared metric + soft-pose loss | held seq-05 | 42.00% | 58.49 cm | 552.22 cm | 167 |
| Geometry-positive oracle | strict held seq-05 | **100.00%** | **0.171 cm** | n/a | **0** |

Capacity selection removed 295/14,855 Tracks.  The removed rows had zero
correct winners and zero clean inliers, versus 24,672 false-attractor wins and
6,024 harmful inliers; all mapping queries retained the requested matching
rank.  This is a real, modest train-only improvement, but it does not repair the
seq-05 failure domain.

The positive-rank audit explains the remaining gap.  Across the four held
mapping sequences, a geometry-positive anchor appears in descriptor Top-1,
Top-4, Top-16, and Top-64 for respectively 19.86%, 36.37%, 52.60%, and 66.14%
of positive query rows.  Seq-05 is not missing geometry: its corresponding
fractions are 21.85%, 38.79%, 54.17%, and 66.07%.  Correct geometry exists in
the candidate set, but independent row ranking creates a coherent wrong pose.

The strict geometry oracle is decisive.  With the seq-05 descriptor bank built
only from seq-02/03/06-supported anchors, projected positive correspondences
localize all 500 seq-05 mapping frames; median positive count is 97.5.  Thus the
Gaussian-rendered images can generate sufficient independent 3D geometry.  The
present bottleneck is assigning query observations to that geometry, not the
Gaussian prior's primitive geometry and not Track coverage.

## Decision

Keep and port the reusable infrastructure:

- rendered-RGB feature materialization with explicit no-source-RGB/no-test
  lineage;
- trajectory-balanced fixed-budget camera-pair graph;
- reciprocal epipolar + cycle/chain Track construction and pure-ray
  triangulation;
- rendered Track positive-teacher materialization;
- mapping-sequence crossfit, capacity selection, rank audit, and geometry
  oracle.

Do not make the current rendered-only map the default, and do not continue
parameter sweeps over prototype count, Top-K PnP, reference-view count, bounded
anchor residuals, or the existing soft-pose objective.  Those distinct tests
all preserve the same seq-05 failure.

The next justified method change is a query-conditioned **set-level** Track
assignment model: it must learn compatible groups of 2D-to-3D assignments
rather than score each query row independently.  It should still use the
Gaussian-rendered mapping views and ray-triangulated Tracks, and should first
be trained/evaluated by mapping-sequence crossfit.  Test evaluation remains
deferred until that train-only gate shows a material improvement on seq-05.

