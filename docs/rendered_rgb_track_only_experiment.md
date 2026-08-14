# Rendered-RGB Track-only localization experiment

## Question

Can the Gaussian prior be used only as an RGB renderer, while all deployed
anchors are reconstructed from rendered-image features and known mapping
poses?  This experiment deliberately excludes original mapping RGB and native
Gaussian anchors from map construction.  Test RGB is read only after the map,
selection, and A1 metric have been frozen.

The two pilot scenes are Stairs (7Scenes) and ShopFacade (Cambridge).  The
machine-readable result is
[`docs/evidence/rendered_rgb_track_only_stairs_shopfacade.json`](evidence/rendered_rgb_track_only_stairs_shopfacade.json).

## Implemented path

1. Render RGB at the original mapping cameras from the frozen Gaussian prior.
2. Run SuperPoint once on each rendered image.
3. Build a nearest-camera pair graph, apply reciprocal descriptor matching and
   known-pose epipolar filtering, then form cycle/chain-consistent Tracks.
4. Triangulate Track geometry from camera rays.  Gaussian primitive positions,
   rendered depth, original mapping RGB, and test queries are not used.
5. Build mapping-only geometric positives and run sequence/blocked-time
   cross-fit.  Remove only repeatedly harmful Track attractors.
6. Feed an empty Gaussian base plus the complete Track universe into the
   standard adaptive selector.  Its Track core, coverage reserve, and pose
   reserve produce a single compact Track-only map.
7. Train the standard bounded low-rank A1 descriptor metric on rendered mapping
   rows, with geometry and Track membership frozen.
8. Only then evaluate real test RGB with the normal sparse frontend, one global
   top-1 map lookup per query row, and one PoseLib solve.

The implementation also supports a one-trajectory scene by using three
contiguous mapping-time blocks for cross-fit.  A compact Track map is allowed
to have zero Gaussian base rows, while serialized Track anchors retain
`source_primitive_ids=-1`.

## Map construction

| Scene | Mapping views | Pairs | All Tracks | Triangulated | Broad input | Harmful exclusions | Final Track anchors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stairs | 2,000 | 7,450 | 72,269 | 15,533 | 14,855 | 295 | 4,283 |
| ShopFacade | 231 | 786 | 35,069 | 15,493 | 14,769 | 117 | 2,521 |

Both final maps contain zero Gaussian anchors and cover every mapping query at
the scene-derived matching-rank target.  ShopFacade's rendered Track geometry
is numerically strong: triangulated-track reprojection median is 0.674 px and
P90 is 1.554 px.  This establishes that rendered RGB can provide useful
geometry independently of Gaussian primitive positions.

## Frozen real-test results

The table reports the mean of seeds 2026/2027/2028.  Lower is better for
translation/rotation errors; higher is better for recall and precision.

| Scene / map | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2cm/2deg recall | 5cm/5deg recall | Raw GT precision 2px | Inlier GT precision 2px | Catastrophic >=1m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stairs rendered Track-only + selector + A1 | 3.939 | 14.905 | 23.325 | 5.355 | 6.20% | 65.53% | 0.107% | 0.666% | 27.0 |
| Stairs full broad Track identity control | 4.373 | 12.221 | 29.994 | 3.554 | 10.03% | 58.50% | 0.138% | 0.667% | 14.0 |
| Stairs existing mixed mainline | 2.135 | 4.946 | 6.873 | 1.024 | 46.60% | 84.87% | 3.621% | 22.426% | 1.0 |
| ShopFacade rendered Track-only + selector + A1 | 2.514 | 5.756 | 8.135 | 0.282 | 40.78% | 78.64% | 7.100% | 42.270% | 1.0 |
| ShopFacade selected Track-only identity control | 2.503 | 5.784 | 8.143 | 0.283 | 39.81% | 78.64% | 7.092% | 42.284% | 1.0 |
| ShopFacade existing mixed mainline | 1.993 | 4.745 | 8.318 | 0.224 | 50.49% | 81.55% | 9.979% | 42.538% | 0.0 |

## Interpretation

The initial route is technically viable but not a universal replacement for
the mixed mainline.

- ShopFacade is the positive result.  A 2,521-anchor map reconstructed only
  from rendered RGB approaches the mixed mainline and even has a slightly
  better P90 translation error.  The standard selector performs most of the
  useful work; A1 adds only about 0.97 percentage points of 2cm recall.
- Stairs' initial learned result is negative: raw precision is 0.11% and the
  tail is poor.  However, this initial experiment compared the selected+A1 map
  only to a much larger full-broad identity map.  It did not yet distinguish a
  render-to-real appearance gap from a harmful learned metric on the selected
  map.
- The initial evidence therefore establishes useful projective geometry, but
  is insufficient for a causal claim about descriptor quality.  The matched
  V1.1 controls below resolve that ambiguity.

The Stairs comparison above did not include an identity-metric control on the
same 4,283 selected anchors.  It therefore cannot by itself attribute the
failure to rendered descriptors: it mixes descriptor learning with a large
map-cardinality change.  The V1.1 experiment below supplies that missing
matched control and supersedes that causal interpretation.

## V1.1: support-aware rendered appearance follow-up

V1.1 keeps the selected Track identities, triangulated positions, map size,
mapping poses, and real-test protocol fixed.  It makes two bounded changes,
both frozen before reading test RGB:

1. At each mapping pose, render seven deterministic appearances from the same
   Gaussian prior: identity, exposure 0.8/1.2, gamma 0.85/1.15, and fixed
   warm/cool white-balance plus contrast transforms.  Detect keypoints only on
   the identity render, describe the same rows in all seven renders, retain the
   six most mutually consistent descriptors, and normalize their mean.
2. Use rendered alpha and depth as support evidence during descriptor fusion
   and A1 teacher construction.  Invalid observations are ignored when a Track
   has supported alternatives; A1 positives must also pass alpha >= 0.05 and
   `|z_anchor-z_render| <= 0.05 + 0.02 |z_render|`.

This remains source-image-free.  Original mapping RGB, test queries, and
Gaussian primitive positions are absent from construction.  Rendered depth is
used only as visibility/support evidence; Track xyz remains the result of
multi-view feature triangulation.  The implementation commit is `3f2a682` and
the machine-readable follow-up is
[`docs/evidence/rendered_rgb_track_only_appearance_v11.json`](evidence/rendered_rgb_track_only_appearance_v11.json).

The identity rerender reproduced the original descriptors closely (minimum
cosine 0.999990 on Stairs and 0.999993 on ShopFacade).  All Track IDs, xyz,
types, source IDs, and map cardinalities are bitwise unchanged.  On Stairs the
support test removed 1,008,157 projected anchor observations from teacher
eligibility and reduced strong pairs from 770,151 to 606,867; on ShopFacade it
removed 278,798 projections and reduced strong pairs from 131,336 to 92,580.

### Frozen V1.1 real-test results

All entries are means over PoseLib seeds 2026/2027/2028.  The Stairs
single-render identity row is the newly added matched control: it uses exactly
the same 4,283 selected Track anchors as the learned variants.

| Scene / fixed selected map | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2cm/2deg recall | 5cm/5deg recall | Raw GT precision 2px | Inlier GT precision 2px | Catastrophic >=1m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stairs single-render identity (matched) | 2.549 | 11.806 | 18.106 | 3.685 | 37.77% | 77.30% | 2.746% | 19.156% | 19.0 |
| Stairs 7-appearance identity metric | 2.520 | 11.488 | 18.614 | 3.513 | 38.03% | 77.03% | 2.744% | 19.156% | 18.33 |
| Stairs 7-appearance + support-aware A1 | **2.491** | 11.639 | **10.566** | 3.660 | 37.73% | **77.70%** | **2.767%** | **19.254%** | 18.67 |
| Stairs existing mixed mainline | 2.135 | **4.946** | 6.873 | **1.024** | **46.60%** | 84.87% | 3.621% | 22.426% | **1.0** |
| ShopFacade single-render selected identity | 2.503 | 5.784 | **8.143** | 0.283 | 39.81% | 78.64% | 7.092% | 42.284% | 1.0 |
| ShopFacade 7-appearance identity metric | **2.489** | 5.818 | 8.275 | 0.285 | 39.81% | 78.64% | 7.091% | **42.358%** | 1.0 |
| ShopFacade 7-appearance + support-aware A1 | 2.501 | 5.781 | 8.270 | 0.283 | 39.81% | 78.64% | 7.094% | 42.330% | 1.0 |
| ShopFacade original single-render A1 | 2.514 | **5.756** | 8.135 | **0.282** | **40.78%** | 78.64% | **7.100%** | 42.270% | 1.0 |

### Revised conclusion

- The former Stairs result was not primarily a render-to-real descriptor
  collapse.  The matched selected identity control already reaches 37.77%
  2cm recall and 2.746% raw precision.  The old A1 result (6.20% and 0.107%)
  was chiefly a learned-metric failure under a teacher that lacked rendered
  support checks.
- Support-aware A1 prevents that failure and improves Stairs median/P90 TE,
  5cm recall, and correspondence precision relative to the matched identity
  control.  The P90 improvement is large (18.106 to 10.566 cm), but mean error
  and catastrophic failures remain far behind the mixed mainline.
- The gentle seven-appearance descriptor ensemble is nearly neutral.  It does
  not materially improve either scene by itself.  On ShopFacade the original
  single-render A1 retains the best 2cm recall and mean/P90 translation error,
  so V1.1 is not a universal replacement.
- The route is therefore retained as a real source-image-free experimental
  pipeline, with a mandatory matched identity control and an A1 non-regression
  requirement.  The next useful work is tail-oriented Track support: rendered
  alpha/depth-aware edge evidence, depth-cycle checks, and bounded component
  splitting or repair.  Stronger render perturbations are a separate
  descriptor hypothesis and must not be conflated with that geometric tail
  repair.

No test query selected the transforms, support thresholds, map, or metric.
Test RGB was used only once the mapping-only artifacts were frozen.

## V1.2: support-certified Track repair and full-chain validation

V1.2 tests the geometric follow-up proposed by V1.1.  It still constructs the
map without original mapping RGB and without Gaussian primitive positions.
Rendered RGB supplies keypoints and descriptors; rendered alpha/depth supplies
only visibility and uncertainty evidence; final xyz is recomputed from camera
rays and feature correspondences.

The repair is deliberately bounded.  Reciprocal, epipolar-filtered matches may
split observations only inside an existing rendered Track, never merge two
source Tracks.  A source Track may yield at most three children.  High-alpha,
smooth-depth edges with cycle error above 8 px or depth disagreement above
three local sigmas are hard rejected; uncertain evidence remains soft rather
than being deleted.  Every retained child is retriangulated with the original
minimum-view, view-bin, parallax, reprojection, and conditioning checks.

Two map hypotheses were evaluated before test:

- **E+R** preserves the V1.1 selected source-Track membership and maps each
  source identity to its strongest repaired broad child.  It retains 2,504 /
  2,521 ShopFacade anchors and 4,256 / 4,283 Stairs anchors.
- **E+R+S** reruns the generic selector on the repaired pool.  This hypothesis
  is mapping-only and was rejected because it improved some typical metrics
  while worsening tail risk or recall.  It was not taken to test.

The resulting structure is useful but not a default accuracy improvement:

| Scene | Source Tracks | Repaired Tracks | Split sources | Broad before | Broad after | Certified observations | Frozen E+R anchors |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade | 35,069 | 36,856 | 2,014 | 14,769 | 14,727 | 239,473 | 2,504 |
| Stairs | 72,269 | 72,497 | 455 | 14,855 | 15,072 | 690,354 | 4,256 |

### Mapping-only support-fold audit

The cross-fit evaluator removes held observations, rebuilds each fold's Track
components, retriangulates fold xyz/covariance, and then runs mapping
self-localization.  Thus the held observation is not present indirectly through
the deployed Track geometry.

| Scene / arm | Catastrophic | CVaR95 TE cm | Mean TE cm | Median TE cm | P90 TE cm | Raw GT precision | 5cm recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade V1.1 control | 7 | 829.113 | 44.417 | 0.564 | 3.224 | 11.491% | 91.775% |
| ShopFacade E+R | 9 | **581.679** | **31.503** | 0.601 | 3.322 | 12.550% | 91.775% |
| ShopFacade E+R+S | 9 | 932.662 | 49.399 | **0.452** | **3.077** | **17.878%** | 91.342% |
| Stairs V1.1 control | 309 | 626.787 | 62.423 | 2.042 | 272.801 | **3.613%** | 70.600% |
| Stairs E+R | 322 | **611.046** | **62.227** | **2.017** | **272.189** | 1.709% | 70.600% |
| Stairs E+R+S | 366 | 752.625 | 76.221 | 2.087 | 283.853 | 1.788% | 69.750% |

E+R was therefore frozen.  ShopFacade selected its exact map-bound identity
metric; Stairs selected its mapping-trained A1 metric because mapping tail
metrics improved while recall changed by only -0.05 percentage points.  Both
teachers are bound to the exact query cache and mapping calibration used to
train/evaluate them.

### Frozen real-test results

All rows below are means over PoseLib seeds 2026/2027/2028.  Test data was read
only after the map, metric, and per-scene choice had been frozen.  GPU1 and
GPU2 were used in parallel where the frontend permitted it; utilization is
bursty because every GPU frontend batch is followed by CPU PoseLib work.

| Scene / frozen route | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2cm recall | 5cm recall | Raw GT precision | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade V1.1 identity | 2.489 | **5.818** | **8.275** | **0.285** | 39.806% | 78.641% | **7.091%** | 1.0 |
| ShopFacade V1.2 E+R identity | **2.434** | 5.929 | 8.393 | 0.292 | **40.777%** | **79.612%** | 6.686% | 1.0 |
| Stairs V1.1 A1 | 2.491 | **11.639** | **10.566** | **3.660** | 37.733% | 77.700% | **2.767%** | **18.667** |
| Stairs V1.2 E+R A1 | **2.458** | 12.430 | 16.606 | 4.269 | **39.233%** | **78.600%** | 2.756% | 24.667 |

V1.2 raises 2cm/5cm recall by +0.971/+0.971 percentage points on ShopFacade
and +1.500/+0.900 points on Stairs.  It simultaneously worsens ShopFacade
mean/P90 translation slightly and worsens the Stairs tail materially: mean TE
+0.790 cm, P90 TE +6.040 cm, and catastrophic failures +6.0.  Two Stairs
queries become seed-stable new catastrophics, and the broader failures cluster
in `seq-01`; this is coherent global Top-1/PoseLib consensus, not a failure of
the local triangulation checks alone.

### V1.2 decision

The support-repair implementation is retained as a source-image-free
structural prototype, but V1.2 is **not promoted to the default map**.  Local
support thresholds, a larger pair set, or another generic selector rerun are
not justified by these results.  The next distinct hypothesis must protect
deployment-rank / false-consensus behavior using mapping-only evidence (for
example, a query-conditioned set-level Track assignment) and must be frozen
before any further test evaluation.  Machine-readable lineage and all formal
hashes are in
[`docs/evidence/rendered_rgb_track_support_v12.json`](evidence/rendered_rgb_track_support_v12.json).
