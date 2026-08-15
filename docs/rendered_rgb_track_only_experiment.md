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

The V1.2 evaluator removed held observations from already frozen full-mapping
Track identities, retriangulated fold xyz/covariance, and then ran mapping
self-localization.  It did **not** rebuild connected components after removing
the held sequence.  The table below therefore measures support-only geometry
and descriptor replay under fixed identity, not independent formation of the
Track identity.  This limitation was discovered during the V1.3 corrective
audit and the earlier stronger wording is withdrawn.

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

## V1.3 corrective audit: post-geometry children and true component cross-fit

V1.3 implements the review findings without introducing a new learned target
or reading test queries:

- all repaired children are ray-triangulated before the broad gate and the
  maximum-three child limit;
- support-filtered reciprocal match rows are serialized once and reused
  exactly when every held fold rebuilds Track components;
- repaired children retain parent, child index, and parent child-count lineage;
  Selector capacity and pose completion count siblings at parent level;
- exact observations become strong positives only when their own reprojection
  and cycle evidence is certified; other exact rows are ambiguous/ignored;
- source-image-free calibration now requires `uses_source_mapping_rgb=false`,
  `uses_test_queries=false`, and `mapping_source=gaussian_render`;
- the frozen membership audit compares one best child (`max1`) with at most two
  complementary view-bin children (`max2`).

The implementation also fixed two follow-up integration bugs found by formal
execution: the initial `max2` view-bin registry was quadratic, and fold replay
assigned the same rebuilt child to both sibling rows.  The registry is now
linear in observations and fold siblings receive distinct quality-ordered
children.  The adaptive compact map also propagates parent/child lineage to
downstream consumers.

The structural result rules out the pre-triangulation cap as the main Stairs
failure.  ShopFacade had 37,143 unbounded children, of which 14,756 were broad
geometry eligible; the corrected cap removed only three eligible excess
children.  Stairs had 72,503 unbounded children and 15,072 eligible children;
the corrected cap removed zero.  `max2` retained 131 extra ShopFacade children
but only six extra Stairs children.

Observation-level certification is a much larger change.  ShopFacade `max1`
contains 95,852 exact observations, of which 80,206 are strong and 15,646 are
ambiguous.  Stairs contains 371,943 exact observations, of which only 167,186
are strong and 204,757 are ambiguous.  Thus the previous Track-level positive
rule was certifying many chain-only or locally inconsistent observations.

### Corrected mapping-only results

V1.1 below is retained as a historical fixed-identity control.  V1.3 uses the
stronger true component cross-fit, so absolute differences also include the
corrected independence protocol.  The `max1` versus `max2` comparison is exact
within V1.3.

| Scene / arm | Catastrophic | New / fixed catastrophics vs V1.1 | CVaR95 TE cm | Mean TE cm | Median TE cm | P90 / P95 TE cm | Raw GT precision | 5cm recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade V1.1 fixed identity | 7 | - | 829.113 | 44.417 | 0.564 | 3.224 / 44.228 | 11.491% | 91.775% |
| ShopFacade corrected `max1` | 8 | 3 / 2 | **583.272** | **31.805** | 0.596 | 3.242 / 45.039 | 10.269% | 90.909% |
| ShopFacade corrected `max2` | 7 | 2 / 2 | 703.684 | 38.101 | 0.592 | 3.693 / 45.051 | 10.318% | 90.909% |
| ShopFacade entity-aware Selector | 8 | 3 / 2 | 992.702 | 52.525 | **0.470** | 3.256 / **20.032** | **16.396%** | 91.342% |
| Stairs V1.1 fixed identity | 309 | - | 626.787 | 62.422 | 2.042 | 272.800 / 317.555 | **3.613%** | **70.600%** |
| Stairs corrected `max1` | 324 | 33 / 18 | **591.506** | **62.098** | 2.005 | **271.375 / 313.145** | 1.087% | 69.050% |
| Stairs corrected `max2` | 324 | 33 / 18 | 594.905 | 62.359 | **2.000** | 271.660 / 313.573 | 1.087% | 69.150% |

The corrected route therefore stops before test.  Shop `max2` keeps the total
catastrophic count at seven but swaps in two new catastrophic queries; the
Selector improves typical precision and P95 while worsening the extreme tail.
Stairs `max1/max2` have the exact same 324-query catastrophic set.  The six
extra siblings recover only 0.10 recall points and slightly worsen mean, P90,
P95, and CVaR95.  No corrected arm satisfies the required combination of no
new catastrophic query, non-regressed mean/P95/CVaR95, and non-regressed
recall.  No V1.3 test evaluation was run.

The first-order conclusion is now narrower and stronger: neither the child-cap
ordering, one-versus-two child retention, nor sibling double counting explains
the deployment tail.  The remaining failure is dominated by fold-unstable
component identity and coherent render-domain false consensus.  Further work
on this route should audit low-confidence chain bridges and pose-level minimal
sets; it should not tune the same depth/cycle thresholds or add another generic
Selector pass.  Machine evidence and artifact hashes are in
[`docs/evidence/rendered_rgb_track_support_v13.json`](evidence/rendered_rgb_track_support_v13.json).

## V1.4: full-mapping feedback without formal cross-fit

V1.4 removes held-out folds from the formal method. All mapping cameras now
participate in Track construction, descriptor materialization, teacher
construction, and deployment feedback. Trajectory and pose-cell labels remain
only as balancing metadata: they prevent densely repeated views from receiving
frame-count-proportional weight, but no sequence is held out from the map.

The remaining anti-self-match rule is query-local. When mapping query `q` is
localized, every affected Track descriptor is rebuilt after removing all
observations from `q`:

```text
Track identity and xyz: all mapping observations
Track descriptor for mapping query q: Fuse(observations whose query != q)
```

The implementation first proves that fusing the complete observation set with
the same robust Track fusion routine reproduces the frozen map descriptor bank
bitwise. It then updates only the affected rows for each query before the same
single global Top-1 match and single PoseLib call. This is not cross-validation
and does not test generalization; it only prevents a query descriptor from
matching a map descriptor that contains itself.

The formal candidate is the full repaired-child universe selected by the
parent/sibling-aware sufficiency selector: 5,788 anchors on ShopFacade and
5,811 on Stairs. It does not use the former fixed `max1`/`max2` child maps. The
metric is frozen identity because the old A1 teacher did not implement the same
query-local exclusion and would reintroduce self-match leakage. Original
mapping RGB and test queries remain absent from construction and selection.

### Full-mapping leave-one-query-observation-out audit

| Scene | Mapping queries | Median TE cm | Mean TE cm | P90 / P95 TE cm | 5cm recall | Catastrophic >=1m |
|---|---:|---:|---:|---:|---:|---:|
| ShopFacade | 231 | 0.342 | 45.272 | 0.741 / 1.099 | 99.134% | 2 |
| Stairs | 2,000 | 0.363 | 6.103 | 0.960 / 1.499 | 96.600% | 22 |

The large ShopFacade mean is caused by two extreme failures: the median, P90,
P95, and 229/231-query 5cm behavior remain sub-centimeter/centimeter scale.
Stairs shows the same separation more mildly. For this reason V1.4 reports
typical accuracy and tail counts together instead of treating mean error alone
as the scene's localization scale.

### Frozen real-test results

The candidate map, metric, calibration, and implementation commit
`a2d4caed3ddd995d40fc2130c232f424933f924e` were frozen before test. Test RGB
was used only for the three final PoseLib seeds 2026/2027/2028. Entries below
are seed means.

| Scene / route | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2cm recall | 5cm recall | Raw GT precision | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade V1.2 E+R identity | 2.434 | 5.929 | 8.393 | 0.292 | 40.777% | 79.612% | 6.686% | 1.0 |
| ShopFacade V1.4 full-child identity | **2.044** | **4.733** | **8.057** | **0.225** | **49.515%** | **82.201%** | **9.752%** | **0.0** |
| Stairs V1.2 E+R A1 | 2.458 | 12.430 | 16.606 | 4.269 | 39.233% | 78.600% | 2.756% | 24.667 |
| Stairs V1.4 full-child identity | **2.273** | **10.926** | **9.335** | **3.142** | **42.800%** | **82.967%** | **2.954%** | **15.667** |

The full-child, sibling-aware R0 candidate improves every listed accuracy
metric on both scenes. Relative to V1.2, ShopFacade mean TE improves by
1.196 cm, 2cm/5cm recall by 8.738/2.589 percentage points, and catastrophic
count from one to zero. Stairs mean/P90 improve by 1.503/7.271 cm,
2cm/5cm recall by 3.567/4.367 points, and catastrophic count by nine per seed.

V1.4 is therefore promoted as the current **source-image-free R0 experimental
baseline**. It does not replace the mixed shared mainline: Stairs still trails
the mixed map most clearly in mean/tail error and catastrophic count. The next
single factor is the preregistered raw/clean render artifact-stability evidence,
followed only if useful by artifact-aware KCS support and unified GWFF-style
observation fusion. Formal cross-fit, fixed maximum-child maps, and the old
self-matched A1 are retired from this route. Exact paths, hashes, mapping-only
reports, test summaries, and deltas are in
[`docs/evidence/rendered_rgb_track_fullmap_v14.json`](evidence/rendered_rgb_track_fullmap_v14.json).

## R1: raw/clean 2DGS artifact stability

R1 was implemented as the next single factor without changing V1.4 Track
membership, xyz, selector rows, or map size. A strict cache audit proves that
all localization query rows and rendered geometry samples remain bitwise
exact; only observation reliability and six artifact annotations change.

ShopFacade passed its mapping-only gate and reduced CVaR95 from 864.87 cm to
164.75 cm. Stairs improved P90 from 0.960 cm to 0.932 cm and kept the same 22
catastrophic queries, but CVaR95 worsened from 114.27 cm to 141.14 cm and raw
precision fell by 0.0856 percentage points. The formal dual-scene gate is
therefore `STOP_R1_BEFORE_TEST_AND_R2`. No R1 test query was evaluated, and
the artifact scalar is not promoted into Track identity, KCS/GWFF, A1, or the
shared default. Full attribution, artifact statistics, exact SHA lineage, and
the tail diagnosis are recorded in
[`docs/rendered_rgb_track_artifact_stability_result.md`](rendered_rgb_track_artifact_stability_result.md)
and
[`docs/evidence/rendered_rgb_track_artifact_stability_result.json`](evidence/rendered_rgb_track_artifact_stability_result.json).

## Conditional fusion, LOO-A1, and completion convergence

The remaining authorized method enhancements were implemented without creating
another scientific gate: conditional artifact-aware observation fusion, a true
leave-one-query-observation-out A1 refresh, and a mapping-only full broad-Track
completion oracle. Conditional fusion helped ShopFacade tail recall but caused
a large Stairs P90 regression; LOO-A1 did not remove the Stairs tail; and the
completion oracle exposed capacity headroom without reducing catastrophic
failures. V1.4 therefore remains the source-image-free baseline, and the
remaining Stairs error is attributed to structured false Track/pose consensus.
See
[`docs/rendered_track_conditional_loo_completion_result.md`](rendered_track_conditional_loo_completion_result.md)
for exact results and artifact lineage.
