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

The route is technically viable but not a universal replacement for the mixed
mainline yet.

- ShopFacade is the positive result.  A 2,521-anchor map reconstructed only
  from rendered RGB approaches the mixed mainline and even has a slightly
  better P90 translation error.  The standard selector performs most of the
  useful work; A1 adds only about 0.97 percentage points of 2cm recall.
- Stairs is the negative result.  Its map is geometrically complete, but raw
  real-test correspondence precision collapses from 3.62% on the mixed
  mainline to 0.11%.  Selection and A1 cannot repair that render-to-real
  appearance gap; A1 trades better median/P90 and 5cm recall for worse mean
  tail and more catastrophic failures.
- The evidence therefore separates geometry from appearance: rendered RGB can
  reconstruct high-quality Track geometry, but the deployable descriptor bank
  depends strongly on how faithfully the Gaussian prior reproduces the real
  image domain.

The reasonable next experiment is not to reintroduce Gaussian anchor geometry.
It is a mapping-only render-domain robustness step: multiple deterministic
appearance-perturbed renders at the same mapping poses, shared Track geometry,
and a descriptor consistency objective.  That should first be validated on
Stairs and retained on ShopFacade before any broader benchmark.  No test query
should participate in choosing those perturbations or training the map.

