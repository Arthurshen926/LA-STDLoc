# V8 V2 pre-association full rebuild result

## Corrected question

The earlier 34,437-Anchor strict map was a post-hoc subset of an already built
Full map.  It did not rerun pairing or Track formation after invalid
observations were removed.  Its coverage loss therefore cannot establish that
V2 evidence is ineffective during mapping.

This experiment tests the requested causal intervention: native SuperPoint
first observes the complete unmasked Gaussian RGB; V2 then removes invalid
detector rows **before** camera-pair association.  Pair matching, Track
components, triangulation, completion and descriptor fusion are all recomputed
from scratch.  The query detector and closed-loop feedback are absent.

## Construction audit

| Quantity | Original Full | V2 full rebuild |
|---|---:|---:|
| Mapping views | 1,487 | 1,487 |
| Detector rows admitted | 3,045,376 | 2,525,043 |
| Matched camera pairs | 5,080 | 5,071 |
| Raw reciprocal/epipolar edges | 4,777,167 | 4,053,897 |
| Track components | 200,278 | 164,562 |
| Track observations | 2,387,189 | 2,016,662 |
| Base Anchors | 199,705 | 164,160 |
| Completion Anchors | 550 | 711 |
| Final Anchors | 200,255 | 164,871 |

V2 removes 520,333 detector rows (17.09%), but the fresh map retains 82.33% of
the Full Anchors.  This differs sharply from the post-hoc strict map, which
retained only 34,437 Anchors (17.20%).  The changed edge, Track and completion
counts prove that the result is a fresh reconstruction rather than a filtered
copy of the old map.

The final map contains 2,017,016 Anchor observations.  An exact audit against
the immutable source-cache row registry found zero V2-invalid observations.
Every Anchor has at least three observations; geometry and descriptors are
finite and descriptor norms remain unit length.

## Frozen 63-render confirmation

Both variants use native SuperPoint, exact global Top-1 matching and one
standard PoseLib call.  Selector and localization parameters are unchanged.

| Map | Median TE | P90 TE | R5 | Catastrophic >=50 cm | Mean TE |
|---|---:|---:|---:|---:|---:|
| Original Full | 0.332 cm | 0.892 cm | 98.41% | 1 | 34.54 cm |
| V2 full rebuild | 0.326 cm | 1.137 cm | 98.41% | 1 | 21.69 cm |

The aggregate median improves slightly and the single catastrophic error falls
from 21.51 m to 13.37 m, but P90 regresses.  Paired results are 27 improved and
36 worsened queries; paired median delta is +0.010 cm with bootstrap 95%
interval [-0.002, +0.035] cm.  Render-domain central improvement is therefore
not established.

## Frozen non-test real-RGB panel

The panel consists of 128 mapping-camera real images fixed before this variant
was evaluated.  They are used only for evaluation: no real RGB descriptor is
written to the map and no result tunes V2 thresholds.

| Map | Median TE | P90 TE | R5 | Catastrophic >=50 cm | Mean TE | Median AE |
|---|---:|---:|---:|---:|---:|---:|
| Original Full | 4.311 cm | 27.019 cm | 58.59% | 10 | 587.70 cm | 0.120 deg |
| V2 full rebuild | 3.658 cm | 27.214 cm | 60.94% | 10 | 365.84 cm | 0.118 deg |

V2 improves median TE by 0.654 cm (15.2%) and R5 by 2.34 percentage points.
At query level, 68 improve and 60 worsen; six R5 failures are recovered and
three successes are lost, for a net gain of three.  Paired median TE delta is
-0.064 cm and its bootstrap 95% interval [-0.168, +0.059] cm crosses zero.
P90 and catastrophic count do not improve, although several very large errors
become substantially smaller.

## Decision

The corrected ablation supplies **positive evidence** for inserting V2 into
map construction.  In particular, it overturns the earlier inference that a
clean-map path necessarily destroys most useful coverage.  The positive result
comes from allowing clean observations to relink into new Tracks and to be
retriangulated, not from deleting old Anchors after the fact.

The result is promoted as the accepted mainline M0 and replaces both the
unfiltered Full initialization and the post-hoc strict-map path.  Subsequent
feedback actions must start from this artifact and pass fresh confirmation;
they may not use the old Full map as an alternate starting point.

The query scene detector remains out of this experiment.  Its supervision
should be regenerated only after the map-side feedback actions have accepted
or rejected Anchors, so that it learns from the final accepted Anchor set rather
than the provisional V2-only map.  The real test remains sealed.

## Artifacts

- Contract: `configs/v8_v2_full_rebuild.yaml`
- Builder: `scripts/materialize_v8_v2_projective_map.py`
- Map: `/mnt/pool/sqy/lafgs_v8_v2_full_rebuild_20260828/StMarysChurch/projective_map/projective_anchor_map.pt`
- Build report: `/mnt/pool/sqy/lafgs_v8_v2_full_rebuild_20260828/StMarysChurch/projective_map/report.json`
- Render confirmation: `/mnt/pool/sqy/lafgs_v8_v2_full_rebuild_20260828/StMarysChurch/confirmation_v2_rebuild.json`
- Real-RGB panel: `/mnt/pool/sqy/lafgs_v8_v2_full_rebuild_20260828/StMarysChurch/real_mapping_panel.json`
