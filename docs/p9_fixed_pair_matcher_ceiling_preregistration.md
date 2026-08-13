# P9 fixed-pair matcher-ceiling preregistration

P9 isolates one question: on the exact frozen nearest camera pairs, does the
bundled XFeat LighterGlue matcher improve correspondence identity over mutual
nearest-neighbour cosine matching when both arms consume the exact same fresh
XFeat rows?  This is a mapping-only Pair Gate.  It does not build Tracks, run a
pose solver, consult test queries, or change the default method.

The machine-readable authority is
`docs/evidence/p9_fixed_pair_matcher_ceiling_preregistration.json`.  Every value
in that file was measured locally before implementation; it contains no inferred
or abbreviated digest.

## Causal boundary

Both arms use extractor E1 from the single bundled
`xfeat-lighterglue.pt` checkpoint.  The checkpoint is exactly 291 tensor keys:
122 keys under `extractor.model.net.` and 169 keys under `matcher.`.  The
extractor prefix is stripped and loaded with `strict=True`.  LighterGlue is
instantiated from the fixed 64-to-96, six-layer, one-head configuration.  Its
169 stripped checkpoint tensors plus the one audited runtime-generated
`confidence_thresholds` buffer must exactly equal the 170-key instantiated
state before another `strict=True` load.  `strict=False`, the unrelated
69-entry `xfeat.pt`, and the high-level `match_xfeat` wrappers are forbidden.

The fresh feature cache recreates masked native mapping RGB and runs exactly one
extractor forward per image.  Detection is fixed at probability `>0.05`, a
single 5x5 NMS, stable score order with row-major tie retention, and scene K
before the valid-mask filter with no refill.  It stores raw XFeat and scaled
native coordinates, normalized 64D descriptors, detector scores, exact row
hashes, dense native depth/alpha resampled onto the declared native grid,
intrinsics, poses, masks, and source-image lineage.  GreatCourt's dimensions
need not be divisible by 32: raw-to-native scaling is explicit and is tested.

The pair table comes only from the attested frozen nearest proposal arm:
7,450 Stairs pairs and 5,254 GreatCourt pairs.  No P8 match row, old XFeat Arm-A
or Arm-B probe, or cached pair-conditioned result is accepted.  During the pair
stage the detector is unavailable by construction.  The control is MNN cosine
with `min_cossim=-1`; the variant is one direct LighterGlue forward per pair.
Both arms then use the same symmetric epipolar threshold of 2 px and the same
confidence `sqrt(left_detector_score * right_detector_score)`.

## Mapping-only depth teacher

Correctness uses only the frozen mapping camera geometry and the dense rendered
depth/alpha fields carried into the fresh feature cache.  Each endpoint samples
positive finite depth at nearest native pixels, requires alpha at least 0.2,
uses the frozen `+0.5` pixel-centre convention, and reprojects through the other
mapping camera.  A row is correct only if both depth-based reprojection errors
are at most 2 px.  Unevaluable rows do not enter the precision denominator.

## Pair Gate

For each scene, LighterGlue must not reduce any of correct count, correct
precision, epipolar-accepted count/rate, exact verified keypoint triangles,
cycle-supported correspondence edges, verified-triangle camera set, or nonempty
pair coverage.  It may not increase graph identity conflicts.  At least one of
correct count, verified triangles, or cycle-supported edges must improve
strictly.  Exact-cycle triangles require the same feature-row identity around
all three fixed pair edges and at most 2 px three-view reprojection error.

A scientific failure writes a valid Stop gate and exits 2.  Invalid lineage or
input writes no gate and exits 1.  A per-scene pass only says that the other
scene is still required.  Even two passes authorize only review of a future
Track implementation; this preregistration never authorizes a real Track run,
pose run, test split, or default change.

## Artifact and review boundary

Feature cache, paired probe, completion marker, and comparator gate use distinct
P9 schemas and content hashes.  Outputs must be fresh and are committed by an
atomic same-directory rename; the completion marker is written last only after
the feature cache and both arms reload and validate.  Spliced, partial, resumed,
or one-arm outputs are invalid.

The preregistration is committed first.  Implementation and CPU tests are a
separate commit.  A later independent review and full CPU suite must sign a
separate implementation registry before either real mapping scene may run.
