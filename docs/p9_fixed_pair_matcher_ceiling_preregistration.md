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
need not be divisible by 32: its 1080x1920 native input is resized to 1056x1920
for E1, and raw coordinates return through `(x*1920/1920,y*1080/1056)`.

More exactly, E1 applies a 65-way softmax, drops the dustbin and unpacks each
8x8 cell before one 5x5 max-pool.  Candidate rows satisfy equality with the
local maximum and strict probability `>0.05`.  `torch.nonzero` establishes
row-major ties; score is locked nearest-interpolated probability times locked
bilinear-interpolated reliability.  The raw `(0,0)` padding sentinel receives
score -1.  Stable descending score order is cut to scene K, positive scores are
kept, and only then the native mask is indexed with PyTorch round-to-even; a
floor lookup is forbidden and a rejected row is never refilled.  The normalized
float32 dense 64D field is sampled bicubically at raw XFeat coordinates and each
float32 row is normalized again.  Per-tensor hashes include dtype, shape, and
contiguous CPU bytes; the row-registry hash is the frozen canonical JSON over
the four tensor hashes and row count.

The pair table comes only from the attested frozen nearest proposal arm:
7,450 Stairs pairs and 5,254 GreatCourt pairs.  No P8 match row, old XFeat Arm-A
or Arm-B probe, or cached pair-conditioned result is accepted.  During the pair
stage the detector is unavailable by construction.  The control is MNN cosine
with `min_cossim=-1`; the variant is one direct LighterGlue forward per pair.
Both arms then use the same symmetric epipolar threshold of 2 px and the same
confidence `sqrt(left_detector_score * right_detector_score)`.

LighterGlue receives `native_xy` without `+0.5`, the exact shared E1 descriptor
rows, and float32 native `[width,height]` image size.  It receives no detector
score.  Only geometry and the depth teacher add `+0.5`.  This distinction is
part of the machine contract rather than an implementation convention.

## Mapping-only depth teacher

Correctness uses only the frozen mapping camera geometry and the dense rendered
depth/alpha fields carried into the fresh feature cache.  Each endpoint samples
positive finite depth at nearest native pixels, requires alpha at least 0.2,
uses the frozen `+0.5` pixel-centre convention, and reprojects through the other
mapping camera.  A row is correct only if both depth-based reprojection errors
are at most 2 px.  Unevaluable rows do not enter the precision denominator.
Correctness is evaluated after the shared epipolar filter, never on raw matcher
output.  When the raw or evaluable denominator is zero, the respective rate or
precision is exactly 0.0.

The count fields are authoritative and the serialized rate/precision fields
are only checked projections.  Every validator recomputes both projections
from nonnegative integer counts and requires
`epipolar_accepted_count <= raw_match_count` and
`correct_correspondence_count <= teacher_evaluable_count <=
epipolar_accepted_count`.  A zero denominator therefore also requires a zero
numerator and is represented by the exact rational `0/1`.  Pair-Gate
non-regression never compares serialized floats: it compares the recomputed
rationals with Python-integer cross multiplication, so equality at a binary
floating-point boundary cannot change a decision.

## Pair Gate

For each scene, LighterGlue must not reduce any of correct count, correct
precision, epipolar-accepted count/rate, exact verified keypoint triangles,
cycle-supported correspondence edges, verified-triangle camera set, or nonempty
pair coverage.  It may not increase graph identity conflicts.  At least one of
correct count, verified triangles, or cycle-supported edges must improve
strictly.  Exact-cycle triangles require the same feature-row identity around
all three fixed pair edges and at most 2 px three-view reprojection error.

The symmetric epipolar error is the maximum of the two point-to-line distances
from `F=K1^-T [t10]_x R10 K0^-1`, with each line norm clamped below at `1e-12`;
acceptance is finite and `<=2.0`.  Camera triples are lexicographically
enumerated from the fixed graph, closed by exact row maps, triangulated by a
six-row homogeneous DLT SVD, and require positive depth in all three views.
Cycle edges are unique `(fixed_pair_index,source_row,target_row)` tuples.
Coverage counts fixed pairs with at least one epipolar-accepted row.  Identity
conflicts union those accepted rows in fixed-pair/source-row order and count,
per connected component and camera, distinct rows beyond the first.

A scientific failure writes a valid Stop gate and exits 2.  Invalid lineage or
input writes no gate and exits 1.  A per-scene pass only says that the other
scene is still required.  Even two passes authorize only review of a future
Track implementation; this preregistration never authorizes a real Track run,
pose run, test split, or default change.

The formal scene order is Stairs then GreatCourt.  A Stairs Stop ends P9 before
any GreatCourt feature-cache/probe build.  GreatCourt requires the exact path
and SHA-256 of a valid passing Stairs Pair Gate before large I/O and embeds that
parent in its own gate.  A separate cross-scene gate then requires both exact
parents plus identical compiled source and runtime/backend identities.  Its
only pass decision is `GO_TO_FIXED_PAIR_TRACK_IMPLEMENTATION_REVIEW`; it still
does not authorize a real Track run.

Before any GreatCourt query-cache/image/checkpoint loading or model forward,
the feature CLI constructs its current producer identity and requires its
compiled identity to equal the already validated Stairs parent identity.

## Frozen mapping image inputs and runtime

The two source-image manifests are frozen beside the preregistration.  They
hash every mapping row as `UTF-8 name + TAB + lowercase per-file SHA-256 + LF`
in exact query-cache/COLMAP mapping order.  Stairs binds 2,000 images,
793,635,630 bytes and digest
`ff39e966b651a3c265cc59200b5cf8f319be4777a5594a7ca0a4ab98f89cd4f9`;
GreatCourt binds 1,531 images, 3,864,188,235 bytes and digest
`21261bb646419272241b08a761a9480e0180f06e9b83a7ea90fb47d1f4a15f67`.
Both mapping/test-name intersections are zero.  Split and mask artifacts are
also SHA-bound in those manifests.

Formal execution uses the exact cybersim Python executable on CPU with CUDA
hidden, float32 extractor/matcher, eval plus inference mode, no autocast,
16 intra-op threads, one inter-op thread, deterministic algorithms, MKLDNN
enabled, highest float32 matmul precision, CPU scaled-dot-product self-attention
and Kornia's manual CPU cross-attention.  Both scene gates must report the same
runtime/backend identity.

Kornia 0.8.2 is runtime-imported only; no Kornia source or weight is vendored.
Its LightGlue source and installed license are independently SHA-bound.  The
recursive local producer closure includes COLMAP parsing as well as dataset,
image, hashing, feature, geometry, all three producers/comparators, the cross
aggregator, preregistration, and both mapping-image manifests.

## Artifact and review boundary

Feature cache, paired probe, completion marker, and comparator gate use distinct
P9 schemas and content hashes.  Outputs must be fresh and are committed by an
atomic same-directory rename; the completion marker is written last only after
the feature cache and both arms reload and validate.  Spliced, partial, resumed,
or one-arm outputs are invalid.

Tensor shapes are exact and are checked before any reshape or squeeze.  In
particular detector scores are `[N]`, depth/alpha/masks are `[H,W]`, pair
indices and every match column are one-dimensional, and each diagnostic is
`[P]`; singleton-expanded forms such as `[N,1]` and `[1,H,W]` are invalid.
The completion marker must contain `failure_recovery` with the exact value
`isolate_root_and_rebuild_both_arms_from_fresh_cache`; missing, null, or edited
values invalidate the completion before artifact loading.

The machine contract enumerates every required top-level/query/arm key, tensor
shape and dtype, and content-hash domain.  The fixed files are
`p9_fixed_pair_feature_cache.pt`, `fixed_pair_match_probe.pt`, its JSON summary,
and `paired_match_completion.json`.  The paired root is assembled in a sibling
temporary directory and renamed only after full reload; the completion is the
only Pair-Gate input, so a direct probe path or a partial arm cannot advance.
Feature cache, both arms, completion, scene gate, and cross gate bind a common
compiled identity.  That identity includes reviewed implementation/preregistered
commits, every recursive source hash, exact runtime/dtype/thread/backend fields,
and a clean-path guard.

The preregistration is committed first.  Implementation and CPU tests are a
separate commit.  A later independent review and full CPU suite must sign a
separate implementation registry before either real mapping scene may run.
