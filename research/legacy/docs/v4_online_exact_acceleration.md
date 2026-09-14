# V4 exact online acceleration

## Scope

This change accelerates the frozen single-stage runtime without changing the
keypoint budget, descriptor precision, global search, correspondence count, or
PoseLib contract.  It contains no ANN, reduced precision, adaptive pruning, or
RANSAC parameter change.

The compact map is normalized once into its final matching bank when
`SparseLocalizer` is constructed.  Exact Top-1 now retains only the best row of
each score chunk and updates the running result directly.  Exact Top-2 retains
two rows per chunk and merges four candidates, rather than concatenating every
score in the chunk.  The default independent global Top-1 path is unchanged.

Pinned reusable 2D/3D host buffers replace two allocating synchronous copies.
Camera intrinsics and the immutable PoseLib camera payload are cached per
calibration.  `profile_mode=True` preserves explicit stage synchronization;
`--deployment-mode` removes intermediate synchronization, uses CUDA events for
stage accounting, and synchronizes only when PoseLib needs CPU correspondences.

## Tie contract

The old `torch.topk` implementation had no stable equal-score contract.  Its
winner changes with CPU versus CUDA and with chunk sizes 1, 2, 3, 4, or 8 for
duplicate descriptors, zero descriptors, and signed zero.  The accelerated
kernel defines the missing contract explicitly:

1. higher cosine score first;
2. equal score: lower compact Anchor row first.

This is a contract clarification, not a claim of bitwise equivalence for
synthetic exact ties.  Random non-tie oracle tests are byte-exact.  Real
ShopFacade and GreatCourt artifact replay found no changed ID or score, and
their PoseLib pose matrices are exactly equal.

## Real artifact benchmark

CUDA_VISIBLE_DEVICES selected an otherwise idle RTX 3090.  Each map replay used
its own rendered feature cache, 2,048 descriptors per query, one warm-up query,
and three measured real test images for the full pipeline.  The standalone
matcher used the same artifact descriptors.  Evidence is stored in
`docs/evidence/v4_online_exact_acceleration_{shopfacade,greatcourt}.json`.

| scene | Anchors | old matcher mean | exact matcher mean | speedup | IDs/scores/poses |
|---|---:|---:|---:|---:|---|
| ShopFacade | 5,794 | 1.268 ms | 0.537 ms | 2.36x | exact |
| GreatCourt | 18,279 | 3.489 ms | 1.288 ms | 2.71x | exact |

The three-image deployment samples measured 84.7 ms/query on ShopFacade and
187.0 ms/query on GreatCourt.  These small samples are implementation checks,
not replacement dataset runtime statistics.  The theoretical two-stage stream
boundary `max(frontend + matching, PoseLib)` was 56.5 ms (17.7 FPS) and 140.3
ms (7.1 FPS), respectively.  The evaluator does not yet execute that pipeline,
so these numbers are explicitly bounds rather than measured stream throughput.

The result also confirms the remaining bottleneck: GreatCourt exact matching
is only 1.29 ms, while standard PoseLib averages 140.3 ms in this sample.
Further large gains on hard scenes require either actual GPU/CPU frame
double-buffering or higher correspondence inlier quality; neither is silently
introduced here.

## Reproduction

Run `python -m scripts.benchmark_online_exact_acceleration --help`.  The script
contains the pre-change matcher as an oracle, fails closed if IDs, scores, or
pose matrices differ, and writes a lineage-friendly JSON result.
