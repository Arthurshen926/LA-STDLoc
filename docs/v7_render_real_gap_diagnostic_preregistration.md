# V7 render--real gap causal diagnostic preregistration

This P0.5 diagnostic follows the completed Situation-B gate. It is post-hoc,
non-formal, and transductive: it reads real test RGB and exact test poses only
to distinguish mechanisms. It cannot select or mutate a map, tune a threshold,
or authorize feedback. Any deployable method must later be developed and
selected without this test split.

The term `render_real_gap` is observational rather than causal. Four hypotheses
are separated before results are read:

1. **Existing-mask effect.** The deployed real evaluation consumes the dataset
   object/sky/distortion mask, whereas the test-pose render diagnostic is
   unmasked. A fixed 2x2 comparison evaluates real/render RGB with and without
   the same dataset mask.
2. **Content contamination.** Real Top-1 rows are partitioned by a support mask
   computed only from the same-pose render's alpha, depth continuity, border,
   and V2 RGB-structure evidence. The detector is unchanged. Standard PoseLib
   is replayed on the supported rows.
3. **Shared-content descriptor mismatch.** Mutually nearest real/render
   keypoints within 2px and inside shared support compare descriptor cosine,
   Top-1 Anchor identity, Top-1/Top-2 margin, and GT@4px correctness.
4. **Geometry ceiling.** Real keypoints are paired offline with unique visible
   Anchors selected by GT projection and render-depth agreement, then replayed
   through the same PoseLib solver. This oracle is never an online input.

A symmetric content intervention uses a feathered, render-derived support mask:

- real inside shared support plus render outside;
- render inside shared support plus real outside.

Both hybrids receive the same existing dataset mask. The feather radius and all
support, pairing, geometry, and decision thresholds are frozen in
`configs/v7_render_real_gap_diagnostic.yaml`. Hybrids use every fourth query;
all non-hybrid diagnostics use all 530 queries. Shards are assigned by global
query index modulo shard count, so no result-dependent sample selection occurs.

The diagnostic first requires `real + dataset mask` to exactly reproduce every
non-timing reference record. Failure invalidates all causal comparisons.
