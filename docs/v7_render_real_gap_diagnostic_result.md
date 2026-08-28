# V7 P0.5 render--real causal diagnostic result

## Conclusion

The render--real difference is not a fixed pose or map-geometry error. It is a
mixture of two upstream correspondence mechanisms:

1. real RGB contains low-consensus content whose Top-1 matches are overwhelmingly
   wrong; and
2. even on mutually visible shared structure, real and rendered SuperPoint
   descriptors rank the 200,255 Anchors differently often enough to leave a
   large matching gap.

The existing dataset mask does not explain the renderer's advantage. A hard
render-support row filter identifies dirty rows but does not materially improve
pose, so it must not be promoted directly into the deployed method.

## Protocol integrity

- All 530 `real + dataset mask` results reproduce the frozen reference pose,
  keypoint count, and inlier count exactly.
- The diagnostic is post-hoc, non-formal, and transductive. It reads test RGB
  and test poses for diagnosis only.
- It selects no map or threshold, authorizes no feedback, and performs zero map
  mutations.
- Full-map, metric, render records, configuration, and shard outputs are
  hash-bound.

## Geometry and fixed-bias test

Using real keypoints but GT-projection/depth-visible, Anchor-unique oracle
correspondences, the unchanged PoseLib solver reaches 0.755cm median TE,
1.685cm P90 TE, and 96.792% R5. The median oracle correspondence count is 737.
This passes the preregistered geometry-sufficiency gate.

For successful real queries, the signed camera-frame translation-error direction
has resultant 0.058 and robust-bias ratio 0.061. The errors do not share a fixed
direction. Rendered queries have a small approximately -0.208cm camera-depth
bias, far below the real median error.

## Existing-mask 2x2 control

| RGB / mask condition | Median TE | P90 TE | R5 |
|---|---:|---:|---:|
| Real, deployed dataset mask | 4.021cm | 12.620cm | 61.887% |
| Real, unmasked | 4.105cm | 12.216cm | 62.453% |
| Render, unmasked | 0.492cm | 1.249cm | 97.358% |
| Render, same dataset mask | 0.532cm | 1.402cm | 96.604% |

The existing mask changes real R5 by -0.566 points and render R5 by -0.755
points. It neither creates nor explains the renderer's advantage.

## Content-support partition

The render-derived proxy uses alpha, positive depth, depth continuity, image
border, and V2 RGB-structure support. The unavailable full distortion raster is
not silently reconstructed, so this is explicitly a V2 proxy.

| Top-1 rows | Correct / total | GT@4px |
|---|---:|---:|
| Real inside shared support | 162,662 / 679,329 | 23.945% |
| Real outside shared support | 14,423 / 390,649 | 3.692% |
| Render inside shared support | 587,263 / 882,813 | 66.522% |
| Render outside shared support | 97,350 / 202,627 | 48.044% |

Real inside/outside precision ratio is 6.49, passing the preregistered content
contamination rule. Nevertheless replaying PoseLib after hard-removing outside
rows changes median TE only from 4.021cm to 3.955cm, leaves R5 exactly 61.887%,
worsens P90 to 13.719cm, and increases catastrophics from 11 to 12. RANSAC had
already rejected many dirty rows, while hard filtering also removes useful
spatial support.

## Shared-content descriptor test

Across all queries, 98,239 mutually nearest real/render keypoint pairs lie
within 2px and shared support. This is only 9.18% of real and 9.05% of render
keypoints, showing low detector repeatability. Within those deliberately stable
pairs:

- mean real/render descriptor cosine is 0.814;
- only 53.44% select the same Top-1 Anchor;
- mean Top-1/Top-2 margin is 0.0456 for real versus 0.0618 for render;
- GT@4px is 63.40% for real versus 79.79% for render.

Thus descriptors at repeatable locations are related, but small changes are
amplified by a globally ambiguous Anchor bank: winner identity and margin remain
substantially less stable on real RGB.

## Symmetric content intervention

On the preregistered every-fourth-query subset (133 queries), both hybrids use
the same feathered support and deployed dataset mask:

- replacing real outside-support content with render improves median TE from
  4.394cm to 3.495cm and R5 from 59.398% to 67.669%;
- adding real outside-support content to a render-supported interior worsens
  median TE from 0.537cm to 1.160cm and R5 from 96.241% to 93.985%.

This detector-level intervention supports a causal content contribution. It is
not a deployable solution: cross-domain seams and inconsistent scene content
create interactions, especially on foliage-heavy tail views. Its direction and
robust median/R5 effects are evidence, not an operating point.

## Method consequence

The next method should not be another rendered-feedback descriptor update and
should not simply feed the current V2 mask into PoseLib. The evidence instead
calls for a leakage-safe real-query reliability mechanism that jointly handles
detector allocation, shared-content repeatability, descriptor margin, and
spatial pose support. It must be designed and selected on mapping-derived or
separate validation imagery; this test diagnostic cannot select it.

## Reproducibility artifacts

- Preregistration: `configs/v7_render_real_gap_diagnostic.yaml`
- Machine-readable report:
  `/mnt/pool/sqy/lafgs_v7_render_real_gap_p05_20260827/StMarysChurch/aggregate_v2/report.json`
- Fixed visual audit:
  `/mnt/pool/sqy/lafgs_v7_render_real_gap_p05_20260827/StMarysChurch/aggregate_v2/visual_audit.png`
- GPU shard manifests:
  `/mnt/pool/sqy/lafgs_v7_render_real_gap_p05_20260827/StMarysChurch/shard0_gpu1/manifest.json`
  and
  `/mnt/pool/sqy/lafgs_v7_render_real_gap_p05_20260827/StMarysChurch/shard1_gpu2/manifest.json`.
