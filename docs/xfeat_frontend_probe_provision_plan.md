# Local XFeat frontend ceiling-probe provision plan

## Decision

The frontend ceiling probe is no longer blocked by the absence of an
independent checkpoint. A bounded, read-only local inventory found one
admissible candidate: the 64D sparse XFeat extractor contained in a clean
historical ULF-Loc worktree. This is an artifact-availability result, not an
accuracy result. No mapping image, test image, GPU, or network was used.

The candidate is:

- weights:
  `/mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean/encoders/XFeat/weights/xfeat.pt`;
- weight SHA256:
  `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`;
- weight size: 6,247,949 bytes;
- parent ULF-Loc commit:
  `b28d53258ab4461ba1a02eaa60ef504e9b82b9ab` (`release code`);
- exact XFeat Git tree:
  `4f804566cb1cf72469b7d7174fba9308885c5c5a`;
- descriptor dimension: 64; and
- code-tree license: Apache-2.0, evidenced by the local `LICENSE` file.

The whole `encoders/XFeat` path is tracked and clean at that commit. Three
historical worktree copies have the same weight SHA. An offline CPU check
strictly loaded all 122 state-dict entries (1,551,625 parameters), with no
missing or unexpected keys, and produced finite 64-channel descriptors plus
detector/reliability outputs on a synthetic 32x32 tensor. The external
worktree was not modified.

## Why XFeat is admissible

`xfeat.pt` drives a single-image model that returns a dense stride-8 64D local
descriptor field, keypoint logits, and a reliability map. Its sparse API
detects, scores, describes, L2-normalizes, and returns top-K keypoints without
conditioning on another image. The descriptors can therefore populate an
independent map bank and be queried by global cosine similarity.

This conclusion applies only to `xfeat.pt`. The adjacent
`xfeat-lighterglue.pt` is a pair-matcher component and is rejected. The
candidate producer must not call XFeat's LighterGlue, semi-dense pair-match,
mutual-match, or refinement APIs.

The repository's existing probe consumer can directly score native 64D
candidate descriptors. It already requires one positive-dimensional, finite,
nonzero descriptor per exact frozen SuperPoint row, normalizes both candidate
query descriptors and candidate anchor banks, and ranks with global cosine.
No consumer or scientific-algorithm change is needed. What is missing is only
an offline producer that converts the locked XFeat output into
`lafgs_frontend_ceiling_probe_cache`, version 1.

## Locked producer contract

The producer must remain mapping-only and start from the frozen query-cache
signature, complete-positive teacher, and the original mapping RGB source
named by the query-cache signature. The query cache does not contain RGB and
must not be treated as an approximate image source.

For every query, reconstruct the exact input used by the frozen sparse
frontend:

1. load the original RGB as float RGB in `[0,1]`;
2. recreate the object/sky/distortion conjunction at the original image size;
3. resize RGB to `native_input_hw` using bilinear interpolation with
   `align_corners=false`;
4. resize the valid mask using nearest-neighbor interpolation;
5. multiply resized RGB by that resized mask before XFeat; and
6. require the result height/width to equal the cached `native_input_hw`.

Preserve XFeat's locked internal preprocessing, but never pretend its resized
coordinates are already native. For cached native size `(Hn, Wn)`, the audited
wrapper computes

```text
Hx = floor(Hn / 32) * 32       Wx = floor(Wn / 32) * 32
rh = Hn / Hx                  rw = Wn / Wx
```

and bilinearly resizes the already masked native RGB to `(Hx, Wx)` with
`align_corners=false`. Reject dimensions below 32. Store both sizes and both
ratios for every query. Detector grid index `(xx, yx)` is returned to the
cached native grid as `(rw*xx, rh*yx)`. Conversely, fixed SuperPoint native row
`(xn, yn)` is sampled at XFeat input coordinate `(xn/rw, yn/rh)`. This simple
ratio is the locked wrapper convention; do not substitute a half-pixel formula
even though PyTorch's resize itself uses `align_corners=false`.

Probe keypoints remain in the cached native **grid-index** convention; the
consumer adds the cached `pixel_center_offset` (normally 0.5) only when
comparing to physical projected pixels. For Arm B, feed the transformed XFeat
coordinate to the locked `InterpolateSparse2d('bicubic')`. Given `(Hx, Wx)`,
that code maps `(xx, yx)` to
`(2*xx/(Wx-1)-1, 2*yx/(Hx-1)-1)` and calls PyTorch `grid_sample` with
`mode=bicubic, align_corners=false` on the normalized stride-8 field. Although
that normalization/formula pairing is unusual, changing it would no longer be
the audited implementation. Validate both directions and the exact sampler
with corner, center, and random synthetic coordinates before materializing a
scene.

A CPU-only precheck exercised this exact contract at deliberately
non-divisible native size `481x641`: it produced XFeat size `480x640`, achieved
zero float32 round-trip error for four corner/center/interior coordinates, and
returned finite `[1,4,64]` bicubic samples. This verifies the transform and
consumer shape contract, not real-image accuracy.

Every probe must bind all of:

- weight path, size, SHA256, and tracked Git blob;
- parent repository commit and XFeat tree ID;
- SHA256 of `modules/model.py`, `modules/xfeat.py`, and
  `modules/interpolator.py`;
- input RGB/preprocessing implementation ID and SHA256;
- query-cache path/signature, teacher path/schema, mapping-only/test-free flags;
- requested K, exact reference-keypoint tensor SHA for each query;
- original/native/XFeat dimensions, both resize transforms, and coordinate
  transform; and
- device/dtype used only as execution provenance, never as a changed factor.

The implementation ID should be a structured string such as
`xfeat_tree_4f804566__producer_<commit>`, not a free-form label.

## Arm B first: fixed-row descriptor identity

Arm B is the least confounded and should run first. It disables XFeat's
detector and pair matcher. For each mapping image:

1. obtain the normalized 64D dense field from the locked model;
2. transform every exact cached SuperPoint `native_keypoints` row into the
   XFeat input grid and sample it with the locked bicubic interpolator above;
3. L2-normalize each sampled row and reject any non-finite/zero row;
4. store `reference_keypoints_sha256` and
   `descriptor_at_reference_keypoints`; and
5. set `descriptor_identity=true`, `detector_repeatability=false`,
   `descriptor_dim=64`, and `requested_keypoint_count` equal to frozen K.

Command draft after the producer is reviewed (the manifest resolves and
fail-closes all artifact/code hashes above):

```bash
PYTHONPATH=/root/STDLoc:/mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean \
  python -m scripts.materialize_xfeat_frontend_probe \
  --arm descriptor \
  --images-root /locked/mapping/rgb/root \
  --query-cache /locked/stairs/query_cache.pt \
  --teacher /locked/stairs/complete_positive_teacher.pt \
  --artifact-manifest /root/STDLoc/docs/frontend_checkpoint_inventory_20260812.json \
  --candidate xfeat_sparse_64d --device cpu --dtype float32 \
  --output /locked/probes/stairs_xfeat64_descriptor.pt

PYTHONPATH=/root/STDLoc python -m scripts.audit_frontend_upper_bound evaluate \
  --state /locked/stairs/compact_anchor_map.pt \
  --query-cache /locked/stairs/query_cache.pt \
  --teacher /locked/stairs/complete_positive_teacher.pt \
  --probe-cache /locked/probes/stairs_xfeat64_descriptor.pt \
  --arm descriptor \
  --output /locked/reports/stairs_xfeat64_descriptor.json
```

The XFeat wrapper automatically chooses CUDA when visible, so the producer
must select its device explicitly and record it. This plan does not authorize a
real-data CPU or GPU run until the producer and coordinate tests are reviewed.

## Arm A second: same-K detector repeatability

Only after Arm B's producer contract is valid should Arm A be materialized.
Use XFeat's single-image sparse detector on the identical locked RGB tensor.
The detector factor is intentionally XFeat-native; do not force SuperPoint's
two-pass NMS onto it. Record the exact native semantics:

- keypoint heatmap from the 65-way cell logits;
- local maximum kernel 5 (radius 2), one pass;
- strict threshold `probability > 0.05` unless separately preregistered before
  any result is seen;
- final score = nearest-sampled keypoint probability times bilinear-sampled
  reliability, using the locked XFeat interpolator coordinate formula;
- exclude the `(0,0)` padding sentinel exactly as the locked wrapper does;
- descending top-K by that final score (stable row-major order breaks exact
  score ties), followed by the locked `(rw, rh)` transform to native grid;
- `detected_count_before_mask` means the positive-score top-K count after the
  coordinate transform but before the frozen native valid-mask row filter; and
- the stored rows/scores are after mask filtering and remain score-sorted.

Arm A uses the same requested K as each cached record. It may return fewer
post-threshold or post-mask rows, which is part of the detector outcome. Store
grid-index coordinates in the cached native domain and never add 0.5 in the
producer. Also store the exact `reference_keypoints_sha256` for every Arm A
query: the current fail-closed consumer requires this registry binding even
when reference-row descriptors are not used.

One code-contract caveat must be resolved before Arm A on any non-divisible
native size. Query-cache construction filters sparse grid coordinates with
nearest-cell `round()`, whereas the current probe validator checks candidate
mask cells with `floor()`. They are identical for integer coordinates, hence
for XFeat when every native `(H, W)` is already divisible by 32, but can differ
after the `(rw, rh)` rescaling above. Arm A must either prove all inputs are
divisible by 32 or first align the validator with the shared frozen mask
helper. The producer must not silently round keypoints or discard disagreement
rows. Arm B is unaffected because it reuses the exact frozen SuperPoint rows.

Command draft:

```bash
PYTHONPATH=/root/STDLoc:/mnt/pool/sqy/G4Splat_runs/cambridge_stmarys_strict_ablation_v2/external_controls/ulfloc_main_b28d532_clean \
  python -m scripts.materialize_xfeat_frontend_probe \
  --arm detector \
  --images-root /locked/mapping/rgb/root \
  --query-cache /locked/stairs/query_cache.pt \
  --teacher /locked/stairs/complete_positive_teacher.pt \
  --artifact-manifest /root/STDLoc/docs/frontend_checkpoint_inventory_20260812.json \
  --candidate xfeat_sparse_64d --device cpu --dtype float32 \
  --detection-threshold 0.05 --nms-kernel 5 \
  --output /locked/probes/stairs_xfeat64_detector.pt

PYTHONPATH=/root/STDLoc python -m scripts.audit_frontend_upper_bound evaluate \
  --state /locked/stairs/compact_anchor_map.pt \
  --query-cache /locked/stairs/query_cache.pt \
  --teacher /locked/stairs/complete_positive_teacher.pt \
  --probe-cache /locked/probes/stairs_xfeat64_detector.pt \
  --arm detector \
  --output /locked/reports/stairs_xfeat64_detector.json
```

Separate probe files are preferred so descriptor-only and detector-only
lineage cannot accidentally borrow fields from one another. Combining the two
arms or running XFeat+LighterGlue would answer a different question.

## Remaining blocked candidates

DISK and DeDoDe code are locally available through Kornia 0.8.2, but their
pretrained detector/descriptor weights are absent. R2D2 exporter code is
present, but its `r2d2_WASF_N16.pt` is absent and the code would download it.
ALIKE has only a wrapper; its submodule and weights are absent. D2-Net was not
found. LoFTR, LightGlue, and `xfeat-lighterglue.pt` are pair-conditioned and
inadmissible. None of these changes the preferred next candidate: locked
single-image XFeat.

## Conclusion

The earlier `BLOCKED_BY_ARTIFACT` conclusion is superseded for the local
inventory: both ceiling-probe arms now have a credible candidate artifact.
The scientific conclusion remains open until a reviewed producer materializes
mapping-only probes and the existing gates are executed. The clean order is
descriptor Arm B first, detector Arm A second, and no full map or pose rebuild
unless the corresponding mechanism gate passes.
