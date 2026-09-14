# Geometry evidence → real-query validation → local extension pilot

## Outcome

The geometry detector has useful associations with GT reprojection proximity,
but it is not a universal bad-Anchor gate. A bounded mapping-only geometry
extension pilot did **not** improve final localization and is not promoted.
No offline self-localization feedback has been enabled. Default F0 is unchanged.
All three localization evaluations in this report completed (182 OldHospital
test images each); there is no pending evaluation behind these conclusions.

## Read-only correspondence telemetry

`scripts.evaluate --audit-correspondences` optionally writes
`correspondence_audit.npz` after solving poses. It contains query offsets,
Anchor rows, GT reprojection errors and PnP inlier membership. GT is not passed
back into matching, PnP or map construction. The outputs are test diagnostics,
not deployment artifacts or physically verified correspondence labels.

The vanilla and MAtCHA telemetry runs reproduced all historical query TE/RE
exactly. Geometry evidence is joined by source map SHA and Anchor row; unseen
Anchor rows remain `not_sampled`, not `unknown` or presumed good. Analyses
include Track-length strata and paired within-query comparisons.

## Measured relation, 2,048 sampled Anchor rows per prior

| Prior / geometry status | Matched rows | Within GT 4 px | PnP inliers | Inlier within GT 4 px |
|---|---:|---:|---:|---:|
| Vanilla / consistent | 3,004 | 14.31% | 716 | 60.06% |
| Vanilla / unstable | 2,158 | 10.33% | 437 | 51.03% |
| Vanilla / unknown | 4,520 | 3.23% | 354 | 41.24% |
| MAtCHA / consistent | 4,065 | 22.76% | 1,105 | 83.71% |
| MAtCHA / unstable | 1,837 | 13.45% | 309 | 79.94% |
| MAtCHA / unknown | 4,282 | 7.05% | 516 | 58.53% |

Within queries having at least five matches of each measured category,
unstable-minus-consistent raw precision averages -3.48 percentage points for
vanilla (180 queries), -8.12 for MAtCHA (173 queries). This is descriptive
association, not causal attribution or independent generalization evidence.

The length-controlled results prohibit a blanket gate: for vanilla Tracks
with >=20 observations, consistent/unstable raw precision is 19.36%/20.91%,
and inlier precision 65.71%/66.92%. For 10–19 observations, raw precision is
14.96%/3.30%, suggesting a narrower intervention may be more useful. Unknown
rows still supply 146/302 correct-under-4px inliers in the two sampled panels;
discarding them could remove coverage.

## Implemented local geometry extension

Pilot inputs are the frozen vanilla map, cache, mapping-only geometry audit
and the cache-bound V2 audit shards. It uses no real query or source RGB
tensor. The choice to investigate shorter unstable Tracks follows the above
test diagnostic, so **policy development is test-adaptive**.

- Target sampled base Tracks classified unstable, with 7–19 observations.
- Project their current XYZ into existing mapping cameras not in the Track.
- Seek actual V2-valid detected rows within 8 px; cosine >=0.8 and local
  best/second margin >=0.04 (singleton candidates still require absolute
  cosine and spatial support).
- Require 0.25 m separation from existing cameras and between additions;
  add at most two views. This is a separation proxy, not guaranteed independent
  noise or visibility. Cross-group parallax/cheirality checks remain necessary.
- Require the extended measurements to pass the same independent cross-group
  prediction check; optimize XYZ with Huber pixel residuals and fixed cameras.
- Bound update by 2% of median depth, retain original-view residual support,
  and require a well-conditioned fit.

This is a **geometry-only ablation**: descriptors, IDs, Anchor count and
descriptor observation CSR remain bitwise unchanged. Added geometry-fit
observations are explicitly stored separately. Covariance and the existing
descriptor-view support geometry are updated; covariance remains a local
model estimate, not certified absolute error. There is no claim that the
new observations have corrected physical feature identity.

Of 210 targets, 173 acquired candidate measurements, 45 passed acceptance.
Median/P90 XYZ shift was 2.38/19.24 cm. Those rows participated in 281 real
query matches, including 51 original PnP inliers. Original and modified maps
produced identical query-to-Anchor match identities and query offsets.

## Final pose validation

| OldHospital vanilla | F0 | Local extension |
|---|---:|---:|
| Median TE cm | 21.118695 | 21.118695 |
| Mean TE cm | 31.131749 | 31.136987 |
| P90 TE cm | 58.432062 | 58.432062 |
| Median RE degrees | 0.281331 | 0.281331 |
| Mean RE degrees | 0.384852 | 0.384888 |
| >=100 cm failures | 4 | 4 |

TE improved on 16 queries, worsened on 30, was unchanged on 136. The mean
change was +0.00524 cm. This is a negative pilot, not a meaningful regression
or a gain. Matcher time remained about 5 ms and RANSAC about 284 ms; no
isolated timing speedup is claimed.

For the fixed original 51 affected inliers, median GT reprojection error
decreased 4.69→4.25 px and the <=4 px count rose 20→23. Across all 281 affected
matches, errors improved on 132 and worsened on 149. Most affected matches
were not original inliers. Small geometric corrections cannot by themselves
correct their potentially wrong feature identities; this pilot did not alter
those identities and did not improve final poses.

## Decision and remaining work

Keep the detector and telemetry as diagnostic tools. Do not deploy a hard
geometry gate, do not expand this extension pilot to all scenes, and do not
claim that passing internal geometry checks establishes real accuracy.

The next high-information work is independently verified physical identity /
fixed-geometry descriptor diagnosis. Further modification of measurements
should be motivated by that diagnosis, rather than another mapping-score
improvement. The limited negative pilot does not establish a full-map or
theoretical upper bound.

Artifacts under `/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/`:

- Each prior: `evaluation/geometry_diagnostic_seed2026`,
  `geometry_query_evidence_panel2048.json`.
- Vanilla: `geometry_extension_panel2048`,
  `evaluation/geometry_extension_panel2048_seed2026`.
- Extension map SHA256:
  `846273d36238f334ed875f3f3311c090205af84bf981216bb7dc898641004108`.
