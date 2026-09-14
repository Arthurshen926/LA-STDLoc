# V36 mapping-quality sparse pose polish

V36 keeps the source-image-free F0 map and its global Top-1 matcher unchanged.
It adds one bounded online feedback step after the standard PoseLib estimate:

1. retain current inliers with reprojection residual at most 4 px;
2. rank their owning Anchors by mapping-only matchability and triangulation
   covariance;
3. keep the best half, with at least 64 rows, and run one local PoseLib polish;
4. accept only if at least 98% of the original inliers survive, the selected-row
   median residual does not increase, and the update stays within 10 cm / 0.06
   degrees.

The policy consumes no source mapping RGB, query GT, test adaptation, query
rendering, dense matching, or offline self-localization feedback. It changes no
Anchor identity or descriptor. The first pose supplies the residual/inlier
feedback; frozen mapping evidence supplies the reliability prior.

## Vanilla 2DGS results

All rows use seed 2026 and the unmodified high-capacity vanilla-2DGS F0 map.

| Scene | Variant | Median TE cm | Mean TE cm | P90 TE cm | Median RE deg | Mean RE deg | P90 RE deg | R5 | Catastrophes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KingsCollege | F0 | 24.1260 | 27.3089 | 52.2142 | 0.17054 | 0.20255 | 0.34820 | 1/343 | 1 |
| KingsCollege | F0 + V36 | **23.7733** | **26.9354** | **52.1087** | **0.16688** | **0.20215** | 0.35070 | **6/343** | 1 |
| OldHospital | F0 | 21.1187 | 31.1317 | 58.4321 | 0.28133 | 0.38485 | 0.79282 | 0/182 | 4 |
| OldHospital | F0 + V36 | **20.9933** | **30.9068** | 58.4321 | 0.28133 | 0.38492 | 0.79282 | **2/182** | 4 |

Across 525 queries, median/mean/P90 TE change from
22.4742/28.6342/53.3251 cm to **22.2439/28.3122/53.2130 cm**. Median/mean/P90
RE change from 0.18517/0.26575/0.51297 degrees to
**0.18112/0.26551/0.51297 degrees**. Paired TE improves on 75 changed queries
and worsens on 19; R5 rises from 1 to 8 and catastrophes remain 5.

The isolated polish stage costs 5.21 ms mean on KingsCollege and 4.08 ms on
OldHospital. Whole-run latency is not compared because these runs overlapped
other reconstruction jobs; stage timing is the relevant incremental cost.

Artifacts:

- KingsCollege summary SHA256: `83f9c67d5c1075db365173ce1a13347bfd886e31975833e08b9bbca101ea103a`
- OldHospital summary SHA256: `35c9b0030d56031ff51a8c953afd7d51a3e65a80ef186c615231acb013d25463`

## Rejected alternatives

- V32 pose-selected descriptor modes found real mapping multimodality, but did
  not improve final TE/RE.
- Exploratory global owner-collapsed modes improved KingsCollege raw 4 px
  correspondence precision from 24.95% to 26.32%, but did not improve median
  TE and lost one R5 success; on OldHospital they worsened P90 TE. They are not
  authorized in the deployment path.
- Reliability-gated Top-8 replacement changed too many identities. Even a
  32-row cap reduced the number of correct correspondences and did not provide
  balanced final-pose gains.
- Projection-first 24 px local candidates generated millions of candidate
  edges on OldHospital but selected exactly the same final correspondences and
  poses as the frozen sparse refinement.

These failures show that mapping reliability should weight trusted geometric
optimization, not act as query-independent evidence for changing Anchor
identity.
