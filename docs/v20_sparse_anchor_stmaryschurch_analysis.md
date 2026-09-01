# V20 StMarysChurch sparse-Anchor closed-loop analysis

Date: 2026-08-31

## Scope and protocol status

This run evaluates a map-side sparse descriptor action with the Query branch
kept native.  The feedback, design protection, and held-out control inputs are
all certified Gaussian novel-view renders (`lafgs_v7_certified_clean_render`),
not real-camera RGB queries and not the official test set.  Consequently the
run can measure Gaussian-domain self-consistency, but it cannot establish a
real-RGB domain gain.

The formal Tier-B teacher is not authorized.  The candidate and its control
replay are therefore analysis-only; control is consumed, confirmation is not
run, and the stable map remains deployed.

## Bound artifacts

- Baseline map SHA256:
  `711855ea46fdaede2e49a306cb56d59ae432a1568a881798c3223b2d36f108f3`
- V20 evidence:
  `/mnt/pool/sqy/lafgs_v20_sparse_anchor_20260831/StMarysChurch/topk_evidence_tier_b_hardened_v2.pt`
- Evidence SHA256:
  `92be43af95bc3838a7b335b056b19779bf4556f93e38d42fc6201cbae68ca6e4`
- Candidate map:
  `/mnt/pool/sqy/lafgs_v20_sparse_anchor_20260831/StMarysChurch/action_positive_only_angle5_hardened_v4/candidate_anchor_map.pt`
- Candidate map SHA256:
  `a0123fa1bd960cc2b3881b3493b17b073150c5d79895e907171174cc3b08b6f7`
- Control decision:
  `/mnt/pool/sqy/lafgs_v20_sparse_anchor_20260831/StMarysChurch/control_hardened_v5_analysis_decision.json`

## Evidence audit

The design split contains 153 pose families; control contains 103 disjoint
families, of which 91 queries are certified `ACCEPT` for exact replay.

The V9 observer supplied 19,235 design training rows.  Exact row-level V9/V19
binding retained 819 rows and rejected 18,416.  In particular, 18,302 had no
decisive V19 truth and 18,416 did not bind the V9 positive to a V19-positive
Anchor (reason counts overlap).  The final competition evidence contains 223
repair rows over 95 unique positive Anchors.

Clean evidence was similarly narrowed from 26,160 inputs to 1,592 explicit
Query-row/V19-truth-bound protection rows.  Ninety-eight legacy Top-2 entries
were themselves V19-positive and are no longer treated as negatives.  No
wrong-winner Anchor had the required clean-positive support in two pose
families, so negative Anchor updates were disabled.

This audit explains why the earlier loop could report a mapping/group-level
gain without producing a valid descriptor update: most proposed row pairs did
not identify the novel-view truth target.

## Sparse action

The action changes exactly 95 selected Anchor descriptor rows and zero
unselected rows.  Query descriptors, geometry, Anchor IDs, and Anchor XYZ stay
unchanged.  Each descriptor is constrained to a 5-degree spherical cap.

- Minimum globally safe action scale: `0.00390625`
- Coordinate-expanded Anchors: `94 / 95`
- Maximum/mean per-Anchor scale: `1.0 / 0.6977385`
- Maximum observed descriptor angle: `4.98323 degrees`
- Design repair Top-1 recoveries: `71 / 223`
- Design repair regressions: `2 / 223`
- Previously established certified-positive wins lost: `0`
- Positive-pair win rate: `0.0 -> 0.31839`
- Clean broken-winner count: `0 / 1592`
- Clean margin-floor violations: `0 / 1592`
- Saved-dtype materialized action audit: `PASS`

The earlier single global-scale candidate recovered only 3 of 223 design
rows.  Per-Anchor coordinate expansion removes that bottleneck while retaining
the exact clean constraints.  The two regressed repair rows were already wrong
Top-1 rows; no previously established certified-positive win was removed.

## Held-out exact control result

Exact control uses all 2,048 detected rows per accepted Query, global cosine
Top-1 matching, and the standard PoseLib replay.  It is not a ranking-loss
proxy.  The aggregator reopens the SHA-bound certified view and proves exact
coverage of all 91 `ACCEPT` source-record SHA entries before risk supervision.

- Top-1 changed: 99 rows across 46 of 91 Queries
- Queries with lower task error: 21 (23.08%)
- Queries with higher task error: 12 (13.19%)
- Paired benefit / harm / net gain: `0.079372 / 0.014405 / 0.064967`
- Net inlier-count change: `+19`
- Maximum single-Query regression: `0.004339`
- R5: `86.8132% -> 86.8132%`
- Catastrophic count: `9 -> 9`
- Total-risk change (baseline minus candidate): `+0.0001912`
- Probability that candidate risk is lower: `0.547`
- 90% bootstrap risk-delta interval: `[-0.001379, 0.000141]`
- Supervisor classification: `ANALYSIS_ONLY`

The direction is encouraging and hard safety passes, but the uncertainty
interval crosses zero and the probability is below both the 0.80 Pareto and
0.95 default/deployment thresholds.  Even with a formally authorized teacher,
this control result would not advance to confirmation.

## Why Tier B remains unauthorized

Tier-B calibration has 150 decisive assignments, all correct, with Wilson
lower bound `0.98228`.  It nevertheless fails the frozen requirements of at
least 200 decisive assignments and at least two active independent families;
only one family is active.  Separate validation is `212 / 219 = 0.96804`, just
below the 0.97 target.  Tier C is diagnostic-only and cannot authorize this map
mutation.

## Conclusion and next valid experiment

Pure Gaussian feedback is not exhausted: it can identify descriptor
competition and produce a small, held-out, directionally positive pose effect.
It is not yet a demonstrated localization improvement, and it says nothing
about real-RGB generalization.

The next method change should prioritize repair rows by exact pose leverage on
design data, instead of increasing the contrastive scope or blindly pushing
more Top-K negatives.  The current action changes only 99 of roughly 186k
control correspondences and leaves R5 unchanged, showing that ranking flips
are often low-leverage for PnP.  Any such tuning must use a newly generated,
pose-family-disjoint control set; the control set reported here is already
consumed.  Formal deployment additionally requires a rebuilt Tier-B
calibration and a separate untouched confirmation set.
