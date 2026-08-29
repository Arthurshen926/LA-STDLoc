# V17 competitive closed-loop attribution and confirmation result

## Outcome

V17 makes the descriptor proposal and delete-only controller optimize the same
deployed state, repairs the virtual-query planner, and adds component-level
confirmation attribution.  The attribution result is decisive: the
competition-aware delete controller works, while the descriptor proposal does
not add repeatable value and must be rolled back.

The final frozen action is:

`V16 active set (164,776 Anchors) + identity descriptor metric`.

It is a **confirmed virtual-query Pareto candidate**, not a real-test result and
not the default test operating point.  No test image, test pose, LOO evidence,
query adapter, detector training, prototype, second pass, or feedback
descriptor copy was used.

## Closed-loop model

The complete feedback path is now:

1. V2 filters invalid rendered observations before association and Track
   construction; the rebuilt map is the frozen 164,871-Anchor M0.
2. V2-valid feedback rows retrieve exact global Top-64 candidates.
3. GT projection, depth, alpha, surface depth, covariance, observation support,
   and view-family support label edges as positive, ambiguous, negative, or
   invalid.
4. V16 recomputes the deployed winner, strongest non-certified competitor,
   Anchor-unique safe correspondence reserve, spatial reserve, and pose reserve
   after every proposed deletion.
5. Exact Top-1 and standard PoseLib response authorizes 95 sequential removals.
6. V17 forms descriptor repair pairs only where a wrong active winner has a
   certified-positive alternative, the observer measured positive task
   response, and the wrong Anchor recurs in at least two pose families.
7. Correct active winners are paired with their strongest non-positive
   competitors as preservation constraints.  Repair and protection rows are
   spatially balanced and normalized per query.
8. One bounded query/map-shared low-rank metric proposal is trained.  It neither
   creates an Anchor nor copies a rendered descriptor into the map.
9. Exact held-out control selects the minimum effective gain.  Fresh
   confirmation is report-only and cannot train or select.
10. Post-confirmation component attribution compares active-only, metric-only,
    and joint actions.  Only the active set survives; deployment restores an
    exact identity metric.

The V17 training evidence contains 7,025 repair rows from 82 pose families and
24,542 correct-winner protection rows from 116 pose families.  There are 2,872
cross-family-authorized wrong competitors.  The full learned direction is
bounded by residual norm 0.05; the smallest control-eligible gain is 0.025.
This remains an analyzed proposal rather than a deployed action.

## Why the previous metric controller failed

The old observer selected descriptor pairs before the V16 active-set action.
Consequently, descriptor learning did not know the deployed active winner or
the competitor exposed after deletion.  V17 rebuilds every pair from the same
active competition state used by the delete controller.

A second problem appeared during validation.  The original supervisor chose
alpha=0.25 because one control outlier produced a very large average risk gain.
On a new 91-query batch this action changed 34 Top-1 rows in one already-hard
query and caused a 2,541-task-unit regression.  The candidate was correctly
rejected.  Active deletion alone was safe on the same batch; the failure came
from metric control energy.

V17 therefore evaluates the minimum-intervention rule: among actions that pass
hard safety, have positive paired net gain, do not reduce R5, and qualify as a
default/Pareto candidate, prefer a default candidate and then the smallest
gain.  This selects alpha=0.025 rather than alpha=0.25.  It is not selected from
confirmation.  This prevents the catastrophic large-gain failure, but final
attribution still finds no incremental metric benefit, so alpha is reset to
zero for deployment.

## Held-out control

The held-out control contains 91 V2-ACCEPT feedback queries.  The analyzed
alpha=0.025 joint proposal has:

| metric | Full V2 | V17 joint |
|---|---:|---:|
| R5 | 84.62% | **85.71%** |
| Q50 task | **0.2354** | 0.2376 |
| Q75 task | 0.5143 | **0.4978** |
| Q90 task | 25.5848 | **25.5315** |
| total risk | 428.8108 | **428.7730** |

It recovers one R5 success, has paired benefit 0.408 versus harm 0.076,
maximum regression 0.0286, and block-bootstrap lower-risk probability 0.965.

## Pose-cell confirmation protocol

The historical planner treated a mapping-camera parent as a finite query
family and eventually exhausted almost every parent.  V17 separates geometric
freshness from statistical blocking:

- freshness is a continuous SE(3) pose cell;
- a candidate must lie outside every registered prior-pose collision ellipsoid
  with translation scale 0.30 mapping baselines and rotation scale 5 degrees;
- mapping support remains inside 0.65--2.35 baselines, matching the existing
  certificate support region;
- one query per source parent is used in each batch;
- source-parent reuse is disclosed, and bootstrap still groups all queries
  from that parent together across batches.

An initial planner attempt wrongly rewarded maximum distance.  Its 96 renders
produced 93 UNCERTAIN decisions because they exceeded the already-frozen 2.5
baseline certificate support.  No certificate threshold was changed.  Those
poses were added to the prior registry, and the planner was corrected to seek
new cells inside the supported region.

After the alpha=0.025 proposal was frozen, two fixed 96-pose batches produced 91
and 92 ACCEPT queries.  The first batch was safe and Pareto-dominant but had
only 0.7965 lower-risk bootstrap probability, just below the 0.80 gate.  One
additional batch was appended without changing the method, action, or
threshold; collection stopped regardless of outcome.  The joint analysis uses
183 queries and 178 source-parent blocks.

## Repeated fresh confirmation and attribution

| metric | Full V2 | active-only | metric-only | joint |
|---|---:|---:|---:|---:|
| R5 | 75.41% | **76.50%** | 75.41% | **76.50%** |
| Q50 task | 0.3930 | 0.3930 | 0.3930 | 0.3930 |
| Q75 task | 0.9853 | **0.9584** | 0.9884 | 0.9677 |
| Q90 task | 34.0316 | 34.0316 | 34.0316 | 34.0316 |
| catastrophic count | 19 | 19 | 19 | 19 |
| total risk | 826.8890 | **826.8618** | 826.8896 | 826.8636 |

Active-only recovers two R5 successes, has paired benefit 1.239 versus harm
0.242, maximum task regression 0.141, and 0.947 lower-risk bootstrap
probability.  It strictly dominates Full and is `PARETO_CANDIDATE`.

Metric-only keeps R5 unchanged, worsens Q75, has only 0.5455 lower-risk
probability, and is `ANALYSIS_ONLY`.  The joint action is still better than
Full, but is slightly worse than active-only in Q75 and total risk and has lower
bootstrap support (0.8915).  Therefore the joint result is carried by the
delete-only action; it is not evidence that descriptor reconstruction works.

For the originally frozen joint proposal, batch one is positive but
`ANALYSIS_ONLY` and batch two is `PARETO_CANDIDATE`; the pooled joint result is
also Pareto.  Component attribution then uses exactly those sealed records and
does not select another metric gain.  It chooses the simpler already
control-authorized V16 active action and rolls the metric back to identity.

## What is and is not solved

The self-localization loop now has a coherent and empirically positive local
delete-only map action:

- observer labels the current plant rather than a historical pair choice;
- controller recomputes the competition/reserve state after every deletion;
- exact Top-1 and PoseLib response bound local harm;
- fresh pose-cell confirmation reproduces active-only R5/Q75 improvement.

Descriptor learning is better specified than before, but is not yet an
effective control action.  Correct-winner protection on design queries does
not guarantee unseen winner stability, and a globally shared low-rank
direction can move many irrelevant rows.  It remains outside the mainline.

The large-scale compression objective is not solved.  Only 95/164,871 Anchors
are removed (0.0576%), so this result cannot support the old several-thousand-
Anchor compression claim.  V17 establishes a safe positive local closed loop;
global sufficiency distillation still needs a separate budgeted controller
whose compression actions pass the same competitive reserve and confirmation
protocol.

This evidence is also render-domain only.  The confirmed action must not be
declared the real-query operating point until a protocol-frozen real
confirmation source exists or the single authorized final test is run.  Test
must not be used to select alpha, active set, detector, or metric.

## Artifacts

- Training report:
  `/mnt/pool/sqy/lafgs_v17_competitive_metric_20260829/StMarysChurch/design_metric/report.json`
- Source metric SHA256:
  `e88834167e7093a7bf9364bce474b8528a0236cfeb6f8fdff0f79033fef61c49`
- Minimum-intervention control decision SHA256:
  `d60fe2832645d00c9a534cb88c8129f424c8d3170a5b1bc6f064e76a50185ac8`
- Repeated confirmation decision SHA256:
  `ba899f9b15ce218bbb26d0e5af1192b8e0b8ae17f6305293b2239a92b6943600`
- Active-only attribution SHA256:
  `7071455919078055af2ed23732d6dfba84beb1bef6b2b38f8f443e6e2e4d28ae`
- Confirmed identity metric SHA256:
  `28785fd1163ce0074359c72b39588bbc880d49cb72bdae83c85faa23538b4a7a`
- Confirmed identity metric:
  `/mnt/pool/sqy/lafgs_v17_competitive_metric_20260829/StMarysChurch/confirmed_active_action/identity_metric.pt`
- Active map SHA256:
  `4d46f55aa36067f925ad08a0e4727f4883064364b96d67122fe6de565163e116`
