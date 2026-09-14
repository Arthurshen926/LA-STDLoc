# V16 competitive localization sufficiency result

## Outcome

V16 replaces the additive global-harmful/static-sufficiency model with an
explicit Top-L competition state and bounded sequential control. The method is
now theoretically aligned with the deployed matcher:

`certified descriptor competition -> Anchor-unique safe correspondences -> safe pose reserve -> exact Top-1/PoseLib action response`.

The implementation works as a safe controller, but neither its active-set
action nor its blockwise joint metric action produced a repeatable improvement
on fresh confirmation. The deployment decision remains **rollback to V2 M0**.
No test RGB, pose, or result was used.

## Frozen scope

- Candidate map: V2 pre-association rebuild, 164,871 Anchors.
- Geometry, Tracks, triangulation, V2 certificate, and deployed single-stage
  localizer are frozen.
- Feedback queries never enter Tracks, observation CSR, or descriptor bank.
- No LOO, detector, query gate, prototype, descriptor copying, second pass, or
  Anchor addition is used.
- The controlled state is `(active Anchor set, bounded shared-metric gain)`.

## Competitive state

For every V2-valid row of each ACCEPT design query, V16 stores the exact
Top-64 Anchor rows and scores. GT pose, alpha, render depth, surface-median
depth, Anchor projection covariance, observation support, and view-family
support partition each edge into:

- certified positive;
- ambiguous (competitor only);
- negative;
- invalid (excluded from supervision, still conservatively unable to become a
  positive).

For active set `S`, V16 recomputes the actual winner, strongest non-certified
competitor, and every positive beating that competitor by delta=0.005. It then
computes an exact maximum-cardinality row-to-Anchor matching, spatial-cell
reserve, and full 6-DoF pose information from only these safe unique matches.

The A/B/C unit test required by the method is executable: if B is the correct
winner, A is a geometrically valid but weaker backup, and wrong C beats A,
deleting B changes the winner to C and fails the reserve gate. Static geometry
can no longer call that deletion safe.

## Bounded controller

The old 164 causal-harmful rows are only action proposals, never global labels.
Starting from Full M0, each proposal is evaluated against the current state:

1. identify only query rows for which the Anchor is a winner, safe positive,
   or strongest non-certified competitor;
2. deactivate it tentatively;
3. recompute competition, unique correspondence, spatial, and pose reserve;
4. reject any reserve decrease or correct-winner loss;
5. rerun standard PoseLib for every query whose deployed Top-1 changed;
6. require positive bounded cumulative task gain from at least two pose
   families, no lost success, and maximum regression <=0.25;
7. commit accepted actions and rebuild state before the next proposal.

This is bounded alternating local search with exact deployment replay, not a
monotone/submodular selector and not a claim of global optimality. Top-64
exhaustion fails closed and requires full retrieval fallback before an action
could be accepted.

On 138 ACCEPT design queries, 95 removals were accepted. Across sequential
design replay their bounded cumulative gain is +2.846 and maximum action
regression is 0.017. Reserve checks rejected 136 proposal evaluations; two more
failed actual task response. The output has 164,776 Anchors (0.0576%
compression).

## Held-out control

On the 91-query held-out control split:

| metric | Full V2 | V16-95 |
|---|---:|---:|
| R5 | 84.62% | **85.71%** |
| Q50 task | **0.2354** | 0.2376 |
| Q75 task | 0.5143 | **0.4978** |
| Q90 task | 4.0000 | 4.0000 |
| total risk | 2.9207 | **2.8961** |

It recovers one failure, loses no success, has maximum regression 0.0221, and
bootstrap lower-risk probability 0.9465. It advances as a Pareto candidate.

## First fresh confirmation: active set only

A new 180-pose plan excludes all earlier V9--V15 plans. V2 accepts 173,
marks 3 uncertain, and rejects 4. Only V16-95 is evaluated.

The planner has one registered stratum (`novel_se3`) in this run and accepts
96.1% of it. ACCEPT/non-ACCEPT median trajectory novelty is 1.00/1.57 m,
view-angle novelty 24.77/23.39 deg, and visible-cell count 15/14. The certificate
therefore removes a slightly more distant tail, but does not erase an entire
planner stratum or collapse spatial coverage.

R5, Q50, Q90, failure count, and catastrophic count are unchanged. Total risk
improves only 0.016%, maximum regression is 0.0051, and bootstrap support is
0.406. The action is safe but statistically neutral, so it is not confirmed.

## Blockwise metric control

With the active set fixed, the corrected V2-valid rank-16 metric is replayed at
gains 0.05/0.10/0.25/0.50/0.75/1.0/1.25. Control selects alpha=0.10 plus
V16-95: R5 remains 85.71%, Q75 becomes 0.4809, total risk improves 0.94%,
maximum regression is 0.0286, and bootstrap support is 0.8945. Larger gains do
not pass.

Because the first fresh confirmation had already been opened, it is not reused.
The planner excludes it and all historical plans. Only 71 fully novel pose
families remain under the frozen novelty gates, so a 64-pose plan is sealed;
V2 accepts 60, marks 3 uncertain, and rejects 1.

This remaining `novel_se3` stratum has a 93.8% acceptance rate; ACCEPT versus
non-ACCEPT median view-angle novelty is 29.61/33.38 deg and visible-cell count
15/12. The smaller sample is caused by exhausted disjoint pose-family capacity,
not by certificate rejection.

On these 60 ACCEPT queries the joint action keeps R5 (96.67%), failures (2),
and catastrophes (1), with maximum regression 0.0299, but total risk worsens
0.079% and lower-risk bootstrap support is 0.2485. It is not confirmed.

## What is fixed and what is not

V16 fixes the central theoretical hole:

- Anchor effects are conditional on query and current active set;
- actual Top-1 reassignment is state, not an after-the-fact diagnostic;
- safe correspondence reserve precedes pose information;
- every accepted action has an exact sequential action-response measurement;
- non-monotonicity is explicit;
- confirmation and rollback remain atomic.

What remains unsolved is transfer strength. The design renders identify real
within-distribution false attractors, but deleting 95/164,871 Anchors affects
too few unseen families. The shared metric has broader reach, yet its learned
direction does not generalize consistently even at alpha=0.10. This is not a
gate problem: both fresh candidates pass the hard safety checks and still lack
positive independent risk evidence.

The next justified work is not broader deletion. It is to regenerate the
metric training objective directly from the competitive state: optimize
certified-positive versus strongest-current-competitor margins, weight each
pose family and query once, protect current correct winners, and train on a
new design split. That learned direction must earn entry through the same gain
curve and a replenished scene/trajectory confirmation source. The current
planner has nearly exhausted strictly disjoint virtual pose families, so merely
resampling this Gaussian prior would no longer provide an honest fresh test.

## Artifacts

- Competitive controller report: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/design_controller/report.json`
- Competitive state: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/design_controller/competitive_state.pt`
- Active-set control decision: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/control_decision.json`
- Active-only confirmation: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/confirmation_decision.json`
- Joint control decision: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/joint_control_decision.json`
- Joint confirmation decision: `/mnt/pool/sqy/lafgs_v16_competitive_closed_loop_20260829/StMarysChurch/joint_confirmation_decision.json`
