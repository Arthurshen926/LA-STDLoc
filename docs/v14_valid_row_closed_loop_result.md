# V14 valid-row self-localization feedback result

## Outcome

V14 repairs the feedback loop around a single, simple controller: remove a very
small set of feedback-identified Anchors, then evaluate the resulting map with
the unchanged one-shot localization plant. The selected action deletes 128 of
164,871 Anchors (0.0776%). It is independently **Pareto-confirmed**, but is not
yet promoted over V2 M0 because confirmation bootstrap support is 0.8455 rather
than the 0.95 default-deployment threshold. No test query was used.

## Historical faults that were repaired

The central fault was a broken V2 contract. The certificate supplied a local
`row_valid` mask, but V9 Observer, V9 confirmation, V10 descriptor
counterfactuals, V10 detector-target generation, and V13 gain-curve evaluation
used all extracted rows. About 20% of rows on ACCEPT feedback renders were known
artifact/blur/unsupported rows. They could improve render-to-render
self-localization through a same-domain shortcut while corrupting the causal
evidence used to update the map. V13 is therefore superseded.

Four additional faults were corrected:

1. Baseline failures were always labelled unresolved. V14 adds
   `causal_recoverable_failure` when independent, Anchor-unique, spatially
   distributed alternatives cross the R5 boundary. Twelve of 229 ACCEPT
   feedback queries satisfy this definition.
2. One query's complete ideal task gain was credited independently to every
   negative Anchor. V14 bounds task credit at 4.0 and distributes it by changed
   row support. Exact removal replay, not this heuristic credit, authorizes an
   action.
3. Catastrophic pose magnitude entered CVaR without a cap and was then counted
   again by a catastrophic penalty. V14 caps continuous task error at 4.0 and
   separately counts failures and catastrophes.
4. Earlier control reused feedback data for both action construction and
   evaluation. V14 seals a 60/40 SHA-ranked pose-family design/control split
   before controller construction.

## Closed-loop structure

The deployed state is only the Anchor active set. The planner samples
interpolation-free virtual poses disjoint from all earlier pose families. V2
certifies each render and filters invalid rows. The Observer measures nominal
success, precision deficit, recoverable failure, unresolved failure, and
unreliable input. The controller proposes delete-only prefixes from exact
single-Anchor PoseLib counterfactuals. The plant then performs complete global
Top-1 matching and one standard PoseLib solve for every joint prefix. The
supervisor applies hard safety, bounded task risk, and pose-family bootstrap.

There is no LOO, Anchor addition, retriangulation, second localization pass,
prototype bank, query gate, or detector inside this map controller. The query
detector remains a separate post-map module. Descriptor reconstruction and the
shared low-rank metric are optional controller arms, not mandatory stages.

## Observer and identification evidence

After V2 row filtering, the 229 ACCEPT feedback queries contain 131 causal
precision deficits, 12 causal recoverable failures, 50 nominal successes, and
36 unresolved failures. The design/control split contains 138/91 ACCEPT
queries. Delete-one replay audited 1,000 candidate Anchors on design data; 164
passed the atomic multi-family and no-lost-success gates. Joint prefixes of
8/16/32/64/128 deletions were then replayed only on control.

The selected Anchors are not rows that V2 forgot to reject: every one already
survived the full V2 pre-association rebuild. They are discovered by a different
signal—each repeatedly becomes a geometrically wrong Top-1 attractor in 2--8
independent design pose families, and deleting it improves exact pose replay.
Their median map matchability is 0.177 versus 0.200 globally, while median
position-covariance norm is 0.00678 versus 0.00276. These weak tendencies are
not used as deletion rules; they only confirm that feedback is complementary to
the static render-validity filter.

| held-out control (91 ACCEPT) | V2 M0 | delete 128 |
|---|---:|---:|
| R5 | 84.62% | **85.71%** |
| Q50 task | 0.2354 | **0.2049** |
| Q75 task | 0.5143 | **0.4719** |
| capped Q90 task | 4.0000 | 4.0000 |
| catastrophic count | 10 | 10 |
| bounded total risk | 2.9207 | **2.8811** |

Relative risk improves 1.36%, one failed query is recovered, no successful
query is lost, maximum capped regression is 0.0097, and bootstrap support is
0.98. This selects `delete_128` before confirmation.

## Fresh independent confirmation

A new 180-pose plan excludes 761 previously consumed pose families. V2 accepts
169, marks 5 uncertain, and rejects 6. Only the frozen `delete_128` action is
evaluated.

| fresh confirmation (169 ACCEPT) | V2 M0 | delete 128 |
|---|---:|---:|
| R5 | 89.94% | **90.53%** |
| Q50 task | 0.2199 | 0.2199 |
| Q75 task | **0.4046** | 0.4089 |
| Q90 task | 1.0115 | **0.9131** |
| catastrophic count | 10 | 10 |
| bounded total risk | 1.6882 | **1.6477** |

Relative risk improves 2.40%, one failed query is recovered, no successful
query is lost, and maximum capped regression is 0.0177. Bootstrap support is
0.8455: above the frozen 0.80 scientific Pareto threshold, below the 0.95
default threshold. The honest decision is `PARETO_CONFIRMED`, not default-map
replacement.

## Other controller arms

The corrected rank-16 shared metric uses only design families, V2-valid rows,
and per-query bounded/row-normalized credit. Across gains 0.05--1.0 on held-out
control, no gain reaches a substantial effect or 0.80 bootstrap support; it is
rejected. This supports keeping the controller delete-only for now.

The historical descriptor and query-detector consumers were repaired to index
the same V2-valid rows as the Observer. Their old negative results are invalid,
but they have not been reintroduced into the active V14 controller. That is
intentional: the verified delete-only action already supplies the cleanest
causal claim, while descriptor or detector learning must earn entry through its
own held-out plant replay.

## Artifacts

- Control decision: `/mnt/pool/sqy/lafgs_v14_valid_row_closed_loop_20260828/StMarysChurch/control_decision_v2.json`
- Selected map: `/mnt/pool/sqy/lafgs_v14_valid_row_closed_loop_20260828/StMarysChurch/active_128/projective_anchor_map.pt`
- Fresh certificate: `/mnt/pool/sqy/lafgs_v14_valid_row_closed_loop_20260828/StMarysChurch/fresh_confirmation_certified/manifest.json`
- Confirmation decision: `/mnt/pool/sqy/lafgs_v14_valid_row_closed_loop_20260828/StMarysChurch/fresh_confirmation_decision_v2.json`
- Corrected metric decision: `/mnt/pool/sqy/lafgs_v14_valid_row_closed_loop_20260828/StMarysChurch/corrected_metric_control_decision.json`
