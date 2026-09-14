# V15 budgeted global-sufficiency closed-loop result

## Outcome

V15 makes global map size a first-class closed-loop control variable. It builds
strictly nested mapping-only and feedback-conditioned maps from the frozen V2
map, replays the unchanged global Top-1 plus standard PoseLib plant, increases
the budget when task safety fails, and sends exactly one control-selected map
to independent confirmation.

The loop operated correctly but the proposed 160,000-Anchor map was **not
confirmed**. The deployable result is therefore `NOT_CONFIRMED`: keep the full
164,871-Anchor V2 map. No test query was used.

This is still a useful method result. Feedback reduced the hard-sufficient core
from 11,432 to 10,587 Anchors and repeatedly beat mapping-only selection at the
same large budget. However, mapping-observation sufficiency is not yet a strong
enough certificate for unseen-pose localization sufficiency.

## Implemented closed loop

The state is the Anchor active set and the controlled resource is its size. The
pipeline is:

1. Start from the complete V2 pre-association rebuild: 164,871 Anchors and
   2,017,016 V2-valid mapping observations.
2. Apply frozen mapping eligibility and hard per-image matching, image-cell,
   view-family, depth-range, and 6-DoF pose-information constraints.
3. Build feedback utility only from the 138 ACCEPT design pose families. Clean
   support and task gain are family-unique and query-bounded. Exact delete-one
   replay contributes 164 causally harmful rows.
4. Let feedback change only Anchor priority. It cannot weaken eligibility or
   any hard mapping constraint.
5. Select the minimum sufficient core, then fill a fixed, nested reliability
   order to the requested global budget.
6. Replay every map with complete global Top-1 matching and exactly one
   standard PoseLib solve on a sealed control batch.
7. Enforce R5 non-decrease, non-increasing catastrophic failures, bounded Q90,
   no large single regression, lower paired total risk, and pose-family
   bootstrap support. If the gate fails, increase map budget rather than relax
   safety.
8. Evaluate only the one control-selected operating point on independent
   confirmation. Confirmation cannot select a fallback.

There is no LOO, Anchor addition, retriangulation, second localization pass,
prototype bank, descriptor adapter, or query detector in this controller.

## Repairs made during implementation

The first budget materialization incorrectly changed the hard safety target
with budget. That made feasibility non-monotonic and is retained only as a
failure audit. The corrected curve uses one constant safety floor at every
budget. Both policies are exactly nested at
12k/20k/32k/48k/64k/80k/96k/112k/128k/144k/152k/160k; every budget at and above
12k meets every hard mapping constraint. An 8k budget is infeasible for both
policies and is rejected.

The legacy supervisor also defined a substantial effect only as localization
improvement. V15 recognizes material map compression as an effect, but never as
an excuse for regression: all task hard checks must still pass and paired total
risk must be strictly lower.

## Design evidence and controller action

Static mapping constraints produce an 11,432-Anchor minimum core. Adding
feedback priority produces a 10,587-Anchor core, 845 Anchors or 7.39% smaller.
Across the full 160k budget, feedback and mapping-only selections differ by
only 144 rows. Nevertheless, feedback removes 163 of the 164 design-time
causally harmful Anchors, while mapping-only removes only 19. The observer and
controller therefore have a real, strong effect; they are not inert.

The fixed safety curve showed that 12k--64k maps are unsafe relative to Full.
The controller expanded the budget without changing selector parameters:

| feedback map | control R5 | total risk | risk change vs Full | bootstrap | result |
|---|---:|---:|---:|---:|---|
| 64k | 88.37% | 1.7801 | -5.69% | 0.2305 | reject |
| 80k | 89.53% | 1.6443 | +2.37% | 0.4010 | reject |
| 96k | 88.37% | 1.6880 | -0.22% | 0.3295 | reject |
| 112k | 88.37% | 1.7868 | -6.09% | 0.2705 | reject |
| 128k | 89.53% | 1.6787 | +0.33% | 0.6400 | reject |
| 144k | 90.12% | 1.6192 | +3.87% | 0.7525 | reject |
| 152k | 89.53% | 1.6317 | +3.12% | 0.7555 | reject |
| 160k | **90.12%** | **1.6014** | **+4.92%** | **0.8720** | Pareto candidate |

The Full control baseline is R5 88.95% and total risk 1.6843. The 160k
feedback map also beats 160k mapping-only by 4.84% relative risk with bootstrap
0.8775. It is the only operating point advanced to confirmation.

## Independent confirmation

The confirmation manifest contains 174 V2 ACCEPT queries from pose families
not used to construct or select the action.

| fresh confirmation | Full V2 | 160k feedback map |
|---|---:|---:|
| R5 | **94.83%** | 94.25% |
| Q50 task | **0.1553** | 0.1571 |
| Q90 task | **0.4785** | 0.4975 |
| catastrophic count | 4 | 4 |
| bounded total risk | **1.1652** | 1.2204 |

Relative risk worsens 4.73%, one successful query is lost, and bootstrap support
for lower risk is only 0.344. Query 68 / pose family 1074 is the decisive tail
failure: Full uses 64 inliers and has 0.37 cm / 0.069 deg error; the compressed
map changes 235 Top-1 assignments, uses 57 inliers, and reaches 26.55 cm /
0.600 deg. Its capped regression is 3.924, violating the pre-registered 2.0
single-query limit. The action is therefore rejected with no fallback search.

## Interpretation

The closed-loop architecture now works procedurally and safely: the observer
changes the map, the plant detects the effect, the budget controller expands
state when needed, and independent confirmation vetoes a non-generalizing
action. What does not yet work is the sufficiency model.

Current hard constraints reconstruct evidence seen along mapping views. They do
not constrain nearest-neighbor margins or correspondence alternatives from a
previously unseen pose family. In addition, a row that is causally harmful on
several design renders is not necessarily globally dispensable. At 160k the
feedback priority nearly removes the entire harmful set, so this
pose-family-specific evidence is over-generalized even though it is applied as
a soft score.

The next justified method change is narrow: add a design/control-only
**novel-view correspondence reserve constraint** to global sufficiency. It
should protect Anchor-unique, spatially distributed Top-1 alternatives and
their match-margin coverage across independent virtual pose families. It must
remain a hard selector constraint, use no confirmation or test rows, and be
ablated separately from feedback priority. Adding networks, descriptors, or a
second localization pass is not supported by this result.

## Artifacts

- Control selection: `/mnt/pool/sqy/lafgs_v15_global_sufficiency_20260828/StMarysChurch/fresh_v4_control_decision.json`
- Confirmation decision: `/mnt/pool/sqy/lafgs_v15_global_sufficiency_20260828/StMarysChurch/fresh_v4_confirmation_decision.json`
- Rejected 160k candidate: `/mnt/pool/sqy/lafgs_v15_global_sufficiency_20260828/StMarysChurch/global_v4_final_expansion/b160k_feedback_conditioned/projective_anchor_map.pt`
- Fixed expansion configs: `configs/v15_global_sufficiency_budget_expansion.yaml` and `configs/v15_global_sufficiency_final_expansion.yaml`
