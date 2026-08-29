# V13 risk-sensitive self-localization feedback result

> **Superseded by V14.** V13 did not apply the V2 `row_valid` mask before
> matching, PoseLib replay, Observer evidence generation, or descriptor-action
> evaluation. Its 229-query control and 153-query confirmation numbers are
> therefore retained only as historical diagnostics and must not support a
> method claim. See `docs/v14_valid_row_closed_loop_result.md`.

## Scope and frozen protocol

V13 starts from the accepted V2 pre-association rebuild M0 (164,871 Anchors).
It does not use test RGB or test poses, LOO, new geometry, query detectors,
multi-prototypes, descriptor copying, or a second localization pass.  The only
controllers are the frozen V9 shared low-rank direction and the strict 87-Anchor
active-set action.  Every reported output comes from exact global Top-1 followed
by one standard PoseLib call.

The executable contract is `configs/v13_risk_sensitive_closed_loop.yaml`.

## Supervisor correction

V13 separates hard safety, scalar task risk, and Pareto value.  Its preregistered
risk is a weighted Q50/Q75/Q90/CVaR95 integral with R5, catastrophic and latency
terms.  It reports benefit and harm magnitudes and uses paired pose-family block
bootstrap instead of a fixed median-gain gate.  A 0.80 lower-risk probability
retains a scientific Pareto candidate; 0.95 is required for a new default.

## Local action-response identification

The frozen V9 metric residual was replayed at gains 0.25--1.25 on all 229 V2
ACCEPT feedback/control queries.  Gain 0.25 reduced aggregate risk but had only
0.643 bootstrap support and one large regression; larger gains were unstable.
This rejects the interpretation that the old metric only needed a relaxed gate.

A second control-only curve tested gains 0.05--0.20, the strict 87-Anchor active
set, and their exact joint actions.  The action-query influence matrix selected
gain 0.05 plus the active set:

| control metric | V2 M0 | selected joint action |
|---|---:|---:|
| Q50 task error | 0.24132 | 0.23342 |
| Q75 task error | 0.63442 | 0.56747 |
| Q90 task error | 363.70955 | 325.86968 |
| CVaR95 task error | 2722.72878 | 2586.36302 |
| total risk | 636.74379 | 599.99451 |
| R5 | 80.79% | 80.79% |
| catastrophic | 30 | 30 |

The paired benefit/harm was 3153.620/0.165, maximum regression was 0.0315 task
unit, and block-bootstrap lower-risk probability was 0.996.  No confirmation
result participated in this selection.

## Fresh confirmation

A new interpolation-free plan excluded 600 pose families consumed by V9--V11.
V2 certification produced 153 ACCEPT, 6 UNCERTAIN, and 1 REJECT query.  Only the
single frozen joint action was evaluated on the 153 ACCEPT queries.

| confirmation metric | V2 M0 | selected joint action |
|---|---:|---:|
| Q50 task error | 0.19643 | 0.19091 |
| Q75 task error | 0.34358 | 0.34707 |
| Q90 task error | 0.61373 | 0.60017 |
| CVaR95 task error | 1662.03608 | 1662.03608 |
| total risk | 333.18871 | 333.18408 |
| R5 | 92.81% | 92.81% |
| catastrophic | 10 | 10 |

Benefit was 0.459, harm 0.246, net task gain +0.214, and the maximum per-query
regression was 0.0371.  Pose-family bootstrap assigns probability 0.842 that the
joint action has lower risk.  It passes every hard safety gate and the fixed
Pareto threshold, but not the 0.95 default threshold.

## Decision

The action is retained as the first independently confirmed feedback-derived
**Pareto candidate**.  It is not installed as the default plant, and V2 M0
remains the default.  A second feedback round is not run because the frozen
protocol permits it only after a default candidate is confirmed.

This result establishes that the feedback loop can produce a transferable,
bounded action when the Observer is connected to exact action-response
identification and joint control.  It does not yet establish a default-map win
or real-query/test improvement.
