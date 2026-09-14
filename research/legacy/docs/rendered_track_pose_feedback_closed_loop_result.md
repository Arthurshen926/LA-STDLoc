# Render-only pose-feedback closed loop

## Outcome

The render-only route now has a complete executable lineage from a frozen
mapping map to real-test evaluation:

```text
rendered RGB observations and ray-triangulated Tracks
  -> V1.4 Track-only map, identity metric, and positive teacher
  -> query-local LOO global-Top-1 + PoseLib mapping feedback
  -> one bounded harmful-attractor map revision
  -> exact mapping PoseLib risk comparison
  -> frozen scene choice
  -> real test through the choice artifact
```

No original mapping RGB, Gaussian primitive center, or test query is used to
construct, revise, or select the map.  Test evaluation still uses one global
Top-1 assignment per query row and one standard PoseLib call.  The implementation
is commit `067a1a7cf5fcfd782a1f90e8ca97e7f7fadfd465` on
`codex/render-track-artifact-r1`; the full CPU suite is 630 passed and one
explicit CUDA smoke skipped.

This closes the former implementation gap where full-mapping self-localization
was only an audit.  Its anchor outcomes now produce a bounded candidate map,
and a predeclared aggregate pose objective chooses between that map and V1.4.
The objective is the existing `pose_policy_loss`, averaged over all mapping
queries.  It combines task-normalized translation and rotation error,
catastrophic error, and solver hypotheses.  There is no collection of strict
per-metric gates and no threshold scan.

## Mapping-only revision

| Scene | V1.4 anchors | Pruned | Mapping risk, base -> revision | Decision |
|---|---:|---:|---:|---|
| Stairs | 5,702 | 57 | 0.236408 -> 0.237137 | retain V1.4 |
| ShopFacade | 5,788 | 15 | 0.239242 -> 0.239096 | select revision |

The Stairs candidate illustrates the remaining method boundary.  The 57
removed Tracks had zero correct-winner events, 2,057 false-attractor events,
and 6,988 harmful solver inliers in the input replay.  Removing them improves
mean TE from 3.678 to 3.625 cm, CVaR95 from 65.744 to 64.680 cm, and raw
precision from 6.326% to 6.393%.  It nevertheless slightly regresses median,
P90, rotation error, and 5 cm recall, so the aggregate pose risk is worse and
the frozen choice remains V1.4.

ShopFacade removes all 15 anchors that satisfy the same rule.  Mapping median
TE improves from 0.342 to 0.337 cm and P95 from 1.099 to 1.091 cm; the two
pre-existing extreme mapping failures remain.  Its small aggregate risk gain
selects the revised 5,773-anchor map.

## Frozen real-test results

Values are means over PoseLib seeds 2026/2027/2028.  The Stairs row is an exact
closed-loop-consumer replay of the retained V1.4 map.  ShopFacade compares the
newly selected revision with its frozen V1.4 control.

| Scene / frozen choice | Median TE cm | Mean TE cm | P90 TE cm | Mean AE deg | 2 cm recall | 5 cm recall | Raw P@2 | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stairs retained V1.4 | 2.144 | 7.193 | 6.560 | 1.622 | 46.967% | 85.100% | 2.987% | 9.67 |
| ShopFacade V1.4 control | 2.044 | **4.733** | 8.057 | **0.225** | 49.515% | 82.201% | **9.752%** | 0 |
| ShopFacade pose-feedback revision | 2.045 | 4.766 | **7.944** | 0.227 | 49.515% | 82.201% | 9.745% | 0 |

The engineering result is positive: the source-image-free route is now closed
from mapping feedback to a SHA-bound final test consumer, with automatic
fallback to the baseline.  The accuracy result is deliberately narrower.  The
ShopFacade revision exchanges 0.033 cm mean error and 0.0077 percentage points
of raw precision for a 0.113 cm P90 improvement; Stairs rejects the revision.
This is not a new cross-scene accuracy Pareto point.

## Scientific conclusion and remaining boundary

The experiment rules out a useful but limited hypothesis: global deletion of
individually harmful anchors is not enough to remove Stairs' structured false
pose consensus.  It can improve tail magnitude and correspondence precision
while worsening the joint query-level pose objective.  This agrees with the
earlier full-pool, conditional-fusion, and LOO-A1 revalidations: the remaining
failure is non-local in the selected Track set.

The render-only branch is therefore method-complete as an experimental
closed-loop pipeline, but it is not the ideal final optimizer.  The next
distinct accuracy hypothesis, if pursued, must operate on coherent Track sets
or pose hypotheses while preserving one final PoseLib call.  Further scans of
the 1% prune fraction, artifact scalar, A1 steps, or generic map capacity are
not justified.  Real/render provider unification and mixed-map integration
remain intentionally outside this branch.

Machine-readable inputs, decisions, hashes, and all six frozen test contracts
are recorded in
[`docs/evidence/rendered_track_pose_feedback_closed_loop_result.json`](evidence/rendered_track_pose_feedback_closed_loop_result.json).
