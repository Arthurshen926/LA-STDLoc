# V9 no-LOO causal closed-loop result

## Frozen scope

V9 starts only from the accepted V2 pre-association full rebuild:

- map: `/mnt/pool/sqy/lafgs_v8_v2_full_rebuild_20260828/StMarysChurch/projective_map/projective_anchor_map.pt`
- map SHA256: `711855ea46fdaede2e49a306cb56d59ae432a1568a881798c3223b2d36f108f3`
- Anchor count: 164,871
- geometry, Track registry, triangulation and Anchor addition: frozen
- permitted map actions: shared bounded descriptor metric, or delete-only active set
- LOO geometry, LOO descriptors and every other LOO variant: forbidden by config and runtime assertions
- feedback queries never enter Track, observation CSR or descriptor bank
- test RGB, test poses and test results: unused

The executable contract is `configs/v9_no_loo_closed_loop.yaml`.

## Query planning and render certification

The V9 planner contains no trajectory interpolation arm. It samples substantial
SE(3) offsets and ambiguity-directed views, then requires distance, view-angle,
Anchor-visibility and spatial-overlap gates.

- mapping cameras: 1,487
- feedback plan: 256 queries, SHA256 `4e1da87728666fa0137351aaf28f6f8fd6f28ee0e57c06a9316cca645fa247d5`
- confirmation plan: 128 queries, SHA256 `d07860e25e83346f7915cfba47692fa4eb2f52b00b315e3da5170bf451160177`
- feedback/confirmation exact pose overlap: 0
- feedback/confirmation source pose-family overlap: 0
- trajectory interpolation candidate count: 0
- LOO used: false

V2 Render Certificate remains hybrid: a whole query is ACCEPT/UNCERTAIN/REJECT,
and only an ACCEPT query can expose locally valid rows.

| batch | ACCEPT | UNCERTAIN | REJECT |
|---|---:|---:|---:|
| feedback (256) | 229 | 25 | 2 |
| confirmation (128) | 119 | 7 | 2 |

Non-ACCEPT rows never train or select an action.

## Causal Top-K observer

The observer performs exact global Top-64 retrieval on the frozen M0. GT pose,
render alpha/depth and the V2 local row mask only label whether a retrieved
candidate is geometrically supported. They do not enter the online plant.

For a query to be a precision deficit, the observer requires:

1. the baseline pose already succeeds;
2. a correct candidate exists inside Top-64;
3. the replacement set is Anchor-unique and spatially dispersed;
4. one fresh standard PoseLib replay actually reduces the frozen task error.

The two GPU shards produced:

| category | queries |
|---|---:|
| causal precision deficit | 126 |
| nominal success | 59 |
| unresolved failure | 44 |
| unreliable query | 27 |

There were 28,674 causal wrong-to-right rows. Requiring the same false-attractor
Anchor to occur in at least two independent pose families retained 4,641 Anchor
candidates and 12,398 ranking rows. This is substantially stronger evidence
than the old false-winner label, but it is still diagnostic evidence rather
than automatic mutation authority.

## Shared bounded metric action

The metric action uses one shared transform for query and map descriptors:

`normalize(d + clip(U SiLU(Vd), 0.05))`, rank 16.

Feedback descriptors enter only the ranking loss. They are never copied into
the map as a descriptor or prototype. Training used 12,398 causal ranking rows
and 45,357 clean-row protection rows.

Training evidence improved the difficult-pair median margin from -0.05328 to
-0.05029, but only 2.51% of those difficult pairs flipped order. Fresh paired
confirmation gave:

| metric | M0 | proposal |
|---|---:|---:|
| median translation | 0.906 cm | 0.968 cm |
| P90 translation | 12.986 cm | 9.460 cm |
| R5 | 86.55% | 87.39% |
| catastrophic queries | 8 | 7 |
| paired median task gain | \- | 0.000 |

The tail and R5 improved, but the preregistered global paired-median gain was
not positive and median precision regressed. Decision: **ROLLBACK**.

Decision artifact:
`/mnt/pool/sqy/lafgs_v9_no_loo_20260828/StMarysChurch/shared_metric_decision.json`.

## Delete-only active-set action

The active-set controller first ranks false attractors by cumulative causal
query gain and pose-family support. For the fixed top 1,000 candidates it then
actually removes one Anchor from every affected query's Top-64 list, substitutes
the next candidate, and reruns standard PoseLib. A label alone cannot authorize
deletion.

An intermediate 202-Anchor proposal exposed a controller defect: median-positive
evidence could still have negative cumulative gain because of a large harmful
outlier. Confirmation was stopped before producing a result. The controller was
made fail-closed by additionally requiring positive cumulative gain and a fixed
minimum median actual gain of 0.001. The superseded proposal remains an audit
artifact and is not a method candidate.

The corrected controller authorized 87 removals (0.0528% of M0). Fresh paired
confirmation gave:

| metric | M0 | strict delete-only proposal |
|---|---:|---:|
| median translation | 0.906 cm | 0.871 cm |
| median task error | 0.1812 | 0.1755 |
| P90 translation | 12.986 cm | 12.986 cm |
| R5 | 86.55% | 86.55% |
| catastrophic queries | 8 | 8 |

Only 57 confirmation queries changed. Their conditional median task gain was
0.00234 and cumulative task gain was +0.243, but 17/57 (29.8%) worsened. This
exceeds the fixed 25% sparse-action safety cap. The whole-batch paired median
gain was also zero. Decision: **ROLLBACK**.

Action-specific decision artifact:
`/mnt/pool/sqy/lafgs_v9_no_loo_20260828/StMarysChurch/active_set_strict_sparse_decision.json`.

## Final method state and conclusion

V9 completed one theoretically separated loop:

`non-interpolated planner → V2 whole-query/local-row certificate → Top-64 causal observer → bounded metric/delete-only proposals → disjoint fresh confirmation → rollback`.

Neither proposal passed its complete safety gate. The chosen map therefore
remains the V2 pre-association full rebuild M0, byte-for-byte unchanged.

The main methodological result is not that feedback has no signal. It has a
strong signal: 126 successful queries admit a better geometric correspondence
set; the metric improves R5/P90; the strict deletion proposal improves median
precision. The remaining problem is action specificity and harm control:

- a rank-16, radius-0.05 global metric is too weak to flip most hard pairs yet
  broad enough to perturb already-correct matches;
- even Anchor deletions with positive feedback counterfactuals transfer
  imperfectly to independent views;
- therefore observer evidence is now useful, but the present controllers are
  not reliable enough to mutate M0.

No LOO mechanism should be reintroduced. Query scene-detector training and the
separate reversible Gaussian counterfactual remain downstream of this frozen
map decision; neither was mixed into V9 map selection.

## Verification

- focused V9 tests: 6 passed
- complete repository suite: 1,046 passed, 1 skipped
- skipped test: optional CUDA renderer release smoke gate
- Ruff on all V9 implementation, runner and test files: PASS
