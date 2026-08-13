# P8 coverage V2 Track and Stage-B preregistration

This addendum freezes the only Track experiment authorized by the formal
cross-scene V2 Stage-A GO.  The machine-readable contract is
`docs/evidence/p8_cycle_verified_fisher_coverage_v2_stage_b_preregistration.json`.

The Track runner is paired and fail-closed.  One process consumes one frozen
scene registry and builds the nearest control first and the coverage selection
second.  Both arms reuse exact rows from the already frozen probe; descriptor
matching and camera-pair selection are unreachable.  A new, nonexistent output
root is mandatory.  Four validated factor/report artifacts must exist before an
atomic completion manifest is written.  A partial directory is never resumable
or a valid Stage-B input.

The base Stage-B gate retains at least 98% of triangulated, broad, and
high-confidence Tracks, permits at most 5% covariance-p90 degradation, and
does not reduce broad-query coverage.  Three additional attestations prove
exact probe-row and matcher reuse.  Stairs also has five hard no-regression
gates against its successful V1 variant and requires exact scientific tensor
and metric parity between the V2 nearest control and the frozen V1 nearest
control.

GreatCourt runs first and stops the sequence on failure.  A scene Pass never
authorizes fullchain.  Only a recursively validated two-scene Stage-B Pass may
request implementation of new V2-aware fullchain lineage; it does not authorize
the existing fullchain runner, mapping pose, formal test queries, or a method
default change.
