# V8 V2-map closed-loop feedback result

## Mainline state

The V2 pre-association full rebuild is now the only accepted M0.  The old
unfiltered Full map and the post-hoc strict map are no longer alternate
initialization arms.

```text
complete render RGB -> native SuperPoint -> V2 row evidence
-> fresh pair association -> fresh Tracks -> triangulation/completion
-> V2 M0 (164,871 Anchors)
```

Map SHA256:
`711855ea46fdaede2e49a306cb56d59ae432a1568a881798c3223b2d36f108f3`.

## Feedback observer

The first map-side round replayed 64 frozen, non-test render queries on V2 M0.
Two queries were not ACCEPT and could not drive updates.  Among the 62 admitted
queries, the continuous observer found 13 precision-deficit cases, 48 nominal
successes and one coverage deficit.  Feedback queries never entered mapping
Tracks, the observation CSR or the descriptor bank.

A small GPU-dependent SuperPoint Top-K tail was detected during replay.  V2 row
evidence is therefore remapped by exact integer keypoint coordinates; the
unmatched 0.6%-1.4% tail is fail-closed invalid.  Positional reuse of a stale
row mask is forbidden.

## Action A: reversible Anchor quarantine

Only false attractors repeated in at least two independent feedback poses and
never observed as a positive alternative were admitted.  This proposed 251
Anchors, or 0.152% of M0, well below the 1% action bound.

| Batch | M0 median TE | Proposal median TE | M0 P90 TE | Proposal P90 TE | R5 |
|---|---:|---:|---:|---:|---:|
| Control (62 ACCEPT) | 0.376 cm | 0.374 cm | 1.045 cm | 0.983 cm | unchanged |
| Fresh confirmation (63) | 0.326 cm | 0.326 cm | 1.137 cm | 1.065 cm | unchanged |
| Real mapping RGB (128, evaluation only) | 3.658 cm | 3.640 cm | 27.214 cm | 27.174 cm | unchanged |

The action has a small consistent tail benefit, but fresh-confirmation median
task error is exactly unchanged.  It fails the frozen median-primary gate and
is rolled back.  The key method lesson is that being a false winner inside a
joint oracle correspondence set is insufficient evidence that deleting that
Anchor changes PoseLib beneficially; a replacement attractor may take its
place.

## Action B: bounded descriptor reconstruction

The controller changed 952 Anchor descriptors using only their original
V2-valid rendered mapping observations.  Feedback descriptors supplied scalar
weight evidence but were never copied into the map.  Each direction change was
bounded to at most five degrees.

| Batch | M0 median TE | Proposal median TE | M0 P90 TE | Proposal P90 TE | R5 |
|---|---:|---:|---:|---:|---:|
| Control (62 ACCEPT) | 0.376 cm | 0.376 cm | 1.045 cm | 1.010 cm | unchanged |
| Fresh confirmation (63) | 0.326 cm | 0.326 cm | 1.137 cm | 1.123 cm | unchanged |
| Real mapping RGB (128, evaluation only) | 3.658 cm | 3.727 cm | 27.214 cm | 27.212 cm | 60.94% -> 60.16% |

The fresh median does not improve and the real evaluation loses one R5
success.  This action is also rolled back.

## Decision

The closed loop executed its complete safety path:

```text
V2 M0 -> feedback observation -> bounded proposal
-> same-RGB paired control -> fresh paired confirmation -> ACCEPT/ROLLBACK
```

Both proposals were rejected and the chosen state remains bit-exact V2 M0.
This is a successful safeguarded round with no accepted mutation, not an
authorization to weaken the gates or tune thresholds from confirmation.

The next admissible map-selection action must test each proposed Anchor with an
actual standard-PoseLib removal counterfactual before grouping it for
quarantine.  A false-winner label alone is no longer actionable.  Query detector
training and the real test remain outside this round.

## Artifacts

- Mainline contract: `configs/v8_v2_feedback_mainline.yaml`
- Feedback runner: `scripts/run_v8_map_feedback.py`
- Paired evaluator: `scripts/evaluate_v8_paired_feedback_maps.py`
- Decision aggregator: `scripts/aggregate_v8_map_feedback.py`
- Decision: `/mnt/pool/sqy/lafgs_v8_v2_feedback_20260828/StMarysChurch/decision.json`
- Control comparison: `/mnt/pool/sqy/lafgs_v8_v2_feedback_20260828/StMarysChurch/paired_control.json`
- Fresh confirmation: `/mnt/pool/sqy/lafgs_v8_v2_feedback_20260828/StMarysChurch/paired_confirmation.json`
