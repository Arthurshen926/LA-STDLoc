# V6 virtual-probe sparse-prototype result

## Frozen intervention

The evaluated implementation is commit `93bea39`.  It keeps the V6 deployment
plant at native SuperPoint, exact global Top-1, and one standard PoseLib solve.
Virtual probes remain observer-only and never become map observations.  The
controller uses only pose-grouped probe indices 0--5; indices 6--7, all mapping
queries, and all test queries are excluded from action design.

For a failed training probe, depth-certified alternatives are ordered by their
ground-truth reprojection gain, but an action is accepted only after replaying
the unchanged Top-1 plus PoseLib plant.  The selected intervention adds 163
normalized appearance prototypes owned by existing Anchors.  Prototype winners
are collapsed to the owner Anchor before PnP, so geometry, Anchor identity,
Anchor count, and the pose solver do not change.

## Probe evidence

The baseline and candidate both contain 16 rendered views: two sensor variants
for each of eight pose probes.  The two held-out pose groups keep all of their
sensor variants in validation.

| scope | baseline R5 | candidate R5 | baseline catastrophe | candidate catastrophe |
|---|---:|---:|---:|---:|
| all probes (16) | 75.00% | 87.50% | 4 | 2 |
| controller train (12) | 66.67% | 83.33% | 4 | 2 |
| independent pose holdout (4) | 100.00% | 100.00% | 0 | 0 |

The two controllable training failures change from 12840.59/13101.42 cm to
1.77/2.78 cm.  The two reverse-view failures have no successful pose-valid
oracle and remain routed to structure/prior rather than descriptor control.

Candidate probe feedback:
`/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v1/candidate_budget256_v2_probe_feedback.json`
(SHA-256 `1c658cb1fac2c4fcf3d96bba1c0e084c8912e1b71f38f0785f196cc860b84dce`).

## Mapping-only independent validation

All 1,487 mapping queries are independent of prototype training.  R5 remains
97.041022%, median/P90 remain 0.290604/0.755513 cm, and catastrophe remains 29.
There are no success/failure flips.  Only 128 of 3,045,376 Top-1 rows change;
12 poses change with a maximum matrix-element delta of `5.8e-4`.  Six
translation errors improve and six regress.  Two exact-correct winners become
negative, without changing pose success.

The explicit seq2 subset contains 352 queries.  R5 remains 98.863636%,
catastrophe remains 2, median/P90 remain 0.285135/0.694376 cm, one translation
error changes, and there are no success/failure flips.

Paired artifact:
`/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v1/candidate_budget256_v2_mapping_paired.json`
(SHA-256 `7c29769f1837df12f26d207cf2c05ec4b9fb6cafee862df64b5eec06d49617cc`).

## Frozen real test

The 530-query StMarysChurch test was run once with seed 2026 after the probe and
mapping checks.  Aggregate accuracy is unchanged from the V6 baseline:

| metric | baseline | candidate |
|---|---:|---:|
| median TE | 4.021415 cm | 4.021415 cm |
| P90 TE | 12.619572 cm | 12.619572 cm |
| R5 | 61.886792% | 61.886792% |
| catastrophe | 11 | 11 |

Ten poses change: five translation errors improve and five regress, with no
success/failure flip.  Matching mean changes from 12.54 ms to 13.35 ms.  The
deployment map grows from 270,134,778 to 270,306,875 bytes (+172,097 bytes).

Test summary:
`/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v1/final_test_seed2026/summary.json`
(SHA-256 `9ab5528dcdb8a5f4ba8ec69692bb9a0766a284db40a8e02c641c2182b40b3a3e`).

## Decision

This is the first V6 action to close the observer-controller-plant loop and
recover actual PoseLib failures while preserving independent probe and mapping
success rates.  It is therefore useful mechanism evidence.  It is not promoted
as an accuracy improvement because the real-test success metric is unchanged.
The remaining limitation is observer-domain coverage: the current Gaussian
probe family excites synthetic pose/sensor failures but still does not reproduce
the real-test failure distribution.  The next intervention must improve probe
domain realism or frontend correspondence coverage, not increase prototype
budget using this test result.
