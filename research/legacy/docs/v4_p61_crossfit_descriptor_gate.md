# P6.1 Cross-fitted Observation Descriptor Gate

## Question and contract

P6.0 showed that surface Anchor observations have much higher dispersion than
Track observations.  P6.1 asks the stricter causal question: does a descriptor
fused from one set of mapping trajectories retrieve the same surface Anchor
better on disjoint held-out trajectories than the current deployment
descriptor?

The gate is mapping-only and audit-only.  It never mutates the Map, uses no test
query, leaves every Track descriptor unchanged, and compares both arms in the
same frozen V3 shared-metric space:

```text
support descriptor = frozen_metric(robust_fuse(raw native observations))
held-out query      = frozen_metric(raw native observation)
baseline bank       = current deployment anchor features
retrieval           = exact global cosine Top-1 over the complete Anchor bank
```

The variant replaces all support-eligible surface descriptors in parallel, so
the held-out result includes the real change in global competition rather than
only measuring positive-pair cosine.

## Heads protocol blocker

The Heads mapping split used by the frozen V3 run contains 1,000 images, all
from `seq-02`.  A trajectory-block A/B cross-fit is therefore impossible.
P6.1 records `requires_at_least_two_mapping_trajectories` and does not fabricate
independence by splitting adjacent frames.  Heads remains a later pose guard if
a variant first passes a valid multi-trajectory mechanism gate.

## Stairs exhaustive balanced trajectory partitions

Stairs has four mapping trajectories: `seq-02`, `seq-03`, `seq-05`, and
`seq-06`.  With two trajectories per fold there are exactly three unique
partitions up to A/B reversal; all three were evaluated.  Eligibility requires
at least two support queries, two support strata, and at least one held-out
observation.  Values below are observation-fused minus current deployment.

| Fold A / Fold B | Bidirectional eligible surface | A→B R@1 | B→A R@1 | Added false winners A→B / B→A | Stable in both directions |
|---|---:|---:|---:|---:|---:|
| 02+05 / 03+06 | 1,826 | 17.49%→4.55% (-12.94 pp) | 18.48%→4.24% (-14.24 pp) | +2,267 / +3,637 | 9 |
| 02+03 / 05+06 | 1,927 | 17.54%→4.21% (-13.33 pp) | 19.22%→6.15% (-13.08 pp) | +3,187 / +3,053 | 12 |
| 02+06 / 03+05 | 1,556 | 17.98%→4.38% (-13.60 pp) | 16.03%→3.42% (-12.61 pp) | +2,017 / +3,042 | 8 |

Across the six directions, held-out positive cosine decreases by 0.113–0.154
on average and positive margin decreases by 0.116–0.160.  Cross-fold descriptor
direction is not itself the decisive gate: 1,045–1,338 eligible Anchors per
partition exceed cosine 0.65, yet only 8–12 preserve mean held-out R@1 and
margin in both directions.  The stable sets contain 18 unique Anchors across
all partitions and only 2 Anchors are stable under every partition.

## Decision

**Stop Obs-all and do not run compact refresh or pose for this descriptor
source.**  The failure is an order of magnitude larger than a pose-gate noise
band and is reproduced in every possible balanced trajectory partition.  A
bounded metric refresh cannot be used to rescue a source descriptor that loses
12.6–14.2 R@1 percentage points before retraining; doing so would change the
question from observation materialization to learning a new identity model.

**Obs-stable is also not a useful deployable branch in its present form.**  At
most 8–12 of 4,795 surface Anchors pass a single partition and only 2 pass all
partitions, which is too little coverage to justify a new materialization and
metric-refresh path.

This does not mean the observations are useless.  It establishes that the
current A1 deployment descriptor is not a noisy approximation to a single raw
observation medoid.  The high P6.0 dispersion is more consistent with
multimodal/contaminated surface identity or with A1 deliberately moving the
descriptor away from its positive mean to preserve global discrimination.
The next independent accuracy factors should therefore be the already-audited
mapping evidence defects: fixed `K_mapping=2048 / NMS=4` and a fixed-budget
parallax-aware pair graph.  A two-prototype oracle is deferred until those
upstream evidence factors have been tested.

## Artifacts

Machine-readable outputs are persisted under:

`/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/audits/observation_descriptor_crossfit`

The implementation is `topology/observation_descriptor_crossfit.py`; the CLI
is `scripts/audit_observation_descriptor_crossfit.py`.

The default Stairs partition is reproduced with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.audit_observation_descriptor_crossfit \
  --registry /tmp/lafgs_stairs_anchor_registry_cov_v4_compat.pt \
  --query-cache /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/query_cache.pt \
  --metric-state /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/map_learning/metric_state_step_1520.pt \
  --output /tmp/lafgs_p61_crossfit/stairs_split_0205.pt \
  --device cuda --score-chunk 256 --cpu-threads 1
```

The other two unique balanced partitions add respectively
`--fold-a-trajectories seq-02,seq-03` and
`--fold-a-trajectories seq-02,seq-06`.
