# Stairs P5.1 Sentinel

## Contract

This sentinel is mapping-only. It uses the frozen Stairs V3 artifacts under
`/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs` and
never reads the test split while selecting candidates or fitting the metric.
The only selector behavior change is an alias-risk tie-break when matching
coverage gain is exactly equal. Alias risk never deletes a candidate and does
not change the precision/core ordering.

All CUDA work in this worktree is pinned to `CUDA_VISIBLE_DEVICES=1`.

## All-candidate alias audit

The eligible universe contains 27,263 candidates:

- 5,079 broad, observation-grounded Track candidates;
- 22,184 mapping-legal surface candidates.

The complete 2,000-query Top-1 graph was built in four mapping-only shards and
merged before risk calibration. Raster visibility was deliberately disabled:
the frozen cache is indexed by canonical rows rather than this new candidate
registry, so candidate legality uses rendered depth/alpha plus ground-truth
reprojection instead of silently applying a misaligned visibility tensor.

| Evidence | Candidates | False wins | Harmful events | Recurrent alias | Risk p50 |
|---|---:|---:|---:|---:|---:|
| Track | 5,079 | 1,146,309 | 317,974 | 5,059 | 0.951 |
| Surface | 22,184 | 643,086 | 116,581 | 15,885 | 0.555 |

Leave-one-trajectory-group-out separability is strong and consistent:

- false-vs-clean AUC: 0.853;
- harmful-vs-clean AUC: 0.904;
- all 8 held-out groups are above random for both targets.

This passes the mechanism gate for the conservative equal-gain selector
tie-break. It does not authorize a risk threshold, candidate deletion, entity
folding, or a descriptor-space change.

## Artifact locations

Large experiment artifacts are not committed:

- candidate universe: `/tmp/lafgs_v4_stairs_p51/selector_candidate_universe.pt`;
- merged Top-1 graph: `/tmp/lafgs_v4_stairs_p51/all_candidate_alias_top1_graph.pt`;
- cross-group audit: `/tmp/lafgs_v4_stairs_p51/all_candidate_alias_risk_audit.pt`;
- selector bundle: `/tmp/lafgs_v4_stairs_p51/selector`.

The selector, compact graph/teacher reconstruction, bounded metric refresh and
mapping pose gate are recorded below once each preceding contract passes.

## Alias-aware selector replay

The full 2,000-query selector replay preserves the V3 budget and every matching
constraint while exchanging candidates only through equal-gain ordering:

| Metric | V3 | Alias tie-break |
|---|---:|---:|
| Final anchors | 7,275 | 7,275 |
| Track / surface | 2,480 / 4,795 | 2,473 / 4,802 |
| Matching reserve | 4,740 | 4,740 |
| Achieved / feasible matching rank | 118,375 / 118,375 | 118,375 / 118,375 |
| Unmet query / rank | 0 / 0 | 0 / 0 |
| Final matching-rank p10 | 56 | 56 |
| Pose additions | 512 | 512 |
| Information logdet p10 | 38.60947 | 38.61205 |
| Translation worst-std p90 | 0.252452 | 0.252948 |

The final sets share 6,914 candidates and exchange 361 in each direction. The
removed candidates have mean supported risk 0.579, 19,402 false wins and 3,472
harmful events; the replacements have mean risk 0.310, 4,608 false wins and
1,045 harmful events. Across the complete selected set this is a reduction of
14,794 false wins and 2,427 harmful events.

The mechanism result is therefore mixed rather than a standalone Go: matching
constraints are exactly preserved and alias evidence improves substantially,
while the information p10 improves only slightly and translation worst-std p90
worsens by 0.20%. The selector is advanced to metric and mapping-pose gates,
which must decide whether the candidate exchange is useful in the real solver.

## Compact evidence and bounded metric refresh

The selector map was not allowed to reuse the frozen 48,000-row canonical
function graph or the 7,275-row V3 teacher/metric by coincidence of row count.
All downstream evidence was rebuilt against the exact selector-map row order:

- compact Top-64 function graph: 7,275 anchors, 2,000 mapping queries;
- raster provenance: 7,326 anchor-source edges over 6,711 source primitives;
- complete-positive teacher: 219,072 positive rows, 236,080 strong pairs,
  820,380 ambiguous pairs and 146,419 exact Track positives;
- bounded metric: 1,520 steps, rank 16, maximum residual norm 0.05, seed 2026,
  no V3 metric warm-start.

The graph merge validates anchor count, source primitive IDs, Track cluster
IDs, anchor type, query order and all mapping-calibrated thresholds. The
teacher validates every query row against the compact graph. The final metric
state has 7,275 landmark indices exactly equal to the trained map anchor IDs.

## Mapping pose gate: Stop

V3 and the alias selector were evaluated on the same deterministic 96-query
uniform mapping gate for RANSAC seeds 2026, 2027 and 2028. No test query was
read. Values below are alias-selector minus V3:

| Seed | Raw P@2 (pp) | Inlier P (pp) | Mean TE (cm) | Median TE (cm) | p90 TE (cm) | p95 TE (cm) | CVaR95 (cm) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | -0.02136 | -0.12556 | +0.29867 | +0.03505 | +0.10913 | +0.34567 | +5.06573 |
| 2027 | -0.02136 | -0.03570 | +0.02181 | +0.04428 | +0.06097 | +0.00784 | +0.07434 |
| 2028 | -0.02136 | -0.03515 | +0.01930 | +0.03317 | +0.06097 | +0.00784 | +0.07434 |

The 100-cm catastrophe count remains zero in both arms, but every seed loses
raw precision, inlier precision, mean/median translation accuracy and all
reported tail metrics. Seed 2026 additionally amplifies CVaR95 from 3.771 cm
to 8.836 cm.

The Stairs decision is therefore **Stop alias-risk equal-gain tie-break**. The
alias audit remains useful diagnostic evidence: it correctly exposes recurrent
identity confusion and should be retained for failure attribution. It is not a
safe deployment objective in this form. Together with the independent Heads
and ShopFacade failures, this prevents the P5.1 tie-break from becoming the
default selector. No test evaluation is authorized or needed for this branch.

## Reproduction outline

The commands use the frozen Stairs artifacts named above and run these modules
in order:

1. `scripts.materialize_selector_candidate_universe`;
2. four `evidence.function_graph --topk 1` shards, then
   `evidence.merge_function_graph` and
   `scripts.audit_all_candidate_alias_risk`;
3. `topology.adaptive_distillation --alias-risk-audit ...`;
4. four `evidence.function_graph --topk 64` shards and merge;
5. four `priors.provenance` shards and merge;
6. four `map_learning.observations` shards and merge;
7. `map_learning.trainer --steps 1520 --rank 16 --metric-residual 0.05`;
8. paired `scripts.evaluate_mapping_cache --query-count 96` calls for seeds
   2026, 2027 and 2028.

Every CUDA command is prefixed by `CUDA_VISIBLE_DEVICES=1`. The provenance
commands additionally place the `g4splat` environment's `bin` first in `PATH`
and its `lib` first in `LD_LIBRARY_PATH`, which is required for the gsplat JIT
to resolve `ninja` and the matching `libstdc++`.
