# ShopFacade P5.1 alias-risk precision sentinel

## Verdict

**Stop the current equal-gain alias-risk tie-break on ShopFacade.** It passes the selector sufficiency gate, but after an exact 176-step compact metric refresh it reduces deterministic raw precision and worsens median and mean mapping translation error in all three RANSAC seeds. Test queries were not evaluated.

## Frozen lineage

- canonical: `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/ShopFacade/canonical/canonical_48000.pt`
- Track payload: `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/ShopFacade/runs/frozen_v1/statistics_combined_1000_frozen_g3_track_provenance_v1/track_micro_anchor_payload.pt`
- sparse mapping cache: `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/ShopFacade/runs/frozen_v1/query_cache_native_sparse_teacher.pt`
- full-resolution mapping cache: `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/ShopFacade/runs/frozen_v1/query_cache_native_fullres_k2048.pt`
- V3 Map/metric: `lafgs_adaptive_v3_validation_20260806/ShopFacade/{topology_ref2p067_stop1e3,map_learning_ref2p067_stop1e3}`
- output root: `/mnt/pool/sqy/lafgs_v4_shopfacade_sentinel_20260812/p5_1`

All selection and evaluation queries are mapping-only. The full-resolution cache is used only where rendered depth/alpha legality is required; the sparse cache remains the selector, teacher, training, and pose-replay contract.

## Candidate and selector gates

The eligible universe is 15,137 candidates: 7,300 broad observation-grounded Tracks and 7,837 mapping-legal surface candidates. Cross-group audit gives false-vs-clean AUC 0.7103 and harmful-vs-clean AUC 0.7410. The selector changes 6,357→6,361 Anchors while preserving matching rank 26,904/26,904 and unmet query/rank 0/0.

The 35 swapped-out candidates have mean alias risk 0.7186; the 39 swapped-in candidates have mean risk 0.4236. The final set removes 392 false wins and 55 harmful events at the cost of 8 clean events. Final information logdet median improves 42.8829→42.9185, while logdet p10 and translation worst-std p90 remain unchanged.

## Exact compact refresh

The new 6,361-row Map receives a newly built Top-64 function graph, raster provenance, and complete-positive teacher. No 6,357-row artifact is remapped or reused. The teacher contains 231 queries, 82,809 positive rows and 87,524 strong pairs.

Metric configuration matches the frozen V3 protocol:

```text
steps=176, checkpoint=176, batch_size=512, topk=64, max_positives=8
rank=16, metric_residual=0.05, learning_rate=2e-4, temperature=0.04
harmful_weight=0.1, trust_weight=1.0, group_dro_eta=0.03
group_dro_max_weight_ratio=1e9, refresh_interval=0, refresh_shards=7
initial_ransac_refresh=true, seed=2026, initial_metric_state=null
task_translation_m=0.05, task_rotation_deg=5, clean_px=4, ransac_px=12
```

An initial trial with the newer config cap `group_dro_max_weight_ratio=3` was stopped before checkpointing because it changed initial maximum group weight from the frozen V3 value 0.83905 to 0.375. The exact run restores 0.83902; its small residual difference is induced by the four-row Map change.

The reusable strict audit is:

```bash
python -m scripts.audit_compact_artifact_lineage \
  --map "$OUT/alias_selector/adaptive_compact_total06361.pt" \
  --function-graph "$OUT/compact_refresh/function_graph.pt" \
  --complete-positive-teacher "$OUT/compact_refresh/complete_positive_teacher.pt" \
  --metric-state "$OUT/compact_refresh/metric_state_step_0176.pt" \
  --output "$OUT/compact_refresh/lineage_audit.json"
```

It verifies that Map/graph/teacher/metric all contain 6,361 rows, metric IDs equal Map Anchor IDs bitwise, graph/teacher query names and every query-row registry are identical, and the metric has no initial checkpoint.

## Mapping-only pose gates

The preliminary fixed 96-query uniform gate produces the same directional warning in all three seeds:

| Mean delta, alias - V3 | Value |
|---|---:|
| Raw precision | -0.01134 pp |
| Inlier precision | +0.02563 pp |
| Median TE | +0.00479 cm |
| Mean TE | +0.00570 cm |
| P90 TE | -0.03524 cm |
| CVaR95 TE | +0.02052 cm |
| Catastrophic >100cm | 0 |

The final gate evaluates all 231 mapping queries:

| Mean delta, alias - V3 | Value |
|---|---:|
| Raw precision | **-0.007924 pp** |
| Inlier precision | +0.007396 pp |
| Median TE | **+0.039363 cm** |
| Mean TE | **+0.004626 cm** |
| P90 TE | -0.012309 cm |
| P95 TE | +0.029161 cm |
| CVaR95 TE | -0.017369 cm |
| Solver inlier ratio | -0.017918 pp |
| Mean hypotheses | +56.16 |
| Catastrophic >100cm | 0 |

Raw precision is seed-independent: 14.192706%→14.184781%. Median and mean TE regress for seeds 2026, 2027, and 2028. P90 is unchanged/improved, but CVaR95 improves by only 0.0156/0.0184/0.0181 cm and does not offset the precision and central-error regressions. The ShopFacade protection rule requires no precision regression, so the gate fails before test evaluation.

## Reproduction commands

With `OUT=/mnt/pool/sqy/lafgs_v4_shopfacade_sentinel_20260812/p5_1`, the main stages are:

```bash
python -m scripts.materialize_selector_candidate_universe \
  --canonical-map "$CANONICAL" --function-graph "$V3_GRAPH" \
  --track-payload "$TRACKS" --query-cache "$SPARSE_CACHE" \
  --scene-calibration "$CALIBRATION" --config configs/paper_mainline.yaml \
  --output "$OUT/selector_candidate_universe.pt"

CUDA_VISIBLE_DEVICES=2 python -m evidence.function_graph \
  --anchor-map "$OUT/selector_candidate_universe.pt" \
  --query-cache "$FULL_CACHE" --deployment-mask-cache "$MASKS" \
  --topk 1 --strong-radius-px 2 --clean-radius-px 4 \
  --ambiguous-radius-px 8 --pnp-reprojection-error-px 12 \
  --harm-radius-px 12 --depth-abs-tolerance-m 0.050000000106815295 \
  --seed 2026 --output "$OUT/all_candidate_top1_function_graph.pt"

python -m scripts.audit_all_candidate_alias_risk \
  --function-graph "$OUT/all_candidate_top1_function_graph.pt" \
  --track-payload "$TRACKS" --candidate-map "$OUT/selector_candidate_universe.pt" \
  --output "$OUT/all_candidate_alias_risk_audit.pt"

CUDA_VISIBLE_DEVICES=2 python -m topology.adaptive_distillation \
  --canonical-map "$CANONICAL" --function-graph "$V3_GRAPH" \
  --complete-positive-teacher "$CANONICAL_TEACHER" \
  --track-payload "$TRACKS" --query-cache "$SPARSE_CACHE" \
  --alias-risk-audit "$OUT/all_candidate_alias_risk_audit.pt" \
  --output-dir "$OUT/alias_selector" --config configs/paper_mainline.yaml
```

Compact graph/provenance/teacher are rebuilt with the standard sharded modules `evidence.function_graph`, `priors.provenance`, `evidence.evidence_graph`, and `map_learning.observations`. Pose comparison uses `scripts.evaluate_mapping_cache` with `--query-count 96` for the preliminary gate and `--query-count 0` for all 231 mapping queries, each with seeds 2026/2027/2028.

The complete machine-readable result is outside Git at `$OUT/shopfacade_p5_1_sentinel_report.json`; large tensor artifacts are intentionally not committed.
