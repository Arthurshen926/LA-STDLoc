# P7 Stairs pair-policy full-chain runbook

## Decision scope

This run changes only the mapping-camera pair policy from `nearest` to
`parallax_diverse`. It uses the frozen Stairs K_mapping=1024/NMS=4 query cache,
Stage-A state, 2DGS prior, visibility cache, descriptors, provenance-assignment
parameters, selector policy, and V3 numeric calibration. It evaluates mapping
poses only (`q=256` uniformly spaced mapping queries, seeds 2026/2027/2028).
Test images remain forbidden.

The current mainline's independent `mapping` block is compatible: for the
640x480 Stairs processed images it resolves to K_mapping=1024 and NMS=4. The
density factor is a recorded no-go and must not be mixed into this run. Sparse
deployment remains independently area-adaptive; this mapping-only gate replays
the frozen cache and does not rerun the frontend.

## Locked inputs

Run from `/root/STDLoc` with the `g4splat` Python environment and the repository
root on `PYTHONPATH`. The values below are the preregistered inputs, not paths
to be discovered by globbing.

```bash
export PYTHONPATH=/root/STDLoc
export PATH=/root/miniconda3/envs/g4splat/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/g4splat/lib:${LD_LIBRARY_PATH:-}

RUN=/mnt/pool/sqy/lafgs_p7_pair_policy_fullchain_20260812/stairs
INPUT=$RUN/inputs
CONTRACT=$RUN/contracts
DATA=/mnt/pool/sqy/datasets/7Scenes_pgt_lafgs_v1/stairs
BASE=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/stage_a/8660_lafgs_map_state.pt
QUERY=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/query_cache.pt
VIS=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/visibility.pt
PLY=/mnt/pool/sqy/indoor_priors_pgt_v1/7Scenes/stairs/lafgs_prior_v1/point_cloud/iteration_30000/point_cloud.ply
PARENT_CAL=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/scene_calibration.json
PAYLOAD=/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/stairs/k1024/provenance_replay/parallax_diverse_track_micro_anchor_payload.pt
PAYLOAD_AUDIT=/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/stairs/k1024/provenance_replay/parallax_diverse_payload_lineage_audit_v2.json
FACTOR=/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/stairs/k1024/parallax_diverse_track_factor.pt
MECHANISM=/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/stairs/k1024/mechanism_gate.json
BOOTSTRAP_MANIFEST=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/tracks_refined/reproducibility_manifest.json
CONFIG=/root/STDLoc/configs/paper_mainline.yaml

QUERY_SHA=8f65f9ad067f40dd9bd7dda99f3b7674a3b9016b4679d29c0df8a54637d863d2
BASE_SHA=949a1b5bdff5f0d72628393b7f02dee526df4c8e0d104739c5562cb5fef19451
VIS_SHA=63b228b445e72ef3bcfe69fd56ec57e8b533e4d384ad343ae3ad37d571e3433b
PLY_SHA=c1d71e3d1984fd8c18436ea0c447a3e75350217bfbb5c70ae03ea63e838f5699
PARENT_CAL_SHA=d3bc0839d73310055d93b895aa5c96fd633bef0fbab276604bffb782335120b2
PAYLOAD_SHA=f1ab21fae713c37ac9725de8bf7eeacf8dcfa02ba91d45883dd789f04cb5a059
PAYLOAD_AUDIT_SHA=55ecc5f23ab1e7064c16fa1db3f76a3129e6606d66e4138def15c9f9fd23cfe7
FACTOR_SHA=5eb12e6936b0783b1e500d785d6165f1bf21b42bdc5f5b71a018b5d7c2bd5811
MECHANISM_SHA=d32e51396392a76910f119b6f78dcb3a1dde00657af1bfda3c33ff690eaa7c52
BOOTSTRAP_MANIFEST_SHA=cb0241f694f6bdd5f4d663bd88273f94217d0a42502a5be97107b5c35a64b8aa
CONFIG_SHA=c522d3a3d692a5e3c4c6db06083ec5ca9682c9f1c2bef49b6bb135b622b352cc
```

Before doing any work, verify all eleven digests. The 10.38 GB query-cache hash
may be slow on the pool filesystem; a timeout is not evidence of a mismatch.

```bash
test "$(sha256sum "$BASE" | cut -d' ' -f1)" = "$BASE_SHA"
test "$(sha256sum "$QUERY" | cut -d' ' -f1)" = "$QUERY_SHA"
test "$(sha256sum "$VIS" | cut -d' ' -f1)" = "$VIS_SHA"
test "$(sha256sum "$PLY" | cut -d' ' -f1)" = "$PLY_SHA"
test "$(sha256sum "$PARENT_CAL" | cut -d' ' -f1)" = "$PARENT_CAL_SHA"
test "$(sha256sum "$PAYLOAD" | cut -d' ' -f1)" = "$PAYLOAD_SHA"
test "$(sha256sum "$PAYLOAD_AUDIT" | cut -d' ' -f1)" = "$PAYLOAD_AUDIT_SHA"
test "$(sha256sum "$FACTOR" | cut -d' ' -f1)" = "$FACTOR_SHA"
test "$(sha256sum "$MECHANISM" | cut -d' ' -f1)" = "$MECHANISM_SHA"
test "$(sha256sum "$BOOTSTRAP_MANIFEST" | cut -d' ' -f1)" = "$BOOTSTRAP_MANIFEST_SHA"
test "$(sha256sum "$CONFIG" | cut -d' ' -f1)" = "$CONFIG_SHA"
```

The payload audit must additionally report `valid=true`,
`pair_policy=parallax_diverse`, all checks true, and bind factor SHA
`5eb12e6936b0783b1e500d785d6165f1bf21b42bdc5f5b71a018b5d7c2bd5811`.

## 1. Fresh workspace preflight

Never resume a failed scientific output directory. Move the whole failed root
aside and start again. Preflight the empty root before copying any input:

```bash
python -m scripts.pair_fullchain_workspace preflight \
  --root "$RUN" \
  --output "$CONTRACT/preflight.json"

mkdir -p "$INPUT"
cp "$PAYLOAD_AUDIT" "$INPUT/payload_lineage_audit.json"
cp "$PARENT_CAL" "$INPUT/parent_scene_calibration.json"
cp "$CONFIG" "$INPUT/paper_mainline.yaml"
```

The report must say `scientific_artifact_count=0` and `valid=true`.

## 2. Variant-bound frozen calibration

Materialize a new sidecar; do not edit or copy the old calibration under a new
name. The producer must require both expected SHA arguments and bind the valid
payload-lineage audit.

```bash
python -m scripts.materialize_pair_factor_calibration \
  --parent "$INPUT/parent_scene_calibration.json" \
  --expected-parent-calibration-sha256 "$PARENT_CAL_SHA" \
  --query-cache "$QUERY" \
  --expected-query-cache-sha256 "$QUERY_SHA" \
  --track-payload "$PAYLOAD" \
  --payload-lineage-audit "$INPUT/payload_lineage_audit.json" \
  --expected-payload-lineage-audit-sha256 "$PAYLOAD_AUDIT_SHA" \
  --expected-pair-budget 7450 \
  --output "$INPUT/frozen_pair_scene_calibration.json"

python -m scripts.pair_fullchain_workspace lock-inputs \
  --root "$RUN" \
  --input base_state="$BASE" --expected-sha256 base_state="$BASE_SHA" \
  --input query_cache="$QUERY" --expected-sha256 query_cache="$QUERY_SHA" \
  --input visibility_cache="$VIS" --expected-sha256 visibility_cache="$VIS_SHA" \
  --input gaussian_ply="$PLY" --expected-sha256 gaussian_ply="$PLY_SHA" \
  --input track_payload="$PAYLOAD" --expected-sha256 track_payload="$PAYLOAD_SHA" \
  --input pair_factor="$FACTOR" --expected-sha256 pair_factor="$FACTOR_SHA" \
  --input mechanism_gate="$MECHANISM" --expected-sha256 mechanism_gate="$MECHANISM_SHA" \
  --input bootstrap_manifest="$BOOTSTRAP_MANIFEST" \
  --expected-sha256 bootstrap_manifest="$BOOTSTRAP_MANIFEST_SHA" \
  --input parent_calibration="$INPUT/parent_scene_calibration.json" \
  --expected-sha256 parent_calibration="$PARENT_CAL_SHA" \
  --input payload_lineage_audit="$INPUT/payload_lineage_audit.json" \
  --expected-sha256 payload_lineage_audit="$PAYLOAD_AUDIT_SHA" \
  --input config="$INPUT/paper_mainline.yaml" \
  --expected-sha256 config="$CONFIG_SHA" \
  --parent-manifest "$CONTRACT/preflight.json" \
  --output "$CONTRACT/locked_inputs.json"

python -m scripts.pair_fullchain_workspace manifest \
  --root "$RUN" --stage frozen_calibration \
  --artifact calibration="$INPUT/frozen_pair_scene_calibration.json" \
  --artifact payload_lineage_audit="$INPUT/payload_lineage_audit.json" \
  --artifact parent_calibration="$INPUT/parent_scene_calibration.json" \
  --artifact config="$INPUT/paper_mainline.yaml" \
  --parent-manifest "$CONTRACT/locked_inputs.json" \
  --output "$CONTRACT/inputs.json"
```

The generated sidecar must retain V3 `statistics`, `parameters`, and `policy`
exactly, hence `metric_steps=1520`. Its sources must bind the variant payload
path/SHA and query path/SHA; lineage must bind expected and actual parent/audit
SHAs. All downstream consumers validate this and must raise rather than
silently recalibrate.

## 3. Canonical evidence from the variant payload

No old canonical map, function graph, provenance, or teacher is permitted.
The output directory is new, so every artifact below is rebuilt.

```bash
CUDA_VISIBLE_DEVICES=2 python -m scripts.build_evidence \
  --base-state "$BASE" \
  --track-payload "$PAYLOAD" \
  --query-cache "$QUERY" \
  --gaussian-ply "$PLY" \
  --gaussian-type 2dgs --sh-degree 3 \
  --visibility-cache "$VIS" \
  --config "$INPUT/paper_mainline.yaml" \
  --scene-calibration "$INPUT/frozen_pair_scene_calibration.json" \
  --function-graph-shards 4 --provenance-shards 4 --observation-shards 4 \
  --output "$RUN/evidence"

python -m common.evidence_contract \
  --verify --output "$RUN/evidence/evidence_contract.json"

python -m scripts.audit_pair_fullchain_lineage \
  --stage canonical \
  --evidence-contract "$RUN/evidence/evidence_contract.json" \
  --expected-track-payload "$PAYLOAD" \
  --output "$CONTRACT/canonical_lineage.json"

python -m scripts.pair_fullchain_workspace manifest \
  --root "$RUN" --stage canonical_evidence \
  --artifact canonical_map="$RUN/evidence/canonical_map.pt" \
  --artifact function_graph_v2="$RUN/evidence/function_graph_v2.pt" \
  --artifact raster_provenance="$RUN/evidence/raster_provenance.pt" \
  --artifact function_graph="$RUN/evidence/function_graph.pt" \
  --artifact positive_teacher="$RUN/evidence/complete_positive_teacher.pt" \
  --artifact evidence_contract="$RUN/evidence/evidence_contract.json" \
  --artifact lineage_audit="$CONTRACT/canonical_lineage.json" \
  --parent-manifest "$CONTRACT/inputs.json" \
  --output "$CONTRACT/canonical.json"
```

`canonical_lineage.json` must be valid and the evidence contract's Track
artifact SHA must equal `PAYLOAD_SHA`.

## 4. Adaptive distillation with frozen numeric thresholds

```bash
CUDA_VISIBLE_DEVICES=2 python -m scripts.distill_map \
  --canonical-map "$RUN/evidence/canonical_map.pt" \
  --function-graph "$RUN/evidence/function_graph.pt" \
  --positive-teacher "$RUN/evidence/complete_positive_teacher.pt" \
  --track-payload "$PAYLOAD" \
  --query-cache "$QUERY" \
  --scene-calibration "$INPUT/frozen_pair_scene_calibration.json" \
  --config "$INPUT/paper_mainline.yaml" \
  --output "$RUN/topology"

COMPACT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["map"])' \
  "$RUN/topology/adaptive_distillation_build.json")

python -m scripts.pair_fullchain_workspace manifest \
  --root "$RUN" --stage adaptive_distillation \
  --artifact compact_map="$COMPACT" \
  --artifact build_report="$RUN/topology/adaptive_distillation_build.json" \
  --artifact selection_provenance="$RUN/topology/adaptive_selection_provenance.pt" \
  --artifact sufficiency_selection="$RUN/topology/unified_sufficiency_selection.pt" \
  --artifact scene_calibration="$RUN/topology/scene_calibration.json" \
  --parent-manifest "$CONTRACT/canonical.json" \
  --output "$CONTRACT/topology.json"
```

The build report's `calibration_contract.mode` must be
`frozen_numeric_pair_factor`, with the exact sidecar path and SHA. A preexisting
report with another contract is rejected.

## 5. Rebuild compact evidence and cold 1520-step metric

The canonical graph argument is only a compatibility input. The CLI forces
`rebuild_function_graph=true`, producing new compact graph V2, raster
provenance, evidence graph, and complete-positive teacher before training.

```bash
CUDA_VISIBLE_DEVICES=2 python -m scripts.train_compact_map \
  --compact-map "$COMPACT" \
  --canonical-function-graph "$RUN/evidence/function_graph.pt" \
  --track-payload "$PAYLOAD" \
  --query-cache "$QUERY" \
  --gaussian-ply "$PLY" --gaussian-type 2dgs --sh-degree 3 \
  --scene-calibration "$INPUT/frozen_pair_scene_calibration.json" \
  --config "$INPUT/paper_mainline.yaml" \
  --function-graph-shards 4 --provenance-shards 4 --observation-shards 4 \
  --output "$RUN/map_learning"

python -m scripts.audit_pair_fullchain_lineage \
  --stage compact --compact-map "$COMPACT" \
  --raster-provenance "$RUN/map_learning/raster_provenance.pt" \
  --complete-positive-teacher "$RUN/map_learning/complete_positive_teacher.pt" \
  --expected-track-payload "$PAYLOAD" \
  --output "$CONTRACT/compact_lineage.json"

python -m scripts.audit_compact_artifact_lineage \
  --map "$RUN/map_learning/anchor_map_step_1520.pt" \
  --function-graph "$RUN/map_learning/compact_function_graph.pt" \
  --complete-positive-teacher "$RUN/map_learning/complete_positive_teacher.pt" \
  --metric-state "$RUN/map_learning/metric_state_step_1520.pt" \
  --output "$CONTRACT/metric_lineage.json"

python -m scripts.pair_fullchain_workspace manifest \
  --root "$RUN" --stage compact_metric \
  --artifact compact_map="$COMPACT" \
  --artifact compact_function_graph_v2="$RUN/map_learning/compact_function_graph_v2.pt" \
  --artifact compact_function_graph="$RUN/map_learning/compact_function_graph.pt" \
  --artifact compact_provenance="$RUN/map_learning/raster_provenance.pt" \
  --artifact compact_teacher="$RUN/map_learning/complete_positive_teacher.pt" \
  --artifact trained_map="$RUN/map_learning/anchor_map_step_1520.pt" \
  --artifact metric="$RUN/map_learning/metric_state_step_1520.pt" \
  --artifact training_report="$RUN/map_learning/training_report.json" \
  --artifact pair_lineage="$CONTRACT/compact_lineage.json" \
  --artifact metric_lineage="$CONTRACT/metric_lineage.json" \
  --parent-manifest "$CONTRACT/topology.json" \
  --output "$CONTRACT/metric.json"
```

The metric must be cold-started: `training_report.config.initial_metric_state`
is null. The metric audit must show exact map/metric anchor-ID alignment and
teacher/function-graph row alignment. Both lineage reports must be valid.

## 6. Mapping pose q256 x 3

Verify the complete chain immediately before pose evaluation; this detects any
artifact change after manifest creation.

```bash
python -m scripts.pair_fullchain_workspace verify --manifest "$CONTRACT/locked_inputs.json"
python -m scripts.pair_fullchain_workspace verify --manifest "$CONTRACT/inputs.json"
python -m scripts.pair_fullchain_workspace verify --manifest "$CONTRACT/canonical.json"
python -m scripts.pair_fullchain_workspace verify --manifest "$CONTRACT/topology.json"
python -m scripts.pair_fullchain_workspace verify --manifest "$CONTRACT/metric.json"

for SEED in 2026 2027 2028; do
  CUDA_VISIBLE_DEVICES=2 python -m scripts.evaluate_mapping_cache \
    --map "$RUN/map_learning/anchor_map_step_1520.pt" \
    --metric-state "$RUN/map_learning/metric_state_step_1520.pt" \
    --complete-positive-teacher "$RUN/map_learning/complete_positive_teacher.pt" \
    --query-cache "$QUERY" \
    --scene-calibration "$INPUT/frozen_pair_scene_calibration.json" \
    --query-count 256 --seed "$SEED" --device cuda \
    --output "$RUN/mapping_pose_q256_v2/seed${SEED}"
done

# Replay the frozen V3 control with the identical query-index rule and seeds.
V3=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs
CONTROL_MAP=$V3/map_learning/anchor_map_step_1520.pt
CONTROL_METRIC=$V3/map_learning/metric_state_step_1520.pt
CONTROL_TEACHER=$V3/map_learning/complete_positive_teacher.pt
CONTROL_CALIBRATION=$V3/evidence/scene_calibration.json
CONTROL_MAP_SHA=5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620
CONTROL_METRIC_SHA=c55818b3ab27e6d1b4ca929d21d235eed5e80238913113ed596a5ea9b436f903
CONTROL_TEACHER_SHA=3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81
CONTROL_CALIBRATION_SHA=d3bc0839d73310055d93b895aa5c96fd633bef0fbab276604bffb782335120b2

for SEED in 2026 2027 2028; do
  CUDA_VISIBLE_DEVICES=2 python -m scripts.evaluate_mapping_cache \
    --map "$CONTROL_MAP" \
    --metric-state "$CONTROL_METRIC" \
    --complete-positive-teacher "$CONTROL_TEACHER" \
    --query-cache "$QUERY" \
    --scene-calibration "$CONTROL_CALIBRATION" \
    --query-count 256 --seed "$SEED" --device cuda \
    --output "$RUN/baseline_mapping_pose_q256_v2/seed${SEED}"
done

python -m scripts.compare_mapping_pose_gate \
  --baseline-seed 2026="$RUN/baseline_mapping_pose_q256_v2/seed2026/mapping_cache_summary.json" \
  --baseline-seed 2027="$RUN/baseline_mapping_pose_q256_v2/seed2027/mapping_cache_summary.json" \
  --baseline-seed 2028="$RUN/baseline_mapping_pose_q256_v2/seed2028/mapping_cache_summary.json" \
  --variant-seed 2026="$RUN/mapping_pose_q256_v2/seed2026/mapping_cache_summary.json" \
  --variant-seed 2027="$RUN/mapping_pose_q256_v2/seed2027/mapping_cache_summary.json" \
  --variant-seed 2028="$RUN/mapping_pose_q256_v2/seed2028/mapping_cache_summary.json" \
  --baseline-map "$CONTROL_MAP" \
  --baseline-metric "$CONTROL_METRIC" \
  --baseline-teacher "$CONTROL_TEACHER" \
  --baseline-query-cache "$QUERY" \
  --baseline-calibration "$CONTROL_CALIBRATION" \
  --variant-map "$RUN/map_learning/anchor_map_step_1520.pt" \
  --variant-metric "$RUN/map_learning/metric_state_step_1520.pt" \
  --variant-teacher "$RUN/map_learning/complete_positive_teacher.pt" \
  --variant-query-cache "$QUERY" \
  --variant-calibration "$INPUT/frozen_pair_scene_calibration.json" \
  --expected-sha256 baseline.map="$CONTROL_MAP_SHA" \
  --expected-sha256 baseline.metric="$CONTROL_METRIC_SHA" \
  --expected-sha256 baseline.teacher="$CONTROL_TEACHER_SHA" \
  --expected-sha256 baseline.calibration="$CONTROL_CALIBRATION_SHA" \
  --expected-sha256 baseline.query_cache="$QUERY_SHA" \
  --expected-sha256 variant.query_cache="$QUERY_SHA" \
  --output "$CONTRACT/mapping_pose_gate_v2.json"

python -m scripts.pair_fullchain_workspace manifest \
  --root "$RUN" --stage mapping_pose_q256x3_v2 \
  --artifact seed2026="$RUN/mapping_pose_q256_v2/seed2026/mapping_cache_summary.json" \
  --artifact seed2027="$RUN/mapping_pose_q256_v2/seed2027/mapping_cache_summary.json" \
  --artifact seed2028="$RUN/mapping_pose_q256_v2/seed2028/mapping_cache_summary.json" \
  --artifact control_seed2026="$RUN/baseline_mapping_pose_q256_v2/seed2026/mapping_cache_summary.json" \
  --artifact control_seed2027="$RUN/baseline_mapping_pose_q256_v2/seed2027/mapping_cache_summary.json" \
  --artifact control_seed2028="$RUN/baseline_mapping_pose_q256_v2/seed2028/mapping_cache_summary.json" \
  --artifact mapping_pose_gate="$CONTRACT/mapping_pose_gate_v2.json" \
  --parent-manifest "$CONTRACT/metric.json" \
  --output "$CONTRACT/pose_v2.json"
```

`compare_mapping_pose_gate` is CPU-only. It does not load the multi-GB query
cache with Torch: it computes a streaming SHA-256, memoized when both arms name
the same resolved path. It loads the two maps, metrics, and teachers on CPU to
require exact Map/metric anchor-ID alignment, exact teacher query-name order,
and equal hashes for both the uniform q256 indices and the selected query names.
It also requires both teacher and calibration contracts to bind the same frozen
query-cache path/SHA, and requires the calibration statistics, parameters, and
policy to be numerically identical. Each version-2 evaluation summary embeds
the actual solver seed; all five input paths and SHA-256 digests; the ordered
teacher registry hash; and the actual selected query indices, index hash, and
name hash. The comparator requires every embedded binding to equal the live
artifact and the recomputed q256 subset. Consequently, old version-1 summaries
must be replayed and cannot authorize this gate. The six summaries must be
distinct files, and their embedded seeds must agree with the explicit
`SEED=PATH` CLI bindings rather than directory names. Every input path and
actual SHA-256 is recorded. They also embed the clean evaluation Git commit and
the SHA-256 of both the evaluator and comparator entrypoints. The comparator
recomputes and requires this identity, so the upstream training commit locked by
`contracts/code.json` (currently `3400516...`) remains explicitly distinct from
the later evaluation-code commit. Run all six V2 replays and the comparator from
that same clean evaluation commit. Optional
`--expected-sha256 ARM.ROLE=SHA256` arguments fail closed on known-digest
mismatches; roles include `map`, `metric`, `teacher`, `query_cache`,
`calibration`, and `seed2026_summary` (similarly for the other seeds).

The `_v2` directories are mandatory. Keep the existing version-1 unattested
summaries untouched; neither overwrite nor promote them into the V2 manifest.

The default structured threshold contract is preregistered as follows:

| scope | metric | required variant result |
|---|---|---|
| each seed | raw GT precision | baseline minus at most 0.005 percentage points |
| each seed | median/mean/p90/CVaR95 translation error | baseline plus at most `max(0.02 cm, 1%)` |
| each seed | median/mean/p90/p95 rotation error | baseline plus at most `max(0.02 deg, 1%)` |
| each seed | 5 cm / 5 deg recall | baseline minus at most 0.1 percentage points |
| each seed | >=100 cm catastrophes | no increase |
| three-seed mean, at least one | median or mean translation error | improve by at least 0.03 cm |
| three-seed mean, at least one | p90 or CVaR95 translation error | improve by at least 0.05 cm |
| three-seed mean, at least one | median/mean/p90/p95 rotation error | improve by at least 0.02 deg |
| three-seed mean, at least one | 5 cm / 5 deg recall | improve by at least 0.2 percentage points |
| three-seed mean, at least one | raw GT precision | improve by at least 0.01 percentage points |

A gate `PASS` requires every per-seed non-regression check and at least one
three-seed-mean substantive improvement. Safe-but-neutral is `STOP`, as is one
bad seed even when the overall mean improves. A custom complete threshold JSON
may be supplied with `--thresholds-json`, and the exact effective contract is
always embedded in the result; publication decisions use the defaults above.

This q256 gate is mapping-only and does not establish test accuracy. Only a
`PASS` authorizes an all-mapping refresh and then a separately preregistered
test evaluation. If the mechanism gains do not survive pose, stop this
pair-policy route despite the strong Track mechanism result.

## Fail-closed checklist

- Never use either `invalid_lineage_attempt` or
  `invalid_spatial_assignment_attempt`.
- No K=2048 cache, density config, changed descriptors, alias selector, surface
  factor, or old canonical/compact artifact may enter this run.
- No old canonical/function graph/provenance/teacher/metric is reused.
- Any failed run root is quarantined; do not delete individual artifacts and
  continue, because pipeline stages otherwise support ordinary silent resume.
- Every sidecar consumer must accept the exact variant binding; any mismatch
  raises and must never fall back to recalibration.
- `scene_calibration.json` remains numerically identical to V3 for statistics,
  parameters, and policy, while its sources/lineage bind the variant payload.
- Canonical and compact lineage audits must bind Track payload SHA
  `f1ab21fa...a059` through every evidence producer.
- All four stage manifests are reverified before pose. A changed path, size,
  SHA, invalid parent, or stale distillation calibration contract stops the run.
