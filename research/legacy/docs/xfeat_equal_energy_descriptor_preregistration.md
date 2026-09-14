# Stairs SP+XFeat equal-energy descriptor preregistration

## Hypothesis and method boundary

The XFeat Arm-B result established real descriptor headroom at the exact frozen
SuperPoint rows, but a native 64D replacement lost `0.02518 pp` Gaussian
Reserve R@1.  The next experiment therefore tests one smaller claim: whether a
single descriptor can retain SuperPoint's Reserve identity while adding
XFeat's Track identity signal.

For every query row and every fused observation bank row, independently
normalize the frozen 256D SuperPoint branch and locked 64D XFeat branch, then
materialize exactly

```text
z = concat(l2(SuperPoint), l2(XFeat)) / sqrt(2).
```

The result is one 320D unit vector, one map bank, one global Top-1, and one dot
product. Its score is exactly

```text
dot(z_query, z_map) = 0.5 * cosine(SuperPoint) + 0.5 * cosine(XFeat).
```

There is no learned fusion parameter, tunable alpha, anchor-type routing,
Track/Reserve-specific bank, candidate detector, topology change, or pose
feedback. This is intentionally a direct and paper-compatible representation
test rather than another branch or selector.

## Frozen inputs

- scene: 7Scenes Stairs mapping split only, 2,000 queries;
- query rows: fresh attested `K=1024/NMS=4` SuperPoint cache, SHA-256
  `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9`;
- map state: frozen V3 `anchor_map_step_1520.pt`, SHA-256
  `5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620`;
- complete-positive teacher: frozen V3, SHA-256
  `3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81`;
- XFeat probe: exact 2,048,000-row Arm-B artifact, SHA-256
  `fc538197bd103cfe9dfc4ef34109218e286d71b814815827a3828d356dc16a3a`;
- XFeat checkpoint: SHA-256
  `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`;
- positive edges, minimum two support views, eight temporal blocks, both
  support/gate directions, and R@1/2/4/8/16/32 remain identical to Arm B.

The evaluator must hash and bind the state, query cache, teacher, probe and
checkpoint. Evaluation and comparison must run from the same clean Git commit;
the report binds the exact evaluator module, runner, and comparator SHA-256.
It must report 320D map memory/MAC cost (exactly `1.25x` the 256D baseline for
equal rows) and may not mix detector Arm A into the report.

## Fixed mechanism gate

All five conditions must pass:

1. selection-to-gate pooled R@1 delta is strictly positive;
2. gate-to-selection pooled R@1 delta is strictly positive;
3. pooled R@8 delta is non-negative within only `1e-12` numeric tolerance;
4. pooled Track Core R@1 delta is non-negative within `1e-12`;
5. pooled Gaussian Reserve R@1 delta is non-negative within `1e-12`.

The gate writes `STOP` and exits 2 if any scientific condition fails. Input,
schema, lineage, dimension, support, split, row-count, or resource-contract
errors fail closed with exit 1 and are not scientific results.

A mechanism `GO` authorizes only a fresh mapping-only 320D descriptor/map
materialization followed by the existing three-seed q256 pose gate. It does not
authorize test evaluation. The later pose gate must preserve anchor IDs,
geometry, topology, query registry, one global Top-1 and one PoseLib call, and
must pass translation, rotation, recall and catastrophe non-regression. Only
then may the same descriptor be checked on 12Scenes `office2_5b`; an outdoor
guard follows before any formal test.

## Locked execution

```bash
PY=/root/miniconda3/envs/g4splat/bin/python
ROOT=/mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs
V3=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs

PYTHONPATH=/root/STDLoc "$PY" -m scripts.audit_frontend_upper_bound evaluate \
  --state "$V3/map_learning/anchor_map_step_1520.pt" \
  --query-cache /mnt/pool/sqy/lafgs_p7_density_factor_20260812/stairs/k1024_nms4/query_cache.pt \
  --teacher "$V3/map_learning/complete_positive_teacher.pt" \
  --probe-cache "$ROOT/xfeat64_descriptor.pt" \
  --arm descriptor_equal_energy \
  --crossfit-blocks 8 --minimum-support-views 2 \
  --topks 1 2 4 8 16 32 \
  --output "$ROOT/sp_xfeat_equal_energy_descriptor_report.json"
```

The fail-closed comparator uses
`--candidate-representation equal_energy_superpoint_candidate`, source
candidate dimension `64`, effective dimension `320`, and the exact report and
input SHA-256 values. The report must exist before its SHA is supplied; no
wildcard or latest-file resolution is allowed.

## Pose-before-deployment extension

The mechanism audit excluded anchors with fewer than two support views. The
frozen deployed map has 7,275 anchors, including exactly 70 single-view
anchors; all 70 are Gaussian Reserve (`anchor_type=0`). Removing or routing
them would change the topology and the scientific factor. Therefore the
versioned preregistration at
`docs/evidence/xfeat_equal_energy_deployment_extension_preregistration.json`
(SHA-256
`3db5001057173589f500adf4f05323993347bd4706832168270d181cb3e3f8f3`)
locks one uniform all-available-view estimator for every anchor with at least
one mapping view. It locks the exact 70-anchor row/ID hashes and type
histogram, and forbids fallback, routing, removal, or topology mutation.
The same manifest separately locks the teacher's topology anchor map
(`a39961d1...d614dc`) and the final trained V3 map. Materialization and the
live pair audit require every anchor ID, XYZ, type, primitive/cluster identity,
and dependency-group field to be bitwise equal; descriptor and training-only
fields may differ.

It also records the deliberate proxy transfer: the mechanism GO used the raw
normalized SuperPoint cross-fit branch, while deployment preserves the frozen
V3 metric/query and V3 anchor-feature branch. This is not claimed to be a
bitwise-identical SuperPoint bank. The extension is adjudicated first by the
Stairs uniform-q256, seeds 2026/2027/2028 pose/tail gate. A PASS only advances
to the mapping-only `office2_5b` tail guard, then an outdoor guard; formal test
data remain frozen.
