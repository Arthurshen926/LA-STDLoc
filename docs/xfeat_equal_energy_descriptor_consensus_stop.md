# Equal-energy descriptor consensus stopping audit

## Decision

The fixed `0.5 * SuperPoint + 0.5 * XFeat` descriptor remains evidence of
row-level complementarity, but the descriptor-fusion line is now a formal
**STOP**.  A query-policy oracle proves that the useful and harmful cases are
separable in principle.  Fixed retrieval-before-image policies do not separate
them safely across all held-out Stairs sequences, and a support-only conformal
fallback removes almost all of the useful headroom.

This is a CPU-only analysis of the already frozen mapping q256x3 postmortem. It
uses no test query, GPU, new descriptor factor, topology edit, pose policy,
Office2/5b run, outdoor run, or weight/threshold search.  P8 is an independent
map-reconstruction factor and is not treated as a remedy for this result.

## Locked evidence

The audit binds the valid formal pose STOP, the hardened postmortem, its row and
PoseLib sidecar, and the exact mapping-only candidate cache/map/teacher.

| Artifact | SHA-256 |
| --- | --- |
| Formal equal-energy pose gate | `5ef1a156e9e2ac54fa0e111a246865269510ee4971d685f0dac52a798e03f947` |
| Hardened postmortem report | `fbc0ea6a3251aca88dc499efe9324ad7a16609c295ee9d873b7f3493dd22998f` |
| Reconstructed row/pose sidecar | `03338ef1cec6fb45b74d4a6be09f73bd5a5087168e61093ca0779baeb2051711` |
| Equal-energy query cache | `8e025d55e30ebcc8c1be90cc34c47dce7faecba20d405b7339f45f01a37df4e5` |
| Equal-energy map | `9a31e6c48ef0624f20236e61a32e7874d912f1786737308a5508b99be8c8c293` |
| Equal-energy teacher | `439cdbd9f974be4f4831d41188ca9153197c7cc0b347aefe964de438b21ab67b` |
| Audit report | `d20df4217d6b50c204763bd02ec3bad6c494c122a4d4d80d451073cb24b423c4` |

The audit report is checked in at
`docs/evidence/xfeat_equal_energy_descriptor_consensus_stop.json`; its external
materialization is
`/mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/descriptor_consensus_stop_audit.json`.
Every source is fail-closed by path and expected SHA.  Both the formal gate and
postmortem must say `uses_test_queries=false`, and the q256 registry must match
the NPZ sidecar exactly.

## What was tested

For each query, the existing three-seed PoseLib loss defines the frozen
advantage

```text
Delta_q = risk_q(A1) - risk_q(equal-energy).
```

Positive `Delta_q` means the already materialized 50/50 descriptor is better.
No new alpha is evaluated.  A policy may choose only `alpha=0` (exact A1
ranking) or `alpha=0.5` (the frozen 320D equal-energy ranking).

The policy sees 19 fixed statistics available before retrieval:

- five distribution statistics for A1-metric SuperPoint rows;
- five for XFeat rows;
- five for the cached dense SuperPoint feature map;
- mean, standard deviation, P10, and P90 of keypoint scores.

The four Stairs sequence prefixes are the outer folds.  Standardization and
fitting use only the other three sequences.  The fixed models are:

1. weighted L2 logistic classification of `sign(Delta_q)`, alpha `1`;
2. L2 ridge regression of continuous `Delta_q`, alpha `1`;
3. the same ridge policy enabled only when a nested support-only, one-sided
   95% conformal lower bound is above zero.

All decision thresholds are exactly zero.  There is no class threshold,
confidence threshold, feature, regularization, fusion-weight, or pose-metric
search.

As a separate parameter-free check, the audit chooses only between the two
already frozen winners using

```text
min(SuperPoint cosine, XFeat cosine).
```

This is explicitly a restricted two-winner audit, not a claim about global
min-score retrieval.  Its selected assignments receive one standard PoseLib
solve at the locked seed 2026.

## Results

The two-policy oracle has large, statistically stable Stairs headroom:

| Result | A1 | Fixed 50/50 | Oracle |
| --- | ---: | ---: | ---: |
| Policy risk | 0.301516 | 0.320508 | **0.263487** |
| Mean TE | 1.1398 cm | 1.5264 cm | **0.9611 cm** |
| P90 TE | 2.0927 cm | 2.2174 cm | **1.8736 cm** |
| CVaR95 TE | 4.1022 cm | 11.6978 cm | **3.2472 cm** |
| 5 cm / 5 deg recall | 98.4375% | 97.6563% | **99.6094%** |
| Mean hypotheses | 2253.7 | 1600.4 | **1818.7** |

The oracle chooses A1 for 120 queries and equal-energy for 136.  Its absolute
risk headroom is `0.03803`, with paired-bootstrap 95% interval
`[0.02825, 0.04649]`, or `12.61%` relative to the best fixed policy.  Thus the
50/50 descriptor is not uniformly bad; its benefit is query-dependent.

That necessary condition is not sufficient for a deployable gate:

| Fixed LOSO policy | Enabled | Risk | Oracle recovered | Outer folds | Mean / P90 / CVaR95 TE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic sign | 182/256 | 0.302896 | -3.63% | 2/4 | 1.2947 / 2.0684 / 7.7371 cm |
| Ridge advantage | 147/256 | **0.286049** | **40.67%** | **3/4** | **1.0734 / 1.9875 / 3.7706 cm** |
| Conformal lower bound | 5/256 | 0.299804 | 4.50% | 4/4 | 1.1322 / 2.0927 / 3.9714 cm |

Continuous advantage contains more useful signal than advantage-sign
classification.  The ridge policy also preserves `+0.34370 pp` raw precision,
improves 5 cm recall to `98.8281%`, and lowers mean hypotheses to `1795.8`.
However, its held-out `seq-03` fold regresses:

| seq-03 | A1 | Ridge gate |
| --- | ---: | ---: |
| Risk | **0.209099** | 0.210955 |
| Mean TE | **0.7611 cm** | 0.7814 cm |
| P90 TE | **1.1997 cm** | 1.4049 cm |
| CVaR95 TE | **1.6896 cm** | 1.8530 cm |

This fails the preregistered `4/4` outer-sequence non-regression condition.
Making the decision support-only conservative does pass all four sequences,
but enables only five queries and recovers `4.50%` of the oracle, below the
fixed one-third minimum.  This is a safe fallback, not an accuracy method.

The agreement alternative also fails.  It changes only 41,448 of 262,144 rows
(`15.81%`, versus `43.68%` for fixed 50/50) and retains `+0.23499 pp` raw
precision.  Yet seed 2026 mean/CVaR95/recall change from
`1.1403 cm / 4.1207 cm / 98.4375%` to
`1.2162 cm / 6.1438 cm / 98.0469%`; one query still reaches `30.41 cm`.
Agreement between marginal experts therefore does not certify a correct
query-level geometric consensus.

## Why one-vector deployment is feasible but not licensed

The STOP is empirical, not an implementation limitation.  Let unit vectors
`b` and `x` be the A1 and XFeat branches, and let an image-shared hard decision
`g_q` be zero or one.  A single 321D map/query representation could be

```text
m_i    = [b_i, x_i, 0] / sqrt(2)
z_{qu} = [b_{qu}, g_q x_{qu}, sqrt(1 - g_q^2)] / sqrt(2).
```

Both are unit vectors and

```text
dot(z_{qu}, m_i) = (dot(b_{qu}, b_i) + g_q dot(x_{qu}, x_i)) / 2.
```

For `g_q=0`, the global ordering is exactly A1; for `g_q=1`, it is exactly the
rejected equal-energy score.  It would therefore remain one vector bank, one
GEMM/global Top-1, and one PoseLib call.  The audit shows that the missing part
is a cross-trajectory-safe decision signal, not vectorization.  The 321D form
is explanatory only and must not be implemented as a production branch.

## Final consequence

Do not implement another classifier, tune the ridge/conformal cutoff, scan
alpha, or expand this factor to Office2/5b, outdoors, or formal test.  The
descriptor conclusion is now:

> XFeat carries real complementary row identity, but none of the tested
> mapping-only, retrieval-before-query policies turns that signal into a
> cross-trajectory-safe one-vector pose improvement.

Frozen V3/A1 remains the descriptor default.  Any future reopening requires a
new source of necessary information, not another estimator over the same 19
statistics.  P8 continues on its own preregistered map-evidence path.

## Reproduction

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/g4splat/bin/python \
  -m scripts.audit_equal_energy_descriptor_consensus \
  --formal-pose-gate /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/mapping_pose_gate.json \
  --expected-formal-pose-gate-sha256 5ef1a156e9e2ac54fa0e111a246865269510ee4971d685f0dac52a798e03f947 \
  --postmortem-report /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/postmortem_cpu_q256x3/postmortem_report.json \
  --expected-postmortem-report-sha256 fbc0ea6a3251aca88dc499efe9324ad7a16609c295ee9d873b7f3493dd22998f \
  --sidecar /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/postmortem_cpu_q256x3/paired_row_pose_sidecar.npz \
  --expected-sidecar-sha256 03338ef1cec6fb45b74d4a6be09f73bd5a5087168e61093ca0779baeb2051711 \
  --candidate-query-cache /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/factor/query_cache_equal_energy_320d.pt \
  --expected-candidate-query-cache-sha256 8e025d55e30ebcc8c1be90cc34c47dce7faecba20d405b7339f45f01a37df4e5 \
  --candidate-map /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/factor/anchor_map_equal_energy_320d.pt \
  --expected-candidate-map-sha256 9a31e6c48ef0624f20236e61a32e7874d912f1786737308a5508b99be8c8c293 \
  --candidate-teacher /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/factor/complete_positive_teacher_equal_energy_320d.pt \
  --expected-candidate-teacher-sha256 439cdbd9f974be4f4831d41188ca9153197c7cc0b347aefe964de438b21ab67b \
  --output /mnt/pool/sqy/lafgs_xfeat_equal_energy_descriptor_fullchain_20260813/stairs/descriptor_consensus_stop_audit.json
```
