# Stairs SP+XFeat equal-energy 320D mapping-pose result

## Decision

The fixed equal-energy descriptor remains a valid mapping-only identity
**mechanism GO**, but its preregistered downstream q256x3 mapping-pose gate is
a formal **STOP**. The gate is valid, all 15 fail-closed lineage checks pass,
and `uses_test_queries=false`; the scientific stop reason is
`PER_SEED_NON_REGRESSION_FAILED`.

This decision does not authorize deployment, `office2_5b`, an outdoor guard,
or formal test evaluation. The default method therefore keeps the frozen V3
SuperPoint descriptor path. The 320D factor remains evidence that the two
descriptors carry complementary identity information, not a deployable pose
improvement.

## Frozen factor and evaluation

The only changed factor is the descriptor representation:

```text
z = concat(l2(V3-metric(SuperPoint-256D)), l2(XFeat-64D)) / sqrt(2)
dot(zq, zm) = 0.5*cosine(V3-SuperPoint) + 0.5*cosine(XFeat)
```

Both arms use the same 7,275 anchor identities and geometry, the same 2,000
mapping-query registry, the same uniform 256-query indices, seeds
2026/2027/2028, one global Top-1 retrieval, and one PoseLib call. The factor
has no learned weight, type routing, dual bank, detector change, topology
change, fallback, anchor removal, or pose feedback. Evaluation code is bound
to clean commit `29797c9bd7df319661d57db1645bef5914245792`.

| Candidate artifact | SHA-256 |
|---|---|
| Descriptor factor contract | `81992f9f418cc64c1cfecaf61a4886ba7a36e6177aa0162d18a1b072fa7cda4f` |
| 320D anchor map | `9a31e6c48ef0624f20236e61a32e7874d912f1786737308a5508b99be8c8c293` |
| Strict-identity 320D metric | `5691305fe1d5ad704cca590f02355480cc19c0f3795efbd69e817e422f4779aa` |
| 320D query cache | `8e025d55e30ebcc8c1be90cc34c47dce7faecba20d405b7339f45f01a37df4e5` |
| Rebound complete-positive teacher | `439cdbd9f974be4f4831d41188ca9153197c7cc0b347aefe964de438b21ab67b` |
| Rebound scene calibration | `b331921383b166bc18b988f7e65a7edf7cb4ea0c8fd6e9d4baa61895841d8944` |
| Formal mapping-pose gate | `5ef1a156e9e2ac54fa0e111a246865269510ee4971d685f0dac52a798e03f947` |

The factor ID is
`2e45f50ab669fe1298f3002f48585b173b5acbe071a9801e6f0064905036224f`.
Direct re-hashing reproduced the contract, map, metric, cache, teacher,
calibration, gate, and all six summary hashes. The gate independently reloads
the factor pair and verifies registry, query selection, calibration, teacher,
strict metric, single-bank retrieval, and PoseLib parity.

| Seed | Baseline summary SHA-256 | 320D summary SHA-256 |
|---:|---|---|
| 2026 | `e2102e6bb209295ba3a6887a5e1c3605b295d8d9dcbc26adce46bf72142a1d56` | `8b0ca99856163bbcd58eb1c6604650348da883625c3add24a8a808d96aea34b0` |
| 2027 | `162b33d7d74f3f1887190e5da646bf788863cb849ac218bb00ce383688d49e09` | `8ca36fa5a2f44144ad13d5bf0ee7e5abf0b1a40850d7a27c26afd4755c0c4bac` |
| 2028 | `c91d8ac634eae320f0e24eb99963aaaba93d76844922e59b3f50f00f5c28c600` | `8b2e44bd78221849272ac482d15c170565671efbd218e05d35fed3d7276690da` |

## Pose result

Three-seed means show the split between identity ranking and pose utility:

| Mapping-pose metric | Frozen V3 | Equal-energy 320D | Delta |
|---|---:|---:|---:|
| raw GT precision | 6.46553% | 7.17010% | **+0.70457 pp** |
| median translation | 0.85935 cm | 0.88103 cm | +0.02169 cm |
| mean translation | 1.13978 cm | 1.52642 cm | **+0.38664 cm** |
| P90 translation | 2.10167 cm | 2.16559 cm | +0.06392 cm |
| CVaR95 translation | 4.10223 cm | 11.69784 cm | **+7.59561 cm** |
| median rotation | 0.25062 deg | 0.22410 deg | **-0.02652 deg** |
| mean rotation | 0.31751 deg | 0.34013 deg | +0.02261 deg |
| P90 rotation | 0.59483 deg | 0.60885 deg | +0.01401 deg |
| P95 rotation | 0.76629 deg | 0.73186 deg | **-0.03443 deg** |
| 5 cm / 5 deg recall | 98.43750% | 97.65625% | **-0.78125 pp** |
| catastrophic >100 cm count | 0 | 0 | 0 |

The raw-precision, median-rotation, and P95-rotation improvements satisfy the
three-seed substantive-signal rule, but substantive signal is not sufficient:
every seed fails the preregistered per-seed non-regression contract.

| Seed | Raw precision delta | Mean TE delta | P90 TE delta | CVaR95 TE delta | Median AE delta | Recall delta | Per-seed gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | +0.70457 pp | +0.38785 cm | +0.05106 cm | +7.58624 cm | -0.02739 deg | -0.78125 pp | **Fail** |
| 2027 | +0.70457 pp | +0.39405 cm | +0.12719 cm | +7.68163 cm | -0.02650 deg | -0.78125 pp | **Fail** |
| 2028 | +0.70457 pp | +0.37801 cm | +0.01350 cm | +7.51895 cm | -0.02568 deg | -0.78125 pp | **Fail** |

## First-principles conclusion

The experiment separates two questions that a pooled retrieval score cannot
collapse. Equal-energy composition improves frozen-row identity ranking and
some rotation statistics, so descriptor headroom is real. Yet the fixed pose
solver receives a worse correspondence set for translation robustness: mean
and CVaR95 translation regress in every seed and recall drops identically.
Because anchor registry, geometry, topology, query registry, calibration,
retrieval count, and pose code are fixed, the failure is localized to the
descriptor-ranking-to-pose interface. This result alone does not distinguish
spatial conditioning, ambiguity, or score calibration as the underlying cause.

Accordingly, the elegant one-vector representation is retained as a mechanism
finding, but the locked 50/50 deployment hypothesis ends here. Retuning its
weight, adding Track/Reserve routing, or searching scene-specific thresholds
would be new factors and is not licensed by this result. The next independent
mainline should test correspondence geometry/conditioning while the frozen V3
descriptor remains the deployment control.
