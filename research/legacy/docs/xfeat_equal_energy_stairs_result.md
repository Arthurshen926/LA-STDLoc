# Stairs SP+XFeat equal-energy descriptor result

## Decision

The preregistered equal-energy descriptor is a mapping-only identity mechanism
**GO**, but its downstream three-seed q256 mapping-pose gate is a formal
**STOP**. All five identity gates pass and justified a fresh 320D factor; all
three pose seeds then fail the preregistered non-regression contract. It is
therefore not the default representation and does not advance to
`office2_5b`, an outdoor guard, or formal test.

The representation remains one vector and one index:

```text
z = concat(l2(SuperPoint-256D), l2(XFeat-64D)) / sqrt(2)
dot(zq, zm) = 0.5*cosine(SuperPoint) + 0.5*cosine(XFeat)
```

It has no learned fusion weight, evidence-type routing, dual bank, candidate
detector, topology change, or pose feedback.

## Locked evidence

| Artifact | SHA-256 |
|---|---|
| 2,048,000-row XFeat probe | `fc538197bd103cfe9dfc4ef34109218e286d71b814815827a3828d356dc16a3a` |
| Equal-energy audit report | `6dbc8394da5c6a498d5648ddbe474f71b82c2baba91eb0ed05e2193dffc63aaa` |
| Fail-closed mechanism gate | `19e6e5797840943bf7d4a0144f1b4cc1b99e2f895dea6437c9e529cc8cc8a623` |
| Frozen V3 state | `5f754ace648336d9f1fca381f29cd7f6164a217ca05b506644f21929e4a9e620` |
| Fresh K1024/NMS4 query cache | `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9` |
| Complete-positive teacher | `3f733debc51aafb7d166ebfb64010de237e3e7542851e647a7a2966f7c609a81` |
| XFeat weights | `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b` |

The report binds clean evaluator commit
`7c4c40bd5d8dfb08ee4b3d4f54a5e36f8a7edcb3` and the exact evaluator,
runner, and comparator file hashes. It contains no detector Arm A result and
attests `mapping_only=true`, `uses_test_queries=false`.

## Result

| Cross-fit identity metric | SuperPoint | Equal-energy 320D | Delta |
|---|---:|---:|---:|
| selection -> gate R@1 | 43.53796% | 48.58052% | **+5.04256 pp** |
| gate -> selection R@1 | 40.28337% | 45.33812% | **+5.05474 pp** |
| pooled R@1 | 41.81647% | 46.86548% | **+5.04901 pp** |
| pooled R@8 | 70.42524% | 77.14637% | **+6.72114 pp** |
| pooled Track Core R@1 | 49.03098% | 55.25992% | **+6.22895 pp** |
| pooled Gaussian Reserve R@1 | 22.05318% | 24.05303% | **+1.99985 pp** |

All five preregistered conditions pass: both directional R@1 deltas are
strictly positive; pooled R@8, Track R@1, and Reserve R@1 are non-regressive.
The candidate bank and cosine MAC count are exactly 1.25x the 256D baseline.

The native XFeat replacement had improved Track R@1 but regressed Reserve by
0.02518 pp. The equal-energy result therefore validates the intended mechanism:
SuperPoint and XFeat contain complementary identity evidence, and a fixed
single-vector composition preserves surface identity while adding Track
identity. It does not justify type-dependent routing or a tunable mixture.

## Downstream pose adjudication

The authorized factor was materialized without fallback while preserving all
7,275 anchor IDs, geometry, types, topology, the 2,000-query mapping registry,
one global Top-1, and one PoseLib call. Its formal q256x3 pose gate is valid and
uses no test query, but stops because every seed regresses mean/CVaR95
translation and 5 cm / 5 deg recall. Three-seed mean raw precision improves by
`+0.70457 pp` and median rotation by `-0.02652 deg`, while mean translation
worsens by `+0.38664 cm`, CVaR95 by `+7.59561 cm`, and recall by
`-0.78125 pp`.

The complete factor lineage, six replay hashes, metrics, and decision are in
`docs/xfeat_equal_energy_stairs_pose_result.md`; machine evidence is in
`docs/evidence/xfeat_equal_energy_stairs_pose_gate.json`. This pose Stop ends
the locked 50/50 deployment line. Formal test remains frozen.
