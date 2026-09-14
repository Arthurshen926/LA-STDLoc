# Stairs XFeat Arm B descriptor-identity result

## Decision

The locked XFeat 64D descriptor is a scientifically valid **STOP before map
rebuild** on Stairs. It improves global identity retrieval in both temporal
cross-fit directions and improves pooled Track Core substantially, but fails
the preregistered exact non-regression check for Gaussian Reserve R@1 by
`-0.02518 pp`. The gate therefore does not authorize a descriptor map, metric
refresh, mapping pose, or formal test run.

This is a mapping-only result. It uses 2,000 mapping images, exactly 1,024
frozen SuperPoint rows per image (2,048,000 rows total), and no test query.

## Frozen inputs and artifacts

| Artifact | Path | SHA-256 |
|---|---|---|
| Fresh-cache equivalence V2 | `/mnt/pool/sqy/lafgs_frontend_ceiling_probe_20260813/stairs/contracts/fresh_cache_equivalence_v2.json` | `b600403c9d3e59f9ef68389dc7a5e321028889bea8515cae80e449c3603522ee` |
| XFeat Arm B probe | `/mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/xfeat64_descriptor.pt` | `fc538197bd103cfe9dfc4ef34109218e286d71b814815827a3828d356dc16a3a` |
| Descriptor report | `/mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/xfeat64_descriptor_report.json` | `77d793e3cc68d820ec5c4d7d78b7684f75a0c5814c977b8307e7511458c80472` |
| Fail-closed gate | `/mnt/pool/sqy/lafgs_xfeat_arm_b_20260813/stairs/descriptor_arm_b_gate.json` | `b1058d2594e6f7a2faab2e5f2e6f59a925feaee9dd1b1c546c9f0aefc7bf09fa` |

The V2 equivalence audit proves exact query order, Track inputs, effective
sparse depth, and native alpha for 2,000/2,000 queries between the legacy V3
cache and fresh `(K,NMS)=(1024,4)` cache. It binds source-cache SHA-256
`8f65f9ad...`, fresh-cache SHA-256 `6f2b5a73...`, and frozen Track-payload
SHA-256 `4e3a9c45...`; `valid=true` and Track-payload reuse is authorized.

The candidate checkpoint is the provenance-locked XFeat artifact with SHA-256
`0f5187fd...`. The probe cache is 544,004,619 bytes. The frozen state, fresh
query cache, and teacher are respectively 15,445,101, 10,380,381,469, and
60,777,421 bytes; these are input/storage costs, not accuracy gates.

## Identity result

| Mapping-only R@K | SuperPoint | XFeat | Delta |
|---|---:|---:|---:|
| selection -> gate R@1 | 43.53796% | 47.60631% | **+4.06835 pp** |
| gate -> selection R@1 | 40.28337% | 44.98298% | **+4.69960 pp** |
| pooled R@1 | 41.81647% | 46.21872% | **+4.40225 pp** |
| pooled R@8 | 70.42524% | 78.35424% | **+7.92901 pp** |
| pooled Track Core R@1 | 49.03098% | 55.28615% | **+6.25518 pp** |
| pooled Gaussian Reserve R@1 | 22.05318% | 22.02800% | **-0.02518 pp** |

Four gates pass: both directional R@1 deltas are strictly positive, pooled
R@8 is non-regressive, and pooled Track Core R@1 is non-regressive. The fifth
gate requires Reserve R@1 delta `>= -1e-12`; the observed
`-0.0002518331975409349` fails it. This exact threshold was registered before
the result and is not rounded to hide the failure.

XFeat is cheaper in this isolated ranking audit because its native descriptor
is 64D versus 256D: bank memory and dot-product MACs are both 0.25x. Across the
two directions its measured CPU matrix-multiply-plus-top-K time is
7.70/7.86 s versus SuperPoint 12.71/13.54 s (0.606x/0.581x). These timings do
not include extraction or end-to-end localization and are not accuracy gates.

## Interpretation

The result disproves the stronger claim that frozen SuperPoint has no
descriptor headroom: an independent 64D representation materially improves
Track identities and pooled R@8 at exactly the same image rows. It does not
show a deployable replacement. Reserve identities remain the fragile domain,
and the locked single-global-bank substitution does not preserve their top-1
rate exactly. The correct method conclusion is therefore narrower: frontend
descriptor quality is a real lever, while one uniform XFeat replacement is
not yet a complete Track-plus-Reserve solution.

The gate was independently re-executed against the exact paths and SHA-256
values. It exited 2 (scientific STOP) and reproduced the persisted gate
byte-for-byte. No map/function graph, metric, mapping-pose, or test artifact
was produced.
