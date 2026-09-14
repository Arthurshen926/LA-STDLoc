# V7 map-observation contamination audit

## Conclusion

The stronger V2 render-validity detector is useful and exposes a real omission
in the Full map construction, but a hard map-side mask is not the complete
solution.

Low-reliability Anchors are about twice as common among wrong real-query Top-1
matches as among correct matches.  Nevertheless, most wrong winners still have
high V2 mapping support.  Strictly retiring only the unquestionably polluted
Anchors has negligible central benefit and causes a severe confirmation-tail
regression.  Rebuilding descriptors from V2-valid mapping observations changes
real-domain behavior and repairs one illustrative tail substantially, but loses
two net R5 successes on the fixed non-test real panel.  No candidate is promoted.

The next module should therefore be a continuous Scene Reliability mechanism:
map-side observation/Anchor reliability plus query-side, map-conditioned row
reliability and descriptor-margin evidence.  It should not be implemented as a
single hard detector mask or as unconditional Gaussian deletion.

## Protocol and lineage

- Full map: 200,255 Projective Anchors, SHA-256
  `78e408cce366af8efea724ade5c35bebbf8f6edb02d62ecf40c2a4531364baed`.
- Mapping observation cache: 1,487 Gaussian-rendered mapping cameras, SHA-256
  `853739523e9bc7652c78dadf085c6199f2b17ec17e3e303027b1ecc4898c6149`.
- The V2 audit uses rendered mapping RGB, alpha, depth, distortion, and the
  exact 3,045,376 cached SuperPoint rows.  It does not redetect rows and
  therefore preserves the 2,385,977-row Anchor observation lineage exactly.
- No source mapping RGB or test RGB enters an Anchor descriptor.  Real mapping
  RGB is used only by a fixed 128-query evaluation panel.
- Test RGB is post-hoc diagnostic only and cannot select a threshold or map.

## What the old mask missed

The original V6 cache retained all 3,045,376 detector rows.  Its alpha mask had
99.9944% pixel support and removed zero rows.  Replaying the full V2 row
certificate retains 2,525,043 rows, or 82.9140%.

| V2 reason | Flagged cached rows |
|---|---:|
| Low RGB structure support | 371,659 |
| Image border | 83,458 |
| Depth discontinuity | 74,430 |
| Extreme distortion | 28,184 |
| Invalid alpha/positive-depth support | 0 |

Reason counts overlap.  The zero invalid-alpha/depth count is decisive: the
smear regions survive the old geometry mask precisely because they are opaque
and have positive rendered depth.  RGB appearance evidence is adding a new
signal rather than duplicating alpha/depth.

## Observation evidence lifted to Anchors

| Anchor class | Count | Fraction |
|---|---:|---:|
| All observations V2-valid | 121,006 | 60.43% |
| Zero V2-valid observations | 23,402 | 11.69% |
| Strict pure contamination: zero valid and at least two view families | 2,944 | 1.47% |
| Mixed contamination but safely reconstructable | 13,582 | 6.78% |
| At least 3 valid observations in at least 2 view families | 34,437 | 17.20% |

Of the 23,402 zero-valid Anchors, 20,458 are supported by only one coarse view
family.  They are suspicious, but the frozen safety rule deliberately does not
retire them without an independent family.  This explains why the strict
retirement arm is much smaller than the raw zero-valid set.

## False-winner enrichment on the read-only 530-query diagnostic

| Real Top-1 rows | Pure-contamination rate | Valid fraction below 25% | Mean valid fraction |
|---|---:|---:|---:|
| Correct | 0.460% | 4.124% | 0.915 |
| Wrong | 0.885% | 8.659% | 0.853 |
| Wrong rows in failed queries | 0.832% | 8.226% | 0.859 |

Pure contamination is 1.92 times as frequent among wrong matches as among
correct matches, and the broader low-reliability rate is about 2.10 times as
large.  This verifies association, but the absolute rates show that map-side
render invalidity explains only a minority of false winners.

For the named tail region, the post-hoc rows are:

| Query index / image | Baseline TE | Wrong Top-1 rows | Wrong rows below 25% valid | Strict-pure wrong rows |
|---|---:|---:|---:|---:|
| 135 / `frame00136` | 12.54cm | 1,970 | 10.81% | 19 |
| 136 / `frame00137` | 399.08cm | 1,989 | 10.11% | 23 |
| 139 / `frame00140` | 1,183.80cm | 1,994 | 8.38% | 11 |
| 140 / `frame00141` | 8,355.64cm | 1,997 | 8.31% | 9 |

The visually smeared region contributes bad Anchors, but most wrong rows in
these cases do not belong to the strict-pure subset.  This is why blanket
retirement cannot repair them.

## Fixed four-arm localization ablation

Descriptors in the reaggregation arms are computed only from original
V2-valid rendered mapping observations.  Updates are capped at 5 degrees and
no query descriptor is copied into the map.

### Non-test real mapping-RGB panel, 128 fixed queries

| Arm | Anchors | Median TE | P90 TE | R5 | Catastrophic >=50cm |
|---|---:|---:|---:|---:|---:|
| Baseline | 200,255 | 4.311cm | 27.019cm | 58.594% | 10 |
| Strict retire | 197,311 | 4.326cm | 28.423cm | 58.594% | 10 |
| Bounded reaggregate | 200,255 | 4.272cm | 25.866cm | 57.031% | 8 |
| Combined | 197,311 | 4.286cm | 25.850cm | 57.031% | 8 |

Reaggregation improves 71 queries and worsens 55, resolves two catastrophics,
and improves median/P90, but changes three baseline successes into failures
while recovering only one failure.  The 1.5625-point R5 regression blocks
promotion.  Strict retirement alone is effectively neutral centrally and
worse at P90.

### Independent 63-query V2-ACCEPT render confirmation

| Arm | Median TE | P90 TE | R5 | Catastrophic >=50cm |
|---|---:|---:|---:|---:|
| Baseline | 0.332cm | 0.892cm | 98.413% | 1 |
| Strict retire | 0.334cm | 0.890cm | 98.413% | 1 |
| Bounded reaggregate | 0.315cm | 0.900cm | 98.413% | 1 |
| Combined | 0.325cm | 0.877cm | 98.413% | 1 |

The count of catastrophics is unchanged, but strict retirement makes query 33
grow from 2,150.85cm to 28,049.63cm.  The combined arm inherits that failure.
For bounded reaggregation, 26 queries improve and 36 worsen, with a paired
median delta of +0.0034cm despite the aggregate median moving down.  It is not
a stable render-domain improvement.

## Named post-hoc cases

| Query index | Baseline | Strict retire | Bounded reaggregate | Combined |
|---|---:|---:|---:|---:|
| 135 | 12.54cm | 12.54cm | 12.85cm | 12.85cm |
| 136 | 399.08cm | 399.08cm | 24.34cm | 24.34cm |
| 139 | 1,183.80cm | 1,183.80cm | 1,183.80cm | 1,183.80cm |
| 140 | 8,355.64cm | 8,355.64cm | 7,715.99cm | 7,715.99cm |

The large #136 improvement comes entirely from descriptor reconstruction, not
Anchor retirement.  It remains outside the 5cm success gate.  #140 remains a
catastrophic failure.  These cases support descriptor contamination as one
mechanism, but disprove the stronger claim that removing the obvious smeared
Anchors is sufficient.

## Method decision

All three candidates are rejected and the Full map remains the active map.

The evidence supports the following next implementation, in order:

1. Keep V2 as the map-side observation reliability branch, producing a
   continuous weight rather than a hard mask.
2. Attribute feedback precision deficits to individual mapping observations;
   reweight only when harmful/positive evidence repeats across independent pose
   families.  Do not replace the existing descriptor with an unconditional
   mean of all V2-valid observations.
3. Add a real-query scene-specific reliability selector that combines RGB
   structure, Top-1 margin, map-Anchor reliability, spatial coverage, and pose
   conditioning.  Detector allocation and descriptor ranking must remain
   separate diagnostics.
4. Only after the cleaned-map mechanism passes fresh confirmation should a
   geometry-preserving real-to-clean-render photometric canonicalizer be tested.
   It must not learn to imitate Gaussian smear.
5. Gaussian removal remains a causal renderer-hygiene experiment.  Permanent
   updates should first be reversible Anchor/observation weights; no Gaussian
   is deleted by this audit.

## Artifacts

- Mapping audit report:
  `/mnt/pool/sqy/lafgs_v7_anchor_contamination_audit_20260827/StMarysChurch/aggregate_v2/report.json`
- Anchor evidence:
  `/mnt/pool/sqy/lafgs_v7_anchor_contamination_audit_20260827/StMarysChurch/aggregate_v2/anchor_contamination_evidence.pt`
- False-winner report:
  `/mnt/pool/sqy/lafgs_v7_anchor_contamination_audit_20260827/StMarysChurch/false_winner_report.json`
- Paired four-arm report:
  `/mnt/pool/sqy/lafgs_v7_anchor_contamination_audit_20260827/StMarysChurch/paired_ablation_report.json`
- Candidate maps are retained under `aggregate_v2/` for reproducibility but are
  explicitly marked `formal_method_selected: false`.
