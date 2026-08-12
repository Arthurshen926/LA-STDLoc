# P7 fixed-budget camera-pair factor gate

## Scope

This gate changes only the mapping-camera pair graph.  It freezes the Stairs
K=1024 mapping query cache, descriptors, selector, density, Stage-A state,
Gaussian prior, triangulation thresholds, and provenance-assignment thresholds.
No test image is consumed.

The opt-in `parallax_diverse` policy keeps the exact nearest-6 global budget,
requires mapping-depth-sample field-of-view overlap, saturates parallax utility,
and diversifies relative pose.  `nearest` remains the default compatibility
policy.

## Track mechanism result

Both policies use exactly 7,450 camera pairs.  All values below are from the
same factor runner and the same exact 4,096-point mapping-only scene sample;
the 46.87% low-parallax control value must not be mixed with older audits that
used a different sample.

| Stairs K=1024 mapping metric | nearest control | parallax-diverse | change |
|---|---:|---:|---:|
| mapping-point pair parallax median | 1.029 deg | 2.888 deg | +180.6% |
| mapping-point pairs below 1 deg | 46.87% | 0.00% | -46.87 pp |
| actual triangulation-pair parallax median | 0.936 deg | 2.671 deg | +185.3% |
| raw reciprocal matches | 4,351,160 | 3,572,139 | -17.9% |
| final accepted matches | 1,760,615 | 1,230,772 | -30.1% |
| total Tracks | 105,782 | 81,535 | -22.9% |
| triangulated Tracks | 17,798 | 31,660 | +77.9% |
| broad eligible Tracks | 16,080 | 27,271 | +69.6% |
| strict eligible Tracks | 11,268 | 20,771 | +84.3% |
| high-confidence Tracks | 70 | 165 | +135.7% |
| triangulated covariance p90 | 0.06864 m2 | 0.03524 m2 | -48.7% |
| broad support/query p10 | 30 | 97 | +223.3% |

The mechanism interpretation is **fewer but more useful pair edges and
Tracks**, not “more Tracks.”  Raw and accepted correspondences fall, while
triangulatable, well-conditioned, broadly supported Tracks increase sharply.
Every preregistered mechanism gate passed.  This authorizes a lineage-correct
pipeline/pose experiment; it is not yet a pose-accuracy claim.

The per-pair sidecar records measured fields only: raw reciprocal matches,
descriptor/epipolar acceptance and rejection, ambiguity rejection, epipolar
recovery, cycle support, graph conflict rejection, final component-edge
contribution, mapping-point overlap/parallax, and actual triangulation
parallax.  Missing measurements remain unavailable rather than fabricated.

## Exact provenance replay

The original frozen Stairs control is reproduced before accepting the variant.
The control audit passes for all six assignment fields, all five Track fields,
all 16 frozen-common geometry fields, all provenance and query-support
diagnostics, the query registry, global primitive IDs, and the mapping-camera
name hash.  Every finite floating-point maximum difference is zero.

With the same frozen splat-provenance assignment, the variant changes:

| assignment metric | nearest control | parallax-diverse |
|---|---:|---:|
| valid provenance observations | 7,023 | 8,618 |
| assigned Tracks | 45 | 103 |
| assigned Gaussian landmarks | 91 | 240 |
| Track-to-landmark group edges | 97 | 248 |
| query-support edges | 10,671 | 11,007 |

The replay is fail-closed on Stage-A SHA, query-cache SHA, Gaussian PLY SHA and
primitive count, K=1024, mapping camera order, image sizes/intrinsics, all
assignment thresholds, and factor single-variable flags.  A spatial-nearest
fallback is forbidden.

## Artifact lineage

- Factor root: `/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/stairs/k1024`
- Mechanism gate: `mechanism_gate.json`
- Frozen-control parity: `provenance_replay/nearest_control_parity.json`
- Valid variant payload:
  `provenance_replay/parallax_diverse_track_micro_anchor_payload.pt`
- Valid variant payload SHA-256:
  `f1ab21fae713c37ac9725de8bf7eeacf8dcfa02ba91d45883dd789f04cb5a059`
- Variant lineage audit:
  `provenance_replay/parallax_diverse_payload_lineage_audit_v2.json`
  (`valid=true`, SHA-256 `55ecc5f23ab1e7064c16fa1db3f76a3129e6606d66e4138def15c9f9fd23cfe7`)
- Variant-bound frozen numeric calibration:
  `provenance_replay/pair_factor_frozen_scene_calibration.json`
  (SHA-256 `0a1631ceb5a29c024ae54b57f81537ebaad6c7e28f9fea9d541ee46ac41ff91a`)
- Frozen parent calibration SHA-256:
  `d3bc0839d73310055d93b895aa5c96fd633bef0fbab276604bffb782335120b2`
- Frozen bootstrap manifest SHA-256:
  `cb0241f694f6bdd5f4d663bd88273f94217d0a42502a5be97107b5c35a64b8aa`
- Frozen query cache SHA-256:
  `8f65f9ad067f40dd9bd7dda99f3b7674a3b9016b4679d29c0df8a54637d863d2`
- Frozen Stage-A SHA-256:
  `949a1b5bdff5f0d72628393b7f02dee526df4c8e0d104739c5562cb5fef19451`

Two earlier attempts are not scientific results:

- `invalid_lineage_attempt`: reused the old Track-identity graph/teacher.
- `invalid_spatial_assignment_attempt`: replaced splat provenance with XYZ
  proximity; its payload SHA `3597f81c...` is forbidden downstream.

The factor runner once contained a conditional proximity-transfer code path.
It has been removed.  A factor artifact is now explicitly Track evidence only
and never owns `assignment`; exact splat provenance is the sole legal
assignment producer.  The archived factors used for this gate do not contain
an assignment field.  Any historical factor that does is invalid and must
never be consumed.

## Current decision

The pair-policy mechanism is **GO** on Stairs.  The method-level conclusion is
still pending.  A valid next experiment must rebuild the canonical map,
canonical function graph, raster provenance, evidence graph, complete positive
teacher, compact map, compact graph/provenance/teacher, metric, and mapping pose
from the variant payload.  It may reuse only Track-independent frozen inputs.
Selector and evidence thresholds must be numerically frozen to V3 through a
variant-bound calibration contract; silent recalibration is a factor leak.
