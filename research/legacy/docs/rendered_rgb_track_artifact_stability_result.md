# Rendered-Track Artifact Stability R1 Result

## Decision

R1 is a valid mapping-only **STOP before test and before R2**. The raw/clean
2DGS signal is informative, but multiplying it into every observation's
descriptor-fusion reliability is not cross-scene safe:

- ShopFacade passes all four preregistered mapping gates and greatly reduces
  the magnitude of its two existing tail failures.
- Stairs improves P90 translation and 5 cm recall without adding a new
  catastrophic query, but fails the CVaR95 and raw-precision guards.

No R1 test query was evaluated. Artifact-aware Track identity, KCS/GWFF,
LOO-A1, lazy completion, and default-map changes remain unauthorized.

## What changed

Relative to frozen V1.4 R0, R1 changes only the observation weight used to
fuse the selected Track descriptor:

```text
w_R1 = w_V1.4 * geometric_mean(
    raw/clean descriptor cosine,
    detector-score stability,
    clean-peak position stability,
    1 - 2DGS distortion exposure,
)
```

The pair graph, keypoint rows, Track components, ray-triangulated xyz,
selector membership, row order, map size, query descriptors, and identity
metric remain fixed. ShopFacade has 5,788 anchors and Stairs has 5,811.

The valid materializer uses `render_mode="RGB"`; alpha and distortion are
independent 2DGS buffers. Earlier attempts that used an incompatible
RGB+expected-depth background shape, lacked the JIT backend, or stopped after
artifact writes but before the large-tensor report were execution-invalid and
are not scientific evidence.

## Exact cache and calibration audit

The R1 cache changes `native_appearance_reliability`, the source tag, and six
artifact annotations. A new independent audit verifies every other shared
query field bitwise, including keypoints, descriptors, scores, intrinsics,
pose, valid mask, alpha, depth, dense rendered planes, query order, and Track
registry. Both scenes passed:

| Scene | Exact queries | Equivalence report SHA-256 | Rebound calibration SHA-256 |
|---|---:|---|---|
| ShopFacade | 231/231 | `119660427a9f9289e868471cd2ac0ae89c15243791bfcab1e5f3e09d7802f33a` | `c3af4efc06a20a927070e6ac990a27eb9f6d2d13d5542ffe30ccdbbd8fc9b04f` |
| Stairs | 2000/2000 | `74f6dd43b7b49ca1fa198f95576e278a23c8938a71e31d4cc8addc10ed4bf168` | `bb06173e992c9752102485e782f29a596898ca0b12f0605ad327a128a567ad01` |

The rebound copies the original mapping-only `statistics`, `parameters`, and
`policy` exactly; it does not estimate a threshold from R1 outcomes.

## Attribution of the V1.4 operating point

The requested C0/C1/C2 audit confirms that V1.4 is a combined operating-point
improvement, not an effect that can be attributed to removing cross-fit alone.
All arms use identity metric and full-mapping query-local LOO.

| Scene / arm | Anchors | Median TE cm | P90 TE cm | Mean TE cm | CVaR95 cm | Raw precision | 5 cm recall | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Shop C0: V1.2 membership | 2,504 | 0.464 | 0.950 | 82.990 | 1588.852 | 15.861% | 99.134% | 2 |
| Shop C1: V1.4 full child | 5,788 | **0.342** | **0.741** | 45.272 | 864.867 | **24.968%** | 99.134% | 2 |
| Shop C2: V1.4 prefix, budget 2,504 | 2,504 | 0.542 | 1.335 | **12.943** | **238.519** | 14.549% | 99.134% | 2 |
| Stairs C0: V1.2 membership | 4,256 | 0.386 | 1.055 | 6.742 | 126.361 | **14.361%** | 95.850% | 22 |
| Stairs C1: V1.4 full child | 5,811 | **0.363** | **0.960** | **6.103** | **114.274** | 7.279% | **96.600%** | 22 |
| Stairs C2: V1.4 prefix, budget 4,256 | 4,256 | 0.367 | 1.061 | 7.049 | 132.798 | 7.199% | 95.900% | 22 |

Stairs benefits consistently from the full repaired-child universe and
capacity. ShopFacade's typical metrics also favor C1, but severe-tail
magnitude is non-monotonic: the smaller C2 prefix has the best mean/CVaR while
retaining the same two catastrophic queries. Map size alone therefore does
not explain the full result.

## R1 mapping-only result

| Scene / arm | Median TE cm | P90 TE cm | Mean TE cm | CVaR95 cm | Raw precision | 5 cm recall | Catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|
| ShopFacade R0 | 0.342 | 0.741 | 45.272 | 864.867 | 24.968% | 99.134% | 2 |
| ShopFacade R1 | **0.332** | 0.749 | **8.898** | **164.754** | 24.745% | 99.134% | 2 |
| Stairs R0 | **0.363** | 0.960 | **6.103** | **114.274** | **7.279%** | 96.600% | 22 |
| Stairs R1 | 0.366 | **0.932** | 7.449 | 141.145 | 7.193% | **96.650%** | 22 |

ShopFacade passes all gates. Stairs fails exactly two:

- `cvar95_te_not_higher`: +26.871 cm;
- `raw_gt_precision_not_lower_by_more_than_0p05pp`: -0.0856 pp.

The Stairs catastrophic set is unchanged (the same 22 queries), but its
magnitudes change strongly. Query 474
`seq-02/frame-000474.color.png` moves from 372.54 cm to 3000.93 cm, while query
471 improves from 794.65 cm to 256.87 cm. Thus the global multiplicative
weight reallocates coherent false-consensus strength rather than monotonically
suppressing bad Tracks. Counting only catastrophic membership would miss this
failure; CVaR correctly catches it.

## Consequence

R1 establishes that raw/clean 2DGS stability carries useful evidence, but the
current scalar use is too coarse. It should not be promoted into Track
identity (R2), KCS support, or a learned metric as-is. Any follow-up must be a
new preregistered hypothesis that makes reliability observation-conditional
and tail-aware, keeps identity R0 as the control, and explicitly guards
coherent false consensus. The present run does not justify another threshold
search or a final test evaluation.

The authoritative gate is
`/mnt/pool/sqy/lafgs_render_track_only_artifact_r1_20260815/artifact_R1_mapping_gate.json`,
SHA-256 `b258a5df90acaf8ca53e10365ae5cff1df72ee480475d0c517605ab69a402335`,
with exit status 2 and empty stderr.
