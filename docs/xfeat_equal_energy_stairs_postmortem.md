# Stairs equal-energy descriptor pose postmortem

## Conclusion

The formal pose **STOP** is now explained more narrowly. The 320D descriptor
does not fail because it lacks identity signal, because the 7,275-anchor map
has bad global geometry, or because PoseLib is unstable across the three
preregistered seeds. The fixed 50/50 score changes `43.68%` of all Top-1
assignments and creates a localized, seed-stable **wrong geometric consensus**
on a small group of Stairs views.

The deployment conclusion is therefore:

- reject the fixed `0.5 * SuperPoint + 0.5 * XFeat` score;
- retain XFeat complementarity and the one-vector/one-bank representation as
  mechanism evidence;
- do not infer that a Track-only, Reserve-only, topology, or RANSAC patch will
  repair this factor;
- keep frozen V3 as the deployment control while P8 is tested independently.

No test query, GPU, new pose policy, topology change, parameter tuning, or
algorithm modification was used in this postmortem.

## Missing sidecar and exact reconstruction

The six original q256 mapping replays retained only aggregate JSON summaries.
The minimum evidence needed for a paired explanation was absent: per-query
Top-1 winners/correctness and per-seed PoseLib inlier row indices. A CPU-only
replay reconstructed those arrays under the same maps, metrics, teacher,
query indices, calibration, PoseLib parameters, and seeds.

| Artifact | SHA-256 |
|---|---|
| Full paired postmortem report | `fbc0ea6a3251aca88dc499efe9324ad7a16609c295ee9d873b7f3493dd22998f` |
| Reconstructed row/pose NPZ sidecar | `03338ef1cec6fb45b74d4a6be09f73bd5a5087168e61093ca0779baeb2051711` |
| Original formal pose gate | `5ef1a156e9e2ac54fa0e111a246865269510ee4971d685f0dac52a798e03f947` |

Candidate pose metrics reproduce the GPU gate exactly for all three seeds.
For the baseline, the only nonzero aggregate differences are mean TE and mean
AE, bounded by `3.67e-5`; the available aggregate artifacts do not identify
whether this comes from CPU/GPU matching arithmetic or another cross-device
numeric detail, so no narrower cause is claimed. All raw Top-1 counts and
decision-relevant behavior reproduce. The pair also re-verifies bitwise-equal
anchor xyz/types, teacher labels, and query geometry.

An independent review matched both arms and all six summaries to the formal
mapping-only gate, whose lineage checks are all true and whose
`uses_test_queries` field is false. The original full postmortem report predates
that fail-closed binding, so the audit script is now hardened to require the
formal gate and its expected SHA, exact artifact and summary hashes, the q256
subset, the ordered seeds, both rebound calibrations, and the fixed solver
protocol before a future replay can emit `uses_test_queries=false`.

## Why raw precision rises while pose worsens

Across `262,144` detector rows, equal-energy fusion changes `114,505` Top-1
winners (`43.6802%`). It converts `2,404` wrong/ambiguous rows to a legal
positive and loses `557` previously correct rows, for the observed net gain of
`1,847` correct rows (`+0.70457 pp`). The precision gain is real, but it
summarizes only marginal row identity; it does not constrain whether all rows
of one query support the same correct camera pose.

Seed 2026 makes the mismatch explicit (the other seeds differ by only a few
solver inliers):

| q256 total | Frozen V3 | Equal-energy | Delta |
|---|---:|---:|---:|
| PoseLib inliers | 97,308 | 113,811 | +16,503 |
| GT-clean inliers | 19,698 | 21,857 | +2,159 |
| GT-harmful inliers | 77,610 | 91,954 | **+14,344** |
| inlier GT precision | 20.243% | 19.205% | **-1.038 pp** |

Thus the candidate supplies a substantially larger RANSAC consensus, but
`86.9%` of the added inliers are GT-harmful. PoseLib is behaving consistently
with its input; the representation has made a wrong scene alias more
self-consistent.

This is not a global coverage loss. The legal positive set becomes larger and
on average has better image coverage and Fisher information. The tail arises
when the solver selects a different consensus mode whose GT-clean information
collapses to zero. Across queries, translation-error change correlates
strongly with clean translation-Fisher loss (`r` about `-0.93`, treating an
empty clean set as zero), but only weakly with raw correct-count change
(`r` about `-0.09`).

## The deterministic tail mode

Twelve of the candidate's thirteen CVaR95 queries are identical across all
three RANSAC seeds (`92.3%` intersection), so the tail is assignment-driven,
not random solver variation. Four queries dominate the regression:

| Query | Baseline TE | 320D TE | Correct-row delta | Clean inliers |
|---|---:|---:|---:|---:|
| `seq-05/frame-000239` | 1.47 cm | **27.67 cm** | +6 | 14 -> **0** |
| `seq-05/frame-000246` | 1.29 cm | **27.38 cm** | -1 | 26 -> **0** |
| `seq-05/frame-000254` | 0.70 cm | **30.27 cm** | +1 | 24 -> **0** |
| `seq-05/frame-000497` | 1.14 cm | **29.43 cm** | -2 | 16 -> **0** |

The first three are neighboring views in one Stairs trajectory. That
localization is consistent with a repeated-structure alias, but the current
full report and NPZ sidecar do not retain estimated pose matrices or a
nearest-mapping-camera lookup. The stronger attribution to a particular other
trajectory segment is therefore intentionally not made here. What the retained
evidence establishes is narrower: many assignments jointly support a larger
consensus that is geometrically wrong under the mapping ground truth.

The candidate recovers three former 5 cm failures (query indices `494`,
`1631`, `1999`) but creates five (`1239`, `1246`, `1254`, `1497`, `1638`),
giving the exact net recall loss of two queries (`-0.78125 pp`).

## Track versus Reserve

The effect is not isolated to one evidence type. Candidate winners shift by a
net `13,053` rows from Track to Reserve. Of the `14,344` added harmful inliers,
`9,809` (`68.4%`) are Reserve and `4,535` are Track, so Reserve aliasing is a
material contributor. However three of the four extreme false consensuses are
Track-majority while the fourth is Reserve-majority. This evidence rejects a
simple Track-shortage explanation and does not license type routing.

## First-principles next step

The causal mismatch is **marginal identity versus set-level pose consensus**.
Anchor xyz and topology are unchanged; the descriptor rewrites which 3D
identities co-occur in each query. A fixed equal-energy weight is therefore
not pose-calibrated and should not be tuned post hoc on this same q256 gate.

P8 attacks a related principle—cycle/Fisher support for a geometrically useful
correspondence set—but it is not the same causal factor. P8 changes which
anchors are retained; this failure occurs on a fixed map because descriptors
form a coherent false mode. P8 should first pass independently with frozen V3.
Only after that would a preregistered crossed experiment establish whether P8
actually suppresses, leaves unchanged, or amplifies the 320D alias mode.

If descriptor fusion is reopened later, the elegant target remains one vector
and one global bank, but its mapping-only calibration objective must include
query-level worst-tail pose consensus (or hard negatives drawn from coherent
wrong-pose cycles), not only pooled row R@1. That is a new hypothesis, not a
continuation of the rejected 0.5 factor.
