# Full-reference Stairs historical method revalidation

## Outcome

The public full-reference SfM initialization is the correct indoor Gaussian
prior contract, and the source-image-free V1.4 map remains the best bounded
operating point on the rebuilt 7Scenes/Stairs prior.  The new prior materially
improves the baseline, but it does not turn any of the recent ShopFacade-driven
enhancements into a better Stairs method.

This replay is deliberately not a new gate.  It fixes the rebuilt prior and the
previously implemented arms, uses mapping-only construction and LOO feedback,
and reads the official test split only after each deployable map is frozen.  No
threshold, rank, step count, map budget, or fusion rule was selected from test.
The exact lineage and result hashes are recorded in
[`docs/evidence/rendered_track_full_reference_history_revalidation.json`](evidence/rendered_track_full_reference_history_revalidation.json).

## The 12Scenes directory is complete

`/mnt/pool/sqy/12scenes` has four top-level apartment/office containers, not
four scenes.  Expanding the hierarchy gives all twelve scenes:

| container | scenes |
|---|---|
| `apt1` | `kitchen`, `living` |
| `apt2` | `bed`, `kitchen`, `living`, `luke` |
| `office1` | `gates362`, `gates381`, `lounge`, `manolis` |
| `office2` | `5a`, `5b` |

All 12/12 published `sfm_gt` reference-model directories contain nonempty
`cameras.bin`, `images.bin`, `points3D.bin`, and `list_test.txt`; the canonical
raw scene hierarchy also contains the corresponding RGB/depth/pose data.  All
12/12 prepared full-reference scene manifests exist.  Together they contain
16,989 mapping images, 5,782 test images, and the following reference point
counts:

| scene | mapping | test | reference points |
|---|---:|---:|---:|
| apt1/kitchen | 744 | 357 | 104,486 |
| apt1/living | 1,035 | 493 | 119,868 |
| apt2/bed | 890 | 244 | 170,537 |
| apt2/kitchen | 782 | 230 | 119,325 |
| apt2/living | 731 | 359 | 121,284 |
| apt2/luke | 1,370 | 624 | 140,385 |
| office1/gates362 | 3,540 | 386 | 418,626 |
| office1/gates381 | 2,950 | 1,053 | 470,507 |
| office1/lounge | 933 | 327 | 120,366 |
| office1/manolis | 1,623 | 807 | 272,973 |
| office2/5a | 1,000 | 497 | 201,753 |
| office2/5b | 1,391 | 405 | 580,447 |

The archive is intact; its SHA-256 is
`be40309b7e3ee5b8c0a663ddef08a3c304996578c70ff029d1549b07264c6043`.
There is therefore nothing to redownload.  This round does not train the other
indoor priors; it uses only the already rebuilt Stairs prior.

## Corrected Stairs operating point

The fixed prior is the 30k 2DGS build initialized from 131,766 published
reference points.  Its final PLY contains 321,622 Gaussians and has SHA-256
`467a397aa89d8f05c79212f36e61d28bd2fe267a3531b8c8ac1a75e53537c964`.
Only Gaussian-rendered mapping RGB participates in Track construction; source
mapping RGB and test queries do not.

Relative to the old retriangulated prior, V1.4 mapping mean/CVaR95 fall from
6.103/114.274 cm to 3.678/65.744 cm and catastrophic queries fall from 22 to
13.  Frozen three-seed test mean/P90 fall from 10.926/9.335 cm to
7.193/6.560 cm, while catastrophic queries fall from 15.67 to 9.67.  The old
Stairs prior was therefore a real limitation.  Prior-dependent negative
results had to be replayed rather than treated as immutable.

## Mapping-only replay

All arms use 2,000 mapping cameras, one global Top-1 match per query row, one
PoseLib call, seed 2026, and query-local descriptor LOO.  C2 and the full-broad
arm are attribution/oracle results, not deployable maps.

| arm | anchors | median cm | P90 cm | mean cm | CVaR95 cm | raw precision | 5 cm recall | catastrophes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rebuilt-prior V1.4 | 5,702 | 0.3637 | 0.9280 | **3.6779** | **65.7437** | 6.3264% | **98.55%** | 13 |
| C2 fixed 4,256 budget | 4,256 | **0.3575** | **0.8761** | 4.4420 | 81.2726 | 6.2610% | 98.15% | 13 |
| R1 scalar artifact weight | 5,702 | 0.3689 | 0.9297 | 4.1821 | 75.7844 | 6.2655% | 98.50% | 13 |
| conditional artifact fusion | 5,702 | 0.3617 | 0.9005 | 3.8510 | 69.2957 | 6.3332% | 98.50% | 13 |
| conditional + LOO-A1 | 5,702 | 0.3643 | 0.9125 | 4.0441 | 73.1491 | 6.3377% | 98.50% | 13 |
| all 15,210 broad Tracks oracle | 15,210 | 0.3909 | 1.0810 | 3.9309 | 70.0959 | **7.7037%** | 97.05% | 13 |

The smaller C2 map and conditional fusion can slightly improve central
quantiles, but both worsen mean/CVaR and recall.  LOO-A1 learns a small ranking
signal without suppressing coherent false poses.  The full pool sharply raises
raw precision and solver inlier ratio, yet worsens every translation quantile
and recall while leaving the catastrophic set size unchanged.  Candidate
capacity is not the limiting factor.

## Frozen real-test replay

R1, conditional fusion, and LOO-A1 were evaluated on all 1,000 official Stairs
test queries with seeds 2026/2027/2028 after their maps were frozen.  Values are
means over seeds.

| arm | median cm | P90 cm | mean cm | mean AE deg | 2 cm recall | 5 cm recall | raw precision | inlier precision | catastrophes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rebuilt-prior V1.4 | 2.1444 | 6.5601 | **7.1927** | **1.6217** | 46.97% | **85.10%** | **2.9868%** | **21.9854%** | 9.67 |
| R1 scalar artifact weight | **2.1141** | **6.5206** | 7.6439 | 1.7574 | **47.17%** | 85.23% | 2.9416% | 21.6626% | 11.00 |
| conditional artifact fusion | 2.1313 | 6.5343 | 7.4724 | 1.6751 | 46.67% | 84.93% | 2.9725% | 21.9397% | **9.33** |
| conditional + LOO-A1 | 2.1190 | 6.5442 | 7.6155 | 1.7025 | 46.93% | 84.60% | 2.9737% | 21.9373% | 10.33 |

These are not failures caused by an overly strict gate.  Each modification
offers a small typical-error benefit, but the aggregate mean, angular error,
precision, recall, or catastrophic tail regresses.  The test result confirms
the mapping diagnosis rather than reversing it.

## Reinterpretation of the ShopFacade results

- R1 remains valid evidence that raw/clean artifact stability is informative
  on ShopFacade: its old mapping mean/CVaR95 improved from 45.27/864.87 cm to
  8.90/164.75 cm.  On rebuilt Stairs, the same mechanism worsens both mapping
  and test tails.  It is scene-dependent evidence, not a default fusion rule.
- Conditional fusion is safer than global scalar weighting and preserved a
  small ShopFacade test benefit, but it still trades central Stairs accuracy
  for worse aggregate pose quality.  No threshold scan is justified.
- The bounded LOO-A1 residual metric is neutral on ShopFacade and harmful on
  both old- and new-prior Stairs.  More steps or rank are not authorized by the
  evidence.
- Full broad-Track completion exposed ShopFacade capacity headroom, but both
  Stairs priors reject simple capacity expansion.  The corrected prior makes
  the same causal conclusion stronger, not weaker.
- P8 and original mapping-RGB pair-selection conclusions are unaffected by the
  Gaussian rebuild because they do not consume rendered Track observations.

## Final method decision

Keep the full-reference-prior V1.4 source-image-free map as the current Stairs
baseline.  Do not merge C2 membership, R1 scalar weighting, conditional fusion,
the 175-step LOO-A1 metric, or full-pool completion into the default method.
Their implementations remain useful as auditable ablations.

The remaining accuracy gap is a structured correspondence/pose-consensus
problem: adding apparently good matches increases precision and inlier ratio
without improving pose.  A future method must change Track identity or robust
geometric consensus, and must be materially different from weight, metric-step,
budget, or completion tuning.  No additional indoor prior builds are required
to support this conclusion.
