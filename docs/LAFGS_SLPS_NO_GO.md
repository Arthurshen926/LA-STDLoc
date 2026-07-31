# SLPS Decision Record

Date: 2026-07-31

## Decision

The learned SLPS correspondence ordering is a No-Go. The frozen LaFGS A1 map
remains the accuracy mainline, and the previously validated fixed P1 ordering
remains an optional efficiency point. SLPS must not write descriptors back to
the map or expand to additional scene-specific searches.

The full 103-query ShopFacade gate showed that only configurations which left
the old ordering nearly unchanged remained safe. A bounded 5% ordering change
regressed, while the adaptive risk model obtained safety by almost always
falling back to A1. The selector cost was also larger than the RANSAC savings.

## Correctness Audit

Negative `track_cluster_ids` mean unknown identity. They had been grouped as
one pseudo-track. The selector now treats each negative row as an independent
singleton for relation grouping, multiplicity, and Beta stability. The four
available frozen maps contained the following unknown-anchor fractions:

| Scene | Unknown anchors |
| --- | ---: |
| GreatCourt | 5,868 / 13,868 (42.31%) |
| KingsCollege | 345 / 8,345 (4.13%) |
| ShopFacade | 453 / 7,145 (6.34%) |
| StMarysChurch | 3,364 / 11,364 (29.60%) |

A 20-query ShopFacade parity audit changed median translation error from
1.6032 cm to 1.6191 cm, with unchanged R2 and R5. The bug is real and fixed,
but it does not reverse the learned-ordering decision.

## Final Allowed Selector Gate

Only a fixed P1 set risk certifier may be evaluated. It may consume normalized
pre-PnP set statistics, choose between the fixed compact set and A1, and must
fall back to A1 under uncertainty. It may not learn or alter row ordering.

The reproducible leave-one-scene-out evaluation is implemented by
`scripts/evaluate_lafgs_fixed_budget_certifier.py`. Its strict success gate is:

- finite-sample false-safe upper bound at most 2%;
- nonzero accepted queries in every held-out scene;
- at least 50% macro hypothesis reduction;
- less than 3 ms certifier overhead;
- no GT, inlier, pose, residual-history, or post-PnP input feature.

Failure of this gate permanently closes learned selector work. Runtime work
must then preserve the complete A1 correspondence set and PoseLib result.

## Fixed-Budget Certifier Result

The single allowed certifier experiment was completed on all four scenes with
consistent frozen A1/P1 artifacts: GreatCourt, KingsCollege, ShopFacade, and
StMarysChurch. OldHospital had no artifact satisfying the same frozen protocol
and was not silently substituted or presented as a fifth fold.

The leave-one-scene-out result is stored at
`/mnt/pool/sqy/stdloc_lafgs_slps_20260731/fixed_budget_certifier_loso_v1.json`.
The held-out macro AUC was 0.4888. No calibration fold could certify a nonempty
P1-512 acceptance set at the 2% one-sided risk limit, so every query fell back
to A1 and the hypothesis reduction was 0%. The 0% false-safe rate is therefore
vacuous rather than a successful efficiency result. This fails the nontrivial
acceptance and 50% hypothesis-reduction gates and permanently closes learned
selector and risk-certifier work under the present evidence contract.

## Exact Preemptive Verification Result

An independent C++ verifier reproduces PoseLib sampling, LO, and final bundle
adjustment while allowing only mathematically safe partial-MSAC rejection. It
never removes or reorders sampling correspondences, and complete scores retain
the original PoseLib summation order. Synthetic tests cover both ordinary and
progressive sampling.

The full 103-query ShopFacade test and its same-load native PoseLib control are
summarized at
`/mnt/pool/sqy/stdloc_lafgs_preemptive_20260731/ShopFacade/preemptive_exact_parity_summary.json`.
All 103 poses, inlier counts, RANSAC iteration counts, and refinement counts
were identical; the maximum pose-matrix difference was exactly zero. Partial
verification reduced residual evaluations by 18.02%, but mean RANSAC runtime
increased from 52.58 ms to 62.65 ms (+19.15%), and mean end-to-end runtime
increased by 6.32%. The exact inlier-trigger guard forces most rows to be
visited, so bookkeeping and verification-order overhead exceed the saved
arithmetic. The implementation is retained as a correctness-valid research
prototype but is not a deployable/default solver.

## Frozen Mainline

- Accuracy: A1 all-correspondence PoseLib.
- Optional compact efficiency point: previously validated fixed P1 ordering.
- Learned SLPS ordering: No-Go.
- Cross-scene fixed-budget risk certifier: No-Go under the strict risk gate.
- Exact preemptive verifier: parity passes, runtime gate fails.
- Descriptor writeback and failed-selector scene expansion remain prohibited.
