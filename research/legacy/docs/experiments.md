# Experiments

Formal reports must include median, mean, and P90 translation error; median and
mean rotation error; 2 cm/2 degree and 5 cm/5 degree recall; raw and RANSAC
inlier GT precision; final anchor count; and runtime/RANSAC iterations.

Prior robustness compares off-the-shelf vanilla 3DGS, vanilla 2DGS, AnySplat
feed-forward Gaussians, and an enhanced 2DGS reference. Only the prior changes.
The mapping images, mapping poses, feature frontend, candidate budget,
topology/A1 schedule, and one-shot PoseLib protocol remain fixed.

Historical ablations and no-go branches are not shipped in this branch. Their
complete source remains in Git tag `archive-full-research-20260803`.

## Repeated-assignment ceiling audit

Before changing the frozen frontend or adding a contextual descriptor, run the
mapping-only ambiguity audit on the trained compact map:

```bash
python scripts/audit_repeated_assignments.py \
  --map /data/run/map_learning/anchor_map_step_0175.pt \
  --metric-state /data/run/map_learning/metric_state_step_0175.pt \
  --complete-positive-teacher /data/run/map_learning/complete_positive_teacher.pt \
  --query-cache /data/run/bootstrap/query_cache.pt \
  --scene-calibration /data/run/map_learning/scene_calibration.json \
  --dataset /data/scene \
  --output /data/run/repeated_assignment_audit.json
```

The report evaluates exact shared-metric global ranking at top-1/2/4/8/16/32.
It records positive recall with both all-row and positive-eligible denominators,
Track Core versus Gaussian Reserve recall and winner breakdowns, exact
positive-to-wrong score margins, repeated false-attractor multiplicity, an
optional image-sharpness stratification, and mapping-only oracle-top-K PoseLib
headroom. The oracle changes only rows whose retrieved top-K contains a legal
positive and retains deployed top-1 for all other rows. It is a diagnostic
ceiling, not a deployable matcher and not a source of test-set model selection.

The first P0 audit used deterministic uniform mapping-view samples from the
indoor detector-density study. Stairs used 256 of 2,000 mapping views with 96
of them replayed through the oracle; Fire used 128 of 2,000 and 32 oracle
views. The denominator below is restricted to rows for which the active-map
teacher exposes at least one legal positive; the JSON report also retains the
all-detector-row denominator.

| Scene | R@1 | R@8 | R@16 | R@32 | Track R@1 / R@16 | Reserve R@1 / R@16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stairs | 48.02% | 72.21% | 78.46% | 83.22% | 70.05% / 90.15% | 17.52% / 63.66% |
| Fire | 53.47% | 82.33% | 87.47% | 90.89% | 58.28% / 86.28% | 17.60% / 86.60% |

On Stairs the exact best-positive minus best-wrong cosine margin has mean
`-0.0201` and median `0.0010`: many legal positives are present but nearly tied
with a wrong identity. Of the 21,795 labeled false top-1 rows, 54.5% already
contain a legal positive in top-16 and 64.2% in top-32. Gaussian Reserve
anchors form 73.2% of the map and win
43.7% of mapping rows, while their labeled-winner precision is only 41.0%.
Track Core positives rank much better, although the largest individual false
attractors are themselves tracks. This supports a shared query/map context
encoder plus reserve consistency control, rather than another reserve-only
descriptor patch.

| Stairs mapping oracle | Median TE | Mean TE | P90 TE | CVaR95 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deployed top-1 | 0.796 cm | 1.035 cm | 2.160 cm | 3.330 cm | 3,171.7 |
| Corrected top-32 | 0.665 cm | 0.889 cm | 1.877 cm | 2.842 cm | 2,879.1 |

The oracle improves all Stairs pose-risk statistics and reduces hypotheses,
so this sample is inconsistent with a solver-only failure. It is a direction
audit, not the context method gate: the current metric has seen all mapping
views. Any learned context baseline must therefore use disjoint alternating
trajectory blocks for descriptor construction/training and pose gating before
test is consulted.

As a cross-dataset sentinel, the same audit was run on 256 mapping views from
the legacy 12Scenes Office2/5b artifact, with 64 views used for oracle PnP and
its registered fixed 12-pixel gate. Positive-eligible R@1/R@16/R@32 is
62.56%/89.48%/91.72%; Track Core is 63.82%/90.41% at R@1/R@16, while Gaussian
Reserve is 28.33%/67.63%. The latter has only 34.0% labeled-winner precision.

| Office2/5b mapping oracle | Median TE | Mean TE | P90 TE | CVaR95 | Hypotheses | >100 cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deployed top-1 | 0.451 cm | 15.782 cm | 1.008 cm | 244.648 cm | 20,145.4 | 3 |
| Corrected top-32 | 0.378 cm | 10.895 cm | 0.959 cm | 167.984 cm | 15,556.0 | 2 |

The large mean/CVaR gap comes from a small catastrophic tail, and the top-32
oracle reduces both by about 31% while cutting hypotheses by 22.8%. Although
this older artifact is not an Adaptive V3 non-inferiority gate, it independently
shows that indoor tail risk contains descriptor-ranking headroom rather than
being explained only by PoseLib.

## Cross-fitted FeatureBooster direction oracle

The official `SuperPoint+Boost-F.pth` was then evaluated as a mapping-only
direction oracle. Each scene was split into alternating contiguous trajectory
blocks and run in both directions. A direction fused map descriptors only from
legal positive observations in its support fold and evaluated only the
disjoint gate fold. Raw SuperPoint and Boost-F used exactly the same observation
edges, view-balanced fusion, minimum two-view anchor support, global cosine
top-1, and one standard PoseLib call. Filtering anchors without support-fold
observations is a diagnostic necessity and is shared by both protocols; these
numbers are therefore a controlled context delta, not a comparison with the
complete deployed map.

```bash
python scripts/evaluate_context_booster_crossfit.py \
  --map /data/run/map_learning/anchor_map_step_1520.pt \
  --complete-positive-teacher /data/run/map_learning/complete_positive_teacher.pt \
  --query-cache /data/run/bootstrap/query_cache.pt \
  --featurebooster-weights ~/.cache/lafgs/SuperPoint+Boost-F.pth \
  --scene-calibration /data/run/map_learning/scene_calibration.json \
  --minimum-support-views 2 \
  --gate-query-count 256 \
  --pose-query-count 96 \
  --output /data/run/map_learning/context_booster_crossfit.json
```

All scenes replayed 256 gate images. Stairs and Fire used 96 PoseLib images;
the legacy Office2/5b artifact used 64 and its registered explicit 12-pixel
fallback. The table reports Boost-F minus its support-matched raw-SuperPoint
control. Negative pose and hypotheses deltas are improvements.

| Scene | R@1 raw -> boost | R@16 raw -> boost | Mean TE delta | P90 TE delta | CVaR95 delta | Hypotheses delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stairs | 32.00% -> 33.23% | 66.38% -> 68.21% | +5.2% | +7.5% | +3.8% | -15.8% |
| Fire | 34.78% -> 35.75% | 82.14% -> 83.59% | -3.3% | -7.5% | -8.8% | -0.6% |
| Office2/5b | 57.55% -> 61.18% | 87.24% -> 89.46% | +43.5% | -4.8% | +45.8% | -14.8% |

The ranking result is stable: Boost-F raises R@1 in all six directed folds,
including Track Core and Gaussian Reserve aggregates in every scene. It also
raises pooled R@16 by 1.83, 1.45, and 2.22 percentage points on Stairs, Fire,
and Office2/5b. This is sufficient evidence that single-image context contains
useful repeated-structure information and promotes the context representation
line over further geometry, topology, or solver patching.

The unbounded official embedding does not pass the pose-risk gate. On the 96
Stairs pose images it improves 42 and worsens 54; one previously 0.78 cm pose
moves to 27.25 cm. On Office2/5b, a 3.37 cm raw pose becomes 399.23 cm, raising
catastrophic failures from three to four and dominating mean/CVaR despite a
better P90. Fire improves in aggregate, but one of its two directions is
slightly pose-regressive. Therefore FeatureBooster itself is not promoted to
deployment and test images remain uninspected.

The next context model must be map-consistent and identity-preserving: a small
dense multi-scale/global-context head with an identity-initialized bounded
residual, trained with repeated-collision negatives, clean-win preservation,
cross-view consistency, and an explicit tail-risk gate. The current topology,
geometry, and one-shot PoseLib path remain frozen. This converts the result
into a precise method constraint: context is useful, while unrestricted
descriptor replacement is unsafe.

## Map-consistent contextual descriptor gate

The first MCCD implementation uses the frozen SuperPoint dense map from the
same detector forward as the sparse keypoints. It samples 3x3, 7x7, and 15x15
masked local averages plus one image-global token, then applies one
identity-initialized two-layer residual head. Query observations and every
mapping observation use the same head; landmark descriptors are fused only
after observation-space adaptation, with equal weight per observing image.
The deployed descriptor remains 256D and localization remains exact global
top-1 followed by one standard PoseLib call.

Training uses only mapping queries and has two stages: a collision/clean-win
list loss against a fixed support-only raw bank, followed by the same loss
against the rebuilt contextual bank. Both temporal-block directions use
identical support observations for raw and contextual maps. The full fit saves
the adapter state hash, base-map anchor IDs, supported anchor rows, context
configuration, and a `uses_test_queries=false` contract.

```bash
python scripts/train_evaluate_context_metric_crossfit.py \
  --map /data/run/map_learning/anchor_map_step_1520.pt \
  --complete-positive-teacher /data/run/map_learning/complete_positive_teacher.pt \
  --query-cache /data/run/bootstrap/query_cache.pt \
  --scene-calibration /data/run/map_learning/scene_calibration.json \
  --minimum-support-views 2 \
  --gate-query-count 256 \
  --pose-query-count 96 \
  --output /data/run/map_learning/mccd_crossfit.json
```

The original hard-clipped bounded residual passes the mapping-only context
gate on all three available indoor sentinels. Values below compare MCCD with
its observation-identical raw-SuperPoint control; negative pose deltas are
improvements.

| Scene | R@1 raw -> MCCD | R@16 raw -> MCCD | Mean TE delta | P90 TE delta | CVaR95 delta | Hypotheses delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stairs | 32.00% -> 33.49% | 66.38% -> 69.17% | -22.8% | -12.1% | -34.3% | -5.4% |
| Fire | 34.78% -> 35.89% | 82.14% -> 84.40% | -1.2% | +1.2% | -1.6% | -0.6% |
| Office2/5b | 57.55% -> 59.76% | 87.24% -> 88.87% | +0.3% | +2.3% | +0.0% | -11.7% |

Stairs ablations keep the same folds, support edges, gate rows, and pose rows.
The full local-plus-global context has the strongest tail interaction even
though local-only has a slightly higher R@1. A 0.05 residual is too
conservative.

| Stairs mapping variant | R@1 delta | Mean TE delta | P90 TE delta | CVaR95 delta |
| --- | ---: | ---: | ---: | ---: |
| Pointwise / zero context | +1.063 pp | -10.9% | -6.5% | -16.1% |
| Local only | +1.525 pp | -11.7% | -7.8% | -16.5% |
| Global only | +1.296 pp | -12.6% | -0.4% | -16.2% |
| Local + global | +1.490 pp | -22.8% | -12.1% | -34.3% |
| Local + global, residual 0.05 | +0.792 pp | -4.2% | -3.0% | -3.1% |

An implementation audit found that the first hard clamp made the residual
trust loss constant after the 0.10 boundary was reached. The retained adapter
therefore uses a smooth radial rational bound. It preserves unit derivative at
the identity, remains strictly below 0.10, and keeps a radial trust gradient.
On Stairs cross-fit its mean/max final residual is 0.0848/0.0992 instead of
0.1000/0.1000. It raises R@1 by 1.294 points and reduces mean/P90/CVaR95 by
13.3%/2.5%/17.8%; 52 of 96 paired poses improve. Legacy hard-clipped artifacts
carry an explicit compatibility path and are not silently reinterpreted.

After the mapping gate, the locked 2048-keypoint protocol was evaluated on all
1,000 Stairs test images. The first attempted run resolved the current adaptive
default to 1024 rows and is excluded; all rows below use the scene's materialized
`factor_config_k2048.yaml`, 12 pixels, seed 2026, global top-1, and one PoseLib
solve. CVaR95 is recomputed from the worst 5% paired test errors.

| Stairs test descriptor protocol | Median TE | Mean TE | P90 TE | CVaR95 | Raw P@2 | 2 cm recall | 5 cm recall | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing learned shared metric | 1.750 cm | 3.795 cm | 6.997 cm | 29.515 cm | 3.274% | 56.0% | 82.3% | 21,845 |
| Raw observation map, identity control | 2.183 cm | 3.919 cm | 6.229 cm | 27.783 cm | 2.769% | 42.4% | 85.7% | 24,852 |
| Raw-map MCCD, hard clip | 2.065 cm | 3.807 cm | 6.512 cm | 26.940 cm | 3.037% | 47.3% | 85.6% | 21,162 |
| Raw-map MCCD, smooth radial bound | 2.099 cm | 3.836 cm | 6.442 cm | 27.026 cm | 2.956% | 45.6% | 85.3% | 21,637 |

This isolates the remaining problem. Context is genuinely useful relative to
the raw observation map: it recovers 3.2--4.9 points of 2 cm recall, improves
raw precision, and reduces the catastrophic tail or RANSAC work. However, the
raw observation map itself discards 13.6 points of tight recall relative to the
existing learned shared-metric map. The current raw-map MCCD is therefore a
tail-robust Pareto branch, not the new default.

The next and only promoted descriptor line is a **metric-preserving contextual
residual uplift**. Query descriptors first pass through the frozen shared
metric. Mapping observations use the same frozen metric and context head, but
their robustly fused contextual residual is added to the existing learned
anchor descriptor. Thus zero initialization exactly reproduces the current A1
query and map descriptors, unsupported anchors remain unchanged, and the
context head can only supply a bounded observation-consistent correction. Its
artifact must pin both the metric state hash and context state hash. The
one-pass Track-First rebuild remains blocked until this stricter P2 variant
passes Stairs plus non-regression sentinels.

## Metric-preserving contextual uplift decision

The decisive follow-up removes the raw-map confound. It freezes the deployed
`SharedLowRankMetric` for every query observation and retains the learned A1
anchor descriptor as the map-side local expert. A context head predicts only a
unit-sphere tangent angle update. Mapping images run the same head, but only
their per-observation tangent updates are view-balanced and fused onto the
learned anchor. Cross-view residual-direction coherence is the map-side gate;
the query head supplies a sigmoid confidence gate. At zero initialization both
query and map descriptors reproduce A1 bitwise.

Training is also restricted by the first-principles failure decomposition. A1
clean winners receive margin preservation. A1 false winners receive collision
supervision only when a legal positive is already in A1 top-32. False rows
without a top-32 positive receive no ranking pressure, preventing the model
from inventing unsupported identities. The metric file hash, context state
hash, support rows, and learned anchor IDs are pinned in every fold artifact.

```bash
python scripts/train_evaluate_metric_context_uplift_crossfit.py \
  --map /data/run/map_learning/anchor_map_step_1520.pt \
  --metric-state /data/run/map_learning/metric_state_step_1520.pt \
  --complete-positive-teacher /data/run/map_learning/complete_positive_teacher.pt \
  --query-cache /data/run/bootstrap/query_cache.pt \
  --scene-calibration /data/run/map_learning/scene_calibration.json \
  --repair-topk 32 \
  --maximum-angle-rad 0.05 \
  --output /data/run/map_learning/metric_context_uplift_crossfit.json
```

On the two Stairs support directions, the training sets contain 51,665/49,821
recoverable false winners and 30,717/26,326 false winners outside top-32. The
latter are excluded from collision supervision. The table below uses the same
256 gate images and 96 PoseLib images for every row. Pose changes are relative
to the support-matched frozen A1 control; negative values are improvements.

| A1 uplift variant | R@1 delta | 2 cm recall delta | Mean TE | P90 TE | CVaR95 | Hypotheses | New false from A1-clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pointwise, context zero | +0.706 pp | -2.083 pp | -22.35% | +0.04% | -64.69% | -0.85% | 0.750% |
| Local dense context | +0.805 pp | 0.000 pp | -0.85% | -3.15% | +1.14% | -3.32% | 0.708% |
| Local + global context | +0.880 pp | -1.042 pp | -0.76% | +2.39% | -2.66% | -5.82% | 0.768% |
| Local + global, 0.025 rad | +0.506 pp | -1.042 pp | -0.19% | +2.23% | +1.17% | -2.22% | 0.379% |

This is stronger evidence than either the raw-map or FeatureBooster result:
context still adds information after preserving A1. Full context gains 0.174
R@1 points beyond pointwise training and cuts substantially more hypotheses;
local context is the best pose-balanced variant, improving 56 and worsening 40
of the paired poses while preserving 2 cm recall. Pointwise training alone
repairs one large tail failure, proving that the recoverable-collision objective
also has an independent effect.

However, no tangent-uplift variant passes the predeclared complete gate. The
local-only variant is closest, but its pooled CVaR95 rises 1.14% and 153 of
21,614 A1-clean rows become new false attractors. Adding the global token helps
retrieval and solver efficiency but harms P90 and one tight-recall query.
Halving the angular cap reduces held-out clean-margin violations from 2.32% to
0.22%, yet does not restore tail non-regression. Therefore this is not merely a
residual-strength tuning problem: a single modified 256D descriptor still
couples local invariance and contextual identifiability too tightly.

The metric-preserving tangent uplift is retained as a reproducible diagnostic,
not promoted to deployment. Test queries, protection scenes, Reserve identity
revision, and Track-First/topology rebuild are deliberately not run because the
Stairs mapping gate failed. The next minimal relaxation is the explicit
dual-expert score proposed by the analysis: preserve the A1 cosine term exactly
and add a separately normalized 16--32D context compatibility term through one
concatenated descriptor bank. It must first show that context corrections can
improve recoverable repeated collisions without changing the local A1 score;
only then should Reserve identity or topology be revisited.

## Explicit dual-expert context score decision

The next experiment implements that relaxation exactly. The frozen 256D A1
term and a separate 32D context term are scored as
`a1_dot + lambda * context_dot`. Mapping context codes are view-balanced but
not renormalized, so their norm is the cross-view concentration gate. Two
orthogonal compensation coordinates make both query and map concatenations
unit norm without contributing to their dot product. Consequently this is
still one normalized descriptor bank, one global top-1, and one PoseLib call;
the A1 score is algebraically unchanged. Unit tests verify both constant norm
and exact score equivalence.

```bash
PYTHONPATH=. python scripts/train_evaluate_context_score_crossfit.py \
  --map /data/run/map_learning/anchor_map_step_1520.pt \
  --metric-state /data/run/map_learning/metric_state_step_1520.pt \
  --complete-positive-teacher /data/run/map_learning/complete_positive_teacher.pt \
  --query-cache /data/run/bootstrap/query_cache.pt \
  --scene-calibration /data/run/map_learning/scene_calibration.json \
  --context-mode global_only \
  --expert-input-scope shared_global \
  --context-kernels 3 \
  --code-dim 32 \
  --context-weights 0.01 \
  --training-context-weight 0.05 \
  --output /data/run/map_learning/context_score_shared_global_crossfit.json
```

The first form retained local per-keypoint context. It improves Stairs R@1 by
0.262/0.440/0.389 points for lambda 0.01/0.02/0.05, but every weight fails the
pose gate. At lambda 0.01, 2 cm recall falls 1.042 points and mean/CVaR95 rise
1.0%/4.4%. This proves that merely separating the score channels does not fix
the consensus problem: independently moving individual keypoint winners can
raise retrieval while damaging the spatially coherent correspondence set.

The decisive correction makes the context code genuinely image-shared. The
head consumes only the global dense SuperPoint token; every keypoint in one
image receives exactly the same code, while A1 remains responsible for local
identity. Lambda 0.01 was retained by the safety-first Stairs gate. The table
uses the same 256 mapping-only gate images and 96 PoseLib images per scene.
Negative relative pose values are improvements.

| Scene | R@1 delta | 2 cm recall delta | Mean TE | P90 TE | CVaR95 | Hypotheses | Strict gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stairs | +0.141 pp | +1.042 pp | -21.97% | -4.93% | -63.41% | -1.74% | pass |
| Fire | +0.106 pp | +1.042 pp | -0.51% | +2.08% | -1.62% | -0.05% | fail P90 |
| Heads | +0.117 pp | 0.000 pp | -2.76% | +1.34% | -2.81% | -0.52% | fail P90 |
| Office2/5b | +0.430 pp | 0.000 pp | -2.05% | +4.97% | -2.25% | -3.63% | fail P90 |
| ShopFacade | +0.105 pp | +1.042 pp | -2.40% | +2.21% | -5.22% | -0.63% | fail P90 |

This is the first context form whose main effect transfers across all five
mapping scenes: R@1, mean, CVaR95, and solver work improve everywhere; tight
recall improves in three scenes and is unchanged in two. It also removes the
catastrophic Stairs outlier. However, it is not promoted: the pooled P90 rises
on every protection scene. Fire and ShopFacade improve P90 inside both
directions yet regress after pooling, demonstrating some 96-query quantile
instability, but Office2/5b also regresses in both directions. The formal
non-regression condition therefore remains unmet.

A final image-shared learned gate tested exact A1 fallback. Its mapping label
was deliberately strict: enable context at training lambda 0.05 only when the
image repairs at least one recoverable A1 error and creates no new false winner
on any A1-clean row. The gate reduced Stairs new false attractors from 180 to 2
and clean-margin violations from 12 to zero, but collapsed toward zero. Held-out
R@1 changes by -0.002 points and pose is effectively identical to A1. This is a
safe fallback but not an accuracy method, so class weighting or gate-threshold
tuning is not promoted post hoc.

The first-principles conclusion is now narrower. A shared image context code
contains real, cross-scene identity information and should remain in the final
method family. The unresolved problem is query-level risk calibration: the
training objective rewards row-wise collision repair but cannot distinguish an
image whose small set of changed winners helps geometric consensus from one
whose changes raise the P90 pose error. The next valid experiment is therefore
a mapping-only query-level expected-clean-inlier/consensus objective (with a
tail term), applied to this shared-global dual score. It is not another lambda,
descriptor-cap, training-step, Reserve, or topology sweep. Test sets, Reserve
revision, and Track-First rebuild remain blocked until that objective preserves
the transferred mean/CVaR gains without the protection-scene P90 regressions.

## Query-level clean-consensus objective decision

That objective was implemented without changing A1, the shared-global context
head, lambda, map topology, or PoseLib. The complete-positive teacher supplies
the edges already certified by mapping GT pose, depth, and visibility. For each
query, the model estimates clean top-1 probability against the hardest legal
negative and averages it into expected clean-inlier mass. Four mapping queries
form one optimization batch; the worst quartile supplies a query-level CVaR
term. The preregistered weights are 0.1 for both mean consensus and tail CVaR.

| Stairs shared-global lambda 0.01 | R@1 delta | 2 cm recall delta | Mean TE | P90 TE | CVaR95 | Hypotheses | New false from A1-clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Row-wise collision objective | +0.141 pp | +1.042 pp | -21.97% | -4.93% | -63.41% | -1.74% | 0.833% |
| + expected clean mass and query CVaR | +0.128 pp | +1.042 pp | -21.98% | -4.48% | -61.81% | -0.22% | 0.796% |

The query-level variant passes the absolute Stairs gate but fails the
incremental gate against the simpler shared-global expert. It slightly reduces
new false attractors and clean-margin violations, yet loses retrieval gain and
most of the solver-work reduction while weakening P90/CVaR. It is therefore
retained as a negative ablation and is not expanded to the protection scenes.

This result distinguishes clean ratio from geometric consensus. PnP does not
only need a high expected number of legal winners; it needs those winners to
span informative image locations and 3D directions. A scalar clean-mass CVaR
cannot distinguish clustered, nearly degenerate correspondences from a
well-conditioned set. If context work continues, the only justified extension
is an expected clean information matrix built from fixed mapping geometry,
with a small `-logdet`/conditioning term at query level. Further lambda, gate,
epoch, descriptor-cap, Reserve, or topology sweeps remain unsupported.

## Expected clean pose-information decision

The final justified extension replaced scalar clean mass with a mapping-only
six-DoF observability target. For each legal query row, the fixed mapping pose,
native camera intrinsics, and candidate anchor xyz define the pixel
reprojection Jacobian with respect to SE(3). Its outer product is weighted by
the model's legal top-1 probability. The resulting 6x6 Fisher matrix is
diagonally whitened by the all-legal reference matrix, and the per-query loss
is the mean log-determinant gap. A worst-quartile CVaR term uses the same
four-query batches. This adds no online branch: deployment remains one
normalized descriptor bank, exact cosine top-1, and one PoseLib call.

The unit gate distinguishes equal-count geometrically complete and degenerate
correspondence sets and has finite gradients. The real Stairs smoke also has
positive whitened minimum eigenvalues and non-saturated information retention,
so the experiment was allowed to run. The preregistered single setting uses
mean and tail weights 0.1, damping 0.001, training lambda 0.05, and deployment
lambda 0.01.

| Stairs shared-global lambda 0.01 | R@1 delta | 2 cm recall delta | Mean TE | P90 TE | CVaR95 | Hypotheses | New false from A1-clean | Margin violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Row-wise collision objective | +0.141 pp | +1.042 pp | -21.97% | -4.93% | -63.41% | -1.74% | 0.833% | 0.056% |
| + expected clean Fisher log-det/CVaR | +0.145 pp | +1.042 pp | -21.48% | -4.89% | -61.92% | -0.07% | 0.787% | 0.019% |

The absolute mapping gate passes, but the incremental gate against the simpler
shared-global expert fails. R@1 changes by only +0.0044 points and tight recall
is identical, while mean, P90, CVaR95, and solver-work gains all weaken. The
paired pose result is 47 better / 49 worse, versus 48 / 48 for the simpler
objective; paired P90 translation delta also worsens from 0.0886 to 0.1101 cm.
The cleaner false-attractor and margin counts therefore do not represent a
pose improvement. This variant remains a negative ablation and is not run on
sentinel scenes.

The route stops here rather than substituting minimum eigenvalue or sweeping
weights. Classical Fisher information assumes known inliers, small Gaussian
residuals, and local least squares. Deployment instead makes a discrete hard
winner choice and uses outlier-contaminated RANSAC minimal sets. The loss sees
the conditioning of legal correspondences but not false-winner leverage or the
probability of drawing an all-inlier, non-degenerate minimal set, which is the
actual estimator bottleneck. Any later context objective must model that
discrete robust-estimation event directly; more log-det, lambda, epoch, gate,
Reserve, or topology adjustment is unsupported.

## PoseLib policy oracle and leave-one-out prior decision

The shared-global result was next treated explicitly as a choice between two
unchanged deployment policies: A1 and A1 plus context lambda 0.01. A mapping-
only oracle replays both policies with PoseLib seeds 2026/2027/2028, averages a
task-normalized translation, rotation, catastrophe, and solver-work loss per
query, and chooses the lower-risk policy. Both directions use 128 uniformly
spaced held-out queries (256 total), with paired query bootstrap intervals.

| Scene | Best fixed risk | Oracle risk | Relative oracle headroom | Absolute headroom 95% CI | Oracle A1/context queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stairs | 0.2979 | 0.2891 | 2.94% | [0.00653, 0.01078] | 134 / 122 |
| Office2/5b | 0.5031 | 0.4996 | 0.71% | [0.00242, 0.00496] | 109 / 147 |

The Stairs oracle is real: relative to fixed context, mean TE falls from 1.153
to 1.111 cm and P90 from 2.166 to 2.063 cm. Office2/5b has much less selection
headroom and the oracle leaves all 21 query-seed catastrophes unchanged. A
scene-independent policy gate therefore has weak protection-scene upside; the
oracle mainly supports a difficult-scene option, not a universal new module.

The more fundamental training correction was also implemented. The context
prototype used for a mapping image removes that image's exact per-anchor view
contribution. Supervision uses unique visible anchors rather than detector
rows, averages Track and Reserve separately, balances dependency families, and
uses both current A1 false winners and 256 context-hard anchors that mapping GT
geometry proves to be outside the camera frustum. The prior term is scaled once
to match the original full-support row-wise loss magnitude; no performance
weight search is used.

| Stairs lambda 0.01 | R@1 delta | 2 cm recall delta | Mean TE | P90 TE | CVaR95 | Hypotheses | New false from A1-clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing row-wise shared context | +0.141 pp | +1.042 pp | -21.97% | -4.93% | -63.41% | -1.74% | 0.833% |
| LOO unique-anchor prior, false negatives only | -0.015 pp | 0.000 pp | -0.63% | +0.16% | +0.002% | +0.06% | 0.167% |
| LOO prior plus explicit incompatible negatives | +0.004 pp | 0.000 pp | -0.25% | +0.23% | +0.14% | -0.19% | 0.176% |

The prior is safe but removes the useful context effect. Without explicit
incompatible negatives, final map-code concentration rises to about 0.97.
Geometry negatives temporarily lower it to 0.92--0.93 after stage one, but it
returns to about 0.96 after refresh. This is not a numerical failure: nearby
indoor images genuinely share most visible anchors, so an image-level binary
compatibility objective is dominated by a common direction and supplies
almost no discriminative top-1 bias. More negative counts, centering, loss
weights, or epochs would only tune the same mismatch.

The preregistered stop condition is met: leave-one-out prior training
eliminates the cross-scene context gain. It is retained as a negative ablation
and is not expanded to sentinel scenes or followed by a generic learned gate.
The original row-wise expert remains evidence that context can help repeated
structure, and its Stairs policy oracle quantifies an upper bound, but the
current context family is frozen rather than forced into the final method.

## Adaptive V2 smoke validation

The mapping-only adaptive topology was isolated on ShopFacade by reusing the
same frozen V1 scaffold, function graph, Track-First payload, and native query
cache. Compact-map labels and the metric were rebuilt, then all 103 test images
were evaluated with seed 2026 and the unchanged one-shot sparse PoseLib path.

| Topology | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen V1 | 7,145 | 1.808 cm | 4.647 cm | 7.312 cm | 10.074% | 40.313% | 81.553% |
| Adaptive V2, aggressive rank target | 4,159 | 1.792 cm | 4.744 cm | 7.784 cm | 9.004% | 44.782% | 81.553% |
| Adaptive V2, retained policy | 6,481 | 1.938 cm | 4.692 cm | 8.132 cm | 10.001% | 42.426% | 82.524% |

The retained policy reduces map capacity by 9.3% while preserving mean error,
raw precision, and recall within a small range, but does not improve median or
P90. The aggressive variant demonstrates additional redundancy but loses raw
cleanliness and RANSAC efficiency. This is a compatibility smoke validation,
not a multi-seed replacement for the registered frozen V1 result.

## Cross-scale adaptive validation

The completed adaptive pipeline was then checked on an outdoor Cambridge scene
and a clean, from-prior indoor rebuild. All reported localization runs use
native full-resolution inputs, uncapped global top-1 matching, one PoseLib
solve, and the complete test split. Pose metrics below are means over seeds
2026/2027/2028 unless noted otherwise.

| Scene / method | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade frozen V1 | 7,145 | 1.808 cm | 4.647 cm | 7.312 cm | 10.074% | 40.313% | 81.553% |
| ShopFacade adaptive | 6,357 | 1.993 cm | 4.74 cm | 8.318 cm | 9.979% | 42.538% | 81.553% |
| Heads frozen V1 | 8,264 | 0.465 cm | 0.575 cm | 1.062 cm | 9.186% | 18.346% | 100.0% |
| Heads adaptive, angular-only PnP | 8,119 | 0.447 cm | 0.721 cm | 1.435 cm | 11.136% | 48.717% | 99.2% |
| Heads adaptive, track-residual PnP | 8,119 | 0.498 cm | 0.619 cm | 1.145 cm | 11.142% | 24.277% | 99.93% |

The indoor angular-only threshold (`3.788 px`) was below the residual scale of
stable mapping tracks. The final mapping-only rule uses their per-track p90
residual at quantile 0.975 (`10.419 px` on Heads), capped at 12 pixels. It
reduces Heads P90 by 20%, restores 5 cm recall, and cuts mean RANSAC hypotheses
from 11,790 to 3,713 relative to the angular-only run. Against frozen V1 it is
a mixed but useful efficiency/cleanliness Pareto point: fewer anchors and half
the detector keypoints, higher raw and inlier GT precision, similar P90 and
recall, but a slightly higher median and mean. ShopFacade automatically remains
at 12 pixels and reproduces the prior adaptive result.

These two scenes establish scale adaptation and catch the indoor solver-gate
failure, but they are not sufficient evidence for universal non-inferiority.
The full Cambridge, 7Scenes, and 12Scenes evaluation remains required.

## Deployment-aware topology revision gate

A one-pass mapping-only revision was tested after compact reconstruction on
ShopFacade. It replayed exact global top-1 plus standard PoseLib on all 231
mapping views, scored counterfactual clean replacements, clean/harmful solver
inliers, full-SE(3) Fisher deletion loss, and matching-rank criticality, then
protected anchors with non-improving replacements in the worst 10% mapping
queries. The revision removed 7 of 6,357 anchors, all from the Gaussian reserve;
the matching-rank p10 remained 159 against a target of 119.

| Test method, three-seed mean | Anchors | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | 2 cm recall | 5 cm recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive V2 | 6,357 | 1.993 cm | 4.7449 cm | 8.318 cm | 9.979% | 42.538% | 50.49% | 81.55% |
| Revision only | 6,350 | 1.981 cm | 4.7439 cm | 8.350 cm | 9.977% | 42.547% | 51.46% | 81.55% |
| Revision + 44-step full-shard refresh | 6,350 | 1.981 cm | 4.7496 cm | 8.350 cm | 9.982% | 42.552% | 51.46% | 82.52% |

The original trainer replayed only one of seven rotating mapping shards when
`refresh_interval=0`. Enabling all shards exposed an uncapped entropic Group-DRO
collapse: one trajectory group reached 98.6% weight after two refreshes. A
capped update (at most three times uniform weight) fixes that failure mode and
kept an effective 5.1--5.3 groups during the short refresh.

The result is mixed rather than Pareto improving: median, clean precision, and
recall improve slightly, while mean, P90, and RANSAC hypotheses do not. The
mapping-only meaningful-improvement gate therefore rejects the revision as a
default method. The frozen adaptive mainline and its one-shot deployment remain
unchanged; the revision is retained as an opt-in, reproducible experiment.

## Cross-fitted swap and duplicate-suppression follow-up

The two remaining topology/deployment hypotheses were tested without changing
the frozen Adaptive V3 map learner. A cross-fitted swap miner used alternating
contiguous mapping-trajectory blocks for proposal and gate. It searched 1,576
unused broad tracks and 3,364 unused Gaussian candidates, and proposed only
four pairs whose positive descriptor could replace a wrong Gaussian winner on
at least two rows. Three one-for-one swaps preserved the 119-row matching-rank
constraint and kept the ShopFacade map at 6,357 anchors (5,726 tracks plus 631
Gaussian reserve anchors).

| ShopFacade mapping fold | Median TE | Mean TE | P90 TE | CVaR95 | Raw P@2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selection, before | 1.801 cm | 2.720 cm | 4.489 cm | 17.427 cm | 15.583% |
| Selection, after | 1.841 cm | 2.717 cm | 4.489 cm | 17.418 cm | 15.588% |
| Held-out gate, before | 1.241 cm | 25.646 cm | 4.416 cm | 461.949 cm | 12.798% |
| Held-out gate, after | 1.241 cm | 25.652 cm | 4.413 cm | 461.949 cm | 12.801% |

The held-out meaningful-improvement gate rejected the swaps. Test was not
consulted. This closes the selection-bias gap in the original prune-only
revision and shows that the observed assignment gains do not transfer into a
material pose improvement.

The second experiment retained only the highest-cosine query match for each
anchor before the same single PoseLib solve. It is opt-in; the default still
passes every global top-1 correspondence to PoseLib.

| Scene / method | Queries | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade adaptive | 103 | 1.993 cm | 4.745 cm | 8.318 cm | 9.979% | 42.538% | 4,318 |
| ShopFacade duplicate suppression | 103 | 1.951 cm | 4.832 cm | 9.515 cm | 12.994% | 44.066% | 2,232 |
| Heads adaptive | 1,000 | 0.498 cm | 0.620 cm | 1.144 cm | 11.142% | 24.278% | 3,713 |
| Heads duplicate suppression | 1,000 | 2.586 cm | 2.786 cm | 4.150 cm | 1.035% | 2.000% | 2,914 |

ShopFacade removes 25.0% of correspondences and approximately halves RANSAC
hypotheses, but worsens mean, P90, and 5 cm recall. Heads removes 28.1% and
fails structurally: descriptor score alone cannot decide which of several
query keypoints assigned to one landmark is geometrically correct. The same
failure appears in two seeds over the complete 1,000-query test split. Online
one-to-one suppression is therefore a No-Go, not part of the paper method.

These results trigger the predeclared stop condition for further structural
revision. Adaptive V3 remains frozen; cross-fitted swap and duplicate
suppression stay as reproducible negative-result modules.

As a release guard, the unmodified deployment option was rerun on all 103
ShopFacade test queries with seed 2026. Every estimated pose matrix, match
count, error metric, precision metric, and RANSAC-iteration count exactly
matched the registered Adaptive V3 artifact.

## Geometry, one-shot guidance, and targeted descriptor follow-up

Three remaining hypotheses were implemented as opt-in mapping-only revisions.
Adaptive V3 remains the default. Geometry and descriptor revisions used
disjoint temporal fit, point-validation, and pose-gate blocks. Guided sampling
used fixed all-mapping matchability statistics and a complete mapping replay;
it is therefore a weaker development gate, not a cross-fitted claim. None of
the three inspected test queries before its declared mapping gate.

The first revision refined only the 633 selected Gaussian reserve positions in
their source-splat local tangent/normal frames. Mapping positives supplied the
reprojection term, while source 2DGS scales and the calibrated surface bounds
formed the trust region. Of 62 reserve anchors with sufficient three-way
observations, 34 passed point validation. Their median displacement was 0.521
cm (P90 1.132 cm, maximum 3.086 cm). On 75 unseen mapping views, median/mean/P90
changed from 1.972/3.464/4.150 cm to 1.955/3.461/4.126 cm, so the weak-positive
gate passed.

| ShopFacade test, three-seed mean | Median TE | Mean TE | P90 TE | Raw P@2 | Inlier P@2 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive V3 | 1.993 cm | 4.7449 cm | 8.318 cm | 9.979% | 42.538% | 4,318.2 |
| Bounded reserve geometry | 1.993 cm | 4.7438 cm | 8.321 cm | 9.981% | 42.545% | 4,318.1 |

The full test result is effectively neutral: mean and precision improve by a
noise-level amount, P90 worsens by 0.002 cm, and recall is unchanged. The
geometry branch is therefore valid evidence that bounded surface refinement
can operate safely, but it is not promoted to the default method.

The second revision preserved every global top-1 correspondence and sorted it
by descriptor margin, mapping-derived anchor matchability, and normalized 3D
covariance before one PoseLib PROSAC solve. The mapping gate showed a small P90
improvement (4.622 to 4.527 cm), but only 25 of 231 poses changed and RANSAC
hypotheses fell by just 0.77%. The complete test result rejected the hypothesis:

| ShopFacade test, three-seed mean | Median TE | Mean TE | P90 TE | CVaR95 | Hypotheses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard one-shot PoseLib | 1.993 cm | 4.7438 cm | 8.321 cm | 39.738 cm | 4,318.1 |
| Guided one-shot PoseLib | 1.993 cm | 4.7535 cm | 8.321 cm | 39.799 cm | 4,321.2 |

The final revision calibrated only Gaussian reserve descriptors that repeatedly
won an incorrect mapping top-1 while retaining legal positives in disjoint fit
and validation blocks. Descriptor changes were bounded to 0.025, and validation
required fewer false wins without losing any existing correct wins. Seven
anchors had sufficient cross-fitted evidence and two passed anchor-level
validation. On the unseen pose gate, mean changed by 0.00013 cm, P90 by 0.0062
cm, and raw P@2 was identical. This did not meet the predeclared meaningful
improvement threshold, so test was not run.

Together these experiments narrow the remaining limitation: reserve geometry
and isolated harmful descriptors can be changed safely, but neither is the
dominant ShopFacade error source. Mapping-only reliability ordering also does
not transfer into better one-shot RANSAC behavior. The frozen Adaptive V3
topology, descriptors, and standard PoseLib deployment remain the registered
mainline; complete multi-scene benchmarking has higher value than further
single-scene local revisions.
