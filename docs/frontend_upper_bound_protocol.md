# Stronger local frontend candidate comparison and ceiling probe

## Decision

The Stairs descriptor arm has now produced a real mapping-only result. Locked
XFeat improves both temporal directions, pooled R@8, and Track Core R@1, so
frozen SuperPoint has measurable descriptor headroom. The strict gate is still
`STOP_BEFORE_MAPPING_ONLY_DESCRIPTOR_REBUILD`: pooled Gaussian Reserve R@1
regresses by 0.02518 pp and fails the preregistered exact non-regression rule.

This is a candidate comparison / ceiling probe, not a mathematical upper bound
unless a future candidate has a separately justified oracle property.  It does
not change the default frontend, map topology, anchor composition, metric,
matcher, PoseLib, or any deployment artifact.  It never consumes test queries.

## First-principles decomposition

After the current P7 density and pair-policy factors have received their
single-factor mapping gates, the remaining correspondence failures must be
separated before another method branch is justified:

1. **Sampling failure:** a depth/visibility-legal 3D identity is not near any
   of the detector's fixed-budget keypoints.
2. **Identity failure:** a legal identity is sampled, but its descriptor does
   not rank the correct frozen-map anchor highly enough.
3. **Consensus/pose failure:** sufficiently good correspondences exist, but the
   one-shot global-top1/PoseLib path does not turn them into a robust pose.

A detector+descriptor replacement in one run cannot say which of (1) and (2)
changed.  A pair matcher changes the candidate graph and scoring problem as
well, so it cannot answer either question.  The new audit consequently has two
independent arms and preserves the current global descriptor-bank problem.

## Existing implementation audit

| Surface | What is already available | Consequence for this protocol |
| --- | --- | --- |
| Frozen SuperPoint | Native detector, two-pass NMS radius 4, top-K sparse API, 256D dense descriptor sampling, and `index + 0.5` pixel-center convention in `features/superpoint.py` | Valid reference frontend; its checkpoint resolves locally and matches the required SHA256. |
| Localization frontend | `features/extractor.py` and `localization/frontend.py` preserve the native sparse interface | A candidate must replay the same resized input, valid mask, K cap, and coordinate convention. |
| Query cache | Version-11 cache construction in `map_learning/bootstrap.py` records native keypoints/descriptors/scores, depth, alpha, valid mask, intrinsics, GT pose, input size, detector K/NMS metadata, and the 0.5 offset | Sufficient for read-only paired scoring after a candidate probe has been materialized; it does not itself contain a stronger frontend output. |
| Complete-positive teacher | `lafgs_v9_active_map_complete_positive_teacher` in `map_learning/observations.py` stores exact query rows plus positive/ambiguous CSR anchor identities under GT geometry, visibility, depth, and provenance constraints | Supplies the frozen legal-positive labels for descriptor R@K without rebuilding topology or inventing labels. |
| Cross-fit/fusion | `temporal_crossfit_split`, view-balanced observation fusion, CSR selection, and exact rank membership already exist | Reused for bidirectional support/gate identity scoring; support observations are identical between control and candidate. |
| FeatureBooster | Strict 256D SuperPoint postprocessor implementation and expected official weight hash exist; the weight is absent on this machine | Not an independent frontend.  Its historical mapping cross-fit already improved identity R@K but failed pose-risk gates on Stairs and Office2/5b, so it is a control, not an untested ceiling answer. |
| MCCD/context family | Raw-map MCCD, metric-preserving uplift, dual expert, shared-global code, query-consensus objective, and LOO priors are implemented and documented | These all transform frozen SuperPoint evidence.  Their mixed/failed transfer gates do not establish the upper limit of a stronger independently pretrained local representation. |

The exact audited code hashes and local artifact inventory are machine-readable
in [`frontend_upper_bound_environment_audit.json`](frontend_upper_bound_environment_audit.json).

## Probe-cache contract

The runner never constructs or downloads a model.  A separately reviewed
producer must materialize a Torch probe with schema
`lafgs_frontend_ceiling_probe_cache`, version 1, containing:

- `mapping_only=true` and `uses_test_queries=false`;
- a `reference` block carrying the exact query-cache signature and teacher
  schema;
- frontend family, implementation/version identifier, local weight path, and
  exact weight SHA256;
- the same requested detector K as every reference cache record;
- explicit detector and/or descriptor capabilities;
- for every mapping query, the SHA256 of the exact reference keypoint tensor;
- for arm A, post-mask candidate keypoints/scores and the pre-mask detection
  count; and
- for arm B, one finite nonzero descriptor at every reference SuperPoint row.

The consumer fails closed on a query-name mismatch, keypoint-registry hash
mismatch, K mismatch, invalid/masked-out keypoint, descriptor-row mismatch,
non-positive candidate dimension, missing or wrong weight hash, any test-query
claim, or a `pair_matcher` frontend family.  Candidate dimension may differ
from SuperPoint's 256D: each protocol builds and scores its bank in its native
space.  Fairness comes from identical keypoint rows, legal edges, support
folds, anchor registry, and top-K ranking—not from an arbitrary equal-dimension
constraint.  Reports therefore include float32 map-bank bytes, dot-product
MACs, and measured ranking wall time for both spaces.

The materializer still needs the frozen mapping image source and preprocessing
lineage referenced by the query-cache signature.  A query cache alone is not a
license to approximate the original RGB transform.

The evaluate CLI additionally requires the exact serialized query-cache and
teacher paths recorded by the producer and verifies both SHA256 values before
scoring. A different teacher with the same schema, an edited cache that retains
the old signature, or an identical byte copy supplied from another artifact
path is rejected. Direct Python callers retain an optional path-free mode for
synthetic/unit use, while all CLI evaluation is source-bound.

## Arm A: detector repeatability ceiling probe

**Question:** with the same requested K, can a stronger detector place legal
keypoints close to more frozen-map identities than SuperPoint?

For each mapping query, the audit:

1. projects every frozen anchor XYZ with the cached GT pose and native
   intrinsics;
2. retains only positive-depth, in-frame anchors whose projected pixel passes
   the frozen valid mask and cached rendered depth/alpha legality thresholds;
3. converts both reference and candidate grid coordinates using the same
   cached pixel-center offset;
4. computes each legal projection's nearest detected keypoint distance; and
5. reports reachable fraction at registered pixel radii, pooled and split into
   Track Core versus Gaussian Reserve.

The target universe is independent of which SuperPoint rows happened to be
detected.  Candidate descriptors, matching scores, teacher identity labels,
PnP, and test images are not used.  “Same K” means the same requested top-K cap;
post-mask counts may differ and are reported because that is part of detector
behavior.

A useful mechanism result must improve the strong-radius pooled reachability
on Stairs while not reducing ambiguous-radius reachability for either Track
Core or Reserve.  Otherwise a full frontend/Pose experiment is not justified.

## Arm B: descriptor identity ceiling probe

**Question:** at exactly the SuperPoint locations already available, does a
stronger independently pretrained representation rank the legal anchor
identity better?

The candidate must sample its descriptor field at the reference SuperPoint
coordinates; its own detector is disabled.  Each temporal-block direction:

1. builds raw-SuperPoint and candidate anchor banks from exactly the same
   support-fold complete-positive edges;
2. averages repeated rows within one image, then contributes one normalized
   descriptor per anchor-observing view;
3. applies the same minimum support-view mask to both banks;
4. evaluates only the disjoint held-out fold with exact global cosine ranking;
   and
5. reports legal-positive R@1/2/4/8/16/32, including Track Core and Reserve
   composition, in both directions and pooled.

Raw SuperPoint uses its native 256D bank and the candidate uses its own positive
dimension.  Both see the same query rows and supported anchors and retrieve the
same number of identities.  Per-direction reports include map-bank memory,
cosine-ranking MAC count, and CPU matrix-multiply-plus-top-K wall time.  These
are deployment costs, not accuracy gates.  If a higher-dimensional candidate
passes, a later projected-to-256 experiment is a separate single factor rather
than something silently folded into this comparison.

No pair-conditioned feature, candidate detector, topology revision, oracle
assignment, learned selector, PnP, or test query enters this arm.  The minimum
mechanism gate is positive R@1 delta in **both** cross-fit directions, pooled
R@8 non-regression, and no pooled R@1 regression for either anchor kind.  A
failure means only that this locked candidate did not exceed SuperPoint under
the fixed identity problem; it does not prove all possible frontends are at the
same ceiling.

## Local artifact audit and blocker

### 2026-08-12 bounded inventory update

The original preflight below was correct for its restricted paths, but a later
bounded inventory of named historical external-control encoder directories
found an admissible local XFeat artifact. The current artifact status is
therefore no longer `BLOCKED_BY_ARTIFACT`: it is
`READY_FOR_PRODUCER_REVIEW`. This update does not supply a result and does not
change the default SuperPoint frontend.

The locked candidate is single-image XFeat with a native 64D dense/sparse
descriptor, weight SHA256
`0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`,
parent ULF-Loc commit
`b28d53258ab4461ba1a02eaa60ef504e9b82b9ab`, and XFeat tree
`4f804566cb1cf72469b7d7174fba9308885c5c5a`. Its checkpoint, code tree,
Apache-2.0 license, CPU strict-load precheck, producer coordinate contract,
and descriptor-first/detector-second progression are recorded in
[`frontend_checkpoint_inventory_20260812.json`](frontend_checkpoint_inventory_20260812.json)
and
[`xfeat_frontend_probe_provision_plan.md`](xfeat_frontend_probe_provision_plan.md).
The adjacent XFeat-LighterGlue checkpoint remains inadmissible as a pair
matcher. The real XFeat Arm B result is now recorded in
[`xfeat_arm_b_stairs_result.md`](xfeat_arm_b_stairs_result.md) and
[`xfeat_arm_b_stairs_gate.json`](evidence/xfeat_arm_b_stairs_gate.json). It
consumed 2,000 mapping images and 2,048,000 fixed rows, used no test query, and
stopped before map/metric/pose.

The read-only preflight on 2026-08-12 found:

- frozen SuperPoint checkpoint: 5,206,086 bytes, verified SHA256
  `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`;
- Kornia 0.8.2 code in the `g4splat` environment, exposing `DISK`, `DeDoDe`,
  `HardNet`, `SOSNet`, and `LoFTR`, but no attested compatible local checkpoint;
- official FeatureBooster expected SHA256
  `5334d9aa861e877a2b99baff0d682e1ac8a749cdd65eb1d4b8bd0a8bb8bf0359`,
  but no file at the restricted local path; and
- a local `loftr_outdoor.ckpt`, SHA256
  `21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`,
  which is deliberately rejected because LoFTR is a pair matcher rather than
  an independently indexable global-map descriptor frontend.

No network access or broad filesystem scan is part of preflight.  Code presence
without a locked checkpoint is not an experiment.  The executable arms remain
blocked until an approved candidate supplies: local immutable weights and
SHA256, locked implementation lineage, offline preprocessing/materialization,
and (for arm B) a positive-dimensional dense field sampleable at the reference
keypoints.

## Minimal progression and pose gate

The shortest defensible sequence is:

1. Finish the two existing P7 Stairs single-factor gates.  Enter this frontend
   line only if neither resolves the failure domain.
2. Provision one reviewed candidate artifact; materialize its mapping-only
   probe without changing deployment.  Do not project it to 256D at this stage.
3. Run arm A and arm B separately on Stairs.  Stop the failing arm; do not
   combine the detector and descriptor just to obtain a favorable aggregate.
4. Only for an arm that passes its mechanism gate, rebuild the corresponding
   **mapping-only** candidate observation/map view while preserving anchor IDs,
   map size, topology, global top-1, one PoseLib call, query registry, K, and
   seed.  Require no increase in catastrophic failures and non-regression of
   median, mean, P90, and CVaR95 translation error, with at least one accuracy
   metric strictly improved.
5. Use 12Scenes `office2_5b` as the first no-regression identity/pose tail guard.
   If detector arm A is the mechanism winner, use Cambridge GreatCourt next as
   the outdoor repeatability guard.  Do not inspect test queries until mapping
   gates pass.

This gate is intentionally stricter than “R@1 increased”: earlier
FeatureBooster and context experiments already proved that row-wise retrieval
improvement can worsen pose tails.

## Reproduction

Environment-only preflight (no model construction):

```bash
PYTHONPATH=. /root/miniconda3/envs/g4splat/bin/python \
  scripts/audit_frontend_upper_bound.py preflight \
  --output docs/frontend_upper_bound_environment_audit.json
```

Generate the exact producer contract from frozen mapping artifacts:

```bash
PYTHONPATH=. /root/miniconda3/envs/g4splat/bin/python \
  scripts/audit_frontend_upper_bound.py contract \
  --query-cache /path/to/query_cache.pt \
  --teacher /path/to/complete_positive_teacher.pt \
  --output /path/to/frontend_probe_contract.json
```

Consume a reviewed probe after its local weight path and hash are available:

```bash
PYTHONPATH=. /root/miniconda3/envs/g4splat/bin/python \
  scripts/audit_frontend_upper_bound.py evaluate \
  --state /path/to/compact_anchor_map.pt \
  --query-cache /path/to/query_cache.pt \
  --teacher /path/to/complete_positive_teacher.pt \
  --probe-cache /path/to/locked_candidate_probe.pt \
  --arm both \
  --output /path/to/frontend_upper_bound_report.json
```

The unit tests use only synthetic mapping artifacts and prove the positive
signal path plus fail-closed hash, family, row, K, and pairing constraints:

```bash
/root/miniconda3/envs/g4splat/bin/python -m pytest -q \
  tests/test_frontend_upper_bound.py
```

## Conclusion

The implementation now cleanly answers two different questions without
inventing another selector. Descriptor headroom is no longer unknown: locked
XFeat raises pooled R@1 by 4.40225 pp and pooled R@8 by 7.92901 pp at the exact
SuperPoint rows. It is not a deployable win because Reserve R@1 fails the
strict non-regression gate, so no downstream rebuild ran. Detector headroom
remains unmeasured. The actionable boundary is representation-by-evidence
type, not a blanket claim that either the method or SuperPoint is optimal.
