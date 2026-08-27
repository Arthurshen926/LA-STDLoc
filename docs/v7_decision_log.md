# V7 decision log

## 2026-08-26 — mainline reset at `9982d2b`

- Created `codex/v7-safe-closed-loop-mainline` from the correctness-fixed base.
- Enabled only P0 identity no-op reproduction.
- Added a recursive source allowlist and legacy import denylist.
- Registered the twelve immutable method invariants and fail-closed artifact checks.
- Deferred novel views, render certification, selection, feedback learning, and
  acquisition until their preceding phase gates pass.

This log records formal-method decisions only. Historical V6 experiment choices
must not be copied here as V7 defaults.

## 2026-08-26 — P0 passed on StMarysChurch

- The no-op output preserved the frozen 200,255-Anchor map and identity metric
  byte-for-byte and tensor-for-tensor.
- All non-timing fields of all 530 test queries matched the frozen baseline.
- The fixed online contract matched: 2,048-keypoint native SuperPoint, exact
  global Top-1, and one standard PoseLib solve.
- The formal recursive import graph contained only the V7 runner and contract;
  no forbidden historical module was reachable.

This pass unlocks P1 only. Test results were used solely for P0 parity and did
not update, tune, or select a map.

## 2026-08-26 — P1 role and planning contract passed

- Registered five disjoint roles with exact allowed and forbidden operations.
- Added a deterministic planner driven by within-sequence baseline and rotation
  statistics, not fixed cross-scene metre thresholds.
- StMarysChurch validation used 1,487 mapping cameras and generated 64 clean
  feedback plus 64 clean confirmation poses. Neither batch duplicated a mapping
  pose, and the two batches were disjoint.
- Feedback and confirmation artifacts explicitly prohibit Track, Anchor CSR,
  and descriptor-bank membership.
- P2 remains disabled until these plans are rendered and certified.

## 2026-08-26 — P2 clean-render certificate passed

- Rendered the two preregistered, mutually disjoint P1 batches from the frozen
  StMarysChurch 2DGS prior: 64 feedback poses on GPU 1 and 64 confirmation poses
  on GPU 2. Mapping RGB was never loaded and the map was not an input.
- Applied native frozen SuperPoint to each complete, unmasked rendered RGB.
  Alpha, depth, black-hole, and distortion evidence was sampled only after
  detection at keypoint rows; distortion remained a local row mask and was not
  converted into a global render-reliability score.
- Feedback decisions were 60 ACCEPT, 2 UNCERTAIN, and 2 REJECT. Confirmation
  decisions were 63 ACCEPT, 1 UNCERTAIN, and 0 REJECT. Every non-ACCEPT record
  is excluded from map updates by schema and contract.
- Visual audit confirmed that the three marginal-keypoint cases contain large
  near-foreground curtain artifacts. The two depth-curtain rejections look
  plausible in RGB but disagree strongly with camera-support expected depth,
  so they remain conservatively rejected.
- Thresholds were not adjusted after observing these outcomes. An earlier
  renderer-environment failure and a development render lacking all supported
  diagnostic channels are retained as non-formal attempts; only the complete
  v3 manifests are formal evidence.
- All 128 record hashes were verified. The frozen Anchor map hash remained
  unchanged, and a complete P0 rerun again matched all 530 test-query records
  with zero non-timing mismatches and zero forbidden imports.

This pass unlocks P3 only. It does not authorize feedback learning or any use of
UNCERTAIN/REJECT observations.

## 2026-08-27 — P3 mapping-evidence feasibility correction

- Reconstructed descriptor dispersion, reprojection error, ray angular
  dispersion, image-cell/depth lineage, and full-SE(3) Fisher contributions
  from the frozen rendered-mapping SuperPoint rows and pure-ray Anchor xyz.
- The first preregistration required two view families in Eligibility. Before
  any localization evaluation, the mapping-only curve proved that this made
  every profile infeasible: 100 mapping queries had zero eligible candidates.
- Corrected the minimum to one, matching the frozen candidate-construction
  contract. View-family diversity remains an explicit Matching Completion
  target rather than a hard precondition that removes all support for a query.
- The four infeasible v3 curve artifacts are retained. No test result, feedback
  result, or localization metric was observed or used for this correction.

## 2026-08-27 — P3 first feasible curve failed and was rescaled

- The mapping-feasible v5 curve selected 11,376 / 8,480 / 5,892 / 3,117
  Anchors. All four maps met their requested feasible layer and pose targets.
- On the 60 P2-ACCEPT novel feedback RGBs, every compressed map worsened median
  translation and R@2 versus the 200,255-Anchor full pool. Lower inlier ratios
  also increased PoseLib iterations, so wall-clock latency became worse rather
  than better. P4 remained locked.
- The cause was a scale error in the curve definition: a 128-row target had
  been labelled "large" despite the immutable 2,048-keypoint online contract.
  The next and final P3 curve uses fixed capacity fractions: 1024, 512, 256,
  and 128 rows for large through aggressive. It does not tune candidate scores
  or thresholds to individual query outcomes.

## 2026-08-27 — P3 unified Selector passed

- The capacity-scaled curve produced 116,658 / 48,581 / 22,131 / 10,801
  Anchors for large / medium / small / aggressive. Every profile met its
  mapping-feasible matching, cell, depth, family, logdet, and minimum-eigenvalue
  targets with zero unmet entries.
- On all 60 P2-ACCEPT novel feedback RGBs, the 48,581-Anchor medium map improved
  median translation from 0.410cm to 0.369cm, P90 from 0.939cm to 0.932cm, mean
  from 41.16cm to 14.64cm, and average runtime from 96.54ms to 82.40ms. R@2 and
  R@5 both remained 98.33%; there were no gained or lost queries at either gate.
- The medium selection used no source mapping RGB, feedback update, or test
  query. It is frozen as the initial map M0. Small and aggressive remain curve
  evidence only and are not deployment candidates.
- A complete post-P3 P0 rerun again preserved the 200,255-Anchor baseline hash,
  matched all 530 test records with zero non-timing differences, and reported
  zero forbidden imports. Those test records did not select M0.

This pass unlocks P4 fixed-plant feedback only. It does not authorize a changed
online matcher, use of non-ACCEPT render records, or test-driven map selection.

## 2026-08-27 — P2 canonical RGB replay correction

- The original certified records persisted only an 8-bit RGB visualization,
  while certification had consumed the renderer's float tensor. That lineage
  was insufficient for the P4 requirement that the same RGB enter the same
  plant, so those records remain non-formal history.
- P2 now persists float16 clean RGB and runs SuperPoint on its exact float32
  replay. Feedback and confirmation were regenerated on GPUs 1 and 2. Their
  decision counts remained 60/2/2 and 63/1/0 for
  ACCEPT/UNCERTAIN/REJECT, respectively.
- Replaying the RGB in a process that has already allocated the compact map
  preserves keypoints and scores bitwise. CUDA convolution ordering moves
  descriptor entries by at most 8.94e-08, so the formal contract records the
  measured 1e-6 numerical bound instead of making a false bitwise claim. No
  quantization or online-plant modification was introduced.

## 2026-08-27 — P4 fixed plant passed

- Added the sole RGB-only localization interface: frozen native SuperPoint,
  exact global cosine Top-1, then one standard PoseLib call. Its signature
  cannot receive GT, alpha, depth, or an oracle correspondence.
- The first implementation was rejected by the import firewall because the
  historical PoseLib wrapper could statically reach group-consensus code. V7
  therefore carries isolated exact Top-1 and standard PoseLib kernels; the
  forbidden formal import count is zero. Unit comparisons match the current
  deployment Top-1 scores/indices and PoseLib pose/inliers exactly.
- On the 48,581-Anchor M0, the feedback batch produced 59 nominal successes,
  one coverage deficit, zero representation deficits, and four unreliable
  renders. All 64 RGBs passed through the same plant; non-ACCEPT records were
  excluded only after localization. Oracle geometry and depth visibility were
  used solely for post-localization routing.
- M0 averaged 78.48ms versus 101.24ms for the 200,255-Anchor full pool. On the
  60 ACCEPT queries both retained 98.33% R@2 and R@5 with the same one
  catastrophic query. M0 median TE/AE was 0.423cm/0.0079deg and P90 TE was
  1.107cm; the full pool was 0.358cm/0.0109deg and 0.785cm. This is an
  acceptable median-primary precision/speed tradeoff under the task-scaled
  soft-tail gate, but the strict P90 non-regression diagnostic is explicitly
  false and is not hidden.

## 2026-08-27 — P5 evidence-only controller terminated the loop safely

- Added bounded multiplicative observation weighting, view-family balancing,
  trimmed reconstruction, angular trust limits, and one-Anchor/one-descriptor
  output. Feedback descriptors only score original rendered-mapping
  observations and can never be copied into the map.
- The controller requires consistent evidence from at least two independent
  pose families. P4 found zero representation-deficit queries, hence zero
  potential Anchors and zero changed Anchors. The proposal SHA is exactly the
  M0 SHA; no candidate pool proposal was materialized.
- P6 fresh-batch confirmation and P7 second round are therefore not run: the
  preregistered `no executable representation deficit` stop condition fires
  before a proposal exists. P8 acquisition is also disabled because only one
  coverage deficit occurred, not repeated deficits across independent ACCEPT
  pose families. This is a successful safeguarded termination, not an
  incomplete training run.
