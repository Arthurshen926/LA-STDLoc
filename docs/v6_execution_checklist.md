# V6 execution checklist

This checklist describes the formal V6 run contract. It does not record a
result before the corresponding SHA-bound artifact exists, and it does not
authorize automatic acceptance or winner selection.

## Frozen construction contract

- [ ] Start from the immutable `v4-render-only-frozen` baseline; do not import
  a V5 adapter or target-materialization experiment.
- [ ] Use Gaussian-rendered mapping images only. Reject source mapping RGB and
  any test-query dependency.
- [ ] Apply rendered-alpha validity before native SuperPoint NMS and Top-K.
- [ ] Build one reciprocal-descriptor, known-pose-epipolar,
  cycle/chain-confidence association graph.
- [ ] Materialize all deployable Anchor coordinates through robust
  multi-camera rays. Rendered depth and Gaussian centers may propose/audit but
  never become the deployed coordinate.
- [ ] SHA-validate the map, identity metric, rendered observation cache,
  association graph, materialization report, scene calibration, and feedback
  calibration binding before launching a formal subprocess.

## Feedback contract

- [ ] Use fresh v8 feedback for the baseline and every candidate.
- [ ] Use F0 fixed-map feedback as the main deployment observer. Keep geometry,
  descriptors, topology, and the Anchor subset fixed across all queries in a
  round.
- [ ] Use F1 fixed-geometry descriptor-leave-self-out only to audit direct
  descriptor self-influence. Use full geometry LOO (F2) only as a stress test.
- [ ] Treat exact projective identity as the only strong descriptor positive.
  A non-identity alternative is pose-valid only after aligned surface-depth
  certification; without depth it remains diagnostic ambiguity/ignore.
- [ ] If the cache has dense rendered depth/alpha but omits redundant sparse
  depth columns, sample the exact detector rows with the frozen raster
  convention and retain alpha/validity fail-closed masking.
- [ ] Keep L1 image-cell visibility, L2 detectability, L3 one-to-one matching,
  and L4 task-scaled pose information as separate recorded targets.
- [ ] Use native SuperPoint, one global cosine Top-1 per query row, and exactly
  one standard PoseLib solve inside every replay.
- [ ] Use the RANSAC threshold from the SHA-bound Gaussian-render mapping-only
  scene calibration. The calibration-binding artifact must attest the same
  map SHA, cache SHA, calibration SHA, ordered-query-registry SHA, and count.
  The 4-pixel setting is only the strict diagnostic protocol.

## Feedback-action holdout

- [ ] Generate the split from the exact baseline feedback SHA with
  `--validation-sequence seq2`.
- [ ] Confirm that D2/D3 descriptor gradients, S1 selection, and R1 targets and
  support observations consume only the training side of that split.
- [ ] Describe `seq2` only as a feedback-action holdout: the immutable initial
  map was already built with all mapping sequences, including `seq2`.
- [ ] Keep all test queries sealed during feedback generation, proposal
  construction, paired diagnostics, and manual review.

## Independent preregistered panel

Use `scripts/run_v6_feedback_core_pipeline.py` as the only formal runner. Run
one arm per invocation so that every arm has the same parent map and baseline
feedback and no candidate is chained into another candidate.

| Arm | Frozen change | Required weights/scope |
|---|---|---|
| DC | Minimal PoseLib-changing winner set plus minimum-norm descriptor action | Training split only; bounded beam/trust region and clean-winner protection |
| D2 | Exact-identity descriptor P1+P2 | `pose_critical_weight=0`, `tail_query_weight=0` |
| D3 | D2 plus P3 pose/tail weighting | `pose_critical_weight=2`, `tail_query_weight=1`; every other descriptor hyperparameter equals D2 |
| S1 | Layered Anchor selection only | Same baseline; training split only |
| R1 | L1-targeted pure-ray reconstruction only | Same baseline; training targets and support only |

For every arm:

- [ ] Produce a full training checkpoint for F1/F2 audit and fresh feedback.
- [ ] Produce a compact deployment map/metric with the same deployed Anchor
  IDs, coordinates, and baked descriptors but without dense training state.
- [ ] Run fresh v8 F0 feedback on the full checkpoint, not the compact export,
  and use F1 as a robustness audit when representation changed.
- [ ] Produce paired diagnostics against the common baseline and record all
  producer, input, output, and calibration SHAs.
- [ ] Do not apply an automatic hard gate, mutate a winner pointer, or advance
  to a second round. Submit the four preregistered reports to external manual
  review.

`scripts/run_closed_loop_projective_distillation.py` and other historical
closed-loop runners are legacy diagnostic/reproduction paths, not formal V6
entry points.

## Fixed-map virtual probe bank

- [ ] Before a CUDA render, put the active environment's `bin` and `lib`
  directories first in `PATH` and `LD_LIBRARY_PATH`, then run the explicit
  2DGS renderer smoke test.  This prevents a cached extension from silently
  resolving the host's older `libstdc++`.
- [ ] Plan interpolation, bounded perturbation, boundary, reverse-view, and
  ambiguity-targeted poses using only mapping evidence.
- [ ] Select by viewpoint novelty, ambiguity co-visibility, pose-cell coverage,
  and artifact risk rather than surface coverage alone.
- [ ] Render from the immutable Gaussian prior and require RGB, alpha, expected
  depth, and z-buffer certification; record median depth, contribution entropy,
  and depth consistency when available.
- [ ] Apply mild exposure/gamma, blur, noise, resize/compression, and local
  occlusion only as observer stress inputs.
- [ ] Assert that virtual probes never enter the map, Anchor observations,
  Track construction, descriptor fusion, or Track view count.

## Final freeze and test

- [ ] Complete manual review using mapping feedback and the declared
  feedback-action holdout only.
- [ ] Freeze exactly one method configuration and compact deployment artifact,
  including their commit and SHA registries.
- [ ] Run the sealed test set once after that freeze.
- [ ] Report the observed result without changing the chosen arm, thresholds,
  map, or configuration. Do not claim an improvement or state of the art before
  the real artifact and result exist.
