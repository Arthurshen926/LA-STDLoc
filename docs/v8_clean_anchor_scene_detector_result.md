# V8 clean-Anchor single-pass scene detector result

## Decision

The synthetic-only scene-specific detector is technically viable but is **not
promoted** to the deployed operating point in this round.

Two stronger claims are rejected:

1. A strict clean map containing only Anchors with at least three V2-valid
   observations from at least two pose families loses too much coverage.
2. A learned low-resolution heatmap cannot replace the native SuperPoint corner
   response directly; doing so collapses keypoints spatially.

The surviving candidate is a single-pass reliability modulation:

```text
one frozen SuperPoint encoder
  -> native corner score * scene-specific reliability
  -> one NMS/Top-K
  -> unchanged SuperPoint descriptors
  -> Full-map exact global Top-1
  -> one standard PoseLib
```

There is no map-conditioned second pass, descriptor adapter, dense refinement,
real-RGB training, test-RGB training, or permanent Gaussian deletion.

## Leakage-safe data

- Scene: StMarysChurch.
- Input: raw Gaussian renders, including renderer artifacts.
- Positive target: spatially balanced projections of depth-consistent clean
  Anchors on V2-valid pixels.
- Negative target: pixels explicitly invalidated by V2.
- Ignore target: borders, depth discontinuities, uncertainty, and uncovered
  content.
- Training/validation/confirmation: 256/128/58 mapping renders.
- Pose-family split: train 7 sequences, validation 2 sequences, confirmation 2
  sequences, with no sequence overlap.
- Confirmation is limited to 58 because the two reserved families contain only
  58 mapping cameras. Validation families were not reused to inflate the count.
- Two seeds were trained on GPUs 1 and 2. Seed 2026 was selected using validation
  loss only; confirmation was then evaluated once.

The selected head's validation positive/negative probability means are
0.805/0.211, with separation 0.593. This verifies that purely synthetic
supervision can learn the clean-Anchor support signal.

## Strict clean-map audit

The candidate map retains 34,437 of 200,255 Anchors (17.20%). Its descriptors
are rebuilt only from original, unmasked rendered observations passing V2.

On the 58-query confirmation set:

| Frontend / map | Median TE | P90 TE | R5 | GT@4px | Spatial cells |
|---|---:|---:|---:|---:|---:|
| Native SuperPoint / strict clean | 1.528 cm | 7.746 cm | 81.03% | 11.99% | 15.98 |
| Scene detector / strict clean | 1.579 cm | 5.904 cm | 84.48% | 12.23% | 15.88 |
| Native SuperPoint / Full | 0.794 cm | 2.576 cm | 98.28% | 35.29% | 15.98 |

The detector repairs part of the strict map's tail, but cannot recover the
coverage removed by hard Anchor filtering. The strict map is rejected.

## Full-map detector isolation

Both arms use the same Full map, frozen descriptors, 2,048-keypoint budget,
exact global Top-1 matcher, and PoseLib call.

| Frontend | Median TE | P90 TE | R5 | GT@4px | Clean-target hit | Spatial cells |
|---|---:|---:|---:|---:|---:|---:|
| Native SuperPoint | 0.794 cm | 2.576 cm | 98.28% | 35.29% | 14.67% | 15.98 |
| Scene reliability modulation | 0.836 cm | 2.568 cm | 100.00% | 35.59% | 15.63% | 15.88 |

The learned detector recovers one R5 failure and loses none, slightly improves
P90, GT@4px, and clean-target allocation, and preserves spatial coverage. It
also worsens aggregate median TE by 0.042 cm. On paired queries, 27 improve and
31 worsen; the paired median delta is +0.017 cm and its 10,000-sample bootstrap
95% interval is [-0.039, +0.069] cm. Therefore the central effect is unresolved
and the preregistered non-regression gate is not passed.

## Failed direct-heatmap arm

Using the upsampled learned heatmap as the complete detector response produced
only one occupied 4x4 cell and 40/40 catastrophic failures in the initial
confirmation probe. The corrected method multiplies native corner responses by
scene reliability. The failed artifact remains available as
`confirmation/isolation_report.json`.

## Safety actions

The implementation now provides reversible contracts for:

- row quarantine: only ACCEPT and V2-valid rows can create feedback evidence;
- Anchor quarantine proposals: repeated evidence from at least two independent
  pose families is required;
- counterfactual Gaussian cleanup: paired rerender/relocalize evidence must
  improve across at least two families with bounded worsening;
- rollback: even a passing counterfactual action authorizes only reversible
  opacity quarantine, never permanent deletion.

These actions are implemented and unit-tested, but no Gaussian quarantine is
accepted by this experiment because no pixel-to-Gaussian contribution artifact
was materialized in this round.

## Method conclusion

V2 should enter the mainline as an observation reliability and detector-target
mechanism, not as a hard map mask. The Full map remains the correct matching
map. Synthetic-only detector training is sufficient to learn artifact-aware
allocation and shows a small tail benefit, but current evidence does not support
replacing native SuperPoint in the deployed operating point.

The real test remains sealed. Running it now would use test to choose an
unpromoted candidate and violate the frozen protocol. The next admissible step
is a larger novel-pose confirmation or a stronger map-derived target that
improves median without sacrificing the observed tail benefit.

## Artifacts

- Contract: `configs/v8_clean_anchor_scene_detector.yaml`
- Strict clean map:
  `/mnt/pool/sqy/lafgs_v8_clean_anchor_scene_detector_20260828/StMarysChurch/clean_map/projective_anchor_map.pt`
- Synthetic dataset:
  `/mnt/pool/sqy/lafgs_v8_clean_anchor_scene_detector_20260828/StMarysChurch/detector_data_v2`
- Selected checkpoint:
  `/mnt/pool/sqy/lafgs_v8_clean_anchor_scene_detector_20260828/StMarysChurch/detector_v2/scene_detector_seed2026.pt`
- Strict-map confirmation:
  `/mnt/pool/sqy/lafgs_v8_clean_anchor_scene_detector_20260828/StMarysChurch/confirmation_v2/clean_map_report.json`
- Full-map confirmation:
  `/mnt/pool/sqy/lafgs_v8_clean_anchor_scene_detector_20260828/StMarysChurch/confirmation_v2/full_map_report.json`
