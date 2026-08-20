# Sufficiency-guided virtual rendering planner (mapping-only probe)

This is an opt-in offline planner layered on the corrected V4 camera registry.
It is not connected to the default renderer, Track builder, selector, or test
evaluator.

## Contract

The planner builds a voxel field from rendered alpha/depth support and Track
geometry.  Each voxel records distinct source-camera families, view bins,
stable observations, Track observations, selected-Track support, and capped
deficit demand.  Candidate poses are bounded to:

- geometry-nearest SE(3) interpolation;
- small source-camera translation or rotation;
- trajectory-envelope boundary expansion;
- reverse views;
- deficit-directed gaze from an existing camera center.

Candidates must remain inside the expanded mapping envelope, near a parent
camera, below an artifact-risk threshold, and depth/alpha-continuous with
already supported surface voxels.  Greedy selection maximizes normalized
capped coverage plus non-negative parallax, appearance-continuity, and
artifact-quality terms.  This is a monotone submodular-plus-modular objective.
Ties are deterministic.  Perturbations of one source camera share a pose
family, and selection permits at most one view per family; downstream
triangulation must retain that same capacity rule.

GT-visible is deliberately `null` in the plan.  It may be attached only as a
post-freeze diagnostic and never participate in candidate generation or
selection.

## Synthetic verification

The focused suite covers diminishing returns, monotonicity with a larger
budget, deterministic ties, duplicate-family attacks, bounded six-source
candidate generation, and rejection of test/source-RGB inputs: `5 passed`.

## ShopFacade read-only headroom

Inputs were the existing 231-view rendered appearance cache, repaired Track
payload, and selected map.  No RGB was rendered and no test query was read.
With 0.25 m voxels, stride-32 surface sampling, 256 bounded candidates, and a
32-view budget:

- 77,120 coverage voxels, 62,581 with nonzero deficit;
- initial capped demand 197,015;
- selected 32 unique pose families, covering 57,666 deficit voxels;
- proxy remaining demand 21,096;
- selected kinds: 29 interpolation, 3 small translation;
- mean normalized parallax 0.8159, appearance continuity 0.9999, artifact risk
  0.1477.

This is an optimistic geometric headroom result, not an accuracy result: it
does not prove that the Gaussian will render repeatable SuperPoint evidence at
those views.  The next authorized experiment would render only the 32 frozen
views, then measure mapping-only repeatability/Track gain while preserving pose
family capacity.  It does not justify full-scene rendering yet.

