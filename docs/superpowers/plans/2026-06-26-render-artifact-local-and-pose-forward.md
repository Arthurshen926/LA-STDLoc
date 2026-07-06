# P18/P19 Render Artifact Ablations

## Goal
Close the render-artifact confound without changing the default LA-STDLoc path:

- P18: suppress teacher supervision locally at projected landmark regions instead of weighting the whole query image.
- P19: evaluate small forward-pose render candidates in an audit-only path before any training integration.

## P18 Design
1. Extend the render artifact module with a sidecar `ArtifactRegionWeightLookup`.
2. Extend artifact audit to optionally emit per-image `.pt` local weight maps plus a manifest.
3. Add `direct_landmark_teacher(..., artifact_weight_map=...)`, sample the map at selected `target_uv`, and pass per-anchor weights into existing descriptor, multiview, full-bank, and anchor losses.
4. Add train CLI args for region-weight manifest/targets/min/power; keep disabled by default.
5. Add tests proving bad-region landmarks are down-weighted while clean landmarks keep weight.

## P19 Design
1. Add a standalone audit script that evaluates original pose plus small camera-forward offsets.
2. Report per-candidate render metrics and best candidate by continuous artifact quality.
3. Do not alter training labels or final test evaluation.

## Verification
1. Unit tests for region map loading/sampling, direct teacher weighting, audit map generation, and script args.
2. Syntax checks for touched scripts.
3. Small smoke/100-step comparison on ShopFacade and OldHospital using existing Cambridge preprocessed data.
