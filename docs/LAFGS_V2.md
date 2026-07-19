# LaFGS V2: Causal Localization-Field Reconstruction

## Current status

The progressive shadow-coreset route in `train_coreset_v2.py` is retained only
for historical reproduction. It is not the current mainline: its accepted output
often fell back to the unchanged strong bank, while its old evaluation protocol
did not pin query resolution.

The current mainline is `train_lafgs_map.py`. It reconstructs a localization
descriptor field on a frozen external MAtCha 2DGS surface, evaluates every phase
with a fixed deployment frontend, and advances only after a joint held-out gate.
This is deliberately a causal sequence rather than one monolithic run.

## Representation

The implementation keeps these quantities distinct:

1. `G_base`: frozen external RGB 2DGS geometry, appearance, normals, and primitive
   identities.
2. `S_strong`: protected localization-bank IDs and mature descriptors.
3. `P_extra`: optional geometry-derived surface candidates. A protected union
   appends these without replacing strong identities.
4. `X_base`: immutable surface anchors.
5. `X_current`: bounded tangent/normal reconstruction derived from `X_base`.
6. `base_uv`: projection of `X_base`, fixed for an observation.
7. `measurement_uv`: an independent local image measurement.
8. `predicted_uv`: projection of `X_current`.

Geometry supervision compares `measurement_uv` with `predicted_uv`; moving an
anchor cannot move its own observation target.

## Causal phases

### Phase 0: scaffold and initialization

- Use the exact strong bank as the protected core.
- Optionally append pure-geometry candidates with `protected_union`.
- Aggregate frozen image-encoder features over support views.
- Use robust MV medoids by default. Existing mature descriptors can be aligned
  by exact ID or blended only on overlapping IDs.
- Split support and validation cameras before cache construction. Test cameras
  are never loaded by map training.

### Phase 1: descriptor reconstruction

Geometry is frozen. Training mixes:

- exact base projections;
- local jitter proposals;
- frozen generic keypoint/grid proposals, including unmatched background;
- true full-bank hard retrieval.

Hard retrieval never injects the source landmark into the forward top-K set.
True source and geometric Recall@1/4/16/64 are reported separately. A retrieval
miss may use a separate missed-positive ranking loss, but it does not alter the
candidate set. Set-valued positives treat a geometrically equivalent surfel
within the positive radius as a valid retrieval.

### Phase 2: bounded surface geometry

This phase is allowed only after Phase 1 passes the fixed-frontend gate.
Descriptors are frozen or used as an EMA teacher. Default bounds are 5 mm in
the tangent plane and 2 mm along the normal; stricter 2 mm/1 mm experiments are
supported. Raw 2DGS XYZ remains immutable.

### Phase 3: PoseLayer

Pose gradients have explicit, mutually exclusive routes:

- `feature`: detach 3D points; keep `measurement_uv` and confidence
  differentiable.
- `geometry`: detach measurements and confidence; update bounded anchors only.

Unit tests require the intended parameter group to receive a nonzero gradient
and the other group to receive none. A PoseLayer branch must outperform a
same-length no-pose continuation on deployment candidates before combination.

### Phase 4: distillation

Landmarks are filtered by real global matchability and false-top1 rate first.
Coverage and translation pose information/FIM are second-stage tie breakers.
FIM is not allowed to rescue globally ambiguous descriptors.

### Phase 5: detector

The map and bank are frozen before scene-detector training. A candidate map must
first pass the fixed strong-detector gate; otherwise detector retraining and all
later phases stop, and the exact strong map is selected.

Dustbin remains off unless inference uses the same explicit unmatched decision
as training. Online rendering and dynamic landmark replacement are not part of
the accepted mainline until the real-query descriptor gate passes.

## Joint checkpoint gate

`scripts/select_lafgs_map_checkpoint.py` requires all candidates and the control
to use the same held-out camera subset and the same evaluation protocol hash. A
candidate must simultaneously:

- improve median translation by at least 0.02 cm;
- not worsen median rotation;
- not worsen raw GT precision at 2 px;
- not worsen RANSAC-inlier GT precision at 2 px;
- not worsen translation pose-information logdet.

If no candidate passes, selection returns the strong control state. Test-set
metrics are not used for checkpoint selection.

## Reproducible evaluation protocol

Formal Cambridge evaluation pins:

- `--source_path /mnt/pool/sqy/Cambridge_stdloc/<scene>`;
- `--images processed`;
- `--resolution 1`;
- `--data_device cpu` for DataLoader image decoding;
- the camera subset and split seed;
- detector, landmark IDs, metadata, descriptor state, and map checkpoint hashes.

Every result writes `evaluation_protocol.json`. Its hash covers query names,
loaded shapes, candidate-split settings, and the SHA256 manifest of query image
contents. The selector rejects legacy or mismatched protocols.

This fixed a high-impact reproduction error: the default `resolution=-1`
silently resized 1920-wide Cambridge queries to 1600 pixels. The same strong
artifacts then produced about 3.28 cm instead of the pinned 3.095 cm result.

Camera metadata no longer stores inherited lazy PIL handles. DataLoader workers
open private image handles, and CUDA-backed camera loading automatically uses
the parent process because forked workers cannot initialize CUDA.

## ShopFacade evidence

With the fixed 46-camera temporal validation split and fixed strong detector:

| State | TE (cm) | AE (deg) | Raw P@2 | Inlier P@2 | Translation logdet | Gate |
|---|---:|---:|---:|---:|---:|---|
| strong control | 2.4517 | 0.1283 | 9.0398% | 41.1052% | 12.4668 | control |
| descriptor F1, step 250 | 2.3369 | 0.1267 | 9.0505% | 41.0205% | 12.4725 | reject |
| descriptor F2, step 500 | 2.3988 | 0.1266 | 9.0526% | 41.0397% | 12.4738 | reject |

F1 improved true source Recall@1 from 16.23% to 20.50%, Recall@16 from 41.77%
to 47.93%, and reduced median top-1 reprojection from 10.53 px to 9.09 px.
However, every F1/F2 checkpoint slightly reduced inlier P@2 or failed another
gate condition. The formal selection therefore keeps the strong control and
does not open geometry, PoseLayer, dustbin, or detector retraining for this
branch. This is a negative descriptor-stage result, not evidence for later
modules.

The pinned full-test reference artifacts are:

- safe strong fallback: 3.0951 cm median TE and 0.1411 degree median AE;
- older candidate-aligned numerical best: 3.0470 cm and 0.1577 degree.

The numerical best is retained as a performance reference, while the strong
fallback is the causally selected state for the corrected phased protocol.

## Entrypoints

The formal ShopFacade entry is:

```bash
scripts/run_lafgs_v2_shopfacade.sh descriptor   # corrected Phase 1
scripts/run_lafgs_v2_shopfacade.sh test_strong  # safe pinned fallback
scripts/run_lafgs_v2_shopfacade.sh test_best    # numerical reference
scripts/run_lafgs_v2_shopfacade.sh verify       # regression suite
```

The former progressive route moved to
`scripts/run_lafgs_v2_progressive_legacy_shopfacade.sh` and is available only
through explicit `legacy_*` modes.
