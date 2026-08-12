# V4 Geometry Materializer Compatibility

## Decision

The geometry path now has one compatibility boundary:

```text
image triangulation + optional accepted surface evidence + surface fallback
                              |
                              v
              materialize_geometry(anchor_evidence)
                              |
                              v
        xyz + covariance + geometry_mode + surface_dependence
```

This is a structural convergence only.  It does not introduce a new geometry
solver, change a coordinate, alter the selector, replace a descriptor, or make
a precision claim.

## Frozen policy

The API makes the existing V3/P5 behavior explicit:

1. an image-stable Track uses its image-only point and uncertainty;
2. a weak Track uses surface-regularized geometry only when that upstream
   update was accepted;
3. a non-Track fallback retains its already materialized surface position;
4. evidence availability and deployed dependence are separate fields.  A
   Track may have surface evidence while its deployed geometry remains purely
   image-triangulated.

`topology/adaptive_distillation.py` now routes the existing Track deployment
choice through this API.  `topology/anchor_registry.py` uses the same API to
annotate an already materialized map and exposes `surface_dependence`.  The
legacy `anchor_type` remains untouched for compatibility.

## Parity result

The audit compares the new implementation against the previous five-field
replacement literally, including tensor dtype and byte representation.  It
also checks the selected Track coordinates against the frozen Map and runs the
Registry localization-tensor compatibility validator.

| Frozen mapping artifact | Anchors | Image-triangulated | Surface-regularized Track | Surface fallback | Result |
|---|---:|---:|---:|---:|---|
| Heads frozen V3/P5.0 | 8,119 | 7,126 | 0 | 993 | PASS |
| Stairs frozen V3/P5.0 | 7,275 | 2,480 | 0 | 4,795 | PASS |
| ShopFacade P5.1 outdoor guard | 6,361 | 5,717 | 0 | 644 | PASS |

For every available legacy field, parity is bitwise:

- `triangulated_xyz`;
- covariance trace and covariance matrix when present;
- reprojection median and P90;
- selected Track coordinates in the final Map;
- every pre-existing localization-facing Registry tensor.

The old V3 maps retain only the Track-core count, not exact core row IDs.  Their
Track payloads also contain no alternative image-only/surface-regularized
fields, so the compatibility materializer is necessarily the identity on all
Track rows.  The newer P5.1 Heads and Stairs artifacts additionally retain
exact core row provenance; they pass the same five-field bitwise audit.  A
synthetic contract exercises the otherwise dormant three-way decision.

The mapping-only reports are stored at:

- `/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/audits/geometry_materializer/heads_v3.json`
- `/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/audits/geometry_materializer/stairs_v3.json`
- `/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/audits/geometry_materializer/shopfacade.json`

## Important audit finding

All three frozen payloads contain zero accepted surface-regularized Tracks.
The code path exists and its three-way behavior is covered by a synthetic
contract, but it is dormant in these deployed artifacts.  The large Stairs
surface count is therefore not evidence that Gaussian depth repaired weak
Tracks: it consists of surface-initialized fallback Anchors.

This distinction matters for the paper.  The current validated method is:

```text
image-only Track geometry + selected surface fallback geometry
```

not a demonstrated adaptive fusion of image and Gaussian geometry.  A future
surface-regularization experiment would need its own mapping residual and pose
gate; this compatibility refactor does not authorize it.

## Preserved legacy annotation

The old Registry copies a selected Track's covariance from the final Track
payload even when its deployed core coordinate is image-only.  That covariance
is audit metadata and is not consumed by localization.  The new API preserves
this behavior through an explicit compatibility override instead of silently
changing historical artifacts.  Aligning covariance source and coordinate
source should be a versioned Registry correction, not mixed into a no-behavior-
change refactor.
