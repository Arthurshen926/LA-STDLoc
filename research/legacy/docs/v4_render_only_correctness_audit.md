# V4 render-only correctness audit (corrected)

Base: `8073cf1`.  This patch preserves the frozen V4 selector and default
method.  In particular, cycle-seeded Precision Core remains an explicit
ablation; the formal runner is unchanged.

## Fixed implementation defects

1. Alpha/depth sampling at fractional sparse rows mixed `floor` in the teacher
   and function graph with nearest-cell `round` in Track/support code.  Both
   raster consumers now share one explicit nearest-cell function.  Integer
   SuperPoint rows remain bitwise unchanged.
2. Virtual-camera identity was only implicit in dataset order and filenames.
   A canonical geometry registry now records its policy, full/selected names,
   pose/K/image-size entries, and a SHA-256 identity.  Base rendering creates
   it; appearance and stability resolve cameras by attested name and validate
   calibration/hash.  `source_mapping_indices` is retained only for archived
   readers and is no longer consumed by these stages.
3. Filename path prefixes are no longer treated as sequence metadata.
   Trajectory-balanced matching requires explicit `sequence_id`; the default
   nearest-camera method is unchanged.

## Attacks and real registry audit

- Dataset order and COLMAP ID mutation cannot change downstream selected
  cameras.
- Registry name/hash/calibration tampering and duplicate geometry fail closed.
- ShopFacade has 231 mapping cameras; geometry order differs from legacy
  filename order in 231/231 positions.  Reversing the downstream dataset list
  still resolves all 231 registry rows exactly.
- No test query or source mapping RGB is used by registry construction.

The centered-pinhole renderer/localizer contract is unchanged.  This audit
does not add off-center or distorted virtual-camera rasterization.

## Verification

- Focused coordinate/registry/cross-stage suite: `51 passed`.
- Clean-commit producer-lineage subset: `31 passed`.
- Full suite: `759 passed, 1 skipped` in `180.24 s`; the skip is the opt-in
  CUDA renderer smoke test.
