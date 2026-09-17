# Baseline compatibility recovery v4

Root: `/mnt/pool/sqy/anygsloc_baseline_recovery_v4_20260915`.
Original pinned baseline sources and failed outputs are preserved. This run has
independent source copies; exact changed-source and reused-asset SHA256 values
are in `compatibility_recovery.json`.

STDLoc renderer: adapt gsplat `[camera, point, xy]` integer radii to the scalar
maximum radius expected by old visibility and densification code. Scalar-radii
versions remain unchanged. Native RGB/descriptor forward AND backward passed a
CUDA synthetic smoke test (`tools/smoke_stdloc_renderer.py`). No matching/loss or
refinement thresholds changed.

ULF-Loc fusion: use its existing `ULFLOC_FUSION_BATCH_SIZE=1` control instead of
10. Preserve image resolution, bilinear upsampling, grid sampling and cosine
weights. Only batching and floating-point summation order change. Weighted
reduction equivalence checked within float32 tolerance, not bitwise identity.
The complete ShopFacade recovery fused all 231 mapping images successfully.

Only ULFLoc/ShopFacade/mapping-semantics-off reuses its completed 30k PLY AND
sampled indices from v3. The native postprocessing path does not retrain or
resample those points. Its new training receipt is explicitly POSTPROCESSING
ONLY; it must not be compared to another method's full build wall time. Prior
training/sampling costs remain in earlier evidence and need stage reconciliation.
All other method/scene/semantic conditions keep native full reconstruction.

Other training jobs depend on that method's representative ShopFacade coarse+
fine evaluation, not merely successful import or first training step. All 120
remaining tasks are registered; previous five successful MAtCHA cost builds are
not rerun. Native baseline reporter runs in the background and validates query
coverage before producing accuracy rows. GPU0 only; SSH-detached watchdog.

This document records fixes and execution protocol, not a claim that all jobs
have completed. Inspect `gpu0/state.json` and `report_status.json` for completion.
