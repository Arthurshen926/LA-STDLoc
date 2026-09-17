# Remaining evidence: execution and accounting contract

## Baselines

Recovery root: `/mnt/pool/sqy/anygsloc_cambridge_recovery_v3_20260915`.
Five Cambridge scenes, STDLoc and ULFLoc, mapping semantics on/off, no query
semantics. Both coarse and fine results and three warmed timing rounds.
125 recovered tasks include five MAtCHA conversion-cost builds. Preserve all old
failed artifacts. Dataset alias fixes a case-sensitive ULFLoc dataset heuristic;
explicit compiler/bin/library paths fix Ninja, crypt.h and libstdc++ discovery.
ShopFacade full training gates the remaining training jobs for each method.
These are startup controls, not claims that the entire reproduction is validated.

## Geometry

`/mnt/pool/sqy/anygsloc_center_control_20260915`: five CPU builds, fifteen GPU
evaluations after the recovery GPU lane finishes. Ray/depth/center use identical
retained Anchor IDs and descriptors. Center projection uses positive camera Z,
one-pixel nearest-Z buffer, nearest projected center within two pixels, and at
least three observations with common valid depth/alpha/center support.
Coordinates from depth/centers use coordinatewise median over the same valid
observations. Triangulation retains its original estimate from the source Track.
Record support loss. This is a controlled coordinate-source comparison, not a
Track-free full-method baseline. Center Z-buffer approximates visibility and does
not represent alpha-composited Gaussian visibility. Two synthetic tests cover
occlusion ordering and world-to-camera/half-pixel conventions.

## Cost table (35 prior/scene rows, plus baseline variants)

Columns: input preparation; semantic preparation; prior inference/reconstruction;
optional +1k optimization; coordinate/export preparation; AnyGSLoc conversion
(render, V2 audit, association/triangulation/fusion/completion, selection/export);
total build wall time; hardware; peak memory; asset lineage; timing quality.
Always include the upstream Gaussian cost in the end-to-end AnyGSLoc view, while
retaining a separate supplied-prior incremental view. Raw and +1k totals share
the raw reconstruction cost; do not treat +1k optimization alone as total cost.

STDLoc/ULFLoc jointly train Gaussian and localization features: record that as a
joint reconstruction stage, not an artificially separated RGB-prior stage. Then
record their detector or sampling/fusion stages and measured full build wall time.
Instrumented stage times are nested in full build time; do not add them twice.
Semantic mask generation is upstream even when not part of AnyGSLoc conversion.
Missing semantic/preparation durations prevent calling a partial sum a complete
end-to-end cost. Learned network pretraining is separate from per-scene build.

Initial historical inventory: `/mnt/pool/sqy/anygsloc_build_cost_audit_20260915`.
Queue timings have polling/startup overhead and may reflect concurrent workloads.
Before summing, verify each timed reconstruction produced the actual PLY used by
the reported localization map. Never mix times from an earlier failed geometry
or another reconstruction variant. Missing/ambiguous values stay null, not zero.
MAtCHA export manifests identify source PLYs but are not reconstruction timers.
Historical inventory is preliminary, not a completed 35-cell cost comparison.
