# V4 offline acceleration

This change targets development-time map construction only. It does not alter
the deployed global Top-1 matcher, Anchor selection, Track policy,
triangulation, or PoseLib.

The public pipeline now emits an atomic `build_timing.json` before its final
manifest. It records coarse method stages as well as every subprocess and
parallel shard. This replaces guesses about the current bottleneck with one
machine-readable profile per scene.

The frozen render-only scene runner also aggregates its existing per-stage
ledgers into `build_timing.json`, distinguishes stages executed in the current
invocation from cache hits, and binds that file into `single_seed_result.json`.

Camera-pair matching now keeps the immutable per-camera descriptor table on the
matching GPU when it occupies no more than 40% of currently free memory. The
fallback is the previous per-pair transfer path. A synthetic RTX 3090 replay
with 96 camera tables and 250 pairs reduced the matching kernel wrapper from
0.1222 s to 0.0507 s (2.409x), with exact Top-1 scores and indices.

Shard concurrency is deliberately configurable rather than silently changed.
`--gpu-workers-per-device 1` prevents long memory-heavy shards from loading
multiple models on one GPU. Zero remains the default because a short synthetic
workload was slower when process startup was serialized. Real-scene timing must
justify enabling the bound.

Historical evidence shows why render/observation batching remains the next
large target. The Stairs render-only Track build took 1384.15 s, of which
matching/Track was 241.09 s and triangulation 55.20 s. ShopFacade took 768.10 s,
of which matching/Track was 42.56 s and triangulation 52.08 s. The remaining
1087.86 s and 673.46 s respectively are dominated by the serialized rendered
observation frontend and associated orchestration.

The next implementation decision must therefore use the new profiles. Reuse
of observation and exact pair-match artifacts is safe and high priority.
Batching render/SuperPoint and persistent workers are only accepted after exact
artifact or downstream pose-equivalence validation.

Machine evidence is stored in
`docs/evidence/v4_offline_acceleration_20260820.json`.
