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

The initial Track build now receives the just-materialized feature-cache
payload in memory. It no longer writes a 2 GiB Stairs cache and immediately
deserializes the same file. Resume runs still load and validate the frozen file.
Track observation coordinates are gathered from one packed ragged table rather
than 1.81 million Python index operations. The real Stairs artifact replay was
bitwise exact and reduced this gather from 8.407 s to 0.052 s; ShopFacade went
from 1.543 s to 0.077 s.

PyTorch CPU threads are now bounded to eight in the standalone render-Track
builder. Thousands of tiny per-Track reductions do not benefit from host-wide
threading: on ShopFacade, four threads fused the full map in 12.38 s versus
37.19 s previously, with bitwise-identical features. A 32-thread 3,000-Track
probe was still running after 60 s, while 1--8 threads took 3.26--3.47 s.

Shard concurrency is deliberately configurable rather than silently changed.
`--gpu-workers-per-device 1` prevents long memory-heavy shards from loading
multiple models on one GPU. Zero remains the default because a short synthetic
workload was slower when process startup was serialized. Real-scene timing must
justify enabling the bound.

Historical evidence was re-read at the source-artifact level. The Stairs
feature-cache stage took 336.03 s (render 113.99 s, SuperPoint 11.70 s), while
the later Track-map stage independently took 1384.15 s. ShopFacade was 38.97 s
and 768.10 s respectively. Therefore the largest unknown is Track
post-processing/artifact handling, not the rendered frontend. Fine-grained
timers have now been added around cache handling, observation materialization,
matching/components, UV gathering, pose bins, triangulation, eligibility,
descriptor fusion and serialization.

An eager packed implementation of all descriptor-fusion rows was also tested
on the real ShopFacade mapping artifacts. It was bitwise exact but slowed the
operation from 37.19 s to 41.55 s, so it was reverted rather than retained.

The final ShopFacade mapping-only replay with eight CPU threads, resident pair
descriptors and packed UV gathering completed the Track-map stage in 105.74 s,
versus the historical 768.10 s record. Every historical Track field was
byte-exact, all Anchor scientific tensors were bitwise exact, and all report
metrics matched. The observed 7.26x combines current runtime/code state; only
the component microbenchmarks should be interpreted as isolated causal
speedups.

A second structural pass removed the remaining Python hot loops without
changing their ordering. Conflict-aware Track assembly now materializes the
already stably-sorted edge columns once before the sequential union pass, uses
an exact integer bitset for per-component camera membership, and maps accepted
pair edges to final Tracks with a dense vectorized lookup. On the same real
ShopFacade artifacts, component assembly fell from 28.44 s to 9.36 s and the
pair sidecar from 11.87 s to 0.49 s. The complete Track build fell from 47.52 s
to 16.91 s (2.81x), while the Track table, diagnostics and complete sidecar
were byte-exact.

Triangulation now computes every immutable observation ray once instead of
re-solving the same camera system per Track, and reuses the already computed
final projection when forming robust covariance weights. A 375,254-observation
ShopFacade ray-only replay reduced 9.75 s to 0.135 s (72.0x) with exact rays;
the complete triangulation stage reduced to 45.60 s and all 29 geometry fields
remained byte-exact.

The resulting full ShopFacade Track-map replay completed in 75.23 s, 1.405x
faster than the preceding 105.74 s optimized run and 10.21x faster than the
768.10 s historical record. The complete Track payload and Anchor map were
byte-exact, not merely metric-equivalent. A hard-scene Stairs Track replay also
remained byte-exact and completed in 102.23 s; its profile is now explicit:
31.57 s pair matching, 60.94 s conflict-aware components, 5.85 s final Track
table and 3.43 s sidecar. This confirms that future optimization should target
the still-sequential conflict-aware union or reuse a completed Track artifact,
not assume the renderer is the dominant cost.

The next implementation decision must therefore use the new profiles. Reuse
of completed observation/Track artifacts across experiment roots is higher
value than pair-match-only caching because the latter would still pay the
conflict-aware union and sidecar costs. Exact pair-match caching remains useful
when the Track policy itself is being iterated.
Batching render/SuperPoint and persistent workers are only accepted after exact
artifact or downstream pose-equivalence validation.

Machine evidence is stored in
`docs/evidence/v4_offline_acceleration_20260820.json`.
