# Functional ablations, registered before new evaluations

## Claims and fixed protocol

Organize the method as geometry, observations, and compact selection. Do not
claim that each internal safeguard independently improves median pose error.
All five Cambridge scenes, seed 2026, frozen Fast settings, no query semantics,
geometry-only rectification support. No test-driven map selection or thresholds.
Previously inspected test results make this a development ablation, not a new
held-out confirmation. Publish all registered cells, including negative results.

## Observation and selection factorial (implemented)

V2 on/off x robust multi-view fusion/single deterministic observation x
projective completion on/off x full/compact map: 16 cells per scene, 80 total.
V2-off uses already rebuilt associations, not deletion of a mask after mapping.
Disable completion BEFORE compact selection. Fusion-off uses the lexicographically
first mapping observation, never a test-selected descriptor. Matching-quality
scores remain those of the source map: this isolates exported descriptor fusion,
not removal of every descriptor-dependent statistic in the construction pipeline.
Same frozen selector parameters in every cell; no retraining or per-cell retuning.

Report each factor's paired effect conditional on the other factors, and V2 x
fusion, fusion x completion, observations x selection interactions. A combined
V2+fusion+completion effect supports the observation block, NOT each member.
Full factorial complements leave-one-out; more combinations do not guarantee
stronger evidence. No post-hoc exclusion of cells or metric shopping.

Metrics: scene median TE/RE, R5/R15, P90 error, failure rate; anchor count,
deployment-map bytes (shared model weights separate), map reduction, warmed core
latency mean/P50/P95, >100ms fraction, full-loop throughput, peak allocated GPU
memory, selector wall time. Always distinguish reciprocal core mean from actual
end-to-end FPS. Three timing rounds, one accuracy seed. GPU0 serial execution;
record contention before claiming isolated hardware benchmarks.

## Geometry controls (pending implementation, NOT queued by this factorial)

Compare fixed-camera multi-ray XYZ, rendered-depth XYZ, and projected Gaussian
center XYZ. Define center visibility/rasterization, pixel tolerance, collisions,
and multi-view aggregation explicitly; never use query GT or match a center to
the triangulated answer in 3D. Keep rendered appearance descriptors identical.
Use common valid Anchor IDs for a coordinate-estimator comparison and report
excluded coverage. Separately use each source's native support for a whole-map
comparison; a common-track XYZ swap does not ablate association/Track construction.
Do not mislabel depth backprojection as Gaussian-center projection.

Geometry x observations x selection overview: 3 geometry sources x 2 observation
block states x 2 selection states = 12 cells/scene. Only materialize after center
definition and matching coordinate conventions pass synthetic tests. Existing
internal geometry leave-one-out experiments go in supplementary evidence.

## Reporting layout

Main paper: three-block combinations plus geometry-source table; selector shown
as accuracy/storage/latency tradeoff, not an asserted accuracy booster. Supplement:
complete observation factorial and internal geometry safety checks. Keep exact
source hashes and distinguish reused map assets from fresh query evaluations.
