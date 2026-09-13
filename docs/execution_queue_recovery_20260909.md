# GPU 0 execution queue recovery

The old `office` replay had a stale running record, empty log, no completed
map and no surviving process. The original termination cause is unknown;
neither OOM nor a specific signal has been established. Its record is now
explicitly interrupted; no old data was deleted.

## Live execution

Queue root: `/mnt/pool/sqy/anygsloc_execution_queue_20260909`.
The detached watchdog owns a lock-protected serial worker on GPU 0 only.
`jobs.json` contains 137 executable stages: five missing primary scenes,
18 fresh Cambridge reconstructions, exports, ten +1k optimizations and
mapping/evaluation stages. Two completed corrected Vanilla 2DGS scene results
remain references and are not retrained. A finished queue is not necessarily
successful: inspect `all_complete` and each job's terminal status.

Every child has an exit code, process identity and heartbeat. Output existence
is checked after successful exit. Failed jobs block dependents only; independent
jobs continue. Surviving orphan children are waited for, not duplicated.
Uncollected completions are conservatively marked interrupted and require an
artifact audit. A watchdog retries worker crashes at most three times; this is
not a guarantee against whole-machine shutdown or both processes being killed.
No destructive retries or silent reuse of partial outputs.

Primary scene order: office2_5b, stairs, pumpkin, redkitchen, office. Office
uses its original cache/V2 shards in an explicit legacy materializer replay,
with cache SHA and shard-scope validation; the original cache lacks new prior
binding and is not retrospectively claimed to have it. Other scenes use fresh
run directories and new stage contracts. Each scene gets seeds 2026/2027.

Cambridge scene order: ShopFacade, OldHospital, KingsCollege, StMarysChurch,
GreatCourt. Each gets Long-LRM raw/+1k, i3DGS initial/+1k and Vanilla 2DGS/3DGS,
except the two already completed Vanilla 2DGS reconstructions. +1k starts from
the newly corrected initialization, never the old distorted prior. No AnySplat.
Large GPU jobs are serial, CPU input preparation runs concurrently at low
priority. No modification of G4Splat or GPU 1/2 jobs.

## GreatCourt exception

`seq5/frame00297.png` is listed in the official mapping split and its raw image
exists, but the supplied COLMAP registry omits it and processed RGB is absent.
The historical MAtCHA preparation summary already reports `skipped_missing: 1`.
Why the upstream registry omitted it is not established by this audit.

An independent dataset view at
`/mnt/pool/sqy/anygsloc_registered_matrix_20260909/datasets/GreatCourt`
contains an explicitly filtered mapping list (1531 registered images) and
`registration_exception.json`. Other files, including all test data, link to
the original dataset. Original lists/raw images are untouched. This aligns the
new prior matrix with the registered camera set used by the existing mainline;
it is not a claim of recovering the unregistered image's pose. No test GT is
used to create a pose. Recovering that extra view would be a separate protocol.

New Vanilla inputs for GreatCourt/ShopFacade/StMarysChurch retain only SfM
points with complete original tracks confined to mapping and length >= 2,
matching the historical mapping-only initialization policy. Retained point
counts: 100130 / 25042 / 193428. Corrected RGB is linked without resampling.
OldHospital/KingsCollege reuse the already audited corrected inputs.

## Verification

Queue failure-isolation, missing-output and stale-running tests, plus image
domain/split checks: 13 passed. DAG generation validates unique IDs and
topological dependencies. All executable paths exist. Actual first job was
launched and heartbeat/process liveness checked. Full job success still needs
the actual results; this document does not mark queued experiments complete.
