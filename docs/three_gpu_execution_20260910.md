# Active three-GPU AnyGSLoc execution

User explicitly allocated all GPUs on 2026-09-10. Active root:
`/mnt/pool/sqy/anygsloc_three_gpu_20260910`.
The preceding single-GPU watchdog, worker and wait-only helper were stopped;
the original office map PID 25984 remains alive. No original reconstruction
or dataset artifact was deleted. All new outputs use this fresh root.

## Lanes

- GPU 0, 44 stages: redkitchen attestation/evaluation, existing office
  completion/evaluation, stairs/pumpkin recovery, then GreatCourt priors.
- GPU 1, 48 stages: StMarysChurch first (including Long-LRM and i3DGS for
  canopy comparisons), then OldHospital corrected prior matrix.
- GPU 2, 48 stages: ShopFacade first, then KingsCollege corrected priors.

Each lane has independent atomic state/heartbeat, logs and a watchdog. DAG
generation validates all dependencies stay within their lane and precede
their consumers. A slow primary map no longer blocks GPU 1/2 reconstructions.
Every scene includes raw/+1k Long-LRM and i3DGS, Vanilla 2DGS and 3DGS,
except the two completed corrected Vanilla 2DGS scene references which are
not retrained. No AnySplat or offline feedback. This covers the agreed
remaining primary + Cambridge prior matrix, not unimplemented paper ablations.

The ShopFacade capped-memory failure is preserved. Its GPU 2 retry starts
fresh, with the original seed, 30k schedule and corrected inputs, **without**
the 55% memory cap. It is deliberately not described as bit-identical
checkpoint continuation (the upstream checkpoint omits RNG/view-stack state).
Each lane ordinarily has one heavy GPU job; office is an inherited CPU-heavy
map with GPU 0 residency. No arbitrary same-GPU full-size training concurrency.

## Frozen implementation

The previous immutable algorithm snapshot was copied, and only the queue
dispatcher was updated to bind physical GPU 0/1/2. All files were then hashed
and made read-only. Each map-stage `--gpu`, environment, cwd and Python path
is bound to its lane/snapshot. External pretrained models and data are not
duplicated; they remain subject to availability/hash checks. Actual algorithm
and input checks remain active. Legacy recovery audit compares the original
immutable snapshot, identical except for queue dispatch, rather than mutable
cleanup scripts in the development workspace.

## Verification and current handoff

Six queue/recovery tests passed; DAG uniqueness/topology checks passed.
i3DGS GPU-enabled CLI import/argument preflight passed. Three watchdogs were
launched and all three first child processes were observed alive. Long-LRM
has loaded the 32 StMarysChurch inputs; this is not yet a completed prior.
Use each `gpuN/state.json` plus liveness, not stale previous queue records,
to report completion. Shared-hardware timing is descriptive until isolated
deployment benchmarking. No result is considered successful merely because
its process exists or its job was submitted.

`live_results.json` is refreshed by a CPU-only collector: lane counts,
failures/blocked jobs and completed evaluation metrics with map/result hashes.
Only successful evaluation jobs enter the result table. This ledger covers
new outputs; existing completed reference scenes are not counted again.
