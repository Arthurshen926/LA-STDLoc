# Frozen-code GPU 0 handoff

Active queue: `/mnt/pool/sqy/anygsloc_frozen_execution_20260910`.
It supersedes the scheduler at `anygsloc_execution_queue_20260909`.
The old watchdog/worker were deliberately terminated after verifying their
command lines; the already running office map PID 25984 was retained and is
not restarted. No GPU 1/2 process was signalled.

## Isolation

571 code/configuration files were copied (not linked) to the active queue's
`code` directory and made read-only. `code_manifest.json` records every file
SHA. Local cwd, PYTHONPATH, config and entrypoint paths in new jobs point to
the snapshot. Native third-party trainers retain their original pinned paths
and environments, which were not modified. Thus newly added cleanup scripts
in the development workspace cannot change this execution's code contract.
Actual input files are still validated; archiving a required file is not
silently ignored. This does not guarantee data remain available if deleted.

## Bounded overlap

The existing office map is currently CPU-heavy with about 8 GiB GPU residency.
ShopFacade Vanilla 2DGS reconstruction starts alongside it on GPU 0 with a
55% PyTorch caching-allocator budget (roughly 13 GiB of the 24 GiB card).
The upstream training entrypoint, full corrected mapping pixels, losses and
30k schedule are unchanged. This budget is not a strict bound on every
external CUDA allocation, nor an assurance against every future OOM.
After this overlap the scheduler waits for office completion before starting
other GPU tasks. Large reconstruction jobs are not arbitrarily co-scheduled.
GPU 1/2 remain reserved for the other session. Any shared-load timings are
descriptive, not isolated latency certification.

## Recovery rather than recomputation

- office: wait for original process completion; verify map/metric/graph SHA;
  evaluate the original completed map from the frozen code.
- redkitchen: compare all recorded stage input hashes and original algorithm
  modules; verify complete map bundle; then evaluate without rebuilding.
- stairs/pumpkin: audit observation-stage dependencies, reuse generated cache,
  rerun V2 audits and materialization using the frozen implementation.
- office2_5b: already complete; not repeated.

The recovery audit exempts only three explicitly named cleanup/archive
scripts, not the localization implementation or data. Tests confirm actual
module/input changes still fail. The audit establishes present content
equivalence; it does not claim uninterrupted historical file monitoring.
Six recovery/queue tests passed.

The new DAG has 140 stages including explicit audit/recovery steps. Existing
completed prior references are retained. A running/queued reconstruction is
not counted as a successful result. Check `state.json`, child liveness and
heartbeat in this active root, not the superseded scheduler's stale records.
