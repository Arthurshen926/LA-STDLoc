# Cambridge semantic-on deployment timing

This independent wrapper benchmarks the accepted repaired five-scene, pure-GS,
unmasked AnyGSLoc Full and same-pool native Compact maps. It does not reconstruct,
select, change descriptors, or edit the scientific Localizer.

**Precomputed semantic masks are inputs. Semantic segmentation-network inference
time is NOT included.** The masks.pkl file is loaded once during initialization.
Per-query CPU conversion, nearest-neighbor resize and object/sky/distortion
intersection are timed separately; GPU transfer/application belongs to core.

## Frozen procedure
- Same 1,918 test queries, same order, SP2048, seed2026, calibrated scene PnP
  thresholds, unchanged Localizer and SuperPoint on every call.
- 50 actual warmup calls before each of three complete query rounds, batch one.
  Warmup names and every measured query (including failures) are retained.
- A single physical GPU UUID for all ten scene/configuration cells, serial.
  An explicit, unexpired parent release is required after indoor jobs finish.
  Release is a prerequisite record, not a substitute for user authorization.
- Before CUDA initialization the selected GPU must have no compute processes.
  After initialization the sole new process is recorded; one-second nvidia-smi
  samples reject additional processes. CPU load, memory, affinity and disk counters
  are recorded, but CPU/disk are not claimed exclusively reserved. Sampling
  cannot prove absence of arbitrarily short external work between samples.
- No query-feature cache, no CUDA empty_cache inside the loop, no PnP patch.

## Timing and storage
core = unchanged Localizer.runtime_ms, including H2D, mask application, SP,
global cosine matching, GPU-to-host correspondences and one PnP call.
frame = synchronized wall time for RGB decode/preprocess, mask composition and
the complete localize call. It excludes metric calculation, JSON writes, model/
map initialization and mask-network inference. Never add core to frame.
Separate per-query decode and mask times, raw per-round rows, pooled distributions
and scene-equal means are retained. Three rounds are not three scenes.
Initialization reports dataset/mask loading and model/map/device time.
CUDA allocated/reserved peaks include resident map/model. Initialization,
each round's warmup, and each round's measured queries have separate peaks. Device-wide memory and host
RSS are distinct telemetry. OS filesystem caches are left untouched.
Storage is actual deployment_map.pt bytes, with per-scene, total and mean values.
Shared SP weights and offline Gaussian/cache assets are not map bytes; they must
not be presented as included. Decimal MB means bytes / 1e6; MiB means / 2**20.

## Preparation (CPU only)
python -m pytest -q tests/test_cambridge_deployment_benchmark.py
python -m experiments.cambridge_deployment_benchmark prepare --root NEW_ROOT

No evaluate call is authorized by preparation. After independent review and
parent GPU release, launch each cell with CUDA_VISIBLE_DEVICES=the released GPU
UUID, keeping Localizer's device cuda:0. Release JSON fields: approved=true,
scope=cambridge_full_compact_formal_timing, protocol_sha256, gpu_uuid,
approval_reference, valid_from_unix, valid_until_unix (at most two hours).
Do not create approval evidence before the actual release.
python -m experiments.cambridge_deployment_benchmark evaluate --root NEW_ROOT
  --scene SCENE --configuration full_or_compact --release RELEASE_JSON
python -m experiments.cambridge_deployment_benchmark report --root NEW_ROOT

Existing output directories are never overwritten. A failed attempt is preserved;
a stale lock needs an explicit process-state check and operator decision.
Old MAtCHA 66.1753 ms is not transferable to these maps/configurations.
