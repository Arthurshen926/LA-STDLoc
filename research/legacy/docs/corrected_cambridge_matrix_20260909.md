# Corrected prior matrix: execution boundary

User authorized the six missing primary scenes and five-scene corrected
Cambridge prior comparison. Existing G4Splat jobs take precedence; no GPU
assignment has been agreed yet. No GPU job was launched in this turn.

CPU preparation is running sequentially at nice 15, with CUDA disabled and
one CPU thread, under:
`/mnt/pool/sqy/anygsloc_corrected_cambridge_matrix_20260909/inputs`.
`preparation.json` records actual stage status. The process preserves a
250 GiB shared-disk free-space reserve and refuses existing destinations.
This is input preparation, not a submitted GPU experiment queue.

## Intended matrix

All five Cambridge scenes: Vanilla 2DGS, Vanilla 3DGS, Long-LRM raw and +1k,
i3DGS initial and +1k. The two completed corrected Vanilla 2DGS scenes from
20260908 are reusable after provenance checks. No AnySplat reconstruction.
All new initialization and +1k supervision must use corrected pixels.
Long-LRM keeps its 32-view initialization contract; i3DGS keeps its own
mapping-stream reconstruction. These are pipeline comparisons, not claims of
equal initial compute/data. Frozen MAtCHA is a reference, not retrained here.

Prepared i3DGS d1/d2 and Long-LRM pose32 inputs explicitly use
`input_image_domain=rectified`; input manifests attest mapping membership and
pixel hashes. Vanilla sparse initialization must be independently bound to
its intended SfM source; do not silently substitute the i3DGS adapter model.

## Six missing primary scenes

- 7Scenes office: observation cache and two V2 shards exist; map unfinished.
- 12Scenes office2_5a: same state.
- 7Scenes pumpkin/redkitchen/stairs and 12Scenes office2_5b: no observation
  cache or complete map in the selected primary directories.

Legacy stage outputs lack the new stage-contract attestation. Do not bypass
the new refusal to reuse them. Either audit cache/prior/camera bindings before
an explicit legacy replay in a fresh map directory, or regenerate in a fresh
run. Do not delete partial directories or mislabel old branch results.

## Resource and reporting rules

Wait for an explicitly assigned GPU; momentary zero utilization is not a
reservation. No stopping G4Splat jobs, modifying its repository, upgrading the
shared environment, or concurrently installing rasterizers. Limit CPU/I/O
concurrency. Accuracy uses full test and seeds 2026/2027 with fixed Base/F0,
no offline feedback or online dense rendering. Shared-load timing must be
marked descriptive; isolated timing is a separate deployment measurement.
