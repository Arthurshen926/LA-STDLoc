# Cambridge ablation closeout execution

Status update (2026-09-13): the report now records 219/219 complete and no missing
outputs. The execution notes below describe the original launch protocol.
Current interpretation and remaining release gates: `release/evidence_status.md`.

The existing 81 evaluations are retained in
`/mnt/pool/sqy/anygsloc_ablation_closeout_20260911`.
The remaining selection/geometry/Compact-descriptor/completion runs are registered in
`/mnt/pool/sqy/anygsloc_final_ablation_20260913/registry.json` (108 evaluations).
The paired coordinate-source experiment is registered in
`/mnt/pool/sqy/anygsloc_depth_ablation_20260913/registry.json` (30 evaluations).

All new evaluation conditions use Fast with 65% match retention, RANSAC minimum
250, maximum 30000, confidence 0.999, and seeds 2026/2027/2028. The environment
limits OMP/MKL/OpenBLAS to four threads. Concurrent timing is telemetry, not an
isolated deployment latency benchmark. Seeds measure solver randomness, not
independent datasets. These settings follow prior test-set exploration; this is
not an untouched held-out validation.

GreatCourt and StMarysChurch use the recovered observations, freshly rebuilt maps,
and their own Compact selections. The Full comparison explicitly uses that same
new input map. Recovered observations must not silently be indexed using old CSR
rows. Descriptor interventions keep the Compact XYZ and anchor set unchanged.
No-completion removal changes capacity and is reported as an auxiliary ablation.

The geometry-source pair keeps identical anchor rows and descriptors. It retains
anchors with at least three finite positive observed depths and alpha >=0.05.
Each depth is backprojected using the observation's own camera, intrinsics, and
the +0.5 pixel-center convention; the world-coordinate componentwise median is
the replacement point. Both ray and depth branches use the common valid rows.
Excluded counts are reported. This tests this explicitly defined depth baseline,
not every possible Gaussian-to-landmark conversion. Other stored attributes are
held fixed; no view-conditioned refinement or covariance-guided solver is enabled.

The collector updates
`/mnt/pool/sqy/anygsloc_ablation_report_20260913/tables.md` and `status.json` every
minute. Missing outputs remain missing in the report, even if a worker exits.

Preliminary interpretation from completed experiments:

- Compression is a supported benefit; do not claim uniformly lossless accuracy.
- Pose completion must be compared with equal-capacity quality selection. Its
  gains differ by scene and metric; KingsCollege is a negative example.
- V2 and cycle constraints help OldHospital, with smaller/mixed effects elsewhere.
- Iterative ray reweighting has not shown a large consistent effect.
- ShopFacade Compact descriptor fusion is only slightly better than ordinary
  averaging; fixed single-view descriptors are worse. Five-scene evidence pending.
- No conclusion on the paired depth experiment before its outputs are complete.
