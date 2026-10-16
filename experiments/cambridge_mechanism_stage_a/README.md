# Cambridge mechanism diagnosis: Stage A

This isolated experiment is four-scene development, not a replacement benchmark.
StMarysChurch is excluded because its per-camera q/t inconsistency affected
training cameras as well as evaluation GT. No retraining or source asset repair occurs.

Five fixed configurations per scene:
- old_native: existing vanilla3DGS compact-v2 map, freshly evaluated.
- old_spatial_B: existing alpha-gated full map, spatial subset at old native count B.
- new_native: existing unmasked full map, unchanged adaptive compact-v2 selector.
- new_spatial_B: unmasked full map, spatial subset at the same frozen B.
- new_spatial_N: unmasked full map, spatial subset at its actual native count N.

B: ShopFacade 12014, OldHospital 9224, KingsCollege 7735, GreatCourt 44631.
N is not forced to equal B. The corresponding same-count comparisons are
old_native vs old_spatial_B and new_native vs new_spatial_N. The fixed-budget
old/new observation-pipeline comparison is old_spatial_B vs new_spatial_B.
This compares the frontend plus its historical cache representation/implementation
package, not a strictly isolated alpha-gating intervention. Historical caches store
dense alpha in float16 and depth in float32; the new cache stores keypoint samples
in float32. Completion's alpha threshold can therefore differ near 0.05. Raster
sampling is nearest-index (round/clamp), not interpolation. Actual differences
are measured separately; no cache is regenerated to force a pure-factor label.
Same-full-map native/spatial comparisons remain selector-only controls. Numerical
code, frozen protocol, source maps, and all five predeclared configurations stay
unchanged.

No classic-native selector is invented. The existing classic artifact lacks the
native identity/geometry reliability and covariance evidence. Stage B is not
enabled by this entry point.

New native-selector evidence is reconstructed exactly from saved Any parts.
Quality is saved identity_reliability * geometry_reliability, covariance is saved
unchanged, CSR local rows are remapped to unfiltered source cache rows.
No dummy qualities or test data are supplied. The original full XYZ/features
must match bitwise before any selection is allowed.

Current query RGB/GT/masks are frozen and verified against the prior matrix in
the evaluation step. All twenty evaluations are fresh, because the historical
query-on campaign did not hash each RGB at its execution time. Historical
results are references only. Shared dataset metadata loading may parse test
poses; mapping inputs are whitelist-only and selectors never receive test poses.

CPU tests:
python -m pytest -q tests/test_cambridge_mechanism_stage_a.py tests/test_cambridge_input_matrix.py

CLI stages:
python -m experiments.cambridge_mechanism_stage_a.driver prepare --root NEW_ROOT
python -m experiments.cambridge_mechanism_stage_a.driver preflight --root NEW_ROOT --scene ShopFacade
python -m experiments.cambridge_mechanism_stage_a.driver select --root NEW_ROOT --scene ShopFacade --kind old
python -m experiments.cambridge_mechanism_stage_a.driver select --root NEW_ROOT --scene ShopFacade --kind new_spatial_B
python -m experiments.cambridge_mechanism_stage_a.driver select --root NEW_ROOT --scene ShopFacade --kind new_native
python -m experiments.cambridge_mechanism_stage_a.driver evaluate --root NEW_ROOT --scene ShopFacade --config old_native --device cuda:0
Repeat evaluate for the other four configurations after independent preflight review.

Each Runner-owned stage must finish within one hour. Do not restart a still-live
child, kill other workloads, use unauthorized detachment, or overwrite frozen
outputs. ShopFacade is a fixed engineering smoke, not a score-selected scene.
Use at most three scene lanes, four torch CPU threads per process, no more than
24 aggregate workers. Full-cache hashes are disk IO, not GPU work. Old large
cache geometry is read with the audited selective tensor reader.

Recorded runtime_ms and loop wall are parallel, unwarmed telemetry, not a formal
real-time benchmark. No paper or production numerical kernel is modified.
