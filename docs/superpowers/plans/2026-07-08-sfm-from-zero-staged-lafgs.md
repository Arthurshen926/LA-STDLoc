# SfM-from-zero staged LaFGS reconstruction

## Goal

Implement an explicit three-stage LaFGS reconstruction path that starts from the COLMAP/SfM point cloud at iteration 0, keeps the original RGB/2DGS photometric training and densify/prune as a geometry scaffold, and progressively makes localization-aware feature/PnP/topology losses the dominant objective.

## Implementation Steps

1. Add parser/default support for a named `sfm_from_zero` LaFGS stage schedule.
   - Expose stage boundaries and loss weights in `train_locaware.py` and `train_lafgs.py`.
   - Bootstrap: RGB-dominant scaffold with multi-view feature initialization.
   - Joint: localization/direct/full-bank/PnP active, raw xyz feedback allowed by explicit flag.
   - Refine: localization-dominant, topology enabled, RGB retained as a weak stabilizer.

2. Integrate RGB/2DGS densify/prune into `train_locaware.py`.
   - Reuse the same densification bookkeeping as `train.py`.
   - Gate it behind an explicit LaFGS RGB scaffold flag and stop it before localization/topology refinement dominates.
   - Refresh feature/geometry anchors after point-count changes.

3. Add runner support for from-zero full experiments.
   - `--lafgs_from_sfm_zero` should not emit `--load_iteration`.
   - Final iteration should be the requested total, not `baseline + steps`.
   - Checkpoint localization should use absolute reconstruction iterations such as 5k/10k/.../30k.
   - Baseline detector retraining remains available as an independent control, not a dependency for LaFGS initialization.

4. Add tests first, then implement.
   - Runner command tests for no baseline load, raw xyz/topology/RGB scaffold flags, and checkpoint iterations.
   - Parser/default tests for the stage schedule and RGB densification controls.
   - Focused pytest verification plus a dry-run command for the ShopFacade experiment.

