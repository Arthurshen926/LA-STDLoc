# LaFGS Feedback Path Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LaFGS reconstruction use localization feedback with correct scale, safer geometry supervision, and detector targets that prefer geometrically useful landmarks.

**Architecture:** Keep the current 2DGS/LaFGS pipeline, but repair the failing paths identified by the 5k A3 experiment. The reconstruction path gets measurable weighted PnP contributions, world-scale surface anchor bounds, and filtered feature/geometry supervision. Detector sampling uses localization utility plus GT reprojection/depth/pose diagnostics instead of self-confidence alone.

**Tech Stack:** Python, PyTorch, existing STDLoc LaFGS training and pytest suites.

---

### Task 1: Surface Anchor Scale and PnP Contribution Diagnostics

**Files:**
- Modify: `scene/gaussian_model.py`
- Modify: `train_locaware.py`
- Modify: `train_lafgs.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [ ] Add tests proving a positive `surfel_loc_radius_floor` gives world-scale displacement even when Gaussian scale is tiny.
- [ ] Add tests proving LaFGS defaults keep PnP feedback at useful weight and expose weighted loss diagnostics.
- [ ] Implement `surfel_loc_radius_floor` in `get_loc_xyz`.
- [ ] Log weighted PnP component contributions and loc-anchor movement stats.

### Task 2: Geometry Feedback Filtering and Feature Match Supervision

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Modify: `train_locaware.py`
- Modify: `train_lafgs.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [ ] Add tests proving geometry match reprojection loss gives gradients to Gaussian features, while geometry reprojection still updates geometry only.
- [ ] Add confidence/margin/entropy/reprojection filtering to both geometry and feature-match branches.
- [ ] Enable conservative geometry-match feature supervision in guarded runner defaults.

### Task 3: Full-Bank and Detector Geometric Quality Gates

**Files:**
- Modify: `localization_training/direct_landmark_teacher.py`
- Modify: `scene/gaussian_model.py`
- Modify: `localization_training/landmark_distill.py`
- Modify: `train_detector.py`
- Test: `tests/test_localization_utility.py`
- Test: `tests/test_detector_soft_targets.py`

- [ ] Add tests proving localization utility penalizes high reprojection error and low pose information.
- [ ] Add tests proving detector landmark metadata can down-weight poor GT-reprojection landmarks.
- [ ] Bias full-bank weights with pose-information by default for LaFGS.
- [ ] Add detector utility quality composition and optional hard caps for low-quality landmarks.

### Task 4: Experiment Runner Defaults

**Files:**
- Modify: `scripts/run_lafgs_cambridge_guarded_pnp.py`
- Test: `tests/test_lafgs_cambridge_experiments.py`

- [ ] Add tests proving guarded PnP experiments use `loc_interval=1`, non-tiny PnP weight, radius floor, conservative filters, and same detector config for baseline/LaFGS.
- [ ] Update runner defaults and manifest command logging.

### Task 5: Verification and Probe

**Files:**
- No new production files expected.

- [ ] Run focused pytest for all touched areas.
- [ ] Run a short ShopFacade probe to verify weighted PnP contribution, anchor movement ratio, and filtered match counts move in the expected direction.
- [ ] Report whether the probe fixes the previously observed “PnP connected but powerless” failure.
