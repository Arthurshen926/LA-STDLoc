# LaFGS Mainline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the LaFGS reconstruction mainline from the attachment as tested code modules and a lightweight CLI surface.

**Architecture:** Add `localization_training/lafgs_reconstruction.py` for MVInit, curriculum policy, soft 3D-to-2D matching, and differentiable PnP losses. Keep the existing renderer, Gaussian model, direct landmark teacher, and STDLoc evaluator intact.

**Tech Stack:** PyTorch, existing `GaussianModel`, existing direct landmark projection helpers, existing weighted Gauss-Newton pose refiner, pytest.

---

### Task 1: MVInit

**Files:**
- Create: `localization_training/lafgs_reconstruction.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests for weighted descriptor aggregation and reliability.
- [x] Implement `MultiViewInitConfig`, `MultiViewInitResult`, `aggregate_multiview_descriptors`, and `apply_multiview_initialization`.
- [x] Add projected-view MVInit builder and train-time MVInit hook for `train_lafgs.py`.
- [x] Run `pytest tests/test_lafgs_reconstruction.py -q`.

### Task 2: Soft 3D-to-2D Correspondence

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests showing a descriptor selects the expected feature-map location and reports entropy confidence.
- [x] Implement `soft_3d_to_2d_correspondences` with optional GT-projection local windows.
- [x] Run `pytest tests/test_lafgs_reconstruction.py -q`.

### Task 3: DiffPnP-Loc Loss

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests proving pose supervision backpropagates to Gaussian descriptors and can optionally backpropagate to 3D points.
- [x] Implement `DifferentiablePnPConfig`, `DifferentiablePnPOutput`, and `differentiable_pnp_pose_loss`.
- [x] Convert DiffPnP outputs into per-landmark topology stats for pose-aware densification.
- [x] Run `pytest tests/test_lafgs_reconstruction.py -q`.

### Task 4: Curriculum Policy and CLI Surface

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Create: `train_lafgs.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests for phase selection and trainable parameter groups.
- [x] Implement `LaFGSCurriculumConfig`, `lafgs_phase_for_iteration`, `lafgs_phase_from_starts`, and `lafgs_trainable_param_names`.
- [x] Add a lightweight `train_lafgs.py` wrapper that runs `train_locaware.py` with LaFGS defaults unless explicitly overridden.
- [x] Wire runtime curriculum into `train_locaware.py` so `train_lafgs.py` advances LocRec -> DiffPnP -> Geometry -> Topology by iteration.
- [x] Run targeted tests and parser smoke checks.

### Task 5: Geometry Residual and Pose-Aware Topology

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Modify: `scene/gaussian_model.py`
- Modify: `train_locaware.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests for scale-bounded geometry residuals and pose-aware split scoring.
- [x] Implement `bounded_geometry_residual_loss` with scale-relative residual limits.
- [x] Add geometry residual loss to geometry/topology phases.
- [x] Update `GaussianModel.compute_split_necessity()` to use ambiguity, PnP residual, repeatability, footprint, confidence, and pose information.
- [x] Run targeted tests and compile checks.

### Task 6: LaFGS Synthetic View Policy

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`
- Modify: `train_locaware.py`
- Test: `tests/test_lafgs_reconstruction.py`

- [x] Write failing tests proving synthetic view sampling is allowed for the direct LocRec teacher.
- [x] Replace dense-only synthetic view gating with `lafgs_should_sample_synthetic_view`.
- [x] Downweight direct synthetic LocRec episodes with `synthetic_view_desc_weight`.
- [x] Default `train_lafgs.py` synthetic feature targets to RGB render -> backbone features, with `loc_feature` retained as an explicit compatibility mode.
- [x] Run targeted tests and compile checks.
