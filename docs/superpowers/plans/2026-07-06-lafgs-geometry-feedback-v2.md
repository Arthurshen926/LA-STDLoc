# LaFGS Geometry Feedback V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve LaFGS differentiable PnP/geometry feedback so it gives a direct, testable descriptor correspondence signal in addition to guarded xyz updates.

**Architecture:** Keep the existing 3D-to-soft-2D DiffPnP path. Add an optional local GT-projection correspondence loss that backpropagates to Gaussian descriptors while keeping xyz detached, and leave the existing guarded geometry reprojection loss responsible for xyz updates.

**Tech Stack:** Python, PyTorch, pytest, existing LaFGS Cambridge runner.

---

### Task 1: Add Failing Unit Test

**Files:**
- Modify: `tests/test_lafgs_reconstruction.py`

- [ ] Add a test named `test_differentiable_pnp_geometry_match_loss_updates_descriptors_without_moving_xyz`.
- [ ] Build four projected landmarks with local feature peaks shifted by one pixel from GT.
- [ ] Call `differentiable_pnp_pose_loss` with all existing pose and xyz geometry weights disabled and the new match loss enabled.
- [ ] Verify RED with `pytest tests/test_lafgs_reconstruction.py::test_differentiable_pnp_geometry_match_loss_updates_descriptors_without_moving_xyz -q`; expected failure is an unknown `DifferentiablePnPConfig` argument.

### Task 2: Implement Descriptor Match Feedback

**Files:**
- Modify: `localization_training/lafgs_reconstruction.py`

- [ ] Add `geometry_match_reprojection_weight` to `DifferentiablePnPConfig`.
- [ ] In `differentiable_pnp_pose_loss`, compute a local soft correspondence around `projected_uv` or GT projection using Gaussian descriptors and detached query features.
- [ ] Build a filtered mask using the same geometry confidence, margin, peak probability, entropy, and max reprojection thresholds.
- [ ] Compute weighted reprojection loss from local soft UV to detached GT projection, with points detached so xyz does not receive this term.
- [ ] Add diagnostics for match correspondences and loss weight.

### Task 3: Expose Runner/CLI Controls

**Files:**
- Modify: `train_locaware.py`
- Modify: `train_lafgs.py`
- Modify: `scripts/run_lafgs_cambridge_guarded_pnp.py`
- Modify: `tests/test_lafgs_cambridge_experiments.py`

- [ ] Add `--lafgs_diff_pnp_geometry_match_reproj_weight`.
- [ ] Forward the value through training config and experiment runner.
- [ ] Set guarded Cambridge experiments to opt in explicitly; keep default 0.0 for backward compatibility.
- [ ] Add or update runner command tests to assert the flag is emitted when configured.

### Task 4: Verify and Run Experiments

**Files:**
- Read: `/mnt/pool/sqy/stdloc_lafgs_cambridge_geom_active_full5_20260706/summary_geomactive_full5.json`
- Create: new experiment root under `/mnt/pool/sqy/`

- [ ] Run focused pytest and py_compile checks.
- [ ] Run a single-scene probe, prioritizing ShopFacade or OldHospital.
- [ ] If probe is not worse than utility-only on median TE and recall, run Cambridge five scenes.
- [ ] Summarize against baseline and utility-only with `scripts/summarize_lafgs_cambridge.py`.

