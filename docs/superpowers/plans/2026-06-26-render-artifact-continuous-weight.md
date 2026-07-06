# Render Artifact Continuous Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous render-artifact teacher weighting mode that preserves the existing severity mode and keeps future per-region/pose-adjust work isolated.

**Architecture:** `localization_training/render_artifacts.py` remains the only module that interprets artifact audit metrics. `train_locaware.py` and `scripts/run_locaware_v03_topology_full.sh` only pass mode/parameter settings and consume an `ArtifactWeightLookup`.

**Tech Stack:** Python stdlib, unittest, existing STDLoc/LA-STDLoc training scripts.

---

### Task 1: Continuous Image-Level Weighting

**Files:**
- Modify: `localization_training/render_artifacts.py`
- Test: `tests/test_render_artifact_weights.py`

- [x] Write failing tests for a new `continuous_quality_weight()` API:
  - Clean rows return `default_weight`.
  - Worse PSNR/SSIM/residual/bias produce lower weights than borderline rows.
  - Low alpha coverage is treated as a quality penalty.
  - Output is clipped to `[min_weight, default_weight]`.

- [x] Implement `continuous_quality_weight()` using normalized metric penalties:
  - bad if `psnr_mean_matched <= mild_psnr`
  - bad if `ssim <= mild_ssim`
  - bad if `residual_frac_025 >= mild_residual`
  - bad if `alpha_cov_05 <= mild_alpha_cov`
  - bad if `mean_abs_bias >= mild_abs_bias`

- [x] Extend `load_artifact_weight_lookup(..., mode="severity", continuous_min_weight=0.70, continuous_power=1.0)` and keep current `severity` behavior unchanged.

### Task 2: CLI and Script Wiring

**Files:**
- Modify: `train_locaware.py`
- Modify: `scripts/run_locaware_v03_topology_full.sh`
- Test: `tests/test_train_locaware_masks.py`
- Test: `tests/test_full_script_args.py`

- [x] Write failing tests for parser/script defaults:
  - `--render_artifact_weight_mode` defaults to `severity`.
  - continuous parameters are exposed.
  - script passes mode and continuous parameters when a weight path is set.

- [x] Add parser arguments and pass them to `load_artifact_weight_lookup`.

- [x] Add env vars to the topology script:
  - `RENDER_ARTIFACT_WEIGHT_MODE`
  - `RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN`
  - `RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER`

### Task 3: Verification and Ablation

**Files:**
- Modify: `LA_update3_closure.md`

- [x] Run unit tests:
  - `/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_render_artifact_weights tests.test_render_artifact_audit tests.test_train_locaware_masks tests.test_full_script_args`

- [x] Run 100-step continuous ablations:
  - ShopFacade seed 2026 with `RENDER_ARTIFACT_WEIGHT_MODE=continuous`.
  - OldHospital seed 2026 with `RENDER_ARTIFACT_WEIGHT_MODE=continuous`.

- [x] Compare against P10/P13 matched severity baselines and update `LA_update3_closure.md`.
