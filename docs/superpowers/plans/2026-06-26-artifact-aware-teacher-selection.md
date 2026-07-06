# Artifact-Aware Teacher Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a P20 teacher gate that combines image-level render quality and local artifact maps, then uses the combined confidence to scale direct landmark teacher loss without changing default behavior.

**Architecture:** Keep artifact detection in `localization_training/render_artifacts.py`, keep direct teacher weighting in `localization_training/direct_landmark_teacher.py`, and expose the new gate through `train_locaware.py` plus `scripts/run_locaware_v03_topology_full.sh`. Defaults preserve P17/P18 behavior; P20 experiments opt into `combined_mean` scaling.

**Tech Stack:** Python, PyTorch, unittest, existing LA-STDLoc shell runner and Cambridge STDLoc preprocessed data.

---

### Task 1: Combine Local And Image Artifact Confidence

**Files:**
- Modify: `localization_training/render_artifacts.py`
- Test: `tests/test_render_artifact_weights.py`

- [ ] **Step 1: Write the failing test**

Add tests that call `combine_artifact_confidence(local_weights, image_weight, mode)` with `product`, `min`, and `none`. The product case must return `local * image`, the min case must cap local weights by the scalar, and the none case must return local weights unchanged.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_render_artifact_weights.ArtifactWeightLookupTest.test_combine_artifact_confidence_modes
```

Expected: fail because `combine_artifact_confidence` is not defined.

- [ ] **Step 3: Implement minimal helper**

Add `combine_artifact_confidence(local_weights, image_weight=1.0, mode="product")` to `localization_training/render_artifacts.py`. It must accept tensor-like local weights, clamp both inputs to `[0, 1]`, support `product`, `min`, and `none`, and raise `ValueError` for unknown modes.

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command. Expected: `OK`.

### Task 2: Scale Direct Teacher Loss By Artifact Confidence

**Files:**
- Modify: `localization_training/direct_landmark_teacher.py`
- Test: `tests/test_direct_landmark_teacher.py`

- [ ] **Step 1: Write failing tests**

Add one test that passes an all-low artifact map plus `artifact_image_weight=0.5`, `artifact_weight_combine_mode="product"`, and `artifact_loss_scale_mode="combined_mean"` to `direct_landmark_teacher`. It must assert that the reported `artifact_teacher_loss_scale` equals the combined mean and that the scaled loss is lower than the unscaled loss. Add a second test for `artifact_loss_scale_mode="none"` showing backward-compatible loss behavior.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_artifact_combined_mean_scales_direct_teacher_loss tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_artifact_loss_scale_none_preserves_legacy_weighted_mean
```

Expected: fail because the new keyword arguments and diagnostics do not exist.

- [ ] **Step 3: Implement direct teacher arguments**

Extend `direct_landmark_teacher` with:

```python
artifact_image_weight=1.0
artifact_weight_combine_mode="product"
artifact_loss_scale_mode="none"
```

Use `combine_artifact_confidence` to compute combined confidence from sampled local weights and image weight. Continue using combined weights for per-landmark weighted means. If `artifact_loss_scale_mode == "combined_mean"`, multiply descriptor, multiview, full-bank, anchor, and aggregate losses by the mean combined confidence. If `artifact_loss_scale_mode == "region_mean"`, scale by the sampled region mean only. If mode is `none`, do not scale losses. Add diagnostics for image weight, combined min/mean, weighted count, and loss scale. Preserve existing `artifact_region_weight_*` diagnostics.

- [ ] **Step 4: Run tests**

Run the same two unittest targets. Expected: `OK`.

### Task 3: Wire P20 Gate Into Training And Shell Runner

**Files:**
- Modify: `train_locaware.py`
- Modify: `scripts/run_locaware_v03_topology_full.sh`
- Test: `tests/test_full_script_args.py`

- [ ] **Step 1: Write failing parser/script tests**

Extend the full script args test to assert the runner forwards:

```bash
--render_artifact_direct_weight_combine_mode
--render_artifact_direct_loss_scale_mode
```

Use env variables:

```bash
RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE=product
RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE=combined_mean
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default
```

Expected: fail because the script does not forward the new args.

- [ ] **Step 3: Implement CLI wiring**

Add parser args in `train_locaware.py`:

```python
--render_artifact_direct_weight_combine_mode {product,min,none}
--render_artifact_direct_loss_scale_mode {none,region_mean,combined_mean}
```

In the direct teacher branch, read the image-level artifact weight before calling `direct_landmark_teacher`. Pass image weight, combine mode, and scale mode to direct teacher. If `combined_mean` consumes the image-level weight, skip the later outer image-level multiplication for direct teacher to avoid double scaling. Dense teacher keeps existing outer multiplication behavior.

Add matching shell env and forwarding in `scripts/run_locaware_v03_topology_full.sh`.

- [ ] **Step 4: Run tests**

Run the same full script args test. Expected: `OK`.

### Task 4: Verification And 100-Step Evidence

**Files:**
- Read logs under `/mnt/pool/sqy/stdloc_la_artifact_filter_logs`
- No production files modified

- [ ] **Step 1: Run targeted unit and static checks**

Run:

```bash
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_render_artifact_weights tests.test_direct_landmark_teacher tests.test_full_script_args
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile localization_training/render_artifacts.py localization_training/direct_landmark_teacher.py train_locaware.py
bash -n scripts/run_locaware_v03_topology_full.sh
git diff --check
```

Expected: all exit with code 0.

- [ ] **Step 2: Run OldHospital P20 100-step**

Run the topology script with:

```bash
CUDA_VISIBLE_DEVICES=0
SCENE=OldHospital
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_update3_p20_teacher_gate_100_seed2026_v1
TOPOLOGY_STEPS=100
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_MAX_MUTATION_EVENTS=0
RENDER_ARTIFACT_WEIGHT_PATH=/mnt/pool/sqy/stdloc_la_render_audit_v1/OldHospital_actual_query_seed2026_base32000.csv
RENDER_ARTIFACT_WEIGHT_MODE=continuous
RENDER_ARTIFACT_WEIGHT_SEVERITIES=severe
RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN=0.25
RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER=0.5
RENDER_ARTIFACT_WEIGHT_TARGETS=teacher
RENDER_ARTIFACT_REGION_WEIGHT_PATH=/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p18_region_OldHospital_seed2026_manifest.csv
RENDER_ARTIFACT_REGION_WEIGHT_ROOT=/mnt/pool/sqy/stdloc_la_update3_p18_region_maps
RENDER_ARTIFACT_REGION_WEIGHT_SEVERITIES=severe
RENDER_ARTIFACT_REGION_WEIGHT_TARGETS=direct
RENDER_ARTIFACT_DIRECT_WEIGHT_COMBINE_MODE=product
RENDER_ARTIFACT_DIRECT_LOSS_SCALE_MODE=combined_mean
```

Expected: training completes and prints a sparse `Result Summary`.

- [ ] **Step 3: Run ShopFacade P20 100-step**

Run the same configuration on GPU1 with ShopFacade manifests:

```bash
CUDA_VISIBLE_DEVICES=1
SCENE=ShopFacade
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_update3_p20_teacher_gate_100_seed2026_v1
RENDER_ARTIFACT_WEIGHT_PATH=/mnt/pool/sqy/stdloc_la_render_audit_v1/ShopFacade_actual_query_seed2026_base32000.csv
RENDER_ARTIFACT_REGION_WEIGHT_PATH=/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p18_region_ShopFacade_seed2026_manifest.csv
```

Expected: training completes and prints a sparse `Result Summary`.

- [ ] **Step 4: Compare against P17/P18/no-filter**

Parse `median_ae`, `median_te`, `recall_5cm_5d`, `recall_2cm_2d`, and `avg_inliers` from the new logs and from the previous no-filter/P17/P18 logs. Positive support requires OldHospital to improve over no-filter in TE or R5 without obvious collapse, and ShopFacade to avoid a clear regression.
