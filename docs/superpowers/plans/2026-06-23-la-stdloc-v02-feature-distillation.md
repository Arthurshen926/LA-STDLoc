# LA-STDLoc v0.2 Feature Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `LA_update0.md` first-batch fixes and produce controlled evidence for whether dense localization feedback can improve fixed sparse Gaussian landmarks.

**Architecture:** Keep geometry, landmark indices, detector, topology, loc opacity, prototype/rank, utility sampling, soft detector, and closed-loop disabled for the first validation. Add a direct landmark-to-query distillation path that samples baseline selected Gaussian landmarks, checks target-view visibility/depth consistency, and optimizes only `_loc_feature` against query observations used by the sparse inference path.

**Tech Stack:** Python 3.8, PyTorch, gsplat renderer, existing `unittest` tests, Cambridge ShopFacade data under `/mnt/pool/sqy/Cambridge_stdloc`.

---

### Task 1: P0 Flag And Feature-Only Semantics

**Files:**
- Modify: `train_locaware.py`
- Test: `tests/test_train_locaware_masks.py` or a new small CLI/config test

- [ ] **Step 1: Write failing tests**
  Add tests proving `--use-loc-opacity/--no-use-loc-opacity` can be parsed, default is false, and feature phase trainables exclude `loc_opacity`.

- [ ] **Step 2: Run tests and confirm failure**
  Run `python -m unittest tests.test_train_locaware_masks`.

- [ ] **Step 3: Implement minimal fix**
  Use `argparse.BooleanOptionalAction`, default `False`, set feature phase trainables to only `loc_feature`, and make loc opacity LR zero unless explicitly enabled outside feature-only.

- [ ] **Step 4: Verify**
  Run the focused test and then full `python -m unittest discover -s tests`.

### Task 2: Restore Spatial Coverage In LA Sampling

**Files:**
- Modify: `localization_training/landmark_distill.py`
- Modify: `train_detector.py`
- Test: `tests/test_landmark_distill.py`

- [ ] **Step 1: Write failing test**
  Add a test where global top-k would select clustered high-utility landmarks but the expected sampler keeps one landmark per spatial neighborhood via `random_knn_score`.

- [ ] **Step 2: Run red test**
  Run `python -m unittest tests.test_landmark_distill`.

- [ ] **Step 3: Implement minimal fix**
  Reuse the baseline spatial sampler with `combined = base + utility` as local score. Keep metadata unchanged.

- [ ] **Step 4: Verify**
  Run focused and full tests.

### Task 3: Direct Landmark-To-Query Distillation

**Files:**
- Create: `localization_training/direct_landmark_teacher.py`
- Modify: `train_locaware.py`
- Test: `tests/test_direct_landmark_teacher.py`

- [ ] **Step 1: Write failing tests**
  Test projection of known Gaussian xyz into GT pose, target-depth consistency filtering, query-feature sampling, cosine direct loss, and per-Gaussian stats/prototype using sampled query descriptors.

- [ ] **Step 2: Run red tests**
  Run `python -m unittest tests.test_direct_landmark_teacher`.

- [ ] **Step 3: Implement teacher**
  Load baseline `sampled_idx.pkl`, select visible landmarks, project to GT view, optionally filter by GT rendered depth/alpha, compute direct cosine loss and descriptor diagnostics, and return exact `full_idx` stats for EMA.

- [ ] **Step 4: Wire training mode**
  Add `--loc_teacher {dense,direct}` and `--landmark_path`. In v0.2 feature phase use direct teacher, set dense teacher weights to zero, and update localization stats only for direct landmark indices.

- [ ] **Step 5: Verify**
  Run focused tests, compile checks, and a 2-iteration smoke train.

### Task 4: Controlled Evaluation Scripts

**Files:**
- Modify: `scripts/run_locaware_cambridge_full.sh`
- Create or modify: `scripts/train_locaware_cambridge.sh`
- Create: `scripts/run_locaware_v02_shopfacade.sh`
- Test: `tests/test_full_script_args.py`

- [ ] **Step 1: Write failing script-argument tests**
  Verify v0.2 script uses baseline model landmark/detector paths, direct teacher, no loc opacity, no topology, hard detector, and saves 500/1000/2000 checkpoints.

- [ ] **Step 2: Implement scripts**
  Add commands for baseline dense evaluation, E1 fixed baseline detector/index evaluation at 33k, E3 final fixed baseline detector/index evaluation, and v0.2 short runs.

- [ ] **Step 3: Verify**
  Run `bash -n` and script argument tests.

### Task 5: Initial ShopFacade Validation

**Files/Artifacts:**
- Input: `/mnt/pool/sqy/Cambridge_stdloc/ShopFacade`
- Output: `/mnt/pool/sqy/stdloc_la_v02_runs/ShopFacade_v02`
- Results: `results/phase-v02-*`

- [ ] **Step 1: Run smoke**
  Train 2 iterations direct teacher from baseline 30k and confirm checkpoint/save works.

- [ ] **Step 2: Run short checkpoints**
  Train 500, 1000, and 2000 iterations feature-only direct distillation.

- [ ] **Step 3: Evaluate controlled sparse pipeline**
  Evaluate each checkpoint using baseline detector and baseline sampled indices. Compare to Phase0 baseline sparse and dense summaries.

- [ ] **Step 4: Report**
  Report descriptor diagnostics, sparse median AE/TE, 5cm/5deg recall, inliers, and whether the fixed sparse pipeline improved without changing geometry/sampling/detector.
