# PnP-Aware Landmark Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small `localization_aware_pnp` detector sampling mode that preserves high localization utility while enforcing 3D landmark diversity for sparse PnP robustness.

**Architecture:** Keep the change isolated to detector landmark sampling. Add a voxel-balanced selector in `localization_training/landmark_distill.py`, expose it through `train_detector.py`, and pass the knobs through the LA pipeline script. Do not change `stdloc.py` inference behavior.

**Tech Stack:** Python, PyTorch, unittest, existing STDLoc training scripts.

---

### Task 1: Add PnP-Balanced Landmark Sampling

**Files:**
- Modify: `localization_training/landmark_distill.py`
- Test: `tests/test_landmark_distill.py`

- [ ] **Step 1: Write the failing test**

Add a test proving `pnp_balance=True` avoids selecting all top utility landmarks from the same 3D voxel:

```python
def test_localization_aware_sample_pnp_balance_limits_voxel_collapse(self):
    from localization_training.landmark_distill import localization_aware_sample

    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    base_score = torch.zeros(5)
    utility = torch.tensor([100.0, 99.0, 98.0, 10.0, 9.0])

    sampled, meta = localization_aware_sample(
        xyz,
        base_score,
        utility,
        num=3,
        k=2,
        min_observations=torch.ones(5, dtype=torch.bool),
        spatial=False,
        pnp_balance=True,
        pnp_voxel_size=1.0,
        pnp_max_per_voxel=1,
    )

    self.assertEqual(set(sampled.tolist()), {0, 3, 4})
    self.assertTrue(bool(meta["pnp_balance"].item()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_landmark_distill.LandmarkDistillTest.test_localization_aware_sample_pnp_balance_limits_voxel_collapse
```

Expected: fail with `TypeError` because `pnp_balance` is not accepted yet.

- [ ] **Step 3: Implement minimal code**

Add `voxel_balanced_score()` and optional `pnp_balance`, `pnp_voxel_size`, `pnp_max_per_voxel` arguments to `localization_aware_sample()`. Use top utility order, keep at most `pnp_max_per_voxel` landmarks per voxel, then refill by score if the voxel quota is too strict.

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command. Expected: pass.

### Task 2: Expose The Sampling Mode

**Files:**
- Modify: `train_detector.py`
- Modify: `scripts/run_la_pseudo_query_pipeline.sh`
- Test: `tests/test_detector_soft_targets.py`
- Test: `tests/test_full_script_args.py`

- [ ] **Step 1: Write failing parser/script tests**

Add parser coverage:

```python
def test_detector_parser_accepts_pnp_aware_sampling_mode(self):
    from train_detector import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--sampling_mode", "localization_aware_pnp"])

    self.assertEqual(args.sampling_mode, "localization_aware_pnp")
```

Add script coverage that `run_la_pseudo_query_pipeline.sh` forwards:

```python
def test_pseudo_query_pipeline_forwards_pnp_sampling_controls(self):
    text = self._read_script("run_la_pseudo_query_pipeline.sh")

    self.assertIn("LA_DETECTOR_PNP_VOXEL_SIZE=${LA_DETECTOR_PNP_VOXEL_SIZE:-0.25}", text)
    self.assertIn("LA_DETECTOR_PNP_MAX_PER_VOXEL=${LA_DETECTOR_PNP_MAX_PER_VOXEL:-8}", text)
    self.assertIn('--pnp_voxel_size "$LA_DETECTOR_PNP_VOXEL_SIZE"', text)
    self.assertIn('--pnp_max_per_voxel "$LA_DETECTOR_PNP_MAX_PER_VOXEL"', text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_detector_parser_accepts_pnp_aware_sampling_mode \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_forwards_pnp_sampling_controls
```

Expected: parser fails because the mode is missing, script test fails because args are not forwarded.

- [ ] **Step 3: Implement minimal code**

In `train_detector.py`, add parser args:

```python
parser.add_argument("--pnp_voxel_size", type=float, default=0.25)
parser.add_argument("--pnp_max_per_voxel", type=int, default=8)
```

Allow `localization_aware_pnp` in sampling choices and call `localization_aware_sample(..., pnp_balance=True, pnp_voxel_size=pnp_voxel_size, pnp_max_per_voxel=pnp_max_per_voxel)` for that mode.

In `scripts/run_la_pseudo_query_pipeline.sh`, add defaults and forward both args to `train_detector.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command. Expected: pass.

### Task 3: Validate And Smoke

**Files:**
- Modify: `LA_update32_pnp_aware_landmark_utility.md`

- [ ] **Step 1: Run targeted tests**

Run:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_landmark_distill \
  tests.test_detector_soft_targets.DetectorSoftTargetsTest.test_detector_parser_accepts_pnp_aware_sampling_mode \
  tests.test_full_script_args.FullRunScriptArgsTest.test_pseudo_query_pipeline_forwards_pnp_sampling_controls
```

Expected: all selected tests pass.

- [ ] **Step 2: Run script syntax checks**

Run:

```bash
bash -n scripts/run_la_pseudo_query_pipeline.sh
bash -n scripts/run_la_oldhospital_objective_ablation.sh
```

Expected: both commands exit 0.

- [ ] **Step 3: Run OldHospital smoke**

Run a small clean-boundary smoke:

```bash
RUN_PSEUDO_QUERY_MANIFEST=0 \
RUN_TEACHER_CACHE=0 \
GPU=0 \
OUT_ROOT=/mnt/pool/sqy/stdloc_la_oldhospital_pnpaware100_seed414_20260701 \
TRAIN_SEED=414 \
LA_ADAPT_STEPS=100 \
LANDMARK_NUM=8192 \
LA_DETECTOR_SAMPLING_MODE=localization_aware_pnp \
LOG_ROOT=/mnt/pool/sqy/stdloc_la_oldhospital_pnpaware100_seed414_20260701/logs \
bash scripts/run_la_oldhospital_objective_ablation.sh
```

Expected: training, detector refresh, and official sparse-only evaluation finish and write `summary.json`.

- [ ] **Step 4: Document results**

Create `LA_update32_pnp_aware_landmark_utility.md` with implementation details, test results, smoke metrics, and whether this mode improves inliers/5cm recall relative to the latest clean/objective references.
