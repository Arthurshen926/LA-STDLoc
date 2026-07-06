# LA Mainline Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current clean LA-STDLoc control path and separate it from experimental synthetic/artifact branches before starting the next student-objective ablation.

**Architecture:** Keep `scripts/run_la_clean_real_train_mainline.sh` as the single-scene clean control wrapper, keep `scripts/run_la_pseudo_query_pipeline.sh` as the full experimental backend, and add a small matrix wrapper for the three validated LA_update29 control runs. Document the boundaries so future experiments compare against a stable control instead of changing the default path.

**Tech Stack:** Bash wrappers, Markdown documentation, existing `unittest` script checks.

---

### Task 1: Document Mainline Boundaries

**Files:**
- Create: `docs/la_stdloc_mainline.md`
- Reference: `LA_update29_clean_mainline_closure.md`

- [ ] **Step 1: Write the mainline document**

Create `docs/la_stdloc_mainline.md` with these sections:

```markdown
# LA-STDLoc Mainline Boundaries

## Status

The previous broad LA-STDLoc goal is paused. The validated control path is the
clean all-train RGB mainline from `LA_update29_clean_mainline_closure.md`.

## Default Control

Use `scripts/run_la_clean_real_train_mainline.sh` for one scene and
`scripts/run_la_clean_control_matrix.sh` for the three validated controls.

The control path uses only real Cambridge train RGB pseudo queries and disables
synthetic RGB, teacher gates, pseudo-query selectors, no-reference valid/support
masks, artifact detector/repair weighting, reliability weighting, and direct
depth checks.

## Experimental Backend

Use `scripts/run_la_pseudo_query_pipeline.sh` only for ablations. It contains
MAtCha/WildGaussians synthetic rendering, spatial pose sampling, teacher cache
valid masks, teacher gates, pseudo-query selection, artifact weights, and
reliability weighting. None of these branches should be treated as default.

## Validated Controls

| Scene | Capacity | Steps | Seed | Output root |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 8192 | 2000 | 301 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` |
| OldHospital | 8192 | 2000 | 302 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_8192_2000_20260630` |
| OldHospital | 16384 | 2000 | 303 | `/mnt/pool/sqy/stdloc_la_clean_mainline_logged_16384_2000_20260630` |

## Next Smaller Objective

The next goal should be an OldHospital student-objective ablation against this
control, focused on high-precision recall and stability. Synthetic RGB and
artifact modules should stay disabled until that objective has a positive or
negative result.
```

- [ ] **Step 2: Check the document**

Run:

```bash
rg -n "Default Control|Experimental Backend|Next Smaller Objective" docs/la_stdloc_mainline.md
```

Expected: three matching section headings.

### Task 2: Add A Clean Control Matrix Wrapper

**Files:**
- Create: `scripts/run_la_clean_control_matrix.sh`
- Test: `tests/test_full_script_args.py`

- [ ] **Step 1: Write the wrapper**

Create `scripts/run_la_clean_control_matrix.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

LOG_ROOT=${LOG_ROOT:-/mnt/pool/sqy/stdloc_la_clean_mainline_logs_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT_8192=${OUT_ROOT_8192:-/mnt/pool/sqy/stdloc_la_clean_mainline_control_8192_2000_$(date +%Y%m%d)}
OUT_ROOT_16384=${OUT_ROOT_16384:-/mnt/pool/sqy/stdloc_la_clean_mainline_control_16384_2000_$(date +%Y%m%d)}
LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-2000}

RUN_SHOP_8192=${RUN_SHOP_8192:-1}
RUN_OLD_8192=${RUN_OLD_8192:-1}
RUN_OLD_16384=${RUN_OLD_16384:-1}

GPU_SHOP_8192=${GPU_SHOP_8192:-0}
GPU_OLD_8192=${GPU_OLD_8192:-1}
GPU_OLD_16384=${GPU_OLD_16384:-2}

SEED_SHOP_8192=${SEED_SHOP_8192:-301}
SEED_OLD_8192=${SEED_OLD_8192:-302}
SEED_OLD_16384=${SEED_OLD_16384:-303}

mkdir -p "$LOG_ROOT"

run_one() {
  local scene=$1
  local capacity=$2
  local seed=$3
  local gpu=$4
  local out_root=$5
  local log_name=$6
  shift 6
  local extra_env=("$@")

  (
    set -o pipefail
    export SCENES="$scene"
    export LA_ADAPT_STEPS="$LA_ADAPT_STEPS"
    export TRAIN_SEED="$seed"
    export GPU="$gpu"
    export OUT_ROOT="$out_root"
    export LA_BOOTSTRAP_LANDMARK_NUM="$capacity"
    export LA_DETECTOR_LANDMARK_NUM="$capacity"
    for item in "${extra_env[@]}"; do
      export "$item"
    done
    bash "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh" 2>&1 | tee "$LOG_ROOT/$log_name"
  ) &
}

if [[ "$RUN_SHOP_8192" == "1" ]]; then
  run_one ShopFacade 8192 "$SEED_SHOP_8192" "$GPU_SHOP_8192" "$OUT_ROOT_8192" shop8192.log
fi

if [[ "$RUN_OLD_8192" == "1" ]]; then
  run_one OldHospital 8192 "$SEED_OLD_8192" "$GPU_OLD_8192" "$OUT_ROOT_8192" old8192.log
fi

if [[ "$RUN_OLD_16384" == "1" ]]; then
  run_one OldHospital 16384 "$SEED_OLD_16384" "$GPU_OLD_16384" "$OUT_ROOT_16384" old16384.log
fi

wait
```

- [ ] **Step 2: Add script test coverage**

In `tests/test_full_script_args.py`, add a constant:

```python
CLEAN_CONTROL_MATRIX_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_la_clean_control_matrix.sh"
```

Add a test:

```python
def test_clean_control_matrix_runs_validated_controls_with_logs(self):
    self.assertTrue(CLEAN_CONTROL_MATRIX_SCRIPT.exists(), "clean control matrix script is missing")
    text = CLEAN_CONTROL_MATRIX_SCRIPT.read_text()
    self.assertIn("LA_ADAPT_STEPS=${LA_ADAPT_STEPS:-2000}", text)
    self.assertIn("RUN_SHOP_8192=${RUN_SHOP_8192:-1}", text)
    self.assertIn("RUN_OLD_8192=${RUN_OLD_8192:-1}", text)
    self.assertIn("RUN_OLD_16384=${RUN_OLD_16384:-1}", text)
    self.assertIn("SEED_SHOP_8192=${SEED_SHOP_8192:-301}", text)
    self.assertIn("SEED_OLD_8192=${SEED_OLD_8192:-302}", text)
    self.assertIn("SEED_OLD_16384=${SEED_OLD_16384:-303}", text)
    self.assertIn('bash "$SCRIPT_DIR/run_la_clean_real_train_mainline.sh" 2>&1 | tee "$LOG_ROOT/$log_name"', text)
```

- [ ] **Step 3: Verify shell syntax**

Run:

```bash
bash -n scripts/run_la_clean_control_matrix.sh scripts/run_la_clean_real_train_mainline.sh scripts/run_la_pseudo_query_pipeline.sh
```

Expected: exit 0.

### Task 3: Verify The Reset

**Files:**
- Verify: `docs/la_stdloc_mainline.md`
- Verify: `scripts/run_la_clean_control_matrix.sh`
- Verify: `tests/test_full_script_args.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_full_script_args.FullRunScriptArgsTest.test_clean_real_train_mainline_hard_disables_experimental_branches tests.test_full_script_args.FullRunScriptArgsTest.test_clean_control_matrix_runs_validated_controls_with_logs
```

Expected: both tests pass.

- [ ] **Step 2: Run diff checks**

Run:

```bash
git diff --check -- docs/la_stdloc_mainline.md scripts/run_la_clean_control_matrix.sh tests/test_full_script_args.py docs/superpowers/plans/2026-06-30-la-mainline-reset.md
```

Expected: no output.

- [ ] **Step 3: Commit only if requested**

The workspace contains many unrelated modified and untracked LA files. Do not
commit this reset automatically unless the user explicitly asks for a commit.
