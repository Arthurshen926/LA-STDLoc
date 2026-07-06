# LA Qualitative Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize LA-STDLoc qualitative diagnostics into an independent module that records each improved configuration batch.

**Architecture:** Add a standalone `la_diagnostics` Python package for loading pose results, render artifact audits, region weight manifests, writing sample-level reports, and generating lightweight contact sheets. Add a CLI wrapper under `scripts/` so each experiment batch can append one structured registry record without modifying training or evaluation code.

**Tech Stack:** Python standard library, NumPy, Pillow, optional PyTorch for `.pt` region weight maps, existing `unittest` test runner.

---

### Task 1: Report API and Tests

**Files:**
- Create: `tests/test_la_qualitative_diagnostics.py`
- Create: `la_diagnostics/__init__.py`
- Create: `la_diagnostics/qualitative.py`

- [ ] Write tests that create synthetic pose results, artifact audit CSVs, region manifests, and images in a temporary directory.
- [ ] Verify the report writer emits `summary.json`, `sample_flow.csv`, `index.md`, contact sheet PNGs, and appends `registry.jsonl`.
- [ ] Verify final-test samples and artifact-teacher samples keep distinct stage/split fields.
- [ ] Implement the minimal module to pass those tests.

### Task 2: CLI Wrapper

**Files:**
- Create: `scripts/run_la_qualitative_diagnostics.py`
- Modify: `tests/test_la_qualitative_diagnostics.py`

- [ ] Add a CLI smoke test for required arguments and output paths.
- [ ] Implement argument parsing that maps experiment batch inputs to the report API.
- [ ] Keep the CLI read-only for experiment inputs and write-only under the requested report directory.

### Task 3: Real Batch Recording

**Files:**
- Output only: `/root/STDLoc/results/la_qualitative_report/*`

- [ ] Run the CLI on the P20 OldHospital no-artifact vs teacher-gate batch.
- [ ] Confirm the report captures final worst/improved/regressed samples and severe teacher artifact samples.
- [ ] Append the batch summary to `registry.jsonl`.

### Task 4: Verification

**Files:**
- Test: `tests/test_la_qualitative_diagnostics.py`

- [ ] Run the focused unit tests.
- [ ] Run `py_compile` on the new package and CLI.
- [ ] Run `git diff --check`.
