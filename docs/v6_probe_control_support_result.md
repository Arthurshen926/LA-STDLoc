# V6 virtual-probe control-support result

## Outcome

This round fixes the observer design and demonstrates a real control-support
effect, but it does **not** improve the real-test success rate.  The result must
not be reported as a new localization-accuracy winner.

## Implemented corrections

- Expanded the observer from 8 poses / 16 views to a balanced 48-pose plan
  with four deterministic sensor variants per pose.
- Balanced pose excitation across interpolation, rotation, translation,
  deficit-directed, boundary, and reverse-view mechanisms.
- Rejected pose groups with insufficient Gaussian alpha/depth support; 44/48
  poses (176 views) passed the 25% valid-keypoint contract.
- Partitioned failures into nominal, descriptor-controllable,
  geometry/solver-limited, and frontend-observation-limited routes.
- Restricted descriptor actions to oracle-recoverable training failures.
- Removed flat-bank concatenation from online sparse-prototype matching.
- Tested two deployment actions: direct probe prototypes and probe-directed
  selection from non-seq2 mapping observations.
- Added a support-expansion action that covers all training-side controllable
  positive Anchors, not only the minimal PoseLib basin-crossing subset.

## Main evidence

Baseline observer (176 views): R5 70.4545%, catastrophic 52.  The 8192-mode
support candidate reaches R5 84.0909%, catastrophic 24.

On the independent 44-view probe holdout, R5 remains 72.7273%, but
catastrophic failures fall 12 to 8, P90 falls 11711.07 cm to 2780.61 cm, CVaR95
falls 27908.72 cm to 13128.17 cm, and mean falls 3210.13 cm to 1266.10 cm.
There are no success-to-failure regressions.

The mechanism behind the earlier lack of transfer is measured directly: the
minimal 645-prototype action covers 479 Anchors, while held-out controllable
failures require 2898 positive Anchors.  Only 18 overlap (0.62%); the median
spatial distance from a held-out positive Anchor to an acted-on Anchor is
2.16 m.  Support expansion improves tail behavior but still does not cross the
5 cm success boundary.

Full mapping evaluation of the 8192-mode candidate keeps R5 at 97.0410% and
catastrophic count at 29.  Median changes 0.29060 to 0.29187 cm and P90 changes
0.75551 to 0.75688 cm, so it is not a mapping-accuracy winner.

Real StMarysChurch test, 530 queries, seed 2026, deployment protocol:

| Metric | Baseline | 8192-mode support candidate | Delta |
|---|---:|---:|---:|
| median TE | 4.0214 cm | 4.0014 cm | -0.0200 cm |
| mean TE | 109.9243 cm | 109.9554 cm | +0.0311 cm |
| P90 TE | 12.6196 cm | 12.9882 cm | +0.3686 cm |
| R2 | 18.3019% | 18.1132% | -0.1887 pp |
| R5 | 61.8868% | 61.8868% | 0 pp |
| catastrophic >=100 cm | 11 | 11 | 0 |

There are 0 gained and 0 lost R5 queries.  Consequently the median reduction
is real but not sufficient evidence of a meaningful localization improvement.

## Decision

Do not promote either sparse-prototype candidate.  The fixed-map observer and
failure routing are retained.  The next controller must share an action across
Anchors and viewpoints (for example a tightly constrained shared/low-rank
metric or trajectory-wide augmentation-consistent map reconstruction), and it
must be selected without further per-query test tuning.

## Artifacts

- Plan: `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v2/probe_plan_balanced_pose48_sensor4.pt`
- Accepted cache: `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v2/probe_cache_balanced_pose48_sensor4_accepted.pt`
- Baseline probe feedback: `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v2/baseline_feedback_balanced_pose48_sensor4_accepted.json`
- Support candidate: `/mnt/pool/sqy/lafgs_v6_feedback_control_20260825/StMarysChurch/probe_sparse_prototype_v2/candidate_coverage_budget8192`
- Final test: `candidate_coverage_budget8192/final_test_seed2026/summary.json`
