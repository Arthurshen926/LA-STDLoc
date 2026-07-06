# LA_update20: Source Ablation and Mainline Refactor Evidence

Date: 2026-06-30

## Purpose

This update closes the immediate question raised by the source/stage cross-tab diagnostics: whether the current synthetic pseudo-query pool is the dominant reason for weak student gains, especially on OldHospital.

The ablation keeps the current stage-aware direct objective and changes only the pseudo-query source pool:

- all-source: `train_rgb,synthetic_rgb`
- train-only: `train_rgb`

Teacher cache is used as diagnostics and soft reliability signal, not as a hard gate.

## Runs

Common command settings:

```bash
OUT_ROOT=/mnt/pool/sqy/stdloc_la_refactor_full_20260630
LA_ADAPT_STEPS=100
TRAIN_SEED=121
PSEUDO_QUERY_STAGE_OBJECTIVE_MODE=direct
PSEUDO_QUERY_SOURCES=train_rgb
RUN_PSEUDO_QUERY_MANIFEST=0
RUN_TEACHER_CACHE=0
RUN_PSEUDO_QUERY_GATE=0
RUN_PSEUDO_QUERY_SELECT=0
RUN_LA_FRONTEND_REFRESH=0
RUN_EVAL=1
```

Outputs:

- ShopFacade: `/mnt/pool/sqy/stdloc_la_refactor_full_20260630/ShopFacade/student_100step_seed121`
- OldHospital: `/mnt/pool/sqy/stdloc_la_refactor_full_20260630/OldHospital/student_100step_seed121`

## Official Sparse-Only Results

| Scene | Source pool | Steps | Seed | Median TE | Median AE | R5 | R2 | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | train_rgb only | 100 | 121 | 3.1276 cm | 0.1650 deg | 73.79% | 25.24% | 427.00 |
| OldHospital | train_rgb only | 100 | 121 | 18.4301 cm | 0.3690 deg | 4.40% | 0.55% | 273.36 |

Reference all-source 100-step seed120 from LA_update19:

| Scene | Source pool | Steps | Seed | Median TE | Median AE | R5 | R2 | Avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | train_rgb + synthetic_rgb | 100 | 120 | 3.1517 cm | 0.1632 deg | 73.79% | 27.18% | 423.66 |
| OldHospital | train_rgb + synthetic_rgb | 100 | 120 | 19.0305 cm | 0.3440 deg | 4.40% | 0.55% | 275.30 |

The seeds are not matched, so this should be read as a directional ablation, not a final paired statistical result.

## Source/Stage Cross-Tab

ShopFacade train-only seed121:

| Source | teacher_ok | dense_improves_sparse | mixed_or_uncertain | dense_rescues_sparse | sparse_failure | dense_regression | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_rgb | 82 | 0 | 17 | 0 | 1 | 0 | 100 |

OldHospital train-only seed121:

| Source | teacher_ok | dense_improves_sparse | mixed_or_uncertain | dense_rescues_sparse | sparse_failure | dense_regression | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_rgb | 10 | 17 | 37 | 26 | 9 | 1 | 100 |

OldHospital all-source seed120 had synthetic-heavy failure:

| Source | teacher_ok | dense_improves_sparse | mixed_or_uncertain | dense_rescues_sparse | sparse_failure | dense_regression | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_rgb | 5 | 13 | 30 | 14 | 4 | 0 | 66 |
| synthetic_rgb | 0 | 0 | 2 | 4 | 25 | 3 | 34 |

## Conclusions

1. Current OldHospital synthetic is a real high-impact confound. In the all-source run, 28/34 synthetic samples were sparse failures or dense regressions.
2. Removing synthetic helps OldHospital median TE directionally, but does not solve the scene. The remaining train_rgb distribution is still weak: only 10/100 teacher_ok and 36/100 dense-rescue or sparse-failure samples.
3. ShopFacade synthetic is not obviously poisonous. Train-only and all-source are close, with small metric trade-offs.
4. A global "always use all synthetic" default is not justified. The training mainline should make synthetic opt-in or scene-policy controlled, and should keep teacher cache as diagnostics/soft reliability rather than a hard selection gate.
5. The next refactor should make the default path explicit:
   - default source pool: `train_rgb`
   - optional source pool: `train_rgb,synthetic_rgb`
   - no teacher hard gate by default
   - no sort-based synthetic selection by default
   - source/stage cross-tabs always logged when a teacher cache exists

