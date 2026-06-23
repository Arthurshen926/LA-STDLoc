# LA_update1 落实闭环记录

日期：2026-06-23

## 结论

`LA_update1.md` 提出的主要工程缺口已经在当前分支形成可运行闭环：

1. v0.3 feature-only 路径已对齐 sparse inference：full-bank bi-MNN、hard negatives、view-diverse memory、baseline anchor 均已接入。
2. full-bank diagnostics、paired per-query analysis、solver-threshold sweep 已落地。
3. 3DGS `GaussianModel` 的 topology mutation 已完成生产路径验证：physical prune、selected split、source-id remap、optimizer/buffer 同步均跑通。
4. topology physical prune/split 已完成三场景三 seed full-length checkpoint 复验：ShopFacade 有明确 pilot 正向，KingsCollege 有边界性 TE 改善信号但 paired CI 跨 0，OldHospital recall/R2 偶有收益但 TE/AE 不稳。
5. Cambridge 三场景 100-step v0.3 已完成 3 seed smoke；full-length v0.3 30500/31000/32000 也已完成 3 场景 x 3 seed sparse-only 评测。
6. KingsCollege/OldHospital label-state topology 已完成从 100-step smoke 到 full-length 30500 checkpoint 的跨场景复验；source-id remap、selected split、physical prune 开关和 sparse-only eval 生产路径均可运行。

仍未宣称完成的是论文级外延：topology full-length 主表已经跑完但效果 mixed，尚不足以支撑“全场景稳定优于 v0.3 最优 checkpoint”的强结论；dense-stage improvement 到 sparse map 的跨场景主结果也尚未形成。

## 已落实的 LA_update1 修改项

| LA_update1 项 | 当前状态 | 证据 |
| --- | --- | --- |
| phase LR 恢复 | 已修复 | `train_locaware._set_phase_lrs()` 每次切 phase 先恢复 `la_base_lr`，再冻结/缩放 |
| full-bank diagnostics | 已实现 | `scripts/diagnose_sparse_descriptors.py`，`full_bank_descriptor_metrics()` |
| paired per-query 统计 | 已实现 | `scripts/analyze_paired_sparse_results.py` |
| solver-invariant sweep | 已实现并补跑 topology/no-split | `scripts/analyze_solver_threshold_sweep.py`，`/mnt/pool/sqy/stdloc_la_update1_solver_sweep/*.json` |
| full-bank bi-MNN + hard negatives | 已实现 | `localization_training/direct_landmark_teacher.py` |
| view-diverse memory | 已实现 | `LandmarkObservationMemory` 记录 view direction、distance、confidence，并做质量/视角替换 |
| anchor/drift 控制 | 已实现 | `--loc_anchor_weight`、baseline feature anchor |
| geometry-balanced selector | 已实现 | `localization_training/geometry_selector.py`，2x2 脚本已落地 |
| dense responsibility / KL | 已实现 smoke 路径 | `localization_training/dense_distill.py`，`scripts/diagnose_dense_responsibility.py` |
| utility 真实标签 | 已实现诊断/外部 state 输入 | `scripts/diagnose_sparse_inliers.py`，`--localization_state_path` |
| topology selected split | 已实现并验证 | `GaussianModel.densify_and_split_selected()`，`LocalizationTopologyController` |
| physical prune/split | 已跑通三场景三 seed full-length 复验；ShopFacade 正向，KingsCollege 边界改善，OldHospital mixed/偏负 | ShopFacade label-state one-event、4-event、over-split；full-length 30500 -> 30600 三场景三 seed topology；source-id/remap audit 全部通过 |
| 多场景/多 seed 入口 | 已实现，并加固 baseline artifact 检查 | `scripts/run_locaware_v03_multiscene.sh`，缺 detector 或缺 baseline 会显式 skip |
| full-length topology 复验入口 | 已实现 | `scripts/run_locaware_v03_topology_full.sh` 会复制 v0.3 checkpoint、生成 sparse label state、开启 physical prune/split、source-id remap，并用 remap 后 landmark 做 sparse-only eval |
| 非 ShopFacade baseline artifact 准备 | 已实现准备/训练入口与资产审计；KingsCollege、OldHospital 均已通过 fallback 补齐 | `scripts/prepare_cambridge_baseline_artifacts.sh` 会校验 `loc_*` PLY 字段，阻止 RGB-only 3DGS map 误入 detector 训练；显式 `TRAIN_MISSING_BASELINE=1` 时可从 Cambridge 数据跑完整 `train.py --train_detector` baseline |
| dense-to-sparse KL 主实验入口 | 已实现并完成 ShopFacade smoke；当前不是正向主结果 | `scripts/run_densekl_v03_cambridge.sh`，`/mnt/pool/sqy/stdloc_la_update1_densekl_smoke/*` |

## 关键实验结果

### topology 4-event 主结果

Run:

```text
/mnt/pool/sqy/stdloc_la_update1_topology_labelstate/ShopFacade_v03_labelstate_phys_split_4evt_20260623_151301
```

Topology events:

```text
30125: physical_prune=153, split=34 parents, children=68, points 342765 -> 342799
30150: physical_prune=0,   split=34 parents, children=68, points 342799 -> 342833
30175: physical_prune=0,   split=34 parents, children=68, points 342833 -> 342867
30200: physical_prune=0,   split=34 parents, children=68, points 342867 -> 342901
```

Remap:

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=342901
```

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-split adapt | 0.162927 | 3.185332 | 0.728155 | 0.242718 | 431.087 |
| topology 4-event | 0.162880 | 2.976683 | 0.737864 | 0.262136 | 431.922 |

Paired no-split vs topology 4-event at 12px:

```text
translation_delta_mean   = -0.2918 cm
translation_delta_median = -0.2056 cm
translation_CI95         = [-0.6046, -0.0378]
rotation_delta_mean      = -0.0154 deg
rotation_delta_median    = -0.0070 deg
rotation_CI95            = [-0.0306, -0.0033]
translation improved/degraded = 61.2% / 38.8%
R5 gain/loss = 2 / 1
R2 gain/loss = 7 / 5
```

### solver-threshold sweep

Sweep thresholds:

```text
2, 4, 6, 8, 10, 12, 16 px
```

No-split vs topology 4-event AUC:

| Metric | no-split | topology 4-event | Delta |
| --- | ---: | ---: | ---: |
| median AE | 0.163828 | 0.158131 | -0.005696 |
| median TE | 3.118784 | 2.983983 | -0.134801 |
| R5 | 0.728155 | 0.746879 | +0.018724 |
| R2 | 0.251040 | 0.250347 | -0.000693 |
| avg inliers | 391.059 | 392.467 | +1.408 |

V03 vs topology 4-event AUC:

| Metric | v03 | topology 4-event | Delta |
| --- | ---: | ---: | ---: |
| median AE | 0.162216 | 0.158131 | -0.004084 |
| median TE | 3.123372 | 2.983983 | -0.139389 |
| R5 | 0.733703 | 0.746879 | +0.013176 |
| R2 | 0.251040 | 0.250347 | -0.000693 |
| avg inliers | 377.329 | 392.467 | +15.138 |

Sweep outputs:

```text
/mnt/pool/sqy/stdloc_la_update1_solver_sweep/nosplit_vs_topology4evt_solver_sweep.json
/mnt/pool/sqy/stdloc_la_update1_solver_sweep/v03_vs_topology4evt_solver_sweep.json
```

### 多场景/多 seed 与 dense-KL 入口

新增多场景调度入口：

```text
scripts/run_locaware_v03_multiscene.sh
```

默认矩阵：

```text
SCENES="ShopFacade KingsCollege OldHospital"
SEEDS="2025 2026 2027"
```

当前资产审计结果：

```text
/mnt/pool/sqy/Cambridge_stdloc/{ShopFacade,KingsCollege,OldHospital} 均存在
/mnt/pool/sqy/stdloc_la_full_runs 目前已有 ShopFacade_baseline / ShopFacade_la / KingsCollege_baseline / OldHospital_baseline
/mnt/pool/sqy/ulfloc_repro_20260607/{KingsCollege,OldHospital} 有 30k RGB 3DGS PLY，但没有 loc_* feature 字段
```

新增 baseline artifact 准备入口：

```text
scripts/prepare_cambridge_baseline_artifacts.sh
```

该脚本现在默认要求 source/target PLY 含 `loc_*` feature 字段；如果 source 只是 RGB-only 3DGS，会在启动 detector 前立即 skip，避免进入相机加载后才在 `GaussianModel.load_ply()` 崩溃。验证命令：

```text
SCENES='ShopFacade KingsCollege' SOURCE_ROOT=/mnt/pool/sqy/ulfloc_repro_20260607 \
  TARGET_ROOT=/mnt/pool/sqy/stdloc_la_full_runs \
  PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python \
  scripts/prepare_cambridge_baseline_artifacts.sh
```

输出：

```text
[prepare baseline] Skip detector for ShopFacade: artifacts already exist.
[prepare baseline] Skip KingsCollege: source point cloud lacks loc_* feature fields: /mnt/pool/sqy/ulfloc_repro_20260607/KingsCollege/point_cloud/iteration_30000/point_cloud.ply.
[prepare baseline] Run train.py --train_detector for a full STDLoc baseline, or set SOURCE_ROOT to an existing feature-3DGS baseline.
```

当没有可复用 feature-3DGS baseline 时，使用显式训练 fallback：

```text
SCENES='KingsCollege OldHospital' TRAIN_MISSING_BASELINE=1 \
  SOURCE_ROOT=/mnt/pool/sqy/ulfloc_repro_20260607 \
  TARGET_ROOT=/mnt/pool/sqy/stdloc_la_full_runs \
  PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python \
  scripts/prepare_cambridge_baseline_artifacts.sh
```

该 fallback 调用的 baseline 命令与 `scripts/run_locaware_cambridge_full.sh` 的 baseline 阶段一致：

```text
train.py -s /mnt/pool/sqy/Cambridge_stdloc/KingsCollege \
  -m <TARGET_ROOT>/KingsCollege_baseline \
  -r 1 -f sp -g 3dgs --images processed --data_device cpu \
  --densify_grad_threshold 0.0004 \
  --position_lr_init 0.000016 \
  --scaling_lr 0.001 \
  --iterations 30000 \
  --train_detector \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --test_detector_iterations 30000 \
  --save_detector_iterations 30000 \
  --detector_folder detector
```

轻量 smoke 使用临时目录和 `PYTHON=/bin/echo` 验证了 fallback 分支会派发上述命令，而不会在默认模式误启长训练。

本轮已用 `TRAIN_MISSING_BASELINE=1` 补齐 KingsCollege 和 OldHospital baseline artifact，并完成字段/文件校验。

KingsCollege:

```text
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/point_cloud/iteration_30000/point_cloud.ply
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/detector/30000_detector.pth
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/detector/sampled_idx.pkl
PLY header: loc_0 ... loc_255
```

KingsCollege 训练日志：

```text
/mnt/pool/sqy/stdloc_la_full_runs/logs/kingscollege_baseline_full_20260623_155944.log
[ITER 30000] Evaluating detector: test loss 0.09767472742358901
[ITER 30000] Evaluating detector: train loss 0.10448088645935058
Training complete.
```

OldHospital:

```text
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/point_cloud/iteration_30000/point_cloud.ply
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/detector/30000_detector.pth
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/detector/sampled_idx.pkl
PLY header: loc_0 ... loc_255
```

OldHospital 训练日志：

```text
/mnt/pool/sqy/stdloc_la_full_runs/logs/oldhospital_baseline_full_20260623_181116.log
[ITER 30000] Evaluating detector: test loss 0.12411996169568418
[ITER 30000] Evaluating detector: train loss 0.11083250045776367
Training complete.
```

因此多场景脚本现在可以进入 KingsCollege 和 OldHospital；如果后续新增场景缺失 baseline 或 detector，仍会显式 skip，而不是静默失败或误跑半成品 baseline。旧验证命令的预期输出已过时：

```text
SCENES='KingsCollege OldHospital' SEEDS='2025' RUN_SWEEP=0 \
  scripts/run_locaware_v03_multiscene.sh
```

输出：

```text
不再因 KingsCollege/OldHospital 缺 baseline model 而 skip。
```

### KingsCollege v0.3 100-step smoke

利用补齐的 KingsCollege baseline，已跑通单场景、单 seed、100-step v0.3 feature-only smoke，并做 12px baseline/v0.3 对照：

```text
CUDA_VISIBLE_DEVICES=0 \
SCENES='KingsCollege' SEEDS='2025' RUN_SWEEP=1 SWEEP_THRESHOLDS='12' \
V03_STEPS='100' V03_MAX_STEP=100 V03_LOC_ANCHORS=2048 \
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_v03_multiscene \
PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python \
scripts/run_locaware_v03_multiscene.sh
```

Artifacts:

```text
/mnt/pool/sqy/stdloc_la_v03_multiscene/KingsCollege/seed_2025/KingsCollege_v03/point_cloud/iteration_30100/point_cloud.ply
/mnt/pool/sqy/stdloc_la_v03_multiscene/logs/kingscollege_v03_seed2025_100step_reproj12_20260623_175821.log
```

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege baseline 30000 | 0.172290 | 15.969916 | 0.014577 | 0.000000 | 565.292 |
| KingsCollege v0.3 30100 | 0.172025 | 15.810496 | 0.020408 | 0.000000 | 562.630 |

Results:

```text
/root/STDLoc/results/phase0-baseline-reproj12-_mnt_pool_sqy_stdloc_la_full_runs_KingsCollege_baseline-20260623_180432/summary.json
/root/STDLoc/results/phase-v03-30100-reproj12-_mnt_pool_sqy_stdloc_la_v03_multiscene_KingsCollege_seed_2025_KingsCollege_v03-20260623_180623/summary.json
/mnt/pool/sqy/stdloc_la_v03_multiscene/KingsCollege/seed_2025/analysis/paired_baseline_vs_v03_100step_reproj12.json
```

Paired baseline vs v0.3:

```text
query_count = 343
translation_delta_mean   = +3.7449 cm
translation_delta_median = -0.0503 cm
translation_CI95         = [-0.3363, 11.6096]
rotation_delta_mean      = -0.0059 deg
rotation_delta_median    = +0.0009 deg
rotation_CI95            = [-0.0172, 0.0015]
translation improved/degraded = 52.8% / 47.2%
R5 gain/loss = 3 / 1
R2 gain/loss = 0 / 0
```

结论：KingsCollege 链路已从 baseline artifact 准备推进到 v0.3 训练/评测闭环。100-step smoke 的 median TE 和 R5 是轻微正向，但 mean TE 受退化 query 拉高、CI 跨 0，只能作为多场景链路可行和弱正向诊断，不能作为多场景主结果。

### KingsCollege topology label-state 4-event smoke

在 KingsCollege v0.3 30100 checkpoint 上，先用 held-out train sparse matching 生成真实标签 state，再开启 conservative topology 4-event 复验。

Label-state diagnostic:

```text
label_state_output = /mnt/pool/sqy/stdloc_la_update1_multiscene_topology/KingsCollege/seed_2025/labels/KingsCollege_v03_30100_train64_sparse_label_state.pt
image_count = 64
avg_matches = 2048.0
avg_correct_matches = 539.0625
avg_inliers = 539.84375
visible_landmark_count = 14796
matched_landmark_count = 11950
inlier_landmark_count = 4505
spearman_utility_inlier_rate = 0.0138
spearman_calibrated_inlier_rate = 0.0285
top_quartile_calibrated_inlier_rate = 0.2041
bottom_quartile_calibrated_inlier_rate = 0.2383
```

结论：真实 sparse label state 已可写回并被 `train_locaware.py --localization_state_path` 恢复；但当前 utility/calibrated score 对 inlier value 的相关性仍很弱，且 top quartile 没有优于 bottom quartile。因此这轮 topology 只能作为保守生产路径复验，不应让 utility 主导 aggressive physical prune。

Run:

```text
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/KingsCollege/seed_2025/KingsCollege_v03_labelstate_topology_4evt_20260623_202737
```

Topology events:

```text
30125: physical_prune=0, split=11 parents, children=22, points 318593 -> 318604
30150: physical_prune=0, split=12 parents, children=24, points 318604 -> 318616
30175: physical_prune=0, split=12 parents, children=24, points 318616 -> 318628
30200: physical_prune=0, split=12 parents, children=24, points 318628 -> 318640
```

Remap:

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=318640
```

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege v0.3 30100 | 0.172025 | 15.810496 | 0.020408 | 0.000000 | 562.630 |
| KingsCollege topology 4-event 30200 | 0.171852 | 15.811232 | 0.020408 | 0.000000 | 562.615 |

Results:

```text
/root/STDLoc/results/phase-update1-kings-topology4evt-reproj12-_mnt_pool_sqy_stdloc_la_update1_multiscene_topology_KingsCollege_seed_2025_KingsCollege_v03_labelstate_topology_4evt_20260623_202737-20260623_203909/summary.json
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/KingsCollege/seed_2025/paired_v03_vs_topology4evt_reproj12.json
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/KingsCollege/seed_2025/paired_baseline_vs_topology4evt_reproj12.json
```

Paired v0.3 vs topology:

```text
query_count = 343
translation_delta_mean   = +0.0069 cm
translation_delta_median = +0.0000 cm
translation_CI95         = [-0.0116, 0.0248]
rotation_delta_mean      = +0.000019 deg
rotation_delta_median    = +0.000000 deg
rotation_CI95            = [-0.000165, 0.000185]
translation improved/degraded = 14.6% / 11.7%
R5 gain/loss = 0 / 0
R2 gain/loss = 0 / 0
```

结论：KingsCollege topology smoke 证明了 selected split、source-id remap、30200 sparse-only eval 的生产链路跨场景可运行；但 accuracy 基本等同于 v0.3，没有额外正向收益。physical prune 在这组 conservative threshold 下没有触发，这与弱 utility 标签诊断一致。

### OldHospital v0.3 100-step smoke

利用补齐的 OldHospital baseline，已跑通单场景、单 seed、100-step v0.3 feature-only smoke，并做 12px baseline/v0.3 对照：

```text
CUDA_VISIBLE_DEVICES=0 \
SCENES='OldHospital' SEEDS='2025' RUN_SWEEP=1 SWEEP_THRESHOLDS='12' \
V03_STEPS='100' V03_MAX_STEP=100 V03_LOC_ANCHORS=2048 \
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_v03_multiscene \
PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python \
scripts/run_locaware_v03_multiscene.sh
```

Artifacts:

```text
/mnt/pool/sqy/stdloc_la_v03_multiscene/OldHospital/seed_2025/OldHospital_v03/point_cloud/iteration_30100/point_cloud.ply
/mnt/pool/sqy/stdloc_la_v03_multiscene/logs/oldhospital_v03_seed2025_100step_reproj12_20260623_201219.log
```

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital baseline 30000 | 0.338004 | 18.394085 | 0.032967 | 0.005495 | 274.808 |
| OldHospital v0.3 30100 | 0.356078 | 19.502401 | 0.038462 | 0.005495 | 273.434 |

Results:

```text
/root/STDLoc/results/phase0-baseline-reproj12-_mnt_pool_sqy_stdloc_la_full_runs_OldHospital_baseline-20260623_201738/summary.json
/root/STDLoc/results/phase-v03-30100-reproj12-_mnt_pool_sqy_stdloc_la_v03_multiscene_OldHospital_seed_2025_OldHospital_v03-20260623_201932/summary.json
/mnt/pool/sqy/stdloc_la_v03_multiscene/OldHospital/seed_2025/analysis/paired_baseline_vs_v03_100step_reproj12.json
```

Paired baseline vs v0.3:

```text
query_count = 182
translation_delta_mean   = -0.2720 cm
translation_delta_median = +0.4782 cm
translation_CI95         = [-1.6235, 0.7975]
rotation_delta_mean      = -0.0027 deg
rotation_delta_median    = +0.0099 deg
rotation_CI95            = [-0.0224, 0.0130]
translation improved/degraded = 42.3% / 57.7%
R5 gain/loss = 3 / 2
R2 gain/loss = 0 / 0
```

结论：OldHospital 链路也已从 baseline artifact 准备推进到 v0.3 训练/评测闭环。100-step smoke 的 R5 是小幅正向，但 median AE/TE 和 inliers 变差，paired CI 跨 0；因此它只能说明多场景入口可运行，不能作为多场景正向主结果。

### OldHospital topology label-state 4-event smoke

在 OldHospital v0.3 30100 checkpoint 上，沿用 KingsCollege 的 label-state 和 conservative topology 设置，完成 4-event selected split 复验。

Label-state diagnostic:

```text
label_state_output = /mnt/pool/sqy/stdloc_la_update1_multiscene_topology/OldHospital/seed_2025/labels/OldHospital_v03_30100_train64_sparse_label_state.pt
image_count = 64
avg_matches = 2048.0
avg_correct_matches = 308.953125
avg_inliers = 310.078125
visible_landmark_count = 13642
matched_landmark_count = 10033
inlier_landmark_count = 3790
spearman_utility_inlier_rate = 0.0409
spearman_calibrated_inlier_rate = 0.0220
top_quartile_calibrated_inlier_rate = 0.1568
bottom_quartile_calibrated_inlier_rate = 0.1545
```

结论：OldHospital 的真实 inlier label state 同样可写回并恢复，但 calibrated utility 只比随机排序略好，不足以支持 aggressive prune。

Run:

```text
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/OldHospital/seed_2025/OldHospital_v03_labelstate_topology_4evt_20260623_204605
```

Topology events:

```text
30125: physical_prune=0, split=8 parents, children=16, points 405348 -> 405356
30150: physical_prune=0, split=8 parents, children=16, points 405356 -> 405364
30175: physical_prune=0, split=8 parents, children=16, points 405364 -> 405372
30200: physical_prune=0, split=8 parents, children=16, points 405372 -> 405380
```

Remap:

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=405380
```

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital v0.3 30100 | 0.356078 | 19.502401 | 0.038462 | 0.005495 | 273.434 |
| OldHospital topology 4-event 30200 | 0.354659 | 19.212361 | 0.038462 | 0.005495 | 273.335 |

Results:

```text
/root/STDLoc/results/phase-update1-oldhospital-topology4evt-reproj12-_mnt_pool_sqy_stdloc_la_update1_multiscene_topology_OldHospital_seed_2025_OldHospital_v03_labelstate_topology_4evt_20260623_204605-20260623_205039/summary.json
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/OldHospital/seed_2025/paired_v03_vs_topology4evt_reproj12.json
/mnt/pool/sqy/stdloc_la_update1_multiscene_topology/OldHospital/seed_2025/paired_baseline_vs_topology4evt_reproj12.json
```

Paired v0.3 vs topology:

```text
query_count = 182
translation_delta_mean   = -0.0924 cm
translation_delta_median = +0.0000 cm
translation_CI95         = [-0.3453, 0.1584]
rotation_delta_mean      = -0.0007 deg
rotation_delta_median    = +0.0000 deg
rotation_CI95            = [-0.0042, 0.0034]
translation improved/degraded = 40.1% / 24.7%
R5 gain/loss = 0 / 0
R2 gain/loss = 0 / 0
```

结论：OldHospital topology smoke 给出弱正向中位 TE/AE 和 paired mean TE 改善，但 CI 跨 0、recall 持平、inliers 略降。它增强了 topology 生产链路跨场景可用性的证据，也说明当前 utility 下 conservative split 比 physical prune 更可靠。

### Cambridge 三场景 3 seed v0.3 100-step smoke

在补齐 KingsCollege / OldHospital baseline artifacts 后，已补跑三场景 3 seed 的 100-step v0.3 feature-only smoke。该组实验只适合作为可行性与稳定性诊断，不替代 full-length 多场景主表。

Run:

```text
SCENES='ShopFacade KingsCollege OldHospital' SEEDS='2026 2027' RUN_SWEEP=1 \
SWEEP_THRESHOLDS='12' V03_STEPS='100' V03_MAX_STEP=100 V03_LOC_ANCHORS=2048 \
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_v03_multiscene \
scripts/run_locaware_v03_multiscene.sh
```

结合已完成的 seed 2025，12px sparse-only 3 seed 平均如下：

| Scene | Baseline AE | v0.3 AE | Baseline TE | v0.3 TE | Baseline R5 | v0.3 R5 | Baseline R2 | v0.3 R2 | Baseline Inliers | v0.3 Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.166517 | 0.161976 | 3.349951 | 3.203474 | 0.728155 | 0.731392 | 0.262136 | 0.255663 | 388.107 | 425.780 |
| KingsCollege | 0.172290 | 0.173076 | 15.969916 | 15.932772 | 0.014577 | 0.016521 | 0.000000 | 0.000000 | 565.292 | 563.715 |
| OldHospital | 0.338004 | 0.345774 | 18.394085 | 18.920540 | 0.032967 | 0.042125 | 0.005495 | 0.007326 | 274.808 | 273.967 |

单 seed 结果：

| Scene | Seed | AE | TE | R5 | R2 | Inliers | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ShopFacade | 2025 | 0.163457 | 3.159915 | 0.737864 | 0.271845 | 416.689 | 相对 baseline 正向 |
| ShopFacade | 2026 | 0.157036 | 3.250086 | 0.718447 | 0.252427 | 427.126 | TE/inliers 正向，R5/R2 回落 |
| ShopFacade | 2027 | 0.165435 | 3.200420 | 0.737864 | 0.242718 | 433.524 | TE/inliers/R5 正向，R2 回落 |
| KingsCollege | 2025 | 0.172025 | 15.810496 | 0.020408 | 0.000000 | 562.630 | TE/R5 轻微正向，inliers 回落 |
| KingsCollege | 2026 | 0.169044 | 15.777218 | 0.011662 | 0.000000 | 565.426 | AE/TE/inliers 正向，R5 回落 |
| KingsCollege | 2027 | 0.178157 | 16.210601 | 0.017493 | 0.000000 | 563.090 | R5 正向，AE/TE/inliers 回落 |
| OldHospital | 2025 | 0.356078 | 19.502401 | 0.038462 | 0.005495 | 273.434 | R5 正向，AE/TE/inliers 回落 |
| OldHospital | 2026 | 0.342521 | 18.465806 | 0.038462 | 0.005495 | 273.522 | R5 正向，TE/AE/inliers 略差 |
| OldHospital | 2027 | 0.338722 | 18.793412 | 0.049451 | 0.010989 | 274.945 | R5/R2/inliers 正向，TE/AE 略差 |

结论：

1. ShopFacade 的 3 seed 平均支持 `feature-only direct sparse distillation` 作为当前最可靠的正向子命题。
2. KingsCollege 平均基本持平，说明多场景入口和 baseline artifact 闭环已可用，但不能提供强正向。
3. OldHospital recall 指标改善，但 median AE/TE 和 inliers 不稳定，说明当前 v0.3 仍需要 full-length 训练或更可靠 utility/teacher。
4. 因此多场景问题已经从“入口和资产缺失”推进到“效果稳定性和 full-length 主实验”。

新增 dense-to-sparse KL 主实验入口：

```text
scripts/run_densekl_v03_cambridge.sh
```

该脚本串联：

1. dense responsibility diagnostics；
2. `train_locaware.py --loc_teacher dense --loc_dense_kl_weight ...`；
3. baseline detector/landmark sparse-only evaluation。

轻量入口验证：

```text
RUN_DIAGNOSTICS=0 RUN_EVAL=0 DENSEKL_STEPS=0 \
  DENSEKL_MODEL=/mnt/pool/sqy/stdloc_la_v03_runs/ShopFacade_v03_100_20260623_114535 \
  SOURCE_MODEL=/mnt/pool/sqy/stdloc_la_v03_runs/ShopFacade_v03_100_20260623_114535 \
  scripts/run_densekl_v03_cambridge.sh
```

输出：

```text
[LA-STDLoc dense-KL] Skip training: found iteration 30100.
```

ShopFacade dense-KL top32 smoke 结果：

```text
Run: /mnt/pool/sqy/stdloc_la_update1_densekl_smoke/ShopFacade_v03_densekl_top32_20260623_140923
Result: /root/STDLoc/results/phase-update1-densekl-v03-top32-30200-reproj12-_mnt_pool_sqy_stdloc_la_update1_densekl_smoke_ShopFacade_v03_densekl_top32_20260623_140923-20260623_141143
```

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| v03 30100 | 0.163457 | 3.159915 | 0.737864 | 0.271845 | 416.689 |
| dense-KL top32 30200 | 0.153583 | 3.350100 | 0.718447 | 0.262136 | 411.417 |

Paired v03 vs dense-KL:

```text
translation_delta_mean   = +0.0180 cm
translation_delta_median = -0.0338 cm
translation_CI95         = [-0.3198, 0.4203]
rotation_delta_mean      = +0.0020 deg
rotation_delta_median    = -0.0018 deg
rotation_CI95            = [-0.0159, 0.0206]
translation improved/degraded = 51.5% / 48.5%
R5 gain/loss = 1 / 3
R2 gain/loss = 3 / 4
```

Dense responsibility diagnostic:

```text
/mnt/pool/sqy/stdloc_la_update1_densekl_smoke/ShopFacade_densekl_responsibility_diag.json
image_count = 8
total_valid_anchor_count = 4081
mean_responsibility_reconstruction_mean_cosine = 0.7012
mean_responsibility_reconstruction_p10_cosine = 0.4026
mean_kl_loss = 7.1779
```

结论：dense responsibility/KL 路径已可运行并有 responsibility 重构信号，但当前 top32/100-step smoke 没有转化为 sparse-only 正向结果；它不能作为主结果，只能作为后续调参和 teacher 设计诊断入口。

### geometry-balanced 2x2 诊断

`LA_update1.md` 建议的 2x2 设计已跑过，当前 ShopFacade pilot 中 geometry-balanced selector 不是主要正向来源。

| Map | Selector | AE | TE | R5 | R2 | Inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | original | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| baseline | balanced | 0.166268 | 3.373933 | 0.699029 | 0.174757 | 252.709 |
| LA/v03 | original | 0.163457 | 3.159915 | 0.737864 | 0.271845 | 416.689 |
| LA/v03 | balanced | 0.169684 | 3.045594 | 0.718447 | 0.203883 | 253.786 |

结论：balanced selector 降低了一部分 TE，但明显牺牲 R5/R2 和 inlier 数；当前主收益仍来自 descriptor/topology 路径，而不是 selector 替换。

## 当前闭环判断

当前证据支持：

```text
localization-aware descriptor learning
+ sparse-label guided conservative physical prune/split
=> better sparse-only relocalization on ShopFacade pilot
+ runnable full-length topology mutation/remap path on Cambridge 3 scenes
```

但结论边界需要保持清晰：

1. 强正向仍主要来自 ShopFacade pilot；Cambridge 三场景 full-length topology 主表是 mixed。
2. 4-event topology 相对 no-split adapt 的证据强于相对 v0.3 最优 checkpoint 的证据。
3. topology 的收益主要体现在部分 seed 的 TE/AE、R5/R2 或 baseline12 对比；对 v0.3 30500 的 paired CI 大多跨 0。
4. physical prune 已触发并验证，但弱 utility 下的 aggressive prune 风险很明显：KingsCollege/OldHospital seed2025 prune 量过大，seed2026/seed2027 只 split 不 prune 反而更稳。
5. topology 当前的主要贡献仍是验证生产 mutation、source remap 和 full-length 复验路径可用；要成为论文主结果，还需要更可靠的 utility/teacher 和 pruning policy。

## 后续未闭合项

这些属于论文主结果要求，不应混入当前 pilot 的完成声明：

1. 多场景：Cambridge 三场景 full-length v0.3 和 topology 30500 -> 30600 三 seed 复验已经完成；未闭合的是效果稳定性，而不是入口或资产链路。
2. 多 seed：ShopFacade topology 相对 baseline 和 seed2025 v0.3 有正向证据；KingsCollege 主要是边界性 TE/AE 改善；OldHospital recall/R2 偶有收益但 TE/AE 不稳，不能作为强正向主表。
3. dense-stage improvement 作为 teacher 的完整闭环：当前 dense responsibility/KL 有 smoke 和代码路径，但 ShopFacade top32/100-step smoke 不是正向结果。
4. topology 的下一步不是继续证明“会不会跑”，而是改进 utility/teacher 或 pruning policy，使 physical prune 不再依赖弱标签下的偶然 aggressive mask，并让 paired CI 从跨 0 变成稳定正向。

当前阻碍不再是 Cambridge 三场景 baseline artifact、单场景 smoke 入口、100-step 多 seed 调度、full-length v0.3 队列或 full-length topology 入口。现有 `/mnt/pool/sqy/ulfloc_repro_20260607` Cambridge map 仍是 RGB-only 3DGS，不含 `loc_*` feature，不能作为 SOURCE_ROOT 直接复用；但 `/mnt/pool/sqy/stdloc_la_full_runs` 下的 ShopFacade/KingsCollege/OldHospital baseline artifact 已满足 v0.3 多场景入口 contract。下一步应聚焦 utility 校准、dense/sparse teacher 质量和 topology policy，而不是再补相同配置的可运行性实验。

## 本轮复核

### full-length v0.3 多场景进展

本轮已启动 Cambridge 三场景 full-length v0.3，多 GPU 并行运行路径：

```text
/mnt/pool/sqy/stdloc_la_v03_full_length
```

当前已落盘的 sparse-only 12px 结果：

| Scene | Seed | Iter | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 2025 | 30500 | 0.155737 | 3.063995 | 0.747573 | 0.310680 | 449.922 |
| ShopFacade | 2025 | 31000 | 0.159487 | 3.008847 | 0.776699 | 0.281553 | 438.291 |
| ShopFacade | 2025 | 32000 | 0.159120 | 2.972945 | 0.766990 | 0.281553 | 428.806 |
| ShopFacade | 2026 | 30500 | 0.154011 | 2.919792 | 0.766990 | 0.300971 | 444.748 |
| ShopFacade | 2026 | 31000 | 0.150140 | 3.218461 | 0.737864 | 0.252427 | 435.117 |
| ShopFacade | 2026 | 32000 | 0.182046 | 3.520231 | 0.640777 | 0.213592 | 422.456 |
| ShopFacade | 2027 | 30500 | 0.159834 | 3.142369 | 0.757282 | 0.320388 | 487.971 |
| ShopFacade | 2027 | 31000 | 0.161231 | 3.172485 | 0.776699 | 0.194175 | 496.835 |
| ShopFacade | 2027 | 32000 | 0.177145 | 3.128850 | 0.737864 | 0.213592 | 499.893 |
| KingsCollege | 2025 | 30500 | 0.176857 | 16.472355 | 0.008746 | 0.000000 | 560.630 |
| KingsCollege | 2025 | 31000 | 0.179776 | 16.547602 | 0.011662 | 0.000000 | 549.367 |
| KingsCollege | 2025 | 32000 | 0.177201 | 16.655494 | 0.008746 | 0.000000 | 549.589 |
| KingsCollege | 2026 | 30500 | 0.175687 | 16.476057 | 0.020408 | 0.000000 | 553.035 |
| KingsCollege | 2026 | 31000 | 0.179526 | 16.286497 | 0.005831 | 0.000000 | 552.108 |
| KingsCollege | 2026 | 32000 | 0.179587 | 16.423497 | 0.014577 | 0.002915 | 546.653 |
| KingsCollege | 2027 | 30500 | 0.175066 | 16.304958 | 0.008746 | 0.000000 | 554.994 |
| KingsCollege | 2027 | 31000 | 0.180118 | 16.533637 | 0.011662 | 0.002915 | 550.845 |
| KingsCollege | 2027 | 32000 | 0.177385 | 16.656110 | 0.011662 | 0.002915 | 547.781 |
| OldHospital | 2025 | 30500 | 0.338012 | 18.108805 | 0.027473 | 0.005495 | 271.253 |
| OldHospital | 2025 | 31000 | 0.353851 | 19.251827 | 0.054945 | 0.000000 | 267.857 |
| OldHospital | 2025 | 32000 | 0.360693 | 20.388210 | 0.038462 | 0.000000 | 265.445 |
| OldHospital | 2026 | 30500 | 0.341424 | 18.043248 | 0.038462 | 0.000000 | 268.286 |
| OldHospital | 2026 | 31000 | 0.348931 | 20.415073 | 0.032967 | 0.000000 | 265.846 |
| OldHospital | 2026 | 32000 | 0.360251 | 19.582635 | 0.032967 | 0.005495 | 263.225 |
| OldHospital | 2027 | 30500 | 0.355030 | 19.257225 | 0.049451 | 0.000000 | 268.736 |
| OldHospital | 2027 | 31000 | 0.351582 | 19.417655 | 0.049451 | 0.000000 | 266.038 |
| OldHospital | 2027 | 32000 | 0.358776 | 19.519626 | 0.049451 | 0.000000 | 262.533 |

阶段性观察：

1. ShopFacade full-length 三个 seed 均支持 30500 是更稳的 topology 起点；seed2026/seed2027 到 32000 均有 R2 或 R5 退化，因此 topology 复验不应只盯最终 iteration。
2. KingsCollege/OldHospital 三个 seed 均已完成 30500/31000/32000 评测；两者 5cm/2deg recall 仍很低，更适合作为边界案例而不是强正向主表。
3. 三张 RTX 3090 已同时用于 full-length/topology 队列；GPU 空闲窗口主要发生在数据加载和评测切换阶段，本轮已利用该窗口并行完成 KingsCollege/OldHospital 的 topology seed 复验。

### full-length topology physical prune/split 复验

复核对象是 full-length v0.3 中当前最稳的候选：

```text
/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
iteration_30500
```

入口脚本：

```text
scripts/run_locaware_v03_topology_full.sh
```

第一轮使用默认 physical prune threshold `-3.0`，验证了 label-state、selected split、source-id remap、sparse-only eval 链路；但 physical prune 未实际触发：

| Variant | physical prune | split parents | point count | remap missing | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.3 30500 | - | - | 342918 | - | 0.155737 | 3.063995 | 0.747573 | 0.310680 | 449.922 |
| topology default | 0 | 136 | 343054 | 0 | 0.151992 | 2.900497 | 0.747573 | 0.291262 | 444.806 |

为避免“开关打开但未触发”的证据缺口，本轮先直接估算 physical prune mask：`rgb=0.002, loc=0.002, utility_threshold=1e-9` 会剪 153 个非保护点；随后在独立目录复跑：

```text
/mnt/pool/sqy/stdloc_la_v03_topology_full/ShopFacade/seed_2025/ShopFacade_v03_topology_from_30500_prune153
```

事件记录：

```text
30525: physical_prune=153, split=34 parents, children=68, points 342765 -> 342799
30550: physical_prune=0,   split=34 parents, children=68, points 342799 -> 342833
30575: physical_prune=0,   split=34 parents, children=68, points 342833 -> 342867
30600: physical_prune=0,   split=34 parents, children=68, points 342867 -> 342901
```

Remap:

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=342901
```

Sparse-only 12px 对比：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline 12px | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| v0.3 30500 | 0.155737 | 3.063995 | 0.747573 | 0.310680 | 449.922 |
| topology prune153 30600 | 0.151992 | 2.900497 | 0.747573 | 0.291262 | 444.777 |

Paired v0.3 30500 vs topology prune153：

```text
query_count = 103
translation_delta_mean   = -0.6638 cm
translation_delta_median = -0.0632 cm
translation_CI95         = [-1.3222, -0.0897]
rotation_delta_mean      = -0.0299 deg
rotation_delta_median    = -0.0040 deg
rotation_CI95            = [-0.0627, -0.0055]
translation improved/degraded = 55.3% / 44.7%
R5 gain/loss = 2 / 2
R2 gain/loss = 5 / 7
```

Paired baseline12 vs topology prune153：

```text
query_count = 103
translation_delta_mean   = -1.0781 cm
translation_delta_median = -0.2760 cm
translation_CI95         = [-2.1455, -0.0848]
rotation_delta_mean      = -0.0491 deg
rotation_delta_median    = -0.0138 deg
rotation_CI95            = [-0.1004, -0.0013]
translation improved/degraded = 65.0% / 35.0%
R5 gain/loss = 7 / 5
R2 gain/loss = 11 / 8
```

为验证该结论不是 seed2025 单点，本轮又在 ShopFacade seed2026 的 full-length v0.3 30500 checkpoint 上用同一组保守 physical prune 参数复跑：

```text
/mnt/pool/sqy/stdloc_la_v03_topology_full/ShopFacade/seed_2026/ShopFacade_v03_topology_from_30500_prune002
```

seed2026 事件记录：

```text
30525: physical_prune=153, split=34 parents, children=68, points 342765 -> 342799
30550: physical_prune=0,   split=34 parents, children=68, points 342799 -> 342833
30575: physical_prune=0,   split=34 parents, children=68, points 342833 -> 342867
30600: physical_prune=0,   split=34 parents, children=68, points 342867 -> 342901
```

seed2026 remap:

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=342901
```

seed2026 sparse-only 12px 对比：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline 12px | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| v0.3 30500 | 0.154011 | 2.919792 | 0.766990 | 0.300971 | 444.748 |
| topology prune002 30600 | 0.150124 | 2.953724 | 0.737864 | 0.281553 | 442.252 |

Paired v0.3 30500 vs seed2026 topology：

```text
query_count = 103
translation_delta_mean   = +0.0030 cm
translation_delta_median = +0.0658 cm
translation_CI95         = [-0.2415, +0.2200]
rotation_delta_mean      = +0.0014 deg
rotation_delta_median    = +0.0028 deg
rotation_CI95            = [-0.0101, +0.0120]
translation improved/degraded = 41.7% / 58.3%
R5 gain/loss = 0 / 3
R2 gain/loss = 3 / 5
```

Paired baseline12 vs seed2026 topology：

```text
query_count = 103
translation_delta_mean   = -0.9233 cm
translation_delta_median = -0.2974 cm
translation_CI95         = [-1.5834, -0.4398]
rotation_delta_mean      = -0.0443 deg
rotation_delta_median    = -0.0152 deg
rotation_CI95            = [-0.0735, -0.0189]
translation improved/degraded = 59.2% / 40.8%
R5 gain/loss = 6 / 5
R2 gain/loss = 12 / 10
```

seed2027 同样在独立目录完成：

```text
/mnt/pool/sqy/stdloc_la_v03_topology_full/ShopFacade/seed_2027/ShopFacade_v03_topology_from_30500_prune002
```

seed2027 artifact 证据：

```text
iteration_30500 point count = 342918
iteration_30600 point count = 342901
iteration_30600 loc_source_index unique = 342765
iteration_30600 duplicate source rows = 136
remap source_count/remapped_count/missing_count = 16384 / 16384 / 0
```

seed2027 sparse-only 12px 对比：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline 12px | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| v0.3 30500 | 0.159834 | 3.142369 | 0.757282 | 0.320388 | 487.971 |
| topology prune002 30600 | 0.161918 | 3.273127 | 0.776699 | 0.262136 | 481.728 |

Paired v0.3 30500 vs seed2027 topology：

```text
query_count = 103
translation_delta_mean   = +0.0953 cm
translation_delta_median = +0.0823 cm
translation_CI95         = [-0.1507, +0.3517]
rotation_delta_mean      = +0.0010 deg
rotation_delta_median    = +0.0009 deg
rotation_CI95            = [-0.0093, +0.0127]
translation improved/degraded = 40.8% / 59.2%
R5 gain/loss = 2 / 0
R2 gain/loss = 2 / 8
```

Paired baseline12 vs seed2027 topology：

```text
query_count = 103
translation_delta_mean   = -1.3413 cm
translation_delta_median = -0.2622 cm
translation_CI95         = [-2.6425, -0.4736]
rotation_delta_mean      = -0.0621 deg
rotation_delta_median    = -0.0065 deg
rotation_CI95            = [-0.1197, -0.0208]
translation improved/degraded = 60.2% / 39.8%
R5 gain/loss = 8 / 3
R2 gain/loss = 10 / 10
```

结论：full-length v0.3 30500 上的 topology 复验已经在 ShopFacade seed2025/seed2026/seed2027 三个 seed 上完成，并保持 source-id remap 完整。三个 seed 的最终 topology artifact 均为 `342901` 点、`342765` unique source id、`136` duplicate source rows，符合 selected split 后 child 复用 source id 的预期。seed2025 相对 v0.3 30500 的 TE/AE 和 paired mean 是正向；seed2026/seed2027 相对 v0.3 30500 基本中性略负、CI 跨 0，但二者相对 baseline12 均明确正向。这支持“localization-aware mapping + 保守 topology mutation 可改善相对 baseline 的中位姿态误差”的闭环，但还不足以宣称 topology 全面优于 v0.3 最优 checkpoint 或 recall 全面提升。

#### KingsCollege / OldHospital full-length topology 边界复验

为避免 topology 结论只停留在 ShopFacade，本轮继续用同一入口在 KingsCollege/OldHospital 的 full-length v0.3 30500 checkpoint 上补齐 seed2025/seed2026/seed2027：

```text
scripts/run_locaware_v03_topology_full.sh
TOPOLOGY_PHYSICAL_RGB_THRESHOLD=0.02
TOPOLOGY_PHYSICAL_LOC_THRESHOLD=0.02
TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD=-3.0
```

三场景 topology 30600 sparse-only 12px 汇总：

| Scene | Seed | Variant | AE | TE | R5 | R2 | Inliers |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 2025 | prune153 | 0.151992 | 2.900497 | 0.747573 | 0.291262 | 444.777 |
| ShopFacade | 2026 | prune002 | 0.150124 | 2.953724 | 0.737864 | 0.281553 | 442.252 |
| ShopFacade | 2027 | prune002 | 0.161918 | 3.273127 | 0.776699 | 0.262136 | 481.728 |
| KingsCollege | 2025 | prune002 | 0.176462 | 16.073017 | 0.008746 | 0.000000 | 559.236 |
| KingsCollege | 2026 | prune002 | 0.173142 | 15.973850 | 0.017493 | 0.000000 | 553.776 |
| KingsCollege | 2027 | prune002 | 0.171134 | 16.069320 | 0.008746 | 0.000000 | 555.423 |
| OldHospital | 2025 | prune002 | 0.348512 | 19.611244 | 0.054945 | 0.010989 | 272.582 |
| OldHospital | 2026 | prune002 | 0.357269 | 19.453447 | 0.060440 | 0.010989 | 270.500 |
| OldHospital | 2027 | prune002 | 0.357129 | 19.923596 | 0.043956 | 0.005495 | 269.412 |

Paired v0.3 30500 vs topology 30600：

| Scene | Seed | TE mean delta | TE median delta | TE CI95 | AE mean delta | AE CI95 | Improved/Degraded | R5 gain/loss | R2 gain/loss |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |
| ShopFacade | 2025 | -0.6638 | -0.0632 | [-1.3222, -0.0897] | -0.0299 | [-0.0627, -0.0055] | 0.553/0.447 | 2/2 | 5/7 |
| ShopFacade | 2026 | +0.0030 | +0.0658 | [-0.2415, +0.2200] | +0.0014 | [-0.0101, +0.0120] | 0.417/0.583 | 0/3 | 3/5 |
| ShopFacade | 2027 | +0.0953 | +0.0823 | [-0.1507, +0.3517] | +0.0010 | [-0.0093, +0.0127] | 0.408/0.592 | 2/0 | 2/8 |
| KingsCollege | 2025 | -5.3834 | -0.0408 | [-16.2538, +0.2772] | -0.0195 | [-0.0691, +0.0068] | 0.513/0.487 | 1/1 | 0/0 |
| KingsCollege | 2026 | -1.2927 | +0.1452 | [-4.4001, +0.4273] | -0.0237 | [-0.0802, +0.0068] | 0.478/0.522 | 3/4 | 0/0 |
| KingsCollege | 2027 | +0.0311 | +0.1179 | [-0.3289, +0.3670] | +0.0003 | [-0.0055, +0.0055] | 0.475/0.525 | 1/1 | 0/0 |
| OldHospital | 2025 | +2.5361 | +0.2040 | [+0.5718, +4.8128] | +0.0356 | [+0.0062, +0.0675] | 0.478/0.522 | 6/1 | 1/0 |
| OldHospital | 2026 | -0.0514 | +0.2396 | [-1.2081, +1.0003] | +0.0010 | [-0.0172, +0.0194] | 0.484/0.516 | 6/2 | 2/0 |
| OldHospital | 2027 | +0.6273 | +0.4528 | [-0.3020, +1.6594] | +0.0098 | [-0.0059, +0.0250] | 0.429/0.571 | 2/3 | 1/0 |

Source-id/remap audit：

| Scene | Seed | Point count | Unique source ids | Duplicate source rows | Inferred physical prune | Remap missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 2025 | 342901 | 342765 | 136 | 153 | 0 |
| ShopFacade | 2026 | 342901 | 342765 | 136 | 153 | 0 |
| ShopFacade | 2027 | 342901 | 342765 | 136 | 153 | 0 |
| KingsCollege | 2025 | 281962 | 281850 | 112 | 36743 | 0 |
| KingsCollege | 2026 | 318717 | 318593 | 124 | 0 | 0 |
| KingsCollege | 2027 | 318717 | 318593 | 124 | 0 | 0 |
| OldHospital | 2025 | 340015 | 339883 | 132 | 65465 | 0 |
| OldHospital | 2026 | 405508 | 405348 | 160 | 0 | 0 |
| OldHospital | 2027 | 405508 | 405348 | 160 | 0 | 0 |

结论：source-id/remap 已闭环，且 selected split 的 child source duplication 行为符合预期。效果层面，ShopFacade 是当前唯一可作为正向 pilot 的场景；KingsCollege 的 aggregate TE/AE 有改善迹象，但 paired median/CI 不支持强结论；OldHospital 的 R5/R2 有局部收益，但 TE/AE 多数偏负。尤其 KingsCollege/OldHospital seed2025 的 physical prune 分别达到约 `36.7k` 和 `65.5k` 个 source，说明弱 utility 下 aggressive prune 会成为风险源；seed2026/seed2027 只发生 split 不发生 prune，结果反而更稳定。

### topology physical prune/split artifact audit

复核对象：

```text
/mnt/pool/sqy/stdloc_la_update1_topology_labelstate/ShopFacade_v03_labelstate_phys_split_4evt_20260623_151301
```

保存后的 checkpoint / PLY 状态：

| Item | Value |
| --- | ---: |
| iteration_30100 point count | 342918 |
| iteration_30200 point count | 342901 |
| net point delta | -17 |
| iteration_30200 unique source ids | 342765 |
| iteration_30200 duplicate source rows | 136 |

该状态与 4-event 记录一致：首个 event 有 physical prune，四个 event 总计引入 split child source duplicates；最终 source-id remap 覆盖全部 sparse landmarks：

```text
source_count=16384
remapped_count=16384
missing_count=0
point_count=342901
```

12px sparse-only 复核：

| Metric | no-split adapt | topology 4-event | Delta |
| --- | ---: | ---: | ---: |
| AE | 0.162927 | 0.162880 | -0.000046 |
| TE | 3.185332 | 2.976683 | -0.208649 |
| R5 | 0.728155 | 0.737864 | +0.009709 |
| R2 | 0.242718 | 0.262136 | +0.019417 |
| Inliers | 431.087379 | 431.922330 | +0.834951 |

paired no-split vs topology 4-event at 12px：

```text
query_count = 103
translation_delta_mean   = -0.2918 cm
translation_delta_median = -0.2056 cm
translation_CI95         = [-0.6046, -0.0378]
rotation_delta_mean      = -0.0154 deg
rotation_delta_median    = -0.0070 deg
rotation_CI95            = [-0.0306, -0.0033]
translation improved/degraded = 61.2% / 38.8%
R5 gain/loss = 2 / 1
R2 gain/loss = 7 / 5
```

solver-threshold sweep AUC delta：

| Metric | Delta |
| --- | ---: |
| median AE | -0.005696 |
| median TE | -0.134801 |
| R5 | +0.018724 |
| R2 | -0.000693 |
| avg inliers | +1.407767 |

## 回归验证

当前核心回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_descriptor_diagnostics \
  tests.test_full_script_args \
  tests.test_localization_utility \
  tests.test_topology_controller \
  tests.test_geometry_selector \
  tests.test_eval_analysis \
  tests.test_train_locaware_masks \
  tests.test_dense_distill \
  tests.test_direct_landmark_teacher \
  tests.test_dense_teacher_losses
```

结果：

```text
Ran 92 tests in 2.758s
OK
```

2026-06-23 复核时默认 `python` 指向 `/root/miniconda3/envs/iclpose/bin/python`，该 PyPy 环境无法加载当前 torch/numpy C extensions；改用上面的 `ulfloc_repro` Python 后同一组 92 个测试通过。

脚本语法检查：

```text
bash -n scripts/prepare_cambridge_baseline_artifacts.sh \
  scripts/run_locaware_v03_multiscene.sh \
  scripts/run_densekl_v03_cambridge.sh \
  scripts/run_locaware_v03_shopfacade.sh \
  scripts/run_geometry_balance_2x2_shopfacade.sh \
  scripts/run_locaware_v03_topology_full.sh
```

结果：

```text
OK
```

新增 paired JSON sanity：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python - <<'PY'
...
PY
```

结果：

```text
paired json sanity OK
```
