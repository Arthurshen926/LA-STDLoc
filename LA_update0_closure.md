# LA_update0 落实闭环记录

日期：2026-06-23

## 总结

`LA_update0.md` 的核心判断是正确的：上一版完整 Phase1-6 退化不能靠继续扫权重解决，必须先修正 sparse-only 因果链。当前分支已经把实现重心从“dense rendered feature 训练”收缩到更可控的 `feature-only + direct sparse landmark distillation`，并补齐了 topology 生产路径。ShopFacade pilot 已经得到初步正向结果；dense-KL 和多场景部分仍属于后续验证，不应混入当前完成声明。

当前可闭合的主结论是：

```text
相同/受控 sparse pipeline
+ direct landmark-to-query feature distillation
+ conservative topology physical prune/split
=> ShopFacade sparse-only pilot 指标改善
```

边界也很明确：

1. 这是 ShopFacade pilot，不是 Cambridge 全量主表。
2. topology 4-event 相对 no-split adapt 的证据强于相对 v0.3 的证据。
3. dense responsibility/KL 路径可运行，但 top32/100-step smoke 没有转化成 sparse-only 正向结果。
4. Cambridge 三场景 baseline artifact 已通过 full fallback 补齐；旧 repro 目录里的 KingsCollege/OldHospital PLY 是 RGB-only，不含 `loc_*`，不能作为 feature-3DGS baseline 直接复用。

## LA_update0 问题对照

| LA_update0 问题 | 当前处理 | 闭环状态 |
| --- | --- | --- |
| P0-1 matching stats 没有归因到具体 Gaussian | 新增 dense responsibility 聚合，把 anchor 统计按 contributor/responsibility scatter 到 visible Gaussian；同时补充 sparse inlier 诊断和 geometry value | 部分闭合：代码路径和诊断已通，dense-KL smoke 尚未正向 |
| P0-2 prototype 使用 Gaussian self-feature | direct teacher 直接采样 query feature 作为 Gaussian descriptor 监督；dense 聚合的 prototype 也来自 query observation；`loc_proto_weight`、`loc_rank_weight` 默认 0 | 已闭合首版 |
| P0-3 LA global top-k 破坏空间覆盖 | v0.2/v0.3 首先固定 baseline sampled_idx 和 baseline detector；新增 full-bank diagnostics、geometry-balanced selector、2x2 诊断，不再把 global top-k 当主路径 | 已规避主风险；geometry-balanced selector 当前不是正向来源 |
| P0-4 3DGS topology mutation 未接通 | `GaussianModel` 补齐 localization buffers、optimizer tensor mutation、`densify_and_split_selected()`、physical prune/split 后 source-id remap；controller 使用显式 selected mask | 已闭合并有物理 prune/split 验证 |
| P0-4 split threshold 可能误分裂全部点 | controller 不再用 `grad_threshold=0` 间接表达 split，而是显式调用 `densify_and_split_selected(selected_mask=...)` | 已闭合 |
| P0-5 feature phase 不是纯 feature-only | `--use_loc_opacity` 改为可关且默认 false；`--feature_only` 只训练 `loc_feature`；loc opacity loss 默认 0 | 已闭合 |
| anchor correspondence 遮挡/错误监督 | direct teacher 接入 target-view depth/alpha consistency；dense responsibility 支持 depth consistency weighting | 已接入，验证范围为 ShopFacade |
| identity contrastive false negatives | direct multiview loss 支持同 landmark 多视图 positives、投影邻域 ignore、hard negatives；新增 view-diverse observation memory | 已闭合首版 |
| utility 是否能预测真实定位价值 | 新增 sparse descriptor/inlier diagnostics、paired per-query analysis、solver-threshold sweep | 诊断闭合；utility 作为主控采样仍需更多证据 |

## 主要代码改动

### Feature-only / direct sparse distillation

- `train_locaware.py`
  - 新增 `--feature_only`，feature-only 只训练 `loc_feature`。
  - 新增 `--loc_teacher {dense,direct}`，当前正向 pilot 使用 direct teacher。
  - 新增 `--loc_direct_weight`、`--loc_multiview_weight`、`--loc_anchor_weight`。
  - `--use_loc_opacity` 默认 false，支持 `--no-use_loc_opacity`。
  - prototype/rank 默认关闭，避免旧语义继续影响 feature-only 因果实验。
- `localization_training/direct_landmark_teacher.py`
  - 实现 baseline selected landmark 到 query feature 的直接蒸馏。
  - 支持 target depth/alpha visibility 过滤。
  - 支持 multiview positives、ignore radius、hard negative margin。
  - `LandmarkObservationMemory` 支持 view direction、distance、confidence 的 view-diverse memory。

### Dense responsibility / KL 路径

- `localization_training/dense_teacher.py`
  - 新增 `aggregate_dense_anchor_stats()`，不再把同一 episode 的全局均值扩展给所有 visible Gaussian。
  - 新增 dense-to-sparse KL loss 和 responsibility reconstruction diagnostics。
- `localization_training/dense_distill.py`
  - 新增 `gaussian_teacher_distribution()`、`dense_to_sparse_kl()`、`responsibility_reconstruction_metrics()`。
- `scripts/diagnose_dense_responsibility.py`
  - 输出 dense responsibility 重构质量和 KL 诊断。
- `scripts/run_densekl_v03_cambridge.sh`
  - 串联 diagnostics、dense-KL training、sparse-only eval。

### Topology

- `scene/gaussian_model.py`
  - 补齐 `_loc_opacity`、localization EMA buffers、source-index state。
  - 拓扑后同步 optimizer tensors 和 localization buffers。
  - 新增 `densify_and_split_selected()`。
- `localization_training/topology_controller.py`
  - physical prune / selected split 走显式 mask。
  - 增加 landmark protection、cooldown、event 统计。
- `scripts/remap_topology_landmarks.py`
  - split/prune 后把原 sampled landmark source id remap 到当前点索引。

### 诊断和实验脚本

- `scripts/diagnose_sparse_descriptors.py`
  - full-bank descriptor metrics。
- `scripts/diagnose_sparse_inliers.py`
  - sparse inlier/value diagnostic，支持 `--localization_state_path`。
- `scripts/analyze_paired_sparse_results.py`
  - paired per-query delta 和 bootstrap CI。
- `scripts/analyze_solver_threshold_sweep.py`
  - solver-threshold sweep AUC。
- `scripts/run_locaware_v03_shopfacade.sh`
  - ShopFacade v0.3 受控 feature-only 入口。
- `scripts/run_locaware_v03_multiscene.sh`
  - 多场景多 seed 调度入口，缺 baseline/detector 会显式 skip。
- `scripts/prepare_cambridge_baseline_artifacts.sh`
  - 校验 baseline PLY 必须含 `loc_*` 字段；必要时可用 `TRAIN_MISSING_BASELINE=1` 跑完整 baseline。

## ShopFacade 结果

### baseline vs v0.3 feature-only

评测：12px sparse-only，固定 ShopFacade。

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| v0.3 feature-only 30100 | 0.163457 | 3.159915 | 0.737864 | 0.271845 | 416.689 |

证据：

```text
/root/STDLoc/results/phase0-baseline-v03sweep-reproj12-_mnt_pool_sqy_stdloc_la_full_runs_ShopFacade_baseline-20260623_120443/summary.json
/root/STDLoc/results/phase-v03-100-30100-reproj12-_mnt_pool_sqy_stdloc_la_v03_runs_ShopFacade_v03_100_20260623_114535-20260623_120525/summary.json
```

结论：在受控 feature-only 设定下，ShopFacade sparse-only 有初步正向结果，符合 `LA_update0` 要求的第一条因果链验证。

### topology 4-event

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

12px sparse-only:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-split adapt | 0.162927 | 3.185332 | 0.728155 | 0.242718 | 431.087 |
| topology 4-event | 0.162880 | 2.976683 | 0.737864 | 0.262136 | 431.922 |

paired no-split vs topology 4-event:

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

solver sweep AUC, no-split vs topology 4-event:

| Metric | no-split | topology | Delta |
| --- | ---: | ---: | ---: |
| median AE | 0.163828 | 0.158131 | -0.005696 |
| median TE | 3.118784 | 2.983983 | -0.134801 |
| R5 | 0.728155 | 0.746879 | +0.018724 |
| R2 | 0.251040 | 0.250347 | -0.000693 |
| avg inliers | 391.059 | 392.467 | +1.408 |

结论：topology 生产路径已经跑通，且 conservative physical prune/split 在 ShopFacade pilot 上带来 TE/R5/inlier 正向趋势。R2 没有提升，不能夸大。

### dense-KL smoke

ShopFacade dense-KL top32/100-step smoke:

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0.3 30100 | 0.163457 | 3.159915 | 0.737864 | 0.271845 | 416.689 |
| dense-KL top32 30200 | 0.153583 | 3.350100 | 0.718447 | 0.262136 | 411.417 |

paired v0.3 vs dense-KL:

```text
translation_delta_mean   = +0.0180 cm
translation_delta_median = -0.0338 cm
translation_CI95         = [-0.3198, 0.4203]
R5 gain/loss = 1 / 3
R2 gain/loss = 3 / 4
```

dense responsibility diagnostic:

```text
image_count = 8
total_valid_anchor_count = 4081
mean_responsibility_reconstruction_mean_cosine = 0.7012
mean_responsibility_reconstruction_p10_cosine = 0.4026
mean_kl_loss = 7.1779
```

结论：dense responsibility/KL 有重构信号，工程路径可运行；当前 smoke 不是正向主结果。

## 多场景进展

数据目录 `/mnt/pool/sqy/Cambridge_stdloc` 可用，包含 ShopFacade、KingsCollege、OldHospital。最初阻碍是非 ShopFacade 的 baseline artifact：

```text
/mnt/pool/sqy/ulfloc_repro_20260607/{KingsCollege,OldHospital}
```

存在 30k RGB 3DGS PLY，但不含 `loc_*` feature 字段，不能直接用于 STDLoc detector/landmark 训练。

已新增 `scripts/prepare_cambridge_baseline_artifacts.sh`，默认会检查 `loc_*` 字段并阻止 RGB-only PLY 误入训练；需要时可显式：

```bash
SCENES='KingsCollege OldHospital' TRAIN_MISSING_BASELINE=1 \
  SOURCE_ROOT=/mnt/pool/sqy/ulfloc_repro_20260607 \
  TARGET_ROOT=/mnt/pool/sqy/stdloc_la_full_runs \
  PYTHON=/root/miniconda3/envs/ulfloc_repro/bin/python \
  scripts/prepare_cambridge_baseline_artifacts.sh
```

本轮已完成 KingsCollege 和 OldHospital full baseline fallback，并通过 artifact/字段校验：

```text
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/point_cloud/iteration_30000/point_cloud.ply  387M
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/detector/30000_detector.pth                  1.5M
/mnt/pool/sqy/stdloc_la_full_runs/KingsCollege_baseline/detector/sampled_idx.pkl                    129K
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/point_cloud/iteration_30000/point_cloud.ply  492M
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/detector/30000_detector.pth                  1.5M
/mnt/pool/sqy/stdloc_la_full_runs/OldHospital_baseline/detector/sampled_idx.pkl                    129K
```

PLY header 已确认包含 `loc_0` ... `loc_255`。训练日志：

```text
/mnt/pool/sqy/stdloc_la_full_runs/logs/kingscollege_baseline_full_20260623_155944.log
[ITER 30000] Evaluating detector: test loss 0.09767472742358901
[ITER 30000] Evaluating detector: train loss 0.10448088645935058
Training complete.

/mnt/pool/sqy/stdloc_la_full_runs/logs/oldhospital_baseline_full_20260623_181116.log
[ITER 30000] Evaluating detector: test loss 0.12411996169568418
[ITER 30000] Evaluating detector: train loss 0.11083250045776367
Training complete.
```

这闭合了 Cambridge 三场景 baseline artifact 缺口；多场景脚本现在可以进入 ShopFacade、KingsCollege 和 OldHospital，不再因缺 baseline/detector 被动 skip。

KingsCollege 已完成单场景、单 seed、100-step v0.3 feature-only smoke，并完成 12px baseline/v0.3 对照：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege baseline 30000 | 0.172290 | 15.969916 | 0.014577 | 0.000000 | 565.292 |
| KingsCollege v0.3 30100 | 0.172025 | 15.810496 | 0.020408 | 0.000000 | 562.630 |

paired baseline vs v0.3:

```text
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

结论：KingsCollege 链路已经从 baseline artifact 准备推进到 v0.3 训练/评测闭环。100-step smoke 的 median TE 和 R5 是轻微正向，但 mean TE 受退化 query 拉高、CI 跨 0，只能作为链路可行和弱正向诊断，不能作为多场景主结果。

OldHospital 也已完成单场景、单 seed、100-step v0.3 feature-only smoke，并完成 12px baseline/v0.3 对照：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital baseline 30000 | 0.338004 | 18.394085 | 0.032967 | 0.005495 | 274.808 |
| OldHospital v0.3 30100 | 0.356078 | 19.502401 | 0.038462 | 0.005495 | 273.434 |

paired baseline vs v0.3:

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

结论：OldHospital v0.3 smoke 进一步证明多场景入口和训练/评测链路可运行，但结果不是正向主结果：R5 小幅提升，median AE/TE 和 inliers 变差，paired CI 跨 0。

## 补充验证：多场景 3 seed 100-step v0.3 smoke

在 `LA_update0` 的主闭环之后，又补跑了 Cambridge 三场景的 100-step v0.3 feature-only smoke。该组实验使用相同 baseline detector / sampled landmarks / 12px sparse-only 评测，只改变 Gaussian localization feature。它适合作为可行性和稳定性补充，不应升级为论文主表。

### 3 seed 平均

| Scene | Baseline AE | v0.3 AE | Baseline TE | v0.3 TE | Baseline R5 | v0.3 R5 | Baseline R2 | v0.3 R2 | Baseline Inliers | v0.3 Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.166517 | 0.161976 | 3.349951 | 3.203474 | 0.728155 | 0.731392 | 0.262136 | 0.255663 | 388.107 | 425.780 |
| KingsCollege | 0.172290 | 0.173076 | 15.969916 | 15.932772 | 0.014577 | 0.016521 | 0.000000 | 0.000000 | 565.292 | 563.715 |
| OldHospital | 0.338004 | 0.345774 | 18.394085 | 18.920540 | 0.032967 | 0.042125 | 0.005495 | 0.007326 | 274.808 | 273.967 |

### 单 seed 结果

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

补充结论：

1. ShopFacade 的 3 seed 平均支持 `feature-only direct sparse distillation` 作为初版正向路径：TE、AE、R5、inliers 均优于 baseline，但 R2 不稳定。
2. KingsCollege 3 seed 平均基本持平：TE/R5 极小幅正向，AE/inliers 轻微回落，不能作为强正向。
3. OldHospital 3 seed 平均显示 recall 提升，但 median AE/TE 和 inliers 不稳定，仍不是正向主结果。
4. 因此 LA_update0 的问题闭环应表述为：核心实现缺口和受控实验链路已闭合，ShopFacade pilot 正向；多场景 smoke 已跑通但结果混合，论文级多场景/多 seed 主结果未闭合。

## 回归验证

最近一次核心回归命令：

```bash
env CUDA_HOME=/usr/local/cuda-11.8 \
  PATH=/usr/local/cuda-11.8/bin:/root/miniconda3/envs/ulfloc_repro/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64 \
  PYTHONPATH=/root/STDLoc \
  /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
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
Ran 91 tests in 2.777s
OK
```

脚本语法检查：

```bash
bash -n scripts/prepare_cambridge_baseline_artifacts.sh \
  scripts/run_locaware_v03_multiscene.sh \
  scripts/run_densekl_v03_cambridge.sh \
  scripts/run_locaware_v03_shopfacade.sh \
  scripts/run_geometry_balance_2x2_shopfacade.sh
```

结果：exit 0。

## 当前闭环状态

已闭合：

1. feature-only 控制变量。
2. direct sparse landmark distillation。
3. query observation prototype 语义。
4. target depth/alpha visibility。
5. topology physical prune/split 生产接线。
6. paired/sweep/diagnostic 工具链。

部分闭合：

1. dense responsibility/KL：路径和诊断闭合，结果未转正。
2. utility 作为 sampling/topology 主控：已有诊断和 topology pilot，仍需多场景验证。
3. geometry-balanced selector：入口和 2x2 诊断闭合，但当前不作为主正向来源。

未闭合：

1. Cambridge 多场景、多 seed 主表。
2. KingsCollege/OldHospital full v0.3、多 seed 和 topology 复验。
3. dense-stage improvement 到 sparse map 的稳定正向结果。
4. topology 在多场景 full-bank v0.3 上的复验。
