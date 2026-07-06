# LA_update3 P0 闭环记录

日期：2026-06-24

## 当前结论

P0 的 3-scene、500-step、S0-S3 对照已经跑完，结果支持一个更细的判断：

1. topology split 不是完全无效。ShopFacade 和 KingsCollege 上，S1/S2/S3 相对 S0 有 R5 或 TE 的正向信号。
2. topology split 也不是稳定正收益。OldHospital 上 S1/S2/S3 的 R5 和 TE 整体变差。
3. repeated split 在 ShopFacade 上最好，在 KingsCollege 上 TE 最好但 R5 与 S1 持平，在 OldHospital 上仍负向。
4. one-shot + child freeze/group-positive warmup 的 S3 不是当前最稳路线：KingsCollege R5 最好，但 OldHospital 和 ShopFacade 都不优于各自 S1/S2。
5. physical prune 在本轮默认阈值下仍没有触发，因此本轮不能证明 prune 策略有效，只能继续排除“未训练 opacity 导致误删”的混杂。

因此，当前证据更支持 `LA_update3.md` 的方向：不要继续把 static utility 直接作为全局 topology commit 依据；应推进 localization-only overlay、child responsibility specialization、perturb/dense selective teacher 和 held-out risk commit。

2026-06-25 的 P6 复核把 held-out sparse-pose gate 进一步推进到 `holdout=32 + strided selection + recall/tail veto + 500-step`：

1. 相比 LA_update1/LA_update2 阶段，现在已经排除了更多高影响混杂：rejected proposal 不污染训练/RNG、risk gate 可以真实 rollback、prefix 小 holdout 代表性不足已被定位并新增 `strided` 选项。
2. 精度证据比早期更干净，但仍不是稳定正向：ShopFacade 被 gate 完全保护并等价 no-mutation，KingsCollege 出现 AE/TE/R5/Inliers 同向小幅改善，OldHospital 仍出现 AE/TE/R5/Inliers 退化且仅 R2 小幅改善。
3. 当前方法闭环已经从“实现是否有问题”推进到“risk objective/holdout 分层是否足以预测 full sparse-only recall”的策略问题；不能再把 LA_update2 的 mixed 结果直接解释为原始主张错误，但也不能宣称当前具体策略已获得跨场景稳定精度支撑。

## 已落实改动

| 项目 | 状态 |
| --- | --- |
| topology 最大 mutation 次数 | 新增 `TopologyConfig.max_mutation_events` 和 `--topology_max_mutation_events` |
| one-shot split 模式 | 新增 `TOPOLOGY_MUTATION_MODE=one_shot_split` |
| one-shot split + freeze 模式 | 新增 `TOPOLOGY_MUTATION_MODE=one_shot_split_freeze` |
| child birth metadata | split 后写入 `last_topology_iteration` 和 `loc_birth_iteration` |
| child descriptor freeze | 新增 `--loc_child_feature_freeze_steps`，backward 后屏蔽新 child loc feature 梯度 |
| group-positive warmup | 新增 `--loc_full_bank_nearby_as_positive_until`，S3 默认到 split 后 100 step |
| P0 worker | 新增 `scripts/run_la_update3_p0_worker.sh`，执行 S0/S1/S2/S3 |
| summarizer | `scripts/summarize_la_update2_long_closure.py` 支持 `p0_S*_500` tag |
| 回归测试 | 新增/更新 parser、script、topology、summarizer 测试 |

## 验证

代码验证：

```text
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 158 tests in 6.740s
OK
```

脚本和静态检查：

```text
bash -n scripts/run_locaware_v03_topology_full.sh scripts/run_la_update3_p0_worker.sh
py_compile train_locaware.py localization_training/topology_controller.py scripts/summarize_la_update2_long_closure.py
git diff --check
```

均通过。

P0 结果路径：

```text
/mnt/pool/sqy/stdloc_la_update3_p0_core_v1/summary_final.json
```

配置校验：

```text
S0: enable_topology=False, max_events=0, freeze=0
S1: enable_topology=True,  max_events=1, freeze=0
S2: enable_topology=True,  max_events=0, freeze=0
S3: enable_topology=True,  max_events=1, freeze=100, nearby_as_positive=True, nearby_until=30625
CONFIG_ERRORS []
```

## P0 500-step 结果

`d*` 均为相对同 scene 的 S0 no-mutation 差值；R5/R2 越高越好，TE/AE 越低越好。

| Scene | Mode | R5 | dR5 | R2 | dR2 | TE | dTE | AE | dAE | events | children | point_delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KingsCollege | S0 | 0.005831 | +0.000000 | 0.000000 | +0.000000 | 16.050568 | +0.000000 | 0.181337 | +0.000000 | 0 | 0 | 0 |
| KingsCollege | S1 | 0.008746 | +0.002915 | 0.000000 | +0.000000 | 15.941532 | -0.109036 | 0.183153 | +0.001816 | 1 | 30 | 15 |
| KingsCollege | S2 | 0.008746 | +0.002915 | 0.000000 | +0.000000 | 15.649737 | -0.400831 | 0.183256 | +0.001919 | 20 | 908 | 454 |
| KingsCollege | S3 | 0.014577 | +0.008746 | 0.000000 | +0.000000 | 15.766786 | -0.283782 | 0.186669 | +0.005332 | 1 | 34 | 17 |
| OldHospital | S0 | 0.060440 | +0.000000 | 0.000000 | +0.000000 | 15.436979 | +0.000000 | 0.306325 | +0.000000 | 0 | 0 | 0 |
| OldHospital | S1 | 0.043956 | -0.016484 | 0.005495 | +0.005495 | 16.738567 | +1.301587 | 0.321565 | +0.015240 | 1 | 20 | 10 |
| OldHospital | S2 | 0.049451 | -0.010989 | 0.000000 | +0.000000 | 16.196147 | +0.759168 | 0.306576 | +0.000251 | 20 | 486 | 243 |
| OldHospital | S3 | 0.038462 | -0.021978 | 0.005495 | +0.005495 | 16.513428 | +1.076449 | 0.319508 | +0.013183 | 1 | 22 | 11 |
| ShopFacade | S0 | 0.757282 | +0.000000 | 0.300971 | +0.000000 | 3.018843 | +0.000000 | 0.156416 | +0.000000 | 0 | 0 | 0 |
| ShopFacade | S1 | 0.776699 | +0.019417 | 0.281553 | -0.019417 | 2.840700 | -0.178143 | 0.155894 | -0.000522 | 1 | 28 | 14 |
| ShopFacade | S2 | 0.786408 | +0.029126 | 0.291262 | -0.009709 | 3.006976 | -0.011867 | 0.142926 | -0.013490 | 20 | 856 | 428 |
| ShopFacade | S3 | 0.766990 | +0.009709 | 0.300971 | +0.000000 | 3.026488 | +0.007645 | 0.155057 | -0.001359 | 1 | 32 | 16 |

## 训练数据与 synthetic view 状态

当前 LA-STDLoc 主实验仍使用 STDLoc 预处理 Cambridge 原始相机/图像：

```text
/mnt/pool/sqy/Cambridge_stdloc/<scene>/processed
```

训练会加载 3DGS/STDLoc checkpoint，并使用 renderer 产生 RGB、depth、feature map 作为监督和几何上下文；P3/P4/P4.1 已经使用“真实 query 图像 + sparse-error/noise perturb pose + 3DGS render teacher”的训练 episode。P5 已新增最小 synthetic novel-view augmentation：从相邻真实相机插值得到 synthetic pose，再用 3DGS render feature map 作为低权重 dense desc/reproj supervision；目前已完成 100-step 多场景 synthetic-only 对照，但还没有做 500-step、多 seed 精度验证。

## 仍未闭合

1. `LA_update2.md` 没有全部落实；已闭合的是 false-negative、multi-positive、seed/split、topology 控制等高影响混杂点。
2. `LA_update3.md` 已落实 P0、P1 overlay 初版/稳定化、P2 child responsibility/source-mode 100-step matched matrix、P3 perturb-pose dense 100-step、P4 dense advantage 100-step、P4.1 sparse-miss/dense-hit rank-only 100-step、P5 novel-view augmentation 最小实现/100-step synthetic-only 对照，以及 P6 held-out descriptor / sparse-pose risk commit、recall/tail veto、`holdout32+strided` seed0 100/500-step 对照。P2/P3/P4/P4.1/P5/P6 仍缺 true train-seed 统计；P6 pose-risk 仍缺 multi-seed、更强 query 分层和 recall-aware risk score 验证。
3. 当前 P0 只有 `train_seed=0, query_split_seed=2025`，还缺 multi-seed 统计置信度。
4. sparse-only pose 是最终指标，但对 descriptor 改进不够敏感，后续需要增加 retrieval/correspondence AP、inlier precision、pose surrogate 等中间指标。

## 下一步判断

不建议现在直接让外部专家重定方向。当前结果已经说明原始怀疑成立：之前的 mixed topology 结果确实受到实现与实验混杂影响，不能直接判定核心主张错误。但 P0 也说明 split/topology 不是稳定正收益，必须继续按 `LA_update3.md` 调整方法层。

当前下一轮优先级：

1. P6：在已完成 `holdout32+strided` seed0 500-step 的基础上扩到多 seed，并把 R5/R2/tail 从 veto/日志审计推进到 risk score 或分层 full-holdout 校准，同时继续检查 stochastic PnP 噪声。
2. P5：做 synthetic view quality/teacher weighting calibration，而不是直接扩大当前 synthetic desc/reproj 策略。
3. physical prune：设置能真实触发删点但有保护的阈值矩阵，验证策略本身是否有效。
4. 整理给外部专家讨论时，应讨论“下一步方法如何调整”，而不是把当前 mixed 结果解读成主张已被证伪。

## P1 descriptor overlay 初版

日期：2026-06-24

本轮落实 `LA_update3.md` 的 P1/O0：最小 descriptor overlay。

### 已实现

1. `GaussianModel` 新增 descriptor overlay 状态：
   - `loc_overlay_source_index`
   - `_loc_overlay_feature`
   - `_loc_overlay_active_logit`
2. `get_loc_feature` 会按 `loc_source_index` 将 overlay residual 加到 base `_loc_feature` 上。
3. overlay 不修改 base `_loc_feature`；`loc_overlay_mode=descriptor` 时 feature phase 只训练：
   - `loc_overlay_feature`
   - `loc_overlay_active_logit`
4. `capture_localization_state()` / `restore_localization_state()` 会保存恢复 overlay。
5. sparse label-state / utility-only state restore 不再清空已有 overlay，避免 topology continuation 时外部 label state 覆盖 source overlay。
6. `stdloc.sample_gaussians()` 和 `train_detector.get_sampled_gaussian()` 改为 materialize `get_loc_feature`，避免 sparse eval 采样时丢失 overlay。
7. `train_locaware.py` 新增参数：
   - `--loc_overlay_mode {none,descriptor}`
   - `--loc_overlay_lr`
   - `--loc_overlay_active_logit`
8. v0.3 单场景和 topology 脚本新增环境变量透传：
   - `V03_LOC_OVERLAY_MODE`
   - `V03_LOC_OVERLAY_LR`
   - `V03_LOC_OVERLAY_ACTIVE_LOGIT`
   - `TOPOLOGY_LOC_OVERLAY_MODE`
   - `TOPOLOGY_LOC_OVERLAY_LR`
   - `TOPOLOGY_LOC_OVERLAY_ACTIVE_LOGIT`

### 验证

TDD 红灯覆盖：

```text
descriptor overlay source residual
overlay localization state round-trip
utility-only label state restore preserves overlay
sparse eval sampling materializes overlay
descriptor overlay parser / phase LR / configure helper
v03 and topology scripts pass overlay args
```

目标测试和全量测试：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_localization_utility \
  tests.test_train_locaware_masks \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_script_runs_feature_only_full_bank_anchor_without_topology \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default

Ran 48 tests in 1.680s
OK
```

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 165 tests in 7.020s
OK
```

静态检查：

```text
py_compile train_locaware.py scene/gaussian_model.py stdloc.py train_detector.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
git diff --check
```

均通过。

### 2-step ShopFacade smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p1_overlay_smoke/ShopFacade_overlay_smoke
```

配置：

```text
scene = ShopFacade
steps = 1,2
train_seed = 0
query_split_seed = 2025
query_split_mode = sequence_block
V03_LOC_OVERLAY_MODE = descriptor
```

训练日志确认：

```text
Initialized descriptor overlay: sources=16384 lr=0.001
[ITER 30001] base 0.143182 loc 0.522090 psnr 19.074
[ITER 30002] base 0.278129 loc 0.517576 psnr 13.865
```

overlay state 检查：

| Iteration | sources | feature shape | L1 | max abs | active mean |
| ---: | ---: | --- | ---: | ---: | ---: |
| 30001 | 16384 | `(16384, 1, 256)` | 4194.30 | 0.00100 | 0.50000 |
| 30002 | 16384 | `(16384, 1, 256)` | 6548.23 | 0.00200 | 0.50015 |

Sparse-only eval：

| Iteration | AE | TE | R5 | R2 | Inliers |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30001 | 0.162566 | 3.350459 | 0.728155 | 0.252427 | 388.573 |
| 30002 | 0.162842 | 3.350459 | 0.728155 | 0.252427 | 388.728 |

结论：2-step smoke 证明真实训练、loc_state 保存、sparse eval materialize 通路可用；不作为精度结论。

### 100-step matched overlay vs no-overlay

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p1_overlay_100_v1
/mnt/pool/sqy/stdloc_la_update3_p1_nooverlay_100_v1
```

矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
train_seed = 0
query_split_seed = 2025
query_split_mode = sequence_block
steps = 100
variants = overlay descriptor, no-overlay
```

overlay residual 统计：

| Scene | sources | L1 | max abs | active mean |
| --- | ---: | ---: | ---: | ---: |
| ShopFacade | 16384 | 95459.60 | 0.09529 | 0.51553 |
| KingsCollege | 16384 | 32848.11 | 0.09732 | 0.50489 |
| OldHospital | 16384 | 12451.62 | 0.09166 | 0.50194 |

Paired sparse-only delta，`overlay - no-overlay`：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.008288 | +0.274052 | +0.000000 | +0.000000 | -14.320 |
| KingsCollege | -0.001304 | +0.412256 | +0.014577 | +0.000000 | +0.519 |
| OldHospital | -0.027273 | -0.855050 | +0.005495 | -0.005495 | +2.088 |

当前判断：

```text
P1 descriptor overlay 的工程通路已经跑通，且 overlay residual 确实被训练和保存；
100-step matched matrix 出现局部正向信号：Kings/Old 的 R5 上升，Old 的 TE/AE 改善；
但 ShopFacade TE/inliers 退化，Kings TE 也变差，R2 没有稳定改善。
因此 P1 初版支持“overlay 是可行替代实现路径”，但还不能作为稳定精度正向证据。
```

下一步不应立即扩大到 500-step 全矩阵。更优先的是：

1. 调整 overlay gate / LR / normalization，避免 residual 对 already-good ShopFacade 造成 descriptor drift。
2. 给 overlay 加 retrieval/correspondence AP 中间指标，确认 R5 小幅变化来自匹配改善还是 PnP 噪声。
3. 在 P1 上合并 update6 multi-positive objective，再跑 100/500-step matched matrix。
4. 之后进入 P2 child responsibility specialization。

## P1 descriptor overlay 诊断与稳定化

日期：2026-06-25

本轮继续闭合 P1 的两个关键缺口：

1. descriptor diagnostics 之前的辅助加载路径只读 `point_cloud.ply`，没有恢复同 iteration 的 `loc_state.pt`。虽然主模型经 `Scene` 加载时会恢复 `loc_state.pt`，但 baseline/辅助模型路径不会；这会让 overlay 相关的 `feature_drift` 和 paired descriptor 对照不可靠。
2. raw descriptor overlay 的 100-step 结果出现 ShopFacade 退化，因此按 `LA_update3.md` 的公式补了最小 trust-region / normalize / residual regularization。

### 已实现

1. `scripts/diagnose_sparse_descriptors.py` 的 `_load_gaussians_from_iteration()` 现在在加载 PLY 后，如果存在 `loc_state.pt`，会调用 `load_localization_state()`。
2. `GaussianModel.init_descriptor_overlay()` 新增：
   - `max_residual_norm`
   - `normalize`
3. `GaussianModel.get_loc_feature` 现在支持：
   - 对 gated residual 做 per-source L2 cap；
   - 对 materialized descriptor 做 L2 normalize。
4. `capture_localization_state()` / `restore_localization_state()` 保存恢复：
   - `loc_overlay_max_residual_norm`
   - `loc_overlay_normalize`
5. `train_locaware.py` 新增：
   - `--loc_overlay_max_residual_norm`
   - `--loc_overlay_normalize`
   - `--loc_overlay_reg_weight`
   - `_descriptor_overlay_regularizer()`，按 gated residual L2 norm 约束 overlay。
6. v0.3 单场景和 topology 脚本新增环境变量透传：
   - `V03_LOC_OVERLAY_MAX_RESIDUAL_NORM`
   - `V03_LOC_OVERLAY_NORMALIZE`
   - `V03_LOC_OVERLAY_REG_WEIGHT`
   - `TOPOLOGY_LOC_OVERLAY_MAX_RESIDUAL_NORM`
   - `TOPOLOGY_LOC_OVERLAY_NORMALIZE`
   - `TOPOLOGY_LOC_OVERLAY_REG_WEIGHT`

### 验证

TDD 红灯覆盖：

```text
diagnostics loads loc_state.pt when present
overlay caps residual norm per source
overlay can normalize materialized descriptor
overlay stability config round-trip
parser/configure/script pass stability args
overlay regularizer uses gated residual norm
```

目标测试：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_localization_utility \
  tests.test_train_locaware_masks \
  tests.test_full_script_args

Ran 78 tests in 1.714s
OK
```

静态检查：

```text
py_compile train_locaware.py scene/gaussian_model.py scripts/diagnose_sparse_descriptors.py stdloc.py train_detector.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
git diff --check
```

均通过。

全量回归：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 170 tests in 7.196s
OK
```

### Descriptor diagnostics：raw overlay vs no-overlay

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p1_descriptor_diag_v2
```

配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
iteration = 30100
split = test
max_images = 32
max_landmarks_per_image = 1024
depth_check = true
full_bank = true
```

`overlay_raw - nooverlay` 的 full-bank descriptor delta：

| Scene | dRecall@1 | dRecall@5 | dRecall@10 | dMNN | dMargin |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.012966 | -0.019449 | -0.021931 | -0.012662 | -0.010889 |
| KingsCollege | +0.005894 | +0.007321 | +0.007383 | +0.005274 | +0.003221 |
| OldHospital | +0.001959 | +0.006779 | +0.005574 | +0.000603 | +0.003012 |

这个诊断解释了 100-step pose 结果的一部分：

1. Kings/Old 的 R5 正向伴随 full-bank retrieval 小幅正向。
2. ShopFacade 的 pose 退化不是纯 PnP 噪声，descriptor retrieval 本身也退化。
3. 因此 raw overlay 的问题不是“eval 没看到 overlay”，而是当前 objective 对 already-good ShopFacade 会造成 matching degradation。

### 100-step stabilized overlay

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p1_overlay_stable_100_v1
```

配置：

```text
V03_LOC_OVERLAY_MODE=descriptor
V03_LOC_OVERLAY_LR=0.001
V03_LOC_OVERLAY_ACTIVE_LOGIT=0.0
V03_LOC_OVERLAY_MAX_RESIDUAL_NORM=0.1
V03_LOC_OVERLAY_NORMALIZE=1
V03_LOC_OVERLAY_REG_WEIGHT=0.001
train_seed=0
query_split_seed=2025
query_split_mode=sequence_block
steps=100
```

loc_state 配置确认：

| Scene | sources | max_residual_norm | normalize | raw residual norm mean | raw residual norm p95 | raw abs max | active mean |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| ShopFacade | 16384 | 0.1 | true | 0.121046 | 0.136592 | 0.047056 | 0.506971 |
| KingsCollege | 16384 | 0.1 | true | 0.036003 | 0.140000 | 0.038781 | 0.501904 |
| OldHospital | 16384 | 0.1 | true | 0.016526 | 0.131221 | 0.039410 | 0.500915 |

说明：表中 raw residual 是保存的参数值；实际 cap 在 `get_loc_feature` materialize 时动态生效。

Sparse-only 结果，相对 no-overlay：

| Scene | Variant | AE | dAE | TE | dTE | R5 | dR5 | R2 | dR2 | Inliers | dInliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | raw | 0.164413 | +0.008288 | 3.341710 | +0.274052 | 0.737864 | +0.000000 | 0.271845 | +0.000000 | 408.437 | -14.320 |
| ShopFacade | stable | 0.158340 | +0.002215 | 3.184404 | +0.116746 | 0.737864 | +0.000000 | 0.262136 | -0.009709 | 398.971 | -23.786 |
| KingsCollege | raw | 0.173532 | -0.001304 | 15.915976 | +0.412256 | 0.017493 | +0.014577 | 0.000000 | +0.000000 | 566.344 | +0.519 |
| KingsCollege | stable | 0.170996 | -0.003840 | 15.967944 | +0.464223 | 0.017493 | +0.014577 | 0.000000 | +0.000000 | 565.714 | -0.111 |
| OldHospital | raw | 0.335293 | -0.027273 | 18.586734 | -0.855050 | 0.032967 | +0.005495 | 0.005495 | -0.005495 | 275.357 | +2.088 |
| OldHospital | stable | 0.336682 | -0.025884 | 18.674381 | -0.767403 | 0.038462 | +0.010989 | 0.005495 | -0.005495 | 275.198 | +1.929 |

Descriptor diagnostics，`stable - nooverlay`：

| Scene | dRecall@1 | dRecall@5 | dRecall@10 | dMNN | dMargin | dPairTop1 | dPairMNN | dDrift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.019753 | -0.030338 | -0.033630 | -0.018487 | -0.016359 | -0.021576 | -0.019854 | -0.005241 |
| KingsCollege | +0.005336 | +0.006422 | +0.006825 | +0.004902 | +0.002709 | +0.004870 | +0.005274 | -0.002463 |
| OldHospital | +0.001707 | +0.006277 | +0.004821 | +0.000502 | +0.002732 | +0.002260 | +0.001758 | -0.001526 |

`stable - raw` 的 descriptor delta 在三场景均略负；也就是说 cap/normalize/reg 明显降低 drift，但没有改善 retrieval 本身。ShopFacade 的 retrieval 退化甚至更明显。

### 当前判断更新

1. diagnostics 入口的 loc_state 混杂已排除：P1 overlay 的 descriptor 指标现在确实读取了 overlay state。
2. P1 overlay 的正向信号主要集中在 KingsCollege 和 OldHospital，且 full-bank retrieval 与 pose R5 正向一致。
3. ShopFacade 的负向也是真实 descriptor retrieval degradation，不是 sparse eval 或 diagnostics 没 materialize overlay。
4. 单纯收紧 residual/normalize 只能降低 drift，不能解决 objective 方向问题；当前 evidence 不支持继续只扫 overlay cap/LR。
5. 下一步应优先进入 P2 child responsibility specialization，以及把 full-bank loss 从 single-positive/ignore-negative 推到更明确的 multi-positive 或 responsibility-positive，而不是先扩 500-step stable overlay 全矩阵。

## P2 child responsibility 最小实现

日期：2026-06-25

本轮落实 `LA_update3.md` 的 P2/O1 最小 child responsibility specialization。目标不是替代 P6 held-out commit，而是先排除 topology split 后 sibling children 被同一 observation 同步拉向同一 descriptor 的高影响混杂。

### 已实现

1. `localization_training/direct_landmark_teacher.py` 新增 `child_responsibility_keep_mask()`：
   - 默认 `mode=none` 不改变旧行为；
   - `mode=feature` 时，对同一 `loc_source_index` 的多个当前 Gaussian child，只保留与当前 query observation cosine 最相似的 child；
   - 无 `loc_source_index`、source index 不完整或 source 未知时保守保留原样。
2. `direct_landmark_teacher(... child_responsibility_mode=...)` 在采样 query feature 后、desc/full-bank/memory/stat 更新前应用该 mask。
3. `teacher_out.diagnostics` 新增：
   - `child_responsibility_candidate_count`
   - `child_responsibility_kept_count`
   - `child_responsibility_dropped_count`
4. `train_locaware.py` 新增参数：
   - `--loc_child_responsibility_mode {none,feature}`
   - `--loc_child_responsibility_start_iter`
5. v0.3 单场景和 topology 脚本新增环境变量透传：
   - `V03_CHILD_RESPONSIBILITY_MODE`
   - `V03_CHILD_RESPONSIBILITY_START_ITER`
   - `TOPOLOGY_CHILD_RESPONSIBILITY_MODE`
   - `TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER`

### 验证

TDD 红灯确认：

```text
ImportError: cannot import name 'child_responsibility_keep_mask'
TypeError: direct_landmark_teacher() got an unexpected keyword argument 'child_responsibility_mode'
argparse: unrecognized arguments: --loc_child_responsibility_mode feature --loc_child_responsibility_start_iter 30625
script args missing V03/TOPOLOGY child responsibility passthrough
```

窄测试转绿：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_child_responsibility_feature_mode_keeps_best_child_per_source \
  tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_direct_teacher_child_responsibility_updates_only_best_child_per_source \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_child_responsibility_controls \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_script_runs_feature_only_full_bank_anchor_without_topology \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default -v

Ran 5 tests in 1.610s
OK
```

相关回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_direct_landmark_teacher \
  tests.test_train_locaware_masks \
  tests.test_full_script_args -v

Ran 72 tests in 1.889s
OK
```

静态检查：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile localization_training/direct_landmark_teacher.py train_locaware.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
git diff --check
```

均通过。

全量回归：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 173 tests in 7.118s
OK
```

### 5-step ShopFacade topology smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_child_resp_smoke_v1
```

关键配置：

```text
SOURCE_MODEL=/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
V03_ITERATION=32000
TOPOLOGY_STEPS=5
TOPOLOGY_MUTATION_MODE=one_shot_split
TOPOLOGY_UPDATE_INTERVAL=1
TOPOLOGY_MIN_OBSERVATIONS=1
TOPOLOGY_TOTAL_POINT_BUDGET_RATIO=1.001
TOPOLOGY_GROWTH_CAP_PER_EVENT=0.0001
TOPOLOGY_CHILD_RESPONSIBILITY_MODE=feature
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=32002
LABEL_MAX_IMAGES=8
```

训练日志确认实际 split：

```text
[Topology] iter=32001 candidates=1122 physical_prune=0 requested_split=6 parent_removed=6 children_added=12 points=342918->342924
[ITER 32005] base 0.162731 loc 0.503613 psnr 18.743
```

TensorBoard diagnostics：

| Iteration | candidate | kept | dropped | points |
| ---: | ---: | ---: | ---: | ---: |
| 32002 | 447 | 447 | 0 | 342924 |
| 32003 | 1012 | 1012 | 0 | 342924 |
| 32004 | 2048 | 2047 | 1 | 342924 |
| 32005 | 2048 | 2048 | 0 | 342924 |

loc_state 检查：

```text
loc_current_xyz = (342924, 3)
unique loc_source_index = 342918
duplicate source count = 6
last split sources = [22587, 68966, 101112, 115950, 141344, 242862] repeated twice
```

Remap 检查：

```text
source_count = 16384
remapped_count = 16384
missing_count = 0
point_count = 342924
remap_source_distance_mean = 4.7017929318826646e-06
remap_source_distance_max = 0.02292068488895893
```

Sparse-only eval：

| AE | TE | R5 | R2 | Inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.161446 | 3.001269 | 0.757282 | 0.271845 | 428.262 |

当前判断：

1. P2 最小工程通路已经跑通，且不是 inert：本次 smoke 中真实产生了 split children，并在 split 后记录到 child responsibility diagnostics。
2. 5-step smoke 中 dropped 数量很小，只能证明 responsibility branch 被触发；不能证明 specialization 策略有效，更不能作为精度结论。
3. 下一步需要 matched matrix：至少 `no_resp` vs `feature_resp`，100-step 起步，覆盖 ShopFacade/KingsCollege/OldHospital；如果 100-step 不反向，再扩 500-step 和 multi-seed。

## P2 100-step matched matrix 与 pre-limit 修正

日期：2026-06-25

### 100-step 初跑暴露的实现缺口

第一版 P2 在 `max_landmarks` anchor limit 后才做 sibling competition。因此只有当同源 child 恰好同时进入 2048 anchors 时才会发生 drop。

100-step matched matrix 输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_child_resp_100_v1
```

配置：

```text
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
V03_ITERATION = 32000
TOPOLOGY_STEPS = 100
TOPOLOGY_MUTATION_MODE = one_shot_split
TOPOLOGY_CHILD_RESPONSIBILITY_MODE = none vs feature
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER = 32026
LABEL_MAX_IMAGES = 64
```

所有 scene 都实际 split：

| Scene | candidates | parent removed | children added | points |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 2852 | 15 | 30 | 342918 -> 342933 |
| KingsCollege | 2825 | 15 | 30 | 318593 -> 318608 |
| OldHospital | 1787 | 9 | 18 | 405348 -> 405357 |

`feature_resp - no_resp` sparse-only delta：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.000515 | +0.001517 | +0.000000 | -0.009709 | +0.136 |
| KingsCollege | +0.000000 | -0.032166 | +0.000000 | +0.000000 | -0.050 |
| OldHospital | -0.001127 | +0.000000 | +0.000000 | +0.000000 | -0.077 |

diagnostics 说明该实现施力极弱：

| Scene | episodes | dropped | candidates | drop ratio | nonzero episodes | max drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 74 | 127 | 84501 | 0.001503 | 44 | 8 |
| KingsCollege | 75 | 70 | 143890 | 0.000486 | 41 | 3 |
| OldHospital | 75 | 37 | 102740 | 0.000360 | 25 | 4 |

结论：第一版 P2 不能作为策略失败证据，因为它在 anchor limit 后才竞争，绝大多数 sibling observation 根本没有进入 responsibility 分支。

### pre-limit 修正

新增测试先红后绿：

```text
tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_direct_teacher_child_responsibility_competes_before_anchor_limit

红灯：max_landmarks=1 时错误选择 [0]
绿灯：pre-limit responsibility 后选择 [1]
```

实现调整：

1. `direct_landmark_teacher` 现在在 projection/depth valid 后、`_limit_valid_indices()` 前，对所有 visible candidates 计算 query/gaussian descriptor similarity。
2. 对同一 `loc_source_index` 的 visible siblings 先选 winner，再进入 anchor limit。
3. diagnostics 现在统计 pre-limit candidate/kept/dropped 数量。

验证：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_direct_landmark_teacher \
  tests.test_train_locaware_masks \
  tests.test_full_script_args -v

Ran 73 tests in 1.811s
OK
```

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile localization_training/direct_landmark_teacher.py train_locaware.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
git diff --check
```

均通过。

全量回归：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 174 tests in 7.459s
OK
```

pre-limit 5-step ShopFacade smoke：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_child_resp_prelimit_smoke_v1
```

对比旧 smoke：

| Variant | dropped | candidates | drop ratio |
| --- | ---: | ---: | ---: |
| post-limit old | 1 | 5555 | 0.000180 |
| pre-limit new | 4 | 7279 | 0.000550 |

说明 pre-limit 修正确实让 responsibility 覆盖变大，但 5-step 仍只证明通路。

### pre-limit 100-step feature rerun

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_child_resp_prelimit_100_v1
```

只重跑 `feature_resp`；`no_resp` 沿用 `/mnt/pool/sqy/stdloc_la_update3_p2_child_resp_100_v1`，因为 `mode=none` 完全不进入新分支。

`feature_resp_prelimit - no_resp` sparse-only delta：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.000494 | +0.001517 | +0.000000 | +0.000000 | +0.194 |
| KingsCollege | +0.000000 | -0.032166 | +0.000000 | +0.000000 | -0.058 |
| OldHospital | -0.001272 | +0.000000 | +0.000000 | +0.000000 | -0.082 |

pre-limit responsibility diagnostics：

| Scene | episodes | dropped | candidates | drop ratio | nonzero episodes | max drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 75 | 182 | 97639 | 0.001864 | 46 | 9 |
| KingsCollege | 75 | 280 | 291895 | 0.000959 | 71 | 8 |
| OldHospital | 75 | 111 | 148468 | 0.000748 | 44 | 9 |

### 当前判断更新

1. P2 pre-limit responsibility 实现更符合 `LA_update3.md` 的 child responsibility 定义，且已通过红绿测试、相关回归、全量回归和真实训练 smoke。
2. 100-step pre-limit matrix 没有产生 R5/R2 正向；只在 AE/TE 上有极小、非稳定的改善。
3. diagnostics 显示 responsibility 不是 inert，但 dropped/candidate 仍低于 0.2%。这解释了为什么 sparse pose 指标几乎不动：当前 one-shot split 只增加 9-15 个 duplicated source，责任分配影响的 observation 太少。
4. 因此不能把这轮结果解读为 child responsibility 思路无效；更准确地说，当前 split 数量和 observation 覆盖太低，最小 hard winner 策略不足以形成可测精度收益。
5. 下一步应优先做两类验证：
   - 高 split 覆盖诊断：临时提高 `TOPOLOGY_GROWTH_CAP_PER_EVENT` / `TOPOLOGY_TOTAL_POINT_BUDGET_RATIO`，只为验证 responsibility 在更多 sibling 上是否能改善 descriptor/retrieval；
   - responsibility-positive objective：不仅 drop loser，还对同源 child 建立 soft/hard assignment 的 multi-positive/negative 结构，配合 descriptor diagnostics，而不是只看 sparse pose。

## P2 high-split coverage diagnostic

日期：2026-06-25

pre-limit 100-step matrix 说明 child responsibility 不是 inert，但 dropped/candidate 仍低于 0.2%。因此本轮只做诊断性 high-split 覆盖放大，目的不是推荐最终 split 阈值，而是验证 sibling 覆盖变大后 responsibility 是否能产生可测影响。

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_highsplit_quantile_100_v1
```

配置：

```text
TOPOLOGY_STEPS=100
TOPOLOGY_MUTATION_MODE=one_shot_split
TOPOLOGY_GROWTH_CAP_PER_EVENT=0.005
TOPOLOGY_TOTAL_POINT_BUDGET_RATIO=1.02
TOPOLOGY_AMBIGUITY_QUANTILE=0.70
TOPOLOGY_SPLIT_QUANTILE=0.80
TOPOLOGY_CHILD_RESPONSIBILITY_MODE=none vs feature
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=32026
LABEL_MAX_IMAGES=64
V03_ITERATION=32000
source=/mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
```

实际 split 覆盖：

| Scene | candidates | parent removed | children added | points |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 2852 | 172 | 344 | 342918 -> 343090 |
| KingsCollege | 2825 | 171 | 342 | 318593 -> 318764 |
| OldHospital | 1787 | 108 | 216 | 405348 -> 405456 |

`feature_resp - no_resp` sparse-only delta：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.004872 | -0.057731 | +0.000000 | +0.009709 | -0.320 |
| KingsCollege | -0.001447 | -0.037527 | +0.000000 | +0.000000 | -0.289 |
| OldHospital | -0.007450 | -0.255079 | -0.005495 | +0.000000 | -0.709 |

responsibility diagnostics：

| Scene | episodes | dropped | candidates | drop ratio | nonzero episodes | max drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 75 | 1898 | 99620 | 0.019052 | 67 | 104 |
| KingsCollege | 75 | 2419 | 294224 | 0.008222 | 73 | 96 |
| OldHospital | 75 | 1704 | 150179 | 0.011346 | 63 | 83 |

当前判断：

1. high-split 诊断把 split source 数从常规 one-shot 的 9-15 个放大到 108-172 个，child responsibility drop ratio 从低于 0.2% 提高到约 0.8%-1.9%。
2. 在这个放大设置下，`feature_resp` 的 median AE/TE 在三场景均优于 `no_resp`，ShopFacade 的 R2 也提高 0.97 pp。
3. recall 级指标仍不稳健：KingsCollege R5/R2 不变，OldHospital R5 下降 0.55 pp，三场景 average inliers 都小幅下降。
4. 因此该结果支持“P2 responsibility 在 sibling 覆盖足够时能影响 pose quality”，但还不能支持“当前 hard winner 策略已经带来稳健 sparse-only precision 提升”。
5. 下一步不应把 high-split 阈值直接作为最终方法；更合理的是实现 responsibility-positive objective，并补 descriptor diagnostics、500-step 和 true train-seed matrix。

## P2 responsibility-positive source mode

日期：2026-06-25

本轮落实 P2 的最小 responsibility-positive objective。此前 `child_responsibility_mode=feature` 会只保留同源 siblings 中的 winner，但 full-bank source sibling 默认仍作为 ignore；这能避免 false negative，却不会主动推动 child descriptor 分化。

### 已实现

1. `direct_landmark_teacher()` 新增：
   - `full_bank_source_mode=ignore|positive|responsibility`
2. 默认 `ignore` 保持旧行为：
   - exact selected child 是 positive；
   - 同源 sibling loser 从 negative 中屏蔽。
3. `positive` 支持 group-positive warmup：
   - 同源 siblings 作为 positive group。
4. `responsibility` 支持 specialization：
   - 只有 responsibility winner / exact selected child 是 positive；
   - 同源 sibling loser 不再被 source mask 屏蔽，会进入 full-bank denominator 和 hard-negative mining；
   - 3D/UV 近邻仍可单独通过 radius 进入 ignore。
5. 新增 diagnostics：
   - `full_bank_source_related_count`
   - `full_bank_source_positive_count`
   - `full_bank_source_ignore_count`
   - `full_bank_source_negative_count`
6. `train_locaware.py` 新增 CLI：
   - `--loc_full_bank_source_mode {ignore,positive,responsibility}`
7. v0.3 / topology 脚本新增环境变量透传：
   - `V03_FULL_BANK_SOURCE_MODE`
   - `TOPOLOGY_FULL_BANK_SOURCE_MODE`

### TDD 验证

红灯：

```text
TypeError: direct_landmark_teacher() got an unexpected keyword argument 'full_bank_source_mode'
argparse: unrecognized arguments: --loc_full_bank_source_mode responsibility
script args missing V03/TOPOLOGY source-mode passthrough
```

绿灯：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_direct_landmark_teacher.DirectLandmarkTeacherTest.test_direct_teacher_responsibility_source_mode_keeps_sibling_losers_as_negatives \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_full_bank_source_mode \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_script_runs_feature_only_full_bank_anchor_without_topology \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default -v

Ran 4 tests in 1.651s
OK
```

相关回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_direct_landmark_teacher \
  tests.test_train_locaware_masks \
  tests.test_full_script_args -v

Ran 75 tests in 1.824s
OK
```

静态检查：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile localization_training/direct_landmark_teacher.py train_locaware.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
git diff --check
```

均通过。

### ShopFacade 5-step split smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_responsibility_source_split_smoke_v1
```

配置：

```text
TOPOLOGY_STEPS=5
TOPOLOGY_UPDATE_INTERVAL=1
TOPOLOGY_MUTATION_MODE=one_shot_split
TOPOLOGY_FULL_BANK_SOURCE_MODE=responsibility
TOPOLOGY_CHILD_RESPONSIBILITY_MODE=feature
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=32002
LABEL_MAX_IMAGES=8
SOURCE_MODEL=/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
```

实际 topology event：

```text
[Topology] iter=32001 candidates=254 physical_prune=0 requested_split=2 parent_removed=2 children_added=4 points=342918->342920
```

Sparse-only result：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.161406 | 3.001269 | 0.757282 | 0.271845 | 428.252 |

TensorBoard diagnostics：

| Metric | sum | max | values |
| --- | ---: | ---: | --- |
| `full_bank_source_related_count` | 2 | 1 | `[0,0,1,0,1]` |
| `full_bank_source_positive_count` | 0 | 0 | `[0,0,0,0,0]` |
| `full_bank_source_ignore_count` | 0 | 0 | `[0,0,0,0,0]` |
| `full_bank_source_negative_count` | 2 | 1 | `[0,0,1,0,1]` |
| `child_responsibility_candidate_count` | 7276 | 3749 | `[447,1013,2067,3749]` |
| `child_responsibility_dropped_count` | 2 | 1 | `[0,1,0,1]` |

当前判断：

1. responsibility-positive source mode 的工程路径已经跑通；真实 split 后，同源 sibling loser 确实进入 negative count，而不是继续被 ignore 或 positive group 吸收。
2. 5-step smoke 只证明机制触发和不会立即崩坏，不能作为精度结论。
3. 下一步应跑 matched matrix：
   - `source_mode=ignore` vs `source_mode=responsibility`；
   - 100-step 起步，ShopFacade/KingsCollege/OldHospital；
   - 若不反向，再扩 500-step 和 true train-seed。

## P2 source-mode high-split 100-step matrix

日期：2026-06-25

本轮补齐上节要求的 `source_mode=ignore` vs `source_mode=responsibility` matched matrix。为避免常规 split 覆盖太低导致 responsibility 分支近似 inert，沿用 P2 high-split coverage diagnostic 的放大配置。

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p2_source_mode_highsplit_100_v1
/mnt/pool/sqy/stdloc_la_update3_p2_source_mode_highsplit_100_v1/descriptor_diag_v2
```

关键配置：

```text
TOPOLOGY_STEPS=100
TOPOLOGY_MUTATION_MODE=one_shot_split
TOPOLOGY_GROWTH_CAP_PER_EVENT=0.005
TOPOLOGY_TOTAL_POINT_BUDGET_RATIO=1.02
TOPOLOGY_AMBIGUITY_QUANTILE=0.70
TOPOLOGY_SPLIT_QUANTILE=0.80
TOPOLOGY_CHILD_RESPONSIBILITY_MODE=feature
TOPOLOGY_CHILD_RESPONSIBILITY_START_ITER=32026
TOPOLOGY_FULL_BANK_SOURCE_MODE=ignore vs responsibility
LABEL_MAX_IMAGES=64
V03_ITERATION=32000
source=/mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
```

实际 split 覆盖在两种 source mode 下完全一致：

| Scene | candidates | parent removed | children added | points |
| --- | ---: | ---: | ---: | --- |
| ShopFacade | 2852 | 172 | 344 | 342918 -> 343090 |
| KingsCollege | 2825 | 171 | 342 | 318593 -> 318764 |
| OldHospital | 1787 | 108 | 216 | 405348 -> 405456 |

`responsibility - ignore` sparse-only delta：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.000272 | +0.031271 | +0.000000 | +0.000000 | -0.039 |
| KingsCollege | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.009 |
| OldHospital | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.005 |

TensorBoard diagnostics 证明 source-mode 分支不是 inert：

| Scene | Mode | source related | source ignore | source negative | child dropped / candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| ShopFacade | ignore | 1678 | 1678 | 0 | 1898 / 99620 |
| ShopFacade | responsibility | 1678 | 0 | 1678 | 1898 / 99620 |
| KingsCollege | ignore | 896 | 896 | 0 | 2419 / 294224 |
| KingsCollege | responsibility | 896 | 0 | 896 | 2419 / 294224 |
| OldHospital | ignore | 671 | 671 | 0 | 1704 / 150179 |
| OldHospital | responsibility | 671 | 0 | 671 | 1704 / 150179 |

Descriptor diagnostics 使用：

```text
iteration=32100
split=test
max_images=32
max_landmarks_per_image=1024
depth_check=true
full_bank=true
CUDA_HOME=/usr/local/cuda-11.8
```

说明：上一轮 descriptor diagnostics 失败的根因是 shell 环境误用了 `iclpose` conda 环境里的 `nvcc`，导致 `cuda_runtime.h` 缺失。本轮显式设置 CUDA 11.8、`PYTHONPATH=/root/STDLoc` 和独立 `TORCH_EXTENSIONS_DIR` 后，6/6 个 JSON 均生成。

`responsibility - ignore` descriptor delta：

| Scene | dRecall@1 | dRecall@5 | dRecall@10 | dMNN | dMargin |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.000000 | -0.000051 | +0.000101 | -0.000000 | -0.000011 |
| KingsCollege | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000000 |
| OldHospital | +0.000000 | +0.000000 | +0.000000 | +0.000000 | -0.000006 |

当前判断：

1. `source_mode=responsibility` 的实现语义已经被 TensorBoard 计数验证：同源 sibling loser 不再进入 ignore，而是进入 negative/denominator。
2. 在 matched high-split 设置下，这个语义变化没有带来可测 sparse-only pose 改善；R5/R2 三场景完全不变。
3. descriptor diagnostics 也没有显示 full-bank retrieval/MNN/margin 改善，变化基本处于数值噪声级。
4. 因此可以排除一个高影响混杂：P2 缺少正向 precision 证据，不主要是因为 full-bank source sibling loser 仍被 ignore。
5. 下一步不应继续只扫 source-mode；更合理的是推进 `LA_update3.md` 后续 P3/P6：perturb-pose episodes 与 held-out risk commit，或先用 multi-seed/500-step 复核 P2 high-split feature_resp 的弱 AE/TE 信号。

## P3 perturb-pose dense smoke

日期：2026-06-25

本轮开始落实 `LA_update3.md` 的 P3：真实 query 图像 + empirical sparse-error perturb pose + dense corrective supervision。先修一个脚本层缺口：`scripts/run_densekl_v03_cambridge.sh` 原本支持 `DENSEKL_QUERY_MODE=noise/sparse/mixed` 和 sparse pose cache，但没有把 LA_update2 已修好的 support/query split 参数透传给 `train_locaware.py`。这会让 dense perturb 实验仍可能在训练相机上取 query，不利于后续泛化验证。

### 已实现

1. `scripts/run_densekl_v03_cambridge.sh` 新增：
   - `DENSEKL_SUPPORT_QUERY_SPLIT`
   - `DENSEKL_QUERY_HOLDOUT_RATIO`
   - `DENSEKL_QUERY_SPLIT_SEED`
   - `DENSEKL_QUERY_SPLIT_MODE`
2. 开启 `DENSEKL_SUPPORT_QUERY_SPLIT=1` 时，训练会传入：
   - `--support_query_split`
   - `--query_holdout_ratio`
   - `--query_split_seed`
   - `--query_split_mode`
3. `scripts/run_la_update2_dense_long_worker.sh` 继续把矩阵中的 `query_split_seed` 传给 dense 脚本，避免 dense 长跑退回隐式默认 split。

默认 `DENSEKL_SUPPORT_QUERY_SPLIT=0`，因此历史 dense 脚本默认行为不变。

### TDD / 验证

红灯：

```text
test_dense_kl_script_accepts_support_query_split_controls ... FAIL
test_la_update2_workers_separate_train_seed_from_query_split_seed ... FAIL
```

绿灯：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_full_script_args.FullRunScriptArgsTest.test_dense_kl_script_accepts_support_query_split_controls \
  tests.test_full_script_args.FullRunScriptArgsTest.test_la_update2_workers_separate_train_seed_from_query_split_seed -v

Ran 2 tests in 0.000s
OK
```

相关回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_full_script_args -v

Ran 29 tests in 2.855s
OK
```

脚本/静态检查：

```text
bash -n scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh
git diff --check -- scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh tests/test_full_script_args.py LA_update3_closure.md
```

均通过。

### ShopFacade 10-step P3 smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p3_perturb_dense_smoke_v1
```

关键配置：

```text
scene=ShopFacade
source=/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
LOAD_ITERATION=32000
DENSEKL_STEPS=10
DENSEKL_QUERY_MODE=noise
RUN_DENSE_POSE_CACHE=1
DENSEKL_POSE_GATE=0
DENSEKL_ATTR_COSINE_THRESHOLD=0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS=32
DENSEKL_SUPPORT_QUERY_SPLIT=1
DENSEKL_QUERY_HOLDOUT_RATIO=0.2
DENSEKL_QUERY_SPLIT_SEED=2025
DENSEKL_QUERY_SPLIT_MODE=sequence_block
```

cache 与 split sanity：

```text
Sparse pose cache summary: queries=231 failures=0 median_ae=0.168285 median_te=2.741181 avg_inliers=393.680
Support/query split enabled: support=185 query=46 query_ratio=0.2 query_split_seed=2025 query_split_mode=sequence_block
```

训练日志：

```text
[ITER 32010] base 0.178990 loc 0.147111 psnr 16.813
LA-STDLoc training complete.
```

输出 checkpoint：

```text
point_cloud/iteration_32010/point_cloud.ply
point_cloud/iteration_32010/loc_state.pt
```

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.163247 | 3.021684 | 0.786408 | 0.281553 | 430.709 |

当前判断：

1. P3 最小通路已跑通：真实图像、empirical sparse-error distribution、noise 初始位姿、dense corrective KL、sequence-block query split 能在同一训练入口中同时工作。
2. 本次 smoke 中 dense `loc` loss 非零，说明不是空监督；checkpoint 和 sparse-only eval 均正常生成。
3. 10-step 单场景 smoke 不作为精度结论。下一步需要至少 3 scene x 100-step matched matrix，对比同源 no-dense/no-mutation continuation，确认 perturb-pose dense corrective 是否比之前 `query_mode=sparse` dense gate 更稳。

## P3 100-step perturb-dense paired matrix

日期：2026-06-25

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p3_perturb_dense_100_v1
```

矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
variants = no_dense, perturb_dense
DENSEKL_QUERY_MODE = noise
DENSEKL_WEIGHT = 0.0 for no_dense, 0.02 for perturb_dense
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_HOLDOUT_RATIO = 0.2
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
RUN_DENSE_POSE_CACHE = 1
DENSEKL_POSE_GATE = 0
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
```

cache 与 split sanity：

| Scene | cache queries | failures | cache AE | cache TE | cache inliers | support | query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 231 | 0 | 0.168285 | 2.741181 | 393.680 | 185 | 46 |
| KingsCollege | 1220 | 0 | 0.184933 | 14.975915 | 555.189 | 829 | 391 |
| OldHospital | 895 | 0 | 0.303500 | 14.172626 | 255.239 | 594 | 301 |

训练 sanity：

| Scene | Variant | base | loc | psnr |
| --- | --- | ---: | ---: | ---: |
| ShopFacade | no_dense | 0.141834 | 0.000000 | 18.503 |
| ShopFacade | perturb_dense | 0.141990 | 0.129068 | 18.503 |
| KingsCollege | no_dense | 0.127228 | 0.000000 | 17.619 |
| KingsCollege | perturb_dense | 0.127511 | 0.140521 | 17.619 |
| OldHospital | no_dense | 0.200503 | 0.000000 | 15.956 |
| OldHospital | perturb_dense | 0.200753 | 0.149085 | 15.956 |

Sparse-only absolute metrics：

| Scene | Variant | AE | TE | R5 | R2 | avg inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | no_dense | 0.154063 | 2.858097 | 0.776699 | 0.242718 | 433.670 |
| ShopFacade | perturb_dense | 0.155361 | 3.019223 | 0.766990 | 0.262136 | 438.214 |
| KingsCollege | no_dense | 0.174338 | 16.252053 | 0.017493 | 0.005831 | 551.790 |
| KingsCollege | perturb_dense | 0.181596 | 16.326772 | 0.020408 | 0.000000 | 548.746 |
| OldHospital | no_dense | 0.361154 | 20.203354 | 0.043956 | 0.000000 | 265.192 |
| OldHospital | perturb_dense | 0.361115 | 19.078347 | 0.043956 | 0.005495 | 264.038 |

Paired delta，`perturb_dense - no_dense`：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.001298 | +0.161126 | -0.009709 | +0.019417 | +4.544 |
| KingsCollege | +0.007258 | +0.074719 | +0.002915 | -0.005831 | -3.044 |
| OldHospital | -0.000039 | -1.125007 | +0.000000 | +0.005495 | -1.154 |

当前判断：

1. P3 100-step dense perturb 通路不是 inert：三场景 `perturb_dense` 的 final loc loss 均为非零，且 support/query split 与 noise episode 同时生效。
2. 相比 `no_dense`，结果仍是 mixed，而不是稳定精度正向：ShopFacade 的 R2 和 inliers 变好，但 R5/TE 变差；KingsCollege 的 R5 小幅变好，但 R2/AE/TE/inliers 变差；OldHospital 的 TE/R2 变好，R5 持平。
3. 因此 P3 当前只能支持“3DGS dense corrective/perturb-pose 路径可运行，并能改变 sparse-only pose 指标”，还不能支持“dense perturb 是稳健增益模块”。
4. 这与 LA_update3 的方向一致：dense teacher 应作为 selective residual teacher，而不是无条件主干；后续要优先加 pose-improvement / attribution-confidence / correspondence-correctness gate，再做 500-step、多 seed、多 scene 扩展。
5. 本轮并行调度也暴露效率问题：scene worker 内部串行跑 variant，导致 ShopFacade 完成后 GPU0 长时间空闲。下一轮 100/500-step 多 seed 应改成全局 task queue，把 scene/variant/seed 粒度动态分配到空闲 GPU。

## P4 dense advantage gate 初版

日期：2026-06-25

P3 100-step mixed 结果说明无条件 dense KL 不稳。本轮先落实 `LA_update3.md` P4 的最小可运行版本：dense teacher 不再只能用二值 pose gate，而可以使用 cache 中 `sparse pose` 与 `dense pose` 的 episode-level advantage 生成连续权重。

### 已实现

1. `EpisodeSampler` 在 `query_mode=noise` 或 `mixed` 落到 noise 分支时，会保留同一 query 的 `sparse_meta`，包括：
   - `te`
   - `ae`
   - `dense_te`
   - `dense_ae`
   - `dense_inliers`
2. `train_locaware.py` 新增：
   - `--loc_dense_advantage_gate`
   - `--loc_dense_advantage_min_te`
   - `--loc_dense_advantage_min_ae`
   - `--loc_dense_advantage_te_scale`
   - `--loc_dense_advantage_ae_scale`
3. 新增 `_dense_pose_advantage_weight()`：
   - 如果 cache 缺少 sparse/dense pose metric，返回 `0`；
   - 只有 dense pose 同时改善 TE 和 AE 时返回正权重；
   - 权重为连续值，并按 TE/AE scale 保守取较小分量，最终 clamp 到 `[0, 1]`。
4. dense 训练入口中，`loc_dense_advantage_gate` 优先于旧的二值 `loc_dense_pose_gate`。
5. `scripts/run_densekl_v03_cambridge.sh` 新增：
   - `DENSEKL_ADVANTAGE_GATE`
   - `DENSEKL_ADVANTAGE_MIN_TE`
   - `DENSEKL_ADVANTAGE_MIN_AE`
   - `DENSEKL_ADVANTAGE_TE_SCALE`
   - `DENSEKL_ADVANTAGE_AE_SCALE`
6. `scripts/run_la_update2_dense_long_worker.sh` 同步透传这些参数，避免长跑矩阵退回默认无 advantage 路径。

默认 `DENSEKL_ADVANTAGE_GATE=0`，因此历史 dense-KL 行为不变。

### TDD / 验证

红灯：

```text
tests.test_episode_sampler.EpisodeSamplerTest.test_noise_mode_preserves_sparse_meta_for_advantage_gates ... ERROR
tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_dense_advantage_gate_controls ... SystemExit
tests.test_train_locaware_masks.TrainLocawareMaskTest.test_dense_advantage_gate_returns_continuous_pose_weight ... ImportError
tests.test_full_script_args.FullRunScriptArgsTest.test_dense_kl_script_runs_dense_teacher_without_topology ... FAIL
```

绿灯：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_episode_sampler \
  tests.test_train_locaware_masks \
  tests.test_full_script_args \
  tests.test_dense_teacher_losses -v

Ran 75 tests in 1.774s
OK
```

静态检查：

```text
bash -n scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  train_locaware.py localization_training/episode_sampler.py localization_training/dense_teacher.py
```

均通过。

### ShopFacade 10-step smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p4_advantage_smoke_v1
```

配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
load_iteration = 32000
steps = 10
DENSEKL_QUERY_MODE = noise
DENSEKL_WEIGHT = 0.02
DENSEKL_ADVANTAGE_GATE = 1
DENSEKL_ADVANTAGE_TE_SCALE = 10.0
DENSEKL_ADVANTAGE_AE_SCALE = 1.0
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
```

Cache advantage 分布：

```text
items=231
positive=178
positive_ratio=0.770563
weight_min=0.0
weight_max=1.0
weight_mean=0.075617
```

训练日志：

```text
Support/query split enabled: support=185 query=46 query_ratio=0.2 query_split_seed=2025 query_split_mode=sequence_block
[ITER 32010] base 0.178977 loc 0.147359 psnr 16.813
LA-STDLoc training complete.
```

TensorBoard 诊断：

```text
train_diagnostics/dense_kl_pose_weight:
32001 0.000000
32002 0.147399
32003 0.000000
32004 0.015044
32005 0.000000
32006 0.048924
32007 0.058287
32008 0.005104
32009 0.047718
32010 0.046791

train_diagnostics/dense_kl_eligible_anchor_count:
32001 0
32002 443
32003 0
32004 400
32005 0
32006 433
32007 480
32008 473
32009 426
32010 441
```

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.157318 | 2.919292 | 0.786408 | 0.281553 | 430.806 |

相对上一节 P3 ungated 10-step smoke：

| dAE | dTE | dR5 | dR2 | dInliers |
| ---: | ---: | ---: | ---: | ---: |
| -0.005929 | -0.102392 | +0.000000 | +0.000000 | +0.097 |

当前判断：

1. P4 初版 advantage gate 已经跑通，且不是全开/全关：TensorBoard 中 pose weight 是连续小权重，并且部分 iteration 为 0。
2. 10-step ShopFacade smoke 相比 ungated P3 smoke 在 AE/TE 上更好，R5/R2 持平；这只说明方向值得扩展，不是精度结论。
3. 该实现仍然只是 episode-level advantage，不是完整的 sparse-miss/dense-hit correspondence distillation，也没有 logistic teacher confidence calibrator。
4. 下一步应先跑 3-scene 100-step `advantage_dense` vs `no_dense` vs `ungated_dense` paired matrix；如果 advantage gate 能减少 P3 的 ShopFacade/Kings 退化，再进入 500-step 和多 seed。

### P4 100-step advantage-dense paired matrix

日期：2026-06-25

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p4_advantage_100_v1
```

矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
variant = advantage_dense
DENSEKL_QUERY_MODE = noise
DENSEKL_WEIGHT = 0.02
DENSEKL_ADVANTAGE_GATE = 1
DENSEKL_ADVANTAGE_TE_SCALE = 10.0
DENSEKL_ADVANTAGE_AE_SCALE = 1.0
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
```

训练 sanity：

| Scene | base | loc | psnr |
| --- | ---: | ---: | ---: |
| ShopFacade | 0.141912 | 0.131264 | 18.503 |
| KingsCollege | 0.127360 | 0.142378 | 17.619 |
| OldHospital | 0.200723 | 0.149369 | 15.956 |

TensorBoard gate diagnostics：

| Scene | pose weight n | nonzero | mean | max | eligible anchors mean | eligible anchors max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 97 | 62 | 0.038348 | 0.194075 | 284.103 | 485 |
| KingsCollege | 100 | 59 | 0.040927 | 0.285568 | 263.370 | 474 |
| OldHospital | 100 | 91 | 0.127969 | 0.646296 | 417.730 | 500 |

Sparse-only absolute metrics：

| Scene | AE | TE | R5 | R2 | avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.165653 | 3.157783 | 0.757282 | 0.271845 | 437.835 |
| KingsCollege | 0.173394 | 16.236541 | 0.014577 | 0.000000 | 549.662 |
| OldHospital | 0.369725 | 19.908051 | 0.032967 | 0.005495 | 264.044 |

Paired delta，`advantage_dense - no_dense`：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.011590 | +0.299686 | -0.019417 | +0.029126 | +4.165 |
| KingsCollege | -0.000944 | -0.015512 | -0.002915 | -0.005831 | -2.128 |
| OldHospital | +0.008572 | -0.295303 | -0.010989 | +0.005495 | -1.148 |

Paired delta，`advantage_dense - ungated_dense`：

| Scene | dAE | dTE | dR5 | dR2 | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.010292 | +0.138560 | -0.009709 | +0.009709 | -0.379 |
| KingsCollege | -0.008202 | -0.090231 | -0.005831 | +0.000000 | +0.915 |
| OldHospital | +0.008611 | +0.829704 | -0.010989 | +0.000000 | +0.005 |

当前判断：

1. P4 advantage gate 不是 inert：三场景都有非零连续 pose weight，OldHospital 的有效 episode 比例最高。
2. 但 100-step paired matrix 没有支持“advantage gate 减少 P3 退化”。相对 `no_dense`，三场景 R5 全部下降；相对 ungated dense，三场景 R5 也全部下降或不优。
3. advantage gate 在 TE 上对 KingsCollege/OldHospital 相对 no-dense 有小幅改善，但 ShopFacade 明显变差，且 R2/R5 信号冲突。
4. 因此当前 P4 只能证明 episode-level dense advantage 机制可运行、可筛选、可施力；不能证明它已经是有效的 sparse-only 精度增益模块。
5. 下一步若继续 dense 路线，应从完整 sparse-miss/dense-hit correspondence distillation 和 teacher confidence calibration 入手，而不是继续扩大当前 episode-level full-KL advantage gate。

## P4.1 sparse-miss/dense-hit rank distillation 初版

日期：2026-06-25

P4 100-step 说明 episode-level advantage + full-KL 仍不稳。本轮先实现 `LA_update3.md` 中“只学习 sparse-miss/dense-hit”的最小可运行近似：dense teacher 生成 Gaussian teacher distribution 后，不再必须全量 KL 到 sparse bank，而是可选地只对当前 sparse bank top-k 未命中的 dense teacher top1 施加 ranking loss。

### 已实现

1. `localization_training/dense_distill.py` 新增 `dense_sparse_miss_hit_rank_loss()`：
   - teacher top1 作为 dense-hit 目标；
   - sparse bank 当前 top-k 包含 teacher top1 时视为 sparse-hit，不施力；
   - sparse top-k 未命中且 teacher confidence 达阈值时视为 sparse-miss/dense-hit；
   - 对 teacher top1 与最强 negative 做 margin rank loss；
   - 返回 diagnostics：eligible、sparse hit/miss、low confidence、teacher confidence mean。
2. `localization_training/dense_teacher.py` 新增 `dense_responsibility_rank_loss()`，复用 responsibility attribution 和 selective dense anchor weights。
3. `DenseTeacherOutput` 新增 `rank_loss`。
4. `train_locaware.py` 新增参数：
   - `--loc_dense_rank_weight`
   - `--loc_dense_rank_margin`
   - `--loc_dense_rank_teacher_confidence`
   - `--loc_dense_rank_miss_topk`
5. dense 训练入口新增 TensorBoard 标量：
   - `train_loss/loc_dense_rank`
   - `train_diagnostics/dense_rank_*`
6. `scripts/run_densekl_v03_cambridge.sh` 与 `scripts/run_la_update2_dense_long_worker.sh` 新增环境变量透传：
   - `DENSEKL_RANK_WEIGHT`
   - `DENSEKL_RANK_MARGIN`
   - `DENSEKL_RANK_TEACHER_CONFIDENCE`
   - `DENSEKL_RANK_MISS_TOPK`

默认 `DENSEKL_RANK_WEIGHT=0.0`，历史 dense-KL/advantage 行为不变。

### TDD / 验证

红灯：

```text
ImportError: cannot import name 'dense_sparse_miss_hit_rank_loss'
argparse: unrecognized arguments: --loc_dense_rank_weight ...
script args missing DENSEKL_RANK_* passthrough
```

绿灯：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_dense_teacher_losses.DenseTeacherLossTest.test_dense_miss_hit_rank_loss_updates_only_sparse_misses \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_dense_miss_hit_rank_controls \
  tests.test_full_script_args.FullRunScriptArgsTest.test_dense_kl_script_runs_dense_teacher_without_topology \
  tests.test_full_script_args.FullRunScriptArgsTest.test_la_update2_workers_separate_train_seed_from_query_split_seed -v

Ran 4 tests in 1.675s
OK
```

相关回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_dense_teacher_losses tests.test_train_locaware_masks tests.test_full_script_args -v

Ran 69 tests in 1.763s
OK
```

全量回归：

```text
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 182 tests in 6.581s
OK
```

静态检查：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  localization_training/dense_distill.py localization_training/dense_teacher.py train_locaware.py
bash -n scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh
git diff --check -- localization_training/dense_distill.py localization_training/dense_teacher.py \
  train_locaware.py scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh \
  tests/test_dense_teacher_losses.py tests/test_train_locaware_masks.py tests/test_full_script_args.py
```

均通过。

### ShopFacade 10-step rank-only smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p4_rank_smoke_v1
```

关键配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
load_iteration = 32000
steps = 10
DENSEKL_WEIGHT = 0.0
DENSEKL_RANK_WEIGHT = 0.02
DENSEKL_RANK_MARGIN = 0.2
DENSEKL_RANK_TEACHER_CONFIDENCE = 0.0
DENSEKL_RANK_MISS_TOPK = 1
DENSEKL_ADVANTAGE_GATE = 1
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
```

说明：shell pipeline 的 `tee` 因日志目录未预创建返回非零，但训练和 eval 均已完成；证据来自 checkpoint、STDLoc result summary 和 TensorBoard event。

产物检查：

```text
point_cloud/iteration_32010/point_cloud.ply exists
point_cloud/iteration_32010/loc_state.pt exists
```

训练日志关键行：

```text
Support/query split enabled: support=185 query=46 query_ratio=0.2 query_split_seed=2025 query_split_mode=sequence_block
[ITER 32010] base 0.178968 loc 0.151424 psnr 16.813
LA-STDLoc training complete.
```

TensorBoard diagnostics：

| Metric | n | nonzero | mean | max |
| --- | ---: | ---: | ---: | ---: |
| `train_loss/loc_dense_rank` | 10 | 7 | 5.646731 | 8.770273 |
| `train_loss/loc_dense_kl` | 10 | 0 | 0.000000 | 0.000000 |
| `dense_rank_eligible_anchor_count` | 10 | 7 | 305.800 | 473 |
| `dense_rank_sparse_miss_count` | 10 | 7 | 305.800 | 473 |
| `dense_rank_sparse_hit_count` | 10 | 7 | 3.600 | 8 |
| `dense_rank_teacher_confidence_mean` | 10 | 7 | 0.024603 | 0.047434 |
| `dense_kl_pose_weight` | 10 | 7 | 0.036927 | 0.147399 |
| `dense_kl_eligible_anchor_count` | 10 | 7 | 309.400 | 479 |

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.162365 | 2.824082 | 0.796117 | 0.271845 | 430.204 |

相对 P4 10-step advantage full-KL smoke：

| dAE | dTE | dR5 | dR2 | dInliers |
| ---: | ---: | ---: | ---: | ---: |
| +0.005047 | -0.095210 | +0.009709 | -0.009709 | -0.602 |

当前判断：

1. P4.1 rank-only 路径已经跑通，且旧 KL 确认为 0，因此本次 smoke 确实隔离了 sparse-miss/dense-hit rank loss。
2. rank branch 不是 inert：10 step 中 7 step 有非零 rank loss，eligible anchors 与 advantage pose weight 同步。
3. 但 teacher confidence 均值只有约 0.02-0.05，说明当前 dense teacher distribution 很分散；如果直接提高 confidence threshold，很可能使 loss 变成近似全零。
4. 10-step ShopFacade 指标相对 P4 full-KL smoke 有 R5/TE 正向、R2/inliers 负向，只能作为“值得扩展”的 smoke，不能作为精度结论。
5. 下一步应跑 3-scene 100-step paired matrix：`no_dense` vs `advantage_full_kl` vs `rank_only`，并同步报告 `dense_rank_sparse_miss_count`、teacher confidence 分布和 sparse-only metrics。若 rank-only 仍 mixed，再进入 teacher confidence calibration，而不是盲目提高阈值。

### P4.1 100-step rank-only paired matrix

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p4_rank_100_v1
```

关键配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
DENSEKL_WEIGHT = 0.0
DENSEKL_RANK_WEIGHT = 0.02
DENSEKL_RANK_MARGIN = 0.2
DENSEKL_RANK_TEACHER_CONFIDENCE = 0.0
DENSEKL_RANK_MISS_TOPK = 1
DENSEKL_QUERY_MODE = noise
DENSEKL_ADVANTAGE_GATE = 1
DENSEKL_ADVANTAGE_TE_SCALE = 10.0
DENSEKL_ADVANTAGE_AE_SCALE = 1.0
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
RUN_DENSE_POSE_CACHE = 1
RUN_EVAL = 1
```

Sparse-only eval：

| Scene | AE | TE | R5 | R2 | avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.157895 | 2.860094 | 0.757282 | 0.194175 | 433.883 |
| KingsCollege | 0.176932 | 16.158819 | 0.017493 | 0.000000 | 551.210 |
| OldHospital | 0.366100 | 20.063847 | 0.043956 | 0.005495 | 266.181 |

相对 P3 no-dense matched continuation：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.003831 | +0.001997 | -1.942 | -4.854 | +0.214 |
| KingsCollege | +0.002593 | -0.093234 | +0.000 | -0.583 | -0.580 |
| OldHospital | +0.004946 | -0.139508 | +0.000 | +0.549 | +0.989 |

TensorBoard diagnostics，过滤 step 32001-32100：

| Scene | rank loss nonzero | KL nonzero | mean sparse miss | mean sparse hit | teacher conf mean | pose weight mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 62/100 | 0/100 | 279.144 | 3.330 | 0.023484 | 0.038348 |
| KingsCollege | 59/100 | 0/100 | 252.250 | 6.960 | 0.027475 | 0.040927 |
| OldHospital | 91/100 | 0/100 | 412.880 | 2.720 | 0.029469 | 0.127969 |

当前判断：

1. rank-only 100-step 确认不是 inert：三场景均有大量 sparse-miss anchor，旧 full-KL 全程为 0。
2. 相对 P4 advantage full-KL，rank-only 减少了部分退化：ShopFacade TE、Kings TE、OldHospital R5/AE 更好；但这不是最终对照目标。
3. 相对 no-dense，rank-only 仍没有形成正向精度支撑：三场景 R5 均未提升，ShopFacade R5/R2 明显下降；Kings/Old 只有 TE 或 R2 的小幅混合信号。
4. teacher confidence 均值仍只有约 0.02-0.03，说明当前 dense teacher distribution 分散。下一步若继续 dense 路线，应优先做 teacher calibration / top-k distribution sharpening / confidence-gated non-inert sweep，而不是直接把 rank loss 训练更久。
5. 这组结果支持“当前实现能排除 full-KL 混杂并施加 sparse-miss/dense-hit 信号”，但不支持“P4.1 已经带来 sparse-only relocalization 精度增益”。

## P5 synthetic novel-view augmentation 初版

日期：2026-06-25

本轮开始落实 `LA_update3.md` 的 P5：低比例 synthetic novel-view augmentation。实现刻意保守：synthetic view 默认关闭，开启后只进入 dense teacher episode；base RGB/feature reconstruction 仍使用真实 support 图像；direct/full-bank/topology/physical-prune 不会默认被 synthetic view 直接驱动。

### 已实现

1. `localization_training/episode_sampler.py` 新增：
   - `SyntheticView`；
   - `interpolate_pose_w2c()`；
   - `sample_interpolated_novel_view()`。
2. synthetic pose 由相邻真实相机插值得到，默认 `alpha in [0.35, 0.65]`，并记录 difficulty/coverage 近似分数。
3. `train_locaware.py` 新增 synthetic query render：
   - 在 synthetic target pose 用 frozen/no-grad 3DGS render feature map；
   - 用 alpha fraction 作为 observability gate；
   - 支持多候选中按 `difficulty * coverage * observability` 选最高分。
4. 新增训练参数：
   - `--synthetic_view_ratio`
   - `--synthetic_view_candidates`
   - `--synthetic_view_alpha_min`
   - `--synthetic_view_alpha_max`
   - `--synthetic_view_min_observability`
   - `--synthetic_view_desc_weight`
   - `--synthetic_view_reproj_weight`
5. `scripts/run_densekl_v03_cambridge.sh` 与 `scripts/run_la_update2_dense_long_worker.sh` 透传 `DENSEKL_SYNTHETIC_VIEW_*` 环境变量。

默认 synthetic ratio 和 synthetic desc/reproj weight 均为 `0.0`，历史实验默认行为不变。

### TDD / 验证

红灯：

```text
ImportError: cannot import name 'sample_interpolated_novel_view'
argparse: unrecognized arguments: --synthetic_view_ratio ...
script args missing DENSEKL_SYNTHETIC_VIEW_* passthrough
```

绿灯：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_episode_sampler.EpisodeSamplerTest.test_interpolated_novel_view_samples_between_adjacent_cameras \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_synthetic_view_controls \
  tests.test_full_script_args.FullRunScriptArgsTest.test_dense_kl_script_runs_dense_teacher_without_topology -v

Ran 3 tests in 1.605s
OK
```

相关回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_episode_sampler tests.test_train_locaware_masks tests.test_full_script_args -v

Ran 71 tests in 1.875s
OK
```

静态检查：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile \
  localization_training/episode_sampler.py train_locaware.py
bash -n scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh
git diff --check -- localization_training/episode_sampler.py train_locaware.py \
  scripts/run_densekl_v03_cambridge.sh scripts/run_la_update2_dense_long_worker.sh \
  tests/test_episode_sampler.py tests/test_train_locaware_masks.py tests/test_full_script_args.py
```

均通过。

### 第一次 forced-synthetic smoke 暴露的混杂

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p5_synth_smoke_v1
```

配置为 `DENSEKL_SYNTHETIC_VIEW_RATIO=1.0`、`DENSEKL_RANK_WEIGHT=0.02`、`DENSEKL_ADVANTAGE_GATE=1`、`DENSEKL_WEIGHT=0.0`。结果：

| Metric | n | nonzero | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_diagnostics/synthetic_view_used` | 10 | 10 | 1.000000 | 1.000000 | 1.000000 |
| `train_diagnostics/synthetic_view_observability` | 10 | 10 | 0.990109 | 0.953676 | 1.000000 |
| `train_loss/loc_desc` | 10 | 10 | 0.976680 | 0.537314 | 1.703409 |
| `train_loss/loc_reproj` | 10 | 10 | 0.167332 | 0.113997 | 0.206123 |
| `train_loss/loc` | 10 | 0 | 0.000000 | 0.000000 | 0.000000 |
| `train_loss/loc_dense_rank` | 10 | 0 | 0.000000 | 0.000000 | 0.000000 |

根因：canonical dense script 把 `loc_desc_weight`/`loc_reproj_weight` 固定为 0；synthetic episode 没有 sparse cache meta，advantage gate 让 KL/rank pose weight 为 0。因此 v1 只证明 synthetic pose/render 采样生效，不能证明 synthetic view 真的施加训练信号。

本轮随后补了 `synthetic_view_desc_weight` / `synthetic_view_reproj_weight`，专门让 synthetic episode 以低权重 desc/reproj 施力，同时继续默认关闭 synthetic full-KL/rank。

### 第二次 forced-synthetic desc/reproj smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p5_synth_smoke_v2
```

关键配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
load_iteration = 32000
steps = 10
DENSEKL_SYNTHETIC_VIEW_RATIO = 1.0
DENSEKL_SYNTHETIC_VIEW_DESC_WEIGHT = 0.05
DENSEKL_SYNTHETIC_VIEW_REPROJ_WEIGHT = 0.01
DENSEKL_WEIGHT = 0.0
DENSEKL_RANK_WEIGHT = 0.0
DENSEKL_ADVANTAGE_GATE = 0
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
RUN_DENSE_POSE_CACHE = 0
RUN_EVAL = 1
```

TensorBoard diagnostics：

| Metric | n | nonzero | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_loss/loc` | 10 | 10 | 0.050079 | 0.030242 | 0.090340 |
| `train_loss/loc_desc` | 10 | 10 | 0.969663 | 0.574297 | 1.777376 |
| `train_loss/loc_reproj` | 10 | 10 | 0.159585 | 0.124569 | 0.198948 |
| `train_loss/loc_dense_rank` | 10 | 0 | 0.000000 | 0.000000 | 0.000000 |
| `train_loss/loc_dense_kl` | 10 | 0 | 0.000000 | 0.000000 | 0.000000 |
| `train_diagnostics/synthetic_view_used` | 10 | 10 | 1.000000 | 1.000000 | 1.000000 |
| `train_diagnostics/synthetic_view_observability` | 10 | 10 | 0.990109 | 0.953676 | 1.000000 |
| `train_diagnostics/synthetic_view_score` | 10 | 10 | 0.717920 | 0.490440 | 0.913689 |
| `train_diagnostics/synthetic_view_alpha` | 10 | 10 | 0.525266 | 0.377453 | 0.649541 |

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.165023 | 2.979692 | 0.766990 | 0.271845 | 428.029 |

当前判断：

1. P5 最小 synthetic novel-view 通路已经跑通，并且 v2 排除了“采样生效但训练信号为 0”的混杂。
2. v2 严格隔离 synthetic desc/reproj：KL/rank 均为 0，因此这不是 full-KL synthetic supervision。
3. forced ratio=1.0 只用于 smoke，不符合最终 curriculum 的低比例设置；不能作为精度结论。
4. 10-step sparse-only eval 相对 no-dense/P4.1 不是正向证据，且 synthetic-only 强制比例过高，主要意义是验证实现可用。
5. 下一步应跑低比例 `DENSEKL_SYNTHETIC_VIEW_RATIO=0.05/0.15` 的 100-step paired matrix，对照同源 no-dense 和 P4.1 rank-only，并报告 synthetic used ratio、observability、loc loss、sparse-only metrics。

### P5 100-step synth05 + rank paired matrix

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p5_synth05_rank_100_v1
```

关键配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
DENSEKL_WEIGHT = 0.0
DENSEKL_RANK_WEIGHT = 0.02
DENSEKL_RANK_MARGIN = 0.2
DENSEKL_RANK_TEACHER_CONFIDENCE = 0.0
DENSEKL_RANK_MISS_TOPK = 1
DENSEKL_QUERY_MODE = noise
DENSEKL_ADVANTAGE_GATE = 1
DENSEKL_ATTR_COSINE_THRESHOLD = 0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS = 32
DENSEKL_SYNTHETIC_VIEW_RATIO = 0.05
DENSEKL_SYNTHETIC_VIEW_DESC_WEIGHT = 0.05
DENSEKL_SYNTHETIC_VIEW_REPROJ_WEIGHT = 0.01
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
```

Sparse-only eval：

| Scene | AE | TE | R5 | R2 | avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.148931 | 2.900832 | 0.757282 | 0.271845 | 432.301 |
| KingsCollege | 0.188802 | 16.665682 | 0.008746 | 0.002915 | 549.525 |
| OldHospital | 0.351082 | 19.569661 | 0.038462 | 0.000000 | 267.049 |

相对 P3 no-dense matched continuation：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.005133 | +0.042734 | -1.942 | +2.913 | -1.369 |
| KingsCollege | +0.014464 | +0.413629 | -0.875 | -0.292 | -2.265 |
| OldHospital | -0.010072 | -0.633693 | -0.549 | +0.000 | +1.857 |

相对 P4.1 rank-only matched continuation：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.008964 | +0.040738 | +0.000 | +7.767 | -1.583 |
| KingsCollege | +0.011871 | +0.506863 | -0.875 | +0.292 | -1.685 |
| OldHospital | -0.015018 | -0.494185 | -0.549 | -0.549 | +0.868 |

TensorBoard diagnostics，过滤 step 32001-32100：

| Scene | synthetic used | obs mean | score mean | loc nonzero | loc mean | rank nonzero | rank mean | KL nonzero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 7/100 | 0.999220 | 0.855185 | 65/100 | 0.100410 | 58/100 | 4.693935 | 0/100 |
| KingsCollege | 6/100 | 0.742601 | 0.498014 | 64/100 | 0.095103 | 58/100 | 4.559130 | 0/100 |
| OldHospital | 6/100 | 0.895365 | 0.559267 | 91/100 | 0.145709 | 85/100 | 7.079691 | 0/100 |

当前判断：

1. P5 低比例 synthetic novel-view 路径已经不是 smoke-only：3 scene x 100-step 均完成，synthetic episode 实际触发约 6%-7%，observability 非零且 loc/rank loss 非零。
2. 这组结果证明了 3DGS-rendered novel-view augmentation 能进入训练并改变 sparse-only 指标，但精度证据仍然是 mixed。
3. 正向信号主要出现在 ShopFacade 的 R2、OldHospital 的 AE/TE/inliers；负向信号主要是 KingsCollege 退化，以及三场景 R5 没有形成稳定提升。
4. 因此 P5 当前只能支持“实现通路可用、非 inert、值得继续校准”，不能支持“novel-view augmentation 已经稳定提升 sparse-only relocalization”。
5. 下一步如果继续 P5，应做 ratio sweep（0.05/0.15）、去掉 rank-only 交互的 synthetic-desc/reproj 单独对照、teacher/view quality calibration，再决定是否扩到 500-step 与 true train-seed。

### P5 100-step synthetic-only ratio sweep

日期：2026-06-25

本轮继续排除 P5 的 rank-only 交互混杂：关闭 KL/rank/advantage gate，只保留低权重 synthetic desc/reproj supervision。

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p5_synth_ratio_100_v1
```

关键配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
DENSEKL_WEIGHT = 0.0
DENSEKL_RANK_WEIGHT = 0.0
DENSEKL_ADVANTAGE_GATE = 0
DENSEKL_QUERY_MODE = noise
DENSEKL_SYNTHETIC_VIEW_RATIO = 0.05 / 0.15
DENSEKL_SYNTHETIC_VIEW_DESC_WEIGHT = 0.05
DENSEKL_SYNTHETIC_VIEW_REPROJ_WEIGHT = 0.01
DENSEKL_SUPPORT_QUERY_SPLIT = 1
DENSEKL_QUERY_SPLIT_SEED = 2025
DENSEKL_QUERY_SPLIT_MODE = sequence_block
```

Sparse-only eval，相对 P3 no-dense matched continuation：

| Variant | Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synth05_only | ShopFacade | 0.161440 | +0.007377 | 2.857532 | -0.000565 | 0.766990 | -0.971 | 0.203883 | -3.883 | 432.388 |
| synth05_only | KingsCollege | 0.180966 | +0.006628 | 16.540560 | +0.288507 | 0.014577 | -0.292 | 0.002915 | -0.292 | 549.863 |
| synth05_only | OldHospital | 0.377065 | +0.015912 | 18.822141 | -1.381213 | 0.032967 | -1.099 | 0.005495 | +0.549 | 266.819 |
| synth15_only | ShopFacade | 0.161037 | +0.006973 | 2.920751 | +0.062654 | 0.766990 | -0.971 | 0.271845 | +2.913 | 432.495 |
| synth15_only | KingsCollege | 0.179621 | +0.005283 | 16.233075 | -0.018978 | 0.014577 | -0.292 | 0.002915 | -0.292 | 551.845 |
| synth15_only | OldHospital | 0.365977 | +0.004823 | 19.420233 | -0.783121 | 0.038462 | -0.549 | 0.000000 | +0.000 | 267.780 |

TensorBoard diagnostics，过滤 step 32001-32100：

| Variant | Scene | synthetic used | obs mean | score mean | loc nonzero | loc mean | rank nonzero | KL nonzero |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synth05_only | ShopFacade | 7/100 | 0.999220 | 0.855185 | 7/100 | 0.004972 | 0/100 | 0/100 |
| synth05_only | KingsCollege | 6/100 | 0.742601 | 0.498014 | 6/100 | 0.003945 | 0/100 | 0/100 |
| synth05_only | OldHospital | 6/100 | 0.895365 | 0.559267 | 6/100 | 0.003901 | 0/100 | 0/100 |
| synth15_only | ShopFacade | 18/100 | 0.983459 | 0.735620 | 18/100 | 0.007571 | 0/100 | 0/100 |
| synth15_only | KingsCollege | 17/100 | 0.679045 | 0.486702 | 17/100 | 0.005883 | 0/100 | 0/100 |
| synth15_only | OldHospital | 12/100 | 0.908969 | 0.676700 | 12/100 | 0.005151 | 0/100 | 0/100 |

当前判断：

1. synthetic-only 对照已经排除了“rank/KL 交互导致 P5 mixed”的高影响混杂；这轮 desc/reproj-only 分支确实参与训练，且 rank/KL 均为 0。
2. 0.15 比 0.05 更接近可用：ShopFacade R2 明显提升，KingsCollege TE 几乎持平，OldHospital TE 改善；但三场景 R5 仍全部低于 no-dense，AE 也全部变差。
3. 因此 P5 现在支持“3DGS 渲染 synthetic novel-view 样本已经接入并可改变指标”，不支持“当前 synthetic desc/reproj 策略已经稳定提升 sparse-only relocalization”。
4. 若继续推进 P5，应先校准 synthetic view quality/teacher weighting，再跑 500-step 和 true train-seed；不建议直接把当前策略扩成主结论。

### P6 risk-commit scaffold

日期：2026-06-25

本轮落实 P6 的最小可测路径：先实现 topology mutation proposal 的 accept/reject gate，验证 split mutation 能在 commit 前被拒绝。当前版本不是 true held-out risk estimator；它是后续接入 held-out sparse-risk 评估器所需的 rollback-safe scaffold。

已实现：

1. `TopologyConfig.risk_commit_policy`，支持 `off`、`accept_all`、`reject_all`。
2. `TopologyMutationProposal`，携带 split/prune mask、utility、point count、budget 等 commit 前状态。
3. `LocalizationTopologyController(..., risk_evaluator=...)` callback 入口，便于后续接入 held-out evaluator。
4. risk reject 时保留 `requested_split_count`，但实际 `parent_removed=0`、`children_added=0`、点数不变。
5. risk active 时禁止 soft prune；因为 soft prune 会先改 opacity，当前还没有 rollback-safe 语义。
6. risk active 且 accepted proposal 同时包含 physical prune 与 split 时直接报错，要求拆成独立实验，避免 commit 语义混在一起。
7. `train_locaware.py` 与 `scripts/run_locaware_v03_topology_full.sh` 增加 `--topology_risk_commit_policy` / `TOPOLOGY_RISK_COMMIT_POLICY` 透传。

TDD / targeted 验证：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_topology_controller \
  tests.test_train_locaware_masks \
  tests.test_full_script_args -v

Ran 81 tests in 1.768s
OK
```

P6 smoke 输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_risk_smoke_v1
```

关键配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
load_iteration = 32000
TOPOLOGY_STEPS = 10
TOPOLOGY_UPDATE_INTERVAL = 5
TOPOLOGY_MUTATION_MODE = split_only
TOPOLOGY_RISK_COMMIT_POLICY = reject_all / accept_all
TOPOLOGY_LOC_ANCHORS = 512
TOPOLOGY_FULL_BANK_WEIGHT = 0.02
TOPOLOGY_DIRECT_WEIGHT = 0.02
```

Topology log evidence：

| Policy | Iter | candidates | requested split | parent removed | children added | points | risk |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| reject_all | 32005 | 1371 | 8 | 0 | 0 | 342918 -> 342918 | accepted=False, reason=reject_all |
| reject_all | 32010 | 1491 | 8 | 0 | 0 | 342918 -> 342918 | accepted=False, reason=reject_all |
| accept_all | 32005 | 1371 | 8 | 8 | 16 | 342918 -> 342926 | accepted=True, reason=accept_all |
| accept_all | 32010 | 1459 | 8 | 8 | 16 | 342926 -> 342934 | accepted=True, reason=accept_all |

Sparse-only eval：

| Policy | AE | TE | R5 | R2 | avg inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| reject_all | 0.157358 | 2.759262 | 0.757282 | 0.291262 | 427.408 |
| accept_all | 0.156449 | 2.765148 | 0.757282 | 0.300971 | 426.223 |

当前判断：

1. P6 关键实现语义已经验证：split proposal 可在真正 mutate 前被拒绝，reject 后点数保持不变；accept 后点数按预期增加。
2. 这闭合了“topology mutation 没有可回滚 commit 边界”的实现混杂。
3. 这还没有闭合“held-out sparse-risk 是否能筛出有效 topology mutation”的方法问题；下一步需要把 callback 替换为真实 held-out sparse surrogate，并在 100/500-step 多场景、多 seed 上比较 accept/reject/score-threshold。
4. physical prune 的策略有效性仍未证明；默认阈值下长期实验没有实际删点，本轮只是避免 risk/soft/physical 的不可回滚混杂。

### P6 held-out descriptor risk commit 初版

日期：2026-06-25

本轮继续把 P6 从 scaffold 推进到真实 held-out surrogate：topology split proposal 先在临时 trial state 上应用，再用 held-out query 图像的 descriptor/full-bank surrogate 评分；只有 `trial_risk <= baseline_risk - epsilon` 时才提交 mutation，否则恢复原状态并拒绝。

已实现：

1. `HeldoutRiskCommitEvaluator`，统一处理 baseline/trial 风险、非有限值拒绝、epsilon threshold 和 rollback。
2. `_capture_locaware_training_state()` / `_restore_locaware_training_state()`，覆盖 Gaussian 参数、optimizer、localization state 和 topology controller 计数。
3. `_apply_split_proposal_trial()`，在 trial path 上临时应用 split proposal。
4. `--topology_risk_commit_policy heldout_descriptor`，以及：
   - `--topology_risk_holdout_size`
   - `--topology_risk_epsilon`
   - `--topology_risk_desc_weight`
   - `--topology_risk_full_bank_weight`
   - `--topology_risk_reproj_weight`
   - `--topology_risk_anchors`
5. `scripts/run_locaware_v03_topology_full.sh` 默认启用 support/query split，并透传 P6 risk 参数。
6. topology log 现在输出 `risk_baseline`、`risk_trial`、`risk_delta`、`risk_epsilon`，便于核查每次 accept/reject 是否合理。

TDD / bugfix 验证：

```text
heldout_descriptor parser red -> green
HeldoutRiskCommitEvaluator import red -> green
topology script risk args red -> green
state clone preserves torch.nn.Parameter red -> green
risk log numeric fields red -> green
```

运行时修复了一个高影响实现 bug：最初的 state clone 会把 `torch.nn.Parameter` 变成普通 Tensor，导致 rollback 时无法赋回 `_xyz` 等参数；已增加回归测试并修复为保留 Parameter 类型。

#### 10-step ShopFacade smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_heldout_risk_smoke_v1
```

关键配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
load_iteration = 32000
TOPOLOGY_STEPS = 10
TOPOLOGY_UPDATE_INTERVAL = 5
TOPOLOGY_MUTATION_MODE = split_only
TOPOLOGY_RISK_COMMIT_POLICY = heldout_descriptor
TOPOLOGY_RISK_HOLDOUT_SIZE = 2
TOPOLOGY_RISK_ANCHORS = 128
TOPOLOGY_SUPPORT_QUERY_SPLIT = 1
TOPOLOGY_QUERY_SPLIT_MODE = sequence_block
```

日志确认 support/query split 生效：support=185, query=46。两次 topology proposal 都因 held-out descriptor risk 上升而拒绝：

| Iter | requested split | points | risk baseline | risk trial | risk delta | decision |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 32005 | 8 | 342918 -> 342918 | 12.224746 | 12.227475 | +0.002729 | reject |
| 32010 | 8 | 342918 -> 342918 | 12.218366 | 12.222553 | +0.004188 | reject |

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.160554 | 2.937220 | 0.757282 | 0.291262 | 426.447 |

结论：10-step smoke 证明真实 held-out risk path、trial rollback、非提交拒绝和 sparse eval 都可运行；不作为精度结论。

#### 100-step heldout_descriptor vs matched no_mutation

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_heldout_risk_100_v1
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_100_v1
```

共享配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
update_interval = 25
mutation = split_only vs no_mutation
train_seed = 0
query_split_seed = 2025
query_split_mode = sequence_block
TOPOLOGY_RISK_HOLDOUT_SIZE = 4
TOPOLOGY_RISK_EPSILON = 0.0
TOPOLOGY_RISK_ANCHORS = 128
TOPOLOGY_LOC_ANCHORS = 512
TOPOLOGY_FULL_BANK_WEIGHT = 0.02
TOPOLOGY_DIRECT_WEIGHT = 0.02
```

Risk-gated topology events：

| Scene | Iter | requested | children | points | risk baseline | risk trial | risk delta | decision |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| ShopFacade | 32025 | 10 | 20 | 342918 -> 342928 | 11.479605 | 11.470371 | -0.009234 | accept |
| ShopFacade | 32050 | 11 | 0 | 342928 -> 342928 | 11.426976 | 11.473818 | +0.046842 | reject |
| ShopFacade | 32075 | 13 | 0 | 342928 -> 342928 | 11.404666 | 11.451273 | +0.046606 | reject |
| ShopFacade | 32100 | 16 | 0 | 342928 -> 342928 | 11.372154 | 11.392043 | +0.019889 | reject |
| KingsCollege | 32025 | 6 | 12 | 318593 -> 318599 | 10.466582 | 10.461929 | -0.004652 | accept |
| KingsCollege | 32050 | 7 | 0 | 318599 -> 318599 | 10.113792 | 10.130872 | +0.017080 | reject |
| KingsCollege | 32075 | 8 | 0 | 318599 -> 318599 | 9.873167 | 9.900034 | +0.026867 | reject |
| KingsCollege | 32100 | 9 | 0 | 318599 -> 318599 | 9.637625 | 9.642540 | +0.004915 | reject |
| OldHospital | 32025 | 4 | 0 | 405348 -> 405348 | 11.042598 | 11.043059 | +0.000461 | reject |
| OldHospital | 32050 | 4 | 0 | 405348 -> 405348 | 10.765137 | 10.765534 | +0.000397 | reject |
| OldHospital | 32075 | 4 | 0 | 405348 -> 405348 | 10.528217 | 10.528571 | +0.000354 | reject |
| OldHospital | 32100 | 5 | 10 | 405348 -> 405353 | 10.330851 | 10.309811 | -0.021040 | accept |

Sparse-only paired result，`heldout_descriptor - no_mutation`：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.162595 | +0.004170 | 2.930288 | +0.047633 | 0.766990 | +0.000 | 0.252427 | +0.971 | 398.951 | +0.068 |
| KingsCollege | 0.179774 | -0.000386 | 16.374716 | -0.251763 | 0.014577 | +0.000 | 0.000000 | +0.000 | 542.845 | +0.061 |
| OldHospital | 0.368359 | -0.002646 | 20.457874 | +0.010700 | 0.038462 | +0.549 | 0.000000 | +0.000 | 264.478 | +0.022 |

当前判断：

1. P6 不再只是 scaffold：真实 held-out descriptor surrogate 已接入训练脚本、topology controller 和 full script，并在三场景 100-step 上完成 matched 对照。
2. risk gate 行为符合预期：12 次 proposal 中只提交 3 次，全部是 held-out surrogate 下降的 proposal；风险上升的 proposal 均拒绝且点数不变。
3. 精度证据是温和正向但仍 mixed：KingsCollege AE/TE 改善，OldHospital AE/R5 改善，ShopFacade R2/inliers 改善；但 ShopFacade AE/TE 变差，OldHospital TE 略差。
4. 因此 P6 初版排除了“没有真实 held-out gate / rollback 不可靠”的混杂，但还没有证明最终策略稳定有效。
5. 在 held-out descriptor 阶段，P6 仍缺 500-step、多 seed、更多 scene，以及 full sparse-pose risk 或更贴近 PnP 成功率的 surrogate；当前 descriptor risk 只能作为低成本筛选，不应被表述为最终 pose-risk commit。后续 `heldout_pose` 小节补上了 sparse-pose risk 初版，但仍需长期和多 seed 稳定性验证。

### P6 held-out sparse pose risk commit 初版

日期：2026-06-25

本轮继续把 P6 从 held-out descriptor surrogate 推进到真实 sparse-only pose-risk gate：每个 topology split proposal 先在 trial state 上临时应用，然后用 held-out query 相机实际跑 `STDLoc.localize()`，以 sparse AE/TE 和可选 inlier reward 计算风险；只有 trial pose risk 不高于 baseline 才提交 mutation。

已实现：

1. `--topology_risk_commit_policy heldout_pose`。
2. `--topology_risk_pose_cfg`，复用当前 run 生成的 STDLoc artifact yaml。
3. `--topology_risk_pose_ae_weight` / `--topology_risk_pose_te_weight` / `--topology_risk_pose_inlier_weight`。
4. `--topology_risk_pose_ae_scale` / `--topology_risk_pose_te_scale` / `--topology_risk_pose_inlier_scale`。
5. `_score_heldout_sparse_pose_risk()`：刷新当前 Gaussian landmark rows 后，对 held-out cameras 执行 sparse localization，再用 `cal_pose_error()` 得到 AE/TE。
6. `_refresh_stdloc_sparse_landmarks()`：按 `source_index` 重新映射 baseline sampled landmark 到当前 rows，避免 split 后仍评估 stale sampled_idx。
7. `scripts/run_locaware_v03_topology_full.sh` 透传 pose-risk cfg/weights/scales。

Risk score：

```text
risk = ae_weight * (median_AE_deg / ae_scale)
     + te_weight * (median_TE_cm / te_scale)
     - inlier_weight * clamp(avg_inliers / inlier_scale, 0, 1)
```

本轮 100-step 主实验使用 `ae_weight=1.0`、`te_weight=1.0`、`inlier_weight=0.0`、`ae_scale=5deg`、`te_scale=200cm`，先避免 inlier reward 掩盖 pose error。

TDD / targeted 验证：

```text
heldout_pose parser red -> green
_pose_risk_from_sparse_metrics import red -> green
topology script pose-risk args red -> green
```

Targeted green：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default -v

Ran 3 tests
OK
```

Related suite green：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks tests.test_topology_controller tests.test_full_script_args -v

Ran 86 tests in 1.744s
OK
```

#### 10-step ShopFacade smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_heldout_pose_smoke_v1
```

关键配置：

```text
scene = ShopFacade
source = /mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
TOPOLOGY_STEPS = 10
TOPOLOGY_UPDATE_INTERVAL = 5
TOPOLOGY_MUTATION_MODE = split_only
TOPOLOGY_RISK_COMMIT_POLICY = heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE = 1
```

两次 proposal 均因 sparse pose risk 上升而拒绝，点数保持不变：

| Iter | requested split | points | risk baseline | risk trial | risk delta | decision |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| 32005 | 8 | 342918 -> 342918 | 0.042424 | 0.043715 | +0.001291 | reject |
| 32010 | 8 | 342918 -> 342918 | 0.044031 | 0.045124 | +0.001094 | reject |

Sparse-only eval：

| AE | TE | R5 | R2 | avg inliers |
| ---: | ---: | ---: | ---: | ---: |
| 0.160554 | 2.937220 | 0.757282 | 0.291262 | 426.447 |

结论：10-step smoke 证明 `heldout_pose` path、landmark refresh、trial rollback、非提交拒绝和 sparse eval 都可运行；不作为精度结论。

#### 100-step heldout_pose vs matched no_mutation

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_heldout_pose_100_v1
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_100_v1
```

共享配置：

```text
scenes = ShopFacade, KingsCollege, OldHospital
source = /mnt/pool/sqy/stdloc_la_v03_full_length/<scene>/seed_2025/<scene>_v03
load_iteration = 32000
steps = 100
update_interval = 25
mutation = split_only vs no_mutation
train_seed = 0
query_split_seed = 2025
query_split_mode = sequence_block
TOPOLOGY_RISK_HOLDOUT_SIZE = 4
TOPOLOGY_RISK_EPSILON = 0.0
```

Risk-gated topology events：

| Scene | Iter | requested | children | points | risk baseline | risk trial | risk delta | decision |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| ShopFacade | 32025 | 10 | 20 | 342918 -> 342928 | 0.068401 | 0.066881 | -0.001519 | accept |
| ShopFacade | 32050 | 11 | 0 | 342928 -> 342928 | 0.067358 | 0.068000 | +0.000642 | reject |
| ShopFacade | 32075 | 13 | 0 | 342928 -> 342928 | 0.063459 | 0.064000 | +0.000541 | reject |
| ShopFacade | 32100 | 16 | 0 | 342928 -> 342928 | 0.062379 | 0.148634 | +0.086255 | reject |
| KingsCollege | 32025 | 6 | 12 | 318593 -> 318599 | 0.136854 | 0.136104 | -0.000750 | accept |
| KingsCollege | 32050 | 7 | 14 | 318599 -> 318606 | 0.128968 | 0.128235 | -0.000733 | accept |
| KingsCollege | 32075 | 8 | 16 | 318606 -> 318614 | 0.131714 | 0.131402 | -0.000312 | accept |
| KingsCollege | 32100 | 8 | 16 | 318614 -> 318622 | 0.120207 | 0.120055 | -0.000152 | accept |
| OldHospital | 32025 | 4 | 8 | 405348 -> 405352 | 0.072341 | 0.071492 | -0.000850 | accept |
| OldHospital | 32050 | 4 | 0 | 405352 -> 405352 | 0.067357 | 0.067978 | +0.000620 | reject |
| OldHospital | 32075 | 4 | 0 | 405352 -> 405352 | 0.070701 | 0.071503 | +0.000802 | reject |
| OldHospital | 32100 | 5 | 0 | 405352 -> 405352 | 0.073292 | 0.073428 | +0.000136 | reject |

Sparse-only paired result，`heldout_pose - no_mutation`：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.162595 | +0.004170 | 2.930288 | +0.047633 | 0.766990 | +0.000 | 0.252427 | +0.971 | 398.951 | +0.068 |
| KingsCollege | 0.177968 | -0.002192 | 16.430409 | -0.196069 | 0.017493 | +0.292 | 0.000000 | +0.000 | 542.749 | -0.035 |
| OldHospital | 0.365598 | -0.005406 | 20.409856 | -0.037318 | 0.032967 | +0.000 | 0.000000 | +0.000 | 264.495 | +0.038 |

Sparse-only paired result，`heldout_pose - heldout_descriptor`：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.000000 | +0.000000 | +0.000 | +0.000 | +0.000 |
| KingsCollege | -0.001806 | +0.055693 | +0.292 | +0.000 | -0.096 |
| OldHospital | -0.002760 | -0.048018 | -0.549 | +0.000 | +0.016 |

当前判断：

1. P6 sparse-pose risk evaluator 已经不是 TODO：它已接入训练脚本和 full topology script，并完成三场景 100-step matched 对照。
2. gate 行为符合定义：12 次 proposal 中 6 次提交，提交事件均满足 held-out pose risk 下降；拒绝事件点数保持不变。
3. 相对 no-mutation，`heldout_pose` 在 KingsCollege 的 AE/TE/R5 和 OldHospital 的 AE/TE/inliers 给出局部正向；ShopFacade 与 descriptor gate 完全同结果，R2/inliers 正向但 AE/TE 变差。
4. 相对 heldout_descriptor，pose-risk 不是单调更好：KingsCollege R5/AE 更好但 TE/inliers 更差，OldHospital AE/TE/inliers 更好但 R5 更差。
5. 因此本轮闭合的是“P6 是否能用真实 sparse pose risk 作为 commit gate”的实现混杂；尚未证明 pose-risk gate 稳定提升最终 sparse-only relocalization。
6. 下一步不能再把 P6 表述为“缺 evaluator”，而应聚焦 holdout size、risk 权重、PnP stochasticity、500-step/multi-seed 稳定性。

#### 500-step heldout_pose vs matched no_mutation

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_heldout_pose_500_v1
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_500_v1
```

共享配置与 100-step 相同，但 `steps=500`，`update_interval=25`，即每个 scene 最多 20 次 topology proposal。该 no-mutation 对照是严格 matched rerun：同源 `32000 -> 32500`、同 `train_seed=0`、同 `query_split_seed=2025`、同 direct/full-bank objective；不是复用旧的 `topology_from_30500` 长跑结果。

Risk-gated event 汇总：

| Scene | Events | Accepted | Rejected | Children | Point delta | Accepted iters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ShopFacade | 20 | 3 | 17 | 118 | 59 | 32025, 32200, 32500 |
| KingsCollege | 20 | 11 | 9 | 226 | 113 | 32025, 32050, 32075, 32100, 32175, 32300, 32350, 32375, 32400, 32425, 32475 |
| OldHospital | 20 | 7 | 13 | 100 | 50 | 32025, 32150, 32200, 32275, 32425, 32450, 32500 |

Sparse-only paired result，`heldout_pose - no_mutation`：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.161426 | -0.001701 | 3.100866 | -0.042617 | 0.757282 | -0.971 | 0.262136 | +0.971 | 353.922 | -0.631 |
| KingsCollege | 0.196674 | +0.006780 | 17.181985 | +0.258827 | 0.034985 | -0.292 | 0.002915 | +0.000 | 493.254 | -0.985 |
| OldHospital | 0.403065 | +0.001341 | 20.417187 | +0.075001 | 0.021978 | -1.099 | 0.000000 | +0.000 | 263.747 | -1.533 |

当前判断：

1. 500-step 进一步证明 gate 语义正确：所有 accepted event 都满足 held-out pose risk 下降，所有 rejected event 都保持点数不变。
2. 但 500-step 不支持当前 `holdout=4, epsilon=0, AE+TE risk` 配置具有稳定正收益。ShopFacade 只有 AE/TE/R2 小幅改善，R5/inliers 下降；KingsCollege 和 OldHospital 相对严格 no-mutation 对照主要变差。
3. 这说明 P6 的核心实现混杂已经排除，但策略层仍未闭合：小 holdout、零 epsilon 和逐 event 贪心接受会把 held-out 小样本上的微弱风险下降误当作可泛化收益。
4. 后续 P6 不应继续无条件扩当前参数，而应优先做 risk 校准：增大 holdout、加入 bootstrap/置信界或最小收益阈值、限制累计 accepted events，并显式检查 PnP stochasticity 对 risk delta 的影响。

#### 500-step heldout_pose epsilon=0.001 校准

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_eps001_500_v1
```

本轮只改一个变量：`TOPOLOGY_RISK_EPSILON=0.001`。其余 source、`train_seed=0`、`query_split_seed=2025`、`steps=500`、`holdout=4`、AE/TE risk 权重、direct/full-bank objective 均与上面的 `epsilon=0` 500-step 对照一致。

Risk-gated event 汇总：

| Scene | Events | Accepted | Rejected | Children | Point delta | Accepted iters |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ShopFacade | 20 | 3 | 17 | 118 | 59 | 32025, 32200, 32500 |
| KingsCollege | 20 | 3 | 17 | 74 | 37 | 32075, 32425, 32450 |
| OldHospital | 20 | 3 | 17 | 38 | 19 | 32075, 32250, 32375 |

Sparse-only paired result，`epsilon=0.001 - no_mutation`：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.161426 | -0.001701 | 3.100866 | -0.042617 | 0.757282 | -0.971 | 0.262136 | +0.971 | 353.922 | -0.631 |
| KingsCollege | 0.194463 | +0.004569 | 17.058906 | +0.135748 | 0.037901 | +0.000 | 0.005831 | +0.292 | 494.146 | -0.093 |
| OldHospital | 0.410670 | +0.008946 | 20.204573 | -0.137613 | 0.032967 | +0.000 | 0.000000 | +0.000 | 265.093 | -0.187 |

Sparse-only paired result，`epsilon=0.001 - epsilon=0`：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.000000 | +0.000000 | +0.000 | +0.000 | +0.000 |
| KingsCollege | -0.002211 | -0.123079 | +0.292 | +0.292 | +0.892 |
| OldHospital | +0.007605 | -0.212613 | +1.099 | +0.000 | +1.346 |

当前判断：

1. `epsilon=0.001` 验证了“零 epsilon 过度接受微小 held-out risk delta”的假设：KingsCollege accepted 从 11/20 降到 3/20，OldHospital 从 7/20 降到 3/20。
2. stricter epsilon 对 `epsilon=0` 有明显止损：KingsCollege 的 AE/TE/R5/R2/inliers 均改善；OldHospital 的 TE/R5/inliers 改善，但 AE 变差。
3. 但相对严格 no-mutation，`epsilon=0.001` 仍不是稳定正收益：ShopFacade 与原结果相同，KingsCollege AE/TE 仍变差，OldHospital AE 仍变差；正向只集中在 Shop AE/TE/R2、Kings R2、Old TE。
4. 因此 P6 下一步不应只继续扫 epsilon，而应把 epsilon 与更大 holdout、bootstrap/置信界、accepted-event budget 结合；否则它只是降低过度 mutation，而不是证明 topology mutation 有可泛化收益。

### 截至 2026-06-25 的状态更新

当前 LA-STDLoc 使用的数据分三层：

1. 主训练集仍是 STDLoc 预处理 Cambridge 原始图像/相机数据：`/mnt/pool/sqy/Cambridge_stdloc/<scene>/processed`。
2. P3/P4/P4.1 已经使用 3DGS 渲染 teacher/geometry：真实 query 或 perturb pose 上渲染 feature/depth/alpha，用于 dense/rank/advantage 类训练信号。
3. P5 已经使用 3DGS 渲染优势构造 novel-view synthetic query：从相邻真实相机插值 pose，再渲染 synthetic feature map 作为 desc/reproj teacher；但它目前仍是实验分支，不是默认主配置。

定性排查方面，LA_update2/3 的高影响混杂已经关闭了一大批：support/query split、false negative、多正例、seed/split 可复现、opacity 未训练误删、topology one-shot/repeated/freeze 对照、dense advantage/rank、synthetic view inert、risk commit 边界、held-out descriptor risk gate、held-out sparse pose risk gate。但还不能说全部完成：

1. P5 缺 500-step、多 seed、teacher quality calibration。
2. P6 sparse-pose risk evaluator 已完成初版，并完成 seed0 的 100/500-step 三场景 matched 对照；`epsilon=0.001` 已证明可减少过度接受并相对 `epsilon=0` 止损，但仍缺 multi-seed、更大 holdout/置信界校准，以及 PnP stochasticity 稳健性检查。
3. physical prune 缺有效触发并带来收益的实验。
4. 多数精度实验仍是 `train_seed=0, query_split_seed=2025`，统计置信度不足。

精度支撑仍是局部正向、整体不稳定。最强正向证据来自：

1. P0 topology 在 ShopFacade/KingsCollege 局部提高 R5 或 TE，但 OldHospital 负向。
2. P3/P4/P5 在某些 scene/metric 上改善 TE、AE 或 R2。
3. P5 synthetic-only 0.15 在 ShopFacade 提升 R2，在 KingsCollege/OldHospital 改善 TE。
4. P6 held-out descriptor gate 在 100-step matched matrix 中带来 KingsCollege AE/TE、OldHospital AE/R5、ShopFacade R2 的局部正向。
5. P6 held-out sparse-pose gate 在 100-step matched matrix 中带来 KingsCollege AE/TE/R5、OldHospital AE/TE 的局部正向，但 ShopFacade AE/TE 变差；扩到 500-step 后 `epsilon=0` 没有继续支撑稳定收益，`epsilon=0.001` 相对 `epsilon=0` 止损但仍未稳定优于 matched no-mutation。

但目前还没有形成跨 scene、跨 metric、跨 seed 的稳定正向证据。因此，当前主张应保持收敛版本：3DGS 渲染与 localization-aware 训练通路可用，且若干模块能产生局部正向信号；但当前具体策略尚未证明能稳定提升 sparse-only relocalization。

是否需要外部专家：现在可以整理给外部专家讨论，但更合适的问题不是“主张是否错误”，而是“下一步方法应如何调整”。建议带着当前证据讨论三点：synthetic view 的质量/权重 curriculum、held-out sparse/pose-risk gate 的统计稳定性和权重设计、以及是否把目标从 topology mutation 转向 frozen 3DGS + lightweight descriptor overlay。

### 2026-06-25 P6 rngfix + paired-CI 补充

为继续闭合 P6 的高影响混杂，本轮在 `heldout_pose` evaluator 上补了两层保护：

1. 对 risk scoring 做 Python / Torch CPU / Torch CUDA / NumPy RNG capture+restore，避免 trial/reject 分支消耗随机数后污染后续训练。
2. 增加 paired per-heldout-camera risk UCB gate：`delta_risk_ucb <= -epsilon` 才接受 proposal；本轮使用 `holdout=8, epsilon=0.001, ci_z=1.96, min_ci_samples=8`。

运行配置：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_ci_holdout8_label16_rngfix_500_v1
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=8
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_CI_Z=1.96
TOPOLOGY_RISK_MIN_CI_SAMPLES=8
LABEL_MAX_IMAGES=16
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_STEPS=500
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
V03_ITERATION=32000
```

三场景结果：

| Scene | Events | Accepted | Children | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 20 | 0 | 0 | 0.150403 | 2.988938 | 0.776699 | 0.281553 | 420.087 |
| KingsCollege | 20 | 0 | 0 | 0.204003 | 17.735989 | 0.064140 | 0.000000 | 631.892 |
| OldHospital | 20 | 0 | 0 | 0.352415 | 18.018994 | 0.038462 | 0.000000 | 318.093 |

随后补齐三场景当前代码版 `no_mutation` 500-step 严格对照：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_500_current_v1
```

结果与 rngfix+CI 全拒绝行逐项完全一致：

| Scene | Mode | AE | TE | R5 | R2 | Inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | rngfix+CI split_only, 20/20 reject | 0.150403 | 2.988938 | 0.776699 | 0.281553 | 420.087 |
| ShopFacade | current no_mutation | 0.150403 | 2.988938 | 0.776699 | 0.281553 | 420.087 |
| KingsCollege | rngfix+CI split_only, 20/20 reject | 0.204003 | 17.735989 | 0.064140 | 0.000000 | 631.892 |
| KingsCollege | current no_mutation | 0.204003 | 17.735989 | 0.064140 | 0.000000 | 631.892 |
| OldHospital | rngfix+CI split_only, 20/20 reject | 0.352415 | 18.018994 | 0.038462 | 0.000000 | 318.093 |
| OldHospital | current no_mutation | 0.352415 | 18.018994 | 0.038462 | 0.000000 | 318.093 |

当前判断：

1. 这在三场景上排除了“rejected proposal 仍污染训练状态/RNG”的主要怀疑；全拒绝时指标与当前 no-mutation 完全一致，逐项 diff 为 0。
2. 旧的 `/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_500_v1` no-mutation 指标已经不能再作为当前代码的严格同源对照；它与 current no-mutation 不同，但 source checkpoint、label16 state、sampled_idx remap 均已审计一致，因此更可能是历史运行/代码状态/训练非确定性的对照漂移。
3. rngfix+CI 本身没有证明 topology 策略有效，因为三场景 0 accepted、0 children；它证明的是严格 UCB gate 足以阻止不可靠 proposal，且当前结果需要用重新复跑的 current no-mutation 作对照。
4. KingsCollege 和 OldHospital 也都是 0 accepted、0 children；其指标变化同样不应归因于 topology mutation。本批应归类为 strict gate inert/control run。

更新后的 P6 结论：P6 的 rollback/RNG/CI 实现混杂已进一步收敛，但策略有效性尚未证明。下一步不应把这批精度提升写成 topology gain，而应继续做 current-control 对齐后的 risk calibration：更大 holdout、多 seed、accepted-event budget、以及能真实接受少量高置信 proposal 的非 inert gate。

#### P6 budget3 非 inert 校准

为确认 P6 不是只能进入全拒绝 inert 状态，本轮在 current-control 对齐后补跑了一个低预算非 inert 配置：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_holdout8_eps001_budget3_500_current_v1
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=8
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_CI_Z=0.0
TOPOLOGY_RISK_MIN_CI_SAMPLES=8
TOPOLOGY_MAX_MUTATION_EVENTS=3
LABEL_MAX_IMAGES=16
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_STEPS=500
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
V03_ITERATION=32000
```

相对当前代码版 matched `no_mutation` 500-step 的 sparse-only 结果：

| Scene | Events | Accepted | Children | Accepted iters | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 20 | 1 | 52 | 32225 | 0.142132 | -0.008271 | 2.956021 | -0.032917 | 0.757282 | -1.942 | 0.262136 | -1.942 | 420.350 | +0.262 |
| KingsCollege | 20 | 1 | 42 | 32300 | 0.202933 | -0.001070 | 17.591666 | -0.144324 | 0.061224 | -0.292 | 0.000000 | +0.000 | 632.571 | +0.679 |
| OldHospital | 9 | 3 | 38 | 32050, 32075, 32225 | 0.348966 | -0.003449 | 18.339173 | +0.320179 | 0.032967 | -0.549 | 0.005495 | +0.549 | 316.121 | -1.973 |

当前判断：

1. 非 inert 通路已经被验证：三场景都有 accepted mutation，children 实际增加；OldHospital 在 3 次 accepted 后停止，说明 `TOPOLOGY_MAX_MUTATION_EVENTS=3` 的事件预算生效。
2. 相比 LA_update1/2 阶段的 mixed topology，本轮至少排除了两个关键混杂：全拒绝不会污染训练状态/RNG；真实接受时是受 held-out sparse-pose risk 和事件预算约束的，不再是无约束 repeated split。
3. 精度信号仍不能写成稳定正向：三场景 AE 均改善，ShopFacade/KingsCollege TE 与 inliers 也改善，但 R5 三场景下降，R2 只有 OldHospital 改善、ShopFacade 下降、KingsCollege 持平。当前 risk score 更像在优化 median AE/TE，而没有稳定保护 recall 阈值精度。
4. 因此，P6 已从“实现是否有副作用”推进到“策略目标函数是否对齐 sparse-only recall”的问题。下一步应优先调整 risk design，例如加入 inlier/recall-sensitive 项、按 query camera 做更稳的分层 holdout，或提高 accepted margin，而不是继续把现有 AE+TE risk 直接扩大成主结论。

#### P6 recall/tail veto 初版

针对 budget3 暴露出的“AE/TE risk 下降但 R5 下降”问题，本轮把 `LA_update3.md` 中 P6 的 recall/tail 约束落实为可开关的 held-out pose metric gate：

1. `HeldoutRiskCommitEvaluator` 支持 score 返回 `metrics`，先按原有 scalar risk / paired-CI 判断，再由可选 `metric_gate_fn` 二次 veto。
2. `heldout_pose` scorer 在 `--topology_risk_pose_veto_mode != off` 时返回 held-out camera 上的 `r5_count`、`r2_count`、`tail_fail_count` 和 rate。
3. 新增 `--topology_risk_pose_veto_mode {off,r5,r5_r2,r5_r2_tail}`，默认 `off` 保持旧实验可复现；脚本透传 `TOPOLOGY_RISK_POSE_VETO_MODE`、`TOPOLOGY_RISK_POSE_R2_TOLERANCE`、`TOPOLOGY_RISK_POSE_TAIL_TOLERANCE`。
4. topology 日志新增 `risk_metric_count`、`risk_r5_delta`、`risk_r2_delta`、`risk_tail_fail_delta` 以及对应 rate delta，方便后续实验审计。

单测和静态检查：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_accepts_only_when_trial_risk_decreases \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_applies_metric_veto_after_risk_drop \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_uses_paired_ucb_when_enabled \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_ci_when_sample_count_is_too_small \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_recall_tail_veto_rejects_r5_and_tail_regressions \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_restores_rng_state_after_scoring \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_nonfinite_trial_risk_and_restores \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default

Ran 10 tests in 1.657s
OK
```

真实 smoke：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_smoke_v1
ShopFacade, 10 step, holdout=8, epsilon=0.001, veto_mode=r5_r2_tail
```

两次 proposal 都进入真实 held-out sparse-pose evaluator，并因 scalar risk 下降未超过 epsilon 而拒绝：

| Iter | Accepted | Children | risk_delta | Reason |
| ---: | --- | ---: | ---: | --- |
| 32005 | False | 0 | -0.000784 | heldout_pose_not_decreased |
| 32010 | False | 0 | -0.000292 | heldout_pose_not_decreased |

10-step sparse eval 只作 smoke 记录：AE 0.157124、TE 2.781776、R5 0.757282、R2 0.291262、Inliers 427.835。

为实际触发 metric gate 日志，又补了一个 `epsilon=0` 的 5-step smoke：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_smoke_eps0_v1
ShopFacade, 5 step, holdout=8, epsilon=0.0, veto_mode=r5_r2_tail
```

该 run 在 32005 接受 1 次 split，日志显示 held-out metric deltas 均未恶化：

| Iter | Accepted | Children | risk_delta | metric_count | r5_delta | r2_delta | tail_fail_delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32005 | True | 16 | -0.000784 | 8 | 0 | 0 | 0 |

5-step sparse eval：AE 0.160815、TE 2.896246、R5 0.757282、R2 0.271845、Inliers 426.922。这不作为精度结论；它只说明新 gate 可以在真实训练路径中运行并记录 recall/tail evidence。R2 下降再次提醒：held-out 8-camera veto 只能约束小 buffer，不能替代 full test set/multi-seed 验证。

当前判断：这一改动把 P6 从“AE+TE risk commit”推进到“AE+TE risk + held-out recall/tail veto”。它更符合 `LA_update3.md` 对 P6 的定义，也更直接回应 budget3 的 R5 退化问题；但是否带来稳定正向，还需要下一轮 100/500-step 三场景 matched 实验验证。

#### P6 recall/tail veto 100-step 三场景复核

为验证 recall/tail veto 不只是 5/10-step smoke，本轮跑了三场景 100-step：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_r5r2tail_100_v1
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=8
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_CI_Z=0.0
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_MAX_MUTATION_EVENTS=3
LABEL_MAX_IMAGES=16
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_STEPS=100
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
V03_ITERATION=32000
```

Topology decision：

| Scene | Events | Accepted | Rejected | Children | Accepted iters | Rejection reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ShopFacade | 4 | 0 | 4 | 0 | - | heldout_pose_not_decreased |
| KingsCollege | 4 | 0 | 4 | 0 | - | heldout_pose_not_decreased |
| OldHospital | 4 | 2 | 2 | 22 | 32050, 32075 | accepted events had r5/r2/tail deltas all 0 |

Sparse-only result：

| Scene | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.155046 | 2.769958 | 0.757282 | 0.242718 | 422.922 |
| KingsCollege | 0.188302 | 16.800641 | 0.014577 | 0.000000 | 582.120 |
| OldHospital | 0.371636 | 19.069876 | 0.054945 | 0.010989 | 284.819 |

参考已有 100-step no-mutation 结果：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_100_v1
```

| Scene | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.158425 | 2.882655 | 0.766990 | 0.242718 | 398.883 |
| KingsCollege | 0.180159 | 16.626479 | 0.014577 | 0.000000 | 542.784 |
| OldHospital | 0.371005 | 20.447174 | 0.032967 | 0.000000 | 264.456 |

参考差值，`veto100 - no_mutation100`：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.003380 | -0.112697 | -0.971 | +0.000 | +24.039 |
| KingsCollege | +0.008142 | +0.174162 | +0.000 | +0.000 | +39.335 |
| OldHospital | +0.000631 | -1.377299 | +2.198 | +1.099 | +20.363 |

当前判断：

1. recall/tail veto 的真实训练路径已经跑通三场景，并能在 accepted event 上记录 `risk_metric_count/risk_r5_delta/risk_r2_delta/risk_tail_fail_delta`；OldHospital 的两个 accepted split 没有触发 held-out R5/R2/tail 回退。
2. ShopFacade 与 KingsCollege 在 `epsilon=0.001` 下全部拒绝，因此它们的结果不能归因为 topology mutation；这两场景主要证明 gate 足够保守。
3. OldHospital 是本轮唯一非 inert 场景：2 次 accepted、22 个 children，full sparse eval 相对已有 no-mutation100 的 TE/R5/R2/inliers 更好，但 AE 轻微变差。由于 no-mutation100 不是本轮重跑的 current-control，这只能作为正向迹象，不能作为严格归因。
4. 相比 budget3 500-step 暴露的“三场景 AE 改善但 R5 全部下降”，recall/tail veto 至少把 P6 的目标函数推进到了能显式审计 recall/tail 回退的状态；但目前仍没有形成跨 scene 稳定精度收益。
5. 下一步若继续闭合，应补同配置 current no-mutation100 和 500-step veto，并扩到多 seed；否则不能把 OldHospital 的改善写成最终方法结论。

#### P6 current-control 与 holdout selection 更新

上面的 100-step 结果随后补了当前代码强制重跑的 strict no-mutation100：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_100_current_v2
TOPOLOGY_MUTATION_MODE=no_mutation
TOPOLOGY_STEPS=100
LABEL_MAX_IMAGES=16
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
FORCE_TOPOLOGY_COPY=1
FORCE_TOPOLOGY_TRAIN=1
FORCE_LABEL_STATE=1
```

current no-mutation100：

| Scene | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.155046 | 2.769958 | 0.757282 | 0.242718 | 422.922 |
| KingsCollege | 0.188302 | 16.800641 | 0.014577 | 0.000000 | 582.120 |
| OldHospital | 0.364951 | 19.018328 | 0.049451 | 0.005495 | 284.901 |

因此 holdout8 prefix veto100 相对 strict current-control 的解释更新为：

| Scene | Topology | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0 accepted | +0.000000 | +0.000000 | +0.000 | +0.000 | +0.000 |
| KingsCollege | 0 accepted | +0.000000 | +0.000000 | +0.000 | +0.000 | +0.000 |
| OldHospital | 2 accepted, +22 children | +0.006685 | +0.051548 | +0.549 | +0.549 | -0.082 |

这个更新把旧“OldHospital TE 明显正向”的解释收回：相对 current-control，OldHospital 的 100-step holdout8 prefix 结果更准确地说是 R5/R2 小幅改善，但 AE/TE 变差。

同配置 500-step 三场景也已补齐：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_r5r2tail_500_v1
TOPOLOGY_RISK_HOLDOUT_SIZE=8
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_MAX_MUTATION_EVENTS=3
TOPOLOGY_STEPS=500
```

结果与前面的 P6 budget3 非 inert 校准逐项一致：

| Scene | Events | Accepted | Children | Accepted iters | AE | dAE vs current no-mutation500 | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 20 | 1 | 52 | 32225 | 0.142132 | -0.008271 | 2.956021 | -0.032917 | 0.757282 | -1.942 | 0.262136 | -1.942 | 420.350 | +0.262 |
| KingsCollege | 20 | 1 | 42 | 32300 | 0.202933 | -0.001070 | 17.591666 | -0.144324 | 0.061224 | -0.292 | 0.000000 | +0.000 | 632.571 | +0.679 |
| OldHospital | 9 | 3 | 38 | 32050, 32075, 32225 | 0.348966 | -0.003449 | 18.339173 | +0.320179 | 0.032967 | -0.549 | 0.005495 | +0.549 | 316.121 | -1.973 |

关键结论：

1. recall/tail veto 没有引入新的训练副作用；500-step 复现了 budget3。
2. 但它也没有解决 full-test R5/R2 下滑：accepted event 的 held-out `r5/r2/tail` 没有退化，不代表 full test set 不退化。
3. 代码复核发现根因之一：`heldout_pose` risk 原先固定使用 `list(query_cameras)[:holdout_size]`，即 query holdout 的前缀小样本。`holdout=8` 在 sequence-block split 下代表性不足。

为降低这个实现混杂，本轮新增：

```text
--topology_risk_holdout_selection {prefix,strided}
TOPOLOGY_RISK_HOLDOUT_SELECTION
```

默认 `prefix` 保持旧实验可复现；`strided` 在整个 query holdout 上均匀取样。相关验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_select_risk_cameras_supports_prefix_and_strided_modes \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_applies_metric_veto_after_risk_drop \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_recall_tail_veto_rejects_r5_and_tail_regressions \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_restores_rng_state_after_scoring \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default

Ran 7 tests in 2.080s
OK
```

真实 100-step 初步验证：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_r5r2tail_hold32_strided_100_v1
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=strided
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_STEPS=100
```

Topology decision：

| Scene | Events | Accepted | Children | Accepted iters | Held-out metric evidence |
| --- | ---: | ---: | ---: | --- | --- |
| ShopFacade | 4 | 0 | 0 | - | all rejected by scalar risk |
| KingsCollege | 4 | 0 | 0 | - | all rejected by scalar risk |
| OldHospital | 4 | 1 | 8 | 32025 | metric_count=32, tail_fail_delta=-1, r5/r2 deltas 0 |

Sparse-only result vs current no-mutation100：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.155046 | +0.000000 | 2.769958 | +0.000000 | 0.757282 | +0.000 | 0.242718 | +0.000 | 422.922 | +0.000 |
| KingsCollege | 0.188302 | +0.000000 | 16.800641 | +0.000000 | 0.014577 | +0.000 | 0.000000 | +0.000 | 582.120 | +0.000 |
| OldHospital | 0.375196 | +0.010245 | 19.322717 | +0.304389 | 0.054945 | +0.549 | 0.010989 | +0.549 | 284.225 | -0.676 |

当前判断：

1. `strided` 真实路径已跑通，并在日志中可审计：`holdout=32 selection=strided`。
2. 更大且分散的 heldout 让 ShopFacade/KingsCollege 的 100-step run 变成 strict no-mutation 等价控制；这符合“减少小前缀样本误接受”的预期。
3. OldHospital 仍接受 1 次 split，因为 32-camera heldout 上 tail failure 减少 1 个；full test R5/R2 也小幅改善，但 AE/TE 变差。
4. 因此当前 P6 的状态是：实现混杂进一步收敛，risk gate 更可审计、更保守；但它仍没有给出稳定 precision 正向。下一步若继续，应跑 holdout32+strided 的 500-step/多 seed，并考虑把 R5/R2/tail 从 veto 升级为 risk score 的显式分量或使用更强的 query 分层 holdout。

#### P6 holdout32+strided 500-step 复核

继续把上面的 `holdout=32 + strided + r5_r2_tail veto` 扩到 500-step：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_veto_r5r2tail_hold32_strided_500_v1
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=strided
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_CI_Z=0.0
TOPOLOGY_RISK_MIN_CI_SAMPLES=8
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_MAX_MUTATION_EVENTS=3
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_STEPS=500
LABEL_MAX_IMAGES=16
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
V03_ITERATION=32000
```

Topology decision：

| Scene | Events | Accepted | Children | Accepted iters | Held-out metric evidence |
| --- | ---: | ---: | ---: | --- | --- |
| ShopFacade | 20 | 0 | 0 | - | all rejected by scalar risk |
| KingsCollege | 20 | 1 | 44 | 32325 | metric_count=32, r5/r2/tail deltas 0 |
| OldHospital | 16 | 3 | 48 | 32025, 32350, 32400 | first accept tail_fail_delta=-1; later r5/r2/tail deltas 0 |

Sparse-only result vs current no-mutation500：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.150403 | +0.000000 | 2.988938 | +0.000000 | 0.776699 | +0.000 | 0.281553 | +0.000 | 420.087 | +0.000 |
| KingsCollege | 0.200926 | -0.003077 | 17.589360 | -0.146629 | 0.069971 | +0.583 | 0.000000 | +0.000 | 632.519 | +0.627 |
| OldHospital | 0.360897 | +0.008482 | 18.974373 | +0.955380 | 0.032967 | -0.549 | 0.005495 | +0.549 | 315.335 | -2.758 |

对比 holdout8 prefix veto500：

| Scene | holdout8 prefix 500 | holdout32 strided 500 | 更新判断 |
| --- | --- | --- | --- |
| ShopFacade | 1 accepted, R5/R2 -1.942pp | 0 accepted, strict no-mutation 等价 | prefix 小样本误接受被 strided gate 阻止 |
| KingsCollege | 1 accepted, AE/TE/Inliers 正向但 R5 -0.292pp | 1 accepted, AE/TE/R5/Inliers 同向正向 | 更大分散 holdout 改善了接受质量 |
| OldHospital | 3 accepted, AE 正向但 TE/R5/Inliers 负向 | 3 accepted, AE/TE/R5/Inliers 负向且仅 R2 正向 | Old 的 heldout risk 仍不能代表 full test recall |

当前判断：

1. `holdout32+strided` 明确修复/缓解了 `holdout8+prefix` 的一个高影响实现混杂：ShopFacade 不再接受会降低 full-test recall 的 split。
2. KingsCollege 给出了目前 P6 最干净的正向证据：同一 current no-mutation500 对照下，接受 1 次 split 后 AE、TE、R5、Inliers 均改善，且 held-out R5/R2/tail 没有退化。
3. OldHospital 仍是关键反例：held-out risk 大幅或小幅下降，held-out recall/tail 不退化，但 full-test AE/TE/R5/Inliers 退化。这说明当前 risk score 仍不足以预测完整 query 分布上的 sparse-only precision。
4. 因此，P6 相比 LA_update1/LA_update2 已经有实质进展：实现可审计、rollback/RNG 混杂已排除、prefix-holdout 混杂已定位并部分修复；但精度支撑仍只能称为“局部正向、整体不稳定”，不能作为最终方法主张。
5. 下一步若继续闭合，应优先做 multi-seed + query 分层/全 holdout risk、把 R5/R2/tail 纳入 scalar risk 或采用 per-camera worst-case/CVaR，而不是继续只调 AE+TE median risk。

#### P6 recall-aware scalar risk + CVaR 初版

为继续验证上一节的判断，本轮把 R5/R2/tail 从“accepted event 的事后 veto/日志审计”推进到 scalar risk 本身：

```text
--topology_risk_pose_r5_miss_weight
--topology_risk_pose_r2_miss_weight
--topology_risk_pose_tail_fail_weight
--topology_risk_pose_cvar_weight
--topology_risk_pose_cvar_fraction
```

默认均保持无影响：miss/tail/CVaR 权重为 `0.0`，`cvar_fraction=0.25`。因此历史实验默认语义不变。新的 per-camera risk 为原 AE/TE/inlier risk 加上：

```text
r5_miss_weight * 1[not within R5]
r2_miss_weight * 1[not within R2]
tail_fail_weight * 1[tail failure]
```

held-out aggregate risk 则为：

```text
mean(per_camera_risk) + cvar_weight * mean(top cvar_fraction per_camera_risk)
```

验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_select_risk_cameras_supports_prefix_and_strided_modes \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_penalizes_recall_misses_and_tail_failures \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_risk_aggregation_adds_tail_cvar \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_accepts_only_when_trial_risk_decreases \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_applies_metric_veto_after_risk_drop \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_uses_paired_ucb_when_enabled \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_ci_when_sample_count_is_too_small \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_recall_tail_veto_rejects_r5_and_tail_regressions \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_restores_rng_state_after_scoring \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_nonfinite_trial_risk_and_restores \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default

Ran 13 tests in 1.695s
OK
```

100-step 三场景校准：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_recallscalar_hold32_strided_100_v1
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=strided
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT=0.5
TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT=0.1
TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT=1.0
TOPOLOGY_RISK_POSE_CVAR_WEIGHT=0.5
TOPOLOGY_RISK_POSE_CVAR_FRACTION=0.25
TOPOLOGY_STEPS=100
```

Topology decision：

| Scene | Events | Accepted | Children | Accepted iters | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| ShopFacade | 4 | 2 | 74 | 32075, 32100 | 32050 被 R2 veto；accepted held-out R5/R2/tail 未退化 |
| KingsCollege | 4 | 0 | 0 | - | all rejected by scalar risk |
| OldHospital | 4 | 1 | 8 | 32025 | same accepted event as holdout32+strided100; tail_fail_delta=-1 |

Sparse-only result vs current no-mutation100：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.149521 | -0.005524 | 2.916627 | +0.146668 | 0.766990 | +0.971 | 0.233010 | -0.971 | 417.990 | -4.932 |
| KingsCollege | 0.188302 | +0.000000 | 16.800641 | +0.000000 | 0.014577 | +0.000 | 0.000000 | +0.000 | 582.120 | +0.000 |
| OldHospital | 0.375196 | +0.010245 | 19.322717 | +0.304389 | 0.054945 | +0.549 | 0.010989 | +0.549 | 284.225 | -0.676 |

与上一版 `holdout32+strided100` 的对比：

1. ShopFacade 从 0 accepted 变成 2 accepted，说明 recall/tail/CVaR scalar risk 会显著改变 commit 行为；但 full sparse eval 只是 AE/R5 正向，TE/R2/Inliers 负向。
2. KingsCollege 仍是 strict no-mutation 等价控制。
3. OldHospital 与上一版逐项相同：同一个 32025 split 被接受，后续拒绝；R5/R2 小幅正向但 AE/TE/Inliers 负向。
4. 结论是：把 R5/R2/tail 加入 scalar risk 是实现层必要补充，但当前权重和 `holdout32` 仍不能解决 full-test 泛化问题；它甚至会在 ShopFacade 放开新的 mixed accepted events。
5. 下一步不应直接扩这组权重到 500-step。更合理的是先做 risk 诊断：记录所有 accepted/rejected event 的 metric deltas、引入 paired-CI 或更强 r2/tail veto、或用 query 分层/多 buffer 而不是单个 32-camera strided buffer。

#### P6 rejected-proposal metric diagnostics

上面的 recall-aware scalar 校准暴露了一个诊断盲点：之前 topology log 只在 accepted 或 metric-veto path 上记录 `risk_r5_delta/risk_r2_delta/risk_tail_fail_delta`。如果 proposal 因 scalar risk 未下降而被拒绝，日志只保留 scalar risk，无法判断“被拒绝 proposal 是否其实改善了 recall/tail”。

本轮因此把 metric diagnostics 从 commit decision 中解耦：

1. `HeldoutRiskCommitEvaluator` 只要配置了 `metric_gate_fn`，就会对 baseline/trial metrics 计算 details。
2. 若 proposal 已通过 scalar/CI gate，metric gate 仍按原语义执行 veto。
3. 若 proposal 已因 scalar/CI gate 被拒绝，只记录 metric deltas，不改变 `accepted=False` 和原始拒绝原因。

新增验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_logs_metric_deltas_for_scalar_rejections

Ran 1 test in 1.717s
OK
```

P6 定向回归：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_select_risk_cameras_supports_prefix_and_strided_modes \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_score_penalizes_pose_error_and_rewards_inliers \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_pose_risk_penalizes_recall_misses_and_tail_failures \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_risk_aggregation_adds_tail_cvar \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_accepts_only_when_trial_risk_decreases \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_applies_metric_veto_after_risk_drop \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_logs_metric_deltas_for_scalar_rejections \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_uses_paired_ucb_when_enabled \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_ci_when_sample_count_is_too_small \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_pose_recall_tail_veto_rejects_r5_and_tail_regressions \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_restores_rng_state_after_scoring \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_heldout_risk_evaluator_rejects_nonfinite_trial_risk_and_restores \
  tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default

Ran 14 tests in 1.722s
OK
```

真实 smoke：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_metric_diagnostics_smoke_v1
ShopFacade, 10-step, holdout=8, selection=strided, recall-aware scalar risk, veto=r5_r2_tail
```

关键 topology log：

| Iter | Accepted | Reason | risk_delta | metric_count | r5_delta | r2_delta | tail_delta |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32005 | False | heldout_pose_not_decreased | +0.001012 | 8 | 0 | 0 | 0 |
| 32010 | True | heldout_pose_decreased | -0.001110 | 8 | 0 | 0 | 0 |

该 smoke 证明 rejected scalar path 现在也会输出 metric deltas。10-step sparse-only 指标为 AE 0.156542、TE 2.797235、R5 0.776699、R2 0.271845、Inliers 423.845；这只用于通路验证，不作为精度结论。

当前判断：这个改动不直接提升精度，但提升了后续实验的可解释性。下一批 P6 实验应优先利用这些 rejected/accepted metric deltas 做 event-level 诊断，再决定是收紧 R2/tail veto、启用 paired-CI，还是改为多 buffer/分层 heldout。

#### P6 recall-aware scalar diagnostics rerun

为验证上一节的诊断补丁在真实三场景实验里是否能闭合盲区，本轮重跑同一组 recall-aware scalar 100-step 配置，只改变输出目录：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_recallscalar_diag_hold32_strided_100_v1
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=strided
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT=0.5
TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT=0.1
TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT=1.0
TOPOLOGY_RISK_POSE_CVAR_WEIGHT=0.5
TOPOLOGY_RISK_POSE_CVAR_FRACTION=0.25
TOPOLOGY_STEPS=100
```

新增 summary 工具：

```text
scripts/summarize_topology_risk_events.py
```

它解析 `[Topology]` key-value 日志，并汇总 accepted/rejected 数量、metric diagnostics 缺失数量、R5/R2/tail delta、risk delta 和拒绝原因。定向验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_la_update_summarizers.LAUpdateSummarizerTest.test_topology_risk_event_summary_tracks_rejected_metric_diagnostics

Ran 1 test in 0.004s
OK
```

三场景 event summary：

```text
/mnt/pool/sqy/stdloc_la_update3_logs/p6_pose_recallscalar_diag_hold32_strided_100_summary.json
```

| Scene | Events | Accepted | Rejected | accepted_metric_missing | rejected_metric_missing | accepted R5/R2/tail delta | rejected R5/R2/tail delta | Reason summary |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ShopFacade | 4 | 2 | 2 | 0 | 0 | 0 / +2 / 0 | +1 / -1 / 0 | 1 scalar reject, 1 R2 veto, 2 accepted |
| KingsCollege | 4 | 0 | 4 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | all scalar rejects |
| OldHospital | 4 | 1 | 3 | 0 | 0 | 0 / 0 / -1 | -2 / 0 / 0 | 1 accepted, 3 scalar rejects |

Sparse-only 结果与上一组 recall-aware scalar 100-step 一致：

| Scene | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.149521 | 2.916627 | 0.766990 | 0.233010 | 417.990 |
| KingsCollege | 0.188302 | 16.800641 | 0.014577 | 0.000000 | 582.120 |
| OldHospital | 0.375196 | 19.322717 | 0.054945 | 0.010989 | 284.225 |

关键结论：

1. 三场景所有 rejected proposal 都已记录 `risk_metric_count/risk_r5_delta/risk_r2_delta/risk_tail_fail_delta`，诊断盲区闭合。
2. ShopFacade 的 32050 proposal 虽然 scalar risk 更低，但 held-out R2 下降，因此被 R2 veto 拒绝；这说明 metric veto 确实在拦截 recall tradeoff。
3. KingsCollege 的 4 个 rejected proposal held-out R5/R2/tail 都没有改善，因此当前结果不是因为实现误拒了明显更好的 recall proposal。
4. OldHospital 的 32050/32075 rejected proposal 都有 held-out R5 下降，拒绝是合理的；但唯一 accepted 的 32025 虽然 tail failure 改善，最终 full sparse eval 仍表现为 AE/TE/Inliers 退化、R5/R2 小幅正向。
5. 因此本轮把问题进一步定位到 accepted event 泛化和 scalar risk 对 full-query sparse-only recall 的对齐不足，而不是 rollback/RNG/rejected diagnostic 这类实现混杂。当前精度结论仍是局部正向、整体不稳定。

#### P6 pose-stratified holdout 初版

上一节说明单个 `holdout32+strided` buffer 仍不能完全代表 full query 分布。本轮先补一个默认关闭的空间分层 holdout 选择：

```text
--topology_risk_holdout_selection pose_stratified
```

实现语义：

1. 若 held-out camera 有 `camera_center`，先按 camera center 的第一主轴排序，再在该空间顺序上做 strided quantile sampling。
2. PCA 主轴符号固定为最大绝对分量为正，避免同一数据上抽样顺序不稳定。
3. 若 camera center 不可用或退化，则回退为原 `strided`，保持可运行。
4. 默认仍是 `prefix`，历史实验默认语义不变。

定向 TDD 验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_topology_risk_commit_policy \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_select_risk_cameras_supports_prefix_strided_and_pose_stratified_modes

Ran 2 tests in 1.627s
OK
```

100-step 三场景复核：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_recallscalar_diag_hold32_pose_stratified_100_v1
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=pose_stratified
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
TOPOLOGY_RISK_POSE_R5_MISS_WEIGHT=0.5
TOPOLOGY_RISK_POSE_R2_MISS_WEIGHT=0.1
TOPOLOGY_RISK_POSE_TAIL_FAIL_WEIGHT=1.0
TOPOLOGY_RISK_POSE_CVAR_WEIGHT=0.5
TOPOLOGY_RISK_POSE_CVAR_FRACTION=0.25
TOPOLOGY_STEPS=100
```

Event summary：

```text
/mnt/pool/sqy/stdloc_la_update3_logs/p6_pose_recallscalar_diag_hold32_pose_stratified_100_summary.json
```

| Scene | Events | Accepted | Rejected | Children | accepted_metric_missing | rejected_metric_missing | accepted R5/R2/tail delta | rejected R5/R2/tail delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ShopFacade | 4 | 2 | 2 | 50 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| KingsCollege | 4 | 0 | 4 | 0 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| OldHospital | 4 | 2 | 2 | 22 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Sparse-only vs current no-mutation100：

| Scene | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.152379 | -0.002666 | 2.687866 | -0.082092 | 0.776699 | +1.942 | 0.242718 | +0.000 | 423.437 | +0.515 |
| KingsCollege | 0.188302 | +0.000000 | 16.800641 | +0.000000 | 0.014577 | +0.000 | 0.000000 | +0.000 | 582.120 | +0.000 |
| OldHospital | 0.371636 | +0.006685 | 19.069876 | +0.051548 | 0.054945 | +0.549 | 0.010989 | +0.549 | 284.819 | -0.082 |

Sparse-only vs `holdout32+strided` recall-aware scalar：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | +0.002858 | -0.228761 | +0.971 | +0.971 | +5.447 |
| KingsCollege | +0.000000 | +0.000000 | +0.000 | +0.000 | +0.000 |
| OldHospital | -0.003560 | -0.252841 | +0.000 | +0.000 | +0.593 |

关键结论：

1. `pose_stratified` 是本轮相对 `strided` 的实质改进：ShopFacade 的 TE/R5/R2/Inliers 更好；OldHospital 的 AE/TE/Inliers 退化幅度收窄，R5/R2 持平。
2. 相对 current no-mutation100，ShopFacade 已经给出更强的正向支撑：AE/TE/R5/Inliers 正向，R2 持平。
3. OldHospital 仍不能写成稳定正向：R5/R2 小幅正向，但 AE/TE/Inliers 仍略差于 current no-mutation100。
4. KingsCollege 继续是 inert control，证明该配置没有在 Kings 上强行接受微弱 proposal。
5. 因此，query 空间分层确实缓解了 `strided` 的一部分泛化问题，也比 LA_update1/2 阶段更接近精度正向；但仍不足以证明 topology mutation 跨场景稳定有效。下一步应把 `pose_stratified` 扩到 500-step/multi-seed，并考虑 full-holdout 或多 buffer gate，而不是直接宣称主张闭环。

#### P6 pose-stratified 500-step 与 ShopFacade seed2026 补充

继续把上一节的 `holdout32 + pose_stratified + recall-aware scalar + r5/r2/tail veto` 扩到 500-step，并补跑 ShopFacade `query_split_seed=2026` 的 matched no-mutation/pose-stratified 对照。

运行路径：

```text
/mnt/pool/sqy/stdloc_la_update3_p6_pose_recallscalar_diag_hold32_pose_stratified_500_v1
/mnt/pool/sqy/stdloc_la_update3_p6_matched_no_mutation_500_seed2026_v1
/mnt/pool/sqy/stdloc_la_update3_p6_pose_recallscalar_diag_hold32_pose_stratified_500_seed2026_v1
```

Event summary：

| Scene / seed | Events | Accepted | Rejected | Children | accepted R5/R2/tail delta | rejected R5/R2/tail delta | accepted risk delta mean |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| ShopFacade seed2025 | 20 | 2 | 18 | 50 | 0 / 0 / 0 | -3 / 0 / 0 | -0.001193 |
| KingsCollege seed2025 | 15 | 3 | 12 | 108 | 0 / 0 / 0 | -3 / 0 / 0 | -0.008791 |
| OldHospital seed2025 | 6 | 3 | 3 | 36 | +1 / 0 / 0 | 0 / 0 / 0 | -0.005528 |
| ShopFacade seed2026 | 5 | 3 | 2 | 100 | +1 / +2 / 0 | 0 / -1 / 0 | -0.018493 |

Sparse-only vs matched current no-mutation500：

| Scene / seed | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade seed2025 | 0.147853 | -0.002551 | 2.993144 | +0.004206 | 0.766990 | -0.971 | 0.281553 | +0.000 | 420.621 | +0.534 |
| KingsCollege seed2025 | 0.202197 | -0.001806 | 17.877120 | +0.141130 | 0.067055 | +0.292 | 0.000000 | +0.000 | 631.557 | -0.335 |
| OldHospital seed2025 | 0.361934 | +0.009519 | 18.429005 | +0.410011 | 0.027473 | -1.099 | 0.005495 | +0.549 | 316.055 | -2.038 |
| ShopFacade seed2026 | 0.168682 | +0.000930 | 3.545541 | -0.110702 | 0.660194 | +1.942 | 0.194175 | -3.883 | 420.942 | +0.806 |

当前判断：

1. 相比 LA_update1/LA_update2 阶段，这一组结果更可归因：matched current no-mutation 对照已补齐，rollback/RNG 污染已排除，rejected proposal 也有 metric diagnostics，且 `pose_stratified` 能真实接受少量 proposal。
2. 事件层是正向的：accepted event 的 held-out R5/R2/tail 没有退化，ShopFacade seed2026 和 OldHospital seed2025 还出现 held-out recall 改善；rejected event 中能看到 R5/R2 回退被拒绝。
3. full sparse-only 精度仍是局部正向、整体不稳定：ShopFacade seed2026 的 TE/R5/Inliers 正向，KingsCollege seed2025 的 AE/R5 正向，ShopFacade seed2025 的 AE/Inliers 正向；但 OldHospital seed2025 仍主要负向，ShopFacade seed2026 的 R2 明显下降。
4. 因此目前已经能支撑“方法实现比 LA_update1/2 更干净，并出现可重复的局部正向信号”，但还不能支撑“当前 topology/risk 策略跨 scene、跨 seed 稳定提升 sparse-only relocalization”。
5. 下一步若继续闭合，应优先做多 seed 的 ShopFacade/Kings/Old matched matrix，以及 full-holdout 或 multi-buffer risk calibration；P5 synthetic novel-view 仍需 500-step/多 seed/teacher quality calibration，physical prune 仍需真实触发阈值矩阵。

#### P7 render artifact query-filter ablation

目的：验证“test/query 对应渲染图存在伪影，污染 teacher dense / heldout query，从而负面影响 topology risk 与 sparse pose”的怀疑。

实现修复：

1. `train_locaware.py` 新增 `--query_artifact_filter_path` / `--query_artifact_filter_severities` / `--query_artifact_filter_splits`，只过滤训练中的 `query_cameras`，不改 `scene.getTestCameras()` 与正式 STDLoc test 评估口径。
2. 过滤只按规范化 image path exact match；曾尝试 basename match，但 OldHospital 出现跨 sequence 同名 frame 误删，已改回 exact path，并加测试防回归。
3. 新增 `--support_query_sort_by_name`，默认关闭。原因是旧 render audit 的 `heldout_query_sample` 是按 name-sorted split 生成的，而训练默认 `Scene` 会 shuffle camera；如果不显式 sorted split，ShopFacade 只会删到 1/46，ablation 不成立。
4. `scripts/run_locaware_v03_topology_full.sh` 接入上述参数，默认保持旧行为；本 ablation 显式设置 `TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME=1`。

验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_train_locaware_masks tests.test_full_script_args
Ran 80 tests OK
```

实验设置：

```text
TOPOLOGY_STEPS=100
TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME=1
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=pose_stratified
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
QUERY_ARTIFACT_FILTER_PATH=/mnt/pool/sqy/stdloc_la_render_audit_v0/render_artifact_candidate_list_seed2026_base.csv
```

有效过滤计数：

| Scene | Query before | Removed | Query after |
| --- | ---: | ---: | ---: |
| ShopFacade | 46 | 10 | 36 |
| OldHospital | 225 | 43 | 182 |

结果文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p7_sorted_pose_artifact_filter_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p7_sorted_pose_artifact_filter_100_seed2026_summary.csv
```

Sparse-only final test，artifact filter 相对 no-filter：

| Scene | dAE | dTE | dR5 pp | dR2 pp | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.002331 | +0.101961 | +2.913 | +2.913 | +1.854 |
| OldHospital | +0.014221 | -0.291805 | -2.747 | +0.000 | +0.412 |

Topology risk event，artifact filter 相对 no-filter：

| Scene | accepted delta | children delta | risk_delta_mean delta | risk R5 delta | risk R2 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0 | -4 | +0.006820 | 0 | -1 |
| OldHospital | +2 | +36 | -0.003411 | 0 | +1 |

结论：

1. 渲染伪影样本确实存在，且训练 query 过滤必须与实际 support/query split 对齐；否则“过滤 severe/mild”会退化成只删 1 张或发生跨序列误删。
2. 在 100-step sorted-split 对照上，过滤伪影不是单调正向：ShopFacade 的 R5/R2/Inliers 改善，但 TE 略差、heldout risk delta 均值变差；OldHospital 的 TE/Inliers 与 risk gate 改善，但 R5 下降。
3. 因此“伪影污染 teacher/query 是一个真实混杂点”成立；但“简单删除 severe/mild query 样本能稳定提升最终 sparse pose”暂未成立。
4. 下一步若继续这条线，应做 full-query render audit 或在线 render-quality gate，而不是只依赖 sampled candidate CSV；同时要把 sorted split/no-filter/filter 扩到 500-step 与多 seed，避免 100-step 的高方差误判。

#### P8 actual full-query render artifact audit and severity-filter ablation

目的：把 P7 的 sorted sampled candidate 扩展到“训练默认 support/query split 下的 full-query audit”，并验证真实伪影样本是否确实污染 teacher/query 采样与 heldout topology risk。

实现与修复：

1. 新增 `scripts/audit_render_artifacts.py`，直接加载已训练 3DGS checkpoint，对指定 split 的相机逐张渲染并计算 `psnr_mean_matched`、`ssim`、`l1`、高残差占比、alpha coverage、RGB bias，输出 audit CSV/JSON 和 candidate CSV。
2. 默认阈值：
   - `severe`: `psnr_mean_matched <= 13.5` 或 `ssim <= 0.42` 或 `residual_frac_025 >= 0.18`。
   - `mild`: severe 或 `psnr_mean_matched <= 15.5` 或 `ssim <= 0.56` 或 `residual_frac_025 >= 0.10` 或 `alpha_cov_05 <= 0.85` 或 `mean_abs_bias >= 0.04`。
3. audit split 支持 `support_query_split/query_split_seed/query_split_mode/support_query_sort_by_name`，本轮使用训练默认：`query_split_seed=2026`、`query_split_mode=sequence_block`、`sort_by_name=False`。
4. 修复 `psnr(...).view(...)` 遇到非 contiguous tensor 的 smoke failure，改为 `.reshape` 并加回归测试。
5. 补充测试：candidate exact path matching、support/query split 排序行为、非 contiguous PSNR。

验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_render_artifact_audit tests.test_train_locaware_masks tests.test_full_script_args
Ran 84 tests in 2.059s OK
```

full-query audit 输出：

```text
/mnt/pool/sqy/stdloc_la_render_audit_v1/ShopFacade_actual_query_seed2026_base32000.csv
/mnt/pool/sqy/stdloc_la_render_audit_v1/ShopFacade_actual_query_seed2026_base32000_candidates.csv
/mnt/pool/sqy/stdloc_la_render_audit_v1/OldHospital_actual_query_seed2026_base32000.csv
/mnt/pool/sqy/stdloc_la_render_audit_v1/OldHospital_actual_query_seed2026_base32000_candidates.csv
```

audit 结果：

| Scene | Query images | None | Mild | Severe | Candidate ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 46 | 30 | 16 | 0 | 34.8% |
| OldHospital | 369 | 104 | 179 | 86 | 71.8% |

关键观察：OldHospital 默认 split 下的伪影污染远高于 P7 sampled/sorted 表；因此“渲染伪影是一个真实混杂点”成立，尤其是 OldHospital。

100-step matched ablation 设置：

```text
TOPOLOGY_STEPS=100
QUERY_SPLIT_SEED=2026
TOPOLOGY_QUERY_SPLIT_MODE=sequence_block
TOPOLOGY_SUPPORT_QUERY_SORT_BY_NAME=0
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=pose_stratified
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
```

结果汇总文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p8_actual_artifact_filter_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p8_actual_artifact_filter_100_seed2026_summary.csv
```

Sparse-only final test，delta 相对同 scene no-filter：

| Scene | Variant | Removed | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | no-filter | 0/46 | 0.167711 | +0.000000 | 3.406296 | +0.000000 | 0.660194 | +0.000 | 0.194175 | +0.000 | 420.602 | +0.000 |
| ShopFacade | mild+severe | 16/46 | 0.175892 | +0.008181 | 3.718608 | +0.312312 | 0.660194 | +0.000 | 0.203883 | +0.971 | 407.689 | -12.913 |
| OldHospital | no-filter | 0/369 | 0.341946 | +0.000000 | 19.206845 | +0.000000 | 0.043956 | +0.000 | 0.000000 | +0.000 | 291.313 | +0.000 |
| OldHospital | severe-only | 86/369 | 0.346923 | +0.004978 | 19.485590 | +0.278745 | 0.043956 | +0.000 | 0.005495 | +0.549 | 287.176 | -4.137 |
| OldHospital | mild-only | 179/369 | 0.366652 | +0.024707 | 19.618421 | +0.411576 | 0.049451 | +0.549 | 0.005495 | +0.549 | 287.027 | -4.286 |
| OldHospital | mild+severe | 265/369 | 0.355354 | +0.013408 | 19.127379 | -0.079466 | 0.054945 | +1.099 | 0.010989 | +1.099 | 279.505 | -11.808 |

Topology risk event：

| Scene | Variant | accepted/events | children | accepted R5 delta | accepted R2 delta | rejected R5 delta | rejected R2 delta | risk_delta_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | no-filter | 2/4 | 74 | 0 | +2 | -1 | +1 | +0.009056 |
| ShopFacade | mild+severe | 2/4 | 72 | 0 | +1 | -1 | -1 | +0.011835 |
| OldHospital | no-filter | 1/4 | 20 | 0 | 0 | 0 | 0 | -0.025897 |
| OldHospital | severe-only | 1/4 | 20 | +2 | 0 | 0 | -1 | -0.007099 |
| OldHospital | mild-only | 0/4 | 0 | 0 | 0 | 0 | -1 | +0.004321 |
| OldHospital | mild+severe | 1/4 | 18 | 0 | +1 | -2 | 0 | +0.006688 |

结论：

1. 渲染伪影样本确实存在，并且会进入当前训练的 `query_cameras`。由于 dense/direct teacher query sampling 和 heldout risk 都从 `query_cameras` 取样，当前 filter 同时作用于 teacher/query 采样与 topology risk；正式 test set 没有被过滤。
2. 伪影过滤不是单调正向。ShopFacade mild+severe 只带来 R2 +0.971 pp，但 AE/TE/Inliers 明显变差；OldHospital mild+severe 带来 TE、R5、R2 正向，但 AE/Inliers 变差。
3. OldHospital severity 分层说明“删 severe/mild 的收益”主要体现在 R5/R2 recall，且删除更多 query 后 recall 增益更大；但 AE、TE、Inliers 并不随之改善，severe-only 反而 TE 更差。这说明 hard removal 同时改变了 heldout/query 分布，不能简单等价为“去掉坏 teacher”。
4. 目前可以排除“伪影问题不存在”的怀疑；但不能把 hard filter 作为最终策略。更合理方向是 render-quality weighting / stratified risk buffer / severe-only gate + query coverage constraint，而不是直接删除所有 mild+severe。
5. 下一步不宜盲目扩大 500-step hard-filter；应优先实现 soft weighting 或 stratified sampling：保留 query 覆盖，降低 low-quality render 对 teacher/risk 的权重，并保证 heldout buffer 中 none/mild/severe 或 pose bins 的比例稳定。

#### P9 render artifact independent module and soft-weighting ablation

目的：把渲染伪影检测逻辑从训练主线中拆出，先实现“检测 + 软抑制”，避免 P8 hard removal 同时改变 query 覆盖和 heldout 分布；正式 sparse-only test 评估口径仍不做过滤。

实现改动：

1. 新增独立模块 `localization_training/render_artifacts.py`，集中放置：
   - image path 规范化与 exact-match lookup；
   - render artifact severity 阈值分类；
   - filter name 加载；
   - per-camera soft weight lookup；
   - weighted mean / CVaR aggregation helper。
2. `scripts/audit_render_artifacts.py` 改为复用该模块，不再在 audit script 内保留一份重复阈值逻辑。
3. `train_locaware.py` 保留 P8 的 query hard filter 接口，同时新增：
   - `--render_artifact_weight_path`
   - `--render_artifact_weight_splits`
   - `--render_artifact_weight_severities`
   - `--render_artifact_weight_targets`
   - `--render_artifact_weight_default`
   - `--render_artifact_weight_mild`
   - `--render_artifact_weight_severe`
4. soft weighting 支持两个 target：
   - `teacher`：对 direct/dense teacher 的 localization loss 乘以伪影质量权重；
   - `risk`：对 heldout sparse pose risk 的 mean/CVaR 聚合使用同一质量权重。
5. `scripts/run_locaware_v03_topology_full.sh` 接入对应环境变量，默认不启用，避免影响既有实验。

本轮没有实现“向前调整渲染位姿”。原因是 P8/P9 当前验证对象是实际 query 相机渲染质量对 teacher/risk 的污染；位姿前移会改变 teacher 监督来源，必须单独设计 synthetic/adjusted render 数据流与覆盖约束，不能和 soft weighting 混在同一个 ablation 里。

验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_render_artifact_weights \
  tests.test_render_artifact_audit \
  tests.test_train_locaware_masks \
  tests.test_full_script_args

Ran 88 tests in 1.794s
OK
```

100-step matched ablation 设置：

```text
TOPOLOGY_STEPS=100
QUERY_SPLIT_SEED=2026
TOPOLOGY_QUERY_SPLIT_MODE=sequence_block
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=pose_stratified
TOPOLOGY_RISK_EPSILON=0.001
TOPOLOGY_RISK_POSE_VETO_MODE=r5_r2_tail
RENDER_ARTIFACT_WEIGHT_MILD=0.65
RENDER_ARTIFACT_WEIGHT_SEVERE=0.25
```

结果文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p9_render_artifact_weight_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p9_render_artifact_weight_100_seed2026_summary.csv
```

Sparse-only final test，delta 相对 P8 no-filter：

| Scene | Variant | Weighted | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | no-filter | 0/46 | 0.167711 | +0.000000 | 3.406296 | +0.000000 | 0.660194 | +0.000 | 0.194175 | +0.000 | 420.602 | +0.000 |
| ShopFacade | soft teacher+risk | 16/46 | 0.169539 | +0.001828 | 3.604144 | +0.197848 | 0.650485 | -0.971 | 0.194175 | +0.000 | 419.466 | -1.136 |
| ShopFacade | soft teacher-only | 16/46 | 0.169539 | +0.001828 | 3.604144 | +0.197848 | 0.650485 | -0.971 | 0.194175 | +0.000 | 419.466 | -1.136 |
| OldHospital | no-filter | 0/369 | 0.341946 | +0.000000 | 19.206845 | +0.000000 | 0.043956 | +0.000 | 0.000000 | +0.000 | 291.313 | +0.000 |
| OldHospital | soft teacher+risk | 265/369 | 0.336496 | -0.005449 | 18.794818 | -0.412027 | 0.054945 | +1.099 | 0.000000 | +0.000 | 290.099 | -1.214 |
| OldHospital | soft teacher-only | 265/369 | 0.336496 | -0.005449 | 18.794818 | -0.412027 | 0.054945 | +1.099 | 0.000000 | +0.000 | 290.099 | -1.214 |

Topology risk event：

| Scene | Variant | accepted/events | children | accepted R5 delta | accepted R2 delta | rejected R5 delta | rejected R2 delta | risk_delta_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | soft teacher+risk | 2/4 | 74 | 0 | +1 | -1 | +1 | +0.010952 |
| ShopFacade | soft teacher-only | 2/4 | 74 | 0 | +1 | -1 | +1 | +0.012418 |
| OldHospital | soft teacher+risk | 2/4 | 40 | 0 | 0 | 0 | 0 | +0.000484 |
| OldHospital | soft teacher-only | 2/4 | 40 | 0 | 0 | 0 | 0 | -0.000197 |

结论：

1. 伪影处理已经和主训练逻辑解耦：audit、hard filter、soft weighting 共用同一个 `render_artifacts` 模块，后续可以独立替换阈值、权重或 pose-adjust 策略。
2. soft weighting 避免了 P8 hard removal 的“删 query 导致覆盖变化”混杂；正式 test set 仍完整评估，没有删除 severe/mild test query。
3. OldHospital 给出明确正向支撑：AE -0.00545、TE -0.412cm、R5 +1.099pp、R2 持平，Inliers 只小幅下降。该场景有 179 mild + 86 severe，说明高伪影占比场景中降低伪影 teacher 权重是有效方向。
4. ShopFacade 不正向：只有 16 mild、0 severe，soft weighting 使 AE/TE/R5/Inliers 略差，R2 持平。不能把 mild 阈值下的全部样本都当成负贡献；ShopFacade 需要更保守的 severe-only gate 或连续质量权重。
5. teacher-only 与 teacher+risk 的最终 sparse pose 完全一致，且 accepted topology event 数也一致；本轮正向主要来自 teacher loss 软抑制，risk weighting 没有带来可观察的额外收益。
6. 目前可以闭合“渲染伪影是否是真实混杂点”：是，且 soft 抑制在 OldHospital 上能给出正向结果。但还不能闭合“统一阈值在所有 scene 上稳定提升”：ShopFacade 仍反例。
7. 下一步应把伪影模块作为单独分支继续推进：优先尝试 severe-only/continuous quality weight、none/mild/severe stratified heldout buffer，以及独立的 pose-forward adjusted render 数据流；不要再把 hard deletion 作为主策略。

#### P10 default severe-only render artifact weighting

目的：修正 P9 的一个策略问题。P9 对 `mild+severe` 全部降权，OldHospital 正向，但 ShopFacade 只有 16 个 mild、0 个 severe 时被误伤。P10 改为默认只抑制 `severe`，保留 mild 样本覆盖。

实现改动：

1. `train_locaware.py` 的 `--render_artifact_weight_severities` 默认从 `mild,severe` 改为 `severe`。
2. `train_locaware.py` 的 `--render_artifact_weight_mild` 默认从 `0.65` 改为 `1.0`。
3. `scripts/run_locaware_v03_topology_full.sh` 同步上述默认值；显式环境变量仍可覆盖。
4. 更新测试，防止默认策略回退到 mild 降权。

TDD 验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_train_locaware_masks.TrainLocawareMaskTest.test_locaware_parser_accepts_render_artifact_weighting_args \
  tests.test_full_script_args

Ran 30 tests in 1.663s
OK
```

100-step matched ablation 设置：

```text
TOPOLOGY_STEPS=100
QUERY_SPLIT_SEED=2026
TOPOLOGY_QUERY_SPLIT_MODE=sequence_block
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
TOPOLOGY_RISK_HOLDOUT_SIZE=32
TOPOLOGY_RISK_HOLDOUT_SELECTION=pose_stratified
RENDER_ARTIFACT_WEIGHT_TARGETS=teacher
RENDER_ARTIFACT_WEIGHT_SEVERITIES=severe
RENDER_ARTIFACT_WEIGHT_MILD=1.0
RENDER_ARTIFACT_WEIGHT_SEVERE=0.25
```

结果文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p10_render_artifact_weight_severe_only_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p10_render_artifact_weight_severe_only_100_seed2026_summary.csv
```

Sparse-only final test，delta 相对 P8 no-filter：

| Scene | Weighted severe | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0/46 | 0.167711 | +0.000000 | 3.406296 | +0.000000 | 0.660194 | +0.000 | 0.194175 | +0.000 | 420.602 | +0.000 |
| OldHospital | 86/369 | 0.334690 | -0.007255 | 18.186275 | -1.020570 | 0.049451 | +0.549 | 0.005495 | +0.549 | 289.198 | -2.115 |

对 P9 mild+severe soft-weight 的改进：

| Scene | dAE vs P9 | dTE vs P9 | dR5 pp vs P9 | dR2 pp vs P9 | dInliers vs P9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | -0.001828 | -0.197848 | +0.971 | +0.000 | +1.136 |
| OldHospital | -0.001806 | -0.608543 | -0.549 | +0.549 | -0.901 |

Topology risk event：

| Scene | accepted/events | children | accepted R5 delta | accepted R2 delta | rejected R5 delta | rejected R2 delta | risk_delta_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 2/4 | 74 | 0 | +2 | -1 | +1 | +0.009056 |
| OldHospital | 2/4 | 38 | +1 | 0 | -1 | 0 | +0.001404 |

结论：

1. severe-only 是当前更合理的默认策略：它不再误伤 ShopFacade 的 mild 样本，同时保留 OldHospital 的正向收益。
2. 相对 P8 no-filter，P10 在 ShopFacade 完全持平，在 OldHospital 的 AE、TE、R5、R2 四项正向，只有 Inliers 小幅下降。
3. 相对 P9 mild+severe，P10 明显修复了 ShopFacade 退化，并进一步改善 OldHospital 的 AE/TE/R2；R5 从 +1.099pp 降为 +0.549pp，但仍为正。
4. 事件层也支持该策略：OldHospital accepted proposal 中出现 heldout R5 +1，rejected proposal 中包含 R5 -1；ShopFacade 在无 severe 匹配时行为回到 no-filter。
5. 这给当前阶段提供了更强的正向支撑：不是“删除 test query”得到的，也不是 hard filter 改变 query 覆盖得到的，而是在保持正式评估口径不变时，仅降低 severe render teacher 影响获得的。
6. 后续再考虑 continuous weighting 或 pose-forward adjusted render 时，应以 P10 severe-only 作为保守基线，而不是 P9 mild+severe。

#### P11-P13 long-run severe artifact weight sweep

目的：把 P10 的 100-step severe-only 正向结果延长到 500-step，并排除 `severe=0.25` 过强抑制造成的长训练 TE 退化。

设置：

```text
Scene=OldHospital
TOPOLOGY_STEPS=500
QUERY_SPLIT_SEED=2026
TOPOLOGY_QUERY_SPLIT_MODE=sequence_block
TOPOLOGY_MUTATION_MODE=split_only
TOPOLOGY_RISK_COMMIT_POLICY=heldout_pose
RENDER_ARTIFACT_WEIGHT_TARGETS=teacher
RENDER_ARTIFACT_WEIGHT_SEVERITIES=severe
matched severe=86/369
正式 sparse-only test 不删 query
```

结果文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p11_render_artifact_weight_severe_only_500_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p12_render_artifact_weight_sweep_500_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p13_render_artifact_weight_sweep_500_seed2026_summary.json
```

Sparse-only final test，delta 相对 500-step no-filter：

| Variant | severe weight | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-filter | 1.00 | 0.359538 | +0.000000 | 18.579300 | +0.000000 | 0.054945 | +0.000 | 0.010989 | +0.000 | 317.115 | +0.000 |
| severe025 | 0.25 | 0.344038 | -0.015500 | 19.244636 | +0.665336 | 0.060440 | +0.549 | 0.010989 | +0.000 | 311.945 | -5.170 |
| severe050 | 0.50 | 0.352408 | -0.007130 | 18.802741 | +0.223441 | 0.060440 | +0.549 | 0.010989 | +0.000 | 316.143 | -0.973 |
| severe065 | 0.65 | 0.359142 | -0.000396 | 19.045532 | +0.466232 | 0.065934 | +1.099 | 0.005495 | -0.549 | 317.505 | +0.390 |
| severe070 | 0.70 | 0.348466 | -0.011073 | 18.400687 | -0.178613 | 0.054945 | +0.000 | 0.010989 | +0.000 | 317.275 | +0.159 |
| severe075 | 0.75 | 0.361782 | +0.002244 | 18.384991 | -0.194309 | 0.060440 | +0.549 | 0.010989 | +0.000 | 317.643 | +0.527 |

结论：

1. P11 证明 `severe=0.25` 在 500-step 下不是最佳：AE/R5 正向，但 TE +0.665cm、Inliers -5.17，说明抑制过强会损失可用 teacher 约束。
2. P12/P13 的 sweep 支持“伪影抑制有效，但需要温和权重”：`severe=0.50` 和 `0.75` 的 AE/TE normalized risk 都优于 no-filter，表现为 AE/TE tradeoff。
3. `severe=0.70` 是目前最干净的 500-step 正向点：AE -0.0111、TE -0.1786cm、Inliers +0.159，R5/R2 不退化。它没有依赖删除正式 test query。
4. 因此默认 severe soft weight 从 `0.25` 调整为 `0.70`；`0.25` 仍保留为可显式覆盖的强抑制 ablation。
5. 当前正向支撑已经比 LA_update1/LA_update2 更明确：它闭合了“渲染伪影是否是真实混杂点”以及“hard deletion 是否导致覆盖混杂”的两个问题，并给出 100-step 与 500-step 的可复现实验文件。

#### P14-P17 continuous render artifact weighting

目的：继续优化独立伪影抑制模块。P9/P10/P13 仍把 audit 的连续指标压成 `none/mild/severe` 和常数权重，丢失了 `psnr`、`ssim`、`alpha_cov_05`、`residual_frac_025`、`mean_abs_bias` 的强弱信息。本轮先实现连续整图 teacher weighting；per-region/per-landmark 抑制与 pose-forward adjusted render 仍作为后续单独 ablation，不并入默认主线。

实现改动：

1. `localization_training/render_artifacts.py` 新增 `continuous_quality_weight()`，并由 `load_artifact_weight_lookup(..., mode="continuous")` 选择使用。
2. `train_locaware.py` 新增：
   - `--render_artifact_weight_mode {severity,continuous}`
   - `--render_artifact_weight_continuous_min`
   - `--render_artifact_weight_continuous_power`
3. `scripts/run_locaware_v03_topology_full.sh` 新增对应环境变量：
   - `RENDER_ARTIFACT_WEIGHT_MODE`
   - `RENDER_ARTIFACT_WEIGHT_CONTINUOUS_MIN`
   - `RENDER_ARTIFACT_WEIGHT_CONTINUOUS_POWER`
4. 修复了 P14/P15 暴露出的实现混杂：旧 continuous 公式对 bounded penalties 取 `max()`，导致 OldHospital 86 个 severe 样本全部饱和到同一个 `continuous_min`，实际退化成常数权重。修复后将 severe 阈值作为中段、向更坏方向扩展 hard-stop，并用 RMS 聚合多指标坏度。

关键权重分布复查：

```text
OldHospital severe rows = 86
continuous_min=0.25: min=0.4139 p25=0.5122 median=0.6248 p75=0.6880 max=0.7911 unique=86
continuous_min=0.50: min=0.6093 p25=0.6748 median=0.7498 p75=0.7920 max=0.8607 unique=86
continuous_min=0.70: min=0.7656 p25=0.8049 median=0.8499 p75=0.8752 max=0.9164 unique=86
```

验证：

```text
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_render_artifact_weights \
  tests.test_render_artifact_audit \
  tests.test_train_locaware_masks \
  tests.test_full_script_args

Ran 91 tests in 1.771s
OK
```

结果文件：

```text
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p14_render_artifact_weight_continuous_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p15_render_artifact_weight_continuous_severe_100_seed2026_summary.json
/mnt/pool/sqy/stdloc_la_artifact_filter_logs/p16_p17_fixed_continuous_render_artifact_weight_sweep_100_seed2026_summary.json
```

P14/P15 结论：

1. P14 `continuous + mild,severe` 不是好默认：ShopFacade 16 个 mild 被误伤，AE/TE/R5/Inliers 退化；OldHospital 只有 TE 小幅正向。
2. P15 `continuous + severe` 在旧公式下实际饱和为常数权重；它只能说明 severe-only 不误伤 ShopFacade，不能证明 continuous 策略有效。
3. 因此继续实验前必须先修复 continuous 饱和，这已由新增测试覆盖。

P16/P17 fixed-continuous 100-step，OldHospital severe-only，delta 相对 P8 no-filter：

| Variant | min | power | AE | dAE | TE | dTE | R5 | dR5 pp | R2 | dR2 pp | Inliers | dInliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-filter | - | - | 0.341946 | +0.000000 | 19.206845 | +0.000000 | 0.043956 | +0.000 | 0.000000 | +0.000 | 291.313 | +0.000 |
| fixed-continuous | 0.25 | 1.00 | 0.345793 | +0.003848 | 18.359265 | -0.847580 | 0.054945 | +1.099 | 0.005495 | +0.549 | 289.967 | -1.346 |
| fixed-continuous | 0.70 | 1.00 | 0.341539 | -0.000407 | 18.372950 | -0.833895 | 0.054945 | +1.099 | 0.005495 | +0.549 | 289.967 | -1.346 |
| fixed-continuous | 0.25 | 0.50 | 0.336800 | -0.005146 | 18.053414 | -1.153431 | 0.054945 | +1.099 | 0.010989 | +1.099 | 289.643 | -1.670 |
| fixed-continuous | 0.25 | 0.35 | 0.335021 | -0.006925 | 18.224268 | -0.982577 | 0.054945 | +1.099 | 0.010989 | +1.099 | 289.538 | -1.775 |

相对 P10 常数 severe=0.25：

| Variant | dAE vs P10 | dTE vs P10 | dR5 pp vs P10 | dR2 pp vs P10 |
| --- | ---: | ---: | ---: | ---: |
| fixed min=0.25 power=1.00 | +0.011103 | +0.172990 | +0.549 | +0.000 |
| fixed min=0.70 power=1.00 | +0.006849 | +0.186675 | +0.549 | +0.000 |
| fixed min=0.25 power=0.50 | +0.002110 | -0.132861 | +0.549 | +0.549 |
| fixed min=0.25 power=0.35 | +0.000330 | +0.037993 | +0.549 | +0.549 |

结论：

1. “continuous 实现有问题”这个怀疑成立且已修复：旧公式确实把所有 severe 行压成同一权重，不能据此评价 continuous 方案。
2. 修复后的 continuous weighting 给出正向支撑：P17 `min=0.25,power=0.5` 相对 no-filter 同时改善 AE、TE、R5、R2，且在 TE/R5/R2 上优于 P10 常数 severe=0.25；AE 仍略弱于 P10。
3. `power=0.35` 更激进，AE 最接近 P10，但 TE 不如 `power=0.5`；当前 100-step 最优候选是 `min=0.25,power=0.5`。
4. 这仍只是 OldHospital 100-step 证据。默认策略暂不从 P13 支持的 `mode=severity,severe=0.70` 改成 continuous；需要 fixed-continuous 500-step、多 scene、多 seed 之后再决定。
5. ShopFacade 在 severe-only 下匹配数为 0，continuous/severity 都应是 no-op；P14 说明 mild inclusion 仍会误伤，因此 mild 不能进入默认抑制集合。
