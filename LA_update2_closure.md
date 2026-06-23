# LA_update2 初步落实闭环记录

日期：2026-06-24

## 结论

`LA_update2.md` 中最影响实验可信度的工程问题已经完成首轮闭环：

1. 训练随机种子和 query split seed 已拆分，`train_locaware.py` 现在在 `safe_state()` 之后按 `--train_seed` 重新设种子。
2. support/query 划分新增 `random`、`sequence_block`、`temporal_block` 三种模式，可避免相邻视频帧随机泄漏到 support/query 两侧。
3. v0.3 多场景脚本已从单层 `SEEDS` 改为 `TRAIN_SEEDS x QUERY_SPLIT_SEEDS` 矩阵，输出目录按 `train_seed_x/query_split_y` 隔离。
4. direct descriptor teacher 已改进 anchor 采样和 full-bank loss：投影网格均衡采样替代纯线性下采样，同源 sibling false negatives 可通过 ignore mask 屏蔽。
5. 单测、脚本语法检查、全量 unittest 均已通过；ShopFacade 2-step smoke 已跑通 train+eval。

这轮结果证明新实现路径可运行，但还不能作为论文级精度结论。真正的精度闭环仍需按新 seed 语义和 block split 重跑 matched continuation 矩阵。

## 已落实改动

| 项目 | 状态 | 位置 |
| --- | --- | --- |
| `--train_seed` | 已新增，且放在 `safe_state(args.quiet)` 后生效 | `train_locaware.py` |
| `--query_split_mode` | 已新增，默认 `random`，支持 `sequence_block` / `temporal_block` | `train_locaware.py`、`localization_training/episode_sampler.py` |
| v0.3 单实验脚本 seed 入口 | 已新增 `V03_TRAIN_SEED`、`V03_QUERY_SPLIT_MODE` | `scripts/run_locaware_v03_shopfacade.sh` |
| 多场景 seed 矩阵 | 已改为 `TRAIN_SEEDS="0 1 2"` 与 `QUERY_SPLIT_SEEDS="2025 2026 2027"` | `scripts/run_locaware_v03_multiscene.sh` |
| topology 路径 seed 目录 | 已改为 `train_seed_x/query_split_y`，训练时传入 `--train_seed` | `scripts/run_locaware_v03_topology_full.sh` |
| descriptor anchor 采样 | 已支持按投影 2D grid round-robin 均衡选点 | `localization_training/direct_landmark_teacher.py` |
| full-bank false negative 控制 | 已支持 `ignore_bank_mask`，direct teacher 会用 `loc_source_index` 屏蔽同源 sibling negatives | `localization_training/direct_landmark_teacher.py` |
| 回归测试 | 已补充 seed/split/parser/script/descriptor 测试 | `tests/` |

## 验证结果

针对性回归：

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest \
  tests.test_episode_sampler \
  tests.test_full_script_args \
  tests.test_train_locaware_masks \
  tests.test_direct_landmark_teacher
```

结果：

```text
Ran 58 tests in 1.706s
OK
```

脚本语法：

```text
bash -n scripts/run_locaware_v03_shopfacade.sh \
  scripts/run_locaware_v03_multiscene.sh \
  scripts/run_locaware_v03_topology_full.sh
```

结果：通过。

全量 unittest：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests
```

结果：

```text
Ran 129 tests in 7.499s
OK
```

说明：不显式设置 CUDA 路径时，当前 shell 会优先使用 `iclpose` conda 环境里的 `nvcc`，`test_renderer_loc_grad` 会因缺 `cuda_runtime.h` 触发 `gsplat_cuda` 编译失败；显式使用 `/usr/local/cuda-11.8` 后通过。

## ShopFacade 2-step smoke

命令要点：

```text
CUDA_VISIBLE_DEVICES=0
SCENE=ShopFacade
BASELINE_MODEL=/mnt/pool/sqy/stdloc_la_full_runs/ShopFacade_baseline
MODEL_ROOT=/mnt/pool/sqy/stdloc_la_update2_smoke
V03_MODEL=/mnt/pool/sqy/stdloc_la_update2_smoke/ShopFacade_train_seed_0_query_split_2025_smoke
V03_STEPS="1 2"
V03_MAX_STEP=2
RUN_SWEEP=0
V03_TRAIN_SEED=0
V03_QUERY_SPLIT_SEED=2025
V03_QUERY_SPLIT_MODE=sequence_block
bash scripts/run_locaware_v03_shopfacade.sh
```

关键日志：

```text
Support/query split enabled: support=185 query=46 query_ratio=0.2 query_split_seed=2025 query_split_mode=sequence_block
[ITER 30001] base 0.143182 loc 0.522090 psnr 19.074
[ITER 30002] base 0.278130 loc 0.517225 psnr 13.865
```

Sparse-only eval：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline sparse, existing | 0.166517 | 3.349951 | 0.728155 | 0.262136 | 388.107 |
| update2 smoke 30001 | 0.162566 | 3.350459 | 0.728155 | 0.252427 | 388.854 |
| update2 smoke 30002 | 0.163860 | 3.350459 | 0.728155 | 0.242718 | 389.524 |

结论：2-step smoke 与 baseline 基本持平，符合“验证通路”预期；不能据此判断方法精度收益。

输出：

```text
/mnt/pool/sqy/stdloc_la_update2_smoke/ShopFacade_train_seed_0_query_split_2025_smoke
/root/STDLoc/results/phase-v03-30001-_mnt_pool_sqy_stdloc_la_update2_smoke_ShopFacade_train_seed_0_query_split_2025_smoke-20260624_030117
/root/STDLoc/results/phase-v03-30002-_mnt_pool_sqy_stdloc_la_update2_smoke_ShopFacade_train_seed_0_query_split_2025_smoke-20260624_030158
```

## 并行实验建议

当前 `nvidia-smi` 显示 3 张 RTX 3090 基本空闲。下一步可并行跑 3 个 scene 的首个 block-split 训练种子：

```text
CUDA_VISIBLE_DEVICES=0 SCENE=ShopFacade V03_TRAIN_SEED=0 V03_QUERY_SPLIT_SEED=2025 V03_QUERY_SPLIT_MODE=sequence_block RUN_SWEEP=0 bash scripts/run_locaware_v03_shopfacade.sh
CUDA_VISIBLE_DEVICES=1 SCENE=KingsCollege V03_TRAIN_SEED=0 V03_QUERY_SPLIT_SEED=2025 V03_QUERY_SPLIT_MODE=sequence_block RUN_SWEEP=0 bash scripts/run_locaware_v03_shopfacade.sh
CUDA_VISIBLE_DEVICES=2 SCENE=OldHospital V03_TRAIN_SEED=0 V03_QUERY_SPLIT_SEED=2025 V03_QUERY_SPLIT_MODE=sequence_block RUN_SWEEP=0 bash scripts/run_locaware_v03_shopfacade.sh
```

建议先不要一次性启动完整 `3 scenes x 3 train seeds x 3 query splits x sweep`，因为训练和 sparse-only eval 都会读写大量 Cambridge 数据。更稳妥的顺序是：

1. 先并行跑 `3 scenes x train_seed=0 x query_split=2025 x sequence_block`，不开 sweep。
2. 若三场景都没有回归，再扩到 `TRAIN_SEEDS="0 1 2"`，固定 `QUERY_SPLIT_SEEDS="2025"`。
3. 最后再补 `QUERY_SPLIT_SEEDS="2026 2027"` 和 solver sweep。

## 仍未闭合

1. selective dense supervision 仍未按 `pose-improvement / attribution confidence / correspondence correctness` 三个 gate 重构。
2. topology 仍需从 physical prune 主张收缩到 split-first、soft-mask prune、risk-controlled commit 的 matched continuation 矩阵。
3. utility v2 的字段拆分、cross-fitting calibrator、Go/No-Go 指标尚未实现。
4. localization-only overlay map 尚未动手。
5. 还没有按新 seed 语义产出正式精度表；旧“三 seed”结果应改称 three query-split repetitions。

## 附件观点后的补充修复

日期：2026-06-24

附件指出的判断是成立的：旧 topology full-length 脚本把多个高影响因素绑在一起，不能用 mixed 结果直接判定 topology 主张错误。当前已补第一轮实现归因修复：

1. `scripts/run_locaware_v03_topology_full.sh` 默认不再回落到 dense teacher，而是复用 v0.3 direct objective：
   - `TRAIN_PHASE=${TRAIN_PHASE:-feature}`
   - `LOC_TEACHER=${LOC_TEACHER:-direct}`
   - direct / multiview / full-bank / anchor 权重与 v0.3 continuation 对齐
   - 默认 `--no-use_loc_opacity`
   - 默认 `TOPOLOGY_GEOMETRY_ANCHOR_WEIGHT=0.0`
2. topology mutation 增加独立模式：
   - `TOPOLOGY_MUTATION_MODE=no_mutation`
   - `TOPOLOGY_MUTATION_MODE=split_only`，默认
   - `TOPOLOGY_MUTATION_MODE=soft_prune`
   - `TOPOLOGY_MUTATION_MODE=physical_prune`
   - `TOPOLOGY_MUTATION_MODE=current_full`，用于复现旧 mixed 脚本路径
3. `LocalizationTopologyController` 修复两个控制参数问题：
   - `ambiguity_quantile` 现在实际过滤低 ambiguity split 候选；
   - split 数量现在会按剩余 `total_point_budget_ratio` 严格裁剪，避免单次事件越过总预算。

新增/更新测试：

```text
tests.test_full_script_args.FullRunScriptArgsTest.test_v03_topology_script_matches_v03_direct_objective_by_default
tests.test_topology_controller.TopologyControllerTest.test_ambiguity_quantile_filters_low_ambiguity_split_candidates
tests.test_topology_controller.TopologyControllerTest.test_topology_update_caps_split_count_to_remaining_total_budget
```

验证：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 131 tests in 6.675s
OK
```

### ShopFacade attribution smoke

从旧 full-length v0.3 checkpoint 分叉，显式指定：

```text
SOURCE_MODEL=/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
V03_ITERATION=30500
TOPOLOGY_STEPS=1
TOPOLOGY_UPDATE_INTERVAL=1
LABEL_MAX_IMAGES=4
TOPOLOGY_MUTATION_MODE=split_only
TRAIN_PHASE=feature
LOC_TEACHER=direct
```

checkpoint 配置核对：

```text
train_phase feature
loc_teacher direct
enable_topology True
topology_enable_physical_prune False
topology_enable_soft_prune False
geometry_anchor_weight 0.0
loc_direct_weight 0.05
loc_full_bank_weight 0.05
loc_multiview_weight 0.03
loc_anchor_weight 0.01
use_loc_opacity False
train_seed 0
```

点数核对：

```text
30500 source: points=342918 unique_sources=342918
30501 smoke:  points=342918 unique_sources=342918
```

本次 1-step smoke 没有实际 split 增长，因此只能证明修复后的 topology continuation 入口已经不再混入 dense teacher、geometry unlock 和 physical prune。它不是 topology 精度结论。

Sparse-only smoke result：

| Run | AE | TE | R5 | R2 | Inliers |
| --- | ---: | ---: | ---: | ---: | ---: |
| corrected direct split-only smoke 30501 | 0.155737 | 3.071543 | 0.747573 | 0.310680 | 449.893 |

输出：

```text
/mnt/pool/sqy/stdloc_la_attachment_smoke/ShopFacade/train_seed_0/query_split_2025/ShopFacade_v03_topology_from_30500
/root/STDLoc/results/phase-v03-topology-30501-_mnt_pool_sqy_stdloc_la_attachment_smoke_ShopFacade_train_seed_0_query_split_2025_ShopFacade_v03_topology_from_30500-20260624_031721
```

### 当前判断更新

旧 OldHospital / KingsCollege mixed topology 结果确实不足以证伪原始 topology 主张，因为旧实验同时改变了 teacher、geometry、optimizer restart、mutation 和 label-state 路径。更准确的表述应是：

```text
descriptor 子主张已有正向迹象；
topology 主张尚未被干净证伪；
dense feedback 与 topology 的完整闭环仍未被干净测试。
```

仍需继续修复/验证：

1. direct+topology 联训下，feature anchor 与 memory 仍依赖 current row index，prune 后存在 silent mis-assignment 风险；需要稳定 `node_id`。
2. `remap_topology_landmarks.py` 目前只读取 `loc_state.pt`，没有 xyz，因此还不能实现 projection-consistent child / virtual parent remap。
3. physical prune 仍未验证 `loc_opacity.grad` 是否非零；在此之前不能把 loc opacity 当作独立定位剪枝门控。
4. 需要按附件 C0-C7 矩阵补跑 matched continuation，至少先跑 C1/C4/C7 的 100-step ShopFacade/Kings/OldHospital 对照。

## 高影响混杂闭合进展

日期：2026-06-24

本轮继续按附件里的怀疑点做实现修复和 matched smoke。结论先行：怀疑不是小概率，已经确认旧 mixed topology 结论至少受到数个实现/实验混杂影响；当前这些高影响混杂已基本从 10-step ShopFacade smoke 中隔离，仍不能据此证伪原始 topology 主张。

### 新增修复

1. 稳定拓扑身份：
   - `GaussianModel` 新增 `loc_node_id`、`loc_parent_node_id`、`loc_source_xyz`，保存 `loc_current_xyz`。
   - split child 继承 `loc_source_index/loc_source_xyz`，但获得新的 `loc_node_id`，并记录 parent node。
   - feature anchor refresh 改为按 `loc_node_id` 对齐，避免 prune/split 后 row-index silent mis-assignment。
2. topology landmark remap：
   - `remap_topology_landmarks.py` 默认使用 `source_distance` remap。
   - remap summary 记录 `remap_source_distance_mean/max`，能区分同 source 的 projection-consistent child 和漂移 child。
3. prune-only attribution：
   - `TopologyConfig.enable_split` 与 CLI `--topology_disable_split`。
   - 脚本新增 `TOPOLOGY_MUTATION_MODE=soft_prune_only` 和 `physical_prune_only`，C5/C6 不再偷偷混入 split。
4. physical prune guard：
   - 默认要求 `loc_opacity_grad_seen=True` 才允许 physical prune。
   - 发现并修复两个实质 bug：
     - external `loc_state` 在 `training_setup()` 后恢复时会替换 `_loc_opacity` Parameter，但 optimizer 仍指向旧 Parameter；
     - `localization_opacity_regularizer = mean + abs(mean-target)` 在默认 target=0.5 且 mean<0.5 时梯度抵消为 0。
   - 修复后 `restore_localization_state()` 会把 `_loc_opacity` 重新绑定进 optimizer；opacity density 项改为平方误差，默认配置下有非零梯度。
5. 旧 mixed 脚本复现路径：
   - `current_full` 明确设为 dense teacher + geometry anchor + physical prune + sparse landmark protection。
   - dense loss 权重从脚本变量传入，避免之前 `current_full` 训练时 dense loss 实为 0。

### 新增/更新测试

```text
tests.test_localization_utility:
  stable loc_node_id / loc_source_xyz / projection-consistent remap
tests.test_train_locaware_masks:
  feature anchor 按 node_id 对齐；physical prune override 参数
tests.test_topology_controller:
  ambiguity quantile 生效；总点数预算裁剪；physical prune guard；prune-only disable split
tests.test_topology_stats:
  restore_external_loc_state 后 loc_opacity optimizer rebind 且可收到梯度
tests.test_locaware_losses:
  loc opacity regularizer 默认 target 下低 opacity 仍有非零梯度
tests.test_full_script_args:
  topology 脚本默认 direct objective；source_distance remap；prune-only 模式；current_full dense 权重
```

验证：

```text
CUDA_HOME=/usr/local/cuda-11.8 \
PATH=/usr/local/cuda-11.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} \
PYTHONPATH=/root/STDLoc \
/root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests

Ran 141 tests in 6.658s
OK
```

### ShopFacade 10-step attribution smoke

统一设置：

```text
DATA_ROOT=/mnt/pool/sqy/Cambridge_stdloc
SOURCE_MODEL=/mnt/pool/sqy/stdloc_la_v03_full_length/ShopFacade/seed_2025/ShopFacade_v03
V03_ITERATION=30500
TOPOLOGY_STEPS=10
TOPOLOGY_UPDATE_INTERVAL=5
LABEL_MAX_IMAGES=4
TRAIN_SEED=0
QUERY_SPLIT_SEED=2025
```

| Run | 变更点 | Points | Child rows | Remap mean/max | AE | TE | R5 | R2 | Inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 no_mutation | direct feature continuation | 342918 | 0 | 0 / 0 | 0.151717 | 3.022385 | 0.776699 | 0.320389 | 450.117 |
| C2 geometry_only | C1 + geometry phase + anchor 0.01 | 342918 | 0 | 0 / 0 | 0.151717 | 3.022385 | 0.776699 | 0.320389 | 450.136 |
| C3 dense_only | dense teacher, no topology | 342918 | 0 | 0 / 0 | 0.154318 | 2.981734 | 0.757282 | 0.330097 | 450.835 |
| C4 split_only | direct + split only | 342920 | 4 | 0.000003 / 0.042758 | 0.152975 | 2.983277 | 0.776699 | 0.310680 | 450.146 |
| C5 soft_prune_only | direct + soft prune, split disabled | 342918 | 0 | 0 / 0 | 0.151717 | 3.022385 | 0.776699 | 0.320389 | 450.117 |
| C6 physical_prune_only | direct + trained loc opacity + physical prune, split disabled, sparse landmarks protected | 342918 | 0 | 0 / 0 | 0.151717 | 3.022385 | 0.776699 | 0.320389 | 450.136 |
| C7 current_full_dense | dense + geometry + physical prune + split | 342925 | 14 | 0.000027 / 0.161012 | 0.150239 | 3.022879 | 0.737864 | 0.310680 | 450.738 |

C6 关键证据：

```text
use_loc_opacity=True
loc_opacity_weight=0.01
topology_enable_physical_prune=True
topology_disable_split=True
topology_protect_landmarks=True
topology_allow_untrained_loc_opacity_prune=False
loc_opacity_mean: 0.421536 -> 0.418638
mean |delta loc_opacity|: 0.002898
```

这说明 physical prune guard 是被真实 loc opacity 训练信号放行，而不是靠 legacy override。

### 当前闭环判断

已确认的实现问题：

1. 旧 topology 结果把 dense teacher、geometry unlock、split、physical prune、label-state/remap 路径混在一起。
2. split 后 landmark remap 只靠 source/utility 会有错误归因风险，已改成 source-distance 一致性。
3. feature anchor 按 row index 对齐确实有 split/prune 后错配风险，已改为 stable node id。
4. physical prune 原先可能使用未训练 loc opacity；进一步发现即使打开 loc opacity，旧 loss 默认也可能 0 梯度，且 external loc_state 会让 optimizer 指向旧 Parameter。
5. C5/C6 旧脚本语义不是 prune-only，已加 disable-split 闭合。

已排除/降低的混杂：

1. C1/C2/C5/C6 在 10-step smoke 中与 no-mutation 基本持平，短程没有显示 geometry-only、soft-prune-only、guarded physical-prune-only 单独造成明显波动。
2. C4 split-only 确实发生拓扑增长，并且 source-distance remap 正常，无 missing；短程结果中性略混，不能作为证伪。
3. C7 mixed 仍表现为 recall@5cm 下降，说明旧 current_full 路径确有混合效应，但不能归因到 topology split 本身。

仍未完全闭合：

1. 当前只是 ShopFacade 10-step smoke；还需要 100/500-step matched continuation、多 seed、多 scene。
2. physical prune 在默认阈值下本轮没有实际删点，只闭合了“训练过 loc opacity 的 guard 与 prune-only 路径”，还没证明 physical prune 策略有效。
3. optimizer restart 的严格 matched 对照仍需单独设计；目前只修了 loc_state 替换 Parameter 的明确 bug。
4. dense feedback 与 topology 的正向闭环仍未被干净证明；C7 仍是混合路径，不是最终方法结论。
