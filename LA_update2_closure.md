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

## 长跑与删点闭环

日期：2026-06-24

本轮继续排除 topology 阶段剩余的高影响混杂，实验根目录固定为：

```text
/mnt/pool/sqy/stdloc_la_update2_long_closure_v2
```

旧的 `/mnt/pool/sqy/stdloc_la_update2_long_closure` 不再使用，因为早期脚本会把复制来的 source checkpoint 误判为目标 iteration 已存在，从而跳过真实 topology continuation。

### 新增修复

1. `scripts/run_locaware_v03_topology_full.sh` 增加 `FORCE_TOPOLOGY_TRAIN=1`：
   - 若目标 final iteration 已存在，会先移除该 iteration 再训练；
   - 避免 copied source checkpoint 污染 100/500-step continuation。
2. `localization_training/topology_controller.py` 修复 physical-prune-only 事件日志：
   - 旧逻辑只在 `split.any()` 时打印 `[Topology]`；
   - 现在 prune-only 也会打印 `physical_prune`、`points=start->after`。
3. 新增长跑 worker：
   - `scripts/run_la_update2_long_worker.sh`
   - 显式使用 `/mnt/pool/sqy/stdloc_la_v03_full_length/{scene}/seed_{seed}/{scene}_v03` 作为 source；
   - 默认覆盖 `3 scenes x 3 seeds x {100,500} steps x {no_mutation, split_only}`；
   - prune sweep 默认覆盖 mild / balanced / active。
4. 新增最终汇总脚本：
   - `scripts/summarize_la_update2_long_closure.py`
   - 只统计目标 final iteration，避免把 source iteration 混进结果；
   - 输出点数、child rows、source distance、opacity、event log 与 STDLoc sparse metrics。

### 实验完整性

最终汇总文件：

```text
/mnt/pool/sqy/stdloc_la_update2_long_closure_v2/summary_final.json
```

完整性检查：

| Family | Tag | Rows |
| --- | --- | ---: |
| core | `core_no_mutation_100` | 9 |
| core | `core_split_only_100` | 9 |
| core | `core_no_mutation_500` | 9 |
| core | `core_split_only_500` | 9 |
| prune | `prune_mild_100` | 5 |
| prune | `prune_balanced_100` | 5 |
| prune | `prune_active_100` | 9 |

所有 55 行都有 STDLoc sparse-only summary。核心矩阵完整覆盖 `3 scenes x 3 seeds x 2 steps x 2 modes = 36` 行。

### Split-only matched continuation

对每个 scene/seed/step 使用 `split_only - no_mutation` 做 paired delta：

| Scene | Steps | n | mean dR5 | mean dR2 | R5 pos/zero/neg | mean child rows |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| ShopFacade | 100 | 3 | -0.971 pp | -1.942 pp | 0/1/2 | 136.0 |
| ShopFacade | 500 | 3 | +0.324 pp | +1.618 pp | 2/0/1 | 703.3 |
| KingsCollege | 100 | 3 | +0.000 pp | +0.097 pp | 0/3/0 | 134.0 |
| KingsCollege | 500 | 3 | -0.000 pp | +0.000 pp | 1/0/2 | 774.0 |
| OldHospital | 100 | 3 | -0.549 pp | -0.183 pp | 1/0/2 | 82.7 |
| OldHospital | 500 | 3 | +0.549 pp | +0.366 pp | 2/0/1 | 419.3 |

聚合结果：

| Scope | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | median dR2 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 100-step | 9 | -0.507 pp | 0.000 pp | 1/4/4 | -0.676 pp | 0.000 pp |
| 500-step | 9 | +0.291 pp | +0.549 pp | 5/0/4 | +0.661 pp | +0.549 pp |
| all core | 18 | -0.108 pp | 0.000 pp | 6/4/8 | -0.007 pp | 0.000 pp |

结论：split-only 已经被干净地从 teacher / geometry / physical prune 中分离出来，也确实产生 child rows；但在 3 scene x 3 seed 的 100/500-step matched continuation 上，R5/R2 收益很小且正负混合。当前数据不支持把 split-only 作为已验证的稳健精度增益。

### Physical prune 删除与策略效果

本轮已排除“默认阈值没有实际删点”的问题：active 阈值在三个 scene、三个 seed 都发生实删，event log 与最终 point count 一致。

| Tag | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean point delta | logged physical prune |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `prune_mild_100` | 5 | -0.161 pp | 0.000 pp | 2/1/2 | -1,593.4 | 6,961 |
| `prune_balanced_100` | 5 | +0.033 pp | +0.292 pp | 3/0/2 | -15,460.4 | 68,396 |
| `prune_active_100` | 9 | +0.047 pp | 0.000 pp | 1/7/1 | -43,536.0 | 391,824 |

`prune_active_100` 分 scene：

| Scene | n | mean dR5 | mean dR2 | mean point delta | logged physical prune |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 3 | +0.324 pp | -1.294 pp | -28,400 | 85,200 |
| KingsCollege | 3 | +0.000 pp | +0.000 pp | -36,743 | 110,229 |
| OldHospital | 3 | -0.183 pp | +0.000 pp | -65,465 | 196,395 |

说明：

1. `point_count_delta < 0` 是最终 checkpoint 的直接证据；`logged physical prune` 是事件日志证据。
2. ShopFacade seed2025 的 mild/balanced 两行是在日志修复前启动的，因此 `logged physical prune=0`，但最终 `point_count_delta` 仍证明发生了实删点。
3. physical prune 的“机制可生效”已经证明；“策略能提升 sparse-only relocalization”尚未证明。active 删除很激进，但 R5/R2 总体近似中性，ShopFacade 的 R2 还下降。

### 本轮判断

现在可以把几个高影响混杂点分开判断：

| 混杂点 | 状态 | 证据 |
| --- | --- | --- |
| copied source checkpoint 导致 false skip | 已排除 | `FORCE_TOPOLOGY_TRAIN=1`，v2 root 重跑 |
| teacher / geometry / topology mutation 混在一起 | 已排除首层 | core 矩阵使用 direct objective，对照 `no_mutation` vs `split_only` |
| split-only 没有真实 topology event | 已排除 | 100-step 平均 4 个 event，500-step 平均 20 个 event |
| physical prune 默认阈值没有删点 | 已排除 | active 9 行均实删，合计 logged physical prune 391,824 |
| prune-only event log 漏报 | 已修复 | prune-only 也记录 `[Topology]` |

更准确的闭环结论是：

```text
旧 mixed topology 结果不能用来证伪原始 topology 主张；
但在排除主要实现与实验混杂后，split-only 的定位收益仍然很弱且不稳健；
physical prune 的执行机制已经跑通，但当前阈值策略没有显示稳健收益，后续应改为 utility 校准和 risk-controlled commit，而不是继续扩大默认物理删点。
```

### 本轮验证

脚本语法与汇总脚本编译：

```text
bash -n /root/STDLoc/scripts/run_locaware_v03_topology_full.sh
bash -n /root/STDLoc/scripts/run_la_update2_long_worker.sh
/root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile /root/STDLoc/scripts/summarize_la_update2_long_closure.py
```

结果：均通过。

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
Ran 142 tests in 6.002s
OK
```

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

## 最终状态更新

上面的“仍未完全闭合”是 ShopFacade 10-step smoke 后的中间状态；截至本文件的“长跑与删点闭环”一节，前两项已经更新：

1. 100/500-step、三 scene、三 seed 的 `no_mutation` vs `split_only` matched continuation 已完成，核心矩阵 36 行，0 个缺失 summary。
2. physical prune 默认无删点的混杂已通过 mild/balanced/active sweep 排除；active 9 行均发生实删点，合计 logged physical prune 391,824。

当前最终判断不再是“需要长跑后再判断”，而是：

```text
实现混杂已经基本排除到首层；
split-only 的定位收益在当前矩阵中很弱且不稳健；
physical prune 的机制有效，但当前阈值策略没有证明能提升 sparse-only relocalization。
```

## dense 长跑与 C6 默认阈值闭合补充

日期：2026-06-24

本节继续闭合上面剩余两项：dense feedback 的 100/500-step、多 scene、多 seed 长跑，以及 C6 默认阈值下 physical prune 是否真的删点。

### 新增修复与脚本

1. `scripts/run_densekl_v03_cambridge.sh`
   - 新增 `DENSEKL_SAVE_STEPS` / `DENSEKL_EVAL_STEPS`，一次 500-step 训练可同时保存和评测 100/500-step。
   - 新增 pose gate 和 selective dense gate 参数入口，并支持 `RUN_DENSE_POSE_CACHE=1`。
   - 新增 `FORCE_DENSEKL_TRAIN=1`。
   - 新增 `strip_future_point_clouds()`，从 full-length source copy 后删除所有 `iteration > LOAD_ITERATION` 的 checkpoint。
2. `scripts/run_locaware_v03_topology_full.sh`
   - 新增 `FORCE_TOPOLOGY_TRAIN=1`。
   - 同样新增 future checkpoint 清理，避免 source 中已有 `iteration_30600/31000` 导致目标实验假跳过训练。
3. 新增 dense 长跑 worker 和 summary：
   - `scripts/run_la_update2_dense_long_worker.sh`
   - `scripts/summarize_la_update2_dense_long.py`

本轮确认并修复了一个高影响实现混杂：`/mnt/pool/sqy/stdloc_la_v03_full_length/*` 的 source model 已含后续 `iteration_30600/31000`，直接 copy 会让 target run 误以为 100/500-step 已训练完成。修复后 dense 长跑均使用 `FORCE_DENSEKL_TRAIN=1` 重跑。

### dense pose-gate 100/500-step 长跑

输出：

```text
/mnt/pool/sqy/stdloc_la_update2_dense_long_v1/summary_final.json
```

矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
seeds = 2025, 2026, 2027
steps = 100, 500
rows = 18
```

关键 sanity check：

| Scene | Pose cache valid | dense pose both-better ratio | Loc-positive iters |
| --- | ---: | ---: | ---: |
| KingsCollege | 1220 / 1220 | 0.550-0.567 | 61-63 / 103 |
| OldHospital | 895 / 895 | 0.834-0.840 | 82-91 / 103 |
| ShopFacade | 231 / 231 | 0.697-0.762 | 82-89 / 103 |

这排除了两个混杂：dense pose metadata 不是空的，pose-gated dense loss 也确实进入训练；此前 strict 10-step smoke 里的 `Loc=0` 主要是过严 gate / sampling 设置导致。

Sparse-only 指标与 `no_mutation` matched continuation 对齐后的结果：

| Steps | n | dR5 mean | dR5 median | R5 pos/zero/neg | dR2 mean | dTE mean |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 100 | 9 | -0.003555 | 0.000000 | 4 / 2 / 3 | -0.001221 | +0.473840 |
| 500 | 9 | -0.013509 | -0.010989 | 3 / 1 / 5 | +0.001689 | +1.309755 |

按 scene 展开：

| Scene | Steps | n | dR5 mean | dR2 mean | dTE mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege | 100 | 3 | +0.007775 | 0.000000 | +0.333471 |
| KingsCollege | 500 | 3 | +0.008746 | 0.000000 | +0.625849 |
| OldHospital | 100 | 3 | -0.005495 | -0.003663 | +1.297779 |
| OldHospital | 500 | 3 | -0.020147 | +0.001832 | +3.301639 |
| ShopFacade | 100 | 3 | -0.012945 | ~0.000000 | -0.209731 |
| ShopFacade | 500 | 3 | -0.029126 | +0.003236 | +0.001777 |

结论：dense pose-gate 分支已经证明“通路可训练、metadata 有效、loss 非零”，但没有证明正向精度收益。尤其 500-step 汇总下 R5 平均低于 no-mutation，TE 平均更差。当前更像是 dense teacher 质量/权重/采样仍有问题，而不是 loss path 没跑起来。

### C6 默认阈值 physical prune

输出：

```text
/mnt/pool/sqy/stdloc_la_update2_prune_default_v1/summary_final.json
```

矩阵：

```text
ShopFacade seeds 2025/2026/2027, steps 100/500
KingsCollege seed 2025, steps 100
OldHospital seed 2025, steps 100
rows = 8
```

汇总：

```text
physical_prune_total = 0
point_count_delta_total = 0
max physical_prune per event = 0
```

所有 8 行与 `no_mutation` 对齐后，`dR5 = 0`、`dR2 = 0`、`dTE = 0`。其中 ShopFacade 500-step 每行有 20 个 topology events，KingsCollege/OldHospital 100-step 每行有 4 个 topology events，但默认阈值均没有候选通过：

```text
KingsCollege seed2025 100: physical_prune_total=0, point_count_delta=0
OldHospital seed2025 100: physical_prune_total=0, point_count_delta=0
ShopFacade seed2025/2026/2027 100/500: physical_prune_total=0, point_count_delta=0
```

结论：C6 默认阈值下的 physical prune 是 inert 策略，不能作为“prune 策略有效”的证据。它只闭合了“训练过 loc opacity 后不会误删”的混杂。之前 active sweep 已证明机制能真实删点，但 active 策略没有稳健精度收益，因此 physical prune 的策略有效性仍未证明。

### 本轮最终判断

用户关于“可能是实现问题而不是主张错误”的怀疑被部分确认：本轮确实发现并修复了 future checkpoint copy 导致假跳过训练、strict dense gate 导致 `Loc=0`、以及 topology source checkpoint 污染等高影响混杂。

修复后更新判断：

```text
1. dense feedback 通路已跑通，但当前 pose-gate-only 100/500-step 结果不支持正向 precision 主张；
2. split-only 长跑收益弱且不稳健；
3. physical prune 默认策略没有实际删点，active 策略能删点但未显示稳健收益；
4. 因此原始强主张不能用旧 mixed 结果证伪，但也尚未被修复后的 clean matrix 支持。
```

后续若要继续证明 physical prune 策略，需要单独设计阈值校准目标或 risk-controlled commit，而不是继续使用当前默认阈值。

## 剩余长跑闭合补充

日期：2026-06-24

本节继续闭合用户指出的两个剩余点：

1. dense feedback 不能只看 10-step ShopFacade smoke，需要 100/500-step、多 scene、多 seed。
2. C6 默认阈值没有删点，不能证明 physical prune 策略有效；需要至少跑一个实际删点的 matched physical-prune-only 长跑矩阵。

### Strict pose-gate dense 长跑

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_dense_strict_pose500_v1/summary_final.json
```

配置差异：

```text
DENSEKL_POSE_GATE_MIN_TE=1.0
DENSEKL_POSE_GATE_MIN_AE=0.03
DENSEKL_STEPS=500
DENSEKL_SAVE_STEPS="100 500"
DENSEKL_EVAL_STEPS="100 500"
scenes = ShopFacade, KingsCollege, OldHospital
seeds = 2025, 2026, 2027
rows = 18
```

与上一轮 loose pose gate 相比，strict gate 明显降低了 eligible episode 比例，但仍保持非零 dense loss。最终 sparse-only paired delta 相对 `no_mutation`：

| Steps | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 100 | 9 | +0.001697 | +0.005495 | 6 / 1 / 2 | -0.001079 | +0.343680 |
| 500 | 9 | -0.012610 | 0.000000 | 3 / 3 / 3 | +0.006759 | +1.306976 |

500-step 分 scene：

| Scene | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| KingsCollege | 3 | +0.006803 | +0.005831 | 3 / 0 / 0 | -0.000972 | +0.661148 |
| OldHospital | 3 | -0.021978 | -0.032967 | 0 / 1 / 2 | +0.001832 | +3.310144 |
| ShopFacade | 3 | -0.022654 | 0.000000 | 0 / 2 / 1 | +0.019417 | -0.050364 |

结论：

```text
strict pose gate 支持“dense 实现不是没跑起来，过宽 gate 确实会引入噪声”的判断；
但 500-step 后整体 R5 仍偏负，TE 明显受 OldHospital 拉低；
因此 dense pose-gate-only 仍不能作为 LA-STDLoc precision 正向主张的证据。
```

### Active physical-prune-only 500-step

输出：

```text
/mnt/pool/sqy/stdloc_la_update3_prune_active500_v1/summary_final.json
```

配置：

```text
TOPOLOGY_MUTATION_MODE=physical_prune_only
TOPOLOGY_STEPS=500
TOPOLOGY_UPDATE_INTERVAL=25
TOPOLOGY_USE_LOC_OPACITY=1
TOPOLOGY_LOC_OPACITY_WEIGHT=0.01
TOPOLOGY_PROTECT_LANDMARKS=1
TOPOLOGY_DISABLE_SPLIT=1
TOPOLOGY_PHYSICAL_RGB_THRESHOLD=0.005
TOPOLOGY_PHYSICAL_LOC_THRESHOLD=0.20
TOPOLOGY_PHYSICAL_UTILITY_THRESHOLD=0.10
scenes = ShopFacade, KingsCollege, OldHospital
seeds = 2025, 2026, 2027
rows = 9
```

所有 9 行都发生实际物理删点，且 `requested_split_total=0`、`children_added_total=0`，因此这组隔离的是 physical prune 本身，而不是 split 或 mixed topology。

| Scene | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE | mean point delta | logged prune |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| KingsCollege | 3 | -0.001944 | -0.002915 | 0 / 1 / 2 | 0.000000 | +0.177117 | -36,743 | 110,229 |
| OldHospital | 3 | +0.003663 | +0.005495 | 2 / 0 / 1 | 0.000000 | +0.047611 | -65,465 | 196,395 |
| ShopFacade | 3 | +0.006472 | +0.009709 | 2 / 1 / 0 | +0.003236 | -0.044586 | -28,400 | 85,200 |
| Overall | 9 | +0.002731 | 0.000000 | 4 / 2 / 3 | +0.001079 | +0.060047 | -43,536 | 391,824 |

关键判断：

```text
C6 默认阈值 inert 的混杂已经闭合：active 阈值下 9/9 行都实删点；
physical prune 的机制有效，并且 active500 没有出现灾难性退化；
但 precision 收益很小、median dR5=0，且 KingsCollege 偏负，所以“physical prune 策略能稳健提升 sparse-only relocalization”仍未被证明。
```

### 更新后的闭环结论

现在对用户怀疑的回答应更精确：

```text
怀疑成立的一面：
旧 mixed topology / dense 结果确实受到 source checkpoint copy、future iteration false skip、
dense gate 过严/过宽、topology mutation 混合、未训练 loc opacity prune 等高影响混杂影响。
这些混杂已经通过实现修复和 matched long-run 基本排除。

怀疑未完全支持的一面：
排除这些混杂后，dense pose-gate-only、split-only、active physical-prune-only 都没有给出稳健正向 precision 证据。
所以目前不能说“原始主张已被证明”；更合理的状态是“旧结果不能证伪主张，但 clean matrix 也尚未支持强主张”。
```

## update4: seed 语义修复与 selective dense/prune 补跑

日期：2026-06-24

本节继续闭合用户指出的剩余两项：

1. 之前所谓 multi-seed 主要是 `query_split_seed`，不是严格的训练随机种子。
2. physical prune 的 active500 虽然证明机制能删点，但 mild/balanced 500-step 仍需补跑以判断策略强度。

### Seed 语义修复

确认问题：

```text
旧 root: /mnt/pool/sqy/stdloc_la_v03_full_length/{scene}/seed_{2025,2026,2027}/{scene}_v03
```

这些目录名里的 `seed_*` 实际更接近 query split repetition；旧 closure 不能当作真实 train-seed variance 证据。

已修复：

1. `train_locaware.py` 已在 `safe_state(args.quiet)` 之后调用 `seed_everything(args.train_seed)`。
2. `scripts/run_densekl_v03_cambridge.sh` 新增 `DENSEKL_TRAIN_SEED` 并传入 `--train_seed`。
3. `scripts/run_la_update2_dense_long_worker.sh` 和 `scripts/run_la_update2_long_worker.sh` 拆分：
   - `TRAIN_SEEDS`
   - `QUERY_SPLIT_SEEDS`
   - 新路径优先使用 `train_seed_${train_seed}/query_split_${query_split_seed}`；
   - 旧路径 `seed_${query_split_seed}` 只作为 legacy fallback。
4. 两个 summary 脚本已兼容新旧 layout，并输出：
   - `train_seed`
   - `query_split_seed`
   - `legacy_seed_layout`

新增回归测试：

```text
tests/test_full_script_args.py::FullRunScriptArgsTest::test_dense_kl_script_accepts_explicit_training_seed
tests/test_full_script_args.py::FullRunScriptArgsTest::test_la_update2_workers_separate_train_seed_from_query_split_seed
tests/test_la_update_summarizers.py
```

### Selective dense attr-cos0.3 100-step

输出：

```text
/mnt/pool/sqy/stdloc_la_update4_dense_attrcos03_100_v1/summary_final.json
```

配置：

```text
DENSEKL_STEPS=100
DENSEKL_POSE_GATE_MIN_TE=1.0
DENSEKL_POSE_GATE_MIN_AE=0.03
DENSEKL_ATTR_COSINE_THRESHOLD=0.3
DENSEKL_MIN_ELIGIBLE_ANCHORS=32
TRAIN_SEEDS=0
QUERY_SPLIT_SEEDS=2025 2026 2027
scenes = ShopFacade, KingsCollege, OldHospital
rows = 9
```

选择 `attr_cosine_threshold=0.3` 的依据：从 strict dense 长跑 TensorBoard 诊断看，p10 reconstruction cosine 大致在 0.21-0.49 区间，0.3 是保守的 selective attribution gate，能过滤明显低可信 episode，同时避免 `Loc=0`。

与 `no_mutation` 100-step matched continuation 对齐：

| Scene | n | mean dR5 | median dR5 | mean dR2 | mean dTE |
| --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege | 3 | +0.006803 | +0.005831 | 0.000000 | +0.220921 |
| OldHospital | 3 | -0.003663 | +0.005495 | +0.003663 | +1.237161 |
| ShopFacade | 3 | +0.009709 | +0.019417 | -0.012945 | -0.270468 |
| Overall | 9 | +0.004283 | +0.005831 | -0.003094 | +0.395872 |

R5 正负计数：

```text
6 positive / 1 zero / 2 negative
```

关键判断：

```text
attr-cos0.3 gate 排除了 dense loss 全零的混杂：9/9 行 loc_positive_iter_count 均为非零。
R5 有弱正向桶效应，但 R2 均值为负，median TE 均值明显变差，尤其 OldHospital。
因此这组 100-step 结果仍不能证明 selective dense feedback 对 sparse-only relocalization 稳健有效。
```

### Selective dense attr-cos0.3 500-step

输出：

```text
/mnt/pool/sqy/stdloc_la_update4_dense_attrcos03_500_v1
```

配置与 100-step 相同，但 `DENSEKL_STEPS=500`，并保存/评测 `100` 与 `500` 两个 checkpoint。该矩阵也使用 `TRAIN_SEEDS=0`、`QUERY_SPLIT_SEEDS=2025 2026 2027`。

| Steps | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 100 | 9 | +0.004283 | +0.005831 | 6 / 1 / 2 | -0.003094 | +0.395872 |
| 500 | 9 | -0.010128 | 0.000000 | 3 / 2 / 4 | +0.010138 | +1.291585 |

500-step 分 scene：

| Scene | n | mean dR5 | mean dR2 | mean dTE |
| --- | ---: | ---: | ---: | ---: |
| KingsCollege | 3 | +0.007775 | -0.000972 | +0.659104 |
| OldHospital | 3 | -0.021978 | +0.005495 | +3.224045 |
| ShopFacade | 3 | -0.016181 | +0.025890 | -0.008394 |

结论：

```text
500-step 后 attr-cos0.3 的表现变成 R2 正、R5 负、TE 明显变差。
这说明 selective dense feedback 可能在个别场景/阈值上提高严格 precision bucket，
但没有提升整体 sparse-only relocalization，且 OldHospital 出现系统性退化。
```

### Train-seed smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update4_dense_attrcos03_trainseed_smoke_v1/summary_final.json
```

矩阵：

```text
scene = ShopFacade
query_split_seed = 2025
train_seed = 1, 2
steps = 100
```

与已完成的 `train_seed=0` 对齐：

| train_seed | R5 | R2 | median TE | dR5 vs no_mutation | dR2 vs no_mutation | dTE vs no_mutation | loc-positive iters |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.766990 | 0.291262 | 2.781322 | -0.019417 | -0.048544 | -0.334845 | 6 |
| 1 | 0.747573 | 0.330097 | 2.899633 | -0.038835 | -0.009709 | -0.216534 | 6 |
| 2 | 0.766990 | 0.330097 | 2.978816 | -0.019417 | -0.009709 | -0.137351 | 0 |

结论：

```text
真实 train_seed 会影响结果，但这个 smoke 没有给出正向证据。
train_seed=2 的 loc-positive iters 为 0，说明 selective gate 在真实训练随机性下仍可能采不到有效 dense supervision。
旧 query-seed 矩阵不能替代 train-seed variance；后续若继续证明 dense 主张，需要显式 train_seed x query_split_seed 分解。
```

### physical-prune mild/balanced 500-step

输出：

```text
/mnt/pool/sqy/stdloc_la_update4_prune_mb500_v1/summary_final.json
/mnt/pool/sqy/stdloc_la_update4_prune_mb500_old_v1/summary_final.json
/mnt/pool/sqy/stdloc_la_update4_prune_mb500_combined_summary_final.json
```

说明：ShopFacade/KingsCollege 使用主 root；OldHospital 为了并行提速使用独立 root。主 root 中有一个被主动中断的 OldHospital duplicate run，没有 final loc_state，不纳入 combined summary。

合并后矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
query_split_seed = 2025, 2026, 2027
tags = prune_mild_500, prune_balanced_500
rows = 18
```

总体结果：

| Tag | n | point delta / seed | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| prune_mild_500 | 9 | scene-dependent | -0.000867 | -0.002915 | 2 / 2 / 5 | +0.005212 | +0.194086 |
| prune_balanced_500 | 9 | scene-dependent | -0.000723 | 0.000000 | 2 / 3 / 4 | +0.004133 | +0.222194 |

按 scene 展开：

| Tag | Scene | n | point delta / seed | mean dR5 | mean dR2 | mean dTE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prune_mild_500 | KingsCollege | 3 | -1,780 | -0.004859 | -0.000972 | +0.353731 |
| prune_mild_500 | OldHospital | 3 | -3,169 | +0.005495 | +0.003663 | +0.259827 |
| prune_mild_500 | ShopFacade | 3 | -1,006 | -0.003236 | +0.012945 | -0.031298 |
| prune_balanced_500 | KingsCollege | 3 | -16,851 | -0.005831 | -0.000972 | +0.295086 |
| prune_balanced_500 | OldHospital | 3 | -33,733 | +0.003663 | +0.003663 | +0.459569 |
| prune_balanced_500 | ShopFacade | 3 | -8,906 | 0.000000 | +0.009709 | -0.088072 |

结论：

```text
mild/balanced 500 在 18/18 行都实际删点，机制混杂已排除。
相比 active500，它们更温和，且 R2 均值小幅正。
但 R5 均值仍接近 0 且略负，TE 均值变差，KingsCollege 对 prune 尤其不友好。
因此 physical prune 的“策略有效性”仍未被证明，只能说当前 mild/balanced 阈值不会造成灾难性退化。
```

## update5: full-bank false-negative ignore 与 split-only 重跑

本轮继续闭合 LA_update2 中 descriptor 主路径的一个高影响混杂：

```text
full-bank hard negatives 会把 3D/2D 近邻当作负样本，可能惩罚同一局部结构的合理多解。
```

### 新增修复

1. `localization_training/direct_landmark_teacher.py`
   - `full_bank_descriptor_stats()` 支持 `ignore_bank_mask`，训练 loss 与诊断 margin 使用同一 false-negative mask。
   - `direct_landmark_teacher()` 新增：
     - `full_bank_ignore_3d_radius`
     - `full_bank_ignore_uv_radius`
   - full-bank ignore mask 现在合并三类 false negative：
     - stable source sibling；
     - 3D 半径内近邻；
     - 同 query pose 下投影 UV 半径内近邻。
   - `DirectLandmarkTeacherOutput` 新增 `diagnostics`，训练时会向 TensorBoard 记录：
     - `full_bank_query_count`
     - `full_bank_bank_count`
     - `full_bank_valid_positive_count`
     - `full_bank_potential_negative_count`
     - `full_bank_ignore_negative_count`
     - `full_bank_effective_negative_count`
     - `full_bank_ignore_negative_ratio`
2. `train_locaware.py`
   - 新增命令行参数：
     - `--loc_full_bank_ignore_3d_radius`
     - `--loc_full_bank_ignore_uv_radius`
3. v03/topology 脚本
   - `scripts/run_locaware_v03_shopfacade.sh`
   - `scripts/run_locaware_v03_topology_full.sh`
   - 默认半径仍为 `0.0`，可通过环境变量显式打开：
     - `V03_FULL_BANK_IGNORE_3D_RADIUS`
     - `V03_FULL_BANK_IGNORE_UV_RADIUS`
     - `TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS`
     - `TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS`

### 单元验证

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_direct_landmark_teacher -v
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_train_locaware_masks tests.test_full_script_args -v
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh scripts/run_la_update2_long_worker.sh
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile train_locaware.py localization_training/direct_landmark_teacher.py
```

结果：

```text
direct teacher: 12/12 passed
parser/script args: 47/47 passed
bash -n / py_compile: passed
full unittest with CUDA_HOME=/usr/local/cuda-11.8: 151/151 passed
```

### 10-step smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update5_desc_ignorefn_smoke_v1/summary_final.json
```

配置：

```text
scene = ShopFacade
query_split_seed = 2025
mode = no_mutation
steps = 10
TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS = 0.1
TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS = 2.0
```

结果：

| Scene | Steps | R5 | R2 | median TE | point delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 10 | 0.776699 | 0.320388 | 3.008794 | 0 |

结论：新参数路径能跑通，并确认 `train_locaware.py` 收到了 `--loc_full_bank_ignore_3d_radius 0.1 --loc_full_bank_ignore_uv_radius 2.0`。

### 100/500-step multi-scene matrix

输出：

```text
/mnt/pool/sqy/stdloc_la_update5_desc_ignorefn_core_v1/summary_final.json
```

矩阵：

```text
scenes = ShopFacade, KingsCollege, OldHospital
train_seed = 0
query_split_seed = 2025, 2026, 2027
modes = no_mutation, split_only
steps = 100, 500
rows = 36
TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS = 0.1
TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS = 2.0
```

完整性检查：

```text
logs = 36
iteration_30600 checkpoints = 18
iteration_31000 checkpoints = 18
GPU processes after completion = none
```

总体绝对指标：

| Tag | n | mean R5 | median R5 | mean R2 | mean TE | mean point delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core_no_mutation_100 | 9 | 0.276467 | 0.043956 | 0.086300 | 11.885187 | 0.0 |
| core_no_mutation_500 | 9 | 0.277323 | 0.060440 | 0.093993 | 11.687521 | 0.0 |
| core_split_only_100 | 9 | 0.278119 | 0.049451 | 0.093851 | 11.910717 | 61.0 |
| core_split_only_500 | 9 | 0.279587 | 0.065934 | 0.093993 | 11.677187 | 376.9 |

与旧 `stdloc_la_update2_long_closure_v2` 的 matched tag 对照：

| Tag | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| core_no_mutation_100 | 9 | +0.002770 | 0.000000 | 4 / 2 / 3 | -0.013087 | -0.358017 |
| core_no_mutation_500 | 9 | -0.002234 | 0.000000 | 2 / 3 / 4 | +0.004602 | +0.192469 |
| core_split_only_100 | 9 | +0.009490 | +0.005831 | 5 / 1 / 3 | +0.001223 | -0.374848 |
| core_split_only_500 | 9 | -0.002880 | -0.002915 | 2 / 2 / 5 | -0.002013 | +0.085881 |

与同一 run 的 no-mutation 对照：

| Steps | n | mean dR5 | median dR5 | R5 pos/zero/neg | mean dR2 | mean dTE |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 100 | 9 | +0.001652 | 0.000000 | 4 / 3 / 2 | +0.007551 | +0.025530 |
| 500 | 9 | +0.002264 | 0.000000 | 4 / 1 / 4 | 0.000000 | -0.010335 |

按 scene 的 split-only minus no-mutation：

| Steps | Scene | mean dR5 | mean dR2 | mean dTE | mean children_added |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | KingsCollege | -0.001944 | 0.000000 | -0.057947 | 143.3 |
| 100 | OldHospital | +0.003663 | 0.000000 | +0.137368 | 84.0 |
| 100 | ShopFacade | +0.003236 | +0.022654 | -0.002832 | 138.7 |
| 500 | KingsCollege | -0.002915 | 0.000000 | +0.100077 | 914.7 |
| 500 | OldHospital | 0.000000 | 0.000000 | -0.165235 | 484.0 |
| 500 | ShopFacade | +0.009709 | 0.000000 | +0.034154 | 862.7 |

### update5 判断

```text
full-bank 3D/UV false-negative ignore 的实现混杂已经闭合：路径可运行、测试覆盖、36 行矩阵完整。
```

但精度结论仍然谨慎：

1. 与旧 matched tag 对照，`core_split_only_100` 有小幅正向：R5 `+0.00949`、R2 `+0.00122`、TE `-0.37485`。
2. 与同 run no-mutation 对照，split-only 的提升更小：100-step R5 `+0.00165`，500-step R5 `+0.00226`，median dR5 均为 `0`。
3. 500-step matched tag 对照仍为负：`core_split_only_500` mean dR5 `-0.00288`，R5 pos/zero/neg 为 `2 / 2 / 5`。
4. scene 依赖明显：ShopFacade split-only 较友好，KingsCollege 在 100/500 都为负，OldHospital 主要改善 TE 而非 R5。

因此，这轮结果支持：

```text
之前 descriptor full-bank hard-negative 确实存在 false-negative 混杂；
修复后 split-only 有更干净的小正向信号；
但该信号仍弱且不稳健，不能作为 LA-STDLoc 精度主张的强证据。
```

下一步不应继续无目标扫 split/prune 阈值。direct teacher 诊断已经补上，用于后续确认 ignore 半径是否过度屏蔽。更合理的后续方向是按 LA_update2.md 进入方法层调整：

1. 做真正的 multi-positive descriptor objective，而不是只 mask false negatives。
2. 评估 3DGS novel/perturbed-view query supervision 是否能补足当前只用原始相机视角的问题。
3. 对 topology mutation 改成 held-out risk commit 或 localization-only overlay map，避免直接改主 Gaussian map 造成不稳定。

## update6: full-bank multi-positive objective

本轮落实 update5 末尾的第一项：把 full-bank 的 3D/UV/source 近邻从“只忽略 false negative”扩展为可选的多正样本目标。

### 新增修复

1. `localization_training/direct_landmark_teacher.py`
   - `full_bank_bimnn_loss()` 新增 `positive_bank_mask`。
   - query-to-bank loss 从单一正样本 CE 改为对多正样本 logits 做 `logsumexp`。
   - hard negative mining 会排除所有正样本。
   - `full_bank_descriptor_stats()` 的 positive probability / margin 也按多正样本统计。
   - full-bank 诊断新增：
     - `full_bank_positive_count`
     - `full_bank_extra_positive_count`
     - 修正后的 `full_bank_effective_negative_count`
2. `direct_landmark_teacher()` 新增 `full_bank_nearby_as_positive`。
   - 关闭时维持 update5 行为：source sibling、3D 近邻、UV 近邻作为 ignore mask。
   - 开启时这些 related entries 进入 `positive_bank_mask`，作为同一个 query anchor 的多正样本。
3. `train_locaware.py` 新增命令行：
   - `--loc_full_bank_nearby_as_positive`
4. v03/topology 脚本新增环境变量：
   - `V03_FULL_BANK_NEARBY_AS_POSITIVE=1`
   - `TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE=1`

### 验证

```text
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_direct_landmark_teacher -v
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest tests.test_train_locaware_masks tests.test_full_script_args -v
PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m py_compile train_locaware.py localization_training/direct_landmark_teacher.py
bash -n scripts/run_locaware_v03_shopfacade.sh scripts/run_locaware_v03_topology_full.sh
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-} PYTHONPATH=/root/STDLoc /root/miniconda3/envs/ulfloc_repro/bin/python -m unittest discover -s tests
```

结果：

```text
direct teacher: 14/14 passed
parser/script args: 47/47 passed
py_compile / bash -n: passed
full unittest with CUDA_HOME=/usr/local/cuda-11.8: 153/153 passed
```

### 10-step smoke

输出：

```text
/mnt/pool/sqy/stdloc_la_update6_multipos_smoke_v1/summary_final.json
```

配置：

```text
scene = ShopFacade
train_seed = 0
query_split_seed = 2025
mode = no_mutation
steps = 10
TOPOLOGY_FULL_BANK_IGNORE_3D_RADIUS = 0.1
TOPOLOGY_FULL_BANK_IGNORE_UV_RADIUS = 2.0
TOPOLOGY_FULL_BANK_NEARBY_AS_POSITIVE = 1
```

结果：

| Scene | Steps | R5 | R2 | median TE | point delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 10 | 0.766990 | 0.320388 | 3.032771 | 0 |

对照 update5 ignore-only 10-step smoke：

| Variant | R5 | R2 | median TE |
| --- | ---: | ---: | ---: |
| update5 ignore-only | 0.776699 | 0.320388 | 3.008794 |
| update6 multi-positive | 0.766990 | 0.320388 | 3.032771 |

结论：

```text
multi-positive full-bank objective 的实现路径已跑通，且不会立即造成 smoke 级别崩坏。
但当前只有 ShopFacade 10-step no-mutation 单点结果，不能据此判断 multi-positive 是否优于 ignore-only。
```

下一步若继续沿 descriptor 主路径推进，应跑：

```text
3 scenes x query_split_seed=2025/2026/2027 x steps=100/500 x no_mutation/split_only
```

并与 update5 ignore-only 的 36 行矩阵做 matched delta。
