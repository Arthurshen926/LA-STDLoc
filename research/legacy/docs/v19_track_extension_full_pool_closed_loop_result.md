# V19 Track Extension、Full-pool 审计与闭环修复结果

日期：2026-08-30  
场景：StMarysChurch  
稳定地图：V2 Full M0，164,871 Anchors，SHA256 `711855ea...`

## 总结

本轮按外部建议的 P0→P6 顺序推进，修正了匹配真值语义，实现了新视角 Track 延伸教师、分级置信授权、Full-pool Sufficiency Audit，以及 mapping Track identity 低秩 Metric 基础训练和 exact PoseLib control/confirmation。没有使用 test、LOO、真实 RGB 或 feedback 写 Track。

最终没有产生可部署地图动作，但得到了两个比 V18 更明确的结构性结论：

1. b12k 的新视角失败主要包含真实的 active-set selection loss。高置信 Track 行中 7,710/11,351 的真 Anchor 不在 b12k；在 Full oracle 可成功的 113 个 query 中，b12k 因选图失去 59 个 oracle 成功。因此 mapping-only 几何核心不能作为定位充分核心。
2. Full M0 的剩余可控缺口主要表现为 descriptor competition。在 11,351 个高置信行中，393 行是真 Anchor 不在 Top-64，2,989 行是真 Anchor 已在 Top-64 但没有赢，7,969 行 Top-1 正确。现阶段优先研究 metric/competition 比继续全局删图更合理。

闭环最终 fail closed：保留 Full V2 M0 + identity metric；不部署 b12k、不新增 Anchor、不部署本轮低秩 Metric。

## 1. P0：评价标签与推理输入分离

所有 V19 artifact 明确记录：

```text
reference_source: mapping_observation_track_membership
reference_available_for_novel_query: false
feedback_enters_track_registry: false
selection_uses_validation: false
loo_used: false
```

mapping observation 的 Track membership 只用于 held-out 教师评价。新视角 query 没有可直接读取的 Anchor 真值；教师只能由冻结的 mapping Tracks 选择性延伸，不能把评价标签伪装成线上输入。

## 2. P1：Novel-view Track Extension Teacher

实现流程：

1. 对 query keypoint 使用渲染 depth 反投影表面点；
2. 扫描完整 164,871-Anchor 地图构造投影/深度/协方差候选，不使用 descriptor Top-L；
3. 将表面点 transport 到候选 Track 的冻结 mapping observations；
4. 使用独立 view-family 几何一致性与 native descriptor 一致性确认 Track；
5. 输出 `UNIQUE / EQUIVALENT / AMBIGUOUS / NONE`，不向 Track CSR 写回 feedback。

关键真值修复：如果完整投影候选包含多个 identity class，即使 observation bank 只支持其中一个，也必须输出 `AMBIGUOUS`。缺少某 Track 的 bank 支持是“未知”，不是该 Track 为假的证据。

同时修复了一个性能漏洞：旧实现对每张 query 都重新遍历 147 万条 bank observations 并重新拼接 mapping descriptors。现在每个 shard 只构造一次 family-balanced prepared Track bank。229-query 三卡物化由预计数十分钟降到约两分钟，算法证据不变。

## 3. P2：precision-coverage 与授权层级

校准/验证按 mapping sequence family 隔离。旧的 pooled precision 选择会被单一序列掩盖：宽松档位在 calibration 是 147/147，但决定性标签全部来自 family 10；在独立 families 上为 210/217。

修复后的授权门同时要求：

- pooled precision 的单侧 95% Wilson 下界；
- 最小决定性样本数；
- 最小 active mapping-family 数；
- validation 不参与阈值选择。

正式结果：

| Tier | 用途 | calibration | independent validation | 授权 |
|---|---|---:|---:|---|
| A | destructive map control | 150/150，1 active family，Wilson LB 98.23% | 212/219，96.80% | 否 |
| B | strong metric control | 同上 | 同上 | 否 |
| C | soft diagnostic / planner | 同上 | 同上 | 是 |

Tier C validation decisive coverage 为 5.48%。它可用于诊断和 planner priority，但不能删除 Anchor、强更新 Metric 或证明需要新增 Anchor。

在 229 个 ACCEPT feedback queries 上，共审计约 37.2 万个 V2-valid rows：11,351 decisive，112,635 ambiguous，其余为 NONE。低 coverage 是主动弃权的结果，不应改写成 coverage deficit。

## 4. P3：Full-pool Sufficiency Audit

审计逐行区分：

- selection deficit：Full truth 存在但 active map 中不存在；
- retrieval miss：active truth 存在但不在 Top-L；
- competition miss：truth 在 Top-L 但 Top-1 错；
- correct winner；
- unresolved identity；
- unresolved empty projection；
- certified candidate-pool deficit。

只有 Tier A 授权时，空投影才可升级为 candidate-pool deficit。本轮 Tier A 未授权，因此 191,000 个 empty-projection rows 均保持 unresolved，不能据此实现 Anchor Addition。

| 指标 | Full M0 | b12k active map |
|---|---:|---:|
| decisive rows | 11,351 | 11,351 |
| selection deficit | 0 | 7,710 |
| retrieval miss | 393 | 125 |
| competition miss | 2,989 | 789 |
| correct winner | 7,969 | 2,727 |
| Full oracle success / 229 | 113 | 113 |
| active oracle success / 229 | 113 | 56 |
| active selection-loss queries | 0 | 59 |
| certified candidate-pool deficit | 0 | 0 |

Full oracle 只使用稀疏的 Tier C decisive rows：76 个 query 不足 4 条对应，40 个即使有足够对应仍未定位成功。因此 113/229 不是 Full M0 的整体定位率，而是当前高精度教师覆盖下的 oracle 下界。它不能与使用全部特征的 baseline 181/229 直接比较。

## 5. P4：Mapping Track identity Metric 基础训练

使用 7 个 track-bank mapping families 的 19,999 条 observation→Track identity 排序证据训练 rank-16、最大 residual norm 0.05 的 query/map 共享低秩 Metric。没有使用 feedback 或 test。

跨 family 结果：

| Split | Raw Top-1 | Metric Top-1 | wrong→truth | truth→wrong |
|---|---:|---:|---:|---:|
| calibration | 85.175% | 85.375% | 14 | 6 |
| independent validation | 87.000% | 87.100% | 13 | 9 |

这是小幅、可复现的正向 identity evidence，但保护仍不充分，不能只凭 retrieval 指标部署。

## 6. Exact PoseLib control 与 confirmation

在 91 个独立 control queries 上固定测试 `alpha={0.25,0.5,1.0}`：

- alpha 1.0：严重尾部回退，拒绝；
- alpha 0.5：未过 P90 和单 query 安全门，拒绝；
- alpha 0.25：R5 84.62%→85.71%，灾难数不增，但 net gain `-0.0172`，仅为 Pareto candidate。

冻结 alpha 0.25 后，在 169 个 confirmation queries 上一次性复现：

- R5：89.94%→89.94%；
- q50：0.21993→0.21871；
- q75：0.40462→0.39908；
- q90：1.01152→1.01846；
- net gain：`-0.44625`；
- lower-risk probability：0.4345；
- classification：`ANALYSIS_ONLY`。

因此 Metric 不部署。这个结果也说明：大规模 mapping identity 基础训练修复了 V18 “只有 21 repair rows”的证据稀疏问题，但 mapping identity accuracy 的微小提高仍不足以稳定改善新视角 PoseLib。

## 7. 当前闭环的完整职责

1. V2 在 pairing/Track 前过滤无效 observation，生成 Full M0。
2. Track Extension Observer 只产生分级诊断，不写 Track。
3. Full-pool Audit 回答现有 Anchor 是否存在，并把 selection/retrieval/competition/unresolved 分开。
4. Controller 只在相应 Tier 授权后开放 delete/reactivate/swap、强 Metric 或 Add。
5. Metric 先做 mapping identity pretraining，再允许高置信 feedback correction；本轮 Tier B 未授权，所以没有反馈微调。
6. 每个候选动作必须经过 exact Top-1 + PoseLib control，再用冻结动作做 confirmation。
7. 任一真值、control 或 confirmation 门失败，保持上一个 stable map/metric。

正式部署决定：

```text
stable map: Full V2 M0
metric: identity
map mutation: none
b12k: rejected (59 oracle selection-loss queries)
Anchor Addition: not authorized
mapping identity Metric: not confirmed
```

## 8. 附件建议落实状态

- P0：完成。
- P1：完成，并修复缺失 Track support 的假唯一问题。
- P2：完成；A/B 不通过，C 只授权软诊断。
- P3：完成；证明 b12k selection loss，尚无新增 Anchor 正证据。
- P4：完成基础训练与独立 family 验证。
- P5：未执行反馈微调，因为 Tier B 未授权；这是安全门的预期行为。
- P6：控制路由与最终部署门完成；本轮无动作通过 confirmation。
- P7：未实现 Anchor Addition，因为 P3 没有 certified candidate-pool deficit。

## 9. 主要产物

- Teacher validation：`/mnt/pool/sqy/lafgs_v19_track_extension_20260830/StMarysChurch/teacher_validation_safe.pt`
- Novel teacher shards：`feedback_teacher_shard{0,1,2}.pt`
- Full M0 audit：`full_pool_sufficiency_m0.pt`
- b12k audit：`full_pool_sufficiency_b12k.pt`
- Mapping identity Metric：`mapping_identity_metric_r16/shared_metric.pt`
- Metric control：`metric_control_decision.json`
- Metric confirmation：`metric_confirmation_decision.json`
- Final deployment：`closed_loop_deployment_decision.json`

