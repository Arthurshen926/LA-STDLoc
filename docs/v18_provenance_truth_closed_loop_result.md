# V18 provenance truth 与自定位闭环修复结果

日期：2026-08-30  
场景：StMarysChurch  
冻结基图：V2 完整重建 M0，164,871 Anchors，SHA256 `711855ea...`

## 最终判断

本轮按外部讨论的顺序，先实现了“完整高斯来源匹配真值”，再修复责任分解、可逆 active set、控制安全边界、动作定向 planner、共享低秩 metric 和部署门。实现和验证揭示了三个必须保留的结论：

1. 当前 Gaussian primitive provenance 不能唯一恢复 Projective Track/Anchor 真值。加入 Query 几何后，独立 mapping-family validation 的 decisive precision 仍只有 69.10%，低于预注册的 99% 替换门。因此 provenance truth 不得进入 controller 或 metric；Top-L 投影关系只能作为显式标注的诊断 fallback，也不能冒充全图真值。
2. 11,432-Anchor mapping-only 核心不具备新视角定位充分性。独立 95-query confirmation 上，相对 Full V2 的 R5 从 84.21% 降至 78.95%，失败 15→20，灾难样本 8→14，65/95 查询退化。几何覆盖充分不等于 descriptor competition 和 PoseLib 充分。
3. 设计集上恢复 Anchor 15459 的正收益不迁移。它在独立 91-query control 上是 `NO_ACTION`，在 95-query confirmation 上是 `NOT_CONFIRMED`。修复后的正式闭环会在 control 阶段拒绝它，不再把 candidate map 当成已部署结果。

所以，本轮让闭环“work”的含义是：observer、controller、control 和 confirmation 的职责已经闭合，且系统能够可靠拒绝不泛化的动作；但尚未得到一个可部署的正向地图修改。当前正式稳定状态仍是 Full V2 M0，而不是 b12k、15459 reactivation 或 metric action。

## 1. 最后一轮建议：匹配真值

### 1.1 已实现

- 全 460,299 Gaussian prior 的 depth-ordered composition provenance；每个 mapping observation 保存累计质量不低于 95% 的 primitive composition。
- mapping observation → Anchor signature、primitive inverted index、跨独立 mapping family 的 observation transport。
- `UNIQUE / EQUIV / AMBIG / NONE` 真值图与 descriptor Top-L competition graph 分离。
- signature design、threshold calibration、independent validation 按 mapping sequence family 60/20/20 隔离；未使用 LOO、test pose 或 test RGB。
- Query 端增加 reprojection、normalized depth residual、projection covariance 和 view-family transport gates。
- 只有通过独立 validation 的 provenance truth 才能授权替换 controller 真值；失败时 fail closed。

### 1.2 验证结果

独立 validation 共 4,000 个 observation rows：

| 真值构造 | decisive coverage | decisive precision | 正确 / decisive |
|---|---:|---:|---:|
| Full-map projection baseline | 36.88% | 95.66% | 1411 / 1475 |
| Provenance only | 62.75% | 51.31% | 1288 / 2510 |
| Provenance + transport + Query geometry | 61.40% | 69.10% | 1697 / 2456 |

结论：primitive provenance 对“由哪些高斯渲染出来”有效，但 Gaussian primitive 与 Projective Track 不是一一对应关系；共享表面、重叠 splats、track 聚合与跨视角混合使它不能直接充当 correspondence identity。替换门为 `controller_replacement_authorized=false`。

显式 fallback 仍保留用于诊断，合同固定为：

- `uses_topl_candidates=true`
- `descriptor_independent_full_map_truth=false`
- `controller_replacement_authorized=false`
- controller/observer 必须显式传入 `--allow-certified-topl-fallback`

## 2. 前两轮建议：observer 与 controller

### 2.1 Counterfactual responsibility observer

每个错误 decisive correspondence 分别重放：删除 query row、删除当前错误 Anchor、替换为 truth Anchor、全 truth oracle。它只回答“哪个有界操作改变标准 PoseLib plant”，不把查询或 Anchor 直接贴成坏样本。

在 138 张 ACCEPT design queries 上，fallback 诊断得到：

- decisive wrong rows：20,731
- 实际逐行审计：3,328
- 删除 row 可控：129
- metric 可控：203
- geometry-limited queries：8
- coverage-limited rows：150,147
- 跨 pose-family 可删除 Anchors：11

这说明“匹配错很多”并不等于“有很多安全地图动作”；大多数误差没有被当前 action space 控制。

### 2.2 可逆 current-state controller

已修复：

- proposal 每轮从当前 competition graph 重算，不复用历史删除授权。
- active set 支持 deactivate/reactivate；inactive truth reactivation 的优先级不再被 inactive redundancy 截断。
- safety floor 检查 Anchor-unique count、spatial cells、pose information logdet/eigenvalue、effective correspondences 和 Top-L exhaustion。
- Top-L floor 只禁止“新产生”的 exhaustion，允许从已有压缩初态逐步修复。
- action proposal 与正收益均按 pose family 三折 crossfit；没有重新引入 LOO。
- 地图规模只作为安全约束后的次级奖励，不允许硬预算覆盖定位风险。

设计侧 best-of-16 找到 Anchor 15459 reactivation：11,432→11,433 Anchors，旧 design records 上 bounded gain `+4.2591`，10 个 pose families 改善、0 lost success。但它在独立 control 上只有 3 改善/2 退化，R5 不变，bootstrap lower-risk probability 0.1825，因此正式决定为 `NO_ACTION`。

### 2.3 三段式 fail-closed 部署门

闭环现在固定为：

1. design：只生成和拟合 frozen action；
2. control：只决定该 frozen action 是否进入 confirmation；
3. confirmation：只做一次独立复现，不训练、不调参、不选臂；
4. control 或 confirmation 任一失败，部署状态自动保留 previous stable map。

追溯应用该 gate 后：

- `reactivate_15459`：`REJECT_CONTROL`；
- 历史 `b160k_feedback_conditioned`：control 曾通过，但 confirmation 失败，`REJECT_CONFIRMATION`；
- 两者都不能替换 stable map。

## 3. 新视角充分性与全局压缩

本轮使用 288 个冻结的 continuous-SE(3) candidate poses，从中预先选择 96 个 confirmation poses；V2 Render Certificate 得到 95 ACCEPT、1 REJECT。没有使用 test、LOO 或 source mapping RGB。

对 b12k 的独立结果：

| 地图 | Anchors | R5 | q50 task | q75 task | catastrophic |
|---|---:|---:|---:|---:|---:|
| Full V2 | 164,871 | 84.21% | 0.2485 | 0.5801 | 8 |
| b12k mapping-only | 11,432 | 78.95% | 0.3964 | 0.8526 | 14 |
| b12k + Anchor 15459 | 11,433 | 78.95% | 0.3964 | 0.8526 | 14 |

因此 93.07% 的压缩率当前只能作为 mapping coverage candidate，不能作为 localization-sufficient 主线 operating point。历史上即使更保守的 b160k（仅约 2.95% 压缩）也在 control 上通过、在独立 confirmation 上失败；现阶段不能声明任何全局压缩 operating point 已被正式确认。

## 4. Action-targeted planner

旧实现把“Anchor 投影在视锥内”当作 intervention exposure。追溯发现 37 个所谓 intervention queries 中，只有 7 个查询的 Top-1 实际受 reactivation 影响，说明几何可见不等于 descriptor competition exposure。

已修复为：

- intervention：action Anchor 可见，并位于该 Anchor 原 mapping observations 的方向支持锥内；删除动作还需同时看见 certified backup；
- necessity：从原观测支持方向看见待删除 Anchor；
- collateral：独立的全局新视角；
- 没有 harmful deletion 时不虚构 necessity quota；不足部分明确回填 collateral。

在同一 288-pose 冻结候选池上，方向门只留下 33 个真正方向受支持的 intervention candidates；旧几何门选择了 38 个。由于 15459 已在 control 被拒绝，修复后没有为它继续消费新的 confirmation。

## 5. Shared low-rank metric

已实现 query/map 共享的有界低秩 metric，而不是只改 query 或只复制 feedback descriptor：

- pose-family balanced repair rows；
- 大规模 clean protection rows；
- residual norm 上界和 identity interpolation `alpha`；
- active-set + metric 联合 exact Top-1/PoseLib replay；
- control 选最小有效 alpha，confirmation 固定该 alpha。

训练证据很弱：21 repair rows / 13 families，对 23,928 protection rows / 113 families；最终 pair accuracy 4.76%，median margin -0.1266。control 选择 `alpha=0.025 + V16 active`，R5 84.62%→85.71%，q75 0.5143→0.4967，bootstrap lower-risk probability 0.9515。

独立 95-query confirmation 上：

- R5：84.21%→84.21%；
- q75：0.5801→0.5447；
- q50：0.2485→0.2556；
- maximum single-query regression：0.1239；
- bootstrap lower-risk probability：0.736；
- Pareto relation：`TRADEOFF`，没有形成可部署确认。

所以 metric 模块保留为候选机制，但当前 action 不进入主线部署。

## 6. 当前完整方法流程

1. V2 有效渲染证据在 pairing/Track 之前逐 observation 过滤，完整重建 clean M0。
2. Full V2 M0 是唯一当前确认的 stable map。
3. mapping-only evidence 只能产生全局压缩候选，不能自行授权地图规模。
4. virtual feedback queries 从 continuous SE(3) 新位姿规划；V2 certificate 先做样本判定，再用 row-valid mask 过滤局部伪影区域。
5. truth graph 与 descriptor competition graph 分离；provenance truth 必须先过 mapping-family validation。
6. observer 用 counterfactual replay 输出 coverage / geometry / row suppression / Anchor suppression / metric responsibility。
7. controller 只允许当前状态下的 reversible delete/reactivate 或有界 shared metric，不增加 Anchor、不重新三角化、不使用 LOO。
8. action-targeted planner 为通过 control 的动作生成 intervention / necessity / collateral confirmation。
9. design → control → confirmation 逐级 fail closed；最终 test 只能在方法和 operating point 全部冻结后运行一次。

## 7. 当前主线结论与剩余缺口

已经成立：

- V2 observation filtering 进入配对前完整建图；
- matching truth 的替换合同、验证门和失败回退；
- 操作级 responsibility observer；
- current-state reversible controller 与安全 floors；
- action-targeted confirmation 与方向支持门；
- design/control/confirmation 部署门。

尚未成立：

- provenance-to-Track 的高精度 correspondence identity；
- 能在独立新视角上确认的全局压缩 operating point；
- 能在 independent confirmation 上复现的 Anchor delete/reactivate action；
- 能正式部署的 shared metric action；
- render-domain confirmation 到真实 test 的迁移结论。

下一轮不应增加新网络、prototype、matcher 或 controller 复杂度。优先级应是：用 Full V2 作为稳定初态，扩大 descriptor-independent truth 的 identity 信息（必须超越 primitive overlap），同时把 control gate 前移到每个地图动作；只有出现独立 control 的实质效应后才生成新的 action-targeted confirmation。真实 test 仍未使用。

## 8. 主要产物

- provenance validation：`/mnt/pool/sqy/lafgs_v18_provenance_truth_20260829/StMarysChurch/provenance_truth_validation_geometry.json`
- responsibility audit：`/mnt/pool/sqy/lafgs_v18_provenance_truth_20260829/StMarysChurch/responsibility_certified_fallback.json`
- metric confirmation：`/mnt/pool/sqy/lafgs_v18_provenance_truth_20260829/StMarysChurch/confirmation_metric_decision.json`
- reactivation control / confirmation：`control_reactivation_decision.json` / `confirmation_reactivation_decision.json`
- global sufficiency confirmation：`confirmation_global_sufficiency_decision.json`
- deployment gates：`reactivation_deployment_gate.json` / `global_compression_deployment_gate.json`
- direction-gated planner audit：`action_targeted_plan_direction_gated.json`

代码回归：Ruff 全通过；相关 32 个 pytest 全通过；`git diff --check` 全通过。
