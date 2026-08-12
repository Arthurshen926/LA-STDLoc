# Evidence-Grounded Anchor V4 收敛计划

## 结论与最终形式

Track 与 Gaussian 不应继续作为两类可部署地标并行扩张。更简洁、也更符合定位问题本质的形式是：

\[
\mathcal A = \{(I_i, O_i, X_i, \Sigma_i, d_i, e_i, s_i)\}_{i=1}^{N},
\]

其中每一行都是一个 **Evidence-Grounded Localization Anchor**：

- `I`：场景实体身份，优先由跨视图观测确定；
- `O`：真实图像观测引用；
- `X, Sigma`：统一几何及其不确定性；
- `d`：由真实观测聚合得到的单个部署描述子；
- `e`：Track、surface、Gaussian lineage 等证据，而不是地标类型；
- `s`：precision、matching completion 或 observability completion 等入选原因。

因此，正确的抽象不是 `Track anchors + Gaussian anchors`，而是：

> 观测身份定义 Anchor；Gaussian 提供表面、可见性和来源证据；一个统一选择器决定是否部署。

这保留了 Gaussian 对弱纹理和弱三角化区域的价值，同时消除“双地标地图”带来的重复身份、重复竞争和论文叙述冗余。

## 第一性原理约束

1. 定位器需要的是稳定的 `2D observation -> 3D identity` 对应，而不是某种上游表示的所有元素。
2. Track、Gaussian、surface 是证据来源；它们不能直接等价于身份类别。
3. 任何删除或合并都必须先证明两个候选指向同一实体。描述子相似只能作为审计量，不能单独触发合并。
4. 精度优化必须优先处理错误对应和身份混淆；继续增加候选、训练步数或 Pose Reserve 不能解决这个根因。
5. 所有超参数、去重阈值和 selector 约束只能由 mapping split 校准；test split 只在方案冻结后使用一次。

## 已确认的当前实现事实

- Coverage 与 Pose 已经在 `leftover_tracks_plus_gaussian_base` 共享候选宇宙、匹配状态和姿态状态；不存在必须再造一个“统一候选分支”的必要。
- 当前主要冗余位于候选构造、身份重叠、几何物化和 `anchor_type` 驱动的语义，而不在最终 PnP（仍是单次求解）。
- `pose_minimum_additions=64` 不能被直接解释成当前 Pose Reserve 的主因。Heads 的历史 V3 产物实际增加 511 行，明显不是被最小 64 行强制出来的；在统一 selector 与信息归一化完成前直接改成 0，不会验证核心假设。
- SuperPoint 的 mapping 与 deployment 实际都使用内部默认 `NMS radius=4`，而配置曾写为 2。V4 前置修复将真实行为显式冻结为 4，不改变已实现算法行为。
- 老 V3 产物通常只有阶段计数，没有每一行的精确 selection provenance。旧产物必须标记为 `legacy_unresolved`，不能从计数猜出逐行原因。

## 已落地的 V4 前置工作

### P0：冻结 V3 基线

`scripts/freeze_v3_baseline.py` 输出不可变清单，记录：

- 配置、Map 和指定父产物的路径与 SHA-256；
- `anchor_ids / anchor_xyz / anchor_features` 的逐 tensor 哈希；
- Anchor registry 哈希、Track/Gaussian fallback 数量；
- 声明 NMS 与历史实际 NMS；
- Git commit、工作树状态和 diff 哈希。

准入条件：后续任何 V4 实验必须引用一个 V3 manifest；若父产物哈希变化，实验立即失败，不允许把基线漂移误当成增益。

### P1：统一 Anchor Registry 兼容层

`topology/anchor_registry.py` 和 `scripts/materialize_anchor_registry.py` 已提供兼容视图：

- 原定位器消费的 ID、坐标、描述子、来源和依赖张量逐位复制；
- 新增 `identity_mode / geometry_mode / selection_reason / evidence_mask`；
- 将 Track 原生观测与 positive teacher 中的真实观测转成统一 Anchor CSR；
- 缺失的 reliability、matchability、alias risk 用 NaN 表示；
- 旧产物没有精确选择记录时明确标记 unresolved。

硬门槛：兼容校验必须 bitwise equal。P1 不允许改变地图大小、匹配、PnP 或精度。

Heads V3 真实产物兼容审计结果：

| 项目 | 数量 |
|---|---:|
| 最终 Anchor | 8119 |
| Track-verified identity | 7126 |
| Multi-view-supported identity（原 Gaussian fallback） | 993 |
| Weak fallback | 0 |
| 真实 observation edges | 159974 |
| 有真实观测支撑的 Anchor | 8119 |
| 逐行 selection reason 可精确恢复 | 0 |

所有定位张量通过 bitwise compatibility check。这个结果说明 993 个原 Gaussian 行并不是无观测冗余，不能按类型整体删除；是否与 Track 重复必须由 P2 等价图回答。该旧 payload 没有 surface-regularization 专用字段，因此本次兼容审计中的 geometry mode 只反映可证实的 legacy 状态，不据此判断 surface 方法效果。

### P1.5：显式前端契约

- mapping CLI、query-cache signature、训练状态、deployment frontend 与 deployment contract 全部记录 `nms_radius`；
- query-cache schema 升级，禁止把未声明 NMS 的旧 cache 静默复用；
- 主配置明确设为真实历史行为 `NMS=4`。

这是实验卫生修复，不作为算法增益申报。

## 最新附件落实状态（2026-08-12）

附件描述的是完整 V4 收敛目标，不是一次改动可以同时验收的单项功能。当前状态如下：

| 附件阶段 | 状态 | 已落实边界 | 尚未授权/完成 |
|---|---|---|---|
| Stage 0 冻结 V3 | 已落实 | manifest、文件/tensor hash、配置与工作树登记 | 各新场景仍需各自生成 manifest |
| Stage 1 统一 Candidate Registry | 已落实兼容层 | 统一 observations、identity/geometry mode、evidence、NaN unknown；定位张量 bitwise equal | 全量 pose replay parity 随真实 selector 重放验收 |
| Stage 2 Dedup/evidence transfer | 审计完成，物理合并 Stop | 等价图、协方差代理、functional audit、union-find component、无删除反事实 | 不实施删点和 evidence transfer；component 仅供风险/selector 使用 |
| Stage 3 统一 geometry materializer | 未实施 | 已统一 Registry 中的 geometry/covariance 语义 | adaptive surface 求解与 mapping gate 尚未通过 |
| Stage 4 统一 Sufficiency Selector | compatibility 完成；P5.1 Stop | 单一 selected state、统一 primary reason/trace；equal-gain alias tie-break 已在 Heads/Stairs/ShopFacade 完成 compact refresh 与 pose gate | 当前 alias tie-break 不部署；不再沿其做局部阈值扩展 |
| Stage 5 observation descriptor | P6.0 机制 Go；P6.1 单 medoid Stop | 分层融合审计完成；Stairs 三种 trajectory cross-fit 的六个方向均大幅损失 held-out R@1 | 不物化当前 raw observation medoid；多 prototype 延后到 P7 evidence 因子之后 |
| Stage 6 室内身份自适应 | density-only No-Go；pair 在 Stairs mapping Go、GreatCourt 机制 Stop | NMS=4 契约、all-candidate alias audit、baseline/parallax pair audit、K_mapping 独立配置、Stairs fullchain/mapping pose 与 GreatCourt 跨域反证 | K=2048 不进入默认；`parallax_diverse` 不升为跨域默认 |

因此答案明确：**附件内容尚未全部落实**。当前已经把它从方向性建议收敛为带硬门槛的阶段实现；任何未通过 mapping gate 的阶段都不会伪装成“已完成”。

## 后续主线及执行门槛

### P2：候选等价图审计（下一主线，只审计不删除）

输入为统一 Registry、Track observations、positive teacher、几何协方差和 Gaussian lineage。建立无向候选等价图，边必须同时满足：

\[
\operatorname{overlap}(O_i,O_j) \ge \tau_o,
\quad
(X_i-X_j)^T(\Sigma_i+\Sigma_j)^{-1}(X_i-X_j) \le \tau_g,
\quad
\operatorname{lineageCompatible}(e_i,e_j).
\]

落实项：

1. 分别统计 Track-Track、Track-Gaussian、Gaussian-Gaussian 的边数、连通分量大小和查询覆盖重叠。
2. 报告同一 query/keypoint 被多个候选竞争的频率，以及这些竞争是否进入错误匹配或 RANSAC harmful evidence。
3. `tau_o / tau_g` 由 mapping-only 正负候选对分布确定，写入场景校准产物，不设跨场景拍脑袋常数。
4. 描述子 cosine 只用于报告“视觉上有多像”，不得独立建边。
5. 首批场景：Heads、Stairs、Office2 或 Office5b、ShopFacade；确认信号后再扩到 Fire、OldHospital。

Go 条件：存在稳定的、跨阈值邻域不剧烈变化的等价分量，并且这些分量对重复竞争/harmful match 有可测贡献。

Stop 条件：若 Track-Gaussian 等价边极少，或分量对阈值极敏感、内部观测不一致，则“显式去重”不是主瓶颈，停止该分支，保留统一 Registry，直接转向 P5/P6 的 selector 与 alias risk。

#### P2 已完成的首轮审计

当前实现包括：

- `topology/anchor_equivalence.py`：只从完全相同的真实 `(query, keypoint)` 观测建立候选边，描述子相似度只报告、不建边；
- `topology/anchor_covariance.py`：用既有 mapping surface bounds 和 Gaussian scale/rotation 补齐只读协方差代理，不改变 Anchor 坐标、描述子或定位行为；
- `scripts/audit_anchor_equivalence.py`：输出候选边、阈值扫描、functional relevance、跨 query-group 支撑分量及可复现 component IDs。

| 场景 | 共享观测候选边 | 校准近邻边 | 跨 query-group 边 | 跨组 Anchor/分量 | 单实体化最大减行 | 跨组 harmful 端点边 |
|---|---:|---:|---:|---:|---:|---:|
| Heads | 862 | 304 | 162 | 245 / 99 | 146（1.80%） | 67 |
| ShopFacade | 157 | 126 | 51 | 86 / 38 | 48（0.76%） | 21 |

Heads 的跨组边关联 295 次已知 harmful events，ShopFacade 关联 47 次。两场景在距离阈值 0.5x/1x/2x 及 observation-containment 扫描中均保留高支撑核心，因此 P2 的“是否存在真实冗余信号”初始 Go 条件通过。另一方面，即使把所有跨组分量都压成单实体，地图最多只减少约 0.8%–1.8%；后续价值主要应来自消除错误竞争，而不是压缩率。

Stairs 的第三场景审计进一步确认“冗余存在”：2116 条共享观测候选边、549 条 mapping-calibrated triage 边、323 条跨 query-group 独立支撑边；后者覆盖 546 个 Anchor/244 个分量，单实体化上界为 302 行（4.15%），关联 1052 次已知 harmful events。全部候选边已有可计算 covariance proxy。该场景包含 70 个 `weak_fallback` identity，但 7275 个最终 Anchor 均存在真实 observation，因此不能按 fallback 标签整体删除。

Gaussian 协方差当前是 mapping-calibrated surface-prior proxy，不是经验定位后验。它已使 Heads、ShopFacade、Stairs 的全部 triage edges 可计算 Mahalanobis 距离，但仍不单独授权任何物理合并。

#### P2.5 无删除反事实结论

反事实严格区分三个 arm：

1. 当前部署匹配；
2. 仅抑制同一 Anchor 的重复 query 对应（multiplicity control）；
3. 在 control 基础上，每个审计 identity component 只保留最高分对应（entity folding）。

比较 3-2 才是 component folding 的净效应；Map、描述子、Top-1 winner、mapping query 与 PnP 参数均保持不变。Heads/ShopFacade/Stairs 各使用固定 96 个 mapping 查询和预注册 RANSAC seeds 2026/2027/2028：

| 场景 | component 对应删减 | raw P@2 变化 | harmful inlier 变化/seed | mean TE 平均变化 | CVaR95 平均变化 | 5cm recall | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| Heads | 120（0.176%） | +0.043 pp | -10 / -5 / -25 | +0.0186 cm | +0.4240 cm | 2/3 seeds -1.042 pp | **Stop physical dedup** |
| ShopFacade | 156（0.110%） | +0.018 pp | -7 / -3 / -5 | +0.0133 cm | +0.2479 cm | 不变 | Inconclusive；不足以 Go |
| Stairs | 136（0.248%） | +0.069 pp | -57 / -60 / -62 | +0.0082 cm | +0.1534 cm | 不变 | **Stop physical dedup** |

Heads 和 Stairs 三个种子均出现 CVaR95 回退且没有一个通过完整 pose gate；Heads 两个种子进一步损失 5cm/5° 召回。ShopFacade 只有一个种子通过完整 pose gate。结论不是“等价信息无效”，而是：

> 当前 component 更适合作为 alias-risk/selection evidence，不能被解释成可安全物理折叠的 landmark identity。

因此不进入 P3 evidence-transfer materialization，不删除任何 Anchor。该分支已及时停止，避免把微小 correspondence precision 增益换成室内尾部姿态风险。

### P3：证据转移式去重（P2 通过后才实施）

每个可信等价分量只保留一个实体身份：

- 有可靠 Track 身份时，Track identity 为主；
- Gaussian 的 surface、visibility、opacity、lineage 证据转移到该实体；
- 没有可靠跨视图身份但存在真实正观测的候选，保留为 `multi_view_supported`；
- 没有足够真实观测的候选只进入 weak fallback 池，不直接获得部署权。

去重不是简单删 Gaussian 行。物化器必须输出 `old_candidate -> entity -> final_anchor` 三段映射，以及每条被吸收证据的来源。Track/Gaussian 数量不再作为方法核心指标，核心指标改为：唯一实体数、观测覆盖、重复竞争率和 harmful rate。

硬门槛：

- mapping 查询的可匹配行 p10、姿态可观测性和有效空间覆盖不得低于 V3；
- 重复 landmark 竞争率必须下降；
- ShopFacade 与 OldHospital 不得出现 precision sentinel 回退；
- 门槛不满足则回退为“标注等价关系但不物理删除”。

### P4：统一几何物化

每个实体只运行一个 geometry materializer：

1. 图像三角化给出 `X_img, Sigma_img`；
2. surface evidence 给出沿法向/深度方向的软约束，而不是另造一个 Gaussian 点；
3. 由 mapping-only 残差估计约束置信度；强图像几何保持不动，弱深度方向才接受 surface regularization；
4. 输出统一 `X, Sigma, geometry_mode` 和证据贡献，不再在下游按 anchor type 分叉。

必须做三个因子：image-only、固定 surface 修正、adaptive surface regularization。主要检查重投影误差、三角化协方差、PnP inlier 与室内平移误差。若 adaptive 方案只改善几何代理量但不改善 mapping self-localization，或伤害 ShopFacade/OldHospital，则停止几何修订。

### P5：单一 Sufficiency Selector

将现有 Core/Coverage/Pose 理解为同一约束优化的三种入选原因，而不是三类地图：

\[
\min_{S \subseteq \mathcal A}|S|
\quad \text{s.t.}\quad
\operatorname{precision}(S) \ge p_0,
\operatorname{matchingCoverage}_q(S) \ge m_q,
\operatorname{poseInfo}_q(S) \ge h_q,
\operatorname{harmfulRate}(S) \le r_0.
\]

实现顺序：

1. 先写 compatibility selector，在相同候选、相同排序和相同阈值下逐行复现 V3 输出。
2. 把 matching state 与 pose information state 放入一个增量状态机；每次添加只记录触发的约束和边际收益。
3. 统一信息增益的分母、query weight 和停止规则，避免不同阶段对“相对增益”使用不同基准。
4. compatibility 等价后，才移除 `pose_minimum_additions`；先验证 Heads 的 511 个 Pose 行中多少真正满足信息约束，不能把改 64 为 0 当成主要实验。
5. 最终导出每行唯一 primary selection reason，并允许附带多个 satisfied constraints。

Go 条件：在相同 mapping 门槛下 Anchor 更少，或相同预算下室内 mapping self-localization 更好；否则保留 V3 greedy 实现，只统一解释层。

#### P5.0 compatibility 已完成

`topology/sufficiency_selector.py` 已把 Precision、Matching Completion、Observability Completion 接入同一个 selected state，并为每个入选候选输出唯一 `primary_selection_reason` 和统一 trace。compatibility policy 仍调用原 matching-rank 与 dynamic pose objective，不改变 eligibility、排序、阈值或停止规则。

Heads 使用 frozen pipeline manifest 指向的精确 V3 输入完成真实重放：

| 项目 | Frozen V3 | Unified compatibility |
|---|---:|---:|
| Precision | 6870 | 6870 |
| Matching completion | 738 | 738 |
| Observability completion | 511 | 511 |
| Final Anchor | 8119 | 8119 |

`anchor_ids / xyz / features / type / source primitive / track cluster / fine identity / dependency / coarse dependency / source dependency` 十组定位张量全部 bitwise equal。审计同时发现并修复了一个容易被普通 count parity 漏掉的问题：旧实现的两次 `torch.unique(sorted=False)` 是非结合的，统一 trace 不能直接用一次 unique 物化，否则实体集合相同但行顺序会漂移。当前实现明确分离 semantic trace order 与 V3 compatibility materialization order。

旧 V3 Map 没有逐行 selection provenance，因此对旧产物只能验证三类 count 和最终定位张量；从本版开始新 artifact 会保存精确统一 trace。P5.0 不申报精度收益。

#### P5.1 all-candidate alias risk 首轮 Go

Heads 已物化 audit-only 的完整 eligible universe：10173 个 broad observation-grounded Track 与 20817 个 mapping-legal surface candidate，共 30990 个候选。风险图使用 1000 个 mapping queries、8 个 trajectory groups、在线一致的 global Top-1 和单次 PoseLib；因旧 raster visibility 是 canonical-row 而不是 primitive-index registry，本轮统一候选合法性明确使用 rendered depth/alpha + GT reprojection，不做错误 source 索引。

风险拆成两个量：

- `alias_risk`：false-winner rate 与 harmful-inlier rate 的 Wilson 95% **下界**取最大，只惩罚有重复证据支持的风险；
- `risk_uncertainty`：Wilson upper-lower width，独立报告，避免把 1/1 偶发错误直接当成确定风险。

无 evidence 保持 NaN，不伪装成低风险；在至少两个 trajectory groups 重现才标为 recurrent alias。

| Evidence | 候选 | False wins | Harmful events | Recurrent alias | Risk p50 |
|---|---:|---:|---:|---:|---:|
| Track | 10173 | 577321 | 142523 | 9920 | 0.851 |
| Surface | 20817 | 193149 | 26402 | 11756 | 0.321 |

Leave-one-trajectory-group-out 的 `false vs clean AUC=0.829`、`harmful vs clean AUC=0.909`，8/8 held-out groups 均高于随机。这确认 Track 不能继续被默认视为 identity-safe，也证明 all-candidate risk 具有跨 trajectory 可分性。

首个行为变量仅在 matching coverage gain 完全相同时，把 lower alias risk 放在旧 utility 之前作为字典序 tie-break；unknown 排在有支持风险之后。默认不提供 risk artifact 时，P5.0 bitwise compatibility 路径不变。尚未授权按 risk 删除 candidate，也尚未改变 Precision/Core 排序。

Heads 首轮真实 selector 重放的机制 gate 已通过：

| 项目 | P5.0 | P5.1 alias tie-break |
|---|---:|---:|
| Final Anchor | 8119 | 8117 |
| Precision / Matching / Observability | 6870 / 738 / 511 | 6870 / 737 / 510 |
| Achieved / feasible matching rank | 58627 / 58627 | 58627 / 58627 |
| Unmet query / rank | 0 / 0 | 0 / 0 |
| Matching-selected mean risk | 0.473 | 0.438 |
| Final information logdet p10 | 41.494 | 41.510 |
| Translation worst-std p90 | 0.1425 | 0.1406 |

新旧最终集合共享 8012 个实体；换出的 107 个候选 risk 均值 0.394、累计 harmful 311，换入的 105 个 risk 均值 0.175、累计 harmful 63。计入后续 observability selection 的小幅反向变化后，最终集合相对 P5.0 仍净减少约 248 次 mapping harmful events 和 1330 次 false wins。

因此 P5.1 通过 **机制/充分性 gate**，但尚未通过 pose-accuracy gate。因为最终 Anchor 集合发生变化，V3 的 8119-row metric state 不允许复用；必须先为 8117-row Map 重建 compact teacher/function graph 并执行相同 bounded metric refresh，之后才能比较 mapping pose。当前不申报精度提升。

#### P5.1 Heads compact refresh 与 mapping pose：Stop

Heads 的 8117-row P5.1 Map 已完成独立的 compact Top-64 function graph、Gaussian
raster provenance、provenance-aligned function graph 和 complete-positive teacher 重建。
teacher 覆盖全部 1000 个 mapping queries，包含 172526 个 positive rows、189351 个
strong pairs 与 414104 个 ambiguous pairs。随后从 identity metric 初始化执行与 V3 相同的
760-step bounded refresh（rank 16、residual 0.05、LR 2e-4、temperature 0.04、harmful
weight 0.1、trust 1.0、Group-DRO eta 0.03/max ratio 3、seed 2026）。严格 lineage audit
确认 Map/graph/teacher/metric 均为 8117 行，metric IDs 与 Map IDs bitwise equal、teacher
rows 与 function graph rows bitwise equal，且 `initial_metric_state=null`。

固定 96 个均匀 mapping queries、seeds 2026/2027/2028 的 P5.1−V3 结果为：

| 指标（三种子均值） | Delta |
|---|---:|
| Raw precision | **-0.00610 pp** |
| Inlier precision | **-0.04939 pp** |
| Median TE | -0.00940 cm |
| Mean TE | -0.00675 cm |
| P90 TE | -0.04140 cm |
| P95 TE | -0.08223 cm |
| CVaR95 TE | **+0.01433 cm** |
| Catastrophic >100cm | 0 |

mean、P90 与 P95 TE 均 3/3 seed 改善，但 raw precision 是确定性下降，inlier precision
也 3/3 下降；CVaR95 仅 1/3 改善、2/3 恶化。结合 ShopFacade 全量 precision sentinel
失败，Heads 对当前 equal-gain alias-risk tie-break 同样判为 **Stop**：不运行 test，不把
当前 tie-break 合入默认 selector。这个结果不否定 all-candidate alias risk 的可分性，而是
说明风险不能只在 matching/pose gain 相等时作为独立替换准则；下一版本必须联合估计
clean utility、observation-descriptor representability 与 tail pose risk。

Heads 完整 artifacts 位于
`/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/7Scenes/heads`，其中
`map_learning/lineage_audit.json` 保存所有内容指纹，mapping pose 位于
`evaluation/pose_gate_q96/{baseline,alias}_seed{2026,2027,2028}`。

#### P5.1 ShopFacade precision sentinel：Stop

ShopFacade 使用冻结 V3 的精确 canonical、Track payload、231 个 mapping query 和 `ref2p067_stop1e3` calibration 重放。完整合法候选池包含 7300 个 Track 与 7837 个 surface candidate。global Top-1 alias graph 仅使用 mapping split，并以 rendered depth/alpha 与 GT reprojection 判定合法性：

| Evidence | 候选 | False wins | Harmful events | Recurrent alias | Risk p50 |
|---|---:|---:|---:|---:|---:|
| Track | 7300 | 279018 | 24345 | 7275 | 0.620 |
| Surface | 7837 | 104718 | 6996 | 6699 | 0.552 |

跨 trajectory group 的 `false vs clean AUC=0.710`、`harmful vs clean AUC=0.741`；7 个可评估 held-out group 均高于随机，第 3 组因标签单类而 AUC 不可定义。alias risk 因而具有真实可分性，但可分性本身不授权部署。

equal-gain alias tie-break 的 selector 机制 gate 通过：Anchor 6357→6361，matching rank 保持 26904/26904，unmet query/rank 保持 0/0；最终 logdet 中位数 42.883→42.919，logdet p10 与 translation worst-std p90 不退。换出 35、换入 39 个候选，平均 risk 0.719→0.424，并净减少 392 次 false wins、55 次 harmful events，同时减少 8 次 clean events。

由于 Anchor registry 已变化，本轮没有复用旧 6357-row teacher 或 metric。为 6361-row Map 重新构建了 compact Top-64 function graph、Gaussian raster provenance、complete-positive teacher，并从 identity 初始化执行与冻结 V3 相同的 176-step bounded metric refresh。Map/graph/teacher/metric 均为 6361 行，metric IDs 与 Map IDs bitwise equal，teacher rows 与 function graph rows bitwise equal，`initial_metric_state=null`。

随后在全部 231 个 mapping queries 上以 seeds 2026/2027/2028 对比冻结 V3：

| 指标（alias - V3，三种子均值） | Delta |
|---|---:|
| Raw precision | **-0.00792 pp** |
| Inlier precision | +0.00740 pp |
| Median TE | **+0.03936 cm** |
| Mean TE | **+0.00463 cm** |
| P90 TE | -0.01231 cm |
| P95 TE | +0.02916 cm |
| CVaR95 TE | -0.01737 cm |
| Catastrophic >100cm | 0 |

Raw precision 是确定性量，三种子均从 14.19271% 降至 14.18478%；median 与 mean TE 均 3/3 种子恶化。P90 与 CVaR95 的三种子均值分别小幅改善 0.01231 cm 和 0.01737 cm，但不足以覆盖 precision sentinel 与中心误差回退。因此 **ShopFacade 对当前 equal-gain alias tie-break 判定为 Stop**：不运行 test split、不将其纳入默认 selector，也不通过场景阈值调参寻找例外。alias graph 继续保留为诊断证据；下一版本必须改变风险与 clean utility 的联合决策形式，而不是把当前 tie-break 扩大成删除或硬惩罚。

#### P5.1 Stairs compact refresh 与 mapping pose：Stop

Stairs 的完整候选池包含 5079 个 Track 与 22184 个 surface candidate。跨 trajectory
group 的 false/harmful AUC 分别为 0.853/0.904；equal-gain tie-break 在保持 7275 个
Anchor、118375/118375 matching rank 和零 unmet query/rank 的同时交换 361 对候选，净减少
14794 次 false wins 与 2427 次 harmful events。该机制信号同样真实，但 translation
worst-std p90 已轻微恶化 0.20%。

随后为新 Map 重建 2000-query compact Top-64 function graph、raster provenance、
complete-positive teacher，并从 identity 初始化执行 1520-step bounded metric refresh。
Map/graph/teacher/metric 的 7275-row anchor-ID 契约全部通过。固定 q96、seeds
2026/2027/2028 的 alias−V3 结果为：raw precision 3/3 均 -0.02136 pp，inlier precision
3/3 均下降，mean、median、p90、p95 与 CVaR95 translation error 也全部 3/3 恶化；seed
2026 的 CVaR95 进一步增加 5.06573 cm。因此 Stairs 判为明确 **Stop**，不运行 test。

完整报告见 `docs/v4_stairs_sentinel.md`，持久化 artifacts 位于
`/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/7Scenes/stairs`。

### P6：完全观测驱动的单 Anchor 描述子

描述子只从 `O_i` 的真实 SuperPoint 观测产生：

- 以鲁棒 medoid/trimmed fusion 聚合观测，不从 Gaussian feature 直接生成第二套身份描述子；
- 输出观测内 dispersion、有效视角数和 representability；
- 零观测或单观测 weak fallback 不允许伪装成高置信实体；
- all-candidate alias risk 在完整候选集合上计算，不能只对 Track 子集统计；
- alias risk 先作为 selector/matcher 的保守权重，不立刻改变描述子空间。

现有单图上下文结果仍可作为 query-conditioned matchability/alias 校准证据，但不再承担“生成一个新身份表示”的任务。若 dispersion/alias risk 不能在 mapping split 上区分 useful 与 harmful match，则停止学习复杂上下文头，保留鲁棒观测聚合。

#### P6.0 observation descriptor compatibility/audit 已落实

`topology/observation_descriptor.py` 和
`scripts/audit_observation_descriptors.py` 已把统一 Anchor Registry 的真实
observation CSR 与 frozen mapping query cache 接通。每个 Anchor 先在同一图像内按
SuperPoint confidence 聚合，再对 `(trajectory, view-bin)` joint strata 等权，最后做
cosine medoid + 20% trimming。产物并列保存 observation descriptor、有效 observation /
query / view group / trajectory 支撑、cosine dispersion、单描述子 representability 和
与现部署 descriptor 的 cosine。契约固定为 `uses_test_queries=false`、`audit_only=true`、
`deployment_descriptor_mutated=false`；零观测行输出无效 mask，不用旧 descriptor 冒充
真实 observation identity。

Heads 的 8119 个部署 Anchor 在当前完整 registry 中均有有效真实 observation，零观测和
单观测均为 0。Track 与 surface 的分布差异明显：

| Heads mapping-only 指标 | Track（7126） | Surface（993） |
|---|---:|---:|
| observation 数 median / p90 | 6 / 55 | 14 / 34 |
| distinct view-bin median / p90 | 2 / 2 | 3 / 6 |
| representability median / p90 | 0.0425 / 0.1052 | 0.1422 / 0.2333 |
| balanced cosine dispersion median / p90 | 0.0218 / 0.0722 | 0.1484 / 0.2744 |
| fused-vs-deployment cosine median / p10 | 0.9998 / 0.9964 | 0.9313 / 0.7960 |

Stairs 独立实跑复现同一趋势，而且暴露出 weak fallback 的实际边界：7275 个 Anchor
均至少有一个有效 observation，但 70 个单观测行全部来自 surface；它们必须显式保持低
支撑标记，不能当作 multi-view identity。其余关键分布如下：

| Stairs mapping-only 指标 | Track（2480） | Surface（4795） |
|---|---:|---:|
| observation 数 median / p90 | 45 / 125 | 12 / 32 |
| trajectory 数 median / p90 | 1 / 1 | 2 / 3 |
| representability median / p90 | 0.0683 / 0.1592 | 0.1546 / 0.2884 |
| balanced cosine dispersion median / p90 | 0.0265 / 0.0808 | 0.1816 / 0.3238 |
| fused-vs-deployment cosine median / p10 | 0.9998 / 0.9964 | 0.9280 / 0.7617 |

这说明 surface observation 多并不等于可由单 descriptor 良好表示；其多视角不一致性约为
Track 的数倍，而且 observation-fused identity 与现部署 descriptor 存在实质差异。因此
Stage 5 不是纯粹改名，但 P6.0 仍只建立机制审计：在相同 compact teacher/function graph、
bounded metric refresh 和 mapping-pose gate 完成前，不替换 `anchor_features`，也不申报精度
收益。

可复现实跑命令和当前 artifact（均为 mapping-only audit，不进入部署）如下：

```bash
python -m scripts.audit_observation_descriptors \
  --registry /tmp/lafgs_heads_anchor_registry_cov_v1.pt \
  --query-cache /mnt/pool/sqy/lafgs_adaptive_v3_validation_20260806/7Scenes/heads_clean_ref2p067/bootstrap/query_cache.pt \
  --output /tmp/lafgs_heads_observation_descriptor_audit_v1.pt

python -m scripts.audit_observation_descriptors \
  --registry /tmp/lafgs_stairs_anchor_registry_cov_v4_compat.pt \
  --query-cache /mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/7Scenes/stairs/bootstrap/query_cache.pt \
  --output /tmp/lafgs_stairs_observation_descriptor_audit_v1.pt
```

二进制结果分别位于 `/tmp/lafgs_heads_observation_descriptor_audit_v1.pt` 和
`/tmp/lafgs_stairs_observation_descriptor_audit_v1.pt`，同名 `.json` 是无需加载 Tensor 的
完整汇总。P6.0 的决策为 **Stage 5 mechanism GO**：进入独立 descriptor
materialization + bounded refresh + mapping-pose 对照；不是直接替换当前部署 descriptor。

#### P6.1 trajectory-block cross-fit：单 observation descriptor Stop

P6.0 的同观测 representability 不能授权写回 descriptor，因此新增
`topology/observation_descriptor_crossfit.py`，在同一冻结 V3 metric 空间执行严格的
support-trajectory→held-out-trajectory exact global Top-1。Track bank 全程不变；variant
只并行替换满足 support 门槛的 surface descriptor；不读取 test、不改 Map。

Heads 的 1000 张 mapping 图像全部来自 `seq-02`，没有第二条独立 mapping trajectory，
因此协议正确返回 `requires_at_least_two_mapping_trajectories`，不把相邻帧切片伪装成
cross-fit。Stairs 的四条 mapping trajectories 则穷举了全部三种平衡划分：

| Stairs fold A / B | 双向 eligible surface | A→B R@1 delta | B→A R@1 delta | 双向不退化 |
|---|---:|---:|---:|---:|
| 02+05 / 03+06 | 1826 | -12.94 pp | -14.24 pp | 9 |
| 02+03 / 05+06 | 1927 | -13.33 pp | -13.08 pp | 12 |
| 02+06 / 03+05 | 1556 | -13.60 pp | -12.61 pp | 8 |

六个方向均新增 2017–3637 个 false winners。虽然每种划分有 1045–1338 个 Anchor 的
双 fold descriptor cosine ≥0.65，但只有 8–12 个同时保持 held-out R@1 与 margin，三种
划分的稳定交集仅 2 个。由此 **Obs-all 与当前 Obs-stable materialization 均 Stop**：
不执行 compact refresh 或 pose，也不通过训练新 metric 掩盖源 descriptor 已经失去全局
身份判别性的事实。高 dispersion 更可能反映 surface identity 多模态/污染，或 A1 为抑制
全局竞争而有意偏离 positive medoid，而不是一个可由简单 medoid 修正的初始化误差。

详细协议与全部结果见 `docs/v4_p61_crossfit_descriptor_gate.md`；真实 artifacts 位于
`/mnt/pool/sqy/lafgs_anchor_identity_p51_validation_20260812/audits/observation_descriptor_crossfit`。

### P7：7Scenes/12Scenes 室内专项

室内问题优先看身份混淆和短基线，而不是继续堆地图元素：

1. Track pair 建图加入 baseline/parallax awareness，分别报告短基线重复纹理边与有效几何边。
2. alias risk 覆盖全部候选和全部 selection reason，检查楼梯、走廊、重复门框等混淆区域。
3. mapping evidence 与 deployment K 是两个独立因子；室内默认保持 V3 等效 `K_map=1024`，`K_map=2048` 只有通过完整单因素 gate 后才可升为默认，禁止两个密度一起变化。
4. 误差分桶到：无匹配、重复身份、几何退化、PnP outlier、召回足够但排序错误。只有最大的桶才进入下一轮方法修改。

核心哨兵：Heads、Fire、Stairs、Office2/Office5b；外部 precision 哨兵：ShopFacade、OldHospital。每次只使用 mapping split 做选择，冻结后再执行 test。

#### P7.0 baseline-aware Track pair graph 审计已落实

`evidence/track_pair_audit.py` 与 `scripts/audit_track_pair_graph.py` 已把冻结
Track pair policy、mapping camera pose/intrinsics 和最终 Track observation graph 接通。
该阶段严格是 `uses_test_queries=false / audit_only=true`：重建现有 candidate camera
pairs，只报告 baseline、视轴变化、时间间隔、mapping 三角化点的共同视野与视差，不改变
pair selection、Track、Map 或部署。

“短基线”采用场景内相对标签：连续 mapping 帧 baseline 的 p75 与当前 candidate
baseline 的 p25 取较大者；因此 `short-baseline=25%` 本身不是方法信号。真正的诊断量是
其中同时具有小视轴变化、高共同视野的 near-repeat proxy，以及不依赖该相对标签的
`parallax < 1 degree` 比例。Heads/Stairs 的冻结候选数均与旧 payload 逐个计数一致：

| mapping-only pair 指标 | Heads | Stairs |
|---|---:|---:|
| Mapping cameras | 1000 | 2000 |
| Candidate / 至少一条 match 的 pair | 3423 / 3423 | 7450 / 7450 |
| Baseline median | 0.0422 m | 0.0409 m |
| Optical-axis change median | 2.465° | 1.739° |
| 同序时间间隔 median / 直接相邻占比 | 6 / 0.000% | 7 / 0.134% |
| Mapping-FoV overlap Jaccard median | 0.928 | 0.932 |
| Mapping-point parallax median | 2.260° | 1.060° |
| Parallax < 1° | 3.30% | **42.71%** |
| 短基线近重复 proxy | 6.51% | 3.06% |
| 高 overlap 且 parallax < 1° | 0.18% | **11.50%** |
| 非短基线、parallax ≥ 1°、正 overlap | 73.71% | **49.54%** |

该结果把 Stairs 的问题定位得比“连续帧太多”更准确：现有 3 cm 最小 baseline 已基本
排除直接相邻帧，但固定 `nearest-6` 仍给 Stairs 分配了大量**高重叠、低视差** pair；其
绝对 baseline 与 Heads 接近，却因为视轴变化和场景深度结构只产生约一半的有效几何
pair。P7 的 pair-graph 机制因此通过第一步 **diagnostic GO**：下一因子应在相同 pair
budget 下，以 overlap 为可行性约束，提高 medium/large-baseline 与实际 mapping-point
parallax 的覆盖，首要哨兵为 Stairs，同时用 Heads 防回退。此结论不代表 pose gain，尚未
授权改变正式构图。

冻结 payload 仍存在一个不可伪造的结构 blocker：它只保存候选/匹配 pair 与 edge 的总数
及最终 Track 连通分量，没有保存每个 pair 的 raw、accepted、cycle-supported、ambiguity
rejected 和 epipolar-rejected edge 计数。最终 Track 的共同观测不能唯一恢复原始 pair
edge，因此“短基线重复纹理 matched-edge 的精确占比”当前必须留空。最小后续 sidecar
接口是每 pair 的 `left/right query id` 加上述六类计数；无需保存描述子或改变部署 payload。

可复现实跑产物：

```bash
python -m scripts.audit_track_pair_graph \
  --track-payload <scene>/bootstrap/tracks_refined/track_micro_anchor_payload.pt \
  --query-cache <scene>/bootstrap/query_cache.pt \
  --reproducibility-manifest <scene>/bootstrap/tracks_refined/reproducibility_manifest.json \
  --output /tmp/lafgs_<scene>_track_pair_graph_audit_v1.pt
```

Heads 与 Stairs 的结果分别位于
`/tmp/lafgs_heads_track_pair_graph_audit_v1.{pt,json}` 和
`/tmp/lafgs_stairs_track_pair_graph_audit_v1.{pt,json}`。

#### P7.1 mapping evidence 高密度协议审计已落实

`evidence/mapping_density_audit.py` 与 `scripts/audit_mapping_density.py` 已把
`K_mapping` 和 `K_deployment` 拆成两个不可混变的协议轴。该阶段仍严格是
`uses_test_queries=false / audit_only=true / pose_gain_claimed=false`：只读取已经物化的
mapping query cache，逐帧核对 detector 请求数、mask 前后 keypoint 数、原生 Tensor 行数、
valid-mask 像素比例和 NMS metadata，并对冻结 function graph 计算内容指纹；没有重跑
detector、GPU cache、Track、Map、metric 或 pose。

真实 artifacts 揭示了一个此前被配置表述掩盖的机制缺口。Adaptive V3 bootstrap 用
deployment 的面积自适应解析结果作为 mapping `native_keypoint_count`。Heads/Stairs 的
processed image 均为 640x480；以 1920x1080 为参考面积时，`2048 * area/reference` 四舍
五入仅为 303，随后被 `keypoint_minimum=1024` 截到 1024。因此当前两个室内 graph 并非
建议协议中的 `K_mapping=2048`，而是从源头只请求了 1024：

| mapping-only density 指标 | Heads | Stairs | ShopFacade |
|---|---:|---:|---:|
| Mapping queries | 1,000 | 2,000 | 231 |
| Native input | 640x480 | 640x480 | 1920x1080 |
| 面积缩放未截断 K | 303 | 303 | 2,048 |
| 解析原因 / 请求 K | minimum clamp / **1,024** | minimum clamp / **1,024** | unclamped / **2,048** |
| Mask 前 keypoints median | 1,024 | 1,024 | 2,048 |
| Mask 后 keypoints p10 / median | 1,024 / 1,024 | 1,024 / 1,024 | 1,990 / 2,028 |
| Valid-mask fraction median | 100.00% | 100.00% | 93.28% |
| 有 keypoint mask-drop 的 query | 0.00% | 0.00% | 96.54% |
| NMS=4 artifact attestation | 缺失 | 缺失 | 缺失 |

最后一行必须按证据边界解释：当前源码和正式 config 都指定 NMS=4，但这三份旧 cache
分别使用 signature v10/v10/v9，signature payload 与逐 query
`native_sparse_metadata` 均未记录 NMS radius。因此审计结论是 **unattested**，不是已证明
使用了错误 NMS。新 cache 必须同时在 cache signature 和逐 query metadata 写入 4，才可
通过这一协议门。

每个审计同时输出独立的 `lafgs_mapping_density_paired_factor_manifest`。manifest 将同一
query-cache signature、同一 function-graph SHA256、同一 Map/metric 固定为
`immutable_mapping_graph`，只允许 `K_deployment` 在 1024/2048 间变化，并显式禁止两个
variant 之间重建 mapping cache 或同时改变 mapping/deployment density。当前 Heads 和
Stairs manifest 因 `K_mapping=1024` 与 NMS 未证明而正确阻断；ShopFacade 已满足
`K_mapping=2048`，但仍被 NMS attestation 阻断。也就是说，P7 的 2048 mapping evidence
建议此前**尚未在室内哨兵落实**，不能把现有 graph 当作已经完成的高密度基线，更不能
从本审计宣称 pose gain。

可复现实跑命令为：

```bash
python -m scripts.audit_mapping_density \
  --scene <scene> \
  --query-cache <mapping-query-cache.pt> \
  --mapping-graph <function-graph.pt> \
  --mapping-keypoints 2048 \
  --deployment-keypoints 1024,2048 \
  --expected-nms-radius 4 \
  --output /tmp/lafgs_mapping_density_audit/<scene>.json
```

真实报告和成对因子输入 manifest 位于
`/tmp/lafgs_mapping_density_audit/{heads,stairs,shopfacade}.json` 与同目录的
`*.paired_factor_manifest.json`。下一步解锁条件不是立即混合改动 deployment，而是先为
Heads/Stairs 各物化一次固定 `K_mapping=2048 / NMS=4` 且 metadata 可证明的新 mapping
cache/graph；随后两个 deployment K variant 必须复用该唯一 graph。该段记录的是审计阶段
当时的解锁条件；Stairs 的后续 density-only Track 因子已在 P7.2 执行并触发 Stop。

#### P7.2 Stairs mapping-density Track 因子：Stop

Stairs 已在相同源码、nearest-6 pair、pair budget、阈值和 seed 2026 下重跑严格的
`K_mapping=1024/2048, NMS=4` 两臂。cache signature 除 K 完全一致；2,000 个 query 的
逐帧 K/NMS metadata、pose/intrinsics/depth/alpha 和 K=2048 的前 1,024 个 sparse rows
全部通过严格配对审计。新提取的 K=1024 control 与冻结 V3 的完整 Track 漏斗逐项完全
一致。

K=2048 将 raw edge、Track、triangulated、broad 和 strict Track 分别提高到 1.830x、
2.151x、2.064x、1.542x 和 1.273x；但 broad covariance median 恶化到 1.229x，超过
预注册上限 1.10x，同时 frozen high-confidence Track 保持 **70 -> 70**。机制门七项中
六项通过、一项失败，因此按协议停止，不重建 function graph/Map，不执行 metric 或 pose。

第一性原理结论是：更多检测点确实扩大了证据覆盖，但新增量主要进入条件较差的组件，
没有扩大最可靠身份集合；室内瓶颈不是单纯的点数不足。保留通用 K_mapping/K_deployment
解耦与审计契约，K=2048 仅作为显式因子配置。完整命令、哈希、环境失败记录和 JSON 见
`docs/v4_p7_mapping_density_factor.md`。

#### P7.3 fixed-budget pair policy：Stairs scene-specific Go，GreatCourt cross-domain Stop

Stairs 的 `parallax_diverse` factor 已通过 mechanism、exact splat-provenance lineage、完整
compact/fullchain rebuild 和 q256x3 mapping-pose gate。mapping pose 三种 seed 均无回退，
三 seed 平均 raw precision +0.5470 pp、median/mean/p90/CVaR95 translation error 分别
-0.0582/-0.1712/-0.2382/-1.6858 cm，5cm/5deg recall 从 98.4375% 增至 100%，
catastrophe 保持 0。该 Go 的代价是 Anchor 从 7,275 增至 9,559；gate SHA-256 为
`e1366dd367b1be8b2ec9797c64da6f9bfde4b370aae7a1acfbf02133fd921a73`，且仍只是
mapping-only 结论。

为检验它能否成为共享方法默认，GreatCourt 在 K=2048/NMS=4、1,531 个 mapping query、
相同精确 5,254 pair budget 下运行独立单因素机制 gate。fresh manifest/cache/frozen Track
SHA-256 分别为 `078d6e8b...`、`0550b59e...`、`f0fec708...`；control report/factor 为
`c46b360b...` / `78cd1d04...`，variant report/factor 为 `48e5ef86...` /
`a6dbcf51...`。最终 gate SHA-256 为
`201362cde50296eac4abc0e7813ee618f970fdae4c248c854aaa81eb14b72f35`，并记录
`uses_test_queries=false`。

六项预注册检查为：精确 pair budget Pass；低视差比例 64.1745% -> 2.3220%（下降
61.8524 pp）Pass；triangulated Track 34,150 -> 32,293、保留率 94.5622% **Fail**；
broad Track 16,985 -> 19,473、保留率 114.6482% Pass；triangulated covariance p90
28.1497 -> 47.2850 m2、比例 1.6798x **Fail**；broad support/query p10 0 -> 3 Pass，
但 control=0 使最后一项只有弱证据力。故 `mechanism_gate_passed=false`，在 variant
provenance replay 前停止；没有运行 variant lineage audit、fullchain、mapping pose 或
formal test。nearest control provenance replay 只用于证明 fresh-cache exact parity。

第一性原理结论是：camera-pair policy 应最大化固定成本下条件良好、可三角化、可定位的
独立身份，而不是最大化 parallax 代理本身。GreatCourt 同时出现更高 parallax、较少
triangulated Tracks 和显著更差 covariance tail，证明当前 objective 不具备跨场景充分性。
因此 Stairs 保留为 scene-specific Go，跨域默认 **No-Go**，共享方法继续使用 `nearest`。
完整结论见 `docs/p7_greatcourt_pair_policy_stop.md`，机器可读摘要见
`docs/evidence/p7_greatcourt_pair_policy_mechanism_stop.json`。

#### P7.4 Stairs XFeat Arm B：存在 descriptor headroom，但严格 Stop

锁定的 XFeat 64D descriptor 已在完全相同的 2,000 张 mapping 图、每图 1,024 个
SuperPoint 行上完成 Arm B，共验证 2,048,000 行；不启用 XFeat detector，不读取 test。
fresh cache equivalence V2 对 query 顺序、Track 输入、effective sparse depth、native alpha
均为 2,000/2,000 exact，并授权复用冻结 Track payload。

XFeat 在两个 temporal-block 方向的 R@1 分别提升 +4.0683 / +4.6996 pp，pooled R@8
提升 +7.9290 pp，Track Core pooled R@1 提升 +6.2552 pp。这证明前端 descriptor quality
确实存在真实 headroom，不能再把 frozen SuperPoint 当作已达 representation ceiling。
但 Gaussian Reserve pooled R@1 从 22.0532% 降至 22.0280%（-0.02518 pp），违反预先
固定的 exact non-regression gate，因此 `mechanism_gate_passed=false`。按协议没有执行
descriptor map/function graph、metric refresh、mapping pose 或 formal test。

这不是“XFeat 无效”，而是“统一替换无法同时保护两类 evidence identity”。共享方法的下一步
应显式研究 evidence-aware descriptor representation/utility，而不是绕过 gate 混入 detector、
pair policy、更多训练步数或 Pose Reserve。完整报告见 `docs/xfeat_arm_b_stairs_result.md`，
机器摘要见 `docs/evidence/xfeat_arm_b_stairs_gate.json`。

## 最小实验矩阵

| 阶段 | 对照 | 变量 | 首要判据 | 决策 |
|---|---|---|---|---|
| P1 | V3 map | Registry view | 核心张量 bitwise equal | 不等即修复，禁止后推 |
| P2 | V3 candidates | 等价图阈值邻域 | 稳定分量、重复竞争贡献 | 无信号即停去重 |
| P3 | V3 | evidence-transfer dedup | coverage 不降、冲突下降 | 不满足则只标注不删除 |
| P4 | image-only | fixed/adaptive surface | mapping pose 与几何同时改善 | 仅代理改善则停 |
| P5 | V3 staged greedy | unified selector | 相同门槛更小/同预算更准 | 无优势保留兼容实现 |
| P6 | current fusion | robust observation fusion + risk | indoor harmful/alias 降低 | 无可分性则停复杂头 |
| P7 | V4 frozen | baseline-aware + K factors | 7/12Scenes 成对改善 | test 只做最终确认 |

所有实验报告必须同时给出：Anchor 数、真实观测覆盖、每 query 匹配行 p10、唯一 landmark match 数、inlier 数、harmful rate、平移/旋转中位数及标准成功率。仅报告最终 pose 中位数不足以判断机制是否成立。

## 当前决策

P2.5 已完成 Heads、ShopFacade、Stairs 的 mapping-only 多种子 gate。物理去重在两个室内场景稳定伤害 CVaR95，Heads 还出现 recall 回退，因此 **P3 物理 evidence-transfer/dedup 分支正式停止**；不再扩到 Office/OldHospital 来为一个已失败机制寻找例外。等价 component 作为 semantic risk feature 保留，但 matcher 和 Map 均不折叠。

P5.1 all-candidate alias risk 已完成完整候选审计、selector replay、compact evidence/teacher
重建、bounded metric refresh 与 mapping pose gate。Heads、ShopFacade、Stairs 虽都显著减少
历史 false/harmful evidence，却分别触发 precision、中心误差或 tail regression；因此
**当前 equal-gain alias-risk tie-break 正式停止，不进入默认方法，也不再围绕其做阈值、
Pose Reserve 或训练步数的局部修补**。alias risk 仅保留为 failure attribution 与后续联合
效用的输入。

P6.1 trajectory-block cross-fit 已证明 raw observation 的单 medoid materialization 会在
Stairs 所有独立 trajectory 划分上大幅损失 held-out 全局身份判别性，因此该分支正式停止，
不进入 compact refresh。P7 density-only 两臂随后证明 K=2048 虽扩大 Track 漏斗，却恶化
broad Track 的典型几何条件且不增加 high-confidence Track，因此同样停止，不进入完整
pose。

固定 K、固定 pair budget 的 `parallax_diverse` 单因素已完成：Stairs 通过 fullchain
mapping-pose gate，但 GreatCourt 在进入 pipeline 前因 triangulated Track retention 和
covariance p90 两项失败而 Stop。故 **不得把 Stairs Go 外推成跨域默认**；当前共享主线保留
`nearest`，也不再围绕当前 parallax objective 做场景后验阈值修补。descriptor、density 与
pair-policy 三个变量仍不得混入同一个因果实验。后续若继续提高室内外共同精度，应先回到
可定位证据效用/前端 correspondence failure 的统一测量，而不是继续堆 parallax、地图点、
Pose Reserve 或训练步数。

XFeat Arm B 进一步收紧了这个结论：identity descriptor 不是完全饱和变量，但增益高度集中于
Track Core，Reserve 的严格非退化未通过。因此当前默认 frontend 仍保留 SuperPoint，且不跑
XFeat map/pose/test；下一主线必须解释并保护 evidence-conditioned identity，而不是把一个
pooled retrieval 增益直接当作完整定位方法增益。

已在任何融合结果生成前预注册唯一的后续 descriptor 因子：将逐行归一化的 256D
SuperPoint 与 64D XFeat 等能量拼接为一个 320D 单描述子。它仍只有一个 bank、一次全局
Top-1 和一次 cosine，分数严格等于两分支 cosine 的 50/50 平均；禁止可调 alpha、类型路由、
双 bank、detector/topology/pose 联动。双向 R@1 必须严格提升，pooled R@8、Track R@1 与
Reserve R@1 必须全部非退化，否则 Stop。完整冻结合同见
`docs/xfeat_equal_energy_descriptor_preregistration.md`。
