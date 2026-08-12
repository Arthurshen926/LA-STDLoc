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
| Stage 4 统一 Sufficiency Selector | compatibility 已完成 | 单一 selected state、统一 primary reason/trace，Heads 真实重放定位张量 bitwise equal | alias-aware tie-break、最少 Pose=0、relative retention 尚未授权 |
| Stage 5 observation descriptor | 部分基础已具备 | 所有现有 V3 最终 Anchor 均可追溯真实 observation；Track 已做鲁棒融合 | Gaussian/base 的统一重物化、dispersion/representability 尚未实现 |
| Stage 6 室内身份自适应 | 部分落实 | NMS=4 契约修复；all-candidate equivalence/harmful 审计扩到 Stairs | baseline-aware pair graph、all-candidate alias cost、高密度因子尚未实施 |

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

### P6：完全观测驱动的单 Anchor 描述子

描述子只从 `O_i` 的真实 SuperPoint 观测产生：

- 以鲁棒 medoid/trimmed fusion 聚合观测，不从 Gaussian feature 直接生成第二套身份描述子；
- 输出观测内 dispersion、有效视角数和 representability；
- 零观测或单观测 weak fallback 不允许伪装成高置信实体；
- all-candidate alias risk 在完整候选集合上计算，不能只对 Track 子集统计；
- alias risk 先作为 selector/matcher 的保守权重，不立刻改变描述子空间。

现有单图上下文结果仍可作为 query-conditioned matchability/alias 校准证据，但不再承担“生成一个新身份表示”的任务。若 dispersion/alias risk 不能在 mapping split 上区分 useful 与 harmful match，则停止学习复杂上下文头，保留鲁棒观测聚合。

### P7：7Scenes/12Scenes 室内专项

室内问题优先看身份混淆和短基线，而不是继续堆地图元素：

1. Track pair 建图加入 baseline/parallax awareness，分别报告短基线重复纹理边与有效几何边。
2. alias risk 覆盖全部候选和全部 selection reason，检查楼梯、走廊、重复门框等混淆区域。
3. mapping evidence 默认固定 `K_map=2048`；deployment K 单独做 1024/2048 因子，禁止 mapping 与 deployment 密度一起变化。
4. 误差分桶到：无匹配、重复身份、几何退化、PnP outlier、召回足够但排序错误。只有最大的桶才进入下一轮方法修改。

核心哨兵：Heads、Fire、Stairs、Office2/Office5b；外部 precision 哨兵：ShopFacade、OldHospital。每次只使用 mapping split 做选择，冻结后再执行 test。

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

下一主线切换为 **P5.1 all-candidate alias risk**：P5.0 compatibility 已通过；随后只在 mapping split 依次验证 all-candidate alias cost、Coverage 下 Pose tie-break、`pose_minimum_additions=0` 和 relative information retention。Stage 3 geometry revision 与 Stage 5 descriptor 重物化暂不并行开启，避免把 selector 机制与几何/表示变化混淆。
