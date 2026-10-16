# 当前论文与复现入口（2026-10-16）

- 活跃编辑稿：[editorial r3](../../paper/AnyGSLoc_TRO_20261016_editorial_r3/README.md)，[正文源码](../../paper/AnyGSLoc_TRO_20261016_editorial_r3/main.tex)。仅正文及其实际图表依赖纳入本次版本管理；旧补充章节没有作为可构建附件发布。
- 复现候选：[release/AnyGSLoc](../../release/AnyGSLoc/README.zh-CN.md)，含参考记录、22 张表、配置及逐实验入口；实际核验范围见其 `docs/validation.md`。编辑稿与该实验记录的完整科学一致性仍需按稿件审查，不能仅凭版本日期认定。
- 下方全部内容保留为 2026-10-10 及更早的历史记录，其中“当前”“最新”只代表当时状态。旧稿链接可能仅在本机可用；历史稿及大产物保留本地并被 Git 忽略。

---

# AnyGSLoc 实验证据与论文定位（更新：2026-10-10）

本目录是当前论文实验的统一入口。数值来自实际结果文件，不把排队、进程退出或历史日志存在当成完整有效结果。

- [当前：compact-v2 写作框架 r4](../../paper/AnyGSLoc_writing_framework_20261010_r4/README.txt)：[八张主表与 S1](../../paper/AnyGSLoc_writing_framework_20261010_r4/tables/table_drafts.pdf)、[实验判断与同步说明](../../paper/AnyGSLoc_writing_framework_20261010_r4/revision_notes.pdf)、[新版九图](../../paper/figures_revision_20261010_v2/revised_nine_figures.pdf)。采用完整部署存储，分开构造/选择消融；Fig4、E1、E3、E4、E5按真实 v2 地图与位姿同步。以下 r3/r2 与旧稿保留。

- [当前写作框架：2026-10-09 r2](../../paper/AnyGSLoc_writing_framework_20261009_r2/README.txt)：[本轮修订说明](../../paper/AnyGSLoc_writing_framework_20261009_r2/revision_notes.pdf)、[七张表稿](../../paper/AnyGSLoc_writing_framework_20261009_r2/tables/table_drafts.pdf)、[逐段写作蓝图](../../paper/AnyGSLoc_writing_framework_20261009_r2/writing_blueprint.pdf)。采用三项方法贡献、分类精度横评、含地标数的本地系统表和八行联合消融；以冻结九图为锚点。当前写作方向以此框架为准，尚未替换下面的完整历史稿件。

- [当前期刊初稿：v7 叙事重构与审稿式修订](../../paper/AnyGSLoc_Transactions_v7/README.md)：[13 页手稿](../../paper/AnyGSLoc_Transactions_v7/main.pdf)、[事实核查与中文修改说明](../../paper/AnyGSLoc_Transactions_v7/REVISION_NOTES_ZH.md)。按 10–13 页 IEEE Transactions 初稿组织，强调无需原始建图 RGB 的场景转换、轨迹地标重建与集合约束选择。本次补强几何与选择的数学联系，重排 24 场景、基线、几何、逐场景压缩和完整组合表，补全查询配置公式。已复算 35 份既有逐查询结果；没有新增实验。以下 9 月版本的写作建议和“最新”标记保留为历史记录，当前稿件以 v7 为准。

- [最新图件：v6 全套 13 图重绘及整合稿（2026-09-26）](../../paper/AnyGSLoc_TRO_v6/README.md)：[图件审阅 PDF](../../paper/AnyGSLoc_TRO_v6/All_figures_review.pdf)、[逐图修改说明](../../paper/AnyGSLoc_TRO_v6/FIGURE_REVISION_ZH.md)。原始结果不变；统一空间示意、真实 mask 原因、离散压缩运行点和跨室内外 query 案例。v5 保留，v6 为当前视觉版本。

- [最新：v5 审稿视角修订的自包含期刊候选稿（2026-09-26）](../../paper/AnyGSLoc_TRO_v5/README.md)：15 页、6 图、10 表，方法/证据对齐、几何与部署图重画、等预算选择器控制复核；[中文修改与投稿判断](../../paper/AnyGSLoc_TRO_v5/REVIEWER_READINESS_ZH.md)。保留 v3/v4，未修改冻结算法。当前写作及投稿材料范围以 v5 为准；按现行 T-RO 规则，必要附录已纳入主 PDF，额外 archive 仅为内部审阅材料，不是拟提交的 supplementary paper。

- [历史：v4 完整英文初稿与可编译源码（2026-09-26）](../../paper/AnyGSLoc_TRO_v4/README.md)：全文重写、图件整合、自动实验表与理论/实现核对的 26 页内部版本；[修改记录与证据边界](../../paper/AnyGSLoc_TRO_v4/REVISION_NOTES_ZH.md)。

- [最新：空间机制版完整图件（2026-09-26）](../../paper/figures_spatial_final/README.md)：参考 DSAC / STDLoc / ULF-Loc 原图，重画总览并展开几何/描述子融合与紧凑选择，完整 13 图审阅 PDF，明确示意与实证的边界。

- [整套投稿图件候选定稿：8 张正文候选 + 3 张补充图](../../paper/figures_submission_candidate/README.md)：V2 改用经过五场景一致性核验的真实 mask；统一总览、机制和定量图，附英文图注和来源清单。

- [最新：多场景机制图与两种总览构图](../../paper/figures_v5/README.md)：不固定场景选样，真实 Gaussian / 缓冲区 / 筛选决策 / 同 Track 几何对照；保留选择反例与选样边界。
- [重制方法总览与投稿图注](../../paper/figures_publication/README.md)：真实高斯渲染、同 Track 局部、当前地图与匹配素材；180 mm 矢量 PDF/SVG，替代旧 F02 草稿。
- [八张图件首版与待补内容（2026-09-25）](figure_suite_progress_20260925.md)：图件位于 `paper/figures_v4/`，提供 PDF/SVG/PNG 和可追溯来源；目前是审阅草稿而非投稿成图。
- [v4 写作、图件设计与证据收尾计划（2026-09-25）](v4_revision_and_figure_plan_20260925.md)：对照 v3、专家建议和相关论文形成的下一阶段方案；不代表已完成新的实验或图件。
- [完整数值附录](evidence_20260917/TABLES.md)：24 场景、基线、先验、全部组合消融和计时。
- [可复核 CSV 与来源清单](evidence_20260917/)：逐场景细节、80 组组合、15 组几何、历史消融、成本与 SHA256。
- [协议说明](../release/semantic_mask_protocol_v2.md)、[组合消融边界](../release/factorial_ablation_protocol.md)、[基线修复说明](../release/baseline_recovery_v4.md)。
- 生成脚本：`tools/build_paper_evidence_bundle.py`。从仓库根目录用 g4splat Python 执行，可重新生成数值文件；原始大模型、图像、逐查询结果仍留在 `/mnt/pool/sqy`，不纳入 Git。

## 一、总体结论与论文定位

推荐定位：**把已有 Gaussian 先验转换为紧凑、可独立部署的稀疏定位地图，以单阶段定位获得接近渲染式两阶段方法的精度，显著降低在线计算和地图点规模。**

不是“所有指标最优”，不是“对任意先验质量不敏感”，也不是“全部模块都显著提高精度”。当前冻结主线没有离线自定位反馈或在线稀疏 refinement，不应沿用历史反馈版本的贡献叙事。

目前证据最强的三点：

1. 多视角三角化的定位坐标总体优于直接使用深度或投影高斯中心。
2. 观测处理整体改善精度与匹配行为，其中描述子融合证据较强，V2 较小，completion 不稳定。
3. 紧凑选择把五场景地图点数减少 67.1%，总体 median 变化较小，但不是无损压缩，也没有证明选择器本身提高 FPS。

## 二、有效证据清单

| 实验 | 已完成证据 | 使用建议 |
|---|---|---|
| 主线 24 场景 | 5 Cambridge、7 7Scenes、12 12Scenes，单 seed 2026 | 正文主表，完整逐场景表放附录 |
| STDLoc/ULF-Loc 复现 | 20 组构建、20 次含 coarse/fine 的测试、60 次计时 | 正文，必须标清 reproduced 和语义协议 |
| 不同先验 | Cambridge 6 种额外先验 × 5 场景，另有 MAtCHA 主线 | 正文紧凑表＋附录全表，包括失败质量的先验 |
| 观测与选择组合 | V2 × fusion × completion × full/compact，80 组精度＋80 组三轮计时 | 正文核心组合，附录公开全部 16 种配置 |
| 几何来源 | 5 场景 × ray/depth/center，固定共同点 | 正文几何证据，但不是全链路几何消融 |
| 历史内部消融 | 新协议下重评的 73 组 | 附录支持性证据，不替代最新重新选择的组合实验 |
| 构建成本 | MAtCHA 转换 5 组、基线构建 receipts、其他先验历史日志 | 当前仅分阶段实测可用，完整总成本表仍待审计 |

旧查询语义 mask 结果、多 seed 旧表、自定位/在线 refinement 历史版本，不与当前协议结果混合汇总。73 组历史消融的参考地图和删点顺序不完全相同；不要把它们与新 factorial 当成完全重复实验。

## 三、主表与公平比较

24 场景宏平均 TE/RE/R5：Cambridge 8.871 cm / 0.1294° / 42.08%；7Scenes 0.633 cm / 0.2083° / 98.58%；12Scenes 0.257 cm / 0.1144° / 98.65%。宏平均是场景 median 的平均，不是把全部查询混合后求 median。

Cambridge 使用 MAtCHA，室内 19 场景使用 Vanilla 2DGS，不能写成 24 场景统一 MAtCHA。保留数据集训练/测试划分及室内使用的位姿标注版本，在投稿方法节明确。

基线建图 mask 关闭时，STDLoc fine 为 8.799 cm / 0.1309° / 37.97%，ULF-Loc fine 为 8.475 cm / 0.1381° / 40.92%。AnyGSLoc R5 与 RE 较好，TE 不是最好。建图 mask 开启时，STDLoc fine TE 为 8.425 cm，ULF-Loc 为 8.985 cm。不能仅选有利协议，也不应将自行复现结果称为论文原报数值。

所有新查询测试禁用语义 mask，但 Cambridge 保留由相机去畸变映射计算的有效采样域；不以 RGB 非黑判断有效性。V2 是 mapping 渲染观测筛选，和 query 语义 mask 是不同机制。

MAtCHA 上游用了语义先验。AnyGSLoc 转换阶段不使用语义，并不代表整个输入生产链没有用语义。若与 baseline mapping-mask-off 比较，要同时给 baseline mask-on 行，并披露上游差异，不能宣称完全同输入控制实验。

目前直接复现比较只覆盖 Cambridge；室内主表不能单靠本项目自身结果就宣称全面 SOTA。2026-09-26 的 v4 已核对 ULF-Loc 汇编的 published 表格，保留来源精度与报告均值，并与本地 reproduced 分表。它们并非统一上游输入、语义或实现协议下的受控排名。

## 四、消融的科学解释

### 几何

固定共同有效 Anchor 和描述子的五场景结果：ray 8.765 cm / 0.133° / 41.33%；depth 9.911 cm / 0.160° / 29.13%；center 11.667 cm / 0.157° / 36.30%。支持多视角坐标估计总体更可靠。

中心对照采用像素 Z-buffer、2 px 最近投影关联、至少三个共同有效观测，并对坐标逐维取中位数；聚合后的点未必是某个实际高斯中心。该可见性规则不是 alpha 合成，结果只适用于这一定义。ray 保留原 Track 的三角化估计，depth/center 使用共同可用观测；不是三种方法完全同观测的三角化比较。共同点仅保留约 65%–85% 地图，正文必须报告覆盖损失。

KingsCollege 的深度 TE 与中心 R5 存在反例，不支持逐场景全面优胜。该实验保留匹配与 Track，仅隔离坐标来源；不要将其写成双向匹配、极线、Cycle、Track 全链路联合消融。

### 观测

紧凑地图全部观测机制关闭：9.471 cm / 0.1385° / 38.74%；全部开启：8.871 cm / 0.1294° / 42.08%。整体 TE 下降约 6.3%，R5 增加 3.34 个百分点。关闭融合后 R5 为 38.44%，是更明确的证据。

V2 单独关闭变化较小；completion 关闭后 TE 8.849 cm，R5 41.94%，说明其全局收益很弱。可以归入观测覆盖机制，不宜单列为核心贡献，但不能隐瞒实际使用。正文给联合观测模块效果，补充材料拆开各开关；联合效果不能反过来证明每个成员都有效。

fusion-off 用按 mapping 名称确定的单观测描述子，保留来源地图的质量评分。这是最终描述子融合消融，不是去掉地图构建过程中的所有多视角统计。completion-off 在选择之前删候选，与历史直接从紧凑成品删点的消融不同。

### 紧凑选择

总点数 535,840 → 176,329（−67.1%），TE 8.762 → 8.871 cm，R5 43.00% → 42.08%。KingsCollege R5 16.91% → 12.83% 的退化不能被宏平均掩盖；OldHospital 有正收益。

完整地图核心耗时 66.52 ms，紧凑地图 67.10 ms：**没有观察到平均在线加速**。正文应把选择器的贡献写成减少存储和检索规模、保持大致精度水平，不要把主线相对 baseline 的全部加速归因于选择器。

2026-09-26 补充核验：已有 `equal_size_quality` 对照保留相同 quality+matching 前缀，并按质量添加与 pose 阶段等量的点，不是从零开始的纯质量选择。20 个 map 的实际点数和源行集合核实后，两个最终分支逐场景预算相同。quality completion 为 8.812 cm / 0.127° / 41.47%，pose completion 为 8.871 cm / 0.129° / 42.08%；pose 提高约 0.61 pp R5，但平均 TE/RE 略差。v5 已纳入正文解释和内部附录表，不宣称信息补全唯一最优。完整核验见 v5 的 `checks/reviewer_experiments_v5.md`。

## 五、先验适配

Vanilla 2DGS、3DGS 的宏平均 TE 分别 9.131、8.809 cm，接近 MAtCHA 的 8.871 cm。这支持跨优化式先验适配，但各先验输入视角、上游训练条件并不完全一致，不能解释为只改变重建算法的严格控制实验。

i3DGS raw/+1k 为 23.653/11.640 cm；Long-LRM raw/+1k 为 717.621/98.798 cm。不能删除这些负结果后声称对任意 Gaussian 质量不敏感。它们说明接口可接入并不等于质量鲁棒；Long-LRM 的异常仍有上游质量/接入解释的不确定性。

## 六、在线效率、存储与离线成本

主线宏平均核心定位 67.1 ms。基线 mapping mask off：STDLoc coarse/fine-total 为 261.1/403.4 ms，ULF-Loc 为 570.8/1369.6 ms。是当前实现、输入和计时配置下的实测，不是论文原报速度，也不能直接写成严格等分辨率条件的算法加速倍数。

按后续确认的写作要求，正文以完整算法调用的 mean latency 为主，不将磁盘读图/GT 评测循环吞吐混作算法 FPS，也不增加尾延迟图。AnyGSLoc 67.10 ms 包含输入搬运与有效域应用，而两基线原生计时从这些步骤之后开始；原生求解器预算也不同。60 份基线计时回执均覆盖完整查询并实际 warmup 50 次，旧 `unwarmed` 标签已核实为陈旧描述。平均低于 100 ms 不等于逐查询截止保证。

地图存储用部署 map 文件实际字节数；共享网络权重单列。baseline 原生目录清点包含日志、冗余模型与其他产物，不是最小部署包；不能拿它与 AnyGSLoc 最小 map 直接计算公平压缩倍数。当前最可靠的是自身点数/地图字节对照；跨方法最小部署存储还要资产裁剪审计。

MAtCHA 已有 PLY 的转换耗时 9.90–47.66 分钟。STDLoc 多数完整构建约 115–132 分钟；ULF-Loc 多数约 18–23 分钟。三者不能直接拼成端到端优劣表：AnyGSLoc 前者不含 prior，baseline 包含联合重建。ULF-Loc ShopFacade off 新 receipt 仅测恢复后处理，旧构建阶段要单独追溯。GreatCourt 的 198 分钟 STDLoc 与 83 分钟 ULF-Loc 记录包含运行环境/加载异常，不能擅自删除，也不能在未拆解前全归因于算法。

35 单元 prior 成本表明确标为 PROVISIONAL；缺失值不是零。总成本应包含输入/语义准备、prior 重建、可选 +1k、导出与转换。基线联合 Gaussian/feature 训练不做虚假的 RGB/feature 时间拆分，阶段计时与总计时不能重复相加。当前没有完整可信的跨方法总构建成本结论。

## 七、论文落位建议

| 正文位置 | 建议内容 | 不应写的结论 |
|---|---|---|
| 主结果表 | 24 场景；Cambridge reproduced coarse/fine 分行 | 全指标最优、室内全面 SOTA |
| 核心消融表 | 几何来源、观测联合、full/compact；精度＋点数＋成本 | 每个子模块都显著有效 |
| 效率表 | 完整算法 mean latency、计时边界、地图点数/字节 | 稳定逐帧 10 Hz、选择器必然加速、未审计跨方法最小存储比 |
| 先验表 | 优化式先验正文摘要，全部先验放附录 | 任意质量先验不敏感 |
| 构建成本 | 暂给明确边界的阶段成本，完整表补齐后再定稿 | 将 supplied-prior 转换与全训练直接比总成本 |
| 补充材料 | 80 组 factorial、73 组历史子模块、15 组几何逐场景、mask 双协议、全部负结果 | 只公开挑选的正结果 |

单 seed 是固定协议，三轮只重复计时。长期根据 test 调整方法，属于开发过程中的测试集适配，不能将本轮称作新的独立 confirmation。若报告显著性，需要预先定义统计方法；现有宏平均差值不是统计显著性的证明。

## 八、剩余收尾事项（不新增方法）

1. 核对 35 单元 prior 计时与最终 PLY 的对应关系、缺失的 MAtCHA/语义准备时间；缺失无法恢复则标 NR，不编造总和。
2. 审计 baseline 最小部署资产与网络权重；已有目录总大小只能作为资产库存。
3. 把输入分辨率、预计算、读图、传输与迭代数写进效率表脚注；必要时补统一边界计时，但不改冻结算法。
4. 逐场景检查异常与失败尾部，保留 KingsCollege 退化、Long-LRM 负结果。
5. 已在 v4 核对引用与 ULF-Loc 汇编的 published 表；继续保持 published/reproduced 分离，不用当前复现值代替原论文数字。

主线方法可以冻结并进入写作，但“实验运行完”不等于“所有论文主张已被证实”。
