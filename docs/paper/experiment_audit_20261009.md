# 论文实验复核与最小补充方案（2026-10-09）

结论：已检查的数字、查询覆盖、语义开关代码和主要消融产物没有显示需要整批作废的证据。当前主要问题是实验能回答的问题，比某些预期主张更窄。不能因负结果而重跑直到符合预期。建议保留现有记录，补整体 selector、经典 RGB 构图和统一求解预算对照；几何估计器对照若承担核心因果结论，需要在严格共同观测集上重做。

本轮只读实验资产并运行 CPU 审计，没有训练模型、运行新的定位精度实验、调整冻结参数或改写论文数值。报告对应当前 v5 与 `docs/paper/evidence_20260917`。

**本轮实际验证范围**

- 306 个证据清单文件 SHA256 全部匹配。
- 从基线原始逐查询结果重算 40 个 method × scene × mask × stage 单元；查询数量、GT 顺序和表格一致。对有效输出从保存的位姿矩阵重新计算 TE/RE，与原始误差字段完全一致。失败保留在分母中。
- 核验 80 个 observation factorial 和 15 个 geometry 单元的查询列表、冻结配置、seed、像素域、几何有效域协议、部署地图哈希和原始结果归约。
- 另外重算当前协议下的 73 个历史消融和 30 个先验单元；均与现有 CSV 一致。这里不是把旧的有查询语义结果混入新协议。
- 在临时目录执行 v5 表格生成器；24 场景、24,700 queries、所有宏平均检查通过，没有重写原论文表格。
- 原方法环境 `g4splat`（Python 3.9.12 / Torch 2.0.1）下，语义查询禁用、stage fallback、factorial 和 selector 的 11 个相关测试通过。
- 在 `ulfloc_repro` 的 Torch 2.4.1 中，同组测试有 1 项失败：测试假定 `torch.unique(sorted=False)` 的顺序在分阶段/一次性拼接下不同。原环境通过，说明存在版本相关的兼容顺序假设；不能据此认定旧实验出错。补实验应固定原构图环境，升级时先做产物一致性验证。本轮没有修改这个兼容性实现。

主要审计产物：`artifacts/experiment_audit_20261009/summary.json`、`baseline_paired.csv`、`ablation_recomputed.csv`、`historical_recomputed.json`、`factorial_effects.json`、`mask_coverage_spotcheck.json`、`table_generation_checks.json`。可复用审计入口为 `tools/audit_experiment_evidence.py`。哈希和数值一致性不等于证明所有建图或语义标签都正确。

**语义 mask：结果为何不符合预期**

实验的处理变量是“建图阶段是否使用语义”，所有 query 均禁用语义，但保留几何去畸变有效域。这不是完整原生 mapping/query 语义联合策略的有无对照，也不是 AnyGSLoc 上游语义的消融。AnyGSLoc 使用的 MAtCha 先验仍有上游语义来源。

`tools/baseline_protocol_v2.py` 的 Off 把 object/sky 层替换成全真，保留同一几何有效域；On 读取原语义层并替换旧的 RGB>0 有效域。原生训练、STDLoc detector、ULF-Loc sampling/fusion 消费这个 masks 对象。On/Off 有独立模型输出，不是训练后删除 masks 文件。20 份 cfg_args 成对比较只相差 model_path。训练命令保持同一数据、初始化来源、学习率和 30k 步；ULF-Loc ShopFacade/Off 使用经记录的 v3 完成 PLY 与采样结果恢复融合，因此它的 v4 建图计时不能当完整训练成本。

语义操作本身也不只是“移除运动物体”：原预处理将所有 `isthing` 区域置为无效，sky 是另一层；训练还会把被 sky 层排除的 GT 像素改白，ULF 融合使用 object × sky × geometry。它可能改变重建、采样位置、描述子和可用覆盖；不能预设必然提升定位。

下表为 On − Off，负 TE / 正 R5 表示改善；R5 单位为百分点：

| 场景 | STDLoc ΔTE cm | STDLoc ΔR5 | ULF-Loc ΔTE cm | ULF-Loc ΔR5 |
|---|---:|---:|---:|---:|
| GreatCourt | −0.594 | +0.526 | +3.830 | −9.474 |
| KingsCollege | −0.844 | 0.000 | −0.875 | 0.000 |
| OldHospital | −0.436 | −0.549 | −0.891 | +1.648 |
| ShopFacade | +0.095 | −4.854 | +0.411 | −0.971 |
| StMarysChurch | −0.093 | +2.830 | +0.075 | −2.264 |
| 五场景等权平均 | −0.374 | −0.409 | +0.510 | −2.212 |

STDLoc 的 TE 在 4/5 场景改善。R5 宏平均下降很大程度来自只有 103 张 query 的 ShopFacade：开启后 3 张跨入成功区、8 张跨出，净少 5 张。StMarysChurch 则净多 15 张，GreatCourt 净多 4 张。若按所有 query 合并计算，STDLoc R5 实际从 33.681% 到 34.359%；既定 scene-macro 却下降。两种统计回答不同问题，不能为了好看换主指标；这个分析只解释其差异。TE 的中位数和 5 cm/5° 阈值成功率本来就不要求同向变化。

ULF-Loc 的宏 TE 退步主要来自 GreatCourt：8.175 → 12.005 cm；R5 29.605% → 20.132%，41 张获益、113 张受损，净少 72 张。剔除它再主张正收益是不允许的；它应是诊断对象。

对现存 mapping masks 的像素覆盖检查显示，GreatCourt object 层平均保留 99.835%，sky 层保留 67.899%，二者联合保留 67.733%；ShopFacade 联合保留 94.516%。这证明开关并非空操作，并提示 GreatCourt 需要重点查 sky/建筑边界及采样覆盖。它还不能证明天空误分割或覆盖损失就是精度下降的原因。

建议先输出 mapping 图像与 mask 对齐、过滤前后可用观测/点数、参考视角覆盖、query 匹配/内点分布及 refinement 前后成对误差。只有发现错位、错误读图、开关遗漏、缓存串用等具体错误才纠正并重跑受影响分支。若要声称语义的稳定因果收益，还需预先固定训练重复/seed，并区分训练随机性与 PnP 随机性；只增加同一地图的 PnP seed 不能衡量建图波动。

可用论文表述：“在统一无查询语义协议下，建图语义过滤的定位收益随方法、场景及指标而变化；当前结果未显示一致精度收益。”不应写成“不用语义 mask 会降低定位精度”，也不应反过来写“语义普遍有害”。

**各模块消融：保留什么，补什么**

| 项目 | 本轮判断 | 后续动作 |
|---|---|---|
| V2 screening | 五场景均有重新关联和重新三角化的 no-V2 构图记录，不是事后删除一个 mask；V2 不等于 semantic mask | 保留 80-cell factorial；描述为条件性收益 |
| 融合 vs 单观测 | deterministic mapping observation 与融合描述子对照有效；源质量分数保留 | 保留；仅能解释导出的 descriptor 改变，不等于移除所有多视图统计 |
| 稳健融合 vs 普通均值 | 已有同一 compact ID/XYZ 的普通均值对照，不能说缺失 | 把已有结果加入正文或附录，无需重复同一实验 |
| Projective completion | factorial 在 selection 前关闭它；旧的直接删 compact 点实验是另一控制 | 主引用 factorial；不要混用两种结果。若主张每点更有效，另做等 K 对照 |
| Full → Compact | 存储压缩证据有效，精度和耗时存在取舍 | 保留，不声称无损或 selector 加速 |
| 整体 selector | 现有 equal_size_quality 只替换最后的 pose completion，前缀相同 | 补同 full map、同每场景 K 的全局 Top-K / 空间均匀 / 完整 selector |
| ray / depth / center | 同 ID 和描述子成立，但 ray 保留原始轨迹估计，另外两者用共同可用观测 | 为隔离估计器重做 ray 的共同观测拟合；如有进一步筛点，三组一起限制并重评 |
| cycle / ray robust reweighting | 开关重构存在，结果混合；不是每个保护步骤都提高 TE 和 R5 | 保留附录，不因负结果反复调阈值重跑 |
| 先验兼容性 | 当前协议下的 30 个额外先验数值一致；Long-LRM 等负结果必须保留 | 不为了排名重做；质量与集成问题尚未分离的场景另作诊断 |

融合的重要细节：在固定 compact 地图中，single 为 9.246 cm / 38.437% R5，普通均值为 8.717 cm / 41.537%，当前稳健融合为 8.871 cm / 42.082%。多视图信息的证据比特定稳健规则的优越性更强。当前规则相对普通均值，TE 差 0.154 cm，R5 高 0.545 pp。

factorial 的八种其他因素条件下，fusion On 的宏 TE、R5 均改善；V2 的宏 TE 改善 6/8、R5 改善 4/8；completion 两指标均仅改善 4/8。这是描述性条件效应，不是八个独立统计重复。特别是 F=1,C=1 时，V2 在 full 上 TE 8.599 → 8.762 cm，在 compact 上 9.158 → 8.871 cm，说明筛选与压缩有交互，不能只摘最后一个数当 V2 的普遍优势。

容量也不固定：all-on compact 共 176,329 点，no-V2 是 178,835 点，no-completion 是 169,038 点。冻结 selector 参数不等于冻结 K；现有 factorial 测的是整个建图流程关闭某因素后的总效应。如果论文强调特定观测本身“更有信息”，需要另加容量受控分析。

ray 去重加权的宏 TE/R5 为 8.787 cm / 41.932%，完整方法为 8.871 / 42.082%；no-cycle 为 9.008 / 42.806%。不要将这些工程保护包装成每项都必要的独立创新。还应区分“无迭代重加权”与“无置信度加权”：当前 no_robust_ray 保留后者及原验收阈值。

**整体 selector 的最小补充协议**

固定同一全量候选地图的 XYZ、描述子、质量分数、可见性与 source row；只改变行索引。每个场景使用已有 selector 的最终 K：GreatCourt 49,009，KingsCollege 11,994，OldHospital 16,589，ShopFacade 12,520，StMarysChurch 86,217。

1. 全局质量 Top-K：使用与当前 selector 相同的 `anchor_matchability`，确定性 tie-break，不保留当前 matching/pose 前缀。
2. 空间均匀：预先定义只依赖 mapping XYZ 的 voxel/grid 或空间分层策略；固定尺度/规则，确定性轮转、截断及补足，最终必须恰好 K。不能为每个 test 场景寻找最优 voxel 尺度，也不能仅用固定 voxel 宽度而输出不同 K。重复点不允许用来补预算。
3. 完整 selector：复用现有冻结结果，但核验同 full map 和导出行序；最好所有子集都按源行排序，避免相似度平局的顺序差异。
4. 可选均匀无放回随机 K：固定数个抽样 seed，报告抽样波动；与求解器 seed 分开。它是基础参照，不需要很多随机方法。

最小需要新增 10 次完整 scene evaluation（Top-K/空间均匀各 5）；完整 selector 五组已有。报告 TE/RE、R5/R15、P90、失败率、实际 map bytes、选择耗时和完整在线时延。预算曲线可随后在预先固定的 0.5K/K/2K（不超过 N）补，但不同预算下完整 selector 必须有明确定义的预算接口，不能把已有输出随意截断后仍称完整算法。

现有 quality-completion 为 8.812 cm / 41.469% R5，pose-completion 为 8.871 / 42.082%。它只支持最后补点的指标取舍，不能代替以上完整选择策略对照。

**rendered-RGB + known-pose triangulation：值得优先补的构造对照**

这个对照直接回答复杂构图相对“Gaussian 仅作 RGB 渲染器”的增量，价值高于继续细分很弱的 safeguard。冻结同一 Gaussian PLY、renderer 版本与参数、渲染视图/相机、RGB 像素、SuperPoint 权重/NMS/预算、reference camera 列表和 online localizer。

仅从这些 RGB 检测和匹配构建轨迹；使用合理的 reciprocal/ratio、已知相机的极线验证、同视图重复观测冲突处理、正深度/夹角/重投影验收；固定相机内外参三角化，可做仅优化 XYZ 的点优化。普通描述子取各观测单位归一化后的算术平均，再单位归一化。不要使用单观测刻意削弱基线。

禁止从 AnyGSLoc 完成后的 anchor/track 出发再称“独立经典构图”；也不要让这个 RGB-only 对照隐式读取 alpha/depth/V2、当前 pose-aware 质量分数或 query GT。只有在逐字段核验缓存是未施加专用观测筛选的 RGB 特征时才能复用，否则从同一 RGB 重提特征。边界/去畸变的几何有效区域与语义、渲染有效性分开定义。

地图预算逐场景恰好上述 K，描述子维数与 dtype 一致，并报告字节数。若经典构图在合理冻结设置下得到的有效点数少于 K，报告真实覆盖不足，并额外在共同可达预算 `min(K,N_classic)` 对两者重评；不能复制点补 K，也不能悄悄把更小地图当等预算完成。

为区分构图和选择，至少并列 classic construction + 普通 Top-K、ours construction + 同一普通 Top-K、ours construction + full selector。Top-K 的质量依据应使用两种地图都可定义的 mapping-only 统计，或另用共同空间选择，避免用本方法独有分数单方面帮助其中一种。另保留 selector 专用消融，它应使用本方法的冻结质量分数。这样的设计既给出完整系统差距，也避免把选择收益全归给三角化/融合。

COLMAP 官方提供已知相机重建与 point triangulation 的路线，可以作为实现参照；必须明确锁住相机参数，不应改成相机可漂移的 SfM 再与已知位姿构图混比：[官方说明](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses)。是否使用 COLMAP 本身不是公平性的关键，输入、普通构造合理性、预算和求解器一致才是。

**定位参数公平性与提高迭代次数**

现有比较保留各自原生 operating point，是可报告的系统对比，但不能解释为严格相同计算预算下的架构速度收益。AnyGSLoc 是 .999 / 250–30,000；两基线 sparse 是 .99999 / 1,000–100,000。两基线 dense 则为 100–1,000 次/次调用，STDLoc 一轮、ULF-Loc 五轮；不能把 sparse 的 100k 上限套到每一层。

Cambridge 的 AnyGSLoc 实际五场景都为 2,048 query keypoints，reprojection threshold 为 11.922–12.000 px；baseline sparse 为 2,048 和 12 px。所以这里最突出的已知 sparse 配置差异是 RANSAC 预算与匹配筛选，不是简单说“我们的点数更少”。dense feature grid、描述子/匹配实现、65% gate 和计时边界仍有差异。

建议保留 native 配置结果，新增 solver-budget-matched 结果；严格标明它只控制求解预算，未自动统一所有算法与计时边界。方法内部消融则必须统一同一 query 特征、匹配策略、阈值、seed 和求解配置。不同像素分辨率下还要控制归一化/角度门限，不是机械填同一个像素数字。

最低成本的 AnyGSLoc 求解预算实验：

| 条件 | success_prob | min_iterations | max_iterations | 目的 |
|---|---:|---:|---:|---|
| A | .999 | 250 | 30,000 | 原配置 |
| B | .999 | 250 | 100,000 | 单改最大上限 |
| C | .99999 | 250 | 100,000 | 在 B 上提高置信度 |
| D | .99999 | 1,000 | 100,000 | 在 C 上提高最低预算，与 baseline sparse 对齐 |

先保存相同地图、同一 query 的全部最终 2D–3D correspondences、K、采样 seed，再在同一 PoseLib 2.0.5 环境重放 A–D；先确认 A 重现原始结果。所有场景完整报告，不按 test 最优选择性换主表。保存实际 iterations、refinements、内点数/比例、重投影目标、触顶比例、solver latency；最终若采用新配置，还需重新测完整 localizer 的 mean/P95 latency，而非把 CPU 重放时间代入旧 67.10 ms。

提高 max_iterations 只对仍未提前停止的 query 有影响。提高 min_iterations 会强迫高内点率案例做更多工作，提高 success_prob 会延长自适应预算。它们可能减少坏假设造成的失败，但不能修复错误地标坐标或有偏描述子，也不保证 benchmark TE 单调下降。PoseLib 用几何评分选择模型，而非 GT 位姿误差。经典 RANSAC 近似预算为 `log(1-p)/log(1-w^s)`；同一 w/s 下 .999 到 .99999 的概率项约增加 1.67 倍，实际还受实现的预算系数、上下界及局部优化影响。参见本项目所用版本的 [PoseLib v2.0.5 实现](https://github.com/PoseLib/PoseLib/blob/v2.0.5/PoseLib/robust/ransac_impl.h)。

当前 `PoseEstimate.diagnostics` 接收了求解器诊断，但 evaluator 仅保存 inlier_count 和最终位姿，没有保存实际迭代次数。因此目前没有依据回答“提高后会提升多少”或“已达到上限”。实验入口也硬编码 FROZEN；扫参应使用独立、显式记录配置的实验入口，不能把几个值改到现有输出目录再覆盖正式产物。

**执行优先级与停止条件**

先冻结上述补充协议并完成诊断记录，再依次做：求解器重放（低成本确认预算影响）、同 K 的整体 selector（直接补当前缺口）、经典 RGB 构图（最关键的整体贡献证据）、严格共同观测的几何对照（收紧几何因果解释）。语义分支只在发现具体实现错误时针对性重做；若要增加重复验证，预先固定 repeats 并全量公开，不能以是否得到预期为停止条件。

普通均值融合、现有 80-cell factorial、完整 Full/Compact、原生 mask On/Off 不需要因为结果不够漂亮而重跑。已使用 test feedback 的开发实验也不会因重新运行或增加 seed 自动变成独立 held-out confirmation。若需要这种主张，应有独立验证/测试设计。
