# AnyGSLoc 投稿实验计划

日期：2026-09-13。状态：**用户已授权执行；先发布回归，再全量评估，复用有效先验**。
目标：完成统一发布版本的证据闭环；不新增方法模块，不用换指标来掩盖负结果。
面向机器人视觉定位顶刊的充分性组织，不等同于承诺录用。

## 1. 借鉴对象与写作方式

[STDLoc](https://arxiv.org/html/2503.19358v1) 的实验分别讨论定位精度、采样/地图规模和各阶段；
借鉴其将精度表与地标数量曲线分开的组织，不把我们的单阶段耗时与其某个子阶段直接比较。
[ULF-Loc](https://arxiv.org/html/2605.04730v1) 将 sampling/features/LGCV、初始与最终定位、
地标规模及地图构建开销分开分析；借鉴其多阈值召回与成本维度，而非复制其模块或只选有利阈值。
二者以论文版本/补充材料/官方实现核验具体协议后填表，不能用论文 reported 数值冒充同机复现。
投稿前再按 [T-RO 作者说明](https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/) 核验格式、材料和政策；本计划不编造强制实机要求。

## 2. 论文只保留三个主张

| 主张 | 必要证据 | 不能据此声称 |
| --- | --- | --- |
| Gaussian 可转换为独立稀疏定位资产 | 多先验正确链路 + 只携带部署资产也能定位 | 不需要任何上游 RGB/标定、隐私保护、任意先验均等精度 |
| 跨视图身份 + 射线几何比简单深度代理更合适 | 固定 Anchor/描述子只换 XYZ；关联干预及结构统计 | 保证恢复错误关联、对所有几何基线都更优 |
| 紧凑地图 + 简单单次求解有实用效率优势 | 等容量选择对照、Full/Compact×Standard/Fast、同机计时、实际 bytes/RAM/VRAM | 压缩必然无损、所有场景10 Hz、点数比例等于完整存储比例 |

稳健聚合、IRLS、候选 completion、mask 等忠实写入实现，不各自立独立创新。
若某个主要主张只得到部分支持，收紧标题和摘要，不用附录弱阳性弥补核心证据缺失。

## 3. 全局实验合同：先锁定，再统一执行

发布设置：Compact + Fast；冻结 SuperPoint；单描述子；一次 PoseLib；不含离线反馈/在线 refinement。
seeds=2026/2027/2028。场景关键点预算、重投影门限来自 mapping-only 标定，记录 resolved 值；
不因某一测试场景结果差临时加阈值/开关。新计划本身不洗掉历史 test 调参事实。

所有单元格记录：release git tree/commit、依赖锁及 GPU/driver/CUDA、prior/cameras/processed RGB 域哈希、
mapping/test 图像名清单与数量、mask 策略和来源、map/metric/export 哈希、selector 版本、
完整 resolved config、seed、命令、成功/失败和排除原因、逐 query 结果与 wall clock。
map_bound metric 在导出时去除，但来源哈希保留。查询输出只写新目录，不覆盖旧结果。

复用条件：上游像素域正确；地图、缓存、metric、配置及解析数值一致；逐帧结果齐全；
旧 runtime 与 release 等价检查通过。否则列为待重跑，不能只复制 summary。
产物缺失/异常/运行中保持对应状态；只在全部 query 已处理、失败有显式记录、表格审计通过后标 complete。
修正双重去畸变前的实验排除，且撤回相关 prior-quality 归因；记录无效原因而不删除历史。

## 4. 主实验矩阵与优先级

| 编号 | 场景/条件 | 输出 | 是否可复用 |
| --- | --- | --- | --- |
| E0 发布一致性门槛 | 小/大地图、空匹配/失败/掩码、CPU/GPU；随后两场完整 query 对照 | ID/XYZ/desc/匹配/PnP输入/最终pose等价报告 | 本轮只有代码级回归；完整场景门槛后做 |
| E1 主表 | Cambridge5 + 7Scenes7 + 12Scenes12，共24场景 ×3seeds | 三数据集精度主表、逐场景附录 | 先核验旧产物，不强制重建 MAtCHA |
| E2 外部对比 | STDLoc、ULF-Loc 与 ours，同一数据划分；两篇均单列初始与最终定位 | reported 与 reproduced 分表；同机精度/耗时/存储 | 不自行实现传统基线；缺失阶段不拿最终值代替 |
| E3 多先验 | Cambridge5 ×7先验设置 ×3seeds，上游均正确处理 | 先验兼容/精度/成本表 | 复用有效重建，不重建 AnySplat |
| E4 核心消融 | 下节4个主组；Cambridge5为主，室内补代表难/易场景 | 因果消融、Pareto曲线 | 已有219次按合同复用，不机械全重跑 |
| E5 部署成本 | 同一 release 的24场景资产统计；同机串行计时；外部可比场景 | latency分布、吞吐、bytes、RAM/VRAM | 并行旧计时不能用于正式表 |
| E6 构建成本 | 多先验/大小场景，拆上游重建与下游转换 | wall/GPU hours、峰值内存、资产大小 | 审计日志，有缺项就待测而非推算 |
| E7 独立验证 | 冻结后的独立序列/场景；有可信参考位姿才报绝对精度 | 不调参迁移与连续部署结果 | 待确定可用数据，不伪装既有test为独立 |

E1最多72次 seed评估，不代表72次建图；每个冻结场景一张地图复用。
E3七设置为 MAtCHA、Vanilla2DGS、Vanilla3DGS、Long-LRM raw/+1k、i3DGS raw/+1k，理论105次 seed评估；
与 E1重叠部分去重。Long-LRM额外两轮优化是附录诊断，不临时择优替代raw/+1k。
逐先验记录实际输入/优化视角数、分辨率、优化步数、总时间；
MAtCHA优化视角子集与完整mapping集合的差异必须列出，不能误写为所有先验同样训练数据量。
上游不可用的单元格明确 N/A/原因，不挪用另一个先验的地图。

本轮不自行实现强传统基线。STDLoc、ULF-Loc 均展示 stage 1 初始位姿和 stage 1+2 最终位姿；
额外记录 stage 2 增量时间，不能用增量冒充完整定位延迟。未报告的阶段记 NR，复现待完成记 pending。
部署成本按阶段列实际必要资产、模型权重、RAM/VRAM、初始化与在线耗时；
构建成本分相机/SfM、语义 mask 等先验生成、Gaussian/特征训练、地标与检测器构建。
三种方法同等记录 object/sky/distortion mask 的来源、模型、生成时间、文件大小及使用阶段。
离线 mask 不逐帧重复计费；若 query 使用预计算 mask，必须注明 mask-ready 条件，
另列包含在线 mask 生成的端到端成本（无法测得则 NR，而非 0）。共享先验只计一次。
本地实测与论文 reported 分表；RTX 3090 与 RTX 4090 的计时不能直接声称同机加速比。

## 5. 消融只保留四个主组

### A. Anchor 几何来源（核心）

相同 Anchor ID、相同描述子、相同 Fast、相同 seed，只把 XYZ 从射线几何替换为观测深度反投影世界坐标的分量中位数。
所有深度使用观测自身内参/位姿和相同pixel center。共同有效行需至少3个有限正深度且alpha≥0.05，报告排除比例。
现有五场景均无排除，但最终输入变动需重新核验，不能预设为0。

主要读出：median TE/RE、R5、P90 TE、≥1m失败/灾难率；Cambridge同时R15/R50。
可解释读出：对应不变率、射线残差、三角化视差/条件数、深度分歧，与最终误差作分层分析而非GT驱动筛选。
控制解释：selection由射线地图产生，因此只是固定当前选择集的几何干预；不能宣称深度方案的全局最佳性能被排除。
如果补直接Gaussian中心基线，必须先定义合法的投影/身份对应，另列诊断，不能把不存在的track凭空配给中心。

### B. 关联前观测有效性与跨视图一致性

reference / noV2 / noCycle，分别干预；不把多种约束一起删掉再归因给某一个。
重建受影响的 tracks 和地图；这是pipeline-level干预，点数变化本来就是后果，需如实报告。
如做固定容量对照，另列容量配平策略，不和自然输出结果混写。

主要读出：同一组最终误差/召回/尾部指标；机制读出：检测数、有效观测数、pair/reciprocal边数、
冲突/循环通过率、track数量与长度、三角化可接受率、重投影残差、最终Anchor数、下游inlier比例。
为何在关联前做：改变候选集合可能改变NN身份，不等价于最终地图少删几个点。
不强求所有场景都涨median；OldHospital强、其他弱也忠实呈现。

### C. 紧凑地图选择（存储主贡献）

主对照 Full / Compact；子对照 quality-only / quality+matching / full-selector。
解释位姿补全时必须增加**同点数**质量补全，不能用更多点打更少点。
保留来源行，区别固定容量控制与自然自适应输出；包括无收益场景。

主要读出：部署map实际bytes、Anchor数、mean/P95延迟、VRAM、median TE/RE、R5/R15/R50及灾难率。
机制读出：每mapping查询可匹配覆盖、图像cell/深度bin/空间体素分布、局部信息特征值；不拿mapping coverage代替定位。
容量曲线预设 Full的10/20/40/60/100%加自然Compact operating point。
若实现是前缀截断而非重新求解，要标明；不能称每个预算的最优选择。曲线上的新策略只用于分析，不反向调主线。
主文精简为精度—实际bytes—延迟 Pareto图，详细selector stages放附录。

### D. Fast operating point（工程取舍，不强行创新）

Full/Compact × Standard/Fast 的2×2；同一地图只改在线设置。
Standard：不保留筛选、confidence .99999、min1000、max100000。
Fast：保留 .65、至少256、confidence .999、min250、max30000。
代表场景再拆 retain-only / budget-only / both；区分减少对应与RANSAC预算的独立作用。
固定 query预算/分辨率/描述子/CPU线程/库版本；不得顺便启用额外refinement或去重。

主要读出：median TE/RE、所有固定召回、P90 TE、失败率；mean/P50/P90/P95延迟、>100ms比例、实际吞吐。
机制读出：原始/保留匹配数量与inlier ratio、RANSAC iterations及耗时占比，按query难度分层。
如果某些场景tail明显下降，应写为accuracy-efficiency trade-off而非“精度完全不变”。

### 附录而非主贡献

single/mean/current descriptor；noIRLS；noCandidateCompletion；mask协议差异；先验预处理bug审计。
completion删除会改变候选/容量，明确干预层级。当前保留completion：不隐藏实现，也不在摘要宣称大贡献。
不把已有大量微小消融全部塞进主文；完整结果表保留在补充材料。

## 6. 指标合同：允许不同重点，不允许结果后选指标

所有组共同底座：每场景median TE(cm)/RE(deg)、R(5cm,5°)、P90 TE、≥1m数量/比例、无姿态失败数量。
Cambridge追加R(15cm,5°)、R(50cm,5°)；室内追加R(1cm,1°)、R(2cm,2°)。
旋转误差采用相对旋转角，平移误差采用相机中心世界距离；统一单位与严格/包含边界（沿用实现 `<`，灾难 `>=`）。
不能把纯平移R5、5m召回与5cm/5°放同一列；论文baseline定义不同则分列、不自行换算。

失败进入所有query的召回分母；无有效pose记为无限误差，并另列求解成功率。
成功子集median可作为补充但必须注明条件，不能通过剔除失败美化主表。
主表用scene-macro；每场景先各seed计算指标再报告mean±std，另附各seed。
pooled必须从逐query合并重算；不将场景median平均值命名为pooled median。
三seed只是求解随机性，不是3个独立数据集；配对置信区间按场景/连续序列分组，报告重采样单元。
给相同query的误差差值/胜平负和灾难变化，避免仅看median忽略难例。

“精度基本保持”的判据在最终执行前和应用容忍度一起定稿；本计划不把任意阈值包装成预注册。
不以统计不显著推断等价；若要equivalence claim，预先定义等价界及置信区间方法。
失败多或样本少时只陈述观测，不作过强显著性推断。

## 7. 正式实时性与存储协议

### 在线计时

- 同一机器、GPU独占，CPU线程固定4；记录CPU型号、频率策略、GPU/驱动、精度、分辨率、query数。
- batch1，预加载map/model，50次mapping/合成预热不计分；完整test串行定位3轮，轮次顺序固定并报告。
- 主口径：准备好的RGB tensor+标定到返回pose，包含掩码/特征/匹配/筛选/主机拷贝/PoseLib和必要同步。
  若mask需额外在线生成，生成时间不能从端到端表消失：另报ready-mask与含生成版本。
- 图像解码/resize/rectification、model/map初始化另外列；可再报从图像字节到pose的应用端到端口径，各baseline一致。
- 分段用CUDA events或同步，不能把异步提交时间当GPU计算时间；分段profile与轻量端到端计时分开测。
- 报mean/P50/P90/P95、max、>100ms比例、实际processed queries / elapsed wall seconds；不以mean倒数冒充系统吞吐。
- 失败query和长RANSAC不能排除；记录调度并发、GC、I/O与暖启动状态。
- “平均10Hz”需要实际吞吐≥10且mean≤100ms；若P95明显超100ms，只能称平均而非稳定100ms截止期。
- 不将其他方法fine-only或GPU-only数字与ours端到端比较；reported的硬件/范围不足则标不可直接比较。

### 存储与内存

三层分别统计：①上游Gaussian/相机；②下游构图cache/track/中间map；③部署所需资产。
部署至少XYZ float32、256D descriptor float32、ID int64，每点张量payload1044 bytes，
另算文件序列化元数据、SuperPoint权重、相机参数；报告精确文件byte和MiB(2^20)，不混MB。
Full和Compact均以相同minimal schema导出，否则CSR是否携带会混淆压缩结论。
共享权重只计一次；逐场景map另计；索引/缓存、host pinned buffers也计入运行内存。
报CUDA allocated/reserved峰值与CPU RSS增量/峰值，初始化与稳态区分；不把torch allocated当系统总显存。
执行最小资产冒烟：移除搜索路径里的研究代码/数据，只有wheel、权重、map、准备好的query和camera仍能定位。
未完成该测试前不能写“无需任何构图依赖部署”。

### 构建成本

上游重建（raw/+1k等）与下游render/extract/associate/triangulate/select/export分别报告wall time、GPU hours、峰值CPU/GPU内存。
缓存命中与冷启动分表；完整mapping相机数与上游优化相机数分列；不要只报最后优化1000步却漏掉前馈/预处理。

## 8. 最终论文表图清单

| 编号 | 内容 | 建议列 |
| --- | --- | --- |
| 表1 | 24场景精度主结果，按数据集拆表 | method/protocol、medianTE/RE、联合召回、失败 |
| 表2 | 公平效率/资源比较 | batch/resolution、mean/P95、throughput、>100ms、mapMiB、weightsMiB、RAM/VRAM |
| 表3 | 先验兼容与成本 | prior/raw+steps、input/optimized/mapping views、TE/RE/recall、构建时间 |
| 表4 | 几何/关联核心消融 | XYZ source/V2/cycle、Anchor数、TE/RE/R5/R15/P90/灾难 |
| 表5 | 压缩与Fast解耦 | map policy、solver policy、bytes、mean/P95、TE/RE/召回 |
| 图1 | 系统流程和资产边界 | 上游/下游离线/在线，不画不存在的反馈循环 |
| 图2 | 资源—精度Pareto与容量曲线 | 实际bytes/latency，不只Anchor数 |
| 图3 | 延迟CDF及耗时构成 | frontend/match/filter-transfer/PnP，含难例 |
| 附录 | 逐scene/seed、弱模块、负结果、数据域审计、全部manifest | 所有主表可回溯到逐query记录 |

## 9. 执行顺序与完成定义

现在不创建运行队列。得到后续执行指令后：

1. E0一致性门槛 + 历史产物valid/development/invalid台账；确定新结果不可复用的最小集合。
2. 锁定独立数据/外部baseline版本/容忍度，冻结manifest；补E1/E3/E4缺口，尽量复用正确先验。
3. 所有吞吐型实验结束后，独占硬件执行E2/E5正式计时；统计部署真实bytes与内存，补E6。
4. E7不调参独立验证；不可用时明示泛化与实机证据限制，不伪造“全面验证”。
5. 最终一次性生成全部表格，检查重复、缺失、单位、失败分母、报错退出和来源；无数值由手工填补。

收尾门槛：所有论文表中单元格都有状态与依据；主张只覆盖被数据支持的范围；
重构包通过等价与最小部署测试；同一发布配置的精度、延迟、资源能联合复现；
既有test调参、输入域bug和负结果披露完整。达到这些才是证据完整，不是每个模块每个指标都赢。
