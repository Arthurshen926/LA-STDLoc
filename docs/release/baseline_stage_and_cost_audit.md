# 外部基线：阶段与成本核算

本轮不自行实现传统稀疏基线。比较行固定为 AnyGSLoc Compact+Fast、
STDLoc stage 1、STDLoc stage 1+2、ULF-Loc stage 1、ULF-Loc stage 1+2。
第二阶段增量只放分解表，不能替代累计延迟。两阶段失败均保留全部 query 分母。

## 已核验论文证据（reported，非同机复现）

| 方法/阶段 | Cambridge R50cm/5° | R15cm/5° | 时间 | 来源 |
|---|---:|---:|---|---|
| STDLoc stage 1 | NR | NR | 43.3 ms，按四个稀疏模块求和 | 原文 Appendix A |
| STDLoc stage 1+2 | 95.4% | 70.8% | 原文 152.3 ms；ULF-Loc 对照 3.9 FPS，分别保留 | 原文 Appendix A；ULF-Loc Table 2 / C.5 |
| ULF-Loc stage 1 | 89.4% | 63.1% | NR | Table 6 case 8 |
| ULF-Loc stage 1+2 | 93.7% | 72.0% | 4.4 FPS | Table 6 case 8 / C.5 |

STDLoc 的 43.3 ms 为 3.7+6.4+17.4+15.8，属于**推导的模块和**，不是新实测吞吐。
ULF-Loc Table 5 在 Heads 报告 STDLoc 50 min / 6566 MB、ULF-Loc 5 min / 1086 MB；
这些是论文定义的训练时间/显存，不能自动当成包含 SfM、mask、所有预处理的完整构建成本。
逐场景最终 TE/RE 另沿用带来源的历史外部数字清单；未报告初始 TE/RE 不补造。

来源：[STDLoc 原文](https://arxiv.org/html/2503.19358v1)、
[ULF-Loc 原文](https://arxiv.org/html/2605.04730v1)。两篇外部计时使用 RTX 4090，
本机 RTX 3090；禁止直接宣称同硬件加速比。

## 必须同时核算的成本

| 项目 | 构建端 | 部署端 |
|---|---|---|
| 相机 / SfM / 初始几何 | 输入来源、视角集合、时间、磁盘 | 实际运行所需标定；不把全训练集强算作部署必需 |
| object/sky 语义 mask | 模型权重、生成时间、输出大小、训练视角数量 | 若 query 需要，列 mask-ready 和含生成的 RGB-only 两个边界 |
| distortion mask / 去畸变 | 单独于语义模型记录，绑定像素域 | 重采样/掩码读取和应用，不误记为语义网络推理 |
| Gaussian / feature field | 原始重建、额外特征训练、峰值显存 | 各阶段实际需要的模型；只有第二阶段需要的资产单列 |
| 地标 / 检测器 | 选择、描述子融合、检测器训练 | 地标、特征权重、检测器权重、索引及 RAM/VRAM |
| 共享预训练模型 | 公共权重单列，不强加历史预训练 GPU hours | 单场景总占用与多场景摊销占用分列 |

STDLoc 正文明确 Cambridge 训练使用动态物体和天空 mask。
ULF-Loc 官方 README 指定 Mask2Former；本地 `ulfloc.py` 的 query 循环也应用
object/sky/distortion 三类预计算 mask。本地文件含研究修改，尚不能当官方原版复现。
AnyGSLoc 当前 Cambridge 同样应用这三类预计算 mask，不能隐去自己的额外先验。
室内无 mask 的场景明确使用 `none`，不是假装运行语义模型。

## 待执行与报告规则

- 官方代码 commit、局部修改、场景模型及数据域尚需完整绑定，再做同机基线复现。
- 缺失时间、权重、阶段结果一律 NR / pending，不填 0。
- 共享现成 mask 的定位实验可以比较 mask-ready 核心时间；不能声称 RGB-only 实时。
- 完整构建账本分别展示“给定现成先验的增量成本”和“从图像开始的总成本”。
- 语义模型仅离线使用时不逐 query 收费；两种使用阶段都出现时分别测量，不重复摊销同一批生成任务。
