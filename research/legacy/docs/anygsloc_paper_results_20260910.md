# AnyGSLoc 当前完整实验结论与论文表格

## 已完成的范围

GreatCourt Long-LRM +3062 两个 seed 最终评估均完成。现在已完成 24 场景
Base、Cambridge 五场景六种替代先验条件共 30 组/60 次 test，以及两个
Long-LRM 两轮优化对照共 4 次 test。**这不等于原协议全部三 seed、全部消融
和部署基准都已完成。**本次核验了 97 个评估文件，包括可用补充 seed 和历史
V31 三个启用场景；不挑最好 seed。

[完整 13 张表格](/mnt/pool/sqy/anygsloc_paper_tables_20260910/paper_tables.md)

绝对路径：`/mnt/pool/sqy/anygsloc_paper_tables_20260910`。每张表同时提供
`table_XX.csv` 与 `table_XX.tex`；`all_evaluations.csv` 提供所有核验评估，
`results_inventory.json` 保留完整测试名单检查、输入地图及结果文件哈希。
`historical_trials_not_for_main_table.csv` 单独列出历史探索，不混入修正后的主表。

## 主线比较（场景等权平均各场景中位数，cm / degree）

| Dataset | STDLoc | ULF-Loc | AnyGSLoc-Base |
|---|---:|---:|---:|
| Cambridge | 10.060 / 0.142 | 8.252 / 0.134 | 8.573 / 0.124 |
| 7Scenes | 0.760 / 0.243 | 0.700 / 0.197 | 0.681 / 0.222 |
| 12Scenes | 0.394 / 0.175 | 0.337 / 0.154 | 0.248 / 0.110 |

数字为 seed 2026，与论文发表结果并列，不是同配置复现。外部平均值从论文
逐场景已取整数据重算，故可能与论文的一位小数 Avg 不同。
来源：[STDLoc Tables 1/2](https://arxiv.org/html/2503.19358v1)，
[ULF-Loc Tables 1/4/A](https://arxiv.org/html/2605.04730v1)。STDLoc 12Scenes
数值取自 ULF-Loc 所列对照，不声称来自 STDLoc 原论文。

## 能支持与不能支持的结论

- Cambridge MAtCHA Base 在平均 RE 上较好，TE 接近但略逊于 ULF-Loc；不是
  每场景都胜出。Vanilla 3DGS 五场景整体为 7.888 cm / 0.115 degree，不能
  与 MAtCHA 逐场景取最优后拼成“主方法”。
- Vanilla 2DGS 整体为 8.599 cm / 0.121 degree，与 MAtCHA 很接近。修正输入
  链路后，“Vanilla 2DGS 普遍严重逊于 MAtCHA”的旧判断不再成立。
- 前馈先验仍敏感：Long-LRM raw/+1k 为 230.291/53.218 cm，i3DGS raw/+1k
  为 22.688/10.652 cm（均为五场景宏平均 TE）。不能宣称任意质量先验皆同精度。
- 7Scenes 平均 TE 具有竞争力，但 5cm/5deg recall 97.72%，低于论文对照；
  Stairs 中位数 1.804cm/0.535degree、mean latency 306ms 是明显短板。
- 12Scenes 平均误差优势较明显，但 5cm/5deg recall 98.92% 仍低于 ULF-Loc
  99.2%；office2_5b 存在灾难性尾部，不能仅用 median 掩盖。
- Long-LRM 两轮预算控制显著改善 median，但没有解决整体鲁棒性：Church
  仍有 134/530 帧 >=1m，Court 有 128/760 帧 >=1m（seed 2026）。其映射数据
  遍历次数和 LR horizon 同时变化，不能将增益完全归因于覆盖遗漏 mapping 图像。

## 实时性

| 条件 | Query-weighted mean ms | P50 ms | P90 ms | 1000/mean Hz |
|---|---:|---:|---:|---:|
| Base / Cambridge MAtCHA | 193.28 | 70.45 | 531.18 | 5.17 |
| Base / Cambridge Vanilla 3DGS | 156.59 | 见完整数据 | 299.29 | 6.39 |
| Base / 7Scenes | 69.40 | 32.91 | 98.65 | 14.41 |
| Base / 12Scenes | 73.39 | 42.29 | 95.27 | 13.63 |

STDLoc 原文 Appendix A 为 **152.3 ms / RTX4090**（约 6.57 Hz），正文约 7 FPS；
ULF-Loc Appendix C.5 为 **4.4 FPS**，其中 STDLoc 对照 **3.9 FPS**。必须保留
来源差异，不选择较慢的 STDLoc 值宣称加速。

本地为 RTX3090，共享机器运行的 localize 调用时间，**不含图像读盘、valid mask
加载和地图初始化，也未统一预热**。室内可以描述为当前记录下平均定位核心吞吐
超过 10Hz，不能说所有场景完整端到端均稳定 10Hz。Cambridge 尚未达到目标。

## 稿件发布前须披露/补齐

1. 24 场景不是全部 MAtCHA：Cambridge 用 MAtCHA，19 个室内场景用 Vanilla 2DGS。
2. 室内采用官方 SfM pseudo-GT 和 reference points 初始化；manifest 明示这些
   点可能包含测试视角重建证据。不能将其表述为严格 train-only 几何初始化。
   可与同 pseudo-GT benchmark 并列，但要交代先验初始化来源并审计协议差异。
3. MAtCHA 优化图像子集与 Vanilla full mapping、Long-LRM 32 输入不同；对比
   不等价于同预算先验重建竞赛。所有 AnyGSLoc 建图本身仍不读真实 mapping RGB。
4. 历史 V31 为 test/scene-calibrated 可选在线分支；正式 Base 不启用离线反馈。
5. 缺少统一三 seed、完整关键组件消融、同硬件且含 I/O 的隔离延迟/峰值内存
   基准。不能把这批完成的矩阵等同于整篇论文的实验已经全部完备。

生成脚本：`scripts/build_anygsloc_paper_tables.py`；外部数字配置：
`configs/paper_baselines_20260910.json`。主表未加入 oracle 或旧输入 bug 结果。
