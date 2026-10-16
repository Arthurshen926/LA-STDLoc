# 投稿图件候选版核验记录

本轮只修改制图工具、图件和文档；未修改 AnyGSLoc 算法、地图或实验指标。

## V2 科学一致性

使用五场景选定 mapping view 的原始 Gaussian prior，核对 prior SHA256 后重新渲染 native RGB、alpha、depth、distortion。全部全帧算子完成后才裁剪 ROI。

| 场景 | 视图 | 稠密 mask 采样与全部历史特征点判断一致率 |
|---|---|---|
| GreatCourt | seq2/frame00141.png | 100% |
| StMarysChurch | seq4/frame00232.png | 100% |
| OldHospital | seq6/frame00053.png | 100% |
| KingsCollege | seq6/frame00040.png | 100% |
| ShopFacade | seq2/frame00053.png | 100% |

没有通过平滑、填洞或插值改善 mask 的观感；没有语义标签。灰化图只用于展示，未送入特征提取器。S03 保留没有触发拒绝的全白子项，并加细边框区分面板与背景。

## 图件整理与检查

- 统一输出目录：`paper/figures_submission_candidate/`。
- 八张正文候选、三张补充图；PDF/SVG/PNG 三格式均可打开，SVG XML 可解析。
- 检查全套联系表及方法总览、问题引入、V2、判定分项的大图；修正问题引入箭头与文字的交叠，缩减 V2 面板留白。
- S02 保留已核验的逐查询样本，重新排为原图 + TE/RE 数值对照，避免沿用带草稿页脚的旧图。
- source manifest 保存 CSV、选样表、历史逐查询证据 manifest 的 SHA256；数值不手工美化。
- 源码通过 `py_compile`；代码/文档通过 `git diff --check`。

## 仍需作者决策，而非新增画图

- F01 与 F07 的精度/速度信息有重叠，可根据正文篇幅二选一或主文/附录拆分。
- F06 的八种组合及 F08 的全部先验可以放附录，正文用核心表格保持凝练。
- 不把 V2 示例写成平均精度显著提升；不把紧凑选择写成零损失或已证实加速。
- baseline 最小部署资产、完整离线成本仍需另外补证，当前图不作无依据的横向存储/构建成本比较。

因此，本版可作为整套排版和图注审阅的候选定稿；不能因图件齐全，就声明尚未审计的成本证据也已完备。
