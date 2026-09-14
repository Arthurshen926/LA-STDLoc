# 整仓重构回执

日期：2026-09-13。分支：`codex/anygsloc-repository-refactor`。
仓库仍是 `/root/STDLoc`；不是另开项目，也不再是只加部署包。
上一轮未提交改动已随工作树保留。本轮没有自动commit、推送、重建先验或启动正式实验。

## 实际完成的结构调整

- 离线、在线、数据、特征、先验读取、评估、配置和资产合同统一在 `anygsloc/`，共57个Python文件（含包入口/初始化文件）。
- 从38个原模块按符号依赖抽出149个顶层函数/类，去掉这些模块中未被主线使用的训练/反馈入口和相关传递导入。
- `mapping/pose_evidence.py`取代从adaptive_distillation导入helper；`mapping/compact.py`不再读取历史jobs.json。
- 构图入口不再经过旧scene脚本，不再自动串接test评估；统一命令为build-map / compact-map / export-map / localize / evaluate。
- 根目录原common/data/evidence/features/localization/map_learning/priors/topology/scripts等历史目录实际迁入research/legacy。
- 旧配置、研究测试、报告、论文草稿/表格和可视化材料一并保留在归档树；当前docs只保留架构、迁移和投稿计划。
- 包只发布anygsloc命名空间，包含离线代码但在线惰性加载，不依赖research/legacy或旧根模块。
- 旧build与lafgs.egg-info移入research/build_artifacts，保留但不纳入Git/发布包。之后正常构建会产生新的build目录。
- 根requirements改为引用pyproject的mapping extra，旧依赖列表和LaFGS CITATION保存在legacy中；正式论文引用待作者/题名确认。

完整清单：`research/repository_moves.json`；抽取来源与SHA：`research/migration_symbols.json`。
相比上一轮，这次真正移除了正式链路对历史源码树的依赖。

## 数值不变性验证

1. 当前主线测试：`python -m pytest tests -q`，**77 passed**。
2. 对149个抽出符号的可执行AST做来源对照：148个在还原import命名空间、忽略docstring排版后相同。
   剩余一个是三角化子进程启动函数，因目录层级改变修改包搜索根；有独立子进程测试。
3. 完整选择器：1200个合成候选，执行历史选择器原始控制流与新入口，选出的Anchor行精确一致。
4. 合成几何链路：3视图/32点，关联→射线三角化→候选materialize→view support→Compact→部署导出通过。
   并行两进程与串行三角化XYZ/描述子逐值相同。无unused候选时，completion按冻结逻辑明确退回base。
5. 真实在线冒烟：ShopFacade同一mapping图像seq2/frame00001.png、实际mask、相同1024关键点/12px回归参数。
   新旧关键点、描述子、query/anchor索引、匹配分数、PnP inliers及最终pose七项逐值一致。
   此为软件回归，不是该场景正式设置的性能评估，也没有用真实mapping RGB重新建图。
6. 建图合同测试：dry-run零输出，改变输入改变合同hash；completed产物hash校验；中断不算完成；partial不覆盖；锁拒绝同时resume。
7. 编译、未定义名称静态检查和git diff空白检查通过。旧算法的schema字符串保留，不改已有tensor协议。

警告：PyTorch TypedStorage弃用；失败样本的无限误差参与NumPy分位数时可有invalid警告，JSON按非有限值null记录，失败仍在分母。
没有忽略失败测试，也没有把旧的全部历史测试称为已重跑；研究测试仍在legacy/tests。

## 安装后的验证

wheel：`/mnt/pool/sqy/anygsloc_release_archive_20260913/repository_v3/anygsloc-0.3.0-py3-none-any.whl`。
本次构建131,827 bytes，不含SuperPoint权重或地图；不能用代码包大小冒充地图存储。
SHA256：`7442531811bce938b1a682de4469d74cf0000576e2cf73d30a6e705bb1c94142`。

安装到独立临时target，Python -I且cwd=/tmp：

- 在线定位冒烟成功，未加载mapping/priors/evaluation。
- 55个非顶层入口子模块全部可导入，未导入任何旧根模块或research。
- 使用已安装代码完成上述合成离线链路（包含真实fresh-process三角化）和导出。
- renderer审计、materialize、render-observations三个内部CLI的help可用；这不是实际Gaussian渲染回归。

## 迁移中发现并处理的依赖

本机 `/root/.cache/lafgs/superpoint_v1.pth` 原是指向旧build/lib下权重的软链接。
移动build后它失效，已改为稳定缓存中的相同权重文件；SHA仍为
`52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e`。
旧链接保存在 `/root/.cache/lafgs/superpoint_v1.pre_repository_refactor.symlink`，原权重也仍在归档build中。
没有换权重/重训练。发布机仍须自行提供该权重，不假定仓库内带有模型文件。

## 归档完整性

对迁移前工作树检查点中的1890个归档文件逐字节核验：**0差异**。当前release文档因继续更新不列入此计数。
检查点及SHA见research/README。迁移是保存历史，不是清空项目；没有删除Gaussian、观测cache、dataset或实验结果。
第一次并发归档出现tar目录变化警告，已经用之后成功完成并校验的检查点代替，不把前者当恢复依据。

## 当前完成边界

**仓库组织、主线依赖解耦、统一入口和代码级回归已完成。**
但不等于所有实际场景都已用新入口走完真实渲染→构图→评估，也不等于已证明新版本的速度或精度提升。
后续统一正式实验前仍需：真实大小场景完整构图/逐query等价检查、独占计时、完整部署bytes/RAM/VRAM、依赖lock与外部基线。
这些仍按投稿实验计划执行，本轮没有启动。
