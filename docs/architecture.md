# 整仓架构：一套正式主线，不是部署旁包

分支codex/anygsloc-repository-refactor，仓库位置仍是/root/STDLoc。
anygsloc统一包含离线、在线、数据和评估；research/legacy只保存研发历史。

## 依赖方向

| 模块 | 允许依赖 | 禁止依赖 |
| --- | --- | --- |
| online | features、artifacts、config、PoseLib | mapping、priors、evaluation、legacy |
| mapping | data、features、priors、utils及纯几何/选择内核 | legacy脚本、反馈训练、test评估 |
| evaluation | online、data、metrics、artifacts | 修改地图、训练/调参 |
| CLI | 按命令惰性加载 | 单图定位时预加载renderer或历史模型 |
| tests | 主线及显式legacy对照 | 把测试sys.path写入正式包 |

正式包不再导入旧scripts、map_learning、topology。按符号依赖抽出内核，不copy整个历史训练程序。
保留数据schema名是为了读取既有产物，不是遗留代码依赖。

## 离线职责迁移

| 原来源 | 新位置 |
| --- | --- |
| probe_rendered_rgb_track_map | mapping/render_observations：只保留render cache，不含旧Track probe分支 |
| audit_v7_mapping_render_quality_shard | mapping/audit_render、render_certificate |
| materialize_v8_v2_projective_map | mapping/materialize及projective内核 |
| v6_anchor_map、v24_anchor_view_support | mapping/anchor_map、view_support |
| full_selector_pilot | mapping/compact：不读jobs.json，不生成消融队列 |
| adaptive_distillation的5个helper | mapping/pose_evidence：没有反馈主程序 |
| sufficiency_selector、dynamic_reserve | mapping/selector、observability、coverage、pose_information |

mapping/build统一编排观察→两份V2审计→Full→Compact/部署导出，使用当前Python解释器。
stage receipt记录运行/完成/失败/中断及输出哈希；代码、先验、相机、划分、mask元数据和参数共同绑定合同。
只对completed且hash相同的阶段resume，文件存在不算成功；partial输出保留，要求明确处理或新目录。
不引入跨项目GPU队列，不自动运行test。

## 在线和评估

online/runtime只负责特征→匹配→筛选→PnP。地图/query归一化次数、tie顺序、pixel-center和solver参数保留。
identity shim只用于兼容旧map导出，不在在线路径训练。
evaluation/runner负责数据集遍历、mapping-only标定、逐帧结果、失败与汇总；与在线算法分开。
失败进入分母；非有限误差JSON写null并标记。progress.jsonl不是完成凭据。
普通计时标telemetry，正式精度/mean/P95/吞吐/内存按投稿计划另做。

## 兼容边界

- 旧Python导入和python -m scripts不再是主线API，旧任务用历史源码快照或独立legacy目录。
- Gaussian/缓存/地图/identity tensor数值和schema不因改名而重新生成。
- 外部实验文件的硬编码路径不批量改写，避免污染既有证据来源。
- 构图额外依赖在.[mapping]；包内存在离线模块不等于在线加载它们。
- 没有借重构扩展原始畸变相机支持，输入域仍须正确矫正且主点居中。
