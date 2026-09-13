# Vanilla 接入修复：扩展 bug 审计与正确输入重训

## 本轮结论

重复去畸变是已逐像素证实的历史输入错误，不能再把历史 Vanilla 与
MAtCHA 的全部差距解释为重建方法上限。本轮修复了更多相邻数据契约问题，
但必须区分“代码路径存在错误”“旧实验实际受影响”“修复后定位改善”。
下列修复中，只有前述重复去畸变已确认影响两个历史 Vanilla 先验。
其他修复没有被冒充为已测得的定位收益。

前一轮固定地图径向补偿的双种子结果及尾部退化，见
[原始实验报告](vanilla_rectification_bug_20260908.md)。它不等于正确输入重训。
本轮不重建 MAtCHA，不恢复离线自定位反馈，不增加在线渲染或稠密匹配，
不使用真实 mapping RGB 作为定位描述子。

## 已实现并验证的修复

| 问题 | 修复与证据 | 历史影响边界 |
|---|---|---|
| LongLRM / i3DGS 对 processed 无条件再次去畸变 | 显式 rectified/distorted 输入域；非零畸变且未指定域时拒绝猜测；真实 OldHospital 32 张 LongLRM 输入逐像素一致 | 确认代码路径问题；未定位到这两条分支的既有完整评估产物，不能推断其历史精度影响 |
| 缩放后内参用名义比例、remap 网格约定不清 | 按真实取整后的输出尺寸缩放 K；COLMAP 半像素中心转换到 OpenCV 整数中心；校正后再 area resize | 合成像素与 K 回归测试通过 |
| FOV-only 加载器丢弃非中心主点 | 准备阶段将像素和 K 一起重映射到中心主点；加载阶段对未校正输入明确报错 | 已审计 20 个实验 manifest / 15 个独立数据集均为中心主点，不是它们历史退化原因 |
| Palette / CMYK 等模式被当作 RGB 数组 | 显式转换可支持的图像模式；不支持位深报错；RGB/RGBA/L 原路径不变 | 防止静默颜色错误，未宣称已有 Cambridge 输入采用这些模式 |
| 测试注册表漏项、重复或划分重叠未充分拒绝 | 测试缺失/重复报错；适配器拒绝 mapping/test 重叠；保留 Cambridge 官方文本表头兼容 | 防止静默减少评估帧；OldHospital 182 帧回放 TE/RE 与原基线逐项完全一致 |
| 阶段缓存只凭输出存在就复用 | 绑定命令、配置/输入文件、代码与关键环境；校验输出 bundle；执行中内容变化也拒绝认证 | 旧无认证缓存不可静默复用，需新输出目录；不会删除旧结果 |
| V2 审计未绑定产生特征的实际先验 | 新缓存绑定 PLY SHA；核对先验、SH、Gaussian 类型和背景；物化校验分片一致性 | 历史无绑定缓存只能显式兼容，不能追认来源可靠 |

### 真实链路 smoke

12 个 OldHospital mapping 视角重新渲染、提取 SuperPoint、V2 审计、
重新关联和三角化、completion 全流程完成，得到 857 个 Anchor。
24,576 个原始观测中 V2 保留 20,201 个。未使用 source RGB 或 test。
这验证接口整合可运行，不是完整场景精度实验。

## 排除与未解决项

1. Old Vanilla 2DGS、Kings Vanilla 2DGS、Old MAtCHA、Old Vanilla 3DGS
   四个先验导出检查：所有保留 PLY 字段逐项 bit-exact，SHA 匹配，
   未发现零四元数或 scale 指数溢出。未发现“导出把几何/外观写坏”。
2. Old Vanilla 原生 rasterizer 对 gsplat 的 12 视角同相机 RGB 对照：
   11 张 PSNR 约 55–59 dB，一张 seq1/frame00244.png 为 32.14 dB。
   图像主体对齐，异常集中在天空/上部区域。近裁剪从 0.01 改成 0.2
   并未消除该异常，因此不提升为默认修复；具体原因仍未确认。
3. 不为让测试变绿而更新历史独立审查凭证：P9 的 reviewed implementation
   hash 因 datasets.py 的真实修改失效；该分支需要重新独立审查。
   它不属于本轮 F0 训练/建图/定位执行链。

## 回归验证

全套 pytest：1379 passed、1 skipped（CUDA smoke）、1 deselected
（上述历史 P9 审查 hash）；CUDA smoke 单独启用后 1 passed。
后续补充了显式零/负输出尺寸拒绝测试；相关定向测试 29 passed，
日志 `/tmp/anygsloc_bug_audit_targeted_20260908.log`。`git diff --check` 通过。
完整测试日志 `/tmp/anygsloc_full_regression_final_20260908.log`。

## 正确输入重训：运行中，不冒报定位结果

实验根目录：`/mnt/pool/sqy/anygsloc_rectified_prior_retrain_20260908`。
正确输入仅含 mapping：OldHospital 895 张、KingsCollege 1,220 张；
复用原 PINHOLE 模型、位姿与 SfM 初始化，输入图像与 sparse 文件逐项校验 SHA。
官方 2DGS commit `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`，
30k 步、原分辨率、SH3、seed0、原损失/增密参数；只修复输入图像域。

- `OldHospital`：GPU0 正常训练中。
- `KingsCollege`：GPU2 第一轮约 14,650 步 CUDA OOM，失败日志完整保留。
  申请单块 18.61 GiB，日志显示已分配约 1.40 GiB / 已预留 18.69 GiB。
  怀疑分配器碎片，但尚未证实。
- `KingsCollege_allocator_retry`：GPU2 独立目录同参数重试，仅设置
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` 并增加检查点。
  第一轮没有检查点，不能伪装为续训；重试是从相同初始 seed 重新训练。
  不删视角、不降低分辨率、不改变模型预算。不保证该策略一定解决 OOM。

每个成功训练会自动依次导出 RGB prior → 同配置 F0 建图 → 完整 test
seed2026 / seed2027；`replay.json` 与分阶段日志记录状态和失败。
当前没有新的“正确输入重训后”定位结果，不能用上一轮补偿数字替代。
共享训练期间的时延不能作为独占硬件 10 Hz 认证。

本项目已反复查看 test，后续数字仍属于 test-adaptive 开发验证，
不能包装成独立泛化证明，也不能据两个场景声称对任意先验质量不敏感。

### 主要审计产物

- `input_contract_audit.json`：数据集相机与先验导出校验。
- `input_adapter_checks/OldHospital_long_lrm/anygsloc_long_lrm_input.json`：真实输入准备。
- `renderer_parity/summary.json`：原生/gsplat/近裁剪对照。
- `pipeline_smoke/projective_map/report.json`：真实缓存绑定链路验证。
- `legacy_baseline_guard_OldHospital_seed2026`：182 帧旧基线不变性回放。

以上相对路径均相对于本节实验根目录。
