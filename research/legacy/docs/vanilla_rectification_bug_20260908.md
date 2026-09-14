# Vanilla 2DGS：重复去畸变接入错误与固定地图补偿

## 结论与边界

2026-09-08 发现并逐像素确认：当前被反复用于弱先验实验的 OldHospital、
KingsCollege Vanilla 2DGS，其历史 Gaussian 训练输入重复去畸变。
不能再把这些先验与 MAtCHA 的全部差距解释成 vanilla 重建算法本身的上限。

此处是数据接入修复与诊断，不是应包装为论文主方法的创新。只凭两个
有接入错误的先验修复结果，也不能声称对任意先验质量不敏感。

保留历史基线、旧 Gaussian、旧地图和默认配置；没有恢复离线自定位反馈，
没有增加在线渲染、稠密匹配或在线地图写回。Gaussian 重训尚未执行。

## 1. 像素级根因证据

每场景均匀选择 12 张 mapping 图像，全部满足两次 bit-exact 复现：

1. `processed = undistort(CLAHE(raw))`，使用原 ULF 预处理的逐通道
   CLAHE(clip=2, tiles=8×8)，及 FP32 原 SIMPLE_RADIAL 相机参数。
2. Vanilla 训练输入 `prior_image = undistort(processed)`。

当前 query 加载器只读取 `processed`，不会再次去畸变。历史训练
`cfg_args` 和 command 指向 `*_mapping_only_undistorted_flat`，后者
指向上述第二次去畸变的图像，并以 PINHOLE 相机训练。

OldHospital 抽查图像额外 remap 的全图 P90 位移约 10.6–13.8 px，
不是可以忽略的浮点或颜色差异。实际 F0 Track 观测的位移 P50/P90：
OldHospital 1.47/7.99 px；KingsCollege 1.25/7.91 px。

该结论针对上述两个确有历史 Vanilla 输入的场景。额外检查
ShopFacade 时发现此先验目录不存在，未扩展为“五场景均确认”。
审计读取 source mapping RGB，但不读取 test RGB/GT，不训练描述子。

## 2. 无需重训的有界补偿实验

固定 F0 的 Anchor ID、Track、观测行、描述子和容量，只改变 XYZ：

- 由已知历史径向参数，将 render grid 上的观测坐标通过 SIMPLE_RADIAL
  正向畸变映射回“第二次去畸变之前”的图像域，再加原前端 +0.5 约定。
- 已知 mapping 相机固定，10 步 Huber-2 Gauss–Newton 重新拟合 XYZ。
  奇异、非有限或负深度的 Track 回退原 XYZ，不删除 Anchor。
- 同时跑 **相同求解器、但不修改观测坐标** 的 refit control。
- 重算 view support，移除不再适用的旧 ray covariance；每个地图仍只存
  一个 256D 描述子，在线仍为相同一次检索与一次 PoseLib。
- 修正观测保存在每个分支的 `observation_override.pt`；后续几何审计
  必须使用该覆盖文件，不能把原缓存坐标误当作新的拟合观测。

它只是历史错误的近似补救：已训练 Gaussian 已经吸收了错误监督，
错误 Track 选择也无法由坐标映射撤销，不等同于正确重建先验。
仅适用于有像素审计确认的这个错误，不能对 MAtCHA 或未知先验盲目启用。

## 3. 完整 test 结果（seed 2026）

OldHospital 182 帧，KingsCollege 343 帧。Base 相同配置，无在线 refinement。

| 场景 / 分支 | Median TE cm | Median RE ° | Mean TE cm | P90 TE cm | >100 cm |
|---|---:|---:|---:|---:|---:|
| Old / F0 | 21.1187 | 0.28133 | 31.1317 | 58.4321 | 4 |
| Old / 仅重拟合 | 21.6598 | 0.28079 | 33.9895 | 58.9921 | 4 |
| Old / 径向补偿 | **12.0407** | **0.22369** | **23.9067** | **55.0329** | **7** |
| Kings / F0 | 24.1260 | 0.17054 | 27.3089 | 52.2142 | 1 |
| Kings / 仅重拟合 | 24.3084 | 0.17350 | 27.5967 | 52.6495 | 1 |
| Kings / 径向补偿 | **17.6275** | **0.14980** | **20.1413** | **39.4537** | 1 |
| Kings / 多视角 depth-lift | 20.2734 | 0.17584 | 23.3330 | 44.2010 | 0 |

径向补偿 TE 改善 Old 147/182、Kings 282/343，median 相对下降约
43.0% / 26.9%。对照说明收益不是“再优化几步”的普遍效果。

Old 的全部 369,956 条 query→Anchor 原始匹配逐项不变；固定原基线
62,761 条 inlier，仅以 test GT 做求解后审计，重投影 median 3.314→2.744 px，
4px 内数量 36,814→42,136。该 GT 统计未参与修正或接受判定。

Old 的新增三条 >1m：基线 84.60→117.56、72.36→108.13、92.58→100.54 cm。
这些是真实尾部退化，不能因总体 median 提升而掩盖，暂不全局替换默认 F0。

## 4. 同时证伪/验证的小改动

固定 OldHospital 双先验地图，两种单实际观测描述子：
camera-bin-balanced cosine medoid、最接近 F0 融合描述子的实际观测。
物化前对全部 Anchor 精确重放 F0 融合，确认 bit-exact。

| 先验 | 原 F0 median TE | balanced medoid | nearest F0 |
|---|---:|---:|---:|
| Vanilla | 21.1187 | 21.8359 | 22.1577 |
| MAtCHA | 9.7066 | 9.2178 | 9.1948 |

Vanilla 两者均不改善主指标，停止作为缩小弱先验差距的方向；MAtCHA 的
小收益不能支持通用替换。这与早期 learned-surface medoid 实验对象不同，
此次直接评估无 metric 学习的 F0 projective Anchors。

早期已有对称 CLAHE/percentile 的独立负结果，本轮不重复光度参数扫描。
depth-lift 已在两个 Vanilla 场景改善 median，但 Old 的 MAtCHA 对照变差，
也不能作为不分先验的默认替代。

## 5. 已实现的接入修复

`scripts/prepare_rectified_prior_replay.py`：

- 要求同场景、mapping-only 的像素复现证据；校验被审计文件 SHA。
- 校验完整 mapping 图像注册表，排除 test；校验原位姿、内参、分辨率一致。
- 为两个场景分别准备正确输入，直接链接已一次去畸变的 `processed`，
  不再次重采样；复用原 mapping-only PINHOLE 模型和 SfM 初始化。
- 新目录保存全部输入图像 SHA 和复制的 sparse 文件 SHA；拒绝覆盖旧目录。

输出根目录：
`/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/corrected_prior_inputs_20260908/`。
OldHospital 895、KingsCollege 1,220 个 mapping 图像。这里只完成输入修复，
**尚未重训 Vanilla Gaussian，也尚未取得修复训练后的定位结果**。
历史 OldHospital 原版 30k 训练日志约 55 分钟；重训时保持相同训练预算、
初始化、分辨率与参数，仅改输入图像域，不重建 MAtCHA。

## 6. 复现产物与统计边界

根目录分别为：

- Old：`/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907/{OldHospital_vanilla_2dgs,OldHospital_matcha}`。
- Kings：`/mnt/pool/sqy/anygsloc_paper_experiments_20260902/CambridgePrior/KingsCollege_vanilla_2dgs`。

新地图：`representative_descriptor_20260908`、`radial_compensation_20260908`、
Kings `depth_geometry_control_20260908`。像素审计：`preprocessing_pixel_audit_20260908.json`。
汇总：Old 根的 `rectification_followup_20260908.json`，9 组新完整 test + 3 组基线。

共享机器 Base 时延：Kings 95.68→99.50 ms，Old 330.10→341.77 ms；
matching 均约 5 ms。补偿没有额外在线计算，但改变 XYZ 可改变 RANSAC
耗时；不能把不同运行时差全部归因于算法，也不能声称 Old 已达 10Hz。
本项目已反复查看 test，本轮属于 test-adaptive 开发，非独立泛化验证。

## 7. 第二随机种子与最终完成状态

seed 2027 完整 test 复核已完成：

| 场景 / 分支 | Median TE cm | Median RE ° | Mean TE cm | P90 TE cm | >100 cm |
|---|---:|---:|---:|---:|---:|
| Old / F0 | 21.1187 | 0.28133 | 31.8464 | 61.9579 | 5 |
| Old / 径向补偿 | 12.2875 | 0.22457 | 25.3404 | 56.0139 | 8 |
| Kings / F0 | 24.1260 | 0.17072 | 27.3797 | 52.6185 | 1 |
| Kings / 径向补偿 | 17.6275 | 0.14980 | 20.1426 | 39.4537 | 1 |
| Kings / depth-lift | 20.2734 | 0.17584 | 23.3343 | 44.2010 | 0 |

补偿对两个 seed 的 median TE/RE、mean TE、P90 TE 均改善，但 Old 的
灾难数也在两个 seed 均多 3 个。因此：保留可选修复分支，不全局替换
F0，不依据 test GT 为特定失败 query 加门控。更优先的下一步是用已准备
好的正确输入重训这两个 Vanilla 先验，再按相同 F0 管线建图与定位，
检验历史错误是否还污染了其他先验质量结论。

两场景全部 ID、描述子、CSR 逐项相等；Kings 686,850 条原始匹配也全部
不变。Kings 固定 seed2027 基线 200,087 条 inlier，GT 重投影 median
3.400→3.121 px，P90 却从 9.071→9.687 px：这同样说明补偿不是逐点
全面正确。GT 仅用于事后审计，没有进入构图。

描述性 paired-query bootstrap（2,000 次、未校正轨迹相关和多轮选择）
对 seed2026 的 median TE 差区间：Old [-11.07,-7.00] cm，
Kings [-8.77,-1.68] cm。不能把这些区间视为独立泛化显著性证明。

本轮共新增 **13 组完整 test、3,332 次 query 定位**，均已结束；
最终账本 `rectification_followup_with_repeat_seed_20260908.json` 含 13 组新评估
及 4 组旧基线引用，记录地图/结果 SHA、配对改善统计和匹配不变性审计。
相关回归测试 **41 passed**，`git diff --check` 通过。
没有仍在后台运行的本轮实验；未启动约 55 分钟的 Vanilla 重训。
