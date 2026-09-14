# Vanilla 2DGS：构图改进续跑与否定结果

更新：2026-09-07。实验分支 `codex/anygsloc-paper-mainline`。

**2026-09-08 重要勘误：历史 Vanilla 输入在 OldHospital/KingsCollege
重复去畸变，已逐像素确认；固定地图补偿取得明显精度改善。
下文为修复前历史结果，不可直接归因于 vanilla 先验算法上限。见
[接入错误与补偿验证](vanilla_rectification_bug_20260908.md)。**

后续状态：同协议 OldHospital 双先验重建与完整 test 已完成，逐 query
TE/RE 重现历史 F0。新的固定几何描述子、几何来源、表面检测与观测修正
实验见 [受控诊断续报](weak_prior_controlled_diagnostics_20260907.md)。
下文“启动”部分是历史记录，不代表当前仍在等待基线重建。

## 当前决策

不更换默认 F0 构图方案。有效行补偿、联合 Track 净化、cycle/core 覆盖
筛选目前均未达到实质改善 median TE/RE 的标准。2× 超采样的小样本
结构检查也不支持直接扩大到全场景。

这些结果否定的是下面列出的具体实现和设置，不能据此断言弱先验
没有提升空间，也不能将所有差距归因于 prior 的不可恢复信息损失。

所有候选只使用 mapping Gaussian renders 和已知 mapping cameras。
无源建图 RGB、无 test GT 构图、无离线自定位反馈、无 query 渲染、
无稠密匹配。最终 xyz 仍由多视图射线三角化产生。
已经反复查看 test 来决定研究方向，因此以下是 test-adaptive 开发实验，
不是独立留出集的泛化证明。

## 中断前已完成的定位实验（本次重新核对产物）

固定 test、seed 2026、identity metric、Base PoseLib，无额外在线 polish。
TE 单位 cm，RE 单位 degree。

| 场景 / 方法 | Median TE | Mean TE | P90 TE | Median RE | >100 cm |
|---|---:|---:|---:|---:|---:|
| Kings / F0 | 24.1260 | 27.3089 | 52.2142 | 0.17054 | 1 |
| Kings / 有效行补偿 3072→2048 | 24.5203 | 27.4223 | 52.8925 | 0.17401 | 1 |
| Kings / 联合净化 LOO10 | 23.9536 | 27.2853 | 52.1888 | 0.17152 | 1 |
| Old / F0 | 21.1187 | 31.1317 | 58.4321 | 0.28133 | 4 |
| Old / 联合净化 LOO10 | 21.1732 | 30.0671 | 57.6040 | 0.28085 | 4 |
| Old / Cycle Core + Coverage Reserve | 21.3711 | 30.9303 | 58.4458 | 0.28517 | 4 |

- 有效行补偿确实恢复了观测：Kings 保留行 2,081,458→2,492,245。
  但 Anchor 也从 78,248 增至 95,629，不是严格同地图容量比较；
  4px 内点 GT precision 57.63%→56.87%，没有转化为更好的定位。
  未继续跑 OldHospital，也未默认启用。
- LOO10 限制每个 Track 最多调整 10% 的观测，仅在三项残差中至少两项
  异常时降权，并保护唯一 pose-bin 观测。Kings/Old 实际降权比例
  3.78%/4.08%。两场景 mean-of-medians TE 仅改善约 0.059 cm。
  Old 的 mean 改善包含少量长尾 query 的较大贡献，不能描述成普遍提精度。
- 更早的全观测联合降权使 Old mean TE 31.13→33.83 cm，灾难数 4→5，
  已否决。保守 20% 版本 median 21.2575 cm，也未晋级。
- Core/reserve 保留全部 cycle Track，只对 chain 做空间 voxel 或 mapping
  图像 cell 的新增覆盖筛选。Old 地图 76,701→68,343 Anchor，匹配时间
  5.010→4.736 ms，但 median TE/RE 均变差。不是附件提出的完整高质量
  Precision Core，不能用这个结果否定所有覆盖感知选择器。
- 这些运行的整段时延受共享 GPU/CPU 负载影响，不能用不同运行的
  total_ms 差直接归因于算法。Old 的 RANSAC 仍是主要耗时项。

产物根目录：
`/mnt/pool/sqy/anygsloc_paper_experiments_20260902/CambridgePrior`。
场景目录为 `{KingsCollege,OldHospital}_vanilla_2dgs`；补偿检测为
`KingsCollege_vanilla_2dgs_vcd3072`。评估子目录分别为
`evaluation/base_seed2026`、`evaluation/base_joint_purification_loo10_seed2026`、
`evaluation/base_cycle_core_coverage_reserve_seed2026`。

## 本次完成：2× 超采样 12 视角检查

OldHospital，固定相同 12 个 mapping cameras，原分辨率输出，2048 个
SuperPoint、NMS=4、同 V2 阈值。2× 渲染后 RGB/alpha 做 area 下采样，
depth 做 alpha 加权下采样，排除非有限/非正深度；相机坐标系不变。

| 同视角统计 | 原分辨率 | 2× 后下采样 |
|---|---:|---:|
| 检测行数 | 24,576 | 24,576 |
| V2 有效行 | 20,201 | 20,064 |
| 低 RGB 结构支持拒绝 | 3,048 | 3,213 |
| 深度断层拒绝 | 848 | 831 |
| 极端 distortion 拒绝 | 161 | 153 |
| border 拒绝 | 584 | 580 |

拒绝原因可重叠，不应求和当成总拒绝行数。
12 个视角中 10 个有效行数下降、2 个上升；有效率约 82.20%→81.64%。
这只说明当前小样本不支持投入全场景，不是最终定位无收益的证明；
未测这个变体的 Track/cycle/covariance，也未测最终 test 位姿。

当前代码重新产生的原分辨率对照与历史缓存，在全部 12 个对应视角的
keypoints、descriptors、scores、K、pose、image size、depth、alpha 上
逐项 tensor 相等。2× 与原分辨率的 K/pose/image size 也逐项相等。
新旧审计的 prior SHA 和 V2 thresholds 相同。

输出位于 OldHospital 场景目录下的 `supersampling_panel12_native` 和
`supersampling_panel12_2x`。2× 审计 SHA256：
`e954512a8dda28b313c259b39d50d2f474c6043ddd809657386cf218a44bc95e`。
2× 缓存 SHA256：
`6ea69f8503fb1d78f44f8ab80d6b9bded7715d18ec2a4b55d5180b8eae87b617`。

## 实现修正与限制

- 修复超采样时 `NaN depth × 0 alpha` 污染有效深度的问题，有回归测试；
  factor=1 保留原数值路径。本实验没有证明这个边界问题解释了历史差距。
- 修复 core/reserve 子集选择在 Anchor 数恰好等于 camera 数时误裁
  `query_bins` 的问题，有回归测试；此前 Old 实验不满足该触发条件。
- 2DGS 包装器实际没有转发 `rasterize_mode=antialiased`，新缓存明确记录
  effective mode，不能仅凭旧配置字段声称 2DGS 已启用该选项。
- 当前联合净化只做一次再加权重三角化；descriptor/depth 的参考是
  leave-one-observation-out，几何重投影不是独立 LOO triangulation。
  descriptor 使用加权均值而非 medoid；尚未融合 V2 连续质量、expected/
  median depth 一致性证据，也不是附件完整的多轮鲁棒净化。
- 旧净化产物曾将装饰器所在的 torch 文件误记为 reconstruction 源哈希，
  当前实现已修复。旧数值结果保留作开发证据，不作为完整代码溯源的晋级产物。

## 协议对照：当时的启动记录（后续已完成）

先在 OldHospital 对 MAtCHA / vanilla 2DGS 都从头执行：render extraction、
V2 audit、association、triangulation、completion、identity metric、test。
不复用旧 `existing_base`，也不启用上述候选。两种 prior 顺序跑在 GPU 0，
保存完整 stage 命令、日志、时间及输入哈希；启动前已记录关键源码哈希，
完成后还需核对运行期间是否发生变化。

新产物根目录：`/mnt/pool/sqy/anygsloc_weak_prior_protocol_20260907`。
当时处于已启动状态；后续已完成，最新核对结果见上方续报。

启动时的关键源码哈希保存在
`docs/evidence/weak_prior_protocol_20260907_source_start.json`。
完成后可用 `python -m scripts.compare_weak_prior_protocol --left <MAtCHA run>
--right <vanilla run>` 检查：所有阶段必须实际新执行；除 prior/output 路径外
命令必须一致；map 文件字节必须匹配报告；构图源码哈希、方法合同、mapping
名字和 test 名字必须一致。检查器明确不把这些检查称为完整 camera tensor
parity 或渲染正确性证明。

下一步的判断依次是：先量化构建版本差；再做固定 xyz/Track 的
mapping RGB descriptor 诊断与 GT geometry oracle（严格隔离于部署）；
若支持 Track 几何瓶颈，再验证同 pair 预算的基线多样性。
不要继续把“更多有效点”“更高内点数”直接等同于最终 pose 提升。

诊断解释也有边界：在渲染 keypoint 坐标处采样真实 RGB descriptor，仍
保留了渲染 keypoint 的位置偏差；它不是“彻底消除外观问题”的 oracle。
用 GT 投影选择 Anchor 则只是乐观的可解性诊断，不能直接证明这些点具有
正确物理身份或几何精度，绝不能进入部署或正常精度表。
