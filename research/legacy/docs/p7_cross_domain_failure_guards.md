# P7 跨域 Failure-Domain 审计与最小验证矩阵

## 结论

P7 的两个单因素不能不加区分地扩到所有数据集：

- `K_mapping=1024 -> 2048` 是 **Stairs 专属的前端证据密度因子**。`12Scenes/office2_5b` 和 Cambridge 已经以 2048 构图，继续扩这个因子既没有机制对照，也会把“保持 2048”误写成增益。
- 固定预算的 camera-pair policy 是 **Stairs 通过后才扩到 GreatCourt** 的几何因子。GreatCourt 已经接近 2048 keypoints，但高置信三角化漏斗非常弱，正好把“特征数量”与“有效几何基线”分开。
- `office2_5b` 不作为新策略的调参场景，只作为独立的 12Scenes correspondence/pose-tail no-regression guard。它的中央误差很好，但有明显灾难尾部；任何室内增益都不能用改善 Stairs 的平均值来掩盖这里的尾部回退。

本轮严格只读：没有修改 Map、selector、metric 或 deployment，没有读取 test 来拟合任何参数，也没有启动新的 GPU 作业。机器可读数值、路径与 SHA-256 在
[`p7_cross_domain_failure_guards.json`](./p7_cross_domain_failure_guards.json) 中。

## Failure domain 不是同一个问题

| 场景 | Mapping 前端 | Track 几何漏斗 | 当前部署 Map | 主要角色 |
|---|---|---|---|---|
| office2_5b | 1391/1391 mapping 图像均为 2048 行；旧 cache 的 NMS 字段未登记 | 179165 Track；23389 可三角化；6830 high-confidence；triangulated parallax median 6.334°，小于 1° 仅 0.47% | 8000 Track + 2113 surface/Gaussian lineage | 室内对应排序与 pose tail guard |
| GreatCourt | 配置为 2048；valid median/p10 为 2018/2004；旧 NMS 未登记 | 206755 Track；34150 可三角化；仅 1933 high-confidence；parallax median 2.270°，小于 1° 达 15.94%；covariance median 0.881 | 11892 Track + 7033 surface/Gaussian lineage | pair-policy 室外扩展 |
| StMarysChurch | 配置为 2048；valid median/p10 为 2020/1998 | 25529 可三角化；8269 high-confidence；parallax median 8.471°，小于 1° 仅 1.83%；covariance median 0.00326 | 16364 Track + 3627 surface/Gaussian lineage | GreatCourt 结果含混时的健康室外对照 |

这里有两个重要区分：

1. `pair-graph parallax` 衡量相机对对场景点形成的预期几何，`triangulated-track parallax` 衡量已经进入最终三角化 Track 的观测几何；两个百分比的分母和语义不同，不能混用。
2. GreatCourt 的 1.336 px median reprojection 看似正常，但只有 5.66% 的 triangulated Tracks 成为 high-confidence，且 covariance median/p90 达 0.881/28.15。问题不是简单“重投影残差偏大”，而是远景深度方向的可观测性弱。

因此第一性原理上的判断是：`office2_5b` 不支持“再加 mapping keypoints”，GreatCourt 则支持检验“同样数量的相机对能否提供更有效的基线”。

## office2_5b：独立的室内尾部哨兵

### K/NMS 与 pair graph

本轮复核了主任务此前生成的两个只读审计，未重新生成：

- density audit：`/tmp/lafgs_mapping_density_audit/office2_5b.json`，SHA-256 `6d280e...b8ee`；
- pair audit：`/tmp/lafgs_office2_5b_track_pair_graph_audit_v1.{json,pt}`，SHA-256 分别为 `75c69d...029c` 与 `8d2434...1dd`。

Density audit 证明 query-cache signature v10 的 1391 个 mapping query 全部请求并保留 2048 行，没有 mask drop。它只能证明 `K_mapping=2048`，由于旧 schema 没有 NMS metadata，`NMS=4` 必须记为 `unattested`，不能从历史默认值反推为已登记契约。

Pair audit 重建出 4882 个 frozen candidate pairs：baseline median 0.0618 m，mapping-point parallax median 1.809°，小于 1° 的 pair 为 13.91%，high-overlap/low-parallax 为 5.02%，effective-geometry proxy 为 69.85%。这不是 Stairs 式“明显缺少有效几何对”的强信号。旧 payload 没有逐 pair 的 raw/accepted/rejected edge sidecar，因此这些是 camera-pair 诊断量，不能被写成 pair policy 的因果收益。

### 已存在的 mapping-only oracle，不是本轮重跑

正式产物中已有 2026-08-11 生成的
`/mnt/pool/sqy/lafgs_indoor_pgt_runs_20260804/12Scenes/office2_5b/repeated_assignment_audit.json`
（SHA-256 `054cdba...d232`）。它使用 frozen Map/metric/teacher、256 个 uniform mapping queries，并在其中固定 64 个 query 做 oracle PnP；`uses_test_queries=false`。

关键结果：

| 指标 | 当前 Top-1 | Legal-positive oracle Top-2 |
|---|---:|---:|
| Positive-eligible recall | 62.56% | 74.46% |
| Median TE | 0.451 cm | 0.399 cm |
| Mean TE | 15.782 cm | 10.933 cm |
| P90 TE | 1.008 cm | 0.970 cm |
| CVaR95 | 244.648 cm | 168.136 cm |
| 5cm/5° recall | 93.75% | 96.875% |
| >100cm catastrophe | 3 | 2 |

Top-16 positive recall 可到 89.48%，但 oracle 仍只消除 3 个 catastrophe 中的 1 个。它证明“候选排名中有可利用上限”，不证明某个可部署 selector 已经存在，也不把 tail 完全归因于 descriptor。尤其 524288 个 replay rows 中只有 68245 行有 teacher-positive 标签，必须避免把 unlabeled row 当负例。

Office2_5b 的正式历史三种子结果为：raw P@2 固定 5.5154%，median TE 固定 0.4663 cm，P90 为 1.257–1.295 cm，但 >100cm catastrophe 为 10/11/11。它因此比 office2_5a 更适合作为 tail guard；后者 raw/inlier precision 更低，但正式结果只有 1 个 catastrophe，更适合 precision 诊断而不是本轮最小尾部哨兵。

## GreatCourt：pair policy 的室外扩展，不是 density 扩展

GreatCourt 的 frozen Track 构建配置已经是 `native_keypoint_count=2048`，scene calibration 的 valid keypoints median/p10 为 2018/2004。将 Stairs 的 density 因子扩到这里没有可识别的自变量。

相反，Track 漏斗提供了明确的 pair-policy 检验动机：34150 个 triangulated Tracks 中只有 1933 个 high-confidence，低于 1° 的比例为 15.94%，covariance trace median/p90 为 0.881/28.15。作为健康对照，StMarysChurch 的相应值为 8269 high-confidence、1.83% 和 0.00326。GreatCourt 的正式 760-query 三种子基线也有稳定尾部：median 9.79–9.93 cm、P90 46.04–47.37 cm、5cm recall 23.42–23.55%，catastrophe 27/31/33。

尝试对 GreatCourt 旧 50 GiB pickle query cache 做 exact pair audit 时，读取阶段约占 15 GiB RAM 并持续共享 NFS I/O，尚未产生结果即被停止。报告因此只引用 frozen training summary 与 Track payload；不会声称已得到 exact pair-level overlap、parallax 或 accepted/rejected match 分布。后续 pair arm 必须在构建时直接写 sidecar，避免再次反序列化整个 cache。

## 最小跨域验证矩阵

| P7 因子 | 首要机制场景 | 12Scenes | Cambridge | 扩展条件 |
|---|---|---|---|---|
| `K1024 vs K2048, NMS=4` | Stairs | office2_5b 只做 K=2048 code-path/parity 与 tail no-regression，不做 density gain claim | 不运行 | Stairs Track funnel 先通过，才重建 compact graph/teacher/metric 和 mapping pose |
| fixed-budget pair policy | Stairs | office2_5b 只做独立 no-regression，不参与阈值选择 | GreatCourt 为第一扩展；含混时才加 StMarysChurch | Stairs 必须在 identical K/cache/pair budget 下先过机制与 pose gate |

每个 pair-policy arm 必须固定：同一 attested query cache、`K_mapping`、NMS、相机对总预算、候选 universe、selector 预算、compact refresh 协议与 mapping query 集。唯一可变项是 pair selection policy。若任何一项同时变化，该结果只能叫 joint intervention，不能支持 P7 pair 因子的论文结论。

## 最小 pose gate

先在固定 256 个 uniform mapping queries 上运行 RANSAC seeds 2026/2027/2028。三种子均值必须同时满足 raw P@2、median TE、P90 TE、CVaR95 和 5cm/5° recall 不退化，三种子的 catastrophe 总数不得增加。中央误差改善不能覆盖 precision 或 catastrophe 回退。

只有 mapping gate 通过，才允许一次冻结后的 formal test：

- office2_5b：固定 405 queries × 3 seeds，执行相同 no-regression gate；
- GreatCourt：固定 760 queries × 3 seeds，除相同 no-regression 外，平均 P90 或 catastrophe 总数至少一个严格改善。

Test 结果不能用于选择 pair-policy 阈值、K、NMS、selector 或 metric；否则哨兵失去独立性。

## 已落实、未落实与停止线

- 已落实：office2_5b K_mapping/readout、pair graph、Track/surface composition、formal tail 和现有 q256/oracle64 lineage 均已审计；GreatCourt/StMarys Track funnel、Map composition 和历史正式 pose 已审计。
- 未落实：GreatCourt exact pair-level sidecar audit、GreatCourt mapping-only repeated-assignment oracle、任何 P7 跨域 compact refresh/pose replay。它们都应等待 Stairs 单因素先过机制门，避免昂贵但无因果价值的全链路跑法。
- 已停止且不应重复：cross-fitted secondary appearance mode / one-for-one two-prototype-like upper bound。Stairs 18 modes 虽改善 median/P90，却使 mean 1.3276→1.3349 cm、CVaR95 8.816→8.938 cm；ShopFacade 11 modes `meaningful_improvement=false`。如果 density 与 pair policy 都失败，下一步应测 frontend correspondence 上限，而不是再次扩当前双原型分支。

最终边界很清楚：**密度只回答 Stairs 是否“看得不够多”，pair policy 回答 Stairs/GreatCourt 是否“看得不够有几何信息”，office2_5b 则负责阻止任何局部增益把室内灾难尾部变坏。**
