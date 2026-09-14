# GreatCourt fixed-budget pair-policy mechanism: Stop

## 结论

GreatCourt 的 `parallax_diverse` 单因素实验在进入 variant provenance replay 和完整
pipeline 之前触发预注册 **Stop**。它在相同 K=2048、NMS=4、1,531 个 mapping query
和精确 5,254 pair budget 下，确实把低视差 pair 大幅移除，却同时使 triangulated Track
保留率跌到 94.562%，并把 triangulated covariance p90 恶化到 control 的 1.680 倍。
六项 gate 中四项通过、两项失败，因此 `mechanism_gate_passed=false`，不得用后续训练或
pose 结果为失败的机制补救。

这不会推翻 Stairs 的 mapping-only Go；它否定的是把当前
`overlap-constrained parallax-aware` policy 升为**跨场景默认策略**。当前跨域决策为
**No-Go，默认继续使用 `nearest`**。Stairs 的收益应解释为场景特定有效性，而不是已经
证明的通用 camera-pair objective。

## 冻结契约与产物

本轮只使用 mapping 数据，`uses_test_queries=false`。两个 arm 共用以下输入：

- fresh-cache manifest SHA-256：
  `078d6e8b9600be57023cede08c837af45bb44e3b09a50d54e6e5ea77bc95febe`
- K=2048/NMS=4 fresh query cache SHA-256：
  `0550b59e3350cc6759515f0904a96243f9849f12bc868d449415fb2758153e92`
- frozen Track payload SHA-256：
  `f0fec708e00ab0f19160485059574eccdb8f6aba0484a257627f9b56a4cc90af`
- sparse-refresh equivalence report SHA-256：
  `9f1593b7143bfd42cc69866ff6bd7d27e36de4d13c87989b677f8b4b7724f5e2`
- nearest exact-control parity SHA-256：
  `2114c403b21c352ec2d345e8c6a99476a88fe72dcf5320fb8da20484e2a015f9`

Factor root 为
`/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/greatcourt/k2048_nms4`。
control report / factor SHA-256 分别为
`c46b360b465c5cff02719dc1ead05b77ff247b9ce1eaad3b4860406e84bbe8f9` /
`78cd1d04ee14f1b954ace0bf225c9ad6169a5c729cb756af612e2272d4a35084`；
variant report / factor SHA-256 分别为
`48e5ef86cd7594d62786231113dffa979442baf30c09b53460d2d5b4248f040b` /
`a6dbcf51b17290e41ab804e1f12353552cb519948d82393545d0cb2846c83027`。

最终 gate 位于 `contracts/mechanism_gate.json`，SHA-256 为
`201362cde50296eac4abc0e7813ee618f970fdae4c248c854aaa81eb14b72f35`；
其自身为 `valid=true`，并精确绑定 manifest/cache/frozen Track、两个 report、K/NMS、
query 顺序哈希、pair budget 和 policy 参数。

## 六项预注册 gate

| gate | nearest control | parallax-diverse | 门槛 | 结果 |
|---|---:|---:|---:|---|
| 精确全局 pair budget | 5,254 | 5,254 | 完全相等 | Pass |
| pair parallax <1° 比例 | 64.1745% | 2.3220% | 至少下降 10 pp | Pass（-61.8524 pp） |
| triangulated Tracks | 34,150 | 32,293 | 保留至少 95% | **Fail（94.5622%）** |
| broad eligible Tracks | 16,985 | 19,473 | 保留至少 98% | Pass（114.6482%） |
| triangulated covariance p90 | 28.1497 m² | 47.2850 m² | 不高于 1.05x | **Fail（1.6798x）** |
| broad support/query p10 | 0 | 3 | 保留至少 95% | Pass；control 为 0，证据力有限 |

辅助诊断与 gate 一致：pair parallax median 从 0.7829° 增至 2.5915°；total Tracks
从 206,755 降到 129,268，而 strict/high-confidence Tracks 分别从 4,382/1,933 增至
7,145/3,667。这说明 policy 不是“完全无效”：它强力改变了 pair 几何并提高部分高门槛
集合；但它没有同时保持可三角化证据量和几何尾部条件。不能用通过的代理量覆盖失败的
必要条件。

## 第一性原理解释与执行边界

camera-pair policy 的目标不是最大化 parallax 本身，而是在固定成本下最大化**可用于稳定
定位的、条件良好的独立 3D 身份证据**。GreatCourt 结果证明 overlap + saturated
parallax + pose diversity 仍不是该效用的充分代理：更高的 pair parallax 可以与更差的
Track covariance tail、略低于门槛的 triangulated yield 同时出现。当前结果不能唯一确定
是 depth distribution、view connectivity 还是 correspondence survival 导致该差异，因而
不授权事后增加场景特定阈值来挽救这一 arm。

Stairs 已完成 exact lineage、fullchain rebuild 和 q256x3 mapping-pose gate，并以
`e1366dd367b1be8b2ec9797c64da6f9bfde4b370aae7a1acfbf02133fd921a73`
绑定其 mapping Go；但 GreatCourt 的独立 Stop 表明这个 Go 只能保留为 scene-specific
结果。论文方法当前仍以 `nearest` 作为跨域默认，`parallax_diverse` 不进入统一主线。

为验证 fresh cache control 等价，nearest control 的 provenance replay/parity 已运行。
在 GreatCourt gate 失败后，**没有运行 variant provenance replay 或 variant lineage
audit，也没有运行 fullchain rebuild、mapping pose 或 formal test**。这既避免浪费计算，
也保证 test 没有参与选择。机器可读结论见
`docs/evidence/p7_greatcourt_pair_policy_mechanism_stop.json`。
