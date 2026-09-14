# P8 GreatCourt `cycle_verified_fisher` mapping-only Stage-A 结果

## 结论

GreatCourt 的冻结 P8 V1 Stage-A 是有效的科学 **Stop**：9 项预注册 gate 中 8 项
通过，唯一失败项是
`verified_triangle_camera_fraction_not_lower=false`。正式 gate 为
`valid=true`、`mapping_only=true`、`uses_test_queries=false`，decision 为
`STOP_BEFORE_TRACK_REUSE`，进程按合同以状态码 2 结束。

因此 P8 V1 的跨域结论已经闭环：Stairs 的 Stage-A/Stage-B 正向结果只保留为
scene-specific mechanism signal；GreatCourt 在 Track 构建前给出了预注册反证。
**不运行 GreatCourt Track/Stage-B、cross-scene aggregator、fullchain、function
graph/Map、mapping pose、formal test，也不把 P8 切换成共享默认。** 当前共享默认继续
保留冻结 V3 `nearest`。

## 冻结合同与正式产物

GreatCourt 使用 1,531 个 mapping query、K=2048、NMS=4、精确 5,254 pair budget
与同一 9,875-pair bounded probe universe。query 顺序 SHA-256 为
`3ac3c28420a68ac72c779f3f0699ce0773745be62a845f72d0fe91024134451b`；candidate
graph 为一个连通分量、零 isolate、最小 degree 5。matcher 仍是冻结的
`0.65 / 0.01 / 2.0px / topK1 / -1.0 / -1.0`。

预注册机器合同
`docs/evidence/p8_cycle_verified_fisher_preregistration.json` 的 SHA-256 为
`39264e748adc735a85f141ac307aad6cba8b6a1d825880ce500ef3d86d6bc728`；本轮没有
看到结果后修改预算、utility、matcher 或 gate。

| 产物 | SHA-256 | content SHA-256 / 说明 |
|---|---|---|
| pair proposals | `de9cc32ca29a932f95c839143ccb7a8c034c9e861c0c9ce9f5bd192a6467abf6` | content `53b03061bfcfa9da0a7db166731d1d51f9144c168d2d32b5c6cb708a381f18e8` |
| bounded match probe | `3064b56497f253955a26e34ddae38fba4d63dda953d82e7f52b68e99de01d392` | content `f55bc491fb1080dde4593b4d3a7c21df33ff2e46af7023c0569be253e1eb7538`；4,038,756 matches |
| P8 pair selection | `e0e297597937b96af50cba2c0b95b055e523733e2601127218f93ad8994e9661` | content `f1e2194a890983aad889ad08194d5d12003ab2f4d295e485b72bf5781e8e3d0d` |
| Stage-A gate | `1f1f9c6665009b897f93f6dce0121e13e2ce74e8997962acf16674491c8d64f5` | `STOP_BEFORE_TRACK_REUSE`；exit 2 |

proposal、probe、selection 和 gate 的文件哈希已直接重算；PT schema/content 字段、JSON
gate、invocation sidecar、mapping-scope V2 equivalence 及退出码也已交叉检查。正式产物由
精确增量 selector 版本
`199c187acd8a6df018e3630fe0babda3739e68c1` 生成。机器可读摘要位于
`docs/evidence/p8_cycle_verified_fisher_greatcourt_result.json`。

## Stage A：效用变强，但闭环相机覆盖显著收缩

selector 严格保持 5,254 pair 预算、candidate graph 的一个连通分量、零 isolate 和最小
degree 1。Fisher utility、verified triangle 数量以及其余六项合同/图约束全部通过；失败
只来自闭环覆盖相机比例。

| Stage-A 指标 | nearest same-probe control | P8 V1 | 变化 / gate |
|---|---:|---:|---:|
| confidence-weighted Fisher utility | 2,132.8152 | 4,489.3332 | **2.1049x**；Pass |
| completed verified keypoint triangles | 1,028,346 | 1,874,006 | **1.8223x**；Pass |
| 参与 verified triangle 的 mapping cameras | 1,524 | 1,230 | **-294** |
| verified-triangle camera fraction | 0.995428 | 0.803396 | **-0.192031 / -19.2031 pp**；Stop |
| selected pair budget | 5,254 | 5,254 | 相同 |
| selected graph components / isolates / min degree | 1 / 0 / 5 | 1 / 0 / 1 | Pass |

这不是执行或 lineage 故障，也不是“图断开”导致的表面失败。P8 selection 的 pair graph
仍全局连通且没有孤立相机，但大量高效用 triangles 集中在更少的相机上：control 只漏掉
7 个 mapping cameras，variant 漏掉 301 个。也就是说，**总闭环数与总 Fisher 信息可以
同时上升，却仍牺牲轨迹中证据的空间/相机覆盖**。这正是预注册时把 camera fraction 单列为
不可补偿硬门的原因；2.1049x utility 和 1.8223x triangle count 不能抵消 19.2031 pp
coverage 回退。

## 第一性原理结论

Stairs 已证明 exact identity closure 与 bearing-Fisher conditioning 能在室内场景转化为
更好的 Track 漏斗；GreatCourt 则证明 P8 V1 的聚合目标仍不充分。问题不是
`cycle_verified_fisher` 完全无信息，而是它以总量为主的选择过程允许效用集中：满足普通
pair-graph connectivity，并不等于每个 mapping camera 都得到至少一个可验证的三视图身份
闭环。

因此最简洁、可证伪的方法结论是：

1. 保留“verified identity closure + dimensionless bearing-Fisher”作为有价值的诊断量，
   但拒绝 P8 V1 作为跨场景 pair policy；
2. 不通过降低 `camera_fraction_not_lower` 门槛、改变 pair budget、重加 parallax 权重或
   继续跑 Track/pose 来挽救同一 V1 假设；
3. 如果未来立项 P8 V2，必须先建立全新的 preregistration。V2 应把
   **verified-triangle camera coverage 变成 selector 内部的 lexicographic hard
   constraint**：先保证同一 probe 上 coverage 不低于 nearest control，再优化闭环/Fisher
   utility；同时重新冻结可行性、确定性 tie-break、复杂度与两场景 gate；
4. 在新合同冻结并经独立实现/合成测试前，不生成 V2 真实场景结果。当前没有获授权的 P8
   下游执行，冻结 V3 `nearest` 继续作为方法默认。
