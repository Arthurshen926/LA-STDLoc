# P8 V2 `cycle_verified_fisher_coverage` Stage-B 正式结果

## 结论

P8 V2 在正式 mapping-only Stage-B 得到科学 **Stop**。GreatCourt 的 8/8 base gates
全部通过，scene decision 为 `SCENE_PASS_REQUIRES_OTHER_SCENE`；Stairs 产物与 lineage
均合法，但唯一失败项 `broad_mapping_query_coverage_not_lower` 使 scene decision 成为
`STOP_SCENE_MECHANISM`。因此没有运行 cross-scene Stage-B aggregator，也没有继续
fullchain lineage、function graph/Map、mapping pose、formal test 或默认切换。冻结 V3
`nearest` 继续作为共享默认。

## 正式产物

两场景都在 commit `783bef63c113bb80f236b1234fb1d6ac85b29e12`、CPU、Python
3.9.12 / Torch 2.0.1 下，以同一 clean producer identity 顺序构建 control -> V2；所有
Track stderr 为空，completion 都是 `complete=true`、`partial=false`、
`uses_test_queries=false`。

| 场景 | completion SHA-256 | Stage-B gate SHA-256 | 结果 |
|---|---|---|---|
| GreatCourt | `21be068fd6b63ab5e2ff0514a75ef1f022a617efef48bbbba6192f837faf5d03` | `6963235eeb219154ce1540fd7c89cb1831d595c6149b64775d2e9de08a363091` | 8/8 Pass；exit 0 |
| Stairs | `39799b4f8c559589ab934992854ec9961eb9df9e56d21f2b7fecca2b00f1ef17` | `059eb4d1b8c6468a26ba9214228d33e937924511e9318f75dac025ac6851ad06` | valid Stop；base 7/8、V1 retention 4/5；exit 2 |

completion、两场景各四个 Track artifacts 与两个 gate 的文件哈希均已直接重算；正式
scene gates 的独立结果审计为 P0/P1/P2=0，并另从两个 Stairs factor tensors 重算了
query-level support。完整路径、12 个 SHA、gate 布尔值、运行状态与逐 query diff 见
`docs/evidence/p8_cycle_verified_fisher_coverage_v2_stage_b_result.json`。

## 指标与唯一失败 query

GreatCourt 的 V2 在保留 99.1861% triangulated Tracks 的同时，broad / high-confidence
Tracks 分别达到 control 的 117.7968% / 160.6250%，covariance p90 降至 60.9543%，
broad-query coverage 从 0.846506 升至 0.862182，因此 8/8 通过。

Stairs 的 aggregate 指标同样大多改善：triangulated / broad / high-confidence Tracks
分别为 control 的 113.5920% / 114.5769% / 121.9512%，covariance p90 降至
68.7943%。但 broad-query coverage 从严格的 1.0 降到 0.99949998。直接从两个
SHA-bound factor tensors 重算得到，唯一丢失 broad support 的是 query index **1933**，
即 `seq-06/frame-000433.color.png`：

| query 1933 | incident pairs | observations | triangulated support | broad support |
|---|---:|---:|---:|---:|
| nearest control | 6 | 459 | 211 | 184 |
| P8 V2 | 1 | 21 | 0 | 0 |

这不是“aggregate 明显变好所以可以忽略”的尾差。预注册要求每个 mapping query 的
broad support 不低于 control；唯一 query 归零已经足以触发合法 Stop。

## Stage-A 与 Stage-B 的边界

Stage-A 的 hard coverage 约束的是 nearest control 中**已完成 verified triangle 的精确
camera target set**；Stage-B 检验的是 full Track construction 后**每个 mapping query**
是否仍有 broad-eligible Track support。两者不是同一集合，也不存在逻辑蕴含。query
1933 不在 Stage-A target/final completed-triangle camera set 中，V2 对它只选择了 1 条且
completed-triangle count 为 0 的 pair，所以 Stage-A 仍可完整通过；到 Stage-B，它无法
形成 triangulated/broad Track，才暴露出 query-level sufficiency 缺口。

因此 V2 的 Stage-A 成功仍是有效的 camera-coverage 证据，但不能外推为 Track query
support 已成立。当前 P8 V2 主线在这里收敛为科学 Stop，不运行 cross-B、fullchain、pose
或 test，也不做场景后验补丁。
