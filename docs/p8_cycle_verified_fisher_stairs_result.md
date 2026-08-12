# P8 Stairs `cycle_verified_fisher` mapping-only 机制结果

## 结论

P8 在 Stairs 的预注册 Stage-A 与 Stage-B gate 均正式通过：Stage A 为 **9/9
Pass**，Stage B 为 **8/8 Pass**。最终 gate 有效、只使用 mapping query，且明确给出
`SCENE_PASS_REQUIRES_OTHER_SCENE`。因此当前结论是 **Stairs scene-specific
mechanism Go**，不是方法默认 Go，也不是定位精度 Go。

这轮结果授权的唯一下一步，是在完全冻结的协议下执行 GreatCourt 的 Stage-A/Stage-B
序列。它**尚未授权** Stairs 或 GreatCourt fullchain、function graph/Map 重建、mapping
pose、`office2_5b`、formal test 或默认策略切换。冻结 V3 `nearest` 仍是当前部署对照。

## 冻结合同与正式产物

两个 Track arm 共用 2,000 个 Stairs mapping query、K=1024、NMS=4、精确 7,450
pair budget 和同一 14,835-pair bounded probe universe。query 顺序 SHA-256 为
`5167c86bcf9665c7cafdaa9d195a1075becda209b457217ca1afe0a39fbf8e4e`；所有正式
gate 均记录 `mapping_only=true`、`uses_test_queries=false`。

预注册机器合同 `docs/evidence/p8_cycle_verified_fisher_preregistration.json` 保持冻结，
SHA-256 为
`39264e748adc735a85f141ac307aad6cba8b6a1d825880ce500ef3d86d6bc728`。其中
`current_status` 是执行前快照，不在看到结果后回写；本文件及配套 result evidence 才是
执行后的正式状态，阈值仍完全来自该冻结合同。

| 产物 | SHA-256 | content SHA-256 / 说明 |
|---|---|---|
| fresh-cache manifest | `cc7f2b7137451f6bbeed24b016fe5ef2ffff09b1322d8151b92e3410242f6539` | 两臂共享 |
| K1024/NMS4 query cache | `6f2b5a73185a98af10278d6d6fa68f1a95eac1907133dfa0678c357cb09e72c9` | mapping-only V2 equivalence |
| mapping-scope equivalence | `b600403c9d3e59f9ef68389dc7a5e321028889bea8515cae80e449c3603522ee` | `uses_test_queries=false` |
| frozen Track payload | `4e3a9c453b3602c066ca82b06694f3b27a690705cdf2ac9bd6cdce707c479a10` | 冻结阈值/lineage 输入 |
| pair proposals | `31f458410cdad5f2ab0aef846532a2ae324f08cd7b07bae47e18de5b7897ab20` | content `d0d028a4dc8736d3d74a94090f6575da5a742353760a088dcd26c6c77f56c47a` |
| bounded match probe | `6fe4a0581335e7e535effbf44611e88709ed517bc10341efc1ce9e71f42a20e6` | content `53d94010d8e8d7d5e28540aef6a6345090d0191e2ab593c48252e18c0d817a58`；3,848,929 matches |
| P8 pair selection | `7d08bed0ead859ae917724beebe22af844b87bbd7e0f9579b834f5915a13c16f` | content `721617c2e084e1c8fe75e29cec6d818a0374d7522977c241880033489f9cf93f` |
| Stage-A gate | `cf77f75be254c432dde2d64cac647ec70cef7d62730b414e77503955acc4c455` | `GO_TO_TRACK_REUSE` |
| nearest same-probe control factor / report | `e7ba71207af19db71031e59d7f2d82bc7774669eeb497bff036c5ccf97d370e1` | report `30a89a51ea03a517bdabd631efe6a7d1b78df39cdc72793d41cb0413614ce49b` |
| P8 variant factor / report | `ec569adce9d272f01aa9550f2b5558143ea740d31668121ba948706f9e5373dc` | report `8b80257bcc3ae3f06da9ca58d5f6fed42c14232f81e8cb3294239a20244ce543` |
| Stage-B gate | `72233ebd66c33e723d12d92f47b66c5fbf3638635768d2a3fd03802bce7e9dcf` | `SCENE_PASS_REQUIRES_OTHER_SCENE` |

直接重算复现了以上文件哈希；PT schema/content 字段、三个 JSON gate、两个 Track
report 和 control/variant lineage 也已独立解析交叉检查。机器可读摘要位于
`docs/evidence/p8_cycle_verified_fisher_stairs_result.json`。

## Stage A：目标确实选中了更完整、信息更强的闭环

selector 在精确相同的 7,450 pair 成本下保留 candidate universe 的两个连通分量，零
isolated camera，最小 degree 为 1。9 项预注册 gate 全部通过。

| Stage-A 指标 | nearest same-probe control | P8 | 变化 |
|---|---:|---:|---:|
| confidence-weighted Fisher utility | 64.03659 | 638.61841 | **9.9727x** |
| completed verified keypoint triangles | 84,878 | 825,469 | **9.7254x** |
| 参与 verified closure 的 mapping camera fraction | 0.2660 | 0.8665 | **+0.6005 / 3.2575x** |
| selected pair budget | 7,450 | 7,450 | 完全相同 |
| selected graph components / isolates | 2 / 0 | 2 / 0 | 保持 candidate topology |

这不是简单把更多相似帧堆进图里：P8 的自变量是同一 bounded probe 上的 exact
descriptor triangle closure 与 dimensionless bearing-Fisher utility，同时把连通性作为硬
约束。结果说明该 objective 在 Stairs 上确实找到比 nearest 更广泛且条件更强的闭环集合。

## Selector 工程执行审计

第一次正式 selector 执行在旧的 `O(verified triangles x budget)` closure 扫描中运行
1,592.086 秒后被授权中止，没有写出 selection 或 Stage-A gate。该 invalid execution
record 的 SHA-256 为
`42145cc49c39dfc0fc6e08300742b476243ab6afb4ece18c85c8f1f27aa0ca10`，其中
`valid_scientific_result=false`、`scientific_decision=null`；因此它只是工程复杂度故障，
不是科学 Stop。

commit `199c187acd8a6df018e3630fe0babda3739e68c1` 将 closure 改为精确增量索引后，同一
正式输入约 188 秒完成并写出上述 selection；其 selection/content hashes 与 Stage-A
gate 随后进入完整 lineage。运行时间只用于证明工程 blocker 已解除，不属于任何科学
gate，也不构成精度主张。

## Stage B：更好的闭环转化成更多且条件更好的 Track

两个 arm 都直接复用了 probe 中的选中行，`track_pair_matches_reused=1`、
`uses_precomputed_pair_matches=true`，没有重新进入 descriptor matcher。Stage B 八项
gate 全部通过。

| Stage-B 指标 | nearest same-probe control | P8 | 变化 / gate |
|---|---:|---:|---:|
| triangulated Tracks | 15,053 | 17,384 | **+15.4853%**；>=0.98x Pass |
| broad eligible Tracks | 14,276 | 16,634 | **+16.5172%**；>=0.98x Pass |
| high-confidence Tracks | 41 | 51 | **+24.3902%**；>=0.98x Pass |
| triangulated covariance p90 | 0.05501186 m2 | 0.03653227 m2 | **-33.5920%**；<=1.05x Pass |
| broad-support mapping-query fraction | 1.0000 | 1.0000 | 无回退 Pass |
| exact probe-row reuse | 1 | 1 | control/variant 均 Pass |
| matcher contract | 同一冻结合同 | 同一冻结合同 | Pass |

字段级 diff 进一步确认：两个 factor 的 manifest、query cache、mapping equivalence、
frozen Track、proposal、probe、selection、Stage-A gate、matcher 参数以及完整 Track science
contract 完全相同；pre-build lineage 唯一预期差异是 `pair_subset_role`，即 nearest
same-probe subset 对 P8 selected subset。density、descriptor 和 selector materialization
mutation flags 在两臂均为 false。历史 frozen Track 只提供冻结阈值与 lineage，并不是这次
same-probe control 必须逐 count 复现的对象；因而科学比较始终是本轮两个可比 arm，而不是
把缺少当前 cache/NMS lineage 的旧 factor 冒充 control。

## 第一性原理结论与下一步

P7 `parallax_diverse` 的失败说明“更大视差”不足以成为跨场景效用；P8 的 Stairs 结果则
给出更强的正向机制证据：在固定匹配成本下，**可验证身份闭环、camera coverage 与几何
信息条件必须共同成立**。Stage-A 的 set-level 信号没有在 Track 构建时消失，反而同时
转化为更多 triangulated/broad/high-confidence Tracks 和更低 covariance tail。这使 P8
比单独最大化 parallax 更接近论文需要的统一 camera-pair principle。

但 Stairs 单场景无法证明跨域充分性，更不能证明 pose 精度。下一步必须原样执行
GreatCourt proposal -> bounded probe -> selection -> Stage A -> same-probe 双 Track ->
Stage B；不得查看 GreatCourt 结果后修改 1.05/0.98/1.05 gates、pair budget、matcher 或
utility。只有 Stairs 与 GreatCourt 都通过独立机制 gate，cross-scene aggregator 才可授权
fullchain 与 q256x3 mapping-pose。若 GreatCourt 任一项失败，P8 在 pose/test 前停止，
`nearest` 继续作为共享默认。
