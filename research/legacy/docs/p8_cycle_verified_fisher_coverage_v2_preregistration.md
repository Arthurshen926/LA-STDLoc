# P8 V2 `cycle_verified_fisher_coverage` 预注册

## 决策与边界

GreatCourt 已正式推翻 P8 V1 作为共享 pair policy：V1 在相同 5,254 pair
预算下把总 verified-triangle utility 提高 2.1049 倍、triangle 数提高 1.8223
倍，却将有完整 verified triangle 的 mapping camera 从 1,524 降到 1,230。
这说明普通 pair-graph 连通并不等价于身份闭环覆盖。
集合审计进一步显示它实际丢失 295 个 control cameras、只新增 1 个；Stairs 虽从 532
增到 1,733，仍丢失 54 个 control cameras并新增 1,255 个。因此 V2 冻结的是 exact set
containment，而不是再次比较 count/fraction。

P8 V2 只修正这一已定位的充分性缺口，并冻结为新的独立策略
`cycle_verified_fisher_coverage`。V1 的 schema、policy、selector 和历史结论均不修改。
V2 仍然只允许 mapping query；当前文件只授权 CPU 实现/测试和后续按本合同生成
mapping-only selector 证据，不授权 Track、fullchain、pose、test 或默认切换。

## 数学定义

候选边集为冻结两臂并集 `E`，预算为 `B`。同一 SHA-bound probe 经 2 px 三视图
几何检验后得到 verified keypoint triangles `T`；每个 `t` 有三条边 `E(t)`、三台相机
`C(t)` 与 V1 已冻结的 dimensionless Fisher utility `u(t)`。

冻结 nearest-control 边集为 `E_n`，`|E_n|=B`。目标相机不是 candidate universe 中
所有出现过 verified triangle 的相机，而是：

```text
Q_target = union { C(t) | t in T and E(t) subset E_n }.
```

选择 `S subset E, |S|=B` 必须按字典序满足：

1. `Q_target` 中每台相机至少属于一个由 `S` 完整选中三边的 verified triangle；
2. `S` 保持 candidate graph 的 component 数、零 isolate、minimum degree >= 1；
3. 在不删除前两阶段边的前提下，继续按 V1 的
   `u(t) / missing_edge_count` closure objective 加完整 triangle bundles；
4. 剩余槽位按 V1 edge Fisher/cycle 顺序填满精确预算。

选 `Q_target` 而非更强的 candidate-universe coverage 有两个原因。第一，它精确表达
GreatCourt V1 失败的 no-regression 基线，而不在看到反例后降低门槛。第二，`E_n`
是精确预算的 coverage witness：它必然完成 `Q_target` 的 triangles。candidate graph 的
component/isolate/degree 由全 candidate spanning forest 单独保证；两种约束的联合 scaffold
仍需显式检查不超过预算。要求覆盖 candidate universe 的所有 potential cameras没有同预算
coverage witness，可能把“新 objective 是否更好”混同为“合同是否根本不可行”。

## 确定性算法与可行性证书

Stage 0 从 precomputed verified table 和 `E_n` 精确重放 `Q_target`。

Stage 1 先按 V1 的全 candidate edge utility 顺序重算 maximum-utility spanning forest，并
补 minimum-degree；这允许使用非-control edge。随后把共享同一 camera/edge triple 的
keypoint cycles 聚合，coverage witness 严格只允许 `E_n` 已完成的 camera triples。每轮先
吸收当前已完整闭合的所有 eligible triples，再在全体 eligible groups 中选择边 bundle；
tie-break 依次为新增 target 数/缺边数（整数交叉乘精确比较）、新增数、较少缺边、聚合
utility/缺边数、keypoint-cycle count、最大单 cycle utility、较小 missing-edge tuple、较早
canonical group。已覆盖 camera 不再重复获得 coverage 奖励。联合 scaffold 若超过 `B`
则 fail closed；正式两场景的预实现诊断证明本冻结算法有充足余量。Stage 2 只增不删，
coverage 和 graph invariant 因而单调保持。

同时冻结以下 hash-bound 证书：probe content、candidate pool、verified-cycle table
content、nearest reference pair table、target camera set、Stage-1 scaffold，以及最终 covered
camera set。sidecar 还显式报告 `lost_control_camera_count=0`、新增 camera IDs/count、
control witness triangle/edge 数与 Stage-1 后剩余预算。validator 从 verified table 独立
重放 sorted target IDs/hash、最终 covered hash 及 exact set-subset；不会把 camera degree
当作 triangle coverage。实现前的只读可行性诊断给出 GreatCourt scaffold
2,347/5,254、Stairs 2,457/7,450，均留下充足 Stage-2 预算；这些数字是 preregistration
feasibility diagnostics，不是 V2 scientific result。

同一只读诊断在填满预算后得到 GreatCourt utility 约 4,248.919、1,717,438 triangles、
1,525 covered cameras；Stairs 约 629.409、812,232、1,764。Stairs 分别保留成功 V1
utility/triangles 的 98.558%/98.396%，并恢复其丢失的全部 54 个 nearest cameras。这些
数值只说明冻结 gate 看起来可行，不替代未来由新 CLI 生成和 hash-bound 的正式结果。

这个确定性 greedy 不声称全局最优。带 triangle bundles、camera set cover、连通性与固定
预算的联合最优化包含 set-cover/Steiner 型组合子问题，一般是 NP-hard。V2 的声明仅是
“有构造性 witness 的硬 invariant + 确定性启发式 utility fill”。

复杂度（已拥有 verified table 时）为：triangle triple 聚合 `O(T log T)`，graph scaffold
`O(E alpha(Q))`，确定性 marginal cover 最坏 `O(|Q_target| G)`（`G` 为 eligible camera
triples），增量 closure `O((T+E) log T)`。`materialize_cycle_verified_triangle_table` 让
selector 直接复用已 hash-bound table；正式 Stage-A comparator 则有意再次从冻结输入
完整重算三视图几何，作为科学 gate 的独立真实性检查。
这里的“独立”包含两层：comparator 在同一冻结 verified table 上独立重放
control/variant subset、camera IDs、证书和 gate；同时从 hash-bound cache/probe 在内存中
完整重物化三视图几何，并要求所有科学字段和 tensors 与输入 table exact 相等。为记录
provenance，table 还绑定 clean producer Git commit、物化 entrypoint、完整
geometry/CLI/prereg source SHA 与 Python/Torch runtime。materializer、selector、comparator
都要求这些 source paths 在当前 worktree clean、hash 完全一致，且 producer commit 是当前
提交的祖先；clean 不是依赖可被 `assume-unchanged` 隐藏的 status，而是逐文件比较当前
bytes 与 `HEAD:path` blob；声明的 producer commit 也必须逐文件重读其 Git blob，并与
identity source SHA 完全相等。producer identity 不是独立真值：即使攻击者保留合法 identity、同步篡改
Fisher/utility 并重签 content hash，也会被 comparator 的完整几何重物化拒绝。

## 失败封闭与冻结 gate

输入/hash/schema/reference 不合法、reference 无 verified triangle 或 joint scaffold 超预算，
属于 fail-closed contract/infeasibility error（exit 1，不产生可误认的 selection）。对一个
有效 exact-budget selection，以下任一项先于 Track 形成科学 Stop，并持久化 V2 Stage-A
JSON（exit 2）：

- target membership 与 same-probe nearest replay 不完全一致；
- 最终未覆盖任一 target camera，或预算/图 hard constraints 失败；
- Stage-1 feasibility scaffold 超过精确预算的一半，未给 utility fill 保留至少一半预算；
- Fisher utility 低于 control 的 1.05 倍；
- completed verified keypoint triangles 低于 control 的 0.98 倍。

coverage gate 是 set-membership hard invariant，不再使用事后 fraction threshold。两场景
均沿用原冻结 K/NMS/budget/candidate/matcher/2px：Stairs
`1024/4/7450/14835`，GreatCourt `2048/4/5254/9875`。先在 GreatCourt 验证 V1 的已知
failure 是否被构造性消除，再在 Stairs 做 no-regression；两场景 Stage-A 均通过前，不做
任何 V2 Track。此顺序不是调参：算法、tie-break 和 gate 已在任何 V2 真实 selector 执行前
冻结。

“同一 probe”也不是调用方运行时自选：合同逐场景冻结 query count/name hash、query-cache
SHA、mapping-scope equivalence SHA、proposal file/content SHA 与 probe file/content SHA。
三个 V2 CLI 都将实际加载产物与这些编译合同逐字段比较；同尺寸的替代 cache/probe 也会
在产物生成前失败。精确 hashes 见机器预注册的 `fixed_scene_contracts`。

Stairs 还必须防止 V2 用 coverage 修复破坏已经成功的 V1 信号：同一 frozen V1 selection
（file SHA `7d08bed0ead859ae917724beebe22af844b87bbd7e0f9579b834f5915a13c16f`，
content SHA `721617c2e084e1c8fe75e29cec6d818a0374d7522977c241880033489f9cf93f`）
在同一 verified table 上重放，V2 的 utility 与 completed keypoint triangles 均须至少保留
其 98%，covered-camera count 不低于 V1，并仍满足 exact nearest target subset。

若将来两场景 Stage-A 均通过，Stage-B 仍使用 V1 已冻结 Track thresholds，并需要新增
V2-aware reuse runner/gate lineage 后才能执行。Stairs 除相对 nearest control 外，还必须
相对成功 V1 Track factor（SHA
`ec569adce9d272f01aa9550f2b5558143ea740d31668121ba948706f9e5373dc`）保留至少 98%
triangulated/broad/high-confidence Track，covariance p90 不超过 V1 的 1.05 倍，broad-query
coverage 不低于 V1。现有 V1 Stage-B runner 不得把 V2 schema 冒充 V1 selection。
单场景 Stage-A pass 永远只输出
`SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE`、`advance=false`。唯一 Track 权限来自
`aggregate_cycle_verified_fisher_coverage_stage_a`：它递归重哈希并加载两场景的
cache/proposal/probe/table/selection（及 Stairs V1 guard），再次 rematerialize verified
geometry、重放 Stage-1/2 selection、subset metrics 与所有 gates。只有两域均 exact pass
才输出 `GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD`，并明确不授权既有 V1 runner；任一有效
科学 failure 持久化 cross-scene Stop，伪造 all-true/self-signed gate 形成 input error。

## 实现与 CPU 验证范围

- `evidence/cycle_verified_fisher.py`：独立 V2 policy/schema、verified-table
  materializer/validator、coverage selector/validator；
- `scripts/materialize_cycle_verified_triangle_table.py`：一次几何表物化；
- `scripts/select_cycle_verified_fisher_coverage_pairs.py`：V2 selector；
- `scripts/compare_cycle_verified_fisher_coverage_stage_a.py`：独立重放和科学 Stop；
- `scripts/aggregate_cycle_verified_fisher_coverage_stage_a.py`：两域递归授权 gate；
- tests：随机小图 exhaustive feasibility oracle、hard coverage/graph/exact-budget invariant、
  hash tamper、verified-table reuse、CLI scientific Stop persistence；现有 V1 tests 保持不变。
  另覆盖 tampered geometry/Fisher + resigned content 且保留合法 producer identity 的拒绝路径。

提交前 CPU suite 为 `405 passed, 1 skipped`，修改范围的 Ruff 与 `git diff --check` 通过；
跳过项仍是需要显式环境开关的既有 CUDA renderer smoke，不是 V2 测试。

当前仍未运行任何真实 V2 selector/GPU/Track/pose/test，V3 `nearest` 继续作为共享默认。
