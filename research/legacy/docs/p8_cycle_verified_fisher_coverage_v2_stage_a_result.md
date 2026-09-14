# P8 V2 `cycle_verified_fisher_coverage` 双场景 Stage-A 正式结果

## 结论

P8 V2 的 Stairs 与 GreatCourt mapping-only Stage-A 均正式通过。两场景都在冻结的
same-probe、精确 pair budget 下完整保留 nearest control 的 completed verified-triangle
camera 集合，`lost_control_camera_count=0`；同时 graph、utility、triangle 及 Stairs V1
no-regression guards 全部通过。双场景递归 gate 的正式 decision 是
`GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD`。

这个 Go 的边界很窄：**只授权实现并验证一个独立 schema/lineage 的 V2-aware
reuse-only Track runner**。它不授权复用 V1 runner，不授权现在运行 V2 Track/Stage-B，
也不授权 fullchain、function graph/Map、mapping pose、formal test 或默认切换。冻结 V3
`nearest` 仍是共享默认。Stage-A 证明的是 mapping evidence sufficiency 已跨室内/室外
两域成立，还不是最终定位精度结论。

## 正式产物与独立真实性检查

所有表、selection 和 gate 均由 clean producer commit
`bc0e1d362dd8441d68193ddfb488f0699f6d5552` 生成；producer identity 绑定 10 个
geometry/CLI/prereg source blobs。正式 comparator 从 SHA-bound cache/probe 完整重物化
三视图几何，并要求 materialized table 的科学字段与 tensors exact 相等。两场景 gate 都记录
`verified_geometry_independently_rematerialized_exact=true`、`mapping_only=true`、
`uses_test_queries=false`，所有正式进程 exit 0 且 stderr 为空。

| 场景 / 产物 | 文件 SHA-256 | content SHA-256 / decision |
|---|---|---|
| Stairs verified table | `18384745d9005c87bc58a84388a3305cf15366613f25d354cf484317b66cebe3` | `42ea0a6e12cb1ba8d32d8b21c5ada8f4b5d2a406b40eece4dac5ffcc219d844b`；1,209,504 triangles |
| Stairs V2 selection | `6a1351ab1e0d246e8d015071d6caa0388f218de4cef5a2658c8b61f6fd7234d4` | `19ac2f2bc998550285fea4c879f6cfe425b67c6397701ac79e212cd2dd5096c6` |
| Stairs Stage-A gate | `7a0722c1ee5750bef9efc9b7b4cb4cf5e54b101772010838d9ec5b5279f65eec` | 14/14；`SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE` |
| GreatCourt verified table | `22d4b2e646c59cace6d12e33cd58349fc8fcfc760d12c7bc0d8b149de1606693` | `babb827814dadf57ec50c1002504e808b9fb1282cc1424806e0120f0b91a20c3`；3,136,243 triangles |
| GreatCourt V2 selection | `a901f7288a200e02e2403aa0c807244303ef0bf4f6de9e2231c78b8de4ab7e62` | `769dba8cfa01b5ffaeabdd1a7d09184a8e16d41bc211441bf0630ae4c7243f7b` |
| GreatCourt Stage-A gate | `be811e3505018bbf0838156f7852ac46f683871ebe688682cba57309d7a7cf17` | 11/11；`SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE` |
| cross-scene Stage-A gate | `b9aecc359af0f66272602901d60777ebcca2b6800769a6c262d4cb0a121c74da` | `GO_TO_V2_AWARE_REUSE_ONLY_TRACK_BUILD`；V1 runner=false |

文件哈希已直接重算；两个 PT table 和两个 PT selection 也在
Python 3.9.12 / Torch 2.0.1 环境中实际加载，并由项目 validator 重新计算 content hash、
检查 schema/tensor/certificate/producer identity。逐字段、sidecar、exit status 与 gate
交叉检查的机器摘要见
`docs/evidence/p8_cycle_verified_fisher_coverage_v2_stage_a_result.json`。

## 硬覆盖集合与 Stage-1 预算

V2 的 target 不是“相机 count 不下降”，而是同一 probe 的 nearest control 已完整闭合
verified triangle 所涉及的**精确相机集合**。Stage 1 只能用 control-completed triangle
witness 覆盖该集合，并在 Stage 2 Fisher fill 前锁定集合 hash；最终必须逐 ID 保持
`target subset selected_covered`。

| 证书 | Stairs | GreatCourt |
|---|---:|---:|
| exact pair budget | 7,450 | 5,254 |
| target camera count | 532 | 1,524 |
| target camera-set SHA-256 | `871af10a...39801d5` | `b687d260...e78377f9` |
| Stage-1 scaffold pairs | 2,457 | 2,347 |
| Stage-1 / budget | 32.98% | 44.67% |
| Stage-2 remaining budget | 4,993 | 2,907 |
| Stage-1 witness-covered target cameras | 532 | 1,524 |
| final covered cameras | 1,764 | 1,525 |
| lost control cameras | **0** | **0** |
| added cameras | 1,232 | 1（camera 416） |

两个 scaffold 都不超过预算的一半，给 utility fill 留出预注册要求的空间。最终 selected
graphs 分别保持 `2/0/1` 与 `1/0/1` 的 components / isolates / minimum degree，且精确
填满预算。

## Stage-A 科学指标

| 场景 / 指标 | nearest same-probe control | P8 V2 | 变化 |
|---|---:|---:|---:|
| Stairs confidence-weighted Fisher utility | 64.036590 | 629.409066 | **9.8289x** |
| Stairs completed verified triangles | 84,878 | 812,232 | **9.5694x** |
| Stairs completed-triangle cameras | 532 | 1,764 | +1,232；lost=0 |
| GreatCourt confidence-weighted Fisher utility | 2,132.815233 | 4,248.919363 | **1.9922x** |
| GreatCourt completed verified triangles | 1,028,346 | 1,717,438 | **1.6701x** |
| GreatCourt completed-triangle cameras | 1,524 | 1,525 | +1；lost=0 |

GreatCourt 因而修复了 V1 已定位的充分性缺口：V1 虽把 aggregate utility/triangles 提高到
4,489.333/1,874,006，却只覆盖 1,230 台相机；V2 以少量 aggregate objective 让步，恢复
全部 1,524 台 control cameras，并额外覆盖 camera 416。该 V1 对比用于解释机制，不是
V2 的新事后 gate。

Stairs 的额外预注册 V1 retention guards 也全部通过：V2 保留 V1 utility 的
98.5579%（>=98%）、triangles 的 98.3964%（>=98%），covered cameras 从 1,733 增到
1,764。也就是说，coverage-hard 修复没有以破坏已成功的室内 V1 信号为代价。

## 执行调度说明与下一步边界

GreatCourt table 物化期间为避免两项 CPU 作业在 32 核机器上过度争用，进程曾被
`SIGSTOP/SIGCONT` 两次：07:01:45--07:04:18（153 s）与
07:06:02--07:16:09（607 s），合计 760 s。暂停保留进程内存状态，没有改动输入、
producer source、算法、gate 或产物；runtime 也不是科学 gate。因此这只是一条调度记录，
不构成科学处理或结果排除理由。

下一步只能依序进行：实现独立 V2-aware reuse-only Track runner 与 Stage-B lineage，先做
CPU/lineage 审核，再由新的授权合同决定是否运行两场景 mapping-only Track。现有 V1 runner
不能把 V2 selection 冒充 V1 授权。在正式两场景 Stage-B 通过之前，fullchain、pose、test
和默认切换继续全部冻结。
