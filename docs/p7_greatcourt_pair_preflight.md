# P7 GreatCourt pair-policy preflight

## 结论

GreatCourt 是 Stairs pair-policy 通过 pose gate 后最有辨识力的外域验证，
但**当前仍为 BLOCK，不能直接启动 factor**。阻塞点不是 GPU、磁盘或缺少 V3
输入，而是旧 49.69 GB query cache 只登记了 `K_mapping=2048`，没有登记 NMS；
直接用它会把“固定 NMS=4”写成未经证明的事实。

代码层的 K=1024 硬编码已经在独立提交 `be5961d` 中泛化：runner、splat
provenance replay 和 payload lineage audit 都要求调用者显式传入 mapping K 与
exact pair budget，并分别与 bootstrap manifest、cache signature、frozen Track
payload 和 pair sidecar 交叉校验。合成的 `K=2048 / budget=5254` 契约已覆盖；
这只是消除实现阻塞，不授权科学运行。

正式次序必须是：fresh `K2048/NMS4` cache -> 与旧 cache 的逐 query Track-input
exact parity -> nearest control 的完整 Track/geometry/assignment parity ->
parallax-diverse mechanism gate -> fresh compact full chain -> mapping pose gate。
任何一环失败都停止 GreatCourt，不通过调 selector、阈值或 pair-policy 参数救场。

机器可读路径、SHA 和阈值见
[`p7_greatcourt_pair_preflight.json`](./p7_greatcourt_pair_preflight.json)。本预检只读，
没有创建下面的输出目录，也没有加载旧 50 GB pickle 或启动 GPU。

## Frozen V3 lineage

V3 root 为
`/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/Cambridge/GreatCourt`。

| 输入 | 冻结路径 | SHA-256 |
|---|---|---|
| Stage-A state | `bootstrap/stage_a/6630_lafgs_map_state.pt` | `4f133a8099dc989542fdc058c4c1582b43658a28fc718b818ccea64916a3e2c3` |
| query cache | `bootstrap/query_cache.pt` | `13a4b23daae0194e92544746abf790eb72cd739890ad681df34878041e22e53a` |
| visibility | `bootstrap/visibility.pt` | `9594e814f9ffa53708b27ae1ae3a18fc55ccd1775001de81e03f4f872ece44c1` |
| 2DGS PLY | `/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731/GreatCourt/prior/rgb_matcha_2dgs/point_cloud/iteration_30000/point_cloud.ply` | `91b2b6b47a2d8c5b8618fae541a5d2b5bde6b90ace381a31e44a56e1cb0362d2` |
| parent calibration | `bootstrap/scene_calibration.json` | `b939fe0de7bc69245e025bcc41867986c477f11f8de984c7d4ffba949f04f38f` |
| frozen Track payload | `bootstrap/tracks_refined/track_micro_anchor_payload.pt` | `f0fec708e00ab0f19160485059574eccdb8f6aba0484a257627f9b56a4cc90af` |
| bootstrap manifest | `bootstrap/tracks_refined/reproducibility_manifest.json` | `97ed0a0848a4c8bd219a82f98c2ee36aebd2258e9f3ea0f214a41a3e4ae8b846` |
| V3 trained map | `map_learning/anchor_map_step_1164.pt` | `7b1d24316494a39ea259792b0a7ab8f5e4489c2144d7583aec3e2c3d45589b08` |
| V3 metric | `map_learning/metric_state_step_1164.pt` | `f45ef61b5e1a798b346775ed6c307491aad431a9e883b2af3996629d5ea746b1` |
| V3 compact teacher | `map_learning/complete_positive_teacher.pt` | `19bcad509e6d3139040a2dd3f874642e337768d9ad8eace862eb3cd55e21f6cd` |
| mainline config | `/root/STDLoc/configs/paper_mainline.yaml` | `c522d3a3d692a5e3c4c6db06083ec5ca9682c9f1c2bef49b6bb135b622b352cc` |

query-cache SHA 来自其冻结 evidence contract；本轮刻意没有重新顺序读取 49.69 GB
文件。其路径、大小和 mtime 与轻量 calibration sidecar 一致。其余表中 SHA 本轮均
直接校验。V3 calibration 在 bootstrap/evidence/map_learning 三处字节完全相同。

旧机制事实为：1531 个 mapping cameras，requested K=2048，valid keypoints
median/p10=2018/2004；nearest-6 产生 5254 个 candidate pairs，其中 5178 个有匹配。
Track 漏斗为 206755 total、34150 triangulated、1933 high-confidence，triangulated
parallax median=2.2704 deg、低于 1 deg 为 15.94%，covariance median/p90=
0.8807/28.1497 m2。它说明此场景适合检验 pair geometry，而不是再次增加 K。

## Fresh roots and capacity

下面两个路径在预检时均不存在，正式运行必须先做 empty-root preflight：

```text
/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/greatcourt/k2048_nms4
/mnt/pool/sqy/lafgs_p7_pair_policy_fullchain_20260812/greatcourt
```

fresh cache 本身约 49.69 GB。旧 GreatCourt root 扣除 query cache 后约 5.03 GB；
按 factor、control replay、variant replay、canonical/compact refresh 和审计 sidecar
计，建议预留 60--70 GB 新空间。pool 当前约有 1009 GB 可用，容量不是 blocker，
但 pool 已使用 98%，失败 root 必须整体隔离，不能在原目录删若干文件后续跑。

## Strict single-factor command draft

以下仅为 preregistered draft，必须等 Stairs pose Go 后执行。所有数值沿用 Stairs
已冻结 policy，不允许根据 GreatCourt 结果调参。

```bash
export PYTHONPATH=/root/STDLoc
export PATH=/root/miniconda3/envs/g4splat/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/g4splat/lib:${LD_LIBRARY_PATH:-}

V3=/mnt/pool/sqy/lafgs_adaptive_v3_full_benchmark_20260807/Cambridge/GreatCourt
DATA=/mnt/pool/sqy/Cambridge_stdloc/GreatCourt
FACTOR=/mnt/pool/sqy/lafgs_p7_pair_policy_factor_20260812/greatcourt/k2048_nms4
FULL=/mnt/pool/sqy/lafgs_p7_pair_policy_fullchain_20260812/greatcourt
OLD_CACHE=$V3/bootstrap/query_cache.pt
FRESH_CACHE=$FACTOR/cache/query_cache.pt
TRACK=$V3/bootstrap/tracks_refined/track_micro_anchor_payload.pt
BOOT=$V3/bootstrap/tracks_refined/reproducibility_manifest.json
BASE=$V3/bootstrap/stage_a/6630_lafgs_map_state.pt
```

第一步必须真正重跑 detector，不能把旧 cache 政名伪装成 NMS-attested cache：

```bash
CUDA_VISIBLE_DEVICES=<reserved> python -m scripts.refresh_mapping_sparse_cache \
  --dataset "$DATA" --source-cache "$OLD_CACHE" --output "$FRESH_CACHE" \
  --mapping-keypoints 2048 --nms-radius 4 --device cuda

python -m scripts.audit_mapping_sparse_refresh_equivalence \
  --source-cache "$OLD_CACHE" --refreshed-cache "$FRESH_CACHE" \
  --source-track-payload "$TRACK" \
  --output "$FACTOR/contracts/sparse_refresh_equivalence.json"
```

只有审计同时满足 1531/1531 Track inputs exact、1531/1531 alpha exact、
1531/1531 metadata 为 K2048/NMS4 且
`content_equivalent_track_payload_reuse_authorized=true` 才继续。记录 fresh cache 的
SHA-256 为 `FRESH_SHA`，随后两个 factor arm 必须使用同一 SHA。

先跑 nearest control；`--expected-*` 缺失或与 manifest/cache/payload 不一致会硬失败：

```bash
CUDA_VISIBLE_DEVICES=<reserved> python -m scripts.run_track_pair_factor \
  --manifest "$BOOT" --query-cache "$FRESH_CACHE" \
  --frozen-track-payload "$TRACK" --output-dir "$FACTOR/nearest" \
  --pair-policy nearest --expected-mapping-keypoints 2048 \
  --expected-pair-budget 5254 --minimum-overlap-jaccard 0.15 \
  --minimum-joint-visibility-points 8 --parallax-saturation-deg 2.0 \
  --diversity-weight 0.20 --candidate-pool-per-camera 48 \
  --scene-points-per-camera 8 --maximum-scene-points 4096 \
  --scene-point-voxel-size-m 0.02 --device cuda

CUDA_VISIBLE_DEVICES=<reserved> python -m scripts.replay_track_provenance_assignment \
  --factor "$FACTOR/nearest/nearest_track_factor.pt" --base-state "$BASE" \
  --query-cache "$FRESH_CACHE" --expected-query-cache-sha256 "$FRESH_SHA" \
  --expected-mapping-keypoints 2048 --expected-pair-budget 5254 \
  --frozen-bootstrap-manifest "$BOOT" \
  --expected-frozen-bootstrap-manifest-sha256 \
  97ed0a0848a4c8bd219a82f98c2ee36aebd2258e9f3ea0f214a41a3e4ae8b846 \
  --output "$FACTOR/nearest/nearest_track_micro_anchor_payload.pt"

python -m scripts.audit_track_payload_parity \
  --reference "$TRACK" \
  --replay "$FACTOR/nearest/nearest_track_micro_anchor_payload.pt" \
  --float-atol 0 \
  --output "$FACTOR/contracts/nearest_control_parity.json"
```

`nearest_reproduces_all_frozen_counts=true` 还不够；最后一个 parity 必须对 Tracks、
全部 frozen-common geometry、六个 assignment 字段、query registry 和 provenance
diagnostics 全部 exact。否则 fresh cache 改变了不止 NMS attestation，GreatCourt
实验停止。

通过后才以完全相同输入和固定参数运行 variant：

```bash
CUDA_VISIBLE_DEVICES=<reserved> python -m scripts.run_track_pair_factor \
  --manifest "$BOOT" --query-cache "$FRESH_CACHE" \
  --frozen-track-payload "$TRACK" --output-dir "$FACTOR/parallax_diverse" \
  --pair-policy parallax_diverse --expected-mapping-keypoints 2048 \
  --expected-pair-budget 5254 --minimum-overlap-jaccard 0.15 \
  --minimum-joint-visibility-points 8 --parallax-saturation-deg 2.0 \
  --diversity-weight 0.20 --candidate-pool-per-camera 48 \
  --scene-points-per-camera 8 --maximum-scene-points 4096 \
  --scene-point-voxel-size-m 0.02 --device cuda

python -m scripts.compare_track_pair_factor \
  --control "$FACTOR/nearest/nearest_track_factor.json" \
  --variant "$FACTOR/parallax_diverse/parallax_diverse_track_factor.json" \
  --output "$FACTOR/contracts/mechanism_gate.json"
```

若 overlap-constrained pool 不能在默认 pool=48 下填满 5254，视为固定 policy
不适用于 GreatCourt，直接 Stop；不允许扩大 pool 或降低 overlap 后继续声称同一因素。
mechanism 通过后，再按 Stairs full-chain runbook 重建 canonical map/graph/provenance/
teacher、compact map/graph/provenance/teacher，并从零训练 frozen `metric_steps=1164`
的 metric。不能复用 V3 canonical/compact 科学产物。

## Preregistered gates

### Mechanism

- K=2048、NMS=4、1531 query order、fresh-cache SHA、global pair budget=5254 全部 exact；
- mapping-point low-parallax fraction 至少下降 10 percentage points；
- triangulated Tracks 至少保留 control 的 95%；
- broad eligible Tracks 至少保留 98%；
- triangulated covariance p90 不得恶化超过 5%；
- broad support/query p10 至少保留 95%。

这是现有 `compare_track_pair_factor` 的同一 gate；GreatCourt 不另设更容易的阈值。

### Mapping pose first

先用 V3 map/metric/compact teacher 与 fresh cache 运行 uniform mapping q256 × seeds
2026/2027/2028，再用 variant full chain 对相同 query-index hash 重放。每个 seed 的
catastrophe count 不得增加；三种子均值上 raw GT precision 允许的最大回退为
0.005 pp，recall_5cm_5deg 为 0.1 pp，translation median/mean/p90/CVaR95 的最大回退为
`max(0.02 cm, 1%)`。此外必须至少有一项实质改善：mean/median TE 下降 0.03 cm，
p90/CVaR95 下降 0.05 cm，recall 上升 0.2 pp，或 raw precision 上升 0.01 pp。
中心指标不能抵消 precision 或 catastrophe 回退。mapping gate 不通过就不碰 test。

### Formal test only after freeze

GreatCourt frozen formal baseline 的三种子 average p90 TE 为 46.4954 cm，
catastrophe total 为 91，raw precision 为 7.32563%，recall_5cm_5deg average 为
23.5088%。正式 760-query × 3 sentinel 必须满足同一 no-regression 条件，并额外要求
average p90 严格低于 46.4954 cm 或 catastrophe total 严格低于 91。test 结果只能作
最终 sentinel，不能用于选择 pair 参数、K、NMS、selector 或 metric。

## Current risk assessment

- **Hard blocker:** legacy GreatCourt NMS 未登记；fresh K2048/NMS4 exact parity 尚未做。
- **Implementation risk:** 1531-camera outdoor proposal pool 可能无法在 overlap>=0.15
  下填满 exact 5254 budget；这是可证伪边界，不是调参邀请。
- **Compute risk:** 2048-keypoint descriptor pair matrices 相比 Stairs K1024 单 pair
  约为四倍，factor 预计明显更慢；50 GB cache 的 NFS load/refresh 也会主导 wall time。
- **Scientific risk:** GreatCourt 的远景深度几何高度病态；即使 parallax/covariance
  mechanism 改善，也可能不转化为 descriptor precision 或 pose。若 Stairs pose 已经
  Stop，则本外域实验没有继续运行的因果价值。

