# V20 稀疏 Anchor Descriptor 自定位闭环运行手册

StMarysChurch 的首次 Gaussian-domain 实测与未部署结论见
[V20 StMarysChurch analysis](v20_sparse_anchor_stmaryschurch_analysis.md)。

## 目标与边界

V20 只修改 Full V2 M0 中少量 Anchor 的 map-side descriptor；query descriptor、Anchor ID、XYZ、Track 和几何全部冻结。训练目标是让高置信真 Anchor 在当前 Top-K competition 中前移，同时用已有正确匹配约束回退。最终收益只由 paired exact global Top-1 + standard PoseLib 的 control 和 fresh confirmation 决定，训练 margin 不能替代定位指标。

配置文件 [`configs/v20_sparse_anchor_closed_loop.yaml`](../configs/v20_sparse_anchor_closed_loop.yaml) **不会被任何 CLI 读取**，只是 protocol defaults。下面的命令显式传入真实参数，不存在 `--config` 接口。

正式数据流为：

```text
Observer pool --pose-family split--> design --------> Top-K evidence --> sparse proposal
                                  \-> held-out control ----------------> exact replay --> control gate
fresh disjoint Observer pool -----------------------> exact replay --> confirmation gate
stable artifacts + frozen candidate + both decisions ----------------> finalizer
```

严格禁止：test/LOO 输入、confirmation 参与训练或选臂、把 feedback descriptor 复制进地图、用 `row_valid` 删除 plant replay 中的伪影/未知检测点。`row_valid` 只决定哪些行可监督 repair；exact localization 始终使用全部 detected rows。

## 0. 准备路径

以下路径都是示例，运行前替换为本机的绝对路径。所有输出脚本都拒绝覆盖已有文件或目录；重复实验应换一个新的 `V20_RUN`。

```bash
cd /root/STDLoc

# 先激活已经安装 torch、PoseLib 和项目依赖的运行环境。
# 使用 `python -m scripts.<name>` 可让仓库根目录稳定进入 import path。

export V20_RUN=/mnt/pool/sqy/lafgs_v20_sparse_anchor_20260831/StMarysChurch
export V20_MAP=/absolute/path/to/full_v2_m0_anchor_map.pt
export V20_STABLE_METRIC=/absolute/path/to/current_stable_identity_metric.pt
export V20_FRESH_CONFIRMATION_CERTIFIED=/absolute/path/to/fresh_confirmation_certified/manifest.json
export V20_FRESH_CONFIRMATION_OBSERVER=/absolute/path/to/fresh_confirmation_observer/manifest.json

OBSERVER_MANIFESTS=(
  /absolute/path/to/observer_shard0/manifest.json
  /absolute/path/to/observer_shard1/manifest.json
  /absolute/path/to/observer_shard2/manifest.json
)

TEACHER_SHARDS=(
  /absolute/path/to/v19_teacher_shard0.pt
  /absolute/path/to/v19_teacher_shard1.pt
  /absolute/path/to/v19_teacher_shard2.pt
)

mkdir -p "$V20_RUN"
```

输入前提：

- `V20_MAP` 是稳定 Full V2 M0；V19 teacher、V14/V9 Observer 和后续 action 必须绑定同一个 map SHA256。
- `TEACHER_SHARDS` 必须是完整的 V19 `feedback_query` shard registry，且不含 test/LOO。
- `OBSERVER_MANIFESTS` 是同一 frozen feedback pool 的完整 no-LOO Observer shards，row policy 为 `v2_row_valid_only`。
- fresh confirmation 来自从未参与 design、teacher calibration、阈值选择或 control 的新 pose families。
- query index 只在各自 certified batch 内有意义；跨 batch 防重复使用原始 certified record SHA256 和 pose family，而不是要求局部 query index 数值不同。

V20 要求 Observer artifact `version: 2`：repair 行必须记录“确实进入 alternative PoseLib”的 row mask，clean protection 必须带显式 query row。旧 V9/V14 v1 artifact 会按设计被拒绝，需要用升级后的 `run_v9_causal_observer.py` 重新生成 Observer，再重新执行 family split；只要 map/certified-batch/validation lineage 一致，V19 teacher shard 可以复用。

## 1. 封存 design/control/confirmation split

先按 pose family 将 Observer pool 固定拆为 design 60% 和 held-out control 40%。只有 design 可以进入证据物化和训练。

```bash
python -m scripts.split_v14_feedback_families \
  --observer-batches "${OBSERVER_MANIFESTS[@]}" \
  --design-fraction 0.60 \
  --seed 1420260828 \
  --output-dir "$V20_RUN/split"
```

`split/control.json` 仍指向 Observer records，不能直接用于 exact replay。将 control 和独立 confirmation 暴露为指向原始 certified render records 的只读 view：

```bash
python -m scripts.run_v9_causal_observer \
  --certified-batch "$V20_FRESH_CONFIRMATION_CERTIFIED" \
  --expected-view-role confirmation_query \
  --map "$V20_MAP" \
  --device cuda \
  --output-dir "$(dirname "$V20_FRESH_CONFIRMATION_OBSERVER")"

python -m scripts.materialize_v14_certified_view \
  --observer-batch "$V20_RUN/split/control.json" \
  --view-role feedback_query \
  --output "$V20_RUN/control_certified_view.json"

python -m scripts.materialize_v14_certified_view \
  --observer-batch "$V20_FRESH_CONFIRMATION_OBSERVER" \
  --view-role confirmation_query \
  --output "$V20_RUN/confirmation_certified_view.json"
```

下面的独立审计用于在训练前尽早发现 split 错误；finalizer 还会再次硬校验三套 batch SHA 和 pose families 两两不重叠，因此不能靠跳过此段绕过：

```bash
python - \
  "$V20_MAP" \
  "$V20_RUN/split/design.json" \
  "$V20_RUN/split/control.json" \
  "$V20_FRESH_CONFIRMATION_OBSERVER" <<'PY'
import json
import sys
from pathlib import Path

import torch

from common.hashing import sha256_file

map_path, *batch_names = map(Path, sys.argv[1:])
map_sha = sha256_file(map_path)
source_sets = []
family_sets = []
for batch_name in batch_names:
    batch = json.loads(batch_name.read_text())
    assert batch["schema"] == "lafgs_v9_no_loo_causal_feedback_batch"
    assert batch["uses_test_queries"] is False
    assert batch["loo_used"] is False
    assert batch["accepted_query_row_policy"] == "v2_row_valid_only"
    assert batch["input"]["map_sha256"] == map_sha
    queries = set()
    sources = set()
    families = set()
    for item in batch["records"]:
        record_path = Path(item["path"])
        assert sha256_file(record_path) == item["sha256"]
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        queries.add(int(record["query_index"]))
        source_path = Path(record["source_record"])
        assert sha256_file(source_path) == record["source_record_sha256"]
        sources.add(record["source_record_sha256"])
        families.add(int(record["pose_family_id"]))
    assert len(queries) == len(batch["records"])
    assert len(sources) == len(batch["records"])
    source_sets.append(sources)
    family_sets.append(families)

for left in range(3):
    for right in range(left + 1, 3):
        assert source_sets[left].isdisjoint(source_sets[right])
        assert family_sets[left].isdisjoint(family_sets[right])
print("PASS: source records and pose families are pairwise disjoint")
PY
```

任一断言失败都应停止并重新封存 split，不能通过改 role 名称绕过。

## 2. 物化 leakage-safe Top-K competition evidence

正式强更新只使用 `tier_b`。Tier C 可以做诊断，但即使其 soft authorization 为真，也不会获得 `strong_feedback_authorized`。

```bash
python -m scripts.materialize_v20_topk_feedback \
  --teacher-shards "${TEACHER_SHARDS[@]}" \
  --observer-manifests "${OBSERVER_MANIFESTS[@]}" \
  --design-batch "$V20_RUN/split/design.json" \
  --anchor-map "$V20_MAP" \
  --teacher-tier tier_b \
  --minimum-wrong-winner-pose-families 2 \
  --minimum-negative-action-clean-pose-families 2 \
  --maximum-repair-rows-per-query 256 \
  --maximum-protection-rows-per-query 256 \
  --output "$V20_RUN/topk_evidence.pt"
```

证据策略：所有 query 行继续留在 plant；只有 V2-valid 且 V19 truth 为 decisive unique/equivalent 的 design rows 可做 repair。多正样本按 equivalence class 合并，训练时对每个 positive 的 listwise loss 取平均，并用 worst-positive margin 阻止只抬高最强 positive。错误 winner 只有在至少两个 pose families 重复出现且 query 的实际 task gain 为正时，才可进入 repeated-negative 模式。clean correct-winner rows 只用于保护，不能创造 repair 方向。

正式训练前检查 Tier B 强授权：

```bash
python - "$V20_RUN/topk_evidence.pt" <<'PY'
import sys
import torch

evidence = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert evidence["teacher_tier"] == "tier_b"
if not evidence["strong_feedback_authorized"]:
    raise SystemExit("STOP: Tier B teacher is not authorized; evidence is analysis-only")
print("PASS: Tier B strong feedback is authorized")
PY
```

若失败，正式闭环的正确结果是保留 stable map；不能降级成 Tier C 后继续声明可部署训练。若只想测量 teacher-unauthorized proposal 的 exact task response，可以走第 4.1 节的 `--allow-analysis-only` 分支，但它会消耗 held-out control，并永久失去本 protocol 的 confirmation/deploy 资格。

关于伪影/无效区域：这些检测点在 exact plant replay 中一律保留，因为它们确实会参与真实 PnP 成败；但不能仅凭“它匹配到了某个 Anchor”就把该 Anchor 当负样本。当前 matcher 强制每行 Top-1、没有 no-match/abstain 类，而且 render invalid 也不等于几何上必然无对应。盲目推远会把未知行误标成负例并伤害合法对应。因此 V20 只把显式认证的 nuisance 分数留作诊断；若以后加入可校准的 rejector，应使用独立 full-row artifact/OOD 证据和新的 control protocol，不能混入本轮 descriptor repair。

## 3. 训练稀疏 map-side descriptor proposal

正式主线只运行最保守的 `positive_only`。`positive_and_repeated_negative` 现在会要求每个被推远 Anchor 至少有两个 pose families 的自身 clean-positive 支持，并把这些行纳入硬保护；但它仍缺少逐 Anchor/group 的 exact PoseLib 反事实授权，因此只允许作为 analysis-only ablation，finalizer 会拒绝部署。即使做 ablation，也不能在看过 control 或 confirmation 后切换。

```bash
python -m scripts.train_v20_sparse_anchor_descriptors \
  --baseline-map "$V20_MAP" \
  --evidence "$V20_RUN/topk_evidence.pt" \
  --mode positive_only \
  --maximum-angle-deg 5.0 \
  --steps 400 \
  --batch-size 512 \
  --learning-rate 0.05 \
  --temperature 0.05 \
  --clean-margin-slack 0.002 \
  --clean-protection-weight 4.0 \
  --angular-regularization-weight 0.01 \
  --minimum-repair-margin-gain 0.001 \
  --minimum-coordinate-ranking-gain 0.00001 \
  --maximum-selected-anchor-count 4096 \
  --seed 20260831 \
  --device cuda \
  --output-dir "$V20_RUN/action_positive_only_angle5"
```

输出包括 `candidate_anchor_map.pt`、candidate-bound `identity_metric.pt` 和 `report.json`。未选 Anchor descriptor 必须 bit-exact 保持；被选 Anchor 受 5° spherical-cap、角度正则和 hard clean-margin 约束。系统先从小到大选择达到最小 repair gain 的全局 seed，再逐 Anchor 扩张；每个 coordinate step 必须保持全部 clean 行的逐行约束，且不能丢掉任何已经建立的 certified-positive win；对仍为错误 Top-1 的 repair 行，best/worst margin 使用聚合非退化约束，允许有界权衡，并要求加权 per-positive ranking loss 至少下降 `minimum_coordinate_ranking_gain`。保存为 baseline 的真实 dtype 后还会重新审计完整 clean bank；exact evaluator 会从绑定 evidence 再算一次，不能只信 report 标志。这样可以避免单个保护瓶颈把所有局部动作一起压到接近零。这里的 clean bank 是设计集上的有界保护样本，不等于所有 mapping/novel views 的完备保证，端到端保护仍由 held-out exact control/confirmation 负责。

coordinate expansion 的报告含义：

- `global_seed_action_scale`：coordinate expansion 之前所有 Anchor 共同通过的初始 seed。
- `post_training_action_scale`：最终 `per_anchor_action_scales` 的最小值；如果所有 Anchor 都继续扩张，它可以大于 global seed。
- `per_anchor_action_scales`：与 `selected_anchor_rows` 一一对应的最终 scale，范围为 `(0, 1]`。
- `per_anchor_observed_angle_deg`：从 baseline 与最终保存 descriptor 直接重算的真实角位移；这是可验证的部署动作量，optimizer scale 仅作诊断。
- `maximum_applied_anchor_scale` / `mean_applied_anchor_scale`：最终 per-Anchor scale 的最大值和均值。
- `coordinate_expanded_anchor_count`：在全局 seed 之上至少成功扩张一次的 Anchor 数；它不是 Top-1 changed rows 数，也不等于定位收益。

进入 exact replay 前必须通过：

```bash
python - "$V20_RUN/action_positive_only_angle5/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
training = report["training"]
scales = training["per_anchor_action_scales"]
angles = training["per_anchor_observed_angle_deg"]
assert report["status"] == "REQUIRES_EXACT_POSE_CONTROL"
assert training["strong_feedback_authorized"] is True
assert training["clean_protection_passed"] is True
assert training["materialized_action_audit"]["passed"] is True
assert training["positive_win_nonregression_passed"] is True
assert training["post_training_action_scale"] > 0.0
assert training["positive_objective"] == "per_positive_listwise_mean"
assert training["selected_anchor_count"] <= 4096
assert len(scales) == training["selected_anchor_count"]
assert len(angles) == training["selected_anchor_count"]
assert all(0.0 < scale <= 1.0 for scale in scales)
assert all(0.0 < angle <= training["maximum_angle_deg"] + 0.05 for angle in angles)
assert 0.0 < training["global_seed_action_scale"] <= min(scales)
assert abs(min(scales) - training["post_training_action_scale"]) < 1e-8
assert abs(max(scales) - training["maximum_applied_anchor_scale"]) < 1e-8
assert 0 <= training["coordinate_expanded_anchor_count"] <= len(scales)
print("PASS:", report["arm"], "may enter exact control")
PY
```

`REJECTED_CLEAN_PROTECTION`、`NO_EFFECT_AFTER_CLEAN_BACKOFF` 或 `ANALYSIS_ONLY_TEACHER_NOT_AUTHORIZED` 都是正式部署路径的停止条件。最后一种状态只可显式进入第 4.1 节的诊断分支。

后续示例的冻结 arm 为：

```bash
export V20_ARM=sparse_positive_only_angle_5
export V20_CANDIDATE_MAP="$V20_RUN/action_positive_only_angle5/candidate_anchor_map.pt"
```

若改过 mode 或 angle，必须从 `report.json.arm` 读取实际名字，并在 control 前冻结，不能根据结果改臂。

## 4. Held-out control：exact global Top-1 + PoseLib

单 shard 是 CLI 默认值；资源充足时可用相同 `--shard-count N` 生成完整的 `0..N-1` registry，aggregate 会拒绝缺 shard 或混合 candidate。聚合时还会逐条重开 SHA-bound certified source record，并把 replay 的 query index 与 pose family 精确绑定后，才允许 pose-family block bootstrap。

正式 control 命令故意不带 `--allow-analysis-only`；默认 evaluator 只接受训练状态为 `REQUIRES_EXACT_POSE_CONTROL` 的已授权 action。

```bash
mkdir -p "$V20_RUN/control"

python -m scripts.evaluate_v20_sparse_map \
  --certified-batch "$V20_RUN/control_certified_view.json" \
  --baseline-map "$V20_MAP" \
  --candidate-map "$V20_CANDIDATE_MAP" \
  --expected-role feedback_query \
  --device cuda \
  --shard-index 0 \
  --shard-count 1 \
  --output "$V20_RUN/control/shard0.json"

python -m scripts.aggregate_v20_sparse_map \
  --shards "$V20_RUN/control/shard0.json" \
  --phase control \
  --output "$V20_RUN/control_decision.json"
```

control 阶段不要传 `--selected-arm`。只有输出同时满足 `selected_arm == V20_ARM` 和 `decision == ADVANCE_TO_CONFIRMATION` 才允许打开 fresh confirmation：

```bash
python - "$V20_RUN/control_decision.json" "$V20_ARM" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1]))
assert decision["selected_arm"] == sys.argv[2]
assert decision["decision"] == "ADVANCE_TO_CONFIRMATION"
print("PASS: frozen control arm may enter confirmation")
PY
```

失败时不要运行 confirmation；稳定部署状态保持不变。

### 4.1 可选的 teacher-unauthorized analysis-only control

`--allow-analysis-only` 不是授权绕过开关。它只允许一个训练状态恰为 `ANALYSIS_ONLY_TEACHER_NOT_AUTHORIZED`、mode 为 `positive_only`、action 非零且 clean protection 已通过的 proposal，在 held-out control 上测量 exact task response。该 flag 不能接受 clean failure、zero-effect 或 repeated-negative proposal。

若决定做此诊断，应使用独立的 analysis-only 输出目录，并在 evaluator 命令中显式增加该 flag：

```bash
mkdir -p "$V20_RUN/analysis_only_control"

python -m scripts.evaluate_v20_sparse_map \
  --certified-batch "$V20_RUN/control_certified_view.json" \
  --baseline-map "$V20_MAP" \
  --candidate-map /absolute/path/to/analysis_only_candidate_anchor_map.pt \
  --expected-role feedback_query \
  --allow-analysis-only \
  --device cuda \
  --shard-index 0 \
  --shard-count 1 \
  --output "$V20_RUN/analysis_only_control/shard0.json"

python -m scripts.aggregate_v20_sparse_map \
  --shards "$V20_RUN/analysis_only_control/shard0.json" \
  --phase control \
  --output "$V20_RUN/analysis_only_control_decision.json"
```

产物会封存 `analysis_only: true` 和 `strong_feedback_authorized: false`。aggregator 即使观察到正向 risk，也必须输出 `selected_arm: null`、`decision: NO_ACTION`；confirmation 会拒绝未授权 arm，finalizer 也要求 control/confirmation 均为 `analysis_only: false`。

更重要的是，这次运行已经查看了 held-out control 的 task response，所以该 control 被消费，不能在 teacher 以后获得授权时重新充当 formal control。若还要正式推进，必须重新封存与 design、旧 control 和 confirmation 都不重叠的新 pose families，并开启新的 protocol/output root。analysis-only 结果只能回答“这个未授权方向在这批 control 上发生了什么”，不能回答“该动作已获得选择或部署资格”。

## 5. Fresh confirmation：只复现，不训练、不选臂

```bash
mkdir -p "$V20_RUN/confirmation"

python -m scripts.evaluate_v20_sparse_map \
  --certified-batch "$V20_RUN/confirmation_certified_view.json" \
  --baseline-map "$V20_MAP" \
  --candidate-map "$V20_CANDIDATE_MAP" \
  --expected-role confirmation_query \
  --device cuda \
  --shard-index 0 \
  --shard-count 1 \
  --output "$V20_RUN/confirmation/shard0.json"

python -m scripts.aggregate_v20_sparse_map \
  --shards "$V20_RUN/confirmation/shard0.json" \
  --phase confirmation \
  --selected-arm "$V20_ARM" \
  --output "$V20_RUN/confirmation_decision.json"
```

confirmation 的分类不能用于换 mode、angle、训练步数或 teacher tier。任何新超参数都必须开启新的 design/control/confirmation protocol，不能复用本次 confirmation。

## 6. 原子 finalize / rollback

只有已经到达 confirmation 的 frozen action 才调用 finalizer：

```bash
python -m scripts.finalize_v20_closed_loop \
  --action-report "$V20_RUN/action_positive_only_angle5/report.json" \
  --control-decision "$V20_RUN/control_decision.json" \
  --confirmation-decision "$V20_RUN/confirmation_decision.json" \
  --stable-map "$V20_MAP" \
  --stable-metric "$V20_STABLE_METRIC" \
  --output "$V20_RUN/closed_loop_deployment_decision.json"
```

`DEPLOY_CANDIDATE` 需要同时满足：

1. Tier B strong feedback 已授权；clean protection 通过且每个 per-Anchor action scale 都在 `(0, 1]`；训练状态为 `REQUIRES_EXACT_POSE_CONTROL`。
2. control 的 `analysis_only` 为 false，选择同一 frozen arm 并明确 `ADVANCE_TO_CONFIRMATION`。
3. confirmation 的 `analysis_only` 也为 false，复现同一 arm，classification 为 `DEFAULT_CANDIDATE` 或 `PARETO_CANDIDATE`，hard safety 全通过。
4. confirmation paired net gain 大于 0，pose-family block bootstrap 的 lower-risk probability 至少 0.95。
5. design/control/confirmation 的 source-record SHA 与 pose families 两两不重叠；三段 ACCEPT registry 都完整覆盖。
6. action report、candidate map、真实 zero-transform identity metric、stable baseline map/metric、per-Anchor diagnostics、control 和 confirmation 的 SHA/arm 绑定完全一致。

任一条件不满足，finalizer 输出 `RETAIN_STABLE`，map mutation 为 `none`。若 action 在 training 或 control 已停止，则不应为了凑齐 finalizer 输入而消费 confirmation；该阶段的 fail-closed 结果已经是保留当前 stable map/metric。

## 7. 最小验证与结果解释

实现级回归测试：

```bash
python -m pytest -q \
  tests/test_v20_feedback.py \
  tests/test_v20_sparse_descriptor.py \
  tests/test_v20_closed_loop.py
```

成功实现闭环不等于 candidate 已被证明有效。只有最终 artifact 中：

```text
formal_deployment_authorized: true
decision: DEPLOY_CANDIDATE
```

才可以替换 stable map。否则报告真实停止原因（teacher 未授权、clean backoff 清零、control 无动作或 confirmation 不成立），不要用 design margin、Top-K repair count 或 mapping-only accuracy 代替 novel-view exact localization 增益。
